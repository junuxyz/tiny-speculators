import copy
import torch
from torch import nn
import torch.nn.functional as F
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
from tiny_speculators.eagle3.config import Eagle3SpeculatorConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RMSNorm, Qwen3RotaryEmbedding
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.cache_utils import DynamicCache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers import PreTrainedModel

from torch.nn.attention.flex_attention import BlockMask
from tiny_speculators.eagle3.vocab import get_selected_target_ids
from tiny_speculators.eagle3.attention import (
    create_ttt_block_mask,
    document_ids,
    flex_attention_forward,
)

ALL_ATTENTION_FUNCTIONS.register(
    "simple_flex_attention",
    flex_attention_forward,
)

class Eagle3DraftModel(PreTrainedModel):

    config_class = Eagle3SpeculatorConfig
    _keys_to_ignore_on_save = ["verifier_norm.weight"]

    def __init__(
            self,
            config: Eagle3SpeculatorConfig,
            verifier: Qwen3ForCausalLM,
            draft_to_target_offsets: torch.Tensor,
            target_to_draft: torch.Tensor,
    ):
        super().__init__(config)
        self.qwen3_config = config.transformer_layer_config
        hidden_size = self.qwen3_config.hidden_size
        
        # Embedding Layer
        verifier_embeddings = verifier.get_input_embeddings()
        self.embed_tokens = copy.deepcopy(verifier_embeddings)
        self.embed_tokens.requires_grad_(False) # freeze
        
        # FC Layer
        self.fc = nn.Linear(3 * hidden_size, hidden_size, bias=False)
        
        # Attention - choose custom FlexAttention implementation
        self.qwen3_config._attn_implementation = "simple_flex_attention"
        
        # RoPE
        self.rotary_emb = Qwen3RotaryEmbedding(self.qwen3_config)
        
        # Decoder Layer
        self.layers = nn.ModuleList([
            Eagle3FirstDecoderLayer(
                self.qwen3_config,
                layer_idx=0,
                norm_before_residual=config.norm_before_residual,
            )
        ])
        
        # draft model final RMSNorm
        self.norm = Qwen3RMSNorm(
            hidden_size=hidden_size,
            eps=self.qwen3_config.rms_norm_eps,
        )

        self.register_buffer(
            "d2t",
            draft_to_target_offsets.to(dtype=torch.long)
        )
        
        self.register_buffer(
            "t2d",
            target_to_draft.to(dtype=torch.bool)
        )
        
        # t2d
        selected_target_ids = get_selected_target_ids(self.t2d) # [draft_vocab_size,]
        
        if selected_target_ids.numel() != self.config.draft_vocab_size:
            raise ValueError(
                "target_to_draft must select exactly "
                f"{self.config.draft_vocab_size} tokens, "
                f"but selected {selected_target_ids.numel()}"
            )

        # d2t
        recovered_target_ids = (
            self.d2t
            + torch.arange( 
                self.d2t.numel(),
                device=self.d2t.device,
            )
        )

        if not torch.equal(recovered_target_ids,
            selected_target_ids.to(recovered_target_ids.device),
        ):
            raise ValueError(
                "draft_to_target_offsets and target_to_draft describe "
                "different vocabularies"
            )

        full_lm_head_weight = verifier.lm_head.weight
        selected_lm_head_weight = full_lm_head_weight[
            selected_target_ids.to(device=full_lm_head_weight.device)]
        # [verifier_vocab_size,] -> [draft_vocab_size,]

        # Verifier model final RMSNorm
        self.verifier_norm = copy.deepcopy(verifier.model.norm)
        self.verifier_norm.requires_grad_(False)

        # LMHead
        self.lm_head = nn.Linear( # [hidden_size, draft_vocab_size]
            in_features=hidden_size,
            out_features=selected_target_ids.numel(),
            bias=False,
            device=selected_lm_head_weight.device,
            dtype=selected_lm_head_weight.dtype,
        )

        with torch.no_grad():
            self.lm_head.weight.copy_(selected_lm_head_weight)
        self.lm_head.requires_grad_(False) # freeze

        verifier_weight = verifier_embeddings.weight
        self.to(
            device=verifier_weight.device,
            dtype=verifier_weight.dtype,
        )

    def forward(
            self,
            hidden_states: torch.Tensor, # [1, seq_len, h]
            input_ids: torch.Tensor,
            lengths: torch.Tensor,
            position_ids: torch.Tensor,
            verifier_last_hidden_states: torch.Tensor,
            loss_mask: torch.Tensor,
        ):

        attention_masks = tuple(
            create_ttt_block_mask(
                lengths=lengths,
                ttt_steps=ttt_step + 1,
            )
            for ttt_step in range(self.config.ttt_steps)
        )
        past_key_values = DynamicCache()
        hidden_states = self.fc(hidden_states)
        doc_ids = document_ids(lengths)
        position_ids = position_ids.clone()
        with torch.no_grad():
            targets = self.lm_head(
                self.verifier_norm(verifier_last_hidden_states)
            )
        draft_tokens = []
        loss = hidden_states.new_zeros((), dtype=torch.float32)
        metrics: dict[str, torch.Tensor] = {}

        for ttt_step in range(self.config.ttt_steps):
            with torch.no_grad():
                token_embeddings = self.embed_tokens(input_ids) # [1, seq_len, h]
            
            hidden_states = torch.cat([token_embeddings, hidden_states], dim=-1)
            # decoder input; [1, seq_len, 2h]

            position_embeddings = self.rotary_emb(hidden_states, position_ids)

            for layer in self.layers:
                hidden_states = layer( # used for the next draft state input
                    hidden_states,
                    attention_mask=attention_masks[ttt_step],
                    position_embeddings=position_embeddings,
                    past_key_values=past_key_values,
                    )
            
            logits = self.lm_head(self.norm(hidden_states))        
            
            # ttt step alignment
            if ttt_step == 0:
                step_logits = logits
                step_targets = targets
                step_loss_mask = loss_mask
            else:
                step_logits = logits[:, :-ttt_step] # exclude logits that exceed ttt_step
                step_targets = targets[:, ttt_step:]
                step_loss_mask = loss_mask[:, ttt_step:]
                same_document = (
                    doc_ids[:-ttt_step] == doc_ids[ttt_step:]
                )
                step_loss_mask = step_loss_mask & same_document[None]
            
            token_loss = F.kl_div(
                F.log_softmax(step_logits.float(), dim=-1),
                F.softmax(step_targets.float(), dim=-1),
                reduction="none",
            ).sum(dim=-1)

            selected_loss = token_loss[step_loss_mask]
            if selected_loss.numel() == 0:
                s_loss = token_loss.sum() * 0.0
                accuracy = token_loss.new_zeros(())
            else:
                s_loss = selected_loss.mean()
                predictions = torch.argmax(step_logits, dim=-1)
                target_tokens = torch.argmax(step_targets, dim=-1)
                accuracy = (
                    predictions[step_loss_mask] == target_tokens[step_loss_mask]
                ).float().mean()
            loss += (self.config.ttt_loss_decay**ttt_step) * s_loss # weight more on earlier steps
            metrics[f"loss_{ttt_step}"] = s_loss.detach()
            metrics[f"accuracy_{ttt_step}"] = accuracy.detach()

            input_ids = torch.argmax(logits, dim=-1) # greedy decoding
            draft_tokens.append(input_ids.detach().clone())    

            input_ids = input_ids + self.d2t[input_ids]

            if ttt_step + 1 < self.config.ttt_steps:
                position_ids += 1

        metrics["loss"] = loss.detach()
        return loss, draft_tokens, metrics

class Eagle3FirstDecoderLayer(Qwen3DecoderLayer):

    def __init__(self,
                 qwen3_config: Qwen3Config,
                 layer_idx: int,
                 norm_before_residual: bool = False,
                 ):
        super().__init__(qwen3_config, layer_idx)
        self.norm_before_residual = norm_before_residual
        
        hidden_size = qwen3_config.hidden_size
        head_dim = qwen3_config.head_dim

        self.self_attn.q_proj = nn.Linear(
            in_features = 2 * hidden_size,
            out_features = qwen3_config.num_attention_heads * head_dim,
            bias = False,
        )
        self.self_attn.k_proj = nn.Linear(
            in_features = 2 * hidden_size,
            out_features = qwen3_config.num_key_value_heads * head_dim, # gqa
            bias = False,
        )
        self.self_attn.v_proj = nn.Linear(
            in_features = 2 * hidden_size,
            out_features = qwen3_config.num_key_value_heads * head_dim, # gqa
            bias = False,
        )
        self.hidden_norm = Qwen3RMSNorm(
            hidden_size=hidden_size,
            eps=qwen3_config.rms_norm_eps,
        )
        
    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: BlockMask,
            position_embeddings: tuple[torch.Tensor, torch.Tensor],
            past_key_values: DynamicCache | None,
            ):

        token_embeddings, draft_state = hidden_states.chunk(chunks=2, dim=-1)

        residual = draft_state
        token_embeddings = self.input_layernorm(token_embeddings)
        draft_state = self.hidden_norm(draft_state)
        if self.norm_before_residual:
            residual = draft_state

        hidden_states = torch.cat([token_embeddings, draft_state], dim=-1)
        
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values, # internally updated in Qwen3Attention
            position_embeddings=position_embeddings,
        )
        hidden_states = hidden_states + residual

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states + residual

        return hidden_states
