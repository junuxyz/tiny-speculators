from transformers import PretrainedConfig
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
import copy

from tiny_speculators.config import Config

DEFAULT_DRAFT_VOCAB_SIZE = 32_000


def default_qwen3_8b_config() -> Qwen3Config:

    return Qwen3Config(
        vocab_size=151_936,
        hidden_size=4_096,
        intermediate_size=12_288,
        num_hidden_layers=36,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        max_position_embeddings=40_960,
        rope_scaling={"rope_type": "default", "rope_theta": 1_000_000},
    )

class Eagle3SpeculatorConfig(PretrainedConfig):

    model_type = "tiny_eagle3"

    def __init__(
        self,
        transformer_layer_config: Qwen3Config | dict | None = None,
        draft_vocab_size: int = DEFAULT_DRAFT_VOCAB_SIZE,
        ttt_steps: int = 3,
        ttt_loss_decay: float = 1.0,
        eagle_aux_hidden_state_layer_ids: list[int] | None = None,
        norm_before_residual: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        if transformer_layer_config is None:
            transformer_layer_config = default_qwen3_8b_config()
        elif isinstance(transformer_layer_config, dict):
            transformer_layer_config = Qwen3Config(**transformer_layer_config)
        else:
            transformer_layer_config = copy.deepcopy(transformer_layer_config) # independent config for draft
        transformer_layer_config.num_hidden_layers = 1
        if hasattr(transformer_layer_config, "layer_types"):
            transformer_layer_config.layer_types = ["full_attention"]

        self.transformer_layer_config = transformer_layer_config
        self.draft_vocab_size = draft_vocab_size
        self.ttt_steps = ttt_steps
        self.ttt_loss_decay = ttt_loss_decay
        self.speculators_model_type = "eagle3"
        self.eagle_aux_hidden_state_layer_ids = (
            eagle_aux_hidden_state_layer_ids
            or list(Config.eagle_aux_hidden_state_layer_ids)
        )
        self.norm_before_residual = norm_before_residual
