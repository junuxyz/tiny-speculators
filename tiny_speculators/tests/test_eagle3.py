import pytest
import torch
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

from tiny_speculators.eagle3.config import Eagle3SpeculatorConfig
from tiny_speculators.eagle3.model import Eagle3DraftModel
from tiny_speculators.eagle3.attention import create_ttt_block_mask
from tiny_speculators.eagle3.data import shift_sample
from tiny_speculators.eagle3.vocab import (
    build_vocab_mappings,
    get_selected_target_ids,
)
from tiny_speculators.config import Config, validate_qwen3_8b_config
from tiny_speculators.scripts.export_vllm import build_vllm_config
from tiny_speculators.scripts.train_eagle3 import load_checkpoint, save_checkpoint


def make_eagle3_draft_model() -> Eagle3DraftModel:
    qwen3_config = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=64,
    )
    verifier = Qwen3ForCausalLM(qwen3_config)
    selected_target_ids = torch.arange(0, 32, 2)
    target_to_draft = torch.zeros(32, dtype=torch.bool)
    target_to_draft[selected_target_ids] = True

    return Eagle3DraftModel(
        config=Eagle3SpeculatorConfig(
            transformer_layer_config=qwen3_config,
            draft_vocab_size=selected_target_ids.numel(),
            ttt_steps=3,
        ),
        verifier=verifier,
        draft_to_target_offsets=(
            selected_target_ids - torch.arange(selected_target_ids.numel())
        ),
        target_to_draft=target_to_draft,
    )


def test_shift_sample_aligns_next_token_targets():
    shifted = shift_sample(
        input_ids=torch.tensor([0, 1, 2, 3]),
        hidden_states=torch.arange(8).reshape(4, 2),
        verifier_last_hidden_states=torch.arange(8, 16).reshape(4, 2),
        loss_mask=torch.tensor([False, True, False, True]),
    )

    assert torch.equal(shifted["input_ids"], torch.tensor([[1, 2, 3]]))
    assert torch.equal(shifted["hidden_states"], torch.arange(6).reshape(1, 3, 2))
    assert torch.equal(
        shifted["verifier_last_hidden_states"],
        torch.arange(10, 16).reshape(1, 3, 2),
    )
    assert torch.equal(shifted["loss_mask"], torch.tensor([[True, False, True]]))
    assert torch.equal(shifted["position_ids"], torch.tensor([[1, 2, 3]]))


def test_vocab_mapping_selects_most_frequent_tokens():
    d2t, t2d = build_vocab_mappings(
        token_frequencies={4: 10, 1: 8, 3: 4, 0: 1},
        draft_vocab_size=3,
        target_vocab_size=6,
    )

    selected_target_ids = get_selected_target_ids(t2d)
    assert torch.equal(selected_target_ids, torch.tensor([1, 3, 4]))
    assert torch.equal(
        torch.arange(d2t.numel()) + d2t,
        selected_target_ids,
    )


def test_qwen3_8b_defaults_are_consistent():
    config = Eagle3SpeculatorConfig()

    validate_qwen3_8b_config(
        config.transformer_layer_config,
        num_hidden_layers=1,
    )
    assert Config.verifier == "Qwen/Qwen3-8B"
    assert Config.max_model_len == 4_096
    assert config.eagle_aux_hidden_state_layer_ids == [2, 18, 33]
    assert config.norm_before_residual is True

    exported = build_vllm_config(config.to_dict(), Config.verifier)
    assert exported["eagle_aux_hidden_state_layer_ids"] == [2, 18, 33]
    assert exported["norm_before_residual"] is True
    assert exported["speculators_config"]["verifier"]["name_or_path"] == (
        "Qwen/Qwen3-8B"
    )


def test_ttt_attention_preserves_documents_causality_and_anchors():
    mask = create_ttt_block_mask(
        lengths=torch.tensor([2, 2]),
        ttt_steps=2,
    )

    def allowed(query: int, key: int) -> bool:
        return bool(mask.mask_mod(None, None, torch.tensor(query), torch.tensor(key)))

    assert allowed(0, 0)
    assert not allowed(0, 1)
    assert not allowed(2, 1)
    assert allowed(2, 2)
    assert allowed(0, 4)
    assert allowed(1, 5)
    assert not allowed(0, 6)


def test_checkpoint_restores_model_optimizer_and_progress(tmp_path):
    model = make_eagle3_draft_model()
    optimizer = torch.optim.AdamW(
        parameter for parameter in model.parameters() if parameter.requires_grad
    )
    model.fc.weight.grad = torch.ones_like(model.fc.weight)
    optimizer.step()
    original_fc_weight = model.fc.weight.detach().clone()

    save_checkpoint(
        tmp_path,
        model,
        optimizer,
        epoch=2,
        best_val_loss=1.25,
        chunk_index=4,
        validation_samples=["data_9.pt"],
    )
    with torch.no_grad():
        model.fc.weight.zero_()

    load_checkpoint(tmp_path, model, optimizer)
    restored_config = Eagle3SpeculatorConfig.from_pretrained(tmp_path)
    trainer_state = torch.load(
        tmp_path / "trainer_state.pt",
        weights_only=True,
    )

    assert torch.equal(model.fc.weight, original_fc_weight)
    assert optimizer.state
    assert restored_config.ttt_steps == model.config.ttt_steps
    assert trainer_state["epoch"] == 2
    assert trainer_state["chunk_index"] == 4
    assert trainer_state["best_val_loss"] == 1.25


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="FlexAttention backward requires CUDA",
)
def test_tiny_forward_computes_loss_and_gradients():
    device = torch.device("cuda")
    model = make_eagle3_draft_model().to(device)
    seq_len = 4
    hidden_size = model.qwen3_config.hidden_size

    loss, metrics = model(
        hidden_states=torch.randn(1, seq_len, 3 * hidden_size, device=device),
        input_ids=torch.tensor([[0, 2, 4, 6]], device=device),
        lengths=torch.tensor([seq_len], device=device),
        position_ids=torch.arange(1, seq_len + 1, device=device)[None],
        verifier_last_hidden_states=torch.randn(1, seq_len, hidden_size, device=device),
        loss_mask=torch.ones(1, seq_len, dtype=torch.bool, device=device),
    )

    assert loss.isfinite()
    assert metrics.keys() >= {"loss", "loss_0", "loss_1", "loss_2"}

    loss.backward()
    assert model.fc.weight.grad is not None
