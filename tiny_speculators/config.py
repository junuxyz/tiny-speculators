from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parent


class Config:
    """Qwen3-8B defaults shared by every pipeline stage."""

    verifier: str = "Qwen/Qwen3-8B"
    dataset: str = "Aeala/ShareGPT_Vicuna_unfiltered"
    data_dir: Path = PACKAGE_ROOT / "data" / "qwen3-8b-sharegpt"
    max_model_len: int = 4_096
    enable_thinking: bool = False
    verifier_num_hidden_layers: int = 36
    eagle_aux_hidden_state_layer_ids: tuple[int, ...] = (2, 18, 33)


QWEN3_8B_CONFIG_DEFAULTS = {
    "hidden_size": 4_096,
    "intermediate_size": 12_288,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "vocab_size": 151_936,
    "max_position_embeddings": 40_960,
}


def validate_qwen3_8b_config(
    config: Any,
    *,
    num_hidden_layers: int = Config.verifier_num_hidden_layers,
) -> None:
    """Fail early when a pipeline stage is not configured for Qwen3-8B."""

    expected = {
        "model_type": "qwen3",
        **QWEN3_8B_CONFIG_DEFAULTS,
        "num_hidden_layers": num_hidden_layers,
    }
    mismatches = []
    for name, expected_value in expected.items():
        actual_value = (
            config.get(name)
            if isinstance(config, dict)
            else getattr(config, name, None)
        )
        if actual_value != expected_value:
            mismatches.append(f"{name}={actual_value!r} (expected {expected_value!r})")
    if mismatches:
        raise ValueError("Expected a Qwen3-8B config: " + ", ".join(mismatches))
