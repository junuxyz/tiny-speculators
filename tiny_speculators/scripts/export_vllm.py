"""Export a native EAGLE-3 checkpoint to vLLM's Speculators format."""

import argparse
import copy
import json
from pathlib import Path

from safetensors.torch import load_file, save_file

from tiny_speculators.config import Config, validate_qwen3_8b_config


REQUIRED_WEIGHTS = {
    "d2t",
    "embed_tokens.weight",
    "fc.weight",
    "layers.0.hidden_norm.weight",
    "layers.0.self_attn.q_proj.weight",
    "lm_head.weight",
    "norm.weight",
    "t2d",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verifier", default=Config.verifier)
    return parser.parse_args()


def build_vllm_config(native_config: dict, verifier: str) -> dict:
    """Translate a native EAGLE-3 config into vLLM Speculators format."""

    transformer_config = copy.deepcopy(native_config["transformer_layer_config"])
    if not isinstance(transformer_config, dict):
        raise TypeError("transformer_layer_config must be a dictionary")
    validate_qwen3_8b_config(transformer_config, num_hidden_layers=1)

    transformer_config["num_hidden_layers"] = 1
    if "layer_types" in transformer_config:
        transformer_config["layer_types"] = ["full_attention"]

    ttt_steps = native_config.get("ttt_steps", 3)
    auxiliary_layers = native_config.get(
        "eagle_aux_hidden_state_layer_ids",
        list(Config.eagle_aux_hidden_state_layer_ids),
    )
    return {
        "architectures": ["Eagle3Speculator"],
        "draft_vocab_size": native_config["draft_vocab_size"],
        "eagle_aux_hidden_state_layer_ids": auxiliary_layers,
        "model_type": "speculators",
        "norm_before_residual": native_config.get("norm_before_residual", False),
        "norm_before_fc": False,
        "fc_norm": False,
        "norm_output": False,
        "speculators_model_type": "eagle3",
        "speculators_config": {
            "algorithm": "eagle3",
            "default_proposal_method": "greedy",
            "proposal_methods": [
                {
                    "accept_tolerance": 0.0,
                    "proposal_type": "greedy",
                    "speculative_tokens": ttt_steps,
                    "verifier_accept_k": 1,
                }
            ],
            "verifier": {
                "architectures": ["Qwen3ForCausalLM"],
                "name_or_path": verifier,
            },
        },
        "transformer_layer_config": transformer_config,
    }


def export_weights(weights_path: Path) -> dict:
    """Load the weights required by the vLLM checkpoint contract."""

    exported = {
        name: tensor
        for name, tensor in load_file(weights_path, device="cpu").items()
        if not name.startswith("verifier_norm.")
    }

    missing = REQUIRED_WEIGHTS - exported.keys()
    if missing:
        raise ValueError(f"Missing required weights: {sorted(missing)}")
    return exported


def export_vllm(checkpoint: Path, output: Path, verifier: str) -> None:
    """Export a native checkpoint as a vLLM-compatible model directory."""

    config_path = checkpoint / "config.json"
    weights_path = checkpoint / "model.safetensors"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    if checkpoint.resolve() == output.resolve():
        raise ValueError("--output must be different from --checkpoint")

    native_config = json.loads(config_path.read_text(encoding="utf-8"))
    vllm_config = build_vllm_config(native_config, verifier)
    weights = export_weights(weights_path)

    output.mkdir(parents=True, exist_ok=True)
    save_file(
        weights,
        output / "model.safetensors",
        metadata={"format": "pt"},
    )
    (output / "config.json").write_text(
        json.dumps(vllm_config, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    export_vllm(args.checkpoint, args.output, args.verifier)
    print(f"Exported vLLM artifact to {args.output}")


if __name__ == "__main__":
    main()
