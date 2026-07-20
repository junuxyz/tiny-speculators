"""Generate target-model hidden states with vLLM offline inference."""

import argparse
from pathlib import Path

import torch
from transformers import AutoConfig
from vllm import LLM, SamplingParams
from vllm.config.kv_transfer import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    example_hidden_states_connector,
)

from tiny_speculators.config import Config, validate_qwen3_8b_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=Config.verifier)
    parser.add_argument("--data", type=Path, default=Config.data_dir)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-model-len", type=int, default=Config.max_model_len)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenized_dir = args.data / "tokenized"
    sources = sorted(tokenized_dir.glob("data_*.pt"))
    if args.max_samples is not None:
        if args.max_samples < 1:
            raise ValueError("--max-samples must be at least 1")
        sources = sources[: args.max_samples]
    if not sources:
        raise FileNotFoundError(f"No data_*.pt files found in {tokenized_dir}")

    output_dir = args.output or args.data / "hidden_states"
    output_dir.mkdir(parents=True, exist_ok=True)

    verifier_config = AutoConfig.from_pretrained(args.model)
    validate_qwen3_8b_config(verifier_config)
    target_layers = [
        *Config.eagle_aux_hidden_state_layer_ids,
        Config.verifier_num_hidden_layers,
    ]  # low, mid, high, final layer
    # launch instance
    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        # instructs vLLM to use the fake “extract_hidden_states” speculative method
        speculative_config={
            "method": "extract_hidden_states",
            "num_speculative_tokens": 1,
            "draft_model_config": {
                "hf_config": {"eagle_aux_hidden_state_layer_ids": target_layers}
            },
        },
        # sets up the custom KV connector; extracts
        # the hidden states from the draft model's layers
        kv_transfer_config=KVTransferConfig(
            kv_connector="ExampleHiddenStatesConnector",
            kv_role="kv_producer",
            kv_connector_extra_config={
                "shared_storage_path": str(output_dir.resolve()),
                "allow_custom_save_path": True,
            },
        ),
    )

    input_ids = []
    output_paths = []

    for path in sources:
        input_ids.append(torch.load(path, weights_only=True)["input_ids"])
        output_paths.append(
            output_dir / f"hs_{path.stem.removeprefix('data_')}.safetensors"
        )

    sampling_params = [
        SamplingParams(
            max_tokens=1,
            extra_args={
                "kv_transfer_params": {"hidden_states_path": str(path.resolve())}
            },
        )
        for path in output_paths
    ]

    outputs = llm.generate([tokens.tolist() for tokens in input_ids], sampling_params)

    for source, expected_ids, expected_path, result in zip(
        sources, input_ids, output_paths, outputs, strict=True
    ):
        saved_path = Path(result.kv_transfer_params["hidden_states_path"])
        if saved_path != expected_path.resolve():
            raise RuntimeError(f"vLLM saved {source.name} to an unexpected path")
        tensors = example_hidden_states_connector.load_hidden_states(str(saved_path))
        if not torch.equal(tensors["token_ids"].long(), expected_ids):
            raise RuntimeError(f"vLLM returned different token IDs for {source.name}")
        if tensors["hidden_states"].shape[:2] != (
            len(expected_ids),
            len(target_layers),
        ):
            raise RuntimeError(f"vLLM returned invalid hidden states for {source.name}")

    print(f"Generated {len(sources)} hidden-state files")


if __name__ == "__main__":
    main()
