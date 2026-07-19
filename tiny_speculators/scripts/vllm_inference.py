"""Load a saved EAGLE-3 checkpoint in vLLM and run speculative decoding."""

import argparse
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from tiny_speculators.config import Config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/root/checkpoints/eagle3"),
    )
    parser.add_argument("--verifier", default=Config.verifier)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-tokens", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.verifier)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=Config.enable_thinking,
    )
    llm = LLM(
        model=str(args.checkpoint),
        max_model_len=Config.max_model_len,
        gpu_memory_utilization=0.8,
        enforce_eager=True,
        disable_log_stats=False,
    )
    outputs = llm.generate(
        [prompt],
        SamplingParams(temperature=0.0, max_tokens=args.max_tokens),
    )

    metrics = {
        metric.name: getattr(metric, "value", getattr(metric, "values", None))
        for metric in llm.get_metrics()
        if metric.name.startswith("vllm:spec_decode_")
    }
    num_drafts = metrics.get("vllm:spec_decode_num_drafts", 0)
    if not isinstance(num_drafts, (int, float)) or num_drafts <= 0:
        raise RuntimeError(f"vLLM did not run the draft model: {metrics}")

    output = outputs[0].outputs[0]
    print(f"text={output.text!r}", flush=True)
    print(f"token_ids={list(output.token_ids)}", flush=True)
    print(f"spec_decode_metrics={metrics}", flush=True)


if __name__ == "__main__":
    main()
