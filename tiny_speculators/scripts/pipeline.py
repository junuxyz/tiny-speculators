"""Run the end-to-end EAGLE-3 training and vLLM demo pipeline."""

import argparse
from pathlib import Path
import subprocess
import sys

from tiny_speculators.config import Config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verifier", default=Config.verifier)
    parser.add_argument("--data", type=Path, default=Config.data_dir)
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Use a prefix of the dataset; defaults to all available samples",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Native checkpoint output; defaults to --resume or checkpoints/eagle3",
    )
    parser.add_argument(
        "--exported-checkpoint",
        type=Path,
        default=Path("checkpoints/eagle3-vllm"),
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--prompt", default="The capital of France is")
    return parser.parse_args()


def run(module: str, *args: str) -> None:
    """Run one pipeline stage with the active Python environment."""

    print(f"\n==> {module}", flush=True)
    subprocess.run(
        [sys.executable, "-m", module, *args],
        check=True,
    )


def main() -> None:
    args = parse_args()
    if args.max_samples is not None and args.max_samples < 2:
        raise ValueError("--max-samples must be at least 2 for validation")
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    checkpoint = args.checkpoint or args.resume or Path("checkpoints/eagle3")
    if checkpoint.resolve() == args.exported_checkpoint.resolve():
        raise ValueError("--checkpoint and --exported-checkpoint must differ")

    max_samples = args.max_samples
    sample_args = (
        [] if max_samples is None
        else ["--max-samples", str(max_samples)]
    )

    if args.resume is None:
        run(
            "tiny_speculators.scripts.prepare_data",
            "--model", args.verifier,
            "--output", str(args.data),
            *sample_args,
        )
        run(
            "tiny_speculators.scripts.vocab_mapping",
            "--data", str(args.data),
        )
        run(
            "tiny_speculators.scripts.generate_hidden_states",
            "--model", args.verifier,
            "--data", str(args.data),
            "--output", str(args.data / "hidden_states"),
            *sample_args,
        )

    train_args = [
        "--data", str(args.data),
        "--verifier", args.verifier,
        "--output", str(checkpoint),
        "--epochs", str(args.epochs),
        *sample_args,
    ]
    if args.resume is not None:
        train_args.extend(["--resume", str(args.resume)])
    run("tiny_speculators.scripts.train_eagle3", *train_args)

    run(
        "tiny_speculators.scripts.export_vllm",
        "--checkpoint", str(checkpoint),
        "--output", str(args.exported_checkpoint),
        "--verifier", args.verifier,
    )
    run(
        "tiny_speculators.scripts.vllm_inference",
        "--checkpoint", str(args.exported_checkpoint),
        "--verifier", args.verifier,
        "--prompt", args.prompt,
    )


if __name__ == "__main__":
    main()
