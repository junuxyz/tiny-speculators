"""Run the end-to-end EAGLE-3 training and vLLM demo pipeline."""

import argparse
from pathlib import Path
import subprocess
import sys
import tempfile

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
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument(
        "--start-sample",
        type=int,
        default=0,
        help="Skip this many prepared samples when extending a chunked run",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=Config.max_model_len,
        help=(
            "Shared token limit for preprocessing, hidden-state extraction, "
            "and packed training"
        ),
    )
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
    parser.add_argument(
        "--start-at",
        choices=("prepare", "vocab", "hidden", "train", "export"),
        help="First stage to run; defaults to prepare, or train with --resume",
    )
    parser.add_argument(
        "--stop-after",
        choices=("prepare", "vocab", "hidden", "train", "export"),
        default="export",
        help="Last stage to run",
    )
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
    if args.chunk_size is not None and args.chunk_size < 2:
        raise ValueError("--chunk-size must be at least 2 for validation")
    if args.chunk_size is not None and args.epochs != 1:
        raise ValueError("--chunk-size supports one epoch per chunk")
    if args.start_sample < 0:
        raise ValueError("--start-sample must be non-negative")
    if args.start_sample and (args.chunk_size is None or args.resume is None):
        raise ValueError("--start-sample requires --chunk-size and --resume")
    if args.max_samples is not None and args.start_sample >= args.max_samples:
        raise ValueError("--start-sample must be smaller than --max-samples")
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    if args.max_sequence_length < 2:
        raise ValueError("--max-sequence-length must be at least 2")

    stages = ("prepare", "vocab", "hidden", "train", "export")
    start_at = args.start_at or (
        "prepare" if args.start_sample else
        "train" if args.resume is not None else
        "prepare"
    )
    start_index = stages.index(start_at)
    stop_index = stages.index(args.stop_after)
    if start_index > stop_index:
        raise ValueError("--start-at must not come after --stop-after")
    extending_chunks = bool(
        args.chunk_size is not None and args.start_sample and args.resume is not None
    )
    if (
        args.resume is not None
        and start_index < stages.index("train")
        and not extending_chunks
    ):
        raise ValueError("--resume may only be used with --start-at train or later")
    def should_run(stage: str) -> bool:
        index = stages.index(stage)
        return start_index <= index <= stop_index

    checkpoint = args.checkpoint or args.resume or Path("checkpoints/eagle3")
    if checkpoint.resolve() == args.exported_checkpoint.resolve():
        raise ValueError("--checkpoint and --exported-checkpoint must differ")

    max_samples = args.max_samples
    sample_args = (
        [] if max_samples is None
        else ["--max-samples", str(max_samples)]
    )

    sequence_limit = str(args.max_sequence_length)

    if should_run("prepare"):
        run(
            "tiny_speculators.scripts.prepare_data",
            "--model", args.verifier,
            "--output", str(args.data),
            "--max-length", sequence_limit,
            *sample_args,
        )

    if should_run("vocab"):
        run(
            "tiny_speculators.scripts.vocab_mapping",
            "--data", str(args.data),
        )

    if should_run("hidden") and args.chunk_size is None:
        run(
            "tiny_speculators.scripts.generate_hidden_states",
            "--model", args.verifier,
            "--data", str(args.data),
            "--output", str(args.data / "hidden_states"),
            "--max-model-len", sequence_limit,
            *sample_args,
        )

    if args.chunk_size is not None:
        sources = sorted((args.data / "tokenized").glob("data_*.pt"))[
            args.start_sample:max_samples
        ]
        completed_epochs = 0
        if args.resume is not None:
            import torch

            trainer_state = torch.load(
                args.resume / "trainer_state.pt",
                map_location="cpu",
                weights_only=True,
            )
            completed_epochs = int(trainer_state["epoch"])
        for chunk_index, start in enumerate(
            range(0, len(sources), args.chunk_size), 1
        ):
            with tempfile.TemporaryDirectory() as directory:
                chunk = Path(directory)
                (chunk / "tokenized").mkdir()
                for source in sources[start:start + args.chunk_size]:
                    (chunk / "tokenized" / source.name).symlink_to(source.resolve())
                (chunk / "vocab_mapping.pt").symlink_to(
                    (args.data / "vocab_mapping.pt").resolve()
                )
                run(
                    "tiny_speculators.scripts.generate_hidden_states",
                    "--model", args.verifier,
                    "--data", str(chunk),
                    "--max-model-len", sequence_limit,
                )
                train_args = [
                    "--data", str(chunk),
                    "--verifier", args.verifier,
                    "--output", str(checkpoint),
                    "--metrics-file", str(checkpoint / "metrics.jsonl"),
                    "--epochs", str(completed_epochs + chunk_index),
                    "--max-batch-tokens", sequence_limit,
                ]
                if args.resume is not None or chunk_index > 1:
                    resume = args.resume if chunk_index == 1 else checkpoint
                    train_args += ["--resume", str(resume)]
                run("tiny_speculators.scripts.train_eagle3", *train_args)
    elif should_run("train"):
        train_args = [
            "--data", str(args.data),
            "--verifier", args.verifier,
            "--output", str(checkpoint),
            "--metrics-file", str(checkpoint / "metrics.jsonl"),
            "--epochs", str(args.epochs),
            "--max-batch-tokens", sequence_limit,
            *sample_args,
        ]
        if args.resume is not None:
            train_args.extend(["--resume", str(args.resume)])
        run("tiny_speculators.scripts.train_eagle3", *train_args)

    if should_run("export"):
        run(
            "tiny_speculators.scripts.export_vllm",
            "--checkpoint", str(checkpoint),
            "--output", str(args.exported_checkpoint),
            "--verifier", args.verifier,
        )

if __name__ == "__main__":
    main()
