"""Prepare data, train EAGLE-3 in chunks, and export it for vLLM."""

import argparse
from pathlib import Path
import subprocess
import sys
import tempfile

import torch

from tiny_speculators.config import Config


STAGES = ("prepare", "vocab", "train", "export")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verifier", default=Config.verifier)
    parser.add_argument("--data", type=Path, default=Config.data_dir)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-sequence-length", type=int, default=Config.max_model_len)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--exported-checkpoint",
        type=Path,
        default=Path("checkpoints/eagle3-vllm"),
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--start-at", choices=STAGES)
    parser.add_argument("--stop-after", choices=STAGES, default="export")
    return parser.parse_args()


def run(module: str, **options: object) -> None:
    args = [
        str(item)
        for name, value in options.items()
        if value is not None
        for item in (f"--{name.replace('_', '-')}", value)
    ]
    print(f"\n==> {module}", flush=True)
    subprocess.run([sys.executable, "-m", module, *args], check=True)


def stage_dataset(directory: Path, data: Path, sources: list[Path]) -> None:
    tokenized = directory / "tokenized"
    tokenized.mkdir()
    for source in sources:
        (tokenized / source.name).symlink_to(source.resolve())
    (directory / "vocab_mapping.pt").symlink_to((data / "vocab_mapping.pt").resolve())


def generate_hidden_states(
    directory: Path,
    data: Path,
    sources: list[Path],
    verifier: str,
    max_model_len: int,
) -> None:
    stage_dataset(directory, data, sources)
    run(
        "tiny_speculators.scripts.generate_hidden_states",
        data=directory,
        model=verifier,
        max_model_len=max_model_len,
    )


def train(args: argparse.Namespace, checkpoint: Path) -> None:
    if args.chunk_size is not None and args.chunk_size < 1:
        raise ValueError("--chunk-size must be at least 1")
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")

    sources = sorted((args.data / "tokenized").glob("data_*.pt"))
    by_name = {source.name: source for source in sources}
    if args.resume is None:
        sources = sources[: args.max_samples]
        if len(sources) < 2:
            raise ValueError("Training requires at least 2 samples")
        indices = torch.randperm(
            len(sources), generator=torch.Generator().manual_seed(42)
        ).tolist()
        sample_order = [sources[index].name for index in indices]
        train_size = max(1, min(len(sources) - 1, int(len(sources) * 0.9)))
        epoch, chunk_index, next_sample = 1, 0, 0
        resume = None
    else:
        state = torch.load(
            args.resume / "trainer_state.pt", map_location="cpu", weights_only=True
        )
        sample_order = state.get("sample_order") or [
            *state.get("training_samples", []),
            *state.get("validation_samples", []),
        ]
        train_size = state.get("train_size") or (
            len(sample_order) - len(state.get("validation_samples", []))
        )
        epoch = int(state["epoch"])
        chunk_index = int(state.get("chunk_index") or 0)
        next_sample = state.get("next_sample")
        resume = args.resume
        if not sample_order or next_sample is None:
            raise ValueError("Checkpoint does not contain training progress")
        if set(sample_order) - by_name.keys():
            raise ValueError("Checkpoint samples do not match the dataset")

    if not 0 < train_size < len(sample_order):
        raise ValueError("Checkpoint contains an invalid training split")
    if not 0 <= next_sample <= train_size:
        raise ValueError("Checkpoint sample offset is out of range")
    if epoch > args.epochs:
        raise ValueError("Checkpoint epoch exceeds --epochs")

    def shuffle_training(current_epoch: int) -> None:
        nonlocal sample_order
        indices = torch.randperm(
            train_size,
            generator=torch.Generator().manual_seed(41 + current_epoch),
        ).tolist()
        sample_order = [sample_order[index] for index in indices] + sample_order[
            train_size:
        ]

    if next_sample == train_size:
        epoch += 1
        chunk_index = next_sample = 0
        if epoch > args.epochs:
            return
        shuffle_training(epoch)

    chunk_size = args.chunk_size or train_size
    with tempfile.TemporaryDirectory() as validation_dir:
        validation = Path(validation_dir)
        generate_hidden_states(
            validation,
            args.data,
            [by_name[name] for name in sample_order[train_size:]],
            args.verifier,
            args.max_sequence_length,
        )

        first_epoch = epoch
        for epoch in range(first_epoch, args.epochs + 1):
            if epoch > first_epoch:
                shuffle_training(epoch)
                chunk_index = next_sample = 0
            training = [by_name[name] for name in sample_order[next_sample:train_size]]
            for start in range(0, len(training), chunk_size):
                chunk_index += 1
                chunk_sources = training[start : start + chunk_size]
                with tempfile.TemporaryDirectory() as chunk_dir:
                    chunk = Path(chunk_dir)
                    generate_hidden_states(
                        chunk,
                        args.data,
                        chunk_sources,
                        args.verifier,
                        args.max_sequence_length,
                    )
                    run(
                        "tiny_speculators.scripts.train_eagle3",
                        data=chunk,
                        validation_data=validation,
                        verifier=args.verifier,
                        output=checkpoint,
                        epoch=epoch,
                        chunk_index=chunk_index,
                        max_batch_tokens=args.max_sequence_length,
                        resume=resume,
                    )
                resume = checkpoint
                next_sample += len(chunk_sources)
                state_path = checkpoint / "trainer_state.pt"
                state = torch.load(state_path, map_location="cpu", weights_only=True)
                state.update(
                    sample_order=sample_order,
                    train_size=train_size,
                    next_sample=next_sample,
                )
                torch.save(state, state_path)


def main() -> None:
    args = parse_args()
    start = STAGES.index(args.start_at or ("train" if args.resume else "prepare"))
    stop = STAGES.index(args.stop_after)
    if start > stop:
        raise ValueError("--start-at must not come after --stop-after")

    checkpoint = args.checkpoint or args.resume or Path("checkpoints/eagle3")
    if checkpoint.resolve() == args.exported_checkpoint.resolve():
        raise ValueError("--checkpoint and --exported-checkpoint must differ")

    def should_run(stage: str) -> bool:
        return start <= STAGES.index(stage) <= stop

    if should_run("prepare"):
        run(
            "tiny_speculators.scripts.prepare_data",
            model=args.verifier,
            output=args.data,
            max_length=args.max_sequence_length,
            max_samples=args.max_samples,
        )
    if should_run("vocab"):
        run("tiny_speculators.scripts.vocab_mapping", data=args.data)
    if should_run("train"):
        train(args, checkpoint)
    if should_run("export"):
        run(
            "tiny_speculators.scripts.export_vllm",
            checkpoint=checkpoint,
            output=args.exported_checkpoint,
            verifier=args.verifier,
        )


if __name__ == "__main__":
    main()
