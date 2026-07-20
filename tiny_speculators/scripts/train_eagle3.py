"""End-To-End Training script for EAGLE-3 Draft Model
1. create & load dataset
2. load vocab mapping
3. load verifier (for token emb, rmsnorm, lmhead)
4. initialize Eagle3Config
5. initialize Eagle3DraftModel
6. initialize optimizer
7. train packed batches
8. save checkpoint
"""

from pathlib import Path
import argparse
import json
import time
from safetensors.torch import load_file
from tiny_speculators.eagle3.data import SampleFileDataset, packed_batches
from tiny_speculators.eagle3.vocab import load_vocab_mapping
from tiny_speculators.config import Config, validate_qwen3_8b_config
from transformers.models.qwen3 import Qwen3ForCausalLM
import torch
from tiny_speculators.eagle3.model import Eagle3DraftModel
from tiny_speculators.eagle3.config import Eagle3SpeculatorConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Config.data_dir)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--verifier", type=str, default=Config.verifier)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--output", type=Path, default=Path("checkpoints/eagle3"))
    parser.add_argument("--metrics-file", type=Path)
    parser.add_argument(
        "--max-batch-tokens",
        type=int,
        default=Config.max_model_len,
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train-ratio", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--resume",
        type=Path,
        help="Checkpoint directory to resume",
    )

    return parser.parse_args()


def save_checkpoint(
    path: Path,
    model: Eagle3DraftModel,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    best_val_loss: float,
    max_samples: int | None,
    train_ratio: float,
    seed: int,
) -> None:
    """Save everything required to resume at the next epoch."""

    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    torch.save(optimizer.state_dict(), path / "optimizer_state_dict.pt")
    torch.save(
        {
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "max_samples": max_samples,
            "train_ratio": train_ratio,
            "seed": seed,
        },
        path / "trainer_state.pt",
    )


def load_trainer_state(path: Path) -> dict:
    """Load the small CPU-only state used to reconstruct a training run."""

    trainer_state_path = path / "trainer_state.pt"
    if not trainer_state_path.exists():
        raise FileNotFoundError(f"Resume file not found: {trainer_state_path}")
    return torch.load(
        trainer_state_path,
        map_location="cpu",
        weights_only=True,
    )


def load_checkpoint(
    path: Path,
    model: Eagle3DraftModel,
    optimizer: torch.optim.Optimizer,
) -> tuple[int, float]:
    """Restore model and optimizer state and return the next epoch to run."""

    weights_path = path / "model.safetensors"
    optimizer_path = path / "optimizer_state_dict.pt"
    for required_path in (weights_path, optimizer_path):
        if not required_path.exists():
            raise FileNotFoundError(f"Resume file not found: {required_path}")

    device = next(model.parameters()).device
    incompatible = model.load_state_dict(
        load_file(weights_path, device=str(device)),
        strict=False,
    )
    missing = set(incompatible.missing_keys)
    unexpected = set(incompatible.unexpected_keys)
    if missing != {"verifier_norm.weight"} or unexpected:
        raise RuntimeError(
            "Checkpoint does not match the model: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )

    optimizer.load_state_dict(
        torch.load(optimizer_path, map_location=device, weights_only=True)
    )
    trainer_state = load_trainer_state(path)
    return (
        int(trainer_state["epoch"]) + 1,
        float(trainer_state["best_val_loss"]),
    )

def main() -> None:
    args = parse_args()
    metrics_file = args.metrics_file
    if metrics_file is not None:
        metrics_file.parent.mkdir(parents=True, exist_ok=True)
    metrics_stream = metrics_file.open("a") if metrics_file is not None else None

    def record(phase: str, epoch: int, step: int, metrics: dict) -> None:
        if metrics_stream is not None:
            metrics_stream.write(
                json.dumps({
                    "time": time.time(),
                    "phase": phase,
                    "epoch": epoch,
                    "step": step,
                    **metrics,
                }) + "\n"
            )
            metrics_stream.flush()
    if args.resume is None:
        args.train_ratio = 0.9 if args.train_ratio is None else args.train_ratio
        args.seed = 42 if args.seed is None else args.seed
    else:
        trainer_state = load_trainer_state(args.resume)
        saved_max_samples = trainer_state["max_samples"]
        if args.max_samples is None:
            args.max_samples = saved_max_samples
        elif args.max_samples != saved_max_samples:
            raise ValueError(
                "--max-samples must match the resumed run: "
                f"expected {saved_max_samples}, got {args.max_samples}"
            )
        for name in ("train_ratio", "seed"):
            saved_value = trainer_state[name]
            current_value = getattr(args, name)
            if current_value is None:
                setattr(args, name, saved_value)
            elif current_value != saved_value:
                raise ValueError(
                    f"--{name.replace('_', '-')} must match the resumed run: "
                    f"expected {saved_value}, got {current_value}"
                )

    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    if args.max_batch_tokens < 1:
        raise ValueError("--max-batch-tokens must be at least 1")
    if not 0.0 < args.train_ratio < 1.0:
        raise ValueError("--train-ratio must be between 0 and 1")

    # 1. create & load dataset
    dataset = SampleFileDataset(
        data_dir=args.data,
        max_samples=args.max_samples
    )
    if not dataset:
        raise FileNotFoundError(f"Data directory not found: {args.data}")
    if len(dataset) < 2:
        raise ValueError("Training with validation requires at least 2 samples")

    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(dataset), generator=generator).tolist()
    train_size = max(1, min(len(dataset) - 1, int(len(dataset) * args.train_ratio)))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # 2. load vocab mapping
    vocab_mapping_path = args.data / "vocab_mapping.pt"
    if not vocab_mapping_path.exists():
        raise FileNotFoundError(f"Vocab mapping directory not found: {args.data / "vocab_mapping.pt"}")
    draft_to_target_offsets, target_to_draft = load_vocab_mapping(vocab_mapping_path)


    # 3. load verifier (for token emb, rmsnorm, lmhead)
    verifier = Qwen3ForCausalLM.from_pretrained(
        args.verifier,
        dtype=(
            torch.bfloat16 if torch.cuda.is_bf16_supported()
            else torch.float32
        )
    ).to("cuda")
    validate_qwen3_8b_config(verifier.config)

    # 4. initialize Eagle3Config
    if args.resume is None:
        config = Eagle3SpeculatorConfig(
            transformer_layer_config=verifier.config,
            draft_vocab_size=draft_to_target_offsets.numel(),
            eagle_aux_hidden_state_layer_ids=list(
                Config.eagle_aux_hidden_state_layer_ids
            ),
        )
    else:
        config = Eagle3SpeculatorConfig.from_pretrained(args.resume)
        validate_qwen3_8b_config(
            config.transformer_layer_config,
            num_hidden_layers=1,
        )

    # 5. initialize Eagle3DraftModel
    model = Eagle3DraftModel(
        config=config,
        verifier=verifier,
        draft_to_target_offsets=draft_to_target_offsets,
        target_to_draft=target_to_draft,
    )

    del verifier # remove verifier memory after initializing draft model

    device = next(model.parameters()).device

    # 6. initialize optimizer
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    start_epoch = 1
    best_val_loss = float("inf")
    if args.resume is not None:
        start_epoch, best_val_loss = load_checkpoint(args.resume, model, optimizer)
        if start_epoch > args.epochs:
            raise ValueError(
                f"Checkpoint completed epoch {start_epoch - 1}, but --epochs is "
                f"{args.epochs}; set --epochs to at least {start_epoch}"
            )
        print(
            f"Resuming from {args.resume} at epoch {start_epoch} "
            f"with best_val_loss={best_val_loss:.6f}",
            flush=True,
        )

    # 7. train packed batches
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        train_totals: dict[str, float] = {}
        epoch_generator = torch.Generator().manual_seed(args.seed + epoch)
        shuffled_offsets = torch.randperm(
            len(train_indices), generator=epoch_generator
        ).tolist()
        shuffled_indices = [train_indices[offset] for offset in shuffled_offsets]
        train_steps = 0
        for step, batch in enumerate(
            packed_batches(dataset, shuffled_indices, args.max_batch_tokens),
            start=1,
        ):
            train_steps = step
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss, _draft_tokens, metrics = model(**batch)

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at step {step}: {loss.item()}")

            loss.backward()
            optimizer.step()
            for key, value in metrics.items():
                train_totals[key] = train_totals.get(key, 0.0) + value.item()
            metric_text = " ".join(
                f"{key}={value.item():.4f}" for key, value in metrics.items()
            )
            print(f"epoch={epoch} step={step} {metric_text}", flush=True)
            record(
                "train",
                epoch,
                step,
                {key: value.item() for key, value in metrics.items()},
            )

        train_metrics = {
            key: value / train_steps
            for key, value in train_totals.items()
        }

        model.eval()
        val_totals: dict[str, float] = {}
        val_steps = 0
        with torch.no_grad():
            for val_steps, batch in enumerate(
                packed_batches(dataset, val_indices, args.max_batch_tokens),
                start=1,
            ):
                batch = {key: value.to(device) for key, value in batch.items()}
                val_loss, _draft_tokens, metrics = model(**batch)
                if not torch.isfinite(val_loss):
                    raise RuntimeError(
                        f"Non-finite validation loss at epoch {epoch}: "
                        f"{val_loss.item()}"
                    )
                for key, value in metrics.items():
                    val_totals[key] = val_totals.get(key, 0.0) + value.item()
                record(
                    "validation",
                    epoch,
                    val_steps,
                    {key: value.item() for key, value in metrics.items()},
                )

        val_metrics = {
            key: value / val_steps
            for key, value in val_totals.items()
        }
        print(
            f"epoch={epoch} "
            f"train_loss={train_metrics['loss']:.6f} "
            f"val_loss={val_metrics['loss']:.6f}",
            flush=True,
        )
        is_best = val_metrics["loss"] < best_val_loss
        record(
            "epoch",
            epoch,
            train_steps,
            {
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"val_{key}": value for key, value in val_metrics.items()},
                "best": is_best,
            },
        )

        if is_best:
            best_val_loss = val_metrics["loss"]
            best_path = args.output / "best"
            save_checkpoint(
                best_path,
                model,
                optimizer,
                epoch=epoch,
                best_val_loss=best_val_loss,
                max_samples=args.max_samples,
                train_ratio=args.train_ratio,
                seed=args.seed,
            )
            print(f"Saved best checkpoint to {best_path}", flush=True)

    # 8. save final checkpoint
    save_checkpoint(
        args.output,
        model,
        optimizer,
        epoch=args.epochs,
        best_val_loss=best_val_loss,
        max_samples=args.max_samples,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )
    print(f"Saved checkpoint to {args.output}", flush=True)
    if metrics_stream is not None:
        metrics_stream.close()

if __name__ == "__main__":
    main()
