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

import argparse
import json
from pathlib import Path
import time

from safetensors.torch import load_file
import torch
from transformers.models.qwen3 import Qwen3ForCausalLM

from tiny_speculators.config import Config, validate_qwen3_8b_config
from tiny_speculators.eagle3.config import Eagle3SpeculatorConfig
from tiny_speculators.eagle3.data import SampleFileDataset, packed_batches
from tiny_speculators.eagle3.model import Eagle3DraftModel
from tiny_speculators.eagle3.vocab import load_vocab_mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--verifier", default=Config.verifier)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--output", type=Path, default=Path("checkpoints/eagle3"))
    parser.add_argument("--max-batch-tokens", type=int, default=Config.max_model_len)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--chunk-index", type=int, required=True)
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def save_checkpoint(
    path: Path,
    model: Eagle3DraftModel,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    chunk_index: int,
    best_val_loss: float,
    validation_samples: list[str],
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    torch.save(optimizer.state_dict(), path / "optimizer_state_dict.pt")
    torch.save(
        {
            "epoch": epoch,
            "chunk_index": chunk_index,
            "best_val_loss": best_val_loss,
            "validation_samples": validation_samples,
        },
        path / "trainer_state.pt",
    )


def load_trainer_state(path: Path) -> dict:
    state_path = path / "trainer_state.pt"
    if not state_path.exists():
        raise FileNotFoundError(f"Resume file not found: {state_path}")
    return torch.load(state_path, map_location="cpu", weights_only=True)


def load_checkpoint(
    path: Path,
    model: Eagle3DraftModel,
    optimizer: torch.optim.Optimizer,
) -> None:
    weights_path = path / "model.safetensors"
    optimizer_path = path / "optimizer_state_dict.pt"
    for required_path in (weights_path, optimizer_path):
        if not required_path.exists():
            raise FileNotFoundError(f"Resume file not found: {required_path}")

    device = next(model.parameters()).device
    incompatible = model.load_state_dict(
        load_file(weights_path, device=str(device)), strict=False
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


def run_phase(
    model: Eagle3DraftModel,
    batches,
    device: torch.device,
    phase: str,
    record,
    optimizer: torch.optim.Optimizer | None = None,
    progress: str = "",
) -> tuple[dict[str, float], int]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    step = 0

    with torch.set_grad_enabled(training):
        for step, batch in enumerate(batches, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            if training:
                optimizer.zero_grad(set_to_none=True)
            loss, metrics = model(**batch)
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite {phase} loss at {progress}step={step}: {loss.item()}"
                )
            if training:
                loss.backward()
                optimizer.step()

            values = {key: value.item() for key, value in metrics.items()}
            for key, value in values.items():
                totals[key] = totals.get(key, 0.0) + value
            if training:
                metrics_text = " ".join(
                    f"{key}={value:.4f}" for key, value in values.items()
                )
                print(f"{progress}step={step} {metrics_text}", flush=True)
            record(phase, step, values)

    return {key: value / step for key, value in totals.items()}, step


def main() -> None:
    args = parse_args()
    if args.epoch < 1 or args.chunk_index < 1:
        raise ValueError("--epoch and --chunk-index must be at least 1")
    if args.max_batch_tokens < 1:
        raise ValueError("--max-batch-tokens must be at least 1")

    dataset = SampleFileDataset(args.data)
    validation_dataset = SampleFileDataset(args.validation_data)
    if not dataset or not validation_dataset:
        raise FileNotFoundError("Training and validation data must not be empty")

    mapping_path = args.data / "vocab_mapping.pt"
    if not mapping_path.exists():
        raise FileNotFoundError(f"Vocab mapping not found: {mapping_path}")
    draft_to_target_offsets, target_to_draft = load_vocab_mapping(mapping_path)

    verifier = Qwen3ForCausalLM.from_pretrained(
        args.verifier,
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
    ).to("cuda")
    validate_qwen3_8b_config(verifier.config)

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
        validate_qwen3_8b_config(config.transformer_layer_config, num_hidden_layers=1)

    model = Eagle3DraftModel(
        config,
        verifier,
        draft_to_target_offsets,
        target_to_draft,
    )
    del verifier # remove verifier weights from gpu memory
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_loss = float("inf")
    if args.resume is not None:
        state = load_trainer_state(args.resume)
        load_checkpoint(args.resume, model, optimizer)
        best_val_loss = float(state["best_val_loss"])

    args.output.mkdir(parents=True, exist_ok=True)
    metrics_stream = (args.output / "metrics.jsonl").open("a")

    def record(phase: str, step: int, metrics: dict) -> None:
        metrics_stream.write(
            json.dumps(
                {
                    "time": time.time(),
                    "phase": phase,
                    "epoch": args.epoch,
                    "chunk_index": args.chunk_index,
                    "step": step,
                    **metrics,
                }
            )
            + "\n"
        )
        metrics_stream.flush()

    progress = f"epoch={args.epoch} chunk_index={args.chunk_index} "
    generator = torch.Generator().manual_seed(42 + args.epoch + args.chunk_index)
    train_indices = torch.randperm(len(dataset), generator=generator).tolist()
    train_metrics, train_steps = run_phase(
        model,
        packed_batches(dataset, train_indices, args.max_batch_tokens),
        device,
        "train",
        record,
        optimizer,
        progress,
    )
    validation_indices = range(len(validation_dataset))
    val_metrics, _ = run_phase(
        model,
        packed_batches(validation_dataset, validation_indices, args.max_batch_tokens),
        device,
        "validation",
        record,
    )
    is_best = val_metrics["loss"] < best_val_loss
    record(
        "chunk",
        train_steps,
        {
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
            "best": is_best,
        },
    )

    validation_samples = [path.name for path in validation_dataset.data_paths]
    if is_best:
        best_val_loss = val_metrics["loss"]
        save_checkpoint(
            args.output / "best",
            model,
            optimizer,
            epoch=args.epoch,
            chunk_index=args.chunk_index,
            best_val_loss=best_val_loss,
            validation_samples=validation_samples,
        )
    save_checkpoint(
        args.output,
        model,
        optimizer,
        epoch=args.epoch,
        chunk_index=args.chunk_index,
        best_val_loss=best_val_loss,
        validation_samples=validation_samples,
    )
    metrics_stream.close()
    print(
        f"{progress}train_loss={train_metrics['loss']:.6f} "
        f"val_loss={val_metrics['loss']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
