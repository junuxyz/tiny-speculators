from argparse import Namespace
from pathlib import Path

import pytest
import torch

from tiny_speculators.scripts import pipeline


def make_args(data: Path, output: Path, chunk_size: int | None, resume=None):
    return Namespace(
        data=data,
        verifier="test",
        chunk_size=chunk_size,
        max_samples=None,
        epochs=1,
        max_sequence_length=16,
        resume=resume,
    )


def make_data(tmp_path: Path) -> Path:
    data = tmp_path / "data"
    (data / "tokenized").mkdir(parents=True)
    (data / "vocab_mapping.pt").touch()
    for index in range(10):
        (data / "tokenized" / f"data_{index}.pt").touch()
    return data


def save_fake_state(options: dict, output: Path) -> None:
    validation = Path(options["validation_data"])
    output.mkdir(exist_ok=True)
    torch.save(
        {
            "epoch": options["epoch"],
            "chunk_index": options["chunk_index"],
            "best_val_loss": 1.0,
            "validation_samples": [
                path.name for path in (validation / "tokenized").iterdir()
            ],
        },
        output / "trainer_state.pt",
    )


def test_chunk_order_and_offset_survive_restart(tmp_path, monkeypatch):
    data = make_data(tmp_path)
    output = tmp_path / "checkpoint"
    completed = []
    attempts = 0
    fail = True

    def fake_run(module, **options):
        nonlocal attempts, fail
        if not module.endswith("train_eagle3"):
            return
        attempts += 1
        chunk = Path(options["data"])
        names = sorted(path.name for path in (chunk / "tokenized").iterdir())
        if fail and attempts == 2:
            raise RuntimeError("interrupted")
        completed.append(names)
        save_fake_state(options, output)

    monkeypatch.setattr(pipeline, "run", fake_run)
    with pytest.raises(RuntimeError, match="interrupted"):
        pipeline.train(make_args(data, output, 3), output)

    fail = False
    pipeline.train(make_args(data, output, 3, output), output)

    state = torch.load(output / "trainer_state.pt", weights_only=True)
    expected = torch.randperm(10, generator=torch.Generator().manual_seed(42))
    expected = [f"data_{index}.pt" for index in expected.tolist()[:9]]
    assert completed == [sorted(expected[i : i + 3]) for i in range(0, 9, 3)]
    assert state["sample_order"][:9] == expected
    assert state["next_sample"] == 9


def test_missing_chunk_size_trains_all_samples_together(tmp_path, monkeypatch):
    data = make_data(tmp_path)
    output = tmp_path / "checkpoint"
    completed = []

    def fake_run(module, **options):
        if module.endswith("train_eagle3"):
            chunk = Path(options["data"])
            completed.append(list((chunk / "tokenized").iterdir()))
            save_fake_state(options, output)

    monkeypatch.setattr(pipeline, "run", fake_run)
    pipeline.train(make_args(data, output, None), output)

    assert [len(chunk) for chunk in completed] == [9]
