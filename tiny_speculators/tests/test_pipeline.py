import sys

import pytest

from tiny_speculators.scripts import pipeline


def test_pipeline_forwards_shared_sequence_limit(monkeypatch):
    calls = []
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pipeline",
            "--max-samples", "10",
            "--max-sequence-length", "4096",
            "--stop-after", "train",
        ],
    )
    monkeypatch.setattr(
        pipeline,
        "run",
        lambda module, *args: calls.append((module, args)),
    )

    pipeline.main()

    assert [module for module, _args in calls] == [
        "tiny_speculators.scripts.prepare_data",
        "tiny_speculators.scripts.vocab_mapping",
        "tiny_speculators.scripts.generate_hidden_states",
        "tiny_speculators.scripts.train_eagle3",
    ]
    assert ("--max-length", "4096") in list(zip(calls[0][1][::2], calls[0][1][1::2]))
    assert ("--max-model-len", "4096") in list(zip(calls[2][1][::2], calls[2][1][1::2]))
    assert ("--max-batch-tokens", "4096") in list(zip(calls[3][1][::2], calls[3][1][1::2]))


def test_pipeline_can_restart_at_training(monkeypatch):
    calls = []
    monkeypatch.setattr(
        sys,
        "argv",
        ["pipeline", "--start-at", "train", "--stop-after", "train"],
    )
    monkeypatch.setattr(
        pipeline,
        "run",
        lambda module, *args: calls.append((module, args)),
    )

    pipeline.main()

    assert [module for module, _args in calls] == [
        "tiny_speculators.scripts.train_eagle3"
    ]


def test_pipeline_rejects_reversed_stage_range(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["pipeline", "--start-at", "train", "--stop-after", "hidden"],
    )

    with pytest.raises(ValueError, match="must not come after"):
        pipeline.main()
