from pathlib import Path

import torch
from safetensors.torch import load_file
from torch.utils.data import Dataset
from collections.abc import Iterable, Iterator


class SampleFileDataset(Dataset):
    """Load tokenized samples and their matching verifier hidden states."""

    def __init__(
        self,
        data_dir: Path,
        max_samples: int | None = None,
    ):
        self.data_dir = data_dir
        self.data_paths = sorted((data_dir / "tokenized").glob("data_*.pt"))
        if not self.data_paths:
            self.data_paths = sorted(data_dir.glob("data_*.pt"))
        if max_samples is not None:
            self.data_paths = self.data_paths[:max_samples]

    def __len__(self) -> int:
        return len(self.data_paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        data_path = self.data_paths[index]
        data_id = data_path.stem.removeprefix("data_")
        data = torch.load(data_path, weights_only=True)

        hidden_path = self.data_dir / "hidden_states" / f"hs_{data_id}.safetensors"
        hidden = load_file(hidden_path)

        if not torch.equal(data["input_ids"], hidden["token_ids"]):
            raise ValueError(f"Token IDs do not match for {data_path.name}")

        all_hidden_states = hidden["hidden_states"]
        return {
            "input_ids": data["input_ids"],
            "hidden_states": all_hidden_states[:, :-1].flatten(start_dim=1),
            "verifier_last_hidden_states": all_hidden_states[:, -1],
            "loss_mask": data["loss_mask"],
        }


def shift_sample(
    input_ids: torch.Tensor,  # [seq_len]
    hidden_states: torch.Tensor,  # [seq_len, 3 * h]
    verifier_last_hidden_states: torch.Tensor,  # [seq_len, h]
    loss_mask: torch.Tensor,  # [seq_len]
) -> dict[str, torch.Tensor]:
    """Align and batch one EAGLE-3 training sample."""
    seq_len = input_ids.shape[0]
    if seq_len < 2:
        raise ValueError("EAGLE-3 samples must contain at least 2 tokens")

    return {
        "input_ids": input_ids[None, 1:],
        "hidden_states": hidden_states[None, :-1],
        "verifier_last_hidden_states": verifier_last_hidden_states[None, 1:],
        "loss_mask": loss_mask[None, 1:],
        "position_ids": torch.arange(1, seq_len, dtype=torch.long)[None],
    }


def pack_samples(samples: list[dict[str, torch.Tensor]]):
    lengths = torch.tensor(
        [sample["input_ids"].shape[1] for sample in samples],
        dtype=torch.long,
    )
    packed = {
        key: torch.cat(
            [sample[key] for sample in samples],
            dim=1,
        )
        for key in samples[0]
    }
    packed["lengths"] = lengths
    return packed


def packed_batches(
    dataset: Dataset,
    indices: Iterable[int],
    max_tokens: int,
) -> Iterator[dict[str, torch.Tensor]]:

    samples: list[dict[str, torch.Tensor]] = []
    tokens = 0

    for index in indices:
        sample = shift_sample(**dataset[index])
        seq_len = sample["input_ids"].shape[1]

        if seq_len > max_tokens:
            raise ValueError(f"seq_len {seq_len} exceeds max_tokens({max_tokens})")

        if samples and tokens + seq_len > max_tokens:
            yield pack_samples(samples)
            samples = []
            tokens = 0

        samples.append(sample)
        tokens += seq_len

    if samples:
        yield pack_samples(samples)
