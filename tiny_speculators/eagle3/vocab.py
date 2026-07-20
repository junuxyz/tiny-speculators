from pathlib import Path
import torch


def load_vocab_mapping(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Load draft-to-target and target-to-draft vocabulary mappings."""

    mappings = torch.load(path, weights_only=True)

    draft_to_target_offsets = mappings["draft_to_target_offsets"].to(dtype=torch.long)
    target_to_draft = mappings["target_to_draft"].to(dtype=torch.bool)

    return draft_to_target_offsets, target_to_draft


def get_selected_target_ids(
    target_to_draft: torch.Tensor,
) -> torch.Tensor:
    """Return target token IDs selected by the draft-vocabulary mask."""

    return target_to_draft.nonzero(as_tuple=False).flatten()


def build_vocab_mappings(
    token_frequencies: dict[int, int],
    draft_vocab_size: int,
    target_vocab_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build compact vocabulary mappings from target-token frequencies."""

    if not 0 < draft_vocab_size <= target_vocab_size:
        raise ValueError(f"draft_vocab_size must be between 1 and {target_vocab_size}")

    sorted_tokens = sorted(
        token_frequencies, key=lambda tid: (-token_frequencies[tid], tid)
    )

    selected_token_ids = sorted_tokens[:draft_vocab_size]

    if len(selected_token_ids) < draft_vocab_size:
        num_observed = len(selected_token_ids)
        num_missing = draft_vocab_size - num_observed

        print(  # log
            f"Only {num_observed:,} observed tokens are available. "
            f"filling {num_missing:,} remaining draft vocabulary slots "
            f"with the lowest unused token IDs."
        )

        current_ids = set(selected_token_ids)
        for tid in range(target_vocab_size):
            if tid not in current_ids:
                selected_token_ids.append(tid)
            if len(selected_token_ids) >= draft_vocab_size:
                break

    selected_token_ids.sort()

    # target_token_id = draft_idx + draft_to_target_offsets[draft_idx]
    draft_to_target_offsets = torch.tensor(
        selected_token_ids, dtype=torch.long
    ) - torch.arange(draft_vocab_size, dtype=torch.long)

    target_to_draft = torch.zeros(target_vocab_size, dtype=torch.bool)
    target_to_draft[selected_token_ids] = True

    return draft_to_target_offsets, target_to_draft
