"""Reduced vocabulary for Draft LMHead"""

import torch
import argparse
from tiny_speculators.config import Config
from pathlib import Path
from tiny_speculators.eagle3.vocab import build_vocab_mappings
from tiny_speculators.eagle3.config import DEFAULT_DRAFT_VOCAB_SIZE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build vocabulary mappings from token frequency distribution"
    )

    parser.add_argument(
        "--draft-vocab-size",
        type=int,
        default=DEFAULT_DRAFT_VOCAB_SIZE,
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=Config.data_dir,
        help="Directory containing vocab_info.pt",
    )

    parser.add_argument(
        "--output-path",
        type=Path,
        help="Defaults to <data>/vocab_mapping.pt",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    path = args.data / "vocab_info.pt"
    if not path.exists():
        raise FileNotFoundError(f"Token frequency file not found: {path}")

    output_path = args.output_path or args.data / "vocab_mapping.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    vocab_info = torch.load(path, weights_only=True)
    token_frequencies: dict[int, int] = vocab_info["token_frequencies"]
    target_vocab_size = vocab_info["vocab_size"]

    draft_to_target_offsets, target_to_draft = build_vocab_mappings(
        token_frequencies=token_frequencies,
        draft_vocab_size=args.draft_vocab_size,
        target_vocab_size=target_vocab_size,
    )

    torch.save(
        {
            "draft_to_target_offsets": draft_to_target_offsets,
            "target_to_draft": target_to_draft,
        },
        output_path,
    )


if __name__ == "__main__":
    main()
