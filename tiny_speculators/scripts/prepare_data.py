"""Tokenize ShareGPT conversations and save final-answer training targets."""

from __future__ import annotations
import argparse
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer
import json
from collections import Counter

from tiny_speculators.config import Config, validate_qwen3_8b_config

def preprocess(row: dict) -> list[dict[str, str]] | None:
    """Convert one ShareGPT row to the chat-template message format."""
    role_map = {
                "human": "user",
                "gpt": "assistant",
                "system": "system"
                }
    
    turns = row.get("conversations")
    if not isinstance(turns, list):
        return None

    messages = []
    for turn in turns:
        if isinstance(turn, str):
            try:
                turn = json.loads(turn)
            except json.JSONDecodeError:
                return None
        elif not isinstance(turn, dict):
            return None
        role = role_map.get(turn.get("from"))
        content = turn.get("value")
        if role is None or not isinstance(content, str) or not content.strip():
            return None
        messages.append({"role": role, "content": content})
    return messages if messages and messages[-1]["role"] == "assistant" else None


def prepare_sample(tokenizer, messages: list[dict[str, str]]) -> tuple[list[int], list[bool]]:
    """Return full conversation tokens and a mask for its final assistant answer."""
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=Config.enable_thinking,
    )
    answer = messages[-1]["content"]
    answer_start = full_text.rfind(answer)
    if answer_start == -1:
        raise ValueError("Could not find the final assistant answer in the chat template")
    answer_end = answer_start + len(answer)

    encoded = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
    loss_mask = [
        token_start < answer_end and token_end > answer_start
        for token_start, token_end in encoded.offset_mapping
    ]
    return encoded.input_ids, loss_mask


def prepare(
    *, model: str, data: str, output: Path, max_samples: int | None, max_length: int
) -> tuple[int, int]:
    """Tokenize valid conversations and save samples with vocabulary statistics."""

    validate_qwen3_8b_config(AutoConfig.from_pretrained(model))
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    output.mkdir(parents=True, exist_ok=True)
    tokenized_dir = output / "tokenized"
    tokenized_dir.mkdir(parents=True, exist_ok=True)
    # len(tokenizer) includes any added tokens, unlike tokenizer.vocab_size.
    token_frequencies: Counter[int] = Counter()
    
    saved = skipped = 0
    for row in load_dataset(data, split="train", streaming=True):
        messages = preprocess(row)
        if messages is None:
            skipped += 1
            continue
        input_ids, loss_mask = prepare_sample(tokenizer, messages)
        if len(input_ids) > max_length:
            skipped += 1
            continue
        input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
        loss_mask_tensor = torch.tensor(loss_mask, dtype=torch.bool)
        
        torch.save(
            {
                "input_ids": input_ids_tensor,
                "loss_mask": loss_mask_tensor,
            },
            tokenized_dir / f"data_{saved:06d}.pt",
        )

        target_ids = input_ids_tensor[loss_mask_tensor]
        token_frequencies.update(target_ids.tolist())

        saved += 1
        if max_samples is not None and saved >= max_samples:
            break

    torch.save(
        {
            "model": model,
            "vocab_size": len(tokenizer),
            "token_frequencies": dict(token_frequencies),
            "target_total": token_frequencies.total(),
        },
        output / "vocab_info.pt",
    )
    return saved, skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=Config.verifier)
    parser.add_argument("--data", default=Config.dataset)
    parser.add_argument("--output", type=Path, default=Config.data_dir)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-length", type=int, default=Config.max_model_len)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    saved, skipped = prepare(**vars(args))
    print(f"Saved {saved} samples to {args.output}; skipped {skipped} samples.")


if __name__ == "__main__":
    main()
