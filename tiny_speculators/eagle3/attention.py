"""The attention pattern for iterative TTT branches."""

import torch
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention


def document_ids(lengths: torch.Tensor) -> torch.Tensor:
    """Expand packed sequence lengths into a document ID per token."""

    return torch.repeat_interleave(
        torch.arange(lengths.numel(), device=lengths.device),
        lengths,
    )


def create_ttt_block_mask(
    lengths: torch.Tensor,
    ttt_steps: int,
) -> BlockMask:
    """Isolate packed documents while preserving iterative TTT attention."""

    lengths = lengths.to(dtype=torch.long)
    device = lengths.device
    doc_ids = document_ids(lengths)
    total_seq_len = doc_ids.numel()

    def mask_mod(_batch, _head, q_index, kv_index):
        anchor_index = kv_index % total_seq_len
        causal = q_index >= kv_index
        valid_query = q_index < total_seq_len
        safe_q_index = q_index.clamp(max=total_seq_len - 1)
        same_document = doc_ids[safe_q_index] == doc_ids[anchor_index]
        same_anchor = q_index == anchor_index

        return valid_query & ((causal & same_document) | same_anchor)

    return create_block_mask(
        mask_mod,
        B=None,
        H=None,
        Q_LEN=total_seq_len,
        KV_LEN=ttt_steps * total_seq_len,
        device=device,
    )


def flex_attention_forward(
        _module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: BlockMask,
        *,
        scaling: float,
        **_kwargs,
) -> tuple[torch.Tensor, None]:
    """Run FlexAttention and adapt its output layout for HuggingFace"""

    num_q_heads = query.shape[1]
    num_kv_heads = key.shape[1]
    enable_gqa = (
        num_q_heads > num_kv_heads
        and num_q_heads % num_kv_heads == 0
        )

    attention_output = flex_attention(
        query=query,
        key=key,
        value=value,
        block_mask=attention_mask,
        enable_gqa=enable_gqa,
        scale=scaling,
    )
    # FlexAttention returns shape: [batch, q_heads, seq, head_dim]
    # HuggingFace Qwen attention expects shape: [batch, seq, q_heads, head_dim]
    attention_output = attention_output.transpose(1, 2).contiguous()

    return attention_output, None
