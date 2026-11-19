"""Cross-attention fusion of image tokens and text tokens."""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn


class FusionOutput(NamedTuple):
    """Logits of the fused classifier and both attention maps."""

    logits: torch.Tensor
    attn_image_to_text: torch.Tensor
    attn_text_to_image: torch.Tensor


def masked_mean(tokens: torch.Tensor, pad_mask: torch.Tensor | None) -> torch.Tensor:
    """Average tokens. ``pad_mask`` is true on positions that must be ignored."""
    if pad_mask is None:
        return tokens.mean(dim=1)
    weights = (~pad_mask).unsqueeze(-1).to(dtype=tokens.dtype)
    denominator = weights.sum(dim=1).clamp(min=1.0)
    return (tokens * weights).sum(dim=1) / denominator


class CrossAttentionBlock(nn.Module):
    """One modality attends to the other: queries from ``query``, keys and values from ``context``."""

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_attention = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm_feed_forward = nn.LayerNorm(d_model)

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        *,
        query_pad: torch.Tensor | None = None,
        context_pad: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attended, weights = self.attention(
            query,
            context,
            context,
            key_padding_mask=context_pad,
            need_weights=True,
            average_attn_weights=False,
        )
        hidden = self.norm_attention(query + attended)
        hidden = self.norm_feed_forward(hidden + self.feed_forward(hidden))
        if query_pad is not None:
            hidden = hidden.masked_fill(query_pad.unsqueeze(-1), 0.0)
        return hidden, weights


class UnimodalHead(nn.Module):
    """Baseline classifier that sees only one modality."""

    def __init__(self, in_dim: int, d_model: int, n_classes: int) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, tokens: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        projected = self.proj(tokens)
        return self.classifier(masked_mean(projected, pad_mask))


class TokenFusion(nn.Module):
    """Fuse modalities with bidirectional cross-attention, then classify.

    Raw image and text embeddings are not concatenated. Each stream is updated by
    ``nn.MultiheadAttention`` whose keys and values come from the other modality.
    The classifier receives the pooled outputs of those two attention blocks.
    Separate unimodal heads never mix the streams and exist only as baselines.
    """

    def __init__(
        self,
        image_dim: int,
        text_dim: int,
        d_model: int,
        n_heads: int,
        n_classes: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.image_proj = nn.Linear(image_dim, d_model)
        self.text_proj = nn.Linear(text_dim, d_model)
        self.image_to_text = CrossAttentionBlock(d_model, n_heads, dropout)
        self.text_to_image = CrossAttentionBlock(d_model, n_heads, dropout)
        self.classifier = nn.Linear(d_model * 2, n_classes)
        self.image_baseline = UnimodalHead(image_dim, d_model, n_classes)
        self.text_baseline = UnimodalHead(text_dim, d_model, n_classes)

    def forward(
        self,
        image_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        text_pad: torch.Tensor,
    ) -> FusionOutput:
        image_hidden = self.image_proj(image_tokens)
        text_hidden = self.text_proj(text_tokens)
        image_fused, attn_image_to_text = self.image_to_text(
            image_hidden,
            text_hidden,
            context_pad=text_pad,
        )
        text_fused, attn_text_to_image = self.text_to_image(
            text_hidden,
            image_hidden,
            query_pad=text_pad,
        )
        pooled = torch.cat(
            [image_fused.mean(dim=1), masked_mean(text_fused, text_pad)],
            dim=-1,
        )
        return FusionOutput(
            logits=self.classifier(pooled),
            attn_image_to_text=attn_image_to_text,
            attn_text_to_image=attn_text_to_image,
        )

    def image_logits(self, image_tokens: torch.Tensor) -> torch.Tensor:
        """Classify from image tokens alone."""
        return self.image_baseline(image_tokens)

    def text_logits(self, text_tokens: torch.Tensor, text_pad: torch.Tensor) -> torch.Tensor:
        """Classify from text tokens alone."""
        return self.text_baseline(text_tokens, text_pad)
