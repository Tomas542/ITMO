"""Cross-attention shapes, masks and the separation of unimodal heads."""

from __future__ import annotations

import inspect

import torch
from torch import nn

from amlftp.model import TokenFusion


def _fusion() -> TokenFusion:
    torch.manual_seed(0)
    return TokenFusion(image_dim=16, text_dim=16, d_model=32, n_heads=4, n_classes=3, dropout=0.0)


def test_fusion_is_bidirectional_cross_attention() -> None:
    model = _fusion()
    attentions = [module for module in model.modules() if isinstance(module, nn.MultiheadAttention)]
    assert len(attentions) == 2
    assert isinstance(model.image_to_text.attention, nn.MultiheadAttention)
    assert isinstance(model.text_to_image.attention, nn.MultiheadAttention)


def test_image_baseline_does_not_take_text() -> None:
    model = _fusion()
    assert "text_tokens" not in inspect.signature(model.image_logits).parameters
    logits = model.image_logits(torch.randn(2, 4, 16))
    assert logits.shape == (2, 3)


def test_attention_weights_sum_to_one() -> None:
    model = _fusion()
    model.eval()
    image = torch.randn(2, 4, 16)
    text = torch.randn(2, 5, 16)
    pad = torch.tensor(
        [[False, False, False, True, True], [False, False, True, True, True]],
    )
    output = model(image, text, pad)
    assert output.logits.shape == (2, 3)
    assert output.attn_image_to_text.shape == (2, 4, 4, 5)
    assert output.attn_text_to_image.shape == (2, 4, 5, 4)
    assert torch.allclose(output.attn_image_to_text.sum(dim=-1), torch.ones(2, 4, 4), atol=1e-4)
    assert torch.allclose(output.attn_text_to_image.sum(dim=-1), torch.ones(2, 4, 5), atol=1e-4)


def test_padding_does_not_change_logits() -> None:
    torch.manual_seed(0)
    model = _fusion()
    model.eval()
    image = torch.randn(2, 4, 16)
    text = torch.randn(2, 5, 16)
    pad = torch.tensor(
        [[False, False, False, True, True], [False, False, True, True, True]],
    )
    first = model(image, text, pad).logits
    changed = torch.where(pad.unsqueeze(-1), torch.randn_like(text), text)
    second = model(image, changed, pad).logits
    assert torch.allclose(first, second, atol=1e-5)


def test_fused_logits_use_both_modalities() -> None:
    torch.manual_seed(0)
    model = _fusion()
    model.eval()
    image = torch.randn(2, 4, 16)
    text = torch.randn(2, 5, 16)
    pad = torch.zeros(2, 5, dtype=torch.bool)
    first = model(image, text, pad).logits
    changed_text = model(image, torch.randn(2, 5, 16), pad).logits
    changed_image = model(torch.randn(2, 4, 16), text, pad).logits
    assert not torch.allclose(first, changed_text, atol=1e-5)
    assert not torch.allclose(first, changed_image, atol=1e-5)
