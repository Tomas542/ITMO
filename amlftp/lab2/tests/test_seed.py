"""Seed snapshots used before training and inside resume checkpoints."""

from __future__ import annotations

import torch

from amlftp.seed import capture_rng_state, restore_rng_state, set_seed


def test_set_seed_repeats_tensor() -> None:
    set_seed(123)
    first = torch.randn(4)
    set_seed(123)
    second = torch.randn(4)
    assert torch.equal(first, second)


def test_rng_state_roundtrip() -> None:
    set_seed(123)
    _ = torch.randn(2)
    state = capture_rng_state()
    expected = torch.randn(5)
    set_seed(0)
    _ = torch.randn(8)
    restore_rng_state(state)
    restored = torch.randn(5)
    assert torch.equal(expected, restored)
