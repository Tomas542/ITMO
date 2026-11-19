"""Random seed control for reproducible and resumable runs."""

from __future__ import annotations

import random
from typing import Any, cast

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Fix Python, NumPy, PyTorch and CUDA generators before any random step."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def capture_rng_state() -> dict[str, Any]:
    """Snapshot generator states so a later run can continue the same stream."""
    cuda_state: list[torch.Tensor] | None = None
    if torch.cuda.is_available():
        cuda_state = torch.cuda.get_rng_state_all()
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": cuda_state,
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore a snapshot produced by :func:`capture_rng_state`."""
    random.setstate(cast(Any, state["python"]))
    np.random.set_state(cast(Any, state["numpy"]))
    torch.set_rng_state(cast(torch.Tensor, state["torch"]))
    cuda_state = state.get("cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cast(list[torch.Tensor], cuda_state))
