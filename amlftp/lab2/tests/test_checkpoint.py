"""Resume checkpoints keep the epoch, seed, optimizer, task labels and weights."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from amlftp.checkpoint import (
    ResumeState,
    SeedMismatchError,
    cpu_state_dict,
    load_resume,
    save_resume,
    validate_resume,
)
from amlftp.config import TrainConfig
from amlftp.model import TokenFusion
from amlftp.predict import require_weight_files
from amlftp.seed import capture_rng_state, restore_rng_state, set_seed

TASK_LABELS = (
    ("masterCategory", ("a", "b", "c")),
    ("gender", ("x", "y")),
)
TASK_LISTS = {"masterCategory": ["a", "b", "c"], "gender": ["x", "y"]}


def _model() -> TokenFusion:
    return TokenFusion(
        image_dim=8,
        text_dim=8,
        d_model=16,
        n_heads=4,
        task_sizes={"masterCategory": 3, "gender": 2},
        dropout=0.0,
    )


def _config(seed: int) -> TrainConfig:
    return TrainConfig(
        seed=seed,
        max_per_class=10,
        d_model=16,
        n_heads=4,
        dropout=0.0,
        image_dim=8,
        text_dim=8,
        task_labels=TASK_LABELS,
    )


def test_resume_checkpoint_roundtrip(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = _model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    image = torch.randn(2, 3, 8)
    text = torch.randn(2, 4, 8)
    pad = torch.zeros(2, 4, dtype=torch.bool)
    target = torch.tensor([[0, 1], [1, 0]])
    logits = model(image, text, pad).logits
    loss = F.cross_entropy(logits["masterCategory"], target[:, 0]) + F.cross_entropy(logits["gender"], target[:, 1])
    loss.backward()
    optimizer.step()

    set_seed(5)
    _ = torch.randn(1)
    rng = capture_rng_state()
    expected = torch.randn(4).clone()
    state = ResumeState(
        completed_epochs=2,
        seed=42,
        max_per_class=10,
        best_macro_f1=0.5,
        history={"train_loss": [0.2]},
        model_state=cpu_state_dict(model),
        best_model_state=cpu_state_dict(model),
        optimizer_state=optimizer.state_dict(),
        rng=rng,
        task_labels=TASK_LISTS,
        d_model=16,
        n_heads=4,
        dropout=0.0,
        image_dim=8,
        text_dim=8,
        image_size=224,
        max_length=32,
    )
    path = tmp_path / "last.pt"
    save_resume(path, state)
    loaded = load_resume(path)
    validate_resume(loaded, _config(42))

    assert loaded.completed_epochs == 2
    assert loaded.seed == 42
    assert loaded.task_labels == TASK_LISTS
    assert list(loaded.task_labels) == ["masterCategory", "gender"]
    assert loaded.optimizer_state["param_groups"][0]["lr"] == pytest.approx(0.01)
    restore_rng_state(loaded.rng)
    assert torch.equal(expected, torch.randn(4))

    restored = _model()
    restored.load_state_dict(loaded.model_state)
    model.eval()
    restored.eval()
    with torch.no_grad():
        original = model(image, text, pad).logits
        copied = restored(image, text, pad).logits
    for name in TASK_LISTS:
        assert torch.allclose(original[name], copied[name])


def test_resume_rejects_a_different_seed(tmp_path: Path) -> None:
    model = _model()
    state = ResumeState(
        completed_epochs=1,
        seed=42,
        max_per_class=10,
        best_macro_f1=0.1,
        history={},
        model_state=cpu_state_dict(model),
        best_model_state=cpu_state_dict(model),
        optimizer_state={},
        rng={},
        task_labels=TASK_LISTS,
        d_model=16,
        n_heads=4,
        dropout=0.0,
        image_dim=8,
        text_dim=8,
        image_size=224,
        max_length=32,
    )
    with pytest.raises(SeedMismatchError):
        validate_resume(state, _config(7))
    assert not (tmp_path / "weights").exists()


def test_missing_weights_are_not_created(tmp_path: Path) -> None:
    missing = tmp_path / "absent"
    with pytest.raises(FileNotFoundError, match="Не найдены веса"):
        require_weight_files(missing)
    assert not missing.exists()
