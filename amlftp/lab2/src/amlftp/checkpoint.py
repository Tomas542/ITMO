"""Checkpoints for resume and for read-only inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from amlftp.config import TrainConfig


class SeedMismatchError(RuntimeError):
    """Resume checkpoint was created with a different seed."""


@dataclass
class ResumeState:
    """Full training state written to ``weights/last.pt``."""

    completed_epochs: int
    seed: int
    max_per_class: int
    best_macro_f1: float
    history: dict[str, list[float]]
    model_state: dict[str, torch.Tensor]
    best_model_state: dict[str, torch.Tensor]
    optimizer_state: dict[str, Any]
    rng: dict[str, Any]
    task_labels: dict[str, list[str]]
    d_model: int
    n_heads: int
    dropout: float
    image_dim: int
    text_dim: int
    image_size: int
    max_length: int


@dataclass
class InferenceState:
    """Best fused weights written to ``weights/model.pt``."""

    model_state: dict[str, torch.Tensor]
    task_labels: dict[str, list[str]]
    seed: int
    d_model: int
    n_heads: int
    dropout: float
    image_dim: int
    text_dim: int
    image_size: int
    max_length: int
    text_model_name: str


def cpu_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    """Clone a module state dict onto CPU so later steps cannot mutate it."""
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def save_resume(path: Path, state: ResumeState) -> None:
    """Write the resume checkpoint. Callers create the parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "completed_epochs": state.completed_epochs,
            "seed": state.seed,
            "max_per_class": state.max_per_class,
            "best_macro_f1": state.best_macro_f1,
            "history": state.history,
            "model_state": state.model_state,
            "best_model_state": state.best_model_state,
            "optimizer_state": state.optimizer_state,
            "rng": state.rng,
            "task_labels": state.task_labels,
            "d_model": state.d_model,
            "n_heads": state.n_heads,
            "dropout": state.dropout,
            "image_dim": state.image_dim,
            "text_dim": state.text_dim,
            "image_size": state.image_size,
            "max_length": state.max_length,
        },
        path,
    )


def load_resume(path: Path) -> ResumeState:
    """Read a resume checkpoint onto CPU."""
    payload = cast(dict[str, Any], torch.load(path, map_location="cpu", weights_only=False))
    return ResumeState(
        completed_epochs=int(payload["completed_epochs"]),
        seed=int(payload["seed"]),
        max_per_class=int(payload["max_per_class"]),
        best_macro_f1=float(payload["best_macro_f1"]),
        history=_as_history(payload["history"]),
        model_state=cast(dict[str, torch.Tensor], payload["model_state"]),
        best_model_state=cast(dict[str, torch.Tensor], payload["best_model_state"]),
        optimizer_state=cast(dict[str, Any], payload["optimizer_state"]),
        rng=cast(dict[str, Any], payload["rng"]),
        task_labels=_as_task_labels(payload["task_labels"]),
        d_model=int(payload["d_model"]),
        n_heads=int(payload["n_heads"]),
        dropout=float(payload["dropout"]),
        image_dim=int(payload["image_dim"]),
        text_dim=int(payload["text_dim"]),
        image_size=int(payload["image_size"]),
        max_length=int(payload["max_length"]),
    )


def save_inference(path: Path, state: InferenceState) -> None:
    """Write the best weights used by the demo."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": state.model_state,
            "task_labels": state.task_labels,
            "seed": state.seed,
            "d_model": state.d_model,
            "n_heads": state.n_heads,
            "dropout": state.dropout,
            "image_dim": state.image_dim,
            "text_dim": state.text_dim,
            "image_size": state.image_size,
            "max_length": state.max_length,
            "text_model_name": state.text_model_name,
        },
        path,
    )


def load_inference(path: Path) -> InferenceState:
    """Read inference weights onto CPU."""
    payload = cast(dict[str, Any], torch.load(path, map_location="cpu", weights_only=False))
    return InferenceState(
        model_state=cast(dict[str, torch.Tensor], payload["model_state"]),
        task_labels=_as_task_labels(payload["task_labels"]),
        seed=int(payload["seed"]),
        d_model=int(payload["d_model"]),
        n_heads=int(payload["n_heads"]),
        dropout=float(payload["dropout"]),
        image_dim=int(payload["image_dim"]),
        text_dim=int(payload["text_dim"]),
        image_size=int(payload["image_size"]),
        max_length=int(payload["max_length"]),
        text_model_name=str(payload["text_model_name"]),
    )


def validate_resume(state: ResumeState, config: TrainConfig) -> None:
    """Stop resume when the seed or the setup differs from the checkpoint."""
    if state.seed != config.seed:
        msg = f"Seed в чекпоинте ({state.seed}) не совпадает с --seed ({config.seed})."
        raise SeedMismatchError(msg)
    if state.max_per_class != config.max_per_class:
        msg = (
            f"max_per_class в чекпоинте ({state.max_per_class}) не совпадает "
            f"с --max-per-class ({config.max_per_class})."
        )
        raise RuntimeError(msg)
    expected_names = [name for name, _labels in config.task_labels]
    expected_labels = {name: list(labels) for name, labels in config.task_labels}
    labels_match = list(state.task_labels) == expected_names and state.task_labels == expected_labels
    architecture_matches = (
        state.d_model == config.d_model
        and state.n_heads == config.n_heads
        and state.dropout == config.dropout
        and state.image_dim == config.image_dim
        and state.text_dim == config.text_dim
        and state.image_size == config.image_size
        and state.max_length == config.max_length
        and labels_match
    )
    if not architecture_matches:
        msg = "Архитектура или списки классов чекпоинта не совпадают с текущим запуском."
        raise RuntimeError(msg)


def _as_history(value: object) -> dict[str, list[float]]:
    if not isinstance(value, dict):
        raise TypeError("История в чекпоинте повреждена")
    history: dict[str, list[float]] = {}
    for key, item in value.items():
        if not isinstance(item, list):
            raise TypeError(f"История '{key}' в чекпоинте повреждена")
        history[str(key)] = [float(number) for number in item]
    return history


def _as_task_labels(value: object) -> dict[str, list[str]]:
    if not isinstance(value, dict):
        raise TypeError("Списки классов в чекпоинте повреждены")
    labels: dict[str, list[str]] = {}
    for key, item in value.items():
        if not isinstance(item, list):
            raise TypeError(f"Классы задачи '{key}' в чекпоинте повреждены")
        labels[str(key)] = [str(label) for label in item]
    return labels
