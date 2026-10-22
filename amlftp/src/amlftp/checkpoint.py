"""Checkpoints for resume and for read-only inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from amlftp.config import TrainConfig


class SeedMismatchError(RuntimeError):
    """Raised when ``--seed`` does not match the checkpoint being resumed."""


class ResumeConfigError(RuntimeError):
    """Raised when a resume checkpoint belongs to a different setup."""


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
    labels: list[str]
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
    labels: list[str]
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
            "labels": state.labels,
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
        labels=[str(label) for label in cast(list[object], payload["labels"])],
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
            "labels": state.labels,
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
        labels=[str(label) for label in cast(list[object], payload["labels"])],
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
        raise SeedMismatchError(f"Seed в чекпоинте ({state.seed}) не совпадает с --seed ({config.seed}).")
    if state.max_per_class != config.max_per_class:
        raise ResumeConfigError(
            f"max_per_class в чекпоинте ({state.max_per_class}) "
            f"не совпадает с --max-per-class ({config.max_per_class})."
        )
    architecture_matches = (
        state.d_model == config.d_model
        and state.n_heads == config.n_heads
        and state.dropout == config.dropout
        and state.image_dim == config.image_dim
        and state.text_dim == config.text_dim
        and state.image_size == config.image_size
        and state.max_length == config.max_length
        and state.labels == list(config.allowed_labels)
    )
    if not architecture_matches:
        raise ResumeConfigError("Архитектура или список классов чекпоинта не совпадает с текущим запуском.")


def _as_history(value: object) -> dict[str, list[float]]:
    if not isinstance(value, dict):
        raise TypeError("История в чекпоинте повреждена")
    history: dict[str, list[float]] = {}
    for key, item in value.items():
        if not isinstance(item, list):
            raise TypeError(f"История '{key}' в чекпоинте повреждена")
        history[str(key)] = [float(number) for number in item]
    return history
