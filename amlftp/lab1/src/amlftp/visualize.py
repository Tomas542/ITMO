"""Plots of metrics, confusion and cross-attention values."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg", force=True)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
import torch  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from PIL import Image  # noqa: E402

from amlftp.metrics import ModelScore  # noqa: E402

SPECIAL_TOKENS = {"[CLS]", "[SEP]", "[PAD]", "[UNK]"}


def text_to_image_map(attention: torch.Tensor, attention_mask: torch.Tensor) -> np.ndarray:
    """Average text→image attention into a square spatial map."""
    weights = _drop_batch(attention).mean(dim=0)
    valid = attention_mask.bool().to(device=weights.device)
    selected = weights[valid]
    if selected.numel() == 0:
        selected = weights
    values = selected.mean(dim=0).detach().cpu().numpy()
    grid = int(math.isqrt(int(values.size)))
    if grid * grid != int(values.size):
        raise ValueError(f"Число визуальных токенов {values.size} не складывается в квадрат")
    return values.reshape(grid, grid)


def image_to_text_weights(attention: torch.Tensor, attention_mask: torch.Tensor) -> np.ndarray:
    """Average image→text attention over heads and image cells."""
    weights = _drop_batch(attention).mean(dim=0).mean(dim=0)
    mask = attention_mask.to(device=weights.device, dtype=weights.dtype)
    return (weights * mask).detach().cpu().numpy()


def visible_tokens(
    tokens: Sequence[str],
    weights: Sequence[float],
    *,
    limit: int = 8,
) -> tuple[list[str], list[float]]:
    """Keep the strongest content tokens for the attention bar chart."""
    pairs = [
        (token.replace("##", ""), float(weight))
        for token, weight in zip(tokens, weights, strict=True)
        if token not in SPECIAL_TOKENS
    ]
    pairs.sort(key=lambda item: item[1], reverse=True)
    top = pairs[:limit]
    return [token for token, _weight in top], [weight for _token, weight in top]


def plot_history(history: dict[str, list[float]], path: Path) -> Path:
    """Save loss and macro-F1 curves."""
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, history["train_loss"], label="Обучение")
    axes[0].plot(epochs, history["val_loss"], label="Валидация")
    axes[0].set_title("Cross-entropy кросс-attention")
    axes[0].set_xlabel("Эпоха")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[1].plot(epochs, history["train_macro_f1"], label="Обучение, fusion")
    axes[1].plot(epochs, history["val_macro_f1"], label="Валидация, fusion")
    axes[1].plot(epochs, history["image_val_macro_f1"], label="Валидация, фото")
    axes[1].plot(epochs, history["text_val_macro_f1"], label="Валидация, текст")
    axes[1].set_title("Macro-F1")
    axes[1].set_xlabel("Эпоха")
    axes[1].set_ylabel("Macro-F1")
    axes[1].legend()
    return _save(figure, path)


def plot_confusion(confusion: Sequence[Sequence[int]], class_names: Sequence[str], path: Path) -> Path:
    """Save an annotated confusion matrix."""
    figure, axis = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        np.asarray(confusion),
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=list(class_names),
        yticklabels=list(class_names),
        ax=axis,
    )
    axis.set_xlabel("Предсказание")
    axis.set_ylabel("Истинный класс")
    axis.set_title("Матрица ошибок кросс-attention")
    return _save(figure, path)


def plot_metric_bars(scores: Sequence[ModelScore], path: Path) -> Path:
    """Save accuracy and macro-F1 of every model."""
    names = [score.name for score in scores]
    positions = np.arange(len(names))
    width = 0.35
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.bar(positions - width / 2, [score.accuracy for score in scores], width, label="Accuracy")
    axis.bar(positions + width / 2, [score.macro_f1 for score in scores], width, label="Macro-F1")
    axis.set_xticks(positions, names)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Значение")
    axis.set_title("Сравнение модальностей на тесте")
    axis.legend()
    return _save(figure, path)


def render_example_figure(
    image: Image.Image,
    attention_map: np.ndarray,
    class_names: Sequence[str],
    probabilities: Sequence[float],
    tokens: Sequence[str],
    token_weights: Sequence[float],
) -> Figure:
    """Build the photo, attention heatmap, class probabilities and token weights."""
    photo = image.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
    heat = _resize_heat(attention_map, photo.size)
    figure, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes[0, 0].imshow(photo)
    axes[0, 0].set_title("Фото")
    axes[0, 0].axis("off")
    axes[0, 1].imshow(photo)
    axes[0, 1].imshow(heat, cmap="magma", alpha=0.55)
    axes[0, 1].set_title("Cross-attention: текст → фото")
    axes[0, 1].axis("off")
    axes[1, 0].barh(list(class_names), list(probabilities))
    axes[1, 0].set_xlim(0, 1)
    axes[1, 0].set_title("Вероятности классов")
    axes[1, 0].set_xlabel("Вероятность")
    if tokens:
        axes[1, 1].barh(list(tokens), list(token_weights))
        axes[1, 1].set_title("Cross-attention: фото → слова")
        axes[1, 1].set_xlabel("Вес")
    else:
        axes[1, 1].set_title("Нет токенов")
        axes[1, 1].axis("off")
    figure.tight_layout()
    return figure


def save_example_figure(
    image: Image.Image,
    attention_map: np.ndarray,
    class_names: Sequence[str],
    probabilities: Sequence[float],
    tokens: Sequence[str],
    token_weights: Sequence[float],
    path: Path,
) -> Path:
    """Save :func:`render_example_figure` to disk."""
    figure = render_example_figure(
        image,
        attention_map,
        class_names,
        probabilities,
        tokens,
        token_weights,
    )
    return _save(figure, path)


def _drop_batch(attention: torch.Tensor) -> torch.Tensor:
    if attention.ndim == 4:
        return attention.squeeze(0)
    if attention.ndim == 3:
        return attention
    raise ValueError(f"Ожидалась карта внимания с 3 или 4 осями, получено {attention.ndim}")


def _resize_heat(attention_map: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    low = float(attention_map.min())
    high = float(attention_map.max())
    span = high - low
    normalized = np.zeros_like(attention_map) if span < 1e-8 else (attention_map - low) / span
    scaled = np.asarray(normalized * 255, dtype=np.uint8)
    heat = Image.fromarray(scaled, mode="L").resize(size, Image.Resampling.BILINEAR)
    return np.asarray(heat)


def _save(figure: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(figure)
    return path
