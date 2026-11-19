"""Dataset filtering and deterministic splits."""

from __future__ import annotations

import io
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

from amlftp.config import TrainConfig


@dataclass(frozen=True)
class SampleRef:
    """One kept row of the source dataset."""

    index: int
    text: str
    label: str


@dataclass(frozen=True)
class PreparedData:
    """Capped, stratified rows plus the original dataset handle."""

    dataset: Any
    labels: tuple[str, ...]
    train: tuple[SampleRef, ...]
    val: tuple[SampleRef, ...]
    test: tuple[SampleRef, ...]
    class_counts: dict[str, int]


def select_labels(
    labels: Sequence[str],
    *,
    min_count: int,
    allowed: Sequence[str] | None = None,
) -> list[str]:
    """Keep labels that are frequent enough and, optionally, on an allow-list."""
    counts = Counter(labels)
    allowed_set = None if allowed is None else set(allowed)
    chosen: list[str] = []
    for name, count in counts.items():
        if count < min_count:
            continue
        if allowed_set is not None and name not in allowed_set:
            continue
        chosen.append(name)
    return sorted(chosen)


def collect_samples(
    categories: Sequence[str | None],
    names: Sequence[str | None],
    allowed: Sequence[str],
) -> list[SampleRef]:
    """Drop empty titles and categories outside ``allowed``."""
    allowed_set = set(allowed)
    samples: list[SampleRef] = []
    for index, (category, name) in enumerate(zip(categories, names, strict=True)):
        if category is None or name is None:
            continue
        text = name.strip()
        if category not in allowed_set or text == "":
            continue
        samples.append(SampleRef(index=index, text=text, label=category))
    return samples


def cap_per_class(labels: Sequence[str], max_per_class: int, seed: int) -> list[int]:
    """Return indices with at most ``max_per_class`` rows of each class."""
    if max_per_class < 1:
        raise ValueError("max_per_class должен быть положительным")
    grouped: dict[str, list[int]] = {}
    for index, label in enumerate(labels):
        grouped.setdefault(label, []).append(index)
    rng = np.random.default_rng(seed)
    kept: list[int] = []
    for label in sorted(grouped):
        indices = np.asarray(grouped[label], dtype=np.int64)
        if indices.size > max_per_class:
            indices = np.sort(rng.choice(indices, size=max_per_class, replace=False))
        kept.extend(int(index) for index in indices)
    kept.sort()
    return kept


def stratified_split(
    labels: Sequence[str],
    seed: int,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split indices into train, validation and test with a fixed seed."""
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("Доли разбиения должны давать 1")
    indices = np.arange(len(labels))
    label_array = np.asarray(list(labels))
    train_idx, rest_idx = train_test_split(
        indices,
        test_size=ratios[1] + ratios[2],
        stratify=label_array,
        random_state=seed,
    )
    rest_labels = label_array[np.asarray(rest_idx)]
    relative_test = ratios[2] / (ratios[1] + ratios[2])
    val_idx, test_idx = train_test_split(
        rest_idx,
        test_size=relative_test,
        stratify=rest_labels,
        random_state=seed,
    )
    return (
        np.asarray(train_idx, dtype=np.int64),
        np.asarray(val_idx, dtype=np.int64),
        np.asarray(test_idx, dtype=np.int64),
    )


def load_prepared_data(config: TrainConfig) -> PreparedData:
    """Download the fashion dataset and build a capped stratified split."""
    from datasets import load_dataset

    print(f"Загрузка датасета {config.dataset_name}", flush=True)
    dataset = load_dataset(config.dataset_name, split="train")
    categories = [_as_optional_str(value) for value in dataset["masterCategory"]]
    names = [_as_optional_str(value) for value in dataset["productDisplayName"]]
    samples = collect_samples(categories, names, config.allowed_labels)
    if not samples:
        raise RuntimeError("После фильтрации не осталось примеров")
    present = select_labels(
        [sample.label for sample in samples],
        min_count=1,
        allowed=config.allowed_labels,
    )
    missing = [label for label in config.allowed_labels if label not in present]
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(f"В датасете нет классов: {joined}")
    capped_positions = cap_per_class([sample.label for sample in samples], config.max_per_class, config.seed)
    capped = [samples[index] for index in capped_positions]
    counts = Counter(sample.label for sample in capped)
    too_small = [f"{label}={counts[label]}" for label in config.allowed_labels if counts[label] < 10]
    if too_small:
        raise RuntimeError("Слишком мало примеров для стратифицированного разбиения: " + ", ".join(too_small))
    train_idx, val_idx, test_idx = stratified_split([sample.label for sample in capped], config.seed)
    print(
        f"Примеров после ограничения: {len(capped)} (train {len(train_idx)}, val {len(val_idx)}, test {len(test_idx)})",
        flush=True,
    )
    for label in config.allowed_labels:
        print(f"  {label}: {counts[label]}", flush=True)
    return PreparedData(
        dataset=dataset,
        labels=config.allowed_labels,
        train=tuple(capped[int(index)] for index in train_idx),
        val=tuple(capped[int(index)] for index in val_idx),
        test=tuple(capped[int(index)] for index in test_idx),
        class_counts={label: int(counts[label]) for label in config.allowed_labels},
    )


def read_image(dataset: Any, index: int) -> Image.Image:
    """Read one RGB image from a Hugging Face dataset row."""
    image = dataset[int(index)]["image"]
    if not isinstance(image, Image.Image):
        raise TypeError(f"Строка {index} не содержит PIL-изображение")
    return image.convert("RGB")


def pil_to_png(image: Image.Image) -> bytes:
    """Encode an image as PNG bytes."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def png_to_pil(payload: bytes) -> Image.Image:
    """Decode PNG bytes produced by :func:`pil_to_png`."""
    return Image.open(io.BytesIO(payload)).convert("RGB")


def _as_optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)
