"""Dataset filtering and deterministic splits for several tasks at once."""

from __future__ import annotations

import io
from collections import Counter
from collections.abc import Mapping, Sequence
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
    labels: dict[str, str]


@dataclass(frozen=True)
class PreparedData:
    """Capped, stratified rows plus the original dataset handle."""

    dataset: Any
    task_labels: dict[str, tuple[str, ...]]
    train: tuple[SampleRef, ...]
    val: tuple[SampleRef, ...]
    test: tuple[SampleRef, ...]
    class_counts: dict[str, dict[str, int]]


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
    task_columns: Mapping[str, Sequence[str | None]],
    names: Sequence[str | None],
    allowed: Mapping[str, Sequence[str]],
) -> list[SampleRef]:
    """Keep rows whose title is non-empty and whose every task label is allowed."""
    if not allowed:
        raise ValueError("Нужна хотя бы одна задача")
    width = len(names)
    for task, column in task_columns.items():
        if len(column) != width:
            raise ValueError(f"Колонка {task} имеет длину {len(column)}, ожидалось {width}")
    missing_columns = [task for task in allowed if task not in task_columns]
    if missing_columns:
        raise ValueError("Нет колонок: " + ", ".join(missing_columns))
    allowed_sets = {task: set(labels) for task, labels in allowed.items()}
    samples: list[SampleRef] = []
    for index, name in enumerate(names):
        if name is None:
            continue
        text = name.strip()
        if text == "":
            continue
        labels: dict[str, str] = {}
        keep = True
        for task, allowed_set in allowed_sets.items():
            value = task_columns[task][index]
            if value is None or value not in allowed_set:
                keep = False
                break
            labels[task] = value
        if keep:
            samples.append(SampleRef(index=index, text=text, labels=labels))
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
    """Download the fashion dataset and build a capped stratified split.

    A row is kept only when every task label is on that task's allow-list.
    The cap and the stratified split both use the first task.
    """
    from datasets import load_dataset

    if not config.task_labels:
        raise ValueError("Нужна хотя бы одна задача")
    primary, primary_labels = config.task_labels[0]
    print(f"Загрузка датасета {config.dataset_name}", flush=True)
    dataset = load_dataset(config.dataset_name, split="train")
    task_columns = {task: [_as_optional_str(value) for value in dataset[task]] for task, _labels in config.task_labels}
    names = [_as_optional_str(value) for value in dataset["productDisplayName"]]
    allowed = dict(config.task_labels)
    samples = collect_samples(task_columns, names, allowed)
    if not samples:
        raise RuntimeError("После фильтрации не осталось примеров")
    for task, labels in config.task_labels:
        present = select_labels([sample.labels[task] for sample in samples], min_count=1, allowed=labels)
        missing = [label for label in labels if label not in present]
        if missing:
            raise RuntimeError(f"В задаче {task} нет классов: {', '.join(missing)}")
    capped_positions = cap_per_class([sample.labels[primary] for sample in samples], config.max_per_class, config.seed)
    capped = [samples[index] for index in capped_positions]
    primary_counts = Counter(sample.labels[primary] for sample in capped)
    too_small = [f"{label}={primary_counts[label]}" for label in primary_labels if primary_counts[label] < 10]
    if too_small:
        raise RuntimeError("Слишком мало примеров для стратифицированного разбиения: " + ", ".join(too_small))
    train_idx, val_idx, test_idx = stratified_split([sample.labels[primary] for sample in capped], config.seed)
    print(
        f"Примеров после ограничения: {len(capped)} (train {len(train_idx)}, val {len(val_idx)}, test {len(test_idx)})",
        flush=True,
    )
    class_counts: dict[str, dict[str, int]] = {}
    for task, labels in config.task_labels:
        counts = Counter(sample.labels[task] for sample in capped)
        class_counts[task] = {label: int(counts[label]) for label in labels}
        print(f"{task}:", flush=True)
        for label in labels:
            print(f"  {label}: {class_counts[task][label]}", flush=True)
    return PreparedData(
        dataset=dataset,
        task_labels=dict(config.task_labels),
        train=tuple(capped[int(index)] for index in train_idx),
        val=tuple(capped[int(index)] for index in val_idx),
        test=tuple(capped[int(index)] for index in test_idx),
        class_counts=class_counts,
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
