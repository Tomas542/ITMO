"""Pure tests for filtering and splitting, without downloading the dataset."""

from __future__ import annotations

import numpy as np

from amlftp.data import cap_per_class, collect_samples, select_labels, stratified_split


def test_select_labels_filters_rare_and_unknown() -> None:
    labels = ["Apparel"] * 5 + ["Accessories"] * 2 + ["Home"] * 10
    chosen = select_labels(labels, min_count=3, allowed=["Apparel", "Accessories", "Footwear"])
    assert chosen == ["Apparel"]


def test_collect_samples_drops_empty_and_disallowed() -> None:
    categories = ["Apparel", "Home", "Footwear", None, "Apparel"]
    names = ["Blue shirt", "Lamp", "  ", "Watch", "Red dress"]
    samples = collect_samples(categories, names, ["Apparel", "Footwear"])
    assert [(sample.index, sample.label, sample.text) for sample in samples] == [
        (0, "Apparel", "Blue shirt"),
        (4, "Apparel", "Red dress"),
    ]


def test_cap_per_class_is_deterministic_and_limited() -> None:
    labels = ["a"] * 10 + ["b"] * 10
    first = cap_per_class(labels, 4, 1)
    assert first == cap_per_class(labels, 4, 1)
    assert first != cap_per_class(labels, 4, 2)
    assert sum(labels[index] == "a" for index in first) == 4
    assert sum(labels[index] == "b" for index in first) == 4


def test_stratified_split_covers_every_row() -> None:
    labels = ["a", "b"] * 50
    train, val, test = stratified_split(labels, seed=0)
    combined = np.concatenate([train, val, test])
    assert len(combined) == 100
    assert len(set(combined.tolist())) == 100
    assert abs(len(train) / 100 - 0.7) < 0.05
    assert abs(len(val) / 100 - 0.15) < 0.05
    assert abs(len(test) / 100 - 0.15) < 0.05
