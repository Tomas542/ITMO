"""Classification metrics for the report tables."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


@dataclass(frozen=True)
class ClassScore:
    """Precision, recall and F1 of one class."""

    name: str
    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True)
class ModelScore:
    """Aggregate scores of one model on one split."""

    name: str
    accuracy: float
    macro_f1: float
    per_class: tuple[ClassScore, ...]
    confusion: tuple[tuple[int, ...], ...]


def score_predictions(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    label_names: Sequence[str],
    *,
    name: str,
) -> ModelScore:
    """Compute accuracy, macro-F1, per-class scores and a confusion matrix."""
    label_ids = list(range(len(label_names)))
    accuracy = float(accuracy_score(y_true, y_pred))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=label_ids,
        average=None,
        zero_division=0,
    )
    macro_f1 = float(np.mean(np.asarray(f1, dtype=np.float64)))
    per_class = tuple(
        ClassScore(
            name=label_names[index],
            precision=float(precision[index]),
            recall=float(recall[index]),
            f1=float(f1[index]),
            support=int(support[index]),
        )
        for index in label_ids
    )
    matrix = confusion_matrix(y_true, y_pred, labels=label_ids)
    confusion = tuple(tuple(int(value) for value in row) for row in matrix.tolist())
    return ModelScore(
        name=name,
        accuracy=accuracy,
        macro_f1=macro_f1,
        per_class=per_class,
        confusion=confusion,
    )
