"""The generated report contains per-task metrics and worked examples."""

from __future__ import annotations

from pathlib import Path

from amlftp.metrics import ClassScore, ModelScore
from amlftp.report import ExampleView, ReportData, TaskReport, write_reports


def _score(name: str, accuracy: float, macro_f1: float) -> ModelScore:
    per_class = (
        ClassScore("Apparel", 0.667, 1.0, 0.8, 2),
        ClassScore("Footwear", 1.0, 0.5, 0.667, 2),
    )
    return ModelScore(
        name=name,
        accuracy=accuracy,
        macro_f1=macro_f1,
        per_class=per_class,
        confusion=((2, 0), (1, 1)),
    )


def test_write_reports_contains_metrics_and_examples(tmp_path: Path) -> None:
    category = TaskReport(
        name="masterCategory",
        scores=(
            _score("Кросс-attention", 0.75, 0.733),
            _score("Только фото", 0.5, 0.4),
        ),
        confusion_figure="figures/confusion_masterCategory.png",
        metrics_figure="figures/metrics_masterCategory.png",
    )
    gender = TaskReport(
        name="gender",
        scores=(_score("Кросс-attention", 0.6, 0.55),),
        confusion_figure="figures/confusion_gender.png",
        metrics_figure="figures/metrics_gender.png",
    )
    data = ReportData(
        seed=42,
        epochs=5,
        completed_epochs=5,
        best_epoch=4,
        best_val_macro_f1=0.81,
        best_val_accuracy=0.84,
        resumed_from=None,
        dataset_name="ashraq/fashion-product-images-small",
        max_per_class=1500,
        primary_task="masterCategory",
        n_train=70,
        n_val=15,
        n_test=15,
        class_counts={
            "masterCategory": {"Apparel": 40, "Footwear": 30},
            "gender": {"Men": 50, "Women": 20},
        },
        tasks=(category, gender),
        history_figure="figures/history.png",
        examples=(
            ExampleView(
                text="Blue running shoe",
                true_labels={"masterCategory": "Footwear", "gender": "Men"},
                pred_labels={"masterCategory": "Footwear", "gender": "Women"},
                probabilities={
                    "masterCategory": {"Apparel": 0.2, "Footwear": 0.8},
                    "gender": {"Men": 0.4, "Women": 0.6},
                },
                figure_path="figures/example_0.png",
            ),
        ),
        image_note="Миниатюры приводятся к 224 px.",
    )
    write_reports(data, tmp_path)
    markdown = (tmp_path / "report.md").read_text(encoding="utf-8")
    html = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert "мультизадач" in markdown
    assert "0.750" in markdown
    assert "Примеры" in markdown
    assert "Кросс-attention" in markdown
    assert "masterCategory" in markdown
    assert "gender" in markdown
    assert "Blue running shoe" in markdown
    assert "0.750" in html
    assert "Примеры" in html
    assert "gender" in html
    assert "Blue running shoe" in html
