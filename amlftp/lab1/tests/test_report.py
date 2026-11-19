"""The generated report contains metrics and worked examples."""

from __future__ import annotations

from pathlib import Path

from amlftp.metrics import ClassScore, ModelScore
from amlftp.report import ExampleView, ReportData, write_reports


def test_write_reports_contains_metrics_and_examples(tmp_path: Path) -> None:
    fusion = ModelScore(
        name="Кросс-attention",
        accuracy=0.75,
        macro_f1=0.733,
        per_class=(
            ClassScore("Apparel", 0.667, 1.0, 0.8, 2),
            ClassScore("Footwear", 1.0, 0.5, 0.667, 2),
        ),
        confusion=((2, 0), (1, 1)),
    )
    image_only = ModelScore(
        name="Только фото",
        accuracy=0.5,
        macro_f1=0.4,
        per_class=fusion.per_class,
        confusion=((1, 1), (1, 1)),
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
        n_train=70,
        n_val=15,
        n_test=15,
        class_counts={"Apparel": 40, "Footwear": 30},
        scores=(fusion, image_only),
        history_figure="figures/history.png",
        confusion_figure="figures/confusion.png",
        metrics_figure="figures/metrics.png",
        examples=(
            ExampleView(
                text="Blue running shoe",
                true_label="Footwear",
                pred_label="Footwear",
                probabilities={"Apparel": 0.2, "Footwear": 0.8},
                figure_path="figures/example_0.png",
            ),
        ),
        image_note="Миниатюры приводятся к 224 px.",
    )
    write_reports(data, tmp_path)
    markdown = (tmp_path / "report.md").read_text(encoding="utf-8")
    html = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert "0.750" in markdown
    assert "Примеры" in markdown
    assert "Кросс-attention" in markdown
    assert "Blue running shoe" in markdown
    assert "0.750" in html
    assert "Примеры" in html
    assert "Blue running shoe" in html
