"""Markdown and HTML reports with metrics and prediction examples."""

from __future__ import annotations

import html
from dataclasses import dataclass
from pathlib import Path

from amlftp.metrics import ModelScore


@dataclass(frozen=True)
class ExampleView:
    """One illustrated prediction in the report."""

    text: str
    true_label: str
    pred_label: str
    probabilities: dict[str, float]
    figure_path: str


@dataclass(frozen=True)
class ReportData:
    """Everything the report renderers need."""

    seed: int
    epochs: int
    completed_epochs: int
    best_epoch: int
    best_val_macro_f1: float
    best_val_accuracy: float
    resumed_from: str | None
    dataset_name: str
    max_per_class: int
    n_train: int
    n_val: int
    n_test: int
    class_counts: dict[str, int]
    scores: tuple[ModelScore, ...]
    history_figure: str
    confusion_figure: str
    metrics_figure: str
    examples: tuple[ExampleView, ...]
    image_note: str


def write_reports(data: ReportData, reports_dir: Path) -> tuple[Path, Path]:
    """Write ``report.md`` and ``report.html`` into ``reports_dir``."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    html_path = reports_dir / "report.html"
    html_path.write_text(render_html(data), encoding="utf-8")
    return html_path


def render_markdown(data: ReportData) -> str:
    """Render the report as Markdown."""
    lines = [
        "# Отчёт: мультимодальная классификация",
        "",
        "## Воспроизведение",
        "",
        f"- Seed зафиксирован до загрузки данных и инициализации модели: `{data.seed}`",
        f"- Запрошено эпох: {data.epochs}",
        f"- Завершено эпох: {data.completed_epochs}",
        f"- Лучшая эпоха по val macro-F1: {data.best_epoch}",
        f"- Val macro-F1 лучшей эпохи: {_fmt(data.best_val_macro_f1)}",
        f"- Val accuracy лучшей эпохи: {_fmt(data.best_val_accuracy)}",
        f"- Возобновление: {_resume_text(data.resumed_from)}",
        "",
        "## Данные",
        "",
        f"Датасет `{data.dataset_name}`. Цель — `masterCategory`.",
        f"На класс не больше {data.max_per_class} примеров.",
        f"Размеры сплитов: train {data.n_train}, val {data.n_val}, test {data.n_test}.",
        "",
        "| Класс | Примеров в ограниченной выборке |",
        "| --- | ---: |",
    ]
    lines.extend(f"| {name} | {count} |" for name, count in data.class_counts.items())
    lines.extend(["", data.image_note, "", "## Модель", ""])
    lines.extend(
        [
            "Фото и текст объединяются двунаправленным cross-attention.",
            "Запросы изображения читают ключи и значения текста, запросы текста читают ключи и значения изображения.",
            "В классификатор попадают уже обновлённые токены.",
            "Строки «Только фото» и «Только текст» — отдельные головы без смешивания модальностей.",
            "",
            "## Метрики на тесте",
            "",
            "| Модель | Accuracy | Macro-F1 |",
            "| --- | ---: | ---: |",
        ]
    )
    for score in data.scores:
        lines.append(f"| {score.name} | {_fmt(score.accuracy)} | {_fmt(score.macro_f1)} |")
    for score in data.scores:
        lines.extend(
            [
                "",
                f"### {score.name}",
                "",
                "| Класс | Precision | Recall | F1 | Support |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in score.per_class:
            lines.append(
                f"| {row.name} | {_fmt(row.precision)} | {_fmt(row.recall)} | {_fmt(row.f1)} | {row.support} |"
            )
    fusion = data.scores[0]
    lines.extend(["", "### Матрица ошибок кросс-attention", ""])
    header = "| Истинный \\ предсказание | " + " | ".join(row.name for row in fusion.per_class) + " |"
    separator = "| --- | " + " | ".join("---:" for _ in fusion.per_class) + " |"
    lines.extend([header, separator])
    for index, counts in enumerate(fusion.confusion):
        name = fusion.per_class[index].name
        values = " | ".join(str(value) for value in counts)
        lines.append(f"| {name} | {values} |")
    lines.extend(
        [
            "",
            "## Графики",
            "",
            f"![Кривые обучения]({data.history_figure})",
            "",
            f"![Сравнение метрик]({data.metrics_figure})",
            "",
            f"![Матрица ошибок]({data.confusion_figure})",
            "",
            "## Примеры",
            "",
        ]
    )
    for number, example in enumerate(data.examples, start=1):
        probs = ", ".join(f"{label} {_fmt(value)}" for label, value in example.probabilities.items())
        lines.extend(
            [
                f"### Пример {number}",
                "",
                f"- Текст: {_plain(example.text)}",
                f"- Истинный класс: {example.true_label}",
                f"- Предсказание: {example.pred_label}",
                f"- Вероятности: {probs}",
                "",
                f"![Пример {number}]({example.figure_path})",
                "",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def render_html(data: ReportData) -> str:
    """Render the same report as a standalone HTML file."""
    count_rows = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{count}</td></tr>" for name, count in data.class_counts.items()
    )
    summary_rows = "".join(
        f"<tr><td>{html.escape(score.name)}</td><td>{_fmt(score.accuracy)}</td><td>{_fmt(score.macro_f1)}</td></tr>"
        for score in data.scores
    )
    per_class_blocks: list[str] = []
    for score in data.scores:
        body = "".join(
            "<tr>"
            f"<td>{html.escape(row.name)}</td>"
            f"<td>{_fmt(row.precision)}</td>"
            f"<td>{_fmt(row.recall)}</td>"
            f"<td>{_fmt(row.f1)}</td>"
            f"<td>{row.support}</td>"
            "</tr>"
            for row in score.per_class
        )
        per_class_blocks.append(
            f"<h3>{html.escape(score.name)}</h3>"
            "<table><thead><tr><th>Класс</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr></thead>"
            f"<tbody>{body}</tbody></table>"
        )
    fusion = data.scores[0]
    class_names = [row.name for row in fusion.per_class]
    confusion_header = "".join(f"<th>{html.escape(name)}</th>" for name in class_names)
    confusion_body = "".join(
        f"<tr><th>{html.escape(name)}</th>" + "".join(f"<td>{value}</td>" for value in row) + "</tr>"
        for name, row in zip(class_names, fusion.confusion, strict=True)
    )
    examples = []
    for number, example in enumerate(data.examples, start=1):
        probs = ", ".join(f"{html.escape(label)} {_fmt(value)}" for label, value in example.probabilities.items())
        examples.append(
            f"<h3>Пример {number}</h3>"
            f"<p>Текст: {html.escape(_plain(example.text))}<br>"
            f"Истинный класс: {html.escape(example.true_label)}<br>"
            f"Предсказание: {html.escape(example.pred_label)}<br>"
            f"Вероятности: {probs}</p>"
            f'<img src="{html.escape(example.figure_path)}" alt="Пример {number}">'
        )
    return f"""<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <title>Отчёт: мультимодальная классификация</title>
  <style>
    body {{ font-family: sans-serif; margin: 2rem auto; max-width: 980px; line-height: 1.45; }}
    img {{ max-width: 100%; }}
    table {{ border-collapse: collapse; margin: 0.5rem 0 1rem; }}
    td, th {{ border: 1px solid #ccc; padding: 0.3rem 0.6rem; }}
  </style>
</head>
<body>
  <h1>Отчёт: мультимодальная классификация</h1>
  <h2>Воспроизведение</h2>
  <ul>
    <li>Seed зафиксирован до загрузки данных и инициализации модели: {data.seed}</li>
    <li>Запрошено эпох: {data.epochs}</li>
    <li>Завершено эпох: {data.completed_epochs}</li>
    <li>Лучшая эпоха по val macro-F1: {data.best_epoch}</li>
    <li>Val macro-F1 лучшей эпохи: {_fmt(data.best_val_macro_f1)}</li>
    <li>Val accuracy лучшей эпохи: {_fmt(data.best_val_accuracy)}</li>
    <li>Возобновление: {html.escape(_resume_text(data.resumed_from))}</li>
  </ul>
  <h2>Данные</h2>
  <p>Датасет <code>{html.escape(data.dataset_name)}</code>. Цель — <code>masterCategory</code>.
  На класс не больше {data.max_per_class} примеров.
  Размеры сплитов: train {data.n_train}, val {data.n_val}, test {data.n_test}.</p>
  <table>
    <thead><tr><th>Класс</th><th>Примеров в ограниченной выборке</th></tr></thead>
    <tbody>{count_rows}</tbody>
  </table>
  <p>{html.escape(data.image_note)}</p>
  <h2>Модель</h2>
  <p>Фото и текст объединяются двунаправленным cross-attention.
  Запросы изображения читают ключи и значения текста, запросы текста читают ключи и значения изображения.
  В классификатор попадают уже обновлённые токены.
  Строки «Только фото» и «Только текст» — отдельные головы без смешивания модальностей.</p>
  <h2>Метрики на тесте</h2>
  <table>
    <thead><tr><th>Модель</th><th>Accuracy</th><th>Macro-F1</th></tr></thead>
    <tbody>{summary_rows}</tbody>
  </table>
  {"".join(per_class_blocks)}
  <h3>Матрица ошибок кросс-attention</h3>
  <table>
    <thead><tr><th>Истинный \\ предсказание</th>{confusion_header}</tr></thead>
    <tbody>{confusion_body}</tbody>
  </table>
  <h2>Графики</h2>
  <img src="{html.escape(data.history_figure)}" alt="Кривые обучения">
  <img src="{html.escape(data.metrics_figure)}" alt="Сравнение метрик">
  <img src="{html.escape(data.confusion_figure)}" alt="Матрица ошибок">
  <h2>Примеры</h2>
  {"".join(examples)}
</body>
</html>
"""


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def _plain(text: str) -> str:
    return " ".join(text.replace("|", "/").split())


def _resume_text(resumed_from: str | None) -> str:
    if resumed_from is None:
        return "обучение с нуля"
    return resumed_from
