"""Markdown and HTML reports with per-task metrics and prediction examples."""

from __future__ import annotations

import html
from dataclasses import dataclass
from pathlib import Path

from amlftp.metrics import ModelScore

MODEL_PARAGRAPH = (
    "Фото и текст объединяются общим двунаправленным cross-attention. "
    "У каждой задачи своя линейная голова на объединённом представлении. "
    "Функция потерь — сумма cross-entropy по задачам без весов. "
    "Строки «Только фото» и «Только текст» — отдельные головы этой же задачи, модальности они не смешивают."
)


@dataclass(frozen=True)
class ExampleView:
    """One illustrated prediction in the report."""

    text: str
    true_labels: dict[str, str]
    pred_labels: dict[str, str]
    probabilities: dict[str, dict[str, float]]
    figure_path: str


@dataclass(frozen=True)
class TaskReport:
    """Fusion and unimodal scores of one task, plus the figure paths."""

    name: str
    scores: tuple[ModelScore, ...]
    confusion_figure: str
    metrics_figure: str


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
    primary_task: str
    n_train: int
    n_val: int
    n_test: int
    class_counts: dict[str, dict[str, int]]
    tasks: tuple[TaskReport, ...]
    history_figure: str
    examples: tuple[ExampleView, ...]
    image_note: str


def write_reports(data: ReportData, reports_dir: Path) -> Path:
    """Write ``report.md`` and ``report.html`` into ``reports_dir``. Returns the HTML path."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "report.md").write_text(render_markdown(data), encoding="utf-8")
    html_path = reports_dir / "report.html"
    html_path.write_text(render_html(data), encoding="utf-8")
    return html_path


def render_markdown(data: ReportData) -> str:
    """Render the report as Markdown."""
    task_list = ", ".join(f"`{task.name}`" for task in data.tasks)
    lines = [
        "# Отчёт: мультизадачная классификация",
        "",
        "## Воспроизведение",
        "",
        f"- Seed зафиксирован до загрузки данных и инициализации модели: `{data.seed}`",
        f"- Запрошено эпох: {data.epochs}",
        f"- Завершено эпох: {data.completed_epochs}",
        f"- Лучшая эпоха по среднему val macro-F1: {data.best_epoch}",
        f"- Средний val macro-F1 лучшей эпохи: {_fmt(data.best_val_macro_f1)}",
        f"- Средний val accuracy лучшей эпохи: {_fmt(data.best_val_accuracy)}",
        f"- Возобновление: {_resume_text(data.resumed_from)}",
        "",
        "## Данные",
        "",
        f"Датасет `{data.dataset_name}`. Задачи: {task_list}.",
        "Строка остаётся в выборке, только если метка каждой задачи входит в её список классов.",
        (
            f"На класс задачи `{data.primary_task}` не больше {data.max_per_class} примеров. "
            "Разбиение стратифицировано по этой задаче."
        ),
        f"Размеры сплитов: train {data.n_train}, val {data.n_val}, test {data.n_test}.",
        "",
    ]
    for task in data.tasks:
        lines.extend(
            [
                f"### {task.name}",
                "",
                "| Класс | Примеров в ограниченной выборке |",
                "| --- | ---: |",
            ]
        )
        lines.extend(f"| {name} | {count} |" for name, count in data.class_counts[task.name].items())
        lines.append("")
    lines.extend([data.image_note, "", "## Модель", "", MODEL_PARAGRAPH, "", "## Графики", ""])
    lines.extend([f"![Кривые обучения]({data.history_figure})", ""])
    for task in data.tasks:
        lines.extend(
            [
                f"## {task.name}",
                "",
                "| Модель | Accuracy | Macro-F1 |",
                "| --- | ---: | ---: |",
            ]
        )
        for score in task.scores:
            lines.append(f"| {score.name} | {_fmt(score.accuracy)} | {_fmt(score.macro_f1)} |")
        for score in task.scores:
            lines.extend(
                [
                    "",
                    f"### {task.name}: {score.name}",
                    "",
                    "| Класс | Precision | Recall | F1 | Support |",
                    "| --- | ---: | ---: | ---: | ---: |",
                ]
            )
            for row in score.per_class:
                lines.append(
                    f"| {row.name} | {_fmt(row.precision)} | {_fmt(row.recall)} | {_fmt(row.f1)} | {row.support} |"
                )
        fusion = task.scores[0]
        lines.extend(["", f"### Матрица ошибок кросс-attention: {task.name}", ""])
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
                f"![Сравнение метрик]({task.metrics_figure})",
                "",
                f"![Матрица ошибок]({task.confusion_figure})",
                "",
            ]
        )
    lines.extend(["## Примеры", ""])
    for number, example in enumerate(data.examples, start=1):
        lines.extend([f"### Пример {number}", "", f"- Текст: {_plain(example.text)}"])
        for task_name, true_label in example.true_labels.items():
            probs = ", ".join(f"{label} {_fmt(value)}" for label, value in example.probabilities[task_name].items())
            lines.append(
                f"- {task_name}: истина {true_label}, предсказание {example.pred_labels[task_name]}. "
                f"Вероятности: {probs}"
            )
        lines.extend(["", f"![Пример {number}]({example.figure_path})", ""])
    lines.append("")
    return "\n".join(lines)


def render_html(data: ReportData) -> str:
    """Render the same report as a standalone HTML file."""
    task_list = ", ".join(f"<code>{html.escape(task.name)}</code>" for task in data.tasks)
    count_blocks: list[str] = []
    for task in data.tasks:
        rows = "".join(
            f"<tr><td>{html.escape(name)}</td><td>{count}</td></tr>"
            for name, count in data.class_counts[task.name].items()
        )
        count_blocks.append(
            f"<h3>{html.escape(task.name)}</h3>"
            "<table><thead><tr><th>Класс</th><th>Примеров в ограниченной выборке</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>"
        )
    task_blocks: list[str] = []
    for task in data.tasks:
        summary_rows = "".join(
            f"<tr><td>{html.escape(score.name)}</td><td>{_fmt(score.accuracy)}</td><td>{_fmt(score.macro_f1)}</td></tr>"
            for score in task.scores
        )
        per_class_blocks: list[str] = []
        for score in task.scores:
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
                f"<h3>{html.escape(task.name)}: {html.escape(score.name)}</h3>"
                "<table><thead><tr><th>Класс</th><th>Precision</th><th>Recall</th>"
                "<th>F1</th><th>Support</th></tr></thead>"
                f"<tbody>{body}</tbody></table>"
            )
        fusion = task.scores[0]
        class_names = [row.name for row in fusion.per_class]
        confusion_header = "".join(f"<th>{html.escape(name)}</th>" for name in class_names)
        confusion_body = "".join(
            f"<tr><th>{html.escape(name)}</th>" + "".join(f"<td>{value}</td>" for value in row) + "</tr>"
            for name, row in zip(class_names, fusion.confusion, strict=True)
        )
        task_blocks.append(
            f"<h2>{html.escape(task.name)}</h2>"
            "<table><thead><tr><th>Модель</th><th>Accuracy</th><th>Macro-F1</th></tr></thead>"
            f"<tbody>{summary_rows}</tbody></table>"
            f"{''.join(per_class_blocks)}"
            f"<h3>Матрица ошибок кросс-attention: {html.escape(task.name)}</h3>"
            "<table><thead><tr><th>Истинный \\ предсказание</th>"
            f"{confusion_header}</tr></thead><tbody>{confusion_body}</tbody></table>"
            f'<img src="{html.escape(task.metrics_figure)}" alt="Сравнение метрик {html.escape(task.name)}">'
            f'<img src="{html.escape(task.confusion_figure)}" alt="Матрица ошибок {html.escape(task.name)}">'
        )
    examples: list[str] = []
    for number, example in enumerate(data.examples, start=1):
        details = []
        for task_name, true_label in example.true_labels.items():
            probs = ", ".join(
                f"{html.escape(label)} {_fmt(value)}" for label, value in example.probabilities[task_name].items()
            )
            details.append(
                f"{html.escape(task_name)}: истина {html.escape(true_label)}, "
                f"предсказание {html.escape(example.pred_labels[task_name])}. Вероятности: {probs}"
            )
        detail_html = "<br>".join(details)
        examples.append(
            f"<h3>Пример {number}</h3>"
            f"<p>Текст: {html.escape(_plain(example.text))}<br>{detail_html}</p>"
            f'<img src="{html.escape(example.figure_path)}" alt="Пример {number}">'
        )
    limit_sentence = (
        f"На класс задачи <code>{html.escape(data.primary_task)}</code> не больше {data.max_per_class} примеров. "
        "Разбиение стратифицировано по этой задаче."
    )
    return f"""<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <title>Отчёт: мультизадачная классификация</title>
  <style>
    body {{ font-family: sans-serif; margin: 2rem auto; max-width: 980px; line-height: 1.45; }}
    img {{ max-width: 100%; }}
    table {{ border-collapse: collapse; margin: 0.5rem 0 1rem; }}
    td, th {{ border: 1px solid #ccc; padding: 0.3rem 0.6rem; }}
  </style>
</head>
<body>
  <h1>Отчёт: мультизадачная классификация</h1>
  <h2>Воспроизведение</h2>
  <ul>
    <li>Seed зафиксирован до загрузки данных и инициализации модели: {data.seed}</li>
    <li>Запрошено эпох: {data.epochs}</li>
    <li>Завершено эпох: {data.completed_epochs}</li>
    <li>Лучшая эпоха по среднему val macro-F1: {data.best_epoch}</li>
    <li>Средний val macro-F1 лучшей эпохи: {_fmt(data.best_val_macro_f1)}</li>
    <li>Средний val accuracy лучшей эпохи: {_fmt(data.best_val_accuracy)}</li>
    <li>Возобновление: {html.escape(_resume_text(data.resumed_from))}</li>
  </ul>
  <h2>Данные</h2>
  <p>Датасет <code>{html.escape(data.dataset_name)}</code>. Задачи: {task_list}.
  Строка остаётся в выборке, только если метка каждой задачи входит в её список классов.
  {limit_sentence}
  Размеры сплитов: train {data.n_train}, val {data.n_val}, test {data.n_test}.</p>
  {"".join(count_blocks)}
  <p>{html.escape(data.image_note)}</p>
  <h2>Модель</h2>
  <p>{html.escape(MODEL_PARAGRAPH)}</p>
  <h2>Графики</h2>
  <img src="{html.escape(data.history_figure)}" alt="Кривые обучения">
  {"".join(task_blocks)}
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
