"""Gradio demo for the multi-task classifier. Weights are read and never created here."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
os.environ.setdefault("HF_HOME", str(Path(tempfile.gettempdir()) / "amlftp-hf"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import gradio as gr  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from PIL import Image  # noqa: E402

from amlftp.config import task_title  # noqa: E402
from amlftp.predict import InferenceBundle, load_bundle, predict  # noqa: E402
from amlftp.visualize import render_example_figure  # noqa: E402

_PREDICT_LOCK = threading.Lock()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse demo arguments. ``--weights`` points at the read-only checkpoint directory."""
    parser = argparse.ArgumentParser(description="Демо мультизадачного классификатора")
    parser.add_argument("--weights", type=Path, default=Path("weights"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    return parser.parse_args(argv)


def load_examples(reports_dir: Path) -> list[list[str]]:
    """Read example photos saved by training. Missing reports simply leave the list empty."""
    manifest_path = reports_dir / "examples" / "manifest.json"
    if not manifest_path.is_file():
        return []
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    examples: list[list[str]] = []
    if not isinstance(payload, list):
        return []
    for item in payload:
        if not isinstance(item, dict):
            continue
        image_path = reports_dir / str(item.get("image", ""))
        text = str(item.get("text", ""))
        if image_path.is_file() and text.strip() != "":
            examples.append([str(image_path), text])
    return examples


def build_demo(bundle: InferenceBundle, examples: list[list[str]]) -> gr.Blocks:
    """Assemble the Gradio interface around the shared cross-attention model."""
    task_names = tuple(bundle.task_labels)
    primary = task_names[0]
    titles = ", ".join(task_title(task) for task in task_names)

    def predict_ui(image: Image.Image | None, text: str | None) -> tuple[dict[str, float] | Figure, ...]:
        if image is None or text is None or text.strip() == "":
            raise gr.Error("Нужны и фото, и название товара.")
        plt.close("all")
        with _PREDICT_LOCK:
            prediction = predict(bundle, image.convert("RGB"), text)
        primary_labels = bundle.task_labels[primary]
        figure = render_example_figure(
            image,
            prediction.attention_map,
            primary_labels,
            [prediction.probabilities[primary][label] for label in primary_labels],
            prediction.tokens,
            prediction.token_weights,
            probability_title=task_title(primary),
        )
        probabilities = tuple(prediction.probabilities[task] for task in task_names)
        return (*probabilities, figure)

    with gr.Blocks(title="Атрибуты товара") as demo:
        gr.Markdown(
            f"""# Атрибуты товара по фото и названию

Общий cross-attention читает фото и название один раз. Отдельные головы предсказывают: {titles}.
Рядом с вероятностями категории показаны веса внимания.
"""
        )
        with gr.Row():
            image_input = gr.Image(type="pil", label="Фото")
            text_input = gr.Textbox(label="Название", lines=2)
        submit = gr.Button("Классифицировать")
        label_outputs = [
            gr.Label(label=task_title(task), num_top_classes=len(bundle.task_labels[task])) for task in task_names
        ]
        plot_output = gr.Plot(label="Cross-attention")
        submit.click(
            predict_ui,
            inputs=[image_input, text_input],
            outputs=[*label_outputs, plot_output],
        )
        if examples:
            gr.Examples(examples=examples, inputs=[image_input, text_input])
    return demo


def main(argv: list[str] | None = None) -> None:
    """Load weights and start the server. A missing checkpoint exits without creating files."""
    args = parse_args(argv)
    try:
        bundle = load_bundle(Path(args.weights))
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1) from exc
    reports_dir = Path(args.reports_dir)
    demo = build_demo(bundle, load_examples(reports_dir))
    demo.launch(
        server_name=str(args.host),
        server_port=int(args.port),
        allowed_paths=[str(reports_dir.resolve())],
    )


if __name__ == "__main__":
    main()
