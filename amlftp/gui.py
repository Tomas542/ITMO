"""Gradio demo. Weights are read from a directory and never created here."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import threading
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path(tempfile.gettempdir()) / "amlftp-hf"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import gradio as gr  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from PIL import Image  # noqa: E402

from amlftp.predict import InferenceBundle, load_bundle, predict  # noqa: E402
from amlftp.visualize import render_example_figure  # noqa: E402

_PREDICT_LOCK = threading.Lock()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse demo arguments. ``--weights`` points at the read-only checkpoint directory."""
    parser = argparse.ArgumentParser(description="Демо мультимодального классификатора")
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
    """Assemble the Gradio interface around the loaded cross-attention model."""

    def predict_ui(image: Image.Image | None, text: str | None) -> tuple[dict[str, float], Figure]:
        if image is None or text is None or text.strip() == "":
            raise gr.Error("Нужны и фото, и название товара.")
        plt.close("all")
        with _PREDICT_LOCK:
            prediction = predict(bundle, image.convert("RGB"), text)
        figure = render_example_figure(
            image,
            prediction.attention_map,
            bundle.labels,
            [prediction.probabilities[label] for label in bundle.labels],
            prediction.tokens,
            prediction.token_weights,
        )
        return prediction.probabilities, figure

    with gr.Blocks(title="Категория товара") as demo:
        gr.Markdown(
            """# Категория товара по фото и названию

Модальности объединяются двунаправленным cross-attention: клетки фото читают слова названия,
а слова читают клетки фото. Рядом с вероятностями показаны эти веса.
"""
        )
        with gr.Row():
            image_input = gr.Image(type="pil", label="Фото")
            text_input = gr.Textbox(label="Название", lines=2)
        submit = gr.Button("Классифицировать")
        label_output = gr.Label(label="Вероятности", num_top_classes=len(bundle.labels))
        plot_output = gr.Plot(label="Cross-attention")
        submit.click(predict_ui, inputs=[image_input, text_input], outputs=[label_output, plot_output])
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
