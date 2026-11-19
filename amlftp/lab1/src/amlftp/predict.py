"""Load a read-only checkpoint and predict one photo plus title."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from transformers import DistilBertTokenizerFast

from amlftp.checkpoint import load_inference
from amlftp.encoders import (
    DistilBertTokens,
    ResNet18Tokens,
    build_image_encoder,
    build_image_transform,
    load_distilbert,
    tokenize_texts,
    transform_image,
)
from amlftp.model import TokenFusion
from amlftp.visualize import image_to_text_weights, text_to_image_map, visible_tokens


@dataclass
class InferenceBundle:
    """Encoders and fusion loaded for the demo."""

    fusion: TokenFusion
    image_encoder: ResNet18Tokens
    text_encoder: DistilBertTokens
    tokenizer: DistilBertTokenizerFast
    labels: list[str]
    image_size: int
    max_length: int
    device: torch.device
    transform: transforms.Compose


@dataclass(frozen=True)
class Prediction:
    """One demo prediction and the attention values used to draw it."""

    label: str
    probabilities: dict[str, float]
    attention_map: np.ndarray
    tokens: list[str]
    token_weights: list[float]


def require_weight_files(weights_dir: Path) -> None:
    """Fail if the read-only weights directory is incomplete. Do not create it."""
    text_dir = weights_dir / "distilbert"
    tokenizer_ready = (text_dir / "tokenizer.json").is_file() or (text_dir / "vocab.txt").is_file()
    required = {
        str(weights_dir / "model.pt"): (weights_dir / "model.pt").is_file(),
        str(weights_dir / "resnet18.pt"): (weights_dir / "resnet18.pt").is_file(),
        str(text_dir / "config.json"): (text_dir / "config.json").is_file(),
        str(text_dir / "tokenizer"): tokenizer_ready,
    }
    missing = [path for path, exists in required.items() if not exists]
    if missing:
        raise FileNotFoundError("Не найдены веса: " + ", ".join(missing))


def load_bundle(weights_dir: Path, device: torch.device | None = None) -> InferenceBundle:
    """Read fusion and encoder weights. The directory is only opened for reading."""
    require_weight_files(weights_dir)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inference = load_inference(weights_dir / "model.pt")
    image_encoder, _backbone = build_image_encoder(weights_dir / "resnet18.pt")
    text_encoder, tokenizer = load_distilbert(inference.text_model_name, weights_dir / "distilbert")
    fusion = TokenFusion(
        image_dim=inference.image_dim,
        text_dim=inference.text_dim,
        d_model=inference.d_model,
        n_heads=inference.n_heads,
        n_classes=len(inference.labels),
        dropout=inference.dropout,
    )
    fusion.load_state_dict(inference.model_state)
    fusion.to(device).eval()
    image_encoder.to(device)
    text_encoder.to(device)
    return InferenceBundle(
        fusion=fusion,
        image_encoder=image_encoder,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        labels=inference.labels,
        image_size=inference.image_size,
        max_length=inference.max_length,
        device=device,
        transform=build_image_transform(inference.image_size),
    )


def predict(bundle: InferenceBundle, image: Image.Image, text: str) -> Prediction:
    """Run bidirectional cross-attention on one image and title."""
    cleaned = text.strip()
    if cleaned == "":
        raise ValueError("Пустое название")
    tensor = transform_image(bundle.transform, image).unsqueeze(0).to(bundle.device)
    input_ids, attention_mask = tokenize_texts(bundle.tokenizer, [cleaned], bundle.max_length)
    input_ids = input_ids.to(bundle.device)
    attention_mask = attention_mask.to(bundle.device)
    with torch.inference_mode():
        image_tokens = bundle.image_encoder(tensor)
        text_tokens = bundle.text_encoder(input_ids, attention_mask)
        output = bundle.fusion(image_tokens, text_tokens, attention_mask == 0)
    probabilities_tensor = torch.softmax(output.logits, dim=-1).squeeze(0).cpu()
    probabilities = {label: float(probabilities_tensor[index]) for index, label in enumerate(bundle.labels)}
    mask = attention_mask.squeeze(0)
    raw_weights = image_to_text_weights(output.attn_image_to_text, mask)
    token_ids = [int(value) for value in input_ids.squeeze(0).cpu().tolist()]
    tokens = [str(token) for token in bundle.tokenizer.convert_ids_to_tokens(token_ids)]
    shown_tokens, shown_weights = visible_tokens(tokens, [float(value) for value in raw_weights])
    pred_index = int(probabilities_tensor.argmax().item())
    return Prediction(
        label=bundle.labels[pred_index],
        probabilities=probabilities,
        attention_map=text_to_image_map(output.attn_text_to_image, mask),
        tokens=shown_tokens,
        token_weights=shown_weights,
    )
