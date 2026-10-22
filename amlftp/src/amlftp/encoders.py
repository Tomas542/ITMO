"""Frozen image and text encoders that emit token sequences."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.models.resnet import ResNet
from transformers import DistilBertModel, DistilBertTokenizerFast

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


def freeze(module: nn.Module) -> None:
    """Stop gradients and keep batch-norm statistics fixed."""
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad = False


class ResNet18Tokens(nn.Module):
    """ImageNet ResNet-18 without pooling, as a 7 by 7 grid of 512-d tokens."""

    def __init__(self, backbone: ResNet) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        net = self.backbone
        hidden = net.conv1(images)
        hidden = net.bn1(hidden)
        hidden = net.relu(hidden)
        hidden = net.maxpool(hidden)
        hidden = net.layer1(hidden)
        hidden = net.layer2(hidden)
        hidden = net.layer3(hidden)
        hidden = net.layer4(hidden)
        return hidden.flatten(2).transpose(1, 2)

    def train(self, mode: bool = True) -> ResNet18Tokens:
        super().train(False)
        return self


class DistilBertTokens(nn.Module):
    """DistilBERT last hidden states, one vector per text token."""

    def __init__(self, model: DistilBertModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden: torch.Tensor = output.last_hidden_state
        return hidden

    def train(self, mode: bool = True) -> DistilBertTokens:
        super().train(False)
        return self


def build_image_transform(image_size: int) -> transforms.Compose:
    """Resize and normalize images the way ImageNet ResNet expects."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def tokenize_texts(
    tokenizer: Any,
    texts: list[str],
    max_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenize a batch to fixed-length ids and an attention mask."""
    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = cast(torch.Tensor, encoded["input_ids"])
    attention_mask = cast(torch.Tensor, encoded["attention_mask"])
    return input_ids, attention_mask


def load_resnet(weights_path: Path | None) -> ResNet:
    """Load ImageNet weights, or a state dict previously saved into ``weights``."""
    if weights_path is None:
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    else:
        backbone = resnet18(weights=None)
        state = cast(dict[str, torch.Tensor], torch.load(weights_path, map_location="cpu", weights_only=True))
        backbone.load_state_dict(state)
    return backbone


def build_image_encoder(weights_path: Path | None) -> tuple[ResNet18Tokens, ResNet]:
    """Return a frozen token encoder and the backbone whose weights can be saved."""
    backbone = load_resnet(weights_path)
    encoder = ResNet18Tokens(backbone)
    freeze(encoder)
    return encoder, backbone


def save_image_encoder(backbone: ResNet, path: Path) -> None:
    """Write ResNet-18 weights. The file is a tensor state dict."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(backbone.state_dict(), path)


def load_distilbert(
    model_name: str,
    weights_dir: Path | None,
) -> tuple[DistilBertTokens, DistilBertTokenizerFast]:
    """Load DistilBERT from the hub, or from a directory mounted with the checkpoint."""
    if weights_dir is None:
        model = DistilBertModel.from_pretrained(model_name)
        tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
    else:
        model = DistilBertModel.from_pretrained(weights_dir, local_files_only=True)
        tokenizer = DistilBertTokenizerFast.from_pretrained(weights_dir, local_files_only=True)
    encoder = DistilBertTokens(model)
    freeze(encoder)
    return encoder, tokenizer


def save_distilbert(encoder: DistilBertTokens, tokenizer: DistilBertTokenizerFast, path: Path) -> None:
    """Copy the frozen text encoder into the weights directory."""
    path.mkdir(parents=True, exist_ok=True)
    encoder.model.save_pretrained(path)
    tokenizer.save_pretrained(path)


def transform_image(transform: transforms.Compose, image: Image.Image) -> torch.Tensor:
    """Apply the image transform and return a CHW tensor."""
    tensor = transform(image.convert("RGB"))
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Преобразование изображения должно возвращать тензор")
    return tensor
