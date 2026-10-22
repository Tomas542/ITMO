"""Encode frozen backbones, train cross-attention, and write the report."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple, cast

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizerFast

from amlftp.checkpoint import (
    InferenceState,
    ResumeState,
    cpu_state_dict,
    load_resume,
    save_inference,
    save_resume,
    validate_resume,
)
from amlftp.config import TrainConfig
from amlftp.data import PreparedData, SampleRef, load_prepared_data, pil_to_png, png_to_pil, read_image
from amlftp.encoders import (
    DistilBertTokens,
    ResNet18Tokens,
    build_image_encoder,
    build_image_transform,
    load_distilbert,
    save_distilbert,
    save_image_encoder,
    tokenize_texts,
    transform_image,
)
from amlftp.metrics import score_predictions
from amlftp.model import TokenFusion
from amlftp.report import ExampleView, ReportData, write_reports
from amlftp.seed import capture_rng_state, restore_rng_state, set_seed
from amlftp.visualize import (
    image_to_text_weights,
    plot_confusion,
    plot_history,
    plot_metric_bars,
    save_example_figure,
    text_to_image_map,
    visible_tokens,
)

CACHE_VERSION = 1
HISTORY_KEYS: tuple[str, ...] = (
    "train_loss",
    "val_loss",
    "train_macro_f1",
    "val_macro_f1",
    "val_accuracy",
    "image_val_macro_f1",
    "text_val_macro_f1",
)
FUSION_NAME = "Кросс-attention"
IMAGE_NAME = "Только фото"
TEXT_NAME = "Только текст"
IMAGE_NOTE = (
    "Исходные фотографии в датасете — миниатюры около 60 px. "
    "Перед энкодером они приводятся к 224 px, карта cross-attention имеет размер 7×7."
)


@dataclass
class SplitCache:
    """Token tensors of one split. Test rows also keep PNG bytes for the report."""

    image_tokens: torch.Tensor
    text_tokens: torch.Tensor
    attention_mask: torch.Tensor
    input_ids: torch.Tensor
    labels: torch.Tensor
    texts: list[str]
    images_png: list[bytes]


class EpochResult(NamedTuple):
    """State after one finished epoch."""

    completed_epochs: int
    history: dict[str, list[float]]
    best_macro_f1: float
    improved: bool
    model_state: dict[str, torch.Tensor]
    best_state: dict[str, torch.Tensor]


EpochCallback = Callable[[EpochResult, torch.optim.Optimizer], None]


class TokenDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Cached encoder tokens for the trainable fusion."""

    def __init__(self, split: SplitCache) -> None:
        self.split = split

    def __len__(self) -> int:
        return int(self.split.labels.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.split.image_tokens[index].float(),
            self.split.text_tokens[index].float(),
            self.split.attention_mask[index].long(),
            self.split.labels[index].long(),
        )


def empty_history() -> dict[str, list[float]]:
    """Create the metric lists stored in the resume checkpoint."""
    return {key: [] for key in HISTORY_KEYS}


def fit_model(
    model: TokenFusion,
    optimizer: torch.optim.Optimizer,
    train_split: SplitCache,
    val_split: SplitCache,
    *,
    label_names: Sequence[str],
    epochs: int,
    start_epoch: int,
    batch_size: int,
    seed: int,
    device: torch.device,
    history: dict[str, list[float]],
    best_macro_f1: float,
    best_state: dict[str, torch.Tensor],
    on_epoch_end: EpochCallback | None = None,
) -> tuple[dict[str, list[float]], float, dict[str, torch.Tensor]]:
    """Train fusion and unimodal heads. Returns history, best macro-F1 and best weights."""
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    for epoch in range(start_epoch, epochs):
        train_loss, train_true, train_fusion, _, _ = _run_epoch(
            model,
            _loader(train_split, batch_size=batch_size, shuffle=True, seed=seed, epoch=epoch),
            criterion,
            device,
            train=True,
            optimizer=optimizer,
        )
        val_loss, val_true, val_fusion, val_image, val_text = _run_epoch(
            model,
            _loader(val_split, batch_size=batch_size, shuffle=False, seed=seed, epoch=epoch),
            criterion,
            device,
            train=False,
            optimizer=None,
        )
        train_score = score_predictions(train_true, train_fusion, label_names, name=FUSION_NAME)
        fusion_score = score_predictions(val_true, val_fusion, label_names, name=FUSION_NAME)
        image_score = score_predictions(val_true, val_image, label_names, name=IMAGE_NAME)
        text_score = score_predictions(val_true, val_text, label_names, name=TEXT_NAME)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_macro_f1"].append(train_score.macro_f1)
        history["val_macro_f1"].append(fusion_score.macro_f1)
        history["val_accuracy"].append(fusion_score.accuracy)
        history["image_val_macro_f1"].append(image_score.macro_f1)
        history["text_val_macro_f1"].append(text_score.macro_f1)
        improved = fusion_score.macro_f1 > best_macro_f1
        if improved:
            best_macro_f1 = fusion_score.macro_f1
            best_state = cpu_state_dict(model)
        completed = epoch + 1
        print(
            f"Эпоха {completed}/{epochs}: loss {train_loss:.4f}/{val_loss:.4f}, "
            f"macro-F1 fusion {fusion_score.macro_f1:.3f}, фото {image_score.macro_f1:.3f}, "
            f"текст {text_score.macro_f1:.3f}",
            flush=True,
        )
        if on_epoch_end is not None:
            on_epoch_end(
                EpochResult(
                    completed_epochs=completed,
                    history=history,
                    best_macro_f1=best_macro_f1,
                    improved=improved,
                    model_state=cpu_state_dict(model),
                    best_state=best_state,
                ),
                optimizer,
            )
    return history, best_macro_f1, best_state


def run_training(config: TrainConfig) -> None:
    """Fix the seed, train or resume, and write reports plus checkpoints."""
    if config.epochs < 1:
        raise ValueError("epochs должен быть положительным")
    if config.resume is not None and not config.resume.is_file():
        raise FileNotFoundError(f"Чекпоинт для возобновления не найден: {config.resume}")

    resume_state: ResumeState | None = None
    if config.resume is not None:
        resume_state = load_resume(config.resume)
        validate_resume(resume_state, config)

    set_seed(config.seed)
    print(f"Seed зафиксирован: {config.seed}", flush=True)

    prepared = load_prepared_data(config)
    label_names = list(prepared.labels)
    if resume_state is not None and resume_state.labels != label_names:
        raise RuntimeError("Список классов в чекпоинте не совпадает с подготовленными данными")

    config.weights_dir.mkdir(parents=True, exist_ok=True)
    config.reports_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}", flush=True)

    image_encoder, text_encoder, tokenizer = _ensure_encoders(config, device)
    train_split, val_split, test_split = _ensure_cache(
        config,
        prepared,
        image_encoder,
        text_encoder,
        tokenizer,
        device,
    )
    image_encoder.cpu()
    text_encoder.cpu()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    model = TokenFusion(
        image_dim=config.image_dim,
        text_dim=config.text_dim,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_classes=len(label_names),
        dropout=config.dropout,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    history = empty_history()
    start_epoch = 0
    best_macro_f1 = -1.0
    best_state = cpu_state_dict(model)
    if resume_state is not None:
        model.load_state_dict(resume_state.model_state)
        optimizer.load_state_dict(resume_state.optimizer_state)
        _move_optimizer_to_device(optimizer, device)
        for key in HISTORY_KEYS:
            history[key] = list(resume_state.history.get(key, []))
        start_epoch = resume_state.completed_epochs
        best_macro_f1 = resume_state.best_macro_f1
        best_state = resume_state.best_model_state
        restore_rng_state(resume_state.rng)
        print(f"Возобновление с эпохи {start_epoch}", flush=True)

    def on_epoch_end(result: EpochResult, epoch_optimizer: torch.optim.Optimizer) -> None:
        _save_epoch(config, result, epoch_optimizer, label_names)

    history, _best_macro_f1, best_state = fit_model(
        model,
        optimizer,
        train_split,
        val_split,
        label_names=label_names,
        epochs=config.epochs,
        start_epoch=start_epoch,
        batch_size=config.batch_size,
        seed=config.seed,
        device=device,
        history=history,
        best_macro_f1=best_macro_f1,
        best_state=best_state,
        on_epoch_end=on_epoch_end,
    )
    model.load_state_dict(best_state)
    model.to(device)
    save_inference(config.weights_dir / "model.pt", _inference_state(config, best_state, label_names))
    _write_report(config, model, tokenizer, test_split, label_names, history, device, prepared)


def _ensure_encoders(
    config: TrainConfig,
    device: torch.device,
) -> tuple[ResNet18Tokens, DistilBertTokens, DistilBertTokenizerFast]:
    resnet_path = config.weights_dir / "resnet18.pt"
    text_dir = config.weights_dir / "distilbert"
    ready = resnet_path.is_file() and (text_dir / "config.json").is_file()
    if ready:
        print("Читаю сохранённые энкодеры", flush=True)
        image_encoder, _backbone = build_image_encoder(resnet_path)
        text_encoder, tokenizer = load_distilbert(config.text_model_name, text_dir)
    else:
        print("Скачиваю ResNet-18 и DistilBERT", flush=True)
        image_encoder, backbone = build_image_encoder(None)
        text_encoder, tokenizer = load_distilbert(config.text_model_name, None)
        save_image_encoder(backbone, resnet_path)
        save_distilbert(text_encoder, tokenizer, text_dir)
    image_encoder.to(device)
    text_encoder.to(device)
    return image_encoder, text_encoder, tokenizer


def _ensure_cache(
    config: TrainConfig,
    prepared: PreparedData,
    image_encoder: ResNet18Tokens,
    text_encoder: DistilBertTokens,
    tokenizer: DistilBertTokenizerFast,
    device: torch.device,
) -> tuple[SplitCache, SplitCache, SplitCache]:
    cache_path = config.weights_dir / "cache.pt"
    labels = list(prepared.labels)
    if cache_path.is_file():
        payload = cast(dict[str, Any], torch.load(cache_path, map_location="cpu", weights_only=False))
        if _cache_matches(payload, config, labels):
            test_split = _split_from_dict(cast(dict[str, Any], payload["test"]))
            if len(test_split.images_png) == int(test_split.labels.shape[0]):
                print(f"Использую кэш токенов {cache_path}", flush=True)
                return (
                    _split_from_dict(cast(dict[str, Any], payload["train"])),
                    _split_from_dict(cast(dict[str, Any], payload["val"])),
                    test_split,
                )
    print("Кодирую замороженные энкодеры в кэш", flush=True)
    label_to_id = {label: index for index, label in enumerate(labels)}
    transform = build_image_transform(config.image_size)
    splits: dict[str, SplitCache] = {}
    for name, refs, keep_images in (
        ("train", prepared.train, False),
        ("val", prepared.val, False),
        ("test", prepared.test, True),
    ):
        print(f"Сплит {name}", flush=True)
        splits[name] = encode_split(
            prepared.dataset,
            refs,
            label_to_id=label_to_id,
            image_encoder=image_encoder,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transform=transform,
            max_length=config.max_length,
            batch_size=config.batch_size,
            device=device,
            keep_images=keep_images,
            expected_image_tokens=config.expected_image_tokens,
            expected_image_dim=config.image_dim,
            expected_text_dim=config.text_dim,
        )
    torch.save(
        {
            "version": CACHE_VERSION,
            "seed": config.seed,
            "max_per_class": config.max_per_class,
            "labels": labels,
            "image_size": config.image_size,
            "max_length": config.max_length,
            "train": _split_to_dict(splits["train"]),
            "val": _split_to_dict(splits["val"]),
            "test": _split_to_dict(splits["test"]),
        },
        cache_path,
    )
    return splits["train"], splits["val"], splits["test"]


def encode_split(
    dataset: Any,
    refs: Sequence[SampleRef],
    *,
    label_to_id: dict[str, int],
    image_encoder: nn.Module,
    text_encoder: nn.Module,
    tokenizer: DistilBertTokenizerFast,
    transform: Any,
    max_length: int,
    batch_size: int,
    device: torch.device,
    keep_images: bool,
    expected_image_tokens: int,
    expected_image_dim: int,
    expected_text_dim: int,
) -> SplitCache:
    """Run frozen encoders and return float16 tokens on CPU."""
    if not refs:
        raise RuntimeError("Пустой сплит")
    image_parts: list[torch.Tensor] = []
    text_parts: list[torch.Tensor] = []
    mask_parts: list[torch.Tensor] = []
    id_parts: list[torch.Tensor] = []
    label_ids: list[int] = []
    texts: list[str] = []
    images_png: list[bytes] = []
    total = len(refs)
    for start in range(0, total, batch_size):
        batch_refs = list(refs[start : start + batch_size])
        images = [read_image(dataset, ref.index) for ref in batch_refs]
        batch_images = torch.stack([transform_image(transform, image) for image in images]).to(device)
        input_ids, attention_mask = tokenize_texts(tokenizer, [ref.text for ref in batch_refs], max_length)
        with torch.inference_mode():
            image_tokens = image_encoder(batch_images)
            text_tokens = text_encoder(input_ids.to(device), attention_mask.to(device))
        if image_tokens.shape[1] != expected_image_tokens or image_tokens.shape[-1] != expected_image_dim:
            raise RuntimeError(
                f"Визуальные токены имеют форму {tuple(image_tokens.shape)}, "
                f"ожидалось N×{expected_image_tokens}×{expected_image_dim}"
            )
        if text_tokens.shape[-1] != expected_text_dim:
            raise RuntimeError(f"Текстовые токены имеют размер {text_tokens.shape[-1]}, ожидалось {expected_text_dim}")
        image_parts.append(image_tokens.detach().cpu().half())
        text_parts.append(text_tokens.detach().cpu().half())
        mask_parts.append(attention_mask.detach().cpu().long())
        id_parts.append(input_ids.detach().cpu().long())
        label_ids.extend(label_to_id[ref.label] for ref in batch_refs)
        texts.extend(ref.text for ref in batch_refs)
        if keep_images:
            images_png.extend(pil_to_png(image) for image in images)
        print(f"  закодировано {min(start + batch_size, total)}/{total}", flush=True)
    return SplitCache(
        image_tokens=torch.cat(image_parts, dim=0),
        text_tokens=torch.cat(text_parts, dim=0),
        attention_mask=torch.cat(mask_parts, dim=0),
        input_ids=torch.cat(id_parts, dim=0),
        labels=torch.tensor(label_ids, dtype=torch.long),
        texts=texts,
        images_png=images_png,
    )


def _run_epoch(
    model: TokenFusion,
    loader: DataLoader[Any],
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
    *,
    train: bool,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, list[int], list[int], list[int], list[int]]:
    if train:
        if optimizer is None:
            raise ValueError("Для обучения нужен оптимизатор")
        model.train()
    else:
        model.eval()
    total_loss = 0.0
    seen = 0
    y_true: list[int] = []
    fusion_pred: list[int] = []
    image_pred: list[int] = []
    text_pred: list[int] = []
    context = torch.enable_grad() if train else torch.inference_mode()
    with context:
        for batch in loader:
            image, text, mask, target = cast(tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], batch)
            image = image.to(device)
            text = text.to(device)
            mask = mask.to(device)
            target = target.to(device)
            pad = mask == 0
            if train and optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            output = model(image, text, pad)
            image_logits = model.image_logits(image)
            text_logits = model.text_logits(text, pad)
            fusion_loss = criterion(output.logits, target)
            if train and optimizer is not None:
                baseline_loss = criterion(image_logits, target) + criterion(text_logits, target)
                (fusion_loss + baseline_loss).backward()
                optimizer.step()
            batch_size = int(target.size(0))
            total_loss += float(fusion_loss.detach().item()) * batch_size
            seen += batch_size
            y_true.extend(int(value) for value in target.detach().cpu().tolist())
            fusion_pred.extend(int(value) for value in output.logits.argmax(dim=1).detach().cpu().tolist())
            image_pred.extend(int(value) for value in image_logits.argmax(dim=1).detach().cpu().tolist())
            text_pred.extend(int(value) for value in text_logits.argmax(dim=1).detach().cpu().tolist())
    return total_loss / max(seen, 1), y_true, fusion_pred, image_pred, text_pred


def _write_report(
    config: TrainConfig,
    model: TokenFusion,
    tokenizer: DistilBertTokenizerFast,
    test_split: SplitCache,
    label_names: list[str],
    history: dict[str, list[float]],
    device: torch.device,
    prepared: PreparedData,
) -> None:
    model.eval()
    _loss, y_true, fusion_pred, image_pred, text_pred = _run_epoch(
        model,
        _loader(test_split, batch_size=config.batch_size, shuffle=False, seed=config.seed, epoch=0),
        nn.CrossEntropyLoss(),
        device,
        train=False,
        optimizer=None,
    )
    scores = (
        score_predictions(y_true, fusion_pred, label_names, name=FUSION_NAME),
        score_predictions(y_true, image_pred, label_names, name=IMAGE_NAME),
        score_predictions(y_true, text_pred, label_names, name=TEXT_NAME),
    )
    figures = config.reports_dir / "figures"
    plot_history(history, figures / "history.png")
    plot_metric_bars(scores, figures / "metrics.png")
    plot_confusion(scores[0].confusion, label_names, figures / "confusion.png")
    examples = _build_examples(config, model, tokenizer, test_split, label_names, y_true, fusion_pred, device)
    best_index = int(np.argmax(np.asarray(history["val_macro_f1"], dtype=np.float64)))
    report = ReportData(
        seed=config.seed,
        epochs=config.epochs,
        completed_epochs=len(history["train_loss"]),
        best_epoch=best_index + 1,
        best_val_macro_f1=history["val_macro_f1"][best_index],
        best_val_accuracy=history["val_accuracy"][best_index],
        resumed_from=None if config.resume is None else str(config.resume),
        dataset_name=config.dataset_name,
        max_per_class=config.max_per_class,
        n_train=len(prepared.train),
        n_val=len(prepared.val),
        n_test=len(prepared.test),
        class_counts=prepared.class_counts,
        scores=scores,
        history_figure="figures/history.png",
        confusion_figure="figures/confusion.png",
        metrics_figure="figures/metrics.png",
        examples=tuple(examples),
        image_note=IMAGE_NOTE,
    )
    markdown_path, html_path = write_reports(report, config.reports_dir)
    print(f"Отчёт: {markdown_path}", flush=True)
    print(f"Отчёт: {html_path}", flush=True)


def _build_examples(
    config: TrainConfig,
    model: TokenFusion,
    tokenizer: DistilBertTokenizerFast,
    split: SplitCache,
    label_names: list[str],
    y_true: Sequence[int],
    y_pred: Sequence[int],
    device: torch.device,
) -> list[ExampleView]:
    examples_dir = config.reports_dir / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)
    views: list[ExampleView] = []
    manifest: list[dict[str, str]] = []
    for number, index in enumerate(pick_example_indices(y_true, y_pred, config.num_examples)):
        image = split.image_tokens[index].float().unsqueeze(0).to(device)
        text = split.text_tokens[index].float().unsqueeze(0).to(device)
        mask = split.attention_mask[index].long().unsqueeze(0).to(device)
        with torch.inference_mode():
            output = model(image, text, mask == 0)
        probabilities_tensor = torch.softmax(output.logits, dim=-1).squeeze(0).cpu()
        probabilities = {
            label: float(probabilities_tensor[label_index]) for label_index, label in enumerate(label_names)
        }
        attention_map = text_to_image_map(output.attn_text_to_image, mask.squeeze(0))
        raw_weights = image_to_text_weights(output.attn_image_to_text, mask.squeeze(0))
        token_ids = [int(value) for value in split.input_ids[index].tolist()]
        tokens = [str(token) for token in tokenizer.convert_ids_to_tokens(token_ids)]
        shown_tokens, shown_weights = visible_tokens(tokens, [float(value) for value in raw_weights])
        photo = png_to_pil(split.images_png[index])
        photo_name = f"{number}.png"
        photo.save(examples_dir / photo_name)
        figure_name = f"figures/example_{number}.png"
        save_example_figure(
            photo,
            attention_map,
            label_names,
            [probabilities[label] for label in label_names],
            shown_tokens,
            shown_weights,
            config.reports_dir / figure_name,
        )
        true_label = label_names[int(y_true[index])]
        pred_label = label_names[int(probabilities_tensor.argmax().item())]
        views.append(
            ExampleView(
                text=split.texts[index],
                true_label=true_label,
                pred_label=pred_label,
                probabilities=probabilities,
                figure_path=figure_name,
            )
        )
        manifest.append({"image": f"examples/{photo_name}", "text": split.texts[index]})
    (examples_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return views


def pick_example_indices(labels: Sequence[int], predictions: Sequence[int], limit: int) -> list[int]:
    """Prefer one correct prediction of each class, then fill in dataset order."""
    chosen: list[int] = []
    seen: set[int] = set()
    for index, (label, prediction) in enumerate(zip(labels, predictions, strict=True)):
        if label == prediction and label not in seen:
            chosen.append(index)
            seen.add(label)
        if len(chosen) == limit:
            return chosen
    for index in range(len(labels)):
        if index not in chosen:
            chosen.append(index)
        if len(chosen) == limit:
            break
    return chosen


def _save_epoch(
    config: TrainConfig,
    result: EpochResult,
    optimizer: torch.optim.Optimizer,
    label_names: list[str],
) -> None:
    save_resume(
        config.weights_dir / "last.pt",
        ResumeState(
            completed_epochs=result.completed_epochs,
            seed=config.seed,
            max_per_class=config.max_per_class,
            best_macro_f1=result.best_macro_f1,
            history={key: list(result.history[key]) for key in HISTORY_KEYS},
            model_state=result.model_state,
            best_model_state=result.best_state,
            optimizer_state=optimizer.state_dict(),
            rng=capture_rng_state(),
            labels=label_names,
            d_model=config.d_model,
            n_heads=config.n_heads,
            dropout=config.dropout,
            image_dim=config.image_dim,
            text_dim=config.text_dim,
            image_size=config.image_size,
            max_length=config.max_length,
        ),
    )
    if result.improved:
        save_inference(config.weights_dir / "model.pt", _inference_state(config, result.best_state, label_names))


def _inference_state(config: TrainConfig, state: dict[str, torch.Tensor], labels: list[str]) -> InferenceState:
    return InferenceState(
        model_state=state,
        labels=labels,
        seed=config.seed,
        d_model=config.d_model,
        n_heads=config.n_heads,
        dropout=config.dropout,
        image_dim=config.image_dim,
        text_dim=config.text_dim,
        image_size=config.image_size,
        max_length=config.max_length,
        text_model_name=config.text_model_name,
    )


def _loader(split: SplitCache, *, batch_size: int, shuffle: bool, seed: int, epoch: int) -> DataLoader[Any]:
    generator = None
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(seed + epoch + 1)
    return DataLoader(TokenDataset(split), batch_size=batch_size, shuffle=shuffle, generator=generator)


def _move_optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for param_state in optimizer.state.values():
        for key, value in list(param_state.items()):
            if isinstance(value, torch.Tensor):
                param_state[key] = value.to(device)


def _cache_matches(payload: dict[str, Any], config: TrainConfig, labels: list[str]) -> bool:
    return (
        payload.get("version") == CACHE_VERSION
        and payload.get("seed") == config.seed
        and payload.get("max_per_class") == config.max_per_class
        and payload.get("labels") == labels
        and payload.get("image_size") == config.image_size
        and payload.get("max_length") == config.max_length
    )


def _split_to_dict(split: SplitCache) -> dict[str, Any]:
    return {
        "image_tokens": split.image_tokens,
        "text_tokens": split.text_tokens,
        "attention_mask": split.attention_mask,
        "input_ids": split.input_ids,
        "labels": split.labels,
        "texts": split.texts,
        "images_png": split.images_png,
    }


def _split_from_dict(payload: dict[str, Any]) -> SplitCache:
    return SplitCache(
        image_tokens=cast(torch.Tensor, payload["image_tokens"]),
        text_tokens=cast(torch.Tensor, payload["text_tokens"]),
        attention_mask=cast(torch.Tensor, payload["attention_mask"]),
        input_ids=cast(torch.Tensor, payload["input_ids"]),
        labels=cast(torch.Tensor, payload["labels"]),
        texts=[str(text) for text in cast(list[object], payload["texts"])],
        images_png=[cast(bytes, image) for image in cast(list[object], payload["images_png"])],
    )
