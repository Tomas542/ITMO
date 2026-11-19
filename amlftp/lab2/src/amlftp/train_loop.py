"""Encode frozen backbones, train the shared trunk and task heads, and write the report."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
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
from amlftp.config import TrainConfig, task_title
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
from amlftp.report import ExampleView, ReportData, TaskReport, write_reports
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

CACHE_VERSION = 2
FUSION_NAME = "Кросс-attention"
IMAGE_NAME = "Только фото"
TEXT_NAME = "Только текст"
IMAGE_NOTE = (
    "Исходные фотографии в датасете — миниатюры около 60 px. "
    "Перед энкодером они приводятся к 224 px, карта cross-attention имеет размер 7×7."
)


def history_keys(task_names: Sequence[str]) -> tuple[str, ...]:
    """Metric names stored in the resume checkpoint, in a stable order."""
    per_task = tuple(f"val_macro_f1_{task}" for task in task_names)
    return (
        "train_loss",
        "val_loss",
        "train_macro_f1",
        "val_macro_f1",
        "val_accuracy",
        *per_task,
    )


@dataclass
class SplitCache:
    """Token tensors of one split. Test rows also keep PNG bytes for the report.

    ``labels`` has shape ``[N, n_tasks]`` in ``task_names`` order.
    """

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


def empty_history(task_names: Sequence[str]) -> dict[str, list[float]]:
    """Create the metric lists stored in the resume checkpoint."""
    return {key: [] for key in history_keys(task_names)}


def fit_model(
    model: TokenFusion,
    optimizer: torch.optim.Optimizer,
    train_split: SplitCache,
    val_split: SplitCache,
    *,
    task_labels: Mapping[str, Sequence[str]],
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
    """Train the shared trunk and every task head.

    The recorded macro-F1 is the unweighted mean across tasks. The best checkpoint
    is the highest mean validation macro-F1 of the fused heads.
    """
    if tuple(task_labels) != model.task_names:
        raise ValueError("Порядок задач модели и меток не совпадает")
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    task_names = model.task_names
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
        train_scores = [
            score_predictions(train_true[name], train_fusion[name], task_labels[name], name=FUSION_NAME)
            for name in task_names
        ]
        fusion_scores = [
            score_predictions(val_true[name], val_fusion[name], task_labels[name], name=FUSION_NAME)
            for name in task_names
        ]
        image_scores = [
            score_predictions(val_true[name], val_image[name], task_labels[name], name=IMAGE_NAME)
            for name in task_names
        ]
        text_scores = [
            score_predictions(val_true[name], val_text[name], task_labels[name], name=TEXT_NAME) for name in task_names
        ]
        mean_val_f1 = _mean([score.macro_f1 for score in fusion_scores])

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_macro_f1"].append(_mean([score.macro_f1 for score in train_scores]))
        history["val_macro_f1"].append(mean_val_f1)
        history["val_accuracy"].append(_mean([score.accuracy for score in fusion_scores]))
        for name, score in zip(task_names, fusion_scores, strict=True):
            history[f"val_macro_f1_{name}"].append(score.macro_f1)

        improved = mean_val_f1 > best_macro_f1
        if improved:
            best_macro_f1 = mean_val_f1
            best_state = cpu_state_dict(model)
        completed = epoch + 1
        detail = ", ".join(
            f"{name} {score.macro_f1:.3f}" for name, score in zip(task_names, fusion_scores, strict=True)
        )
        image_mean = _mean([score.macro_f1 for score in image_scores])
        text_mean = _mean([score.macro_f1 for score in text_scores])
        print(
            f"Эпоха {completed}/{epochs}: loss {train_loss:.4f}/{val_loss:.4f}, "
            f"средний macro-F1 fusion {mean_val_f1:.3f}, фото {image_mean:.3f}, текст {text_mean:.3f} ({detail})",
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
    if not config.task_labels:
        raise ValueError("Нужна хотя бы одна задача")
    if config.resume is not None and not config.resume.is_file():
        raise FileNotFoundError(f"Чекпоинт для возобновления не найден: {config.resume}")

    resume_state: ResumeState | None = None
    if config.resume is not None:
        resume_state = load_resume(config.resume)
        validate_resume(resume_state, config)

    set_seed(config.seed)
    print(f"Seed зафиксирован: {config.seed}")

    prepared = load_prepared_data(config)
    task_labels = prepared.task_labels
    task_label_lists = {name: list(labels) for name, labels in task_labels.items()}
    if resume_state is not None and (
        list(resume_state.task_labels) != list(task_label_lists) or resume_state.task_labels != task_label_lists
    ):
        raise RuntimeError("Списки классов в чекпоинте не совпадают с подготовленными данными")

    config.weights_dir.mkdir(parents=True, exist_ok=True)
    config.reports_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}")

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

    model = TokenFusion(
        image_dim=config.image_dim,
        text_dim=config.text_dim,
        d_model=config.d_model,
        n_heads=config.n_heads,
        task_sizes={name: len(labels) for name, labels in task_labels.items()},
        dropout=config.dropout,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    history = empty_history(model.task_names)
    start_epoch = 0
    best_macro_f1 = -1.0
    best_state = cpu_state_dict(model)
    if resume_state is not None:
        model.load_state_dict(resume_state.model_state)
        optimizer.load_state_dict(resume_state.optimizer_state)
        _move_optimizer_to_device(optimizer, device)
        for key in history_keys(model.task_names):
            history[key] = list(resume_state.history.get(key, []))
        start_epoch = resume_state.completed_epochs
        best_macro_f1 = resume_state.best_macro_f1
        best_state = resume_state.best_model_state
        restore_rng_state(resume_state.rng)
        print(f"Возобновление с эпохи {start_epoch}")

    def on_epoch_end(result: EpochResult, epoch_optimizer: torch.optim.Optimizer) -> None:
        _save_epoch(config, result, epoch_optimizer, task_label_lists)

    history, _best_macro_f1, best_state = fit_model(
        model,
        optimizer,
        train_split,
        val_split,
        task_labels=task_labels,
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
    save_inference(
        config.weights_dir / "model.pt",
        _inference_state(config, best_state, task_label_lists),
    )
    _write_report(config, model, tokenizer, test_split, task_labels, history, device, prepared)


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
    task_names = [name for name, _labels in config.task_labels]
    if cache_path.is_file():
        payload = cast(dict[str, Any], torch.load(cache_path, map_location="cpu", weights_only=False))
        if _cache_matches(payload, config):
            test_split = _split_from_dict(cast(dict[str, Any], payload["test"]))
            if len(test_split.images_png) == int(test_split.labels.shape[0]):
                print(f"Использую кэш токенов {cache_path}", flush=True)
                return (
                    _split_from_dict(cast(dict[str, Any], payload["train"])),
                    _split_from_dict(cast(dict[str, Any], payload["val"])),
                    test_split,
                )
    print("Кодирую замороженные энкодеры в кэш", flush=True)
    label_to_id = {task: {label: index for index, label in enumerate(labels)} for task, labels in config.task_labels}
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
            task_names=task_names,
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
            "task_labels": {name: list(labels) for name, labels in config.task_labels},
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
    task_names: Sequence[str],
    label_to_id: dict[str, dict[str, int]],
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
    label_rows: list[list[int]] = []
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
        for ref in batch_refs:
            label_rows.append([label_to_id[task][ref.labels[task]] for task in task_names])
        texts.extend(ref.text for ref in batch_refs)
        if keep_images:
            images_png.extend(pil_to_png(image) for image in images)
        print(f"  закодировано {min(start + batch_size, total)}/{total}", flush=True)
    return SplitCache(
        image_tokens=torch.cat(image_parts, dim=0),
        text_tokens=torch.cat(text_parts, dim=0),
        attention_mask=torch.cat(mask_parts, dim=0),
        input_ids=torch.cat(id_parts, dim=0),
        labels=torch.tensor(label_rows, dtype=torch.long),
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
) -> tuple[
    float,
    dict[str, list[int]],
    dict[str, list[int]],
    dict[str, list[int]],
    dict[str, list[int]],
]:
    if train:
        if optimizer is None:
            raise ValueError("Для обучения нужен оптимизатор")
        model.train()
    else:
        model.eval()

    task_names = model.task_names
    total_loss = 0.0
    seen = 0
    y_true: dict[str, list[int]] = {name: [] for name in task_names}
    fusion_pred: dict[str, list[int]] = {name: [] for name in task_names}
    image_pred: dict[str, list[int]] = {name: [] for name in task_names}
    text_pred: dict[str, list[int]] = {name: [] for name in task_names}

    context = torch.enable_grad() if train else torch.inference_mode()
    with context:
        for batch in loader:
            image, text, mask, target = cast(tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], batch)
            image = image.to(device)
            text = text.to(device)
            mask = mask.to(device)
            target = target.to(device)
            if target.ndim != 2 or target.shape[1] != len(task_names):
                raise RuntimeError(f"Ожидались метки формы N×{len(task_names)}, получено {tuple(target.shape)}")

            pad = mask == 0
            output = model(image, text, pad)
            image_logits = model.image_logits(image)
            text_logits = model.text_logits(text, pad)
            fusion_loss = _task_loss(criterion, output.logits, target, task_names)
            if train and optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                baseline_loss = _task_loss(criterion, image_logits, target, task_names) + _task_loss(
                    criterion, text_logits, target, task_names
                )
                (fusion_loss + baseline_loss).backward()
                optimizer.step()

            batch_size = int(target.size(0))
            total_loss += float(fusion_loss.detach().item()) * batch_size
            seen += batch_size
            for index, name in enumerate(task_names):
                y_true[name].extend(int(value) for value in target[:, index].detach().cpu().tolist())
                fusion_pred[name].extend(
                    int(value) for value in output.logits[name].argmax(dim=1).detach().cpu().tolist()
                )
                image_pred[name].extend(
                    int(value) for value in image_logits[name].argmax(dim=1).detach().cpu().tolist()
                )
                text_pred[name].extend(int(value) for value in text_logits[name].argmax(dim=1).detach().cpu().tolist())
    return total_loss / max(seen, 1), y_true, fusion_pred, image_pred, text_pred


def _task_loss(
    criterion: nn.CrossEntropyLoss,
    logits: Mapping[str, torch.Tensor],
    target: torch.Tensor,
    task_names: Sequence[str],
) -> torch.Tensor:
    """Unweighted sum of per-task cross-entropies. ``target`` columns follow ``task_names``."""
    losses = [criterion(logits[name], target[:, index]) for index, name in enumerate(task_names)]
    return torch.stack(losses).sum()


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("Нет значений для среднего")
    return float(sum(values) / len(values))


def _write_report(
    config: TrainConfig,
    model: TokenFusion,
    tokenizer: DistilBertTokenizerFast,
    test_split: SplitCache,
    task_labels: Mapping[str, Sequence[str]],
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
    figures = config.reports_dir / "figures"
    plot_history(history, figures / "history.png", model.task_names)
    task_reports: list[TaskReport] = []
    for task in model.task_names:
        names = list(task_labels[task])
        scores = (
            score_predictions(y_true[task], fusion_pred[task], names, name=FUSION_NAME),
            score_predictions(y_true[task], image_pred[task], names, name=IMAGE_NAME),
            score_predictions(y_true[task], text_pred[task], names, name=TEXT_NAME),
        )
        confusion_figure = f"figures/confusion_{task}.png"
        metrics_figure = f"figures/metrics_{task}.png"
        title = task_title(task)
        plot_confusion(scores[0].confusion, names, figures / f"confusion_{task}.png", title=f"Матрица ошибок: {title}")
        plot_metric_bars(scores, figures / f"metrics_{task}.png", title=f"Сравнение модальностей: {title}")
        task_reports.append(
            TaskReport(
                name=task,
                scores=scores,
                confusion_figure=confusion_figure,
                metrics_figure=metrics_figure,
            )
        )
    examples = _build_examples(
        config,
        model,
        tokenizer,
        test_split,
        task_labels,
        y_true,
        fusion_pred,
        device,
    )
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
        primary_task=model.task_names[0],
        n_train=len(prepared.train),
        n_val=len(prepared.val),
        n_test=len(prepared.test),
        class_counts=prepared.class_counts,
        tasks=tuple(task_reports),
        history_figure="figures/history.png",
        examples=tuple(examples),
        image_note=IMAGE_NOTE,
    )
    html_path = write_reports(report, config.reports_dir)
    print(f"Отчёт: {html_path}", flush=True)


def _build_examples(
    config: TrainConfig,
    model: TokenFusion,
    tokenizer: DistilBertTokenizerFast,
    split: SplitCache,
    task_labels: Mapping[str, Sequence[str]],
    y_true: Mapping[str, Sequence[int]],
    y_pred: Mapping[str, Sequence[int]],
    device: torch.device,
) -> list[ExampleView]:
    examples_dir = config.reports_dir / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)
    views: list[ExampleView] = []
    manifest: list[dict[str, str]] = []
    primary = model.task_names[0]
    chosen = pick_example_indices(y_true[primary], y_pred[primary], config.num_examples)

    for number, index in enumerate(chosen):
        image = split.image_tokens[index].float().unsqueeze(0).to(device)
        text = split.text_tokens[index].float().unsqueeze(0).to(device)
        mask = split.attention_mask[index].long().unsqueeze(0).to(device)

        with torch.inference_mode():
            output = model(image, text, mask == 0)
        probabilities: dict[str, dict[str, float]] = {}
        true_labels: dict[str, str] = {}
        pred_labels: dict[str, str] = {}
        for task in model.task_names:
            names = list(task_labels[task])
            probabilities_tensor = torch.softmax(output.logits[task], dim=-1).squeeze(0).cpu()
            probabilities[task] = {label: float(probabilities_tensor[i]) for i, label in enumerate(names)}
            pred_labels[task] = names[int(probabilities_tensor.argmax().item())]
            true_labels[task] = names[int(y_true[task][index])]

        attention_map = text_to_image_map(output.attn_text_to_image, mask.squeeze(0))
        raw_weights = image_to_text_weights(output.attn_image_to_text, mask.squeeze(0))
        token_ids = [int(value) for value in split.input_ids[index].tolist()]
        tokens = [str(token) for token in tokenizer.convert_ids_to_tokens(token_ids)]
        shown_tokens, shown_weights = visible_tokens(tokens, [float(value) for value in raw_weights])

        photo = png_to_pil(split.images_png[index])
        photo_name = f"{number}.png"
        photo.save(examples_dir / photo_name)
        figure_name = f"figures/example_{number}.png"
        primary_names = list(task_labels[primary])
        save_example_figure(
            photo,
            attention_map,
            primary_names,
            [probabilities[primary][label] for label in primary_names],
            shown_tokens,
            shown_weights,
            config.reports_dir / figure_name,
            probability_title=task_title(primary),
        )
        views.append(
            ExampleView(
                text=split.texts[index],
                true_labels=true_labels,
                pred_labels=pred_labels,
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
    task_labels: dict[str, list[str]],
) -> None:
    save_resume(
        config.weights_dir / "last.pt",
        ResumeState(
            completed_epochs=result.completed_epochs,
            seed=config.seed,
            max_per_class=config.max_per_class,
            best_macro_f1=result.best_macro_f1,
            history={key: list(result.history[key]) for key in history_keys(tuple(task_labels))},
            model_state=result.model_state,
            best_model_state=result.best_state,
            optimizer_state=optimizer.state_dict(),
            rng=capture_rng_state(),
            task_labels=task_labels,
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
        save_inference(
            config.weights_dir / "model.pt",
            _inference_state(config, result.best_state, task_labels),
        )


def _inference_state(
    config: TrainConfig,
    state: dict[str, torch.Tensor],
    task_labels: dict[str, list[str]],
) -> InferenceState:
    return InferenceState(
        model_state=state,
        task_labels=task_labels,
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


def _cache_matches(payload: dict[str, Any], config: TrainConfig) -> bool:
    return (
        payload.get("version") == CACHE_VERSION
        and payload.get("seed") == config.seed
        and payload.get("max_per_class") == config.max_per_class
        and _task_labels_match(payload.get("task_labels"), config)
        and payload.get("image_size") == config.image_size
        and payload.get("max_length") == config.max_length
    )


def _task_labels_match(stored: object, config: TrainConfig) -> bool:
    if not isinstance(stored, dict):
        return False
    expected_names = [name for name, _labels in config.task_labels]
    expected = {name: list(labels) for name, labels in config.task_labels}
    normalized: dict[str, list[str]] = {}
    for key, value in stored.items():
        if not isinstance(value, list):
            return False
        normalized[str(key)] = [str(label) for label in value]
    return list(normalized) == expected_names and normalized == expected


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
