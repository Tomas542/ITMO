"""Training configuration for hard-parameter-sharing multi-task classification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Column order is the order of label-tensor columns and of the task heads.
# usage drops Smart Casual, Party, Travel and Home: each has fewer than 70 rows.
DEFAULT_TASKS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("masterCategory", ("Accessories", "Apparel", "Footwear", "Personal Care")),
    ("gender", ("Boys", "Girls", "Men", "Unisex", "Women")),
    ("season", ("Fall", "Spring", "Summer", "Winter")),
    ("usage", ("Casual", "Ethnic", "Formal", "Sports")),
)

TASK_TITLES: dict[str, str] = {
    "masterCategory": "Категория",
    "gender": "Пол",
    "season": "Сезон",
    "usage": "Назначение",
}


def task_title(name: str) -> str:
    """Human-readable task name. Unknown tasks keep the dataset field name."""
    return TASK_TITLES.get(name, name)


@dataclass(frozen=True)
class TrainConfig:
    """Arguments that define one reproducible training run."""

    seed: int = 42
    epochs: int = 5
    max_per_class: int = 1500
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.01
    image_size: int = 224
    max_length: int = 32
    d_model: int = 256
    n_heads: int = 4
    dropout: float = 0.1
    image_dim: int = 512
    text_dim: int = 768
    weights_dir: Path = Path("weights")
    reports_dir: Path = Path("reports")
    resume: Path | None = None
    dataset_name: str = "ashraq/fashion-product-images-small"
    text_model_name: str = "distilbert-base-uncased"
    task_labels: tuple[tuple[str, tuple[str, ...]], ...] = DEFAULT_TASKS
    num_examples: int = 6
    expected_image_tokens: int = 49
