"""Train the image-text classifier."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent / "data" / "hf"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from amlftp.checkpoint import ResumeConfigError, SeedMismatchError  # noqa: E402
from amlftp.config import TrainConfig  # noqa: E402
from amlftp.train_loop import run_training  # noqa: E402


def parse_args(argv: list[str] | None = None) -> TrainConfig:
    """Parse the training command into a config. The seed is stored, not applied here."""
    parser = argparse.ArgumentParser(description="Обучение мультимодального классификатора с cross-attention")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-per-class", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weights-dir", type=Path, default=Path("weights"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--resume", type=Path, default=None)
    args = parser.parse_args(argv)
    return TrainConfig(
        seed=int(args.seed),
        epochs=int(args.epochs),
        max_per_class=int(args.max_per_class),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        weights_dir=Path(args.weights_dir),
        reports_dir=Path(args.reports_dir),
        resume=None if args.resume is None else Path(args.resume),
    )


def main(argv: list[str] | None = None) -> None:
    """Run training and turn expected setup errors into a non-zero exit."""
    try:
        run_training(parse_args(argv))
    except (SeedMismatchError, ResumeConfigError, FileNotFoundError, ValueError) as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
