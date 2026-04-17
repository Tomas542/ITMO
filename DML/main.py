import argparse

import pytorch_lightning as pl

from data import get_dataloader
from models import EmoClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=bool, default=False)
    parser.add_argument("--test", type=bool, default=False)
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--feature_type", type=str, choices=["tf_idf", "w2v", "mel", "mfcc", "conformer"], required=True)
    parser.add_argument("--transfer_learning", type=bool, default=True)
    return parser.parse_args()


def main(args) -> None:
    pl.seed_everything(42)
    lm = EmoClassifier(args.feature_type, num_classes=3)
    metric = "val_f1"
    mode = "max"
    callbacks = [
        pl.callbacks.EarlyStopping(
            monitor=metric,
            mode=mode,
            patience=2,
            min_delta=0.02,
            verbose=False,
        ),
        pl.callbacks.ModelCheckpoint(
            monitor=metric,
            mode=mode,
            dirpath="checkpoints",
            filename=f"ckpt_{args.feature_type}",
            auto_insert_metric_name=False,
            save_top_k=1,
        ),
    ]
    trainer = pl.Trainer(
        max_epochs=15,
        num_sanity_val_steps=0,
        enable_model_summary=False,
        gradient_clip_val=1,
        deterministic=True,
        benchmark=False,
        callbacks=callbacks,
        logger=False,
    )
    ckpt_path = args.ckpt_path
    if args.train:
        train_loader = get_dataloader(args.feature_type, "train")
        val_loader = get_dataloader(args.feature_type, "val")
        trainer.fit(lm, train_dataloaders=train_loader, val_dataloaders=val_loader)
        ckpt_path = "best"
    if args.test:
        if ckpt_path is None:
            raise RuntimeError("No checkpoint path was passed")
        test_loader = get_dataloader(args.feature_type, "test")
        trainer.test(lm, dataloaders=test_loader, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main(parse_args())
