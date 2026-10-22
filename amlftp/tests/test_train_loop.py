"""A tiny cached run checks that one epoch trains and can be continued."""

from __future__ import annotations

import torch

from amlftp.checkpoint import cpu_state_dict
from amlftp.model import TokenFusion
from amlftp.train_loop import EpochResult, SplitCache, empty_history, fit_model, pick_example_indices


def _cache(count: int, seed: int) -> SplitCache:
    generator = torch.Generator().manual_seed(seed)
    return SplitCache(
        image_tokens=torch.randn(count, 4, 8, generator=generator),
        text_tokens=torch.randn(count, 5, 8, generator=generator),
        attention_mask=torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.long).repeat(count, 1),
        input_ids=torch.ones(count, 5, dtype=torch.long),
        labels=torch.arange(count) % 3,
        texts=["item"] * count,
        images_png=[],
    )


def test_fit_one_epoch_updates_weights_and_history() -> None:
    torch.manual_seed(0)
    model = TokenFusion(image_dim=8, text_dim=8, d_model=16, n_heads=4, n_classes=3, dropout=0.0)
    before = cpu_state_dict(model)["classifier.weight"].clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    seen: list[int] = []

    def on_epoch(result: EpochResult, _optimizer: torch.optim.Optimizer) -> None:
        seen.append(result.completed_epochs)

    history, best_f1, _best_state = fit_model(
        model,
        optimizer,
        _cache(20, 1),
        _cache(8, 2),
        label_names=["a", "b", "c"],
        epochs=1,
        start_epoch=0,
        batch_size=8,
        seed=3,
        device=torch.device("cpu"),
        history=empty_history(),
        best_macro_f1=-1.0,
        best_state=cpu_state_dict(model),
        on_epoch_end=on_epoch,
    )
    assert seen == [1]
    assert len(history["train_loss"]) == 1
    assert len(history["val_macro_f1"]) == 1
    assert best_f1 >= 0
    assert not torch.equal(before, model.state_dict()["classifier.weight"].cpu())


def test_pick_examples_covers_correct_classes() -> None:
    chosen = pick_example_indices([0, 1, 1, 2, 0], [0, 0, 1, 2, 1], 3)
    assert chosen == [0, 2, 3]
