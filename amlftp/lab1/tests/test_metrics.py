"""Known-value checks for the reported classification scores."""

from __future__ import annotations

import pytest

from amlftp.metrics import score_predictions


def test_scores_match_a_fixed_confusion() -> None:
    score = score_predictions([0, 0, 1, 1], [0, 0, 1, 0], ["Apparel", "Footwear"], name="Кросс-attention")
    assert score.accuracy == pytest.approx(0.75)
    assert score.macro_f1 == pytest.approx((0.8 + (2 / 3)) / 2)
    assert score.per_class[0].precision == pytest.approx(2 / 3)
    assert score.per_class[0].recall == pytest.approx(1)
    assert score.per_class[0].f1 == pytest.approx(0.8)
    assert score.per_class[1].precision == pytest.approx(1)
    assert score.per_class[1].recall == pytest.approx(0.5)
    assert score.per_class[1].f1 == pytest.approx(2 / 3)
    assert score.confusion == ((2, 0), (1, 1))
