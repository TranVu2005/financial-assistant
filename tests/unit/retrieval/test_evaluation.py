import pytest

from financial_report_qa.retrieval.evaluation import score_at_10


def test_score_at_10_uses_fixed_precision_denominator() -> None:
    metrics = score_at_10(predicted=("a", "x"), gold=("a", "b"))

    assert metrics.true_positive == 1
    assert metrics.precision == pytest.approx(0.1)
    assert metrics.recall == pytest.approx(0.5)
    assert metrics.f2 == pytest.approx(5 * 0.1 * 0.5 / (4 * 0.1 + 0.5))


def test_score_at_10_rejects_duplicate_predictions() -> None:
    with pytest.raises(ValueError, match="unique"):
        score_at_10(predicted=("a", "a"), gold=("a",))
