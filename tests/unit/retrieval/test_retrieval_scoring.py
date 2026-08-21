"""F2 macro và MRR5: công thức chấm truy hồi thật của dashboard (8 cột
TABLES/DOCS x Precision/Recall/F2/MRR5), không chỉ F2 tóm tắt trong PDF thể lệ."""

from __future__ import annotations

import pytest

from financial_report_qa.retrieval.retrieval_scoring import (
    f2_score,
    macro_f2,
    macro_mrr5,
    mrr5_score,
    sweep_k,
)


def test_perfect_retrieval_scores_f2_one() -> None:
    assert f2_score(["a"], ["a"]) == pytest.approx(1.0)


def test_no_overlap_scores_f2_zero() -> None:
    assert f2_score(["b"], ["a"]) == pytest.approx(0.0)


def test_empty_prediction_scores_f2_zero() -> None:
    assert f2_score([], ["a"]) == pytest.approx(0.0)


def test_f2_weights_recall_four_times_precision() -> None:
    """1 gold, dự đoán 10 bảng có chứa gold: P=0.1, R=1.0
    F2 = 5*0.1*1.0 / (4*0.1 + 1.0) = 0.5/1.4 = 0.357"""
    predicted = ["a"] + [f"x{i}" for i in range(9)]
    assert f2_score(predicted, ["a"]) == pytest.approx(0.5 / 1.4, abs=1e-6)


def test_f2_duplicates_do_not_inflate_score() -> None:
    assert f2_score(["a", "a"], ["a"]) == pytest.approx(1.0)


def test_mrr5_scores_one_when_gold_is_first() -> None:
    assert mrr5_score(["a", "x", "y"], ["a"]) == pytest.approx(1.0)


def test_mrr5_scores_by_reciprocal_rank() -> None:
    """Gold ở vị trí 3 (1-indexed) -> 1/3."""
    assert mrr5_score(["x", "y", "a"], ["a"]) == pytest.approx(1 / 3)


def test_mrr5_ignores_hits_beyond_top_five() -> None:
    predicted = ["x1", "x2", "x3", "x4", "x5", "a"]
    assert mrr5_score(predicted, ["a"]) == pytest.approx(0.0)


def test_mrr5_uses_best_rank_when_multiple_gold_present() -> None:
    predicted = ["x", "a", "b"]  # a ở rank 2, b không xuất hiện
    assert mrr5_score(predicted, ["a", "b"]) == pytest.approx(0.5)


def test_mrr5_order_sensitive_unlike_f2() -> None:
    """Cùng TẬP bảng, khác THỨ TỰ -- F2 phải bằng nhau, MRR5 phải khác nhau.
    Đây là bất biến mà bug 'dùng build_cell_frame làm nguồn thứ tự' (Task 6)
    sẽ vi phạm mà không bị F2 phát hiện."""
    gold = ["a"]
    first_ranked = ["a", "b", "c"]
    last_ranked = ["c", "b", "a"]
    assert f2_score(first_ranked, gold) == pytest.approx(f2_score(last_ranked, gold))
    assert mrr5_score(first_ranked, gold) != pytest.approx(mrr5_score(last_ranked, gold))


def test_macro_f2_averages_across_questions() -> None:
    predictions = {1: ["a"], 2: ["b"]}
    gold = {1: ["a"], 2: ["x"]}
    assert macro_f2(predictions, gold) == pytest.approx(0.5)


def test_macro_mrr5_averages_across_questions() -> None:
    predictions = {1: ["a", "x"], 2: ["x", "b"]}
    gold = {1: ["a"], 2: ["b"]}
    assert macro_mrr5(predictions, gold) == pytest.approx((1.0 + 0.5) / 2)


def test_sweep_k_returns_both_metrics_per_k() -> None:
    ranked = {1: ["a", "b", "c", "d", "e", "f"]}
    gold = {1: ["a"]}
    result = sweep_k(ranked, gold, ks=(1, 5))
    assert set(result[1]) == {"f2", "mrr5"}
    assert result[1]["f2"] == pytest.approx(1.0)
    assert result[1]["mrr5"] == pytest.approx(1.0)
