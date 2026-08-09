from pathlib import Path

import pytest

from financial_report_qa.retrieval.contracts import (
    FilterDecision,
    GoldRetrievalQuestion,
    MetricExpansion,
    RetrievalCandidate,
    RetrievalFilters,
    RetrievalTrace,
    TableMetadata,
)
from financial_report_qa.retrieval.evaluation import (
    evaluate_retrieval,
    score_at_10,
    write_report,
)


def _table_id(character: str) -> str:
    return f"tbl_{character * 64}"


def _question_id(character: str) -> str:
    return f"retq_{character * 64}"


def _question() -> GoldRetrievalQuestion:
    table_id = _table_id("a")
    return GoldRetrievalQuestion.model_validate(
        {
            "question_id": _question_id("1"),
            "question": "Doanh thu cua VCB la bao nhieu?",
            "intent": "lookup",
            "filters": {"company_codes": ["VCB"]},
            "gold_table_ids": [table_id, _table_id("b")],
            "reviewed_by": "reviewer",
            "reviewed_at": "2026-08-08T00:00:00+00:00",
            "gold_evidence": [
                {
                    "table_id": table_id,
                    "relative_path": "VCB/report.txt",
                    "line_start": 1,
                    "line_end": 2,
                    "verified": True,
                },
                {
                    "table_id": _table_id("b"),
                    "relative_path": "VCB/report.txt",
                    "line_start": 3,
                    "line_end": 4,
                    "verified": True,
                },
            ],
            "dataset_fingerprint": "c" * 64,
        }
    )


class _RetrieverFixture:
    def retrieve(
        self, query: str, *, filters: RetrievalFilters, k: int, question_id: str
    ) -> RetrievalTrace:
        assert query == "Doanh thu cua VCB la bao nhieu?"
        assert filters.company_codes == ("VCB",)
        assert k == 10
        return RetrievalTrace(
            question_id=question_id,
            query=query,
            query_tokens=("doanh", "thu"),
            eligible_count=2,
            filter_decisions=(
                FilterDecision(
                    field="company_codes",
                    requested_values=("VCB",),
                    matched_count_before_intersection=2,
                    eligible_count_after_intersection=2,
                ),
            ),
            results=(
                RetrievalCandidate(
                    table_id=_table_id("a"),
                    score=1.25,
                    rank=1,
                    metadata=TableMetadata(
                        table_id=_table_id("a"),
                        doc_id="doc_a",
                        company_code="VCB",
                        periods=(),
                        source_path="VCB/report.txt",
                        line_start=1,
                        line_end=2,
                    ),
                    snippet="Doanh thu",
                    matched_tokens=("doanh", "thu"),
                ),
            ),
            metric_expansions=(
                MetricExpansion(
                    alias_tokens=("doanh", "thu"),
                    canonical_metric="net_revenue",
                    added_tokens=("net", "revenue"),
                ),
            ),
        )


def test_score_at_10_uses_fixed_precision_denominator() -> None:
    metrics = score_at_10(predicted=("a", "x"), gold=("a", "b"))

    assert metrics.true_positive == 1
    assert metrics.precision == pytest.approx(0.1)
    assert metrics.recall == pytest.approx(0.5)
    assert metrics.f2 == pytest.approx(5 * 0.1 * 0.5 / (4 * 0.1 + 0.5))


def test_score_at_10_rejects_duplicate_predictions() -> None:
    with pytest.raises(ValueError, match="unique"):
        score_at_10(predicted=("a", "a"), gold=("a",))


def test_evaluation_records_query_evidence_and_partial_hit_taxonomy() -> None:
    """Would fail if scores/tokens/filter counts or missing gold are dropped from reports."""
    report = evaluate_retrieval(_RetrieverFixture(), (_question(),))

    result = report.per_question[0]
    assert result.failure == "partial_gold_hits"
    assert result.predicted_table_ids == (_table_id("a"),)
    assert result.gold_table_ids == (_table_id("a"), _table_id("b"))
    assert result.missing_gold_table_ids == (_table_id("b"),)
    assert result.trace.results[0].score == pytest.approx(1.25)
    assert result.trace.results[0].matched_tokens == ("doanh", "thu")
    assert result.trace.filter_decisions[0].eligible_count_after_intersection == 2
    assert result.trace.metric_expansions[0].canonical_metric == "net_revenue"


def test_evaluate_retrieval_rejects_non_fixed_k() -> None:
    """Would fail if evaluation silently reports a non-@10 metric."""
    with pytest.raises(ValueError, match="fixed at 10"):
        evaluate_retrieval(_RetrieverFixture(), (_question(),), k=5)


def test_write_report_uses_day8_names_and_terminal_newlines(tmp_path: Path) -> None:
    """Would fail if artifacts are not the specified replayable Day 8 report pair."""
    report = evaluate_retrieval(_RetrieverFixture(), (_question(),))

    json_path, markdown_path = write_report(report, tmp_path)

    assert json_path.name == "retrieval-day8-cccccccccccc.json"
    assert markdown_path.name == "retrieval-day8-cccccccccccc.md"
    assert json_path.read_bytes().endswith(b"\n")
    assert markdown_path.read_bytes().endswith(b"\n")
