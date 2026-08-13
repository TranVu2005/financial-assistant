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
    evaluate_retrieval_v2,
    score_at_10,
    score_extended_at_10,
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


def test_extended_score_uses_only_top_10_and_r_for_their_named_metrics() -> None:
    """A hit below rank 10 must not leak into any metric, including F2@R."""
    predicted = (
        "x1",
        "gold-a",
        "x3",
        "x4",
        "gold-b",
        "x6",
        "x7",
        "x8",
        "x9",
        "x10",
        "gold-c",
    )

    metrics = score_extended_at_10(predicted, ("gold-a", "gold-b", "gold-c"))

    assert metrics.true_positive == 2
    assert metrics.precision_at_10 == pytest.approx(0.2)
    assert metrics.recall_at_3 == pytest.approx(1 / 3)
    assert metrics.recall_at_5 == pytest.approx(2 / 3)
    assert metrics.recall_at_10 == pytest.approx(2 / 3)
    assert metrics.f2_at_10 == pytest.approx(5 / 11)
    assert metrics.mrr == pytest.approx(0.5)
    assert metrics.precision_at_r == pytest.approx(1 / 3)
    assert metrics.f2_at_r == pytest.approx(5 / 9)


class _DiagnosticRetriever:
    def __init__(self, table_ids: tuple[str, ...]) -> None:
        self.table_ids = table_ids
        self.requested_k: list[int] = []

    def retrieve(
        self, query: str, *, filters: RetrievalFilters, k: int, question_id: str
    ) -> RetrievalTrace:
        self.requested_k.append(k)
        return RetrievalTrace(
            question_id=question_id,
            query=query,
            query_tokens=("doanh", "thu"),
            eligible_count=len(self.table_ids),
            filter_decisions=(),
            results=tuple(
                RetrievalCandidate(
                    table_id=table_id,
                    score=1 / rank,
                    rank=rank,
                    metadata=TableMetadata(
                        table_id=table_id,
                        doc_id=f"doc-{rank}",
                        company_code="VCB",
                        periods=("2023",),
                        source_path="VCB/report.txt",
                        line_start=rank,
                        line_end=rank,
                    ),
                    snippet="Doanh thu",
                )
                for rank, table_id in enumerate(self.table_ids[:k], start=1)
            ),
        )


class _KSensitiveRetriever(_DiagnosticRetriever):
    """Expose clients that incorrectly assume rankings are prefix-stable across k."""

    def retrieve(
        self, query: str, *, filters: RetrievalFilters, k: int, question_id: str
    ) -> RetrievalTrace:
        if k == 10:
            self.table_ids = (_table_id("b"),)
        else:
            self.table_ids = (_table_id("a"),)
        return super().retrieve(
            query, filters=filters, k=k, question_id=question_id
        )


def _question_with_dimensions(
    character: str,
    *,
    gold: tuple[str, ...],
    periods: tuple[str, ...],
    statement_types: tuple[str, ...] = (),
) -> GoldRetrievalQuestion:
    return GoldRetrievalQuestion.model_validate(
        {
            "question_id": _question_id(character),
            "question": f"Cau hoi {character}",
            "intent": "lookup",
            "filters": {
                "company_codes": ["VCB"],
                "periods": list(periods),
                "statement_types": list(statement_types),
            },
            "gold_table_ids": list(gold),
            "reviewed_by": "reviewer",
            "reviewed_at": "2026-08-08T00:00:00+00:00",
            "gold_evidence": [
                {
                    "table_id": table_id,
                    "relative_path": "VCB/report.txt",
                    "line_start": index,
                    "line_end": index,
                    "verified": True,
                }
                for index, table_id in enumerate(gold, start=1)
            ],
            "dataset_fingerprint": "c" * 64,
        }
    )


def test_evaluate_retrieval_v2_keeps_diagnostics_out_of_metrics() -> None:
    """Changing diagnostic depth may reveal a rank but cannot change top-10 metrics."""
    gold = _table_id("b")
    ranked = tuple(_table_id(character) for character in "acdef01234" + "b")
    retriever = _DiagnosticRetriever(ranked)
    question = _question_with_dimensions("2", gold=(gold,), periods=("2023",))

    report = evaluate_retrieval_v2(retriever, (question,), diagnostic_k=11)

    result = report.per_question[0]
    assert retriever.requested_k == [10, 11]
    assert result.metrics.recall_at_10 == 0
    assert result.metrics.f2_at_r == 0
    assert result.gold_rank_beyond_10 == {gold: 11}
    assert len(result.trace.results) == 10
    assert result.predicted_table_ids == ranked[:10]


def test_evaluate_retrieval_v2_defaults_to_100_and_records_unranked_gold() -> None:
    """Changing the diagnostic default or dropping null misses must fail this test."""
    retriever = _DiagnosticRetriever((_table_id("a"),))
    gold = _table_id("b")
    question = _question_with_dimensions("6", gold=(gold,), periods=("2023",))

    report = evaluate_retrieval_v2(retriever, (question,))

    assert retriever.requested_k == [10, 100]
    assert report.per_question[0].gold_rank_beyond_10 == {gold: None}


def test_evaluate_retrieval_v2_scores_an_independent_k_10_retrieval() -> None:
    """A k-sensitive retriever must not let diagnostic depth change metric inputs."""
    retriever = _KSensitiveRetriever(())
    gold = _table_id("b")
    question = _question_with_dimensions("7", gold=(gold,), periods=("2023",))

    report = evaluate_retrieval_v2(retriever, (question,))

    assert retriever.requested_k == [10]
    assert report.macro.recall_at_10 == 1
    assert report.macro.mrr == 1


def test_evaluate_retrieval_v2_reports_all_required_breakdown_labels() -> None:
    """Wrong cardinality/filter/era classification must change the report keys."""
    tables = tuple(_table_id(character) for character in "abcdef")
    questions = (
        _question_with_dimensions("3", gold=(tables[0],), periods=("2019",)),
        _question_with_dimensions(
            "4",
            gold=(tables[1], tables[2]),
            periods=("2022", "2023"),
            statement_types=("income_statement",),
        ),
        _question_with_dimensions(
            "5", gold=(tables[3], tables[4], tables[5]), periods=("2024", "2025")
        ),
    )

    report = evaluate_retrieval_v2(_DiagnosticRetriever(tables), questions, diagnostic_k=10)

    assert set(report.by_gold_cardinality) == {"one_table", "two_tables", "three_or_more"}
    assert set(report.by_period_cardinality) == {"one_period", "multiple_periods"}
    assert set(report.by_statement_filter) == {"filtered", "unfiltered"}
    assert set(report.by_report_era) == {"2015_2019", "2020_2023", "2024_2025"}
    assert report.macro.true_positive == 6
    assert report.macro.recall_at_3 == pytest.approx(2 / 3)
    assert report.macro.mrr == pytest.approx(7 / 12)
    assert report.macro.f2_at_r == pytest.approx(11 / 18)
    assert report.by_statement_filter["filtered"].f2_at_r == pytest.approx(5 / 6)


@pytest.mark.parametrize("diagnostic_k", [0, 9])
def test_evaluate_retrieval_v2_rejects_diagnostics_shallower_than_metrics(
    diagnostic_k: int,
) -> None:
    """A diagnostic cutoff below ten cannot support fixed top-10 scoring."""
    with pytest.raises(ValueError, match="at least 10"):
        evaluate_retrieval_v2(
            _DiagnosticRetriever((_table_id("a"),)), (_question(),), diagnostic_k=diagnostic_k
        )


def test_evaluate_retrieval_v2_rejects_non_fixed_metric_cutoff() -> None:
    """Allowing a configurable metric cutoff would make report labels misleading."""
    with pytest.raises(ValueError, match="fixed at 10"):
        evaluate_retrieval_v2(
            _DiagnosticRetriever((_table_id("a"),)), (_question(),), k=5
        )
