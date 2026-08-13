from pathlib import Path

import pytest
from pydantic import ValidationError

from financial_report_qa.retrieval.contracts import (
    GoldRetrievalQuestion,
    RetrievalCandidate,
    RetrievalFilters,
    RetrievalTrace,
    TableMetadata,
)
from financial_report_qa.retrieval.evaluation import (
    RetrievalEvaluationReportV2,
    evaluate_retrieval_v2,
)
from financial_report_qa.retrieval.failure_evaluation import (
    FailureRootCauseAnnotation,
    RetrievalFailureCase,
    build_failure_report,
    write_failure_report,
)


def _table_id(character: str) -> str:
    return f"tbl_{character * 64}"


def _question_id(character: str) -> str:
    return f"retq_{character * 64}"


def _question(character: str, gold: tuple[str, ...]) -> GoldRetrievalQuestion:
    return GoldRetrievalQuestion.model_validate(
        {
            "question_id": _question_id(character),
            "question": f"Cau hoi {character}",
            "intent": "lookup",
            "filters": {"company_codes": ["VCB"], "periods": ["2023"]},
            "gold_table_ids": list(gold),
            "reviewed_by": "reviewer",
            "reviewed_at": "2026-08-14T00:00:00+07:00",
            "gold_evidence": [
                {
                    "table_id": table_id,
                    "relative_path": "VCB/report.txt",
                    "line_start": rank,
                    "line_end": rank,
                    "verified": True,
                }
                for rank, table_id in enumerate(gold, start=1)
            ],
            "dataset_fingerprint": "4" * 64,
        }
    )


class _FailureRetriever:
    def retrieve(
        self, query: str, *, filters: RetrievalFilters, k: int, question_id: str
    ) -> RetrievalTrace:
        question_character = question_id.removeprefix("retq_")[0]
        table_ids: tuple[str, ...]
        if question_character == "1":
            table_ids = (_table_id("a"),)
        elif question_character == "2":
            table_ids = tuple(_table_id(character) for character in "bdef012345" + "c")
        else:
            table_ids = (_table_id("e"),)
        return RetrievalTrace(
            question_id=question_id,
            query=query,
            query_tokens=("doanh", "thu"),
            eligible_count=20,
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
                        statement_type="income_statement",
                        title="Bao cao ket qua kinh doanh",
                        source_path="VCB/report.txt",
                        line_start=rank,
                        line_end=rank,
                    ),
                    snippet="Doanh thu",
                    matched_tokens=("doanh", "thu"),
                )
                for rank, table_id in enumerate(table_ids[:k], start=1)
            ),
        )


def _evaluation(*, diagnostic_k: int = 100) -> RetrievalEvaluationReportV2:
    questions = (
        _question("1", (_table_id("a"),)),
        _question("2", (_table_id("b"), _table_id("c"))),
        _question("3", (_table_id("d"),)),
    )
    return evaluate_retrieval_v2(
        _FailureRetriever(), questions, diagnostic_k=diagnostic_k
    )


def _annotations() -> tuple[FailureRootCauseAnnotation, ...]:
    return (
        FailureRootCauseAnnotation(
            question_id=_question_id("2"),
            root_cause="ranking_only",
            note="Gold c is eligible and appears at diagnostic rank 11.",
        ),
        FailureRootCauseAnnotation(
            question_id=_question_id("3"),
            root_cause="unknown",
            note="Gold d is absent through rank 100; source inspection is inconclusive.",
        ),
    )


def test_build_failure_report_preserves_top_10_trace_and_diagnostic_ranks() -> None:
    """Dropping trace fields, successes filtering, or rank/null evidence must fail."""
    report = build_failure_report(_evaluation(), _annotations())

    assert report.evaluated_question_count == 3
    assert report.failure_count == 2
    assert [item.question_id for item in report.failures] == [
        _question_id("2"),
        _question_id("3"),
    ]
    partial = report.failures[0]
    assert partial.failure == "partial_gold_hits"
    assert partial.gold_table_ids == (_table_id("b"), _table_id("c"))
    assert partial.missing_gold_table_ids == (_table_id("c"),)
    assert partial.gold_rank_beyond_10 == {_table_id("c"): 11}
    assert partial.predicted_table_ids == tuple(
        candidate.table_id for candidate in partial.trace.results
    )
    assert len(partial.trace.results) == 10
    assert partial.trace.results[0].score == pytest.approx(1)
    assert partial.trace.results[0].matched_tokens == ("doanh", "thu")
    assert partial.trace.results[0].metadata.statement_type == "income_statement"
    assert report.failures[1].gold_rank_beyond_10 == {_table_id("d"): None}
    assert report.failure_counts == {
        "no_eligible_documents": 0,
        "no_index_tokens": 0,
        "zero_gold_hits": 1,
        "partial_gold_hits": 1,
    }
    assert report.root_cause_counts["ranking_only"] == 1
    assert report.root_cause_counts["unknown"] == 1


@pytest.mark.parametrize("annotations", [(), (_annotations()[0],)])
def test_build_failure_report_rejects_missing_manual_labels(
    annotations: tuple[FailureRootCauseAnnotation, ...],
) -> None:
    """A failure without a manually supplied root cause cannot be exported."""
    with pytest.raises(ValueError, match="exactly match failure question IDs"):
        build_failure_report(_evaluation(), annotations)


def test_build_failure_report_rejects_a_shallow_diagnostic_evaluation() -> None:
    """A diagnostic_k=10 run cannot be exported under the required cutoff of 100."""
    shallow_evaluation = _evaluation(diagnostic_k=10)

    with pytest.raises(ValueError, match="requires diagnostic_k=100"):
        build_failure_report(shallow_evaluation, _annotations())


def test_unknown_root_cause_requires_an_explanatory_note() -> None:
    """An unexplained unknown would make the Day 14 taxonomy unauditable."""
    with pytest.raises(ValidationError):
        FailureRootCauseAnnotation(
            question_id=_question_id("3"), root_cause="unknown", note=""
        )


def test_failure_case_rejects_a_diagnostic_rank_outside_100() -> None:
    """A rank beyond diagnostic_k=100 must be represented as null, never as 101+."""
    case_payload = build_failure_report(_evaluation(), _annotations()).failures[0].model_dump(
        mode="json"
    )
    missing_id = case_payload["missing_gold_table_ids"][0]
    case_payload["gold_rank_beyond_10"] = {missing_id: 101}

    with pytest.raises(ValidationError, match="between 11 and 100"):
        RetrievalFailureCase.model_validate(case_payload)


def test_write_failure_report_is_byte_stable_and_auditable(tmp_path: Path) -> None:
    """Changing names, ordering, line endings, or human-readable evidence must fail."""
    report = build_failure_report(_evaluation(), _annotations())

    first_paths = write_failure_report(report, tmp_path / "first")
    second_paths = write_failure_report(report, tmp_path / "second")

    assert [path.name for path in first_paths] == [
        "failures-444444444444.json",
        "failures-444444444444.md",
    ]
    assert [path.read_bytes() for path in first_paths] == [
        path.read_bytes() for path in second_paths
    ]
    assert all(path.read_bytes().endswith(b"\n") for path in first_paths)
    markdown = first_paths[1].read_text(encoding="utf-8")
    assert "ranking_only" in markdown
    assert "diagnostic rank 11" in markdown
    assert "income_statement" in markdown
    assert "doanh, thu" in markdown
