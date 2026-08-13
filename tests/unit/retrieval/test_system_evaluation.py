from __future__ import annotations

from pathlib import Path

from financial_report_qa.retrieval.contracts import (
    GoldRetrievalQuestion,
    RetrievalCandidate,
    RetrievalFilters,
    RetrievalTrace,
    TableMetadata,
)
from financial_report_qa.retrieval.evaluation import evaluate_retrieval
from financial_report_qa.retrieval.system_evaluation import (
    RetrievalSystemReportV2,
    derive_system_report_v2,
    write_system_report_v2,
)


def _table_id(character: str) -> str:
    return f"tbl_{character * 64}"


def _question() -> GoldRetrievalQuestion:
    return GoldRetrievalQuestion.model_validate(
        {
            "question_id": "retq_" + "1" * 64,
            "question": "Doanh thu VCB 2023?",
            "intent": "lookup",
            "filters": {"company_codes": ["VCB"], "periods": ["2023"]},
            "gold_table_ids": [_table_id("a"), _table_id("b")],
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
                for rank, table_id in enumerate((_table_id("a"), _table_id("b")), start=1)
            ],
            "dataset_fingerprint": "c" * 64,
        }
    )


class _RetrieverFixture:
    def retrieve(
        self, query: str, *, filters: RetrievalFilters, k: int, question_id: str
    ) -> RetrievalTrace:
        table_id = _table_id("a")
        return RetrievalTrace(
            question_id=question_id,
            query=query,
            query_tokens=("doanh", "thu"),
            eligible_count=1,
            filter_decisions=(),
            results=(
                RetrievalCandidate(
                    table_id=table_id,
                    score=1,
                    rank=1,
                    metadata=TableMetadata(
                        table_id=table_id,
                        doc_id="doc-a",
                        company_code="VCB",
                        periods=("2023",),
                        source_path="VCB/report.txt",
                        line_start=1,
                        line_end=1,
                    ),
                    snippet="Doanh thu",
                ),
            ),
        )


def test_derive_system_report_v2_persists_all_breakdowns_and_question_evidence(
    tmp_path: Path,
) -> None:
    question = _question()
    legacy = evaluate_retrieval(_RetrieverFixture(), (question,))
    source = tmp_path / "legacy.json"
    source.write_text(legacy.model_dump_json(), encoding="utf-8")

    report = derive_system_report_v2(
        system_name="fixture-bm25",
        source_path=source,
        source_kind="legacy",
        questions=(question,),
    )

    assert report.question_count == len(report.per_question) == 1
    assert set(report.by_intent) == {"lookup"}
    assert set(report.by_gold_cardinality) == {"two_tables"}
    assert set(report.by_period_cardinality) == {"one_period"}
    assert set(report.by_statement_filter) == {"unfiltered"}
    assert set(report.by_report_era) == {"2020_2023"}
    assert report.source_sha256
    assert report.per_question[0].metrics.recall_at_10 == 0.5


def test_write_system_report_v2_round_trips_deterministically(tmp_path: Path) -> None:
    legacy = evaluate_retrieval(_RetrieverFixture(), (_question(),))
    source = tmp_path / "legacy.json"
    source.write_text(legacy.model_dump_json(), encoding="utf-8")
    report = derive_system_report_v2(
        system_name="fixture-bm25",
        source_path=source,
        source_kind="legacy",
        questions=(_question(),),
    )

    first = write_system_report_v2(report, tmp_path / "first")
    second = write_system_report_v2(report, tmp_path / "second")

    assert [path.read_bytes() for path in first] == [path.read_bytes() for path in second]
    parsed = RetrievalSystemReportV2.model_validate_json(first[0].read_bytes())
    assert parsed == report
