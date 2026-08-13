"""Day 12 expansion-grid evaluation contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pytest

from financial_report_qa.planning.entity_contracts import QueryEntities
from financial_report_qa.retrieval.contracts import (
    GoldRetrievalQuestion,
    RetrievalFilters,
    TableMetadata,
)
from financial_report_qa.retrieval.evaluation import RetrievalEvaluationReport, RetrievalMetrics
from financial_report_qa.retrieval.expansion_contracts import (
    ExpansionParams,
    ExpansionTrace,
    RerankedCandidate,
    RerankFeatures,
)
from financial_report_qa.retrieval.expansion_evaluation import (
    deterministic_projection,
    evaluate_expansion_grid,
)
from financial_report_qa.retrieval.graph_service import TableGraphService
from financial_report_qa.retrieval.service import RetrievalService

_FINGERPRINT = "422df141c935d46bfd14302abec50f32380e6e4c012159f8ad0ae5560c8a446a"
_REFERENCE = RetrievalMetrics(
    true_positive=105,
    precision=0.1499999999999999,
    recall=0.880952380952381,
    f2=0.4224545295973871,
)


def _table_id(character: str) -> str:
    return f"tbl_{character * 64}"


@dataclass
class _Base:
    result_ids: tuple[str, ...]

    def retrieve(
        self, query: str, *, filters: RetrievalFilters, k: int, question_id: str
    ) -> ExpansionTrace:
        return ExpansionTrace(
            query=query,
            params=ExpansionParams(),
            entities=QueryEntities(question=query),
            eligible_count=len(self.result_ids),
            seed_count=len(self.result_ids),
            expanded_count=len(self.result_ids),
            dropped_out_of_filter=0,
            dropped_contradicted=0,
            results=tuple(
                RerankedCandidate(
                    table_id=value,
                    rank=index,
                    score=1 / index,
                    source="seed",
                    features=RerankFeatures(
                        seed_rrf=1 / index, neighbor_rrf=0, supporting_seed_count=0
                    ),
                    contradiction_count=0,
                    metadata=TableMetadata(
                        table_id=value, doc_id="doc", source_path="a.txt", line_start=1, line_end=1
                    ),
                    snippet="test",
                )
                for index, value in enumerate(self.result_ids[:k], start=1)
            ),
        )


def _patch_service(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "financial_report_qa.retrieval.expansion_evaluation.GraphExpansionService",
        lambda base, graph_service, params: base,
    )


def _question(
    *, question_id: str, gold: tuple[str, ...], periods: tuple[str, ...] = ()
) -> GoldRetrievalQuestion:
    return GoldRetrievalQuestion.model_validate(
        {
            "question_id": question_id,
            "question": "doanh thu",
            "intent": "lookup",
            "filters": {"periods": list(periods)},
            "gold_table_ids": list(gold),
            "reviewed_by": "reviewer",
            "reviewed_at": "2026-08-08T00:00:00+00:00",
            "gold_evidence": [
                {
                    "table_id": table_id,
                    "relative_path": "VCB/a.txt",
                    "line_start": 1,
                    "line_end": 2,
                    "verified": True,
                }
                for table_id in gold
            ],
            "dataset_fingerprint": _FINGERPRINT,
        }
    )


def _bm25_report() -> RetrievalEvaluationReport:
    return RetrievalEvaluationReport(
        dataset_fingerprint=_FINGERPRINT,
        question_count=70,
        macro=_REFERENCE,
        by_intent={"lookup": _REFERENCE},
        per_question=(),
    )


def test_grid_point_with_alpha_zero_matches_the_base_predictions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The alpha=0 anchor must preserve the base ranking exactly."""
    monkeypatch.setattr(
        "financial_report_qa.retrieval.expansion_evaluation._validate_bm25_reference",
        lambda report: None,
    )
    _patch_service(monkeypatch)
    question = _question(question_id="retq_" + "1" * 64, gold=(_table_id("a"),))
    params = ExpansionParams(
        relations=("same_document",),
        alpha=0,
        fan_out=1,
        seed_depth=50,
        rrf_k=60,
        expand_non_seeds=False,
    )
    report = evaluate_expansion_grid(
        cast(RetrievalService, _Base((_table_id("b"), _table_id("a")))),
        cast(TableGraphService, object()),
        (question,),
        _bm25_report(),
        grid=(params,),
    )
    assert report.grid[0].per_question[0].predicted_table_ids == (_table_id("b"), _table_id("a"))


def test_report_breaks_down_by_gold_and_period_cardinality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "financial_report_qa.retrieval.expansion_evaluation._validate_bm25_reference",
        lambda report: None,
    )
    _patch_service(monkeypatch)
    questions = (
        _question(question_id="retq_" + "1" * 64, gold=(_table_id("a"),), periods=("2023",)),
        _question(
            question_id="retq_" + "2" * 64,
            gold=(_table_id("b"), _table_id("c")),
            periods=("2022", "2023"),
        ),
    )
    params = ExpansionParams(
        relations=("same_document",),
        alpha=0,
        fan_out=1,
        seed_depth=50,
        rrf_k=60,
        expand_non_seeds=False,
    )
    report = evaluate_expansion_grid(
        cast(RetrievalService, _Base((_table_id("a"), _table_id("b"), _table_id("c")))),
        cast(TableGraphService, object()),
        questions,
        _bm25_report(),
        grid=(params,),
    )
    point = report.grid[0]
    assert set(point.by_gold_cardinality) == {"one_table", "multiple_tables"}
    assert set(point.by_period_cardinality) == {"one_period", "multiple_periods"}


def test_report_records_caveat_without_a_default_system(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "financial_report_qa.retrieval.expansion_evaluation._validate_bm25_reference",
        lambda report: None,
    )
    _patch_service(monkeypatch)
    params = ExpansionParams(
        relations=("same_document",),
        alpha=0,
        fan_out=1,
        seed_depth=50,
        rrf_k=60,
        expand_non_seeds=False,
    )
    report = evaluate_expansion_grid(
        cast(RetrievalService, _Base((_table_id("a"),))),
        cast(TableGraphService, object()),
        (_question(question_id="retq_" + "3" * 64, gold=(_table_id("a"),)),),
        _bm25_report(),
        grid=(params,),
    )
    dumped = report.model_dump()
    assert "default_system" not in dumped
    assert "70-question" in report.evidence_caveat
    assert "Day 14" in report.decision_reason


def test_deterministic_projection_scrubs_latency() -> None:
    from financial_report_qa.retrieval.expansion_evaluation import ExpansionGridReport

    report = ExpansionGridReport.model_construct(
        dataset_fingerprint=_FINGERPRINT,
        question_count=0,
        bm25_reference=_REFERENCE,
        grid=(),
        best_params=ExpansionParams(
            relations=("same_document",),
            alpha=0,
            fan_out=1,
            seed_depth=50,
            rrf_k=60,
            expand_non_seeds=False,
        ),
        decision_reason="Day 14 decides.",
        evidence_caveat="11/70",
    )
    assert "latency" not in deterministic_projection(report)
