"""Payload batch v2: LLM quyết cả operation lẫn dòng (spec 2026-08-23 §6.2)."""

from __future__ import annotations

from financial_report_qa.planning.entity_parser import parse_query_entities
from financial_report_qa.planning.row_choice_batch import build_batch_payload
from financial_report_qa.retrieval.row_documents import RowMetadata
from financial_report_qa.retrieval.row_fusion_contracts import RowFusedCandidate

TABLE_A = "tbl_" + "a" * 64


def _candidate(rank: int, *, label: str = "Doanh thu thuần", company: str = "ACB"):
    return RowFusedCandidate(
        row_id=f"{TABLE_A}|row_{rank}",
        table_id=TABLE_A,
        row_idx=rank,
        rank=rank,
        fused_score=1.0 / rank,
        metadata=RowMetadata(
            table_id=TABLE_A,
            row_idx=rank,
            company_code=company,
            row_label_raw=label,
            title="Báo cáo KQKD",
            periods=("2023",),
        ),
        snippet=label,
    )


def test_payload_carries_companies_and_periods_for_operation_choice() -> None:
    entities = parse_query_entities("Tra cứu doanh thu thuần của ACB năm 2023.")
    payload = build_batch_payload(7, entities.question, entities, (_candidate(1),))
    assert payload["question_id"] == 7
    assert payload["companies"] == ["ACB"]
    assert payload["periods"] == ["2023"]


def test_candidate_carries_company_code_so_multi_company_picks_are_possible() -> None:
    entities = parse_query_entities("Tra cứu doanh thu thuần của ACB năm 2023.")
    payload = build_batch_payload(7, entities.question, entities, (_candidate(1, company="VIC"),))
    candidates = payload["candidates"]
    assert isinstance(candidates, list)
    assert candidates[0]["company_code"] == "VIC"
    assert candidates[0]["index"] == 0


def test_payload_never_leaks_a_cell_value_or_score() -> None:
    """Bất biến N7 -- đây là thứ giữ cho cam kết chống hardcode còn đứng vững."""
    entities = parse_query_entities("Tra cứu doanh thu thuần của ACB năm 2023.")
    payload = build_batch_payload(7, entities.question, entities, (_candidate(1),))
    blob = repr(payload)
    assert "fused_score" not in blob
    assert "value" not in blob
    for candidate in payload["candidates"]:
        assert "value" not in candidate
        assert "fused_score" not in candidate
