"""Nhánh 2 là một đường thẳng (spec 2026-08-23 §6, nguyên tắc N6).

Không thang tầng, không candidate switching, không context expansion. Một
câu hỏi đi qua đúng một chuỗi bước; hỏng ở đâu thì hỏng rõ ở đó.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from tests.unit.conftest import DOC_ID, TABLE_ID

from financial_report_qa.data.dataset_builder import (
    CELL_SCHEMA,
    DOCUMENT_SCHEMA,
    PLACEMENT_SCHEMA,
    TABLE_SCHEMA,
)
from financial_report_qa.planning.cell_grounding import ground_question
from financial_report_qa.planning.entity_parser import parse_query_entities
from financial_report_qa.planning.question_plan import RowChoiceDecision
from financial_report_qa.retrieval.row_documents import RowMetadata
from financial_report_qa.retrieval.row_fusion_contracts import RowFusedCandidate


def test_ground_question_reports_llm_decision_as_the_only_plan_source(
    release_dir: Path, execution_settings, fusion_rows, table_ids
) -> None:
    entities = parse_query_entities("Tra cứu doanh thu thuần của ACB năm 2023.")
    result = ground_question(
        entities=entities,
        decision=RowChoiceDecision(question_id=1, operation="lookup", chosen=(0,)),
        fusion_rows=fusion_rows,
        candidate_table_ids=table_ids,
        release_dir=release_dir,
        execution_settings=execution_settings,
    )
    assert result.plan_source == "llm_decision"


def test_ground_question_fails_cleanly_when_no_plan_can_be_assembled(
    release_dir: Path, execution_settings, table_ids
) -> None:
    """Không ứng viên -> thất bại có mã, không exception, không tầng thứ hai."""
    entities = parse_query_entities("Tra cứu doanh thu thuần của ACB năm 2023.")
    result = ground_question(
        entities=entities,
        decision=None,
        fusion_rows=(),
        candidate_table_ids=table_ids,
        release_dir=release_dir,
        execution_settings=execution_settings,
    )
    assert result.status == "failed"
    assert result.error_code == "no_row_candidates"


def test_cell_grounding_has_no_recovery_ladder_left() -> None:
    """Ghim N6: các tầng đã bỏ không được lặng lẽ quay lại."""
    from financial_report_qa.planning import cell_grounding

    for gone in (
        "ground_with_recovery",
        "_candidate_switching",
        "_context_expansion",
        "choose_row_label",
    ):
        assert not hasattr(cell_grounding, gone), f"{gone} thuộc thang tầng đã bỏ"


@pytest.fixture
def duplicate_label_release(tmp_path: Path) -> Path:
    """Một release có hai dòng **trùng nhãn, khác giá trị**.

    Đây chính là hình dạng mà nhánh row-choice sinh ra để giải quyết: đo được
    4.37% dòng trùng nhãn khác giá trị, do OCR lặp dòng. Nhãn không phân biệt
    được chúng, nên chỉ có `(table_id, row_idx)` mới nói được lấy dòng nào.
    """
    release_dir = tmp_path / "dup-release"
    release_dir.mkdir()
    documents = [
        {
            "doc_id": DOC_ID,
            "repo_id": "repo",
            "revision": "1",
            "relative_path": "ACB/2023/ACB_financial_statements_2023_consolidated_extracted.txt",
            "company_code": "ACB",
            "report_year": 2023,
            "statement_scope": "consolidated",
            "sha256": "0" * 64,
            "file_size_bytes": 10,
            "encoding": "utf-8",
            "inventory_status": "ready",
            "ruleset_version": "1",
            "normalization_fingerprint": "0" * 64,
        }
    ]
    tables = [
        {
            "table_id": TABLE_ID,
            "doc_id": DOC_ID,
            "source_ordinal": 0,
            "title_raw": "Bao cao ket qua kinh doanh",
            "statement_type": "income_statement",
            "unit_raw": "VND",
            "unit_normalized": "vnd",
            "line_start": 1,
            "line_end": 10,
            "row_count": 2,
            "column_count": 2,
            "quality_score": 0.9,
            "csv_path": None,
        }
    ]
    cells = [
        {
            "cell_id": "cell_" + letter * 64,
            "table_id": TABLE_ID,
            "row_idx": row_idx,
            "col_idx": 1,
            "row_label_raw": "Doanh thu thuan",
            "row_label_canonical": None,
            "row_group_context_raw": None,
            "column_label_raw": "Năm 2023",
            "column_label_canonical": None,
            "value_raw": str(value),
            "value_numeric": Decimal(value),
            "period": "2023",
            "unit": "VND",
            "source_line_start": 5 + row_idx,
            "source_line_end": 5 + row_idx,
            "extraction_confidence": 0.9,
        }
        for letter, row_idx, value in (("a", 0, "100"), ("d", 2, "250"))
    ]
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.Table.from_pylist(documents, schema=DOCUMENT_SCHEMA), release_dir / "documents.parquet"
    )
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.Table.from_pylist(tables, schema=TABLE_SCHEMA), release_dir / "tables.parquet"
    )
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.Table.from_pylist(cells, schema=CELL_SCHEMA), release_dir / "cells.parquet"
    )
    placements = [
        {
            "table_id": TABLE_ID,
            "row_idx": cell["row_idx"],
            "col_idx": cell["col_idx"],
            "cell_id": cell["cell_id"],
        }
        for cell in cells
    ]
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.Table.from_pylist(placements, schema=PLACEMENT_SCHEMA),
        release_dir / "placements.parquet",
    )
    return release_dir


def _dup_candidate(rank: int, row_idx: int) -> RowFusedCandidate:
    return RowFusedCandidate(
        row_id=f"{TABLE_ID}|row_{row_idx}",
        table_id=TABLE_ID,
        row_idx=row_idx,
        rank=rank,
        snippet="Doanh thu thuan | Năm 2023",
        metadata=RowMetadata(
            table_id=TABLE_ID,
            row_idx=row_idx,
            company_code="ACB",
            row_label_raw="Doanh thu thuan",
            periods=("2023",),
        ),
        fused_score=1.0 / rank,
        bm25_score=1.0 / rank,
        dense_score=0.0,
    )


def test_llm_row_choice_survives_compilation_when_labels_collide(
    duplicate_label_release: Path, execution_settings, table_ids
) -> None:
    """Dòng LLM chọn phải là dòng được trả lời, kể cả khi hạng 1 trùng nhãn.

    `compile_grounded` gọi `bind_plan_to_rows` trước, và hàm đó tự chọn lại
    dòng bằng `nhãn khớp + min(rank)`. Với plan đã position-bound từ quyết
    định của LLM, việc chọn lại chỉ có thể *huỷ* lựa chọn đó -- và nó huỷ
    đúng trong trường hợp duy nhất mà lựa chọn ấy có giá trị: hai dòng trùng
    nhãn. Ở đây LLM chọn dòng 2 (=250), hạng 1 là dòng 0 (=100).
    """
    entities = parse_query_entities("Tra cứu doanh thu thuần của ACB năm 2023.")
    result = ground_question(
        entities=entities,
        decision=RowChoiceDecision(question_id=1, operation="lookup", chosen=(1,)),
        fusion_rows=(_dup_candidate(1, 0), _dup_candidate(2, 2)),
        candidate_table_ids=table_ids,
        release_dir=duplicate_label_release,
        execution_settings=execution_settings,
    )
    assert result.status == "accepted", result.error_code
    assert result.compiled is not None
    assert result.compiled.answer == Decimal("250")
