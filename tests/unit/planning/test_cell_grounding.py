import json
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock, patch

from financial_report_qa.core.config import ExecutionSettings
from financial_report_qa.planning.cell_grounding import ground_with_recovery
from financial_report_qa.planning.entity_contracts import QueryEntities
from financial_report_qa.planning.llm_client import LLMClient
from financial_report_qa.retrieval.row_documents import RowMetadata
from financial_report_qa.retrieval.row_fusion_contracts import RowFusedCandidate

TABLE_ID = "tbl_" + "1" * 64
DOC_ID = "doc_" + "a" * 64
CELL_ID = "cell_" + "a" * 64

_ALLOW_LOOKUP = ExecutionSettings(timeout_seconds=5, max_rows=20000, allow_operations=("lookup",))

def _write_release(tmp_path: Path) -> Path:
    release_dir = tmp_path / "release"
    release_dir.mkdir(exist_ok=True)
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
            "row_count": 1,
            "column_count": 2,
            "quality_score": 0.9,
            "csv_path": None,
        }
    ]
    cells = [
        {
            "cell_id": CELL_ID,
            "table_id": TABLE_ID,
            "row_idx": 0,
            "col_idx": 1,
            "row_label_raw": "Doanh thu thuan",
            "row_label_canonical": "net_revenue",
            "row_group_context_raw": None,
            "column_label_raw": "Năm 2023",
            "column_label_canonical": None,
            "value_raw": "100",
            "value_numeric": Decimal("100"),
            "period": "2023",
            "unit": "VND",
            "source_line_start": 5,
            "source_line_end": 5,
            "extraction_confidence": 0.9,
        }
    ]
    placements = [
        {
            "table_id": TABLE_ID,
            "row_idx": cell["row_idx"],
            "col_idx": cell["col_idx"],
            "cell_id": cell["cell_id"],
        }
        for cell in cells
    ]
    
    import pyarrow as pa
    import pyarrow.parquet as pq

    from financial_report_qa.data.dataset_builder import (
        CELL_SCHEMA,
        DOCUMENT_SCHEMA,
        PLACEMENT_SCHEMA,
        TABLE_SCHEMA,
    )
    
    pq.write_table(pa.Table.from_pylist(documents, schema=DOCUMENT_SCHEMA), release_dir / "documents.parquet")
    pq.write_table(pa.Table.from_pylist(tables, schema=TABLE_SCHEMA), release_dir / "tables.parquet")
    pq.write_table(pa.Table.from_pylist(cells, schema=CELL_SCHEMA), release_dir / "cells.parquet")
    pq.write_table(pa.Table.from_pylist(placements, schema=PLACEMENT_SCHEMA), release_dir / "placements.parquet")
    
    manifest = {
        "schema_version": "v1",
        "dataset_fingerprint": "0" * 64,
        "source_manifest_sha256": "0" * 64,
        "document_count": len(documents),
        "table_count": len(tables),
        "cell_count": len(cells),
    }
    (release_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return release_dir



def test_ground_with_recovery_success_attempt_0(tmp_path: Path) -> None:
    release_dir = _write_release(tmp_path)
    entities = QueryEntities(
        question="Tra cứu Doanh thu thuan của ACB năm 2023.",
        company_codes=("ACB",),
        periods=("2023",),
        metrics=(),
        metric_phrases=("Doanh thu thuan",),
        operation="lookup",
        spans=(),
    )

    # Attempt 0: Rule-based ground_raw_metric should find "Doanh thu thuan"
    res = ground_with_recovery(
        question="Tra cứu Doanh thu thuan của ACB năm 2023.",
        entities=entities,
        retrieved=(TABLE_ID,),
        row_labels=("Doanh thu thuan",),
        fusion_rows=(),
        release_dir=release_dir,
        execution_settings=_ALLOW_LOOKUP,
    )

    assert res.status == "accepted"
    assert res.plan_source == "rule_raw_grounded"
    assert res.plan is not None
    assert res.plan.metric.raw_text == "Doanh thu thuan"
    # No fusion rows given -- nothing to score confidence against.
    assert res.grounding_score is None


def test_ground_with_recovery_with_llm_cell_grounding_attempt_0(tmp_path: Path) -> None:
    release_dir = _write_release(tmp_path)
    entities = QueryEntities(
        question="Tra cứu chỉ số ACB năm 2023.",
        company_codes=("ACB",),
        periods=("2023",),
        metrics=(),
        metric_phrases=("chỉ số",),
        operation="lookup",
        spans=(),
    )

    # Mock choose_row_label to select "Doanh thu thuan"
    with patch("financial_report_qa.planning.cell_grounding.choose_row_label", return_value="Doanh thu thuan"):
        llm_client = MagicMock(spec=LLMClient)
        res = ground_with_recovery(
            question="Tra cứu chỉ số ACB năm 2023.",
            entities=entities,
            retrieved=(TABLE_ID,),
            row_labels=("Doanh thu thuan",),
            fusion_rows=(),
            release_dir=release_dir,
            execution_settings=_ALLOW_LOOKUP,
            llm_client=llm_client,
        )

        assert res.status == "accepted"
        assert res.plan_source == "llm_cell_grounded"
        assert res.plan is not None
        assert res.plan.metric.raw_text == "Doanh thu thuan"


def test_ground_with_recovery_candidate_switching_attempt_1(tmp_path: Path) -> None:
    release_dir = _write_release(tmp_path)
    entities = QueryEntities(
        question="Tra cứu chỉ số ACB năm 2023.",
        company_codes=("ACB",),
        periods=("2023",),
        metrics=(),
        metric_phrases=("chỉ số",),
        operation="lookup",
        spans=(),
    )

    fusion_rows = (
        RowFusedCandidate(
            row_id="1",
            table_id=TABLE_ID,
            row_idx=0,
            rank=1,
            snippet="",
            metadata=RowMetadata(
                table_id=TABLE_ID,
                row_idx=0,
                row_label_raw="Doanh thu ao",
                row_label_canonical="unknown",
            ),
            fused_score=0.9,
            bm25_score=1.0,
            dense_score=0.0,
        ),
        RowFusedCandidate(
            row_id="2",
            table_id=TABLE_ID,
            row_idx=0,
            rank=2,
            snippet="",
            metadata=RowMetadata(
                table_id=TABLE_ID,
                row_idx=0,
                row_label_raw="Doanh thu thuan",
                row_label_canonical="net_revenue",
            ),
            fused_score=0.8,
            bm25_score=0.8,
            dense_score=0.0,
        ),
    )

    # Attempt 0: LLM chooses "Doanh thu ao" (which doesn't exist in cell frame, returning metric_not_found)
    # Attempt 1: Candidate switching tries "Doanh thu thuan" (which succeeds!)
    with patch("financial_report_qa.planning.cell_grounding.choose_row_label", return_value="Doanh thu ao"):
        llm_client = MagicMock(spec=LLMClient)
        res = ground_with_recovery(
            question="Tra cứu chỉ số ACB năm 2023.",
            entities=entities,
            retrieved=(TABLE_ID,),
            row_labels=("Doanh thu ao", "Doanh thu thuan"),
            fusion_rows=fusion_rows,
            release_dir=release_dir,
            execution_settings=_ALLOW_LOOKUP,
            llm_client=llm_client,
        )

        assert res.status == "accepted"
        assert res.plan_source == "llm_cell_grounded_recovered"
        assert res.plan is not None
        assert res.plan.metric.raw_text == "Doanh thu thuan"
        assert res.recovery_attempts == 1
        # "Doanh thu thuan" candidate's own fused_score, not "Doanh thu ao"'s.
        assert res.grounding_score == 0.8


def test_ground_with_recovery_context_expansion_attempt_2(tmp_path: Path) -> None:
    release_dir = _write_release(tmp_path)
    entities = QueryEntities(
        question="Tra cứu chỉ số ACB năm 2023.",
        company_codes=("ACB",),
        periods=("2023",),
        metrics=(),
        metric_phrases=("chỉ số",),
        operation="lookup",
        spans=(),
    )

    fusion_rows = (
        RowFusedCandidate(
            row_id="1",
            table_id=TABLE_ID,
            row_idx=0,
            rank=1,
            snippet="Doanh thu ao snippet",
            metadata=RowMetadata(
                table_id=TABLE_ID,
                row_idx=0,
                row_label_raw="Doanh thu ao",
                row_label_canonical="unknown",
            ),
            fused_score=0.9,
            bm25_score=1.0,
            dense_score=0.0,
        ),
    )

    # Attempt 0: LLM chooses "Doanh thu ao" (which doesn't exist, returning metric_not_found)
    # Attempt 1: Candidate switching has no other candidates
    # Attempt 2: Context expansion runs and chooses "Doanh thu thuan" (which succeeds!)
    with patch("financial_report_qa.planning.cell_grounding.choose_row_label", return_value="Doanh thu ao"), \
         patch("financial_report_qa.planning.cell_grounding.choose_row_label_with_context", return_value="Doanh thu thuan"):
        llm_client = MagicMock(spec=LLMClient)
        res = ground_with_recovery(
            question="Tra cứu chỉ số ACB năm 2023.",
            entities=entities,
            retrieved=(TABLE_ID,),
            row_labels=("Doanh thu ao",),
            fusion_rows=fusion_rows,
            release_dir=release_dir,
            execution_settings=_ALLOW_LOOKUP,
            llm_client=llm_client,
        )

        assert res.status == "accepted"
        assert res.plan_source == "llm_cell_grounded_context_expanded"
        assert res.plan is not None
        assert res.plan.metric.raw_text == "Doanh thu thuan"
        assert res.recovery_attempts == 1
        # fusion_rows only scored "Doanh thu ao", never "Doanh thu thuan" --
        # the context-expanded pick has no fusion confidence to report.
        assert res.grounding_score is None


def _write_two_row_release(tmp_path: Path) -> Path:
    """Like `_write_release`, but with a second real, compilable row --
    needed to test the ladder preferring a higher-ranked *alternative*
    candidate over a low-ranked one that already compiled."""
    release_dir = tmp_path / "release_two_rows"
    release_dir.mkdir(exist_ok=True)
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
            "cell_id": CELL_ID,
            "table_id": TABLE_ID,
            "row_idx": 0,
            "col_idx": 1,
            "row_label_raw": "Doanh thu thuan",
            "row_label_canonical": "net_revenue",
            "row_group_context_raw": None,
            "column_label_raw": "Năm 2023",
            "column_label_canonical": None,
            "value_raw": "100",
            "value_numeric": Decimal("100"),
            "period": "2023",
            "unit": "VND",
            "source_line_start": 5,
            "source_line_end": 5,
            "extraction_confidence": 0.9,
        },
        {
            "cell_id": "cell_" + "b" * 64,
            "table_id": TABLE_ID,
            "row_idx": 1,
            "col_idx": 1,
            "row_label_raw": "Doanh thu ao",
            "row_label_canonical": None,
            "row_group_context_raw": None,
            "column_label_raw": "Năm 2023",
            "column_label_canonical": None,
            "value_raw": "999",
            "value_numeric": Decimal("999"),
            "period": "2023",
            "unit": "VND",
            "source_line_start": 6,
            "source_line_end": 6,
            "extraction_confidence": 0.9,
        },
    ]
    placements = [
        {
            "table_id": TABLE_ID,
            "row_idx": cell["row_idx"],
            "col_idx": cell["col_idx"],
            "cell_id": cell["cell_id"],
        }
        for cell in cells
    ]

    import pyarrow as pa
    import pyarrow.parquet as pq

    from financial_report_qa.data.dataset_builder import (
        CELL_SCHEMA,
        DOCUMENT_SCHEMA,
        PLACEMENT_SCHEMA,
        TABLE_SCHEMA,
    )

    pq.write_table(pa.Table.from_pylist(documents, schema=DOCUMENT_SCHEMA), release_dir / "documents.parquet")
    pq.write_table(pa.Table.from_pylist(tables, schema=TABLE_SCHEMA), release_dir / "tables.parquet")
    pq.write_table(pa.Table.from_pylist(cells, schema=CELL_SCHEMA), release_dir / "cells.parquet")
    pq.write_table(pa.Table.from_pylist(placements, schema=PLACEMENT_SCHEMA), release_dir / "placements.parquet")

    manifest = {
        "schema_version": "v1",
        "dataset_fingerprint": "0" * 64,
        "source_manifest_sha256": "0" * 64,
        "document_count": len(documents),
        "table_count": len(tables),
        "cell_count": len(cells),
    }
    (release_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return release_dir


def test_ground_with_recovery_falls_back_to_low_confidence_when_nothing_better(
    tmp_path: Path,
) -> None:
    release_dir = _write_release(tmp_path)
    entities = QueryEntities(
        question="Tra cứu chỉ số ACB năm 2023.",
        company_codes=("ACB",),
        periods=("2023",),
        metrics=(),
        metric_phrases=("chỉ số",),
        operation="lookup",
        spans=(),
    )

    # Attempt 0's pick compiles, but only ranks 5th in fusion -- outside the
    # default top-3 confidence threshold -- and it is the only real row in
    # this release, so no better alternative can ever be found.
    fusion_rows = (
        RowFusedCandidate(
            row_id="1",
            table_id=TABLE_ID,
            row_idx=0,
            rank=5,
            snippet="",
            metadata=RowMetadata(
                table_id=TABLE_ID,
                row_idx=0,
                row_label_raw="Doanh thu thuan",
                row_label_canonical="net_revenue",
            ),
            fused_score=0.1,
            bm25_score=0.1,
            dense_score=0.0,
        ),
    )

    with patch(
        "financial_report_qa.planning.cell_grounding.choose_row_label",
        return_value="Doanh thu thuan",
    ), patch(
        "financial_report_qa.planning.cell_grounding.choose_row_label_with_context",
        return_value=None,
    ):
        llm_client = MagicMock(spec=LLMClient)
        res = ground_with_recovery(
            question="Tra cứu chỉ số ACB năm 2023.",
            entities=entities,
            retrieved=(TABLE_ID,),
            row_labels=("Doanh thu thuan",),
            fusion_rows=fusion_rows,
            release_dir=release_dir,
            execution_settings=_ALLOW_LOOKUP,
            llm_client=llm_client,
        )

    assert res.status == "accepted"
    assert res.low_confidence is True
    assert res.plan_source == "llm_cell_grounded"
    assert res.plan is not None
    assert res.plan.metric.raw_text == "Doanh thu thuan"
    assert res.grounding_score == 0.1


def test_ground_with_recovery_prefers_confident_alternative_over_low_rank_pick(
    tmp_path: Path,
) -> None:
    release_dir = _write_two_row_release(tmp_path)
    entities = QueryEntities(
        question="Tra cứu chỉ số ACB năm 2023.",
        company_codes=("ACB",),
        periods=("2023",),
        metrics=(),
        metric_phrases=("chỉ số",),
        operation="lookup",
        spans=(),
    )

    # Attempt 0 picks "Doanh thu ao" -- it compiles (it's a real row in this
    # release) but only ranks 5th. "Doanh thu thuan" ranks 1st; the ladder
    # must prefer it over keeping Attempt 0's low-confidence pick.
    fusion_rows = (
        RowFusedCandidate(
            row_id="1",
            table_id=TABLE_ID,
            row_idx=1,
            rank=5,
            snippet="",
            metadata=RowMetadata(
                table_id=TABLE_ID,
                row_idx=1,
                row_label_raw="Doanh thu ao",
                row_label_canonical=None,
            ),
            fused_score=0.1,
            bm25_score=0.1,
            dense_score=0.0,
        ),
        RowFusedCandidate(
            row_id="2",
            table_id=TABLE_ID,
            row_idx=0,
            rank=1,
            snippet="",
            metadata=RowMetadata(
                table_id=TABLE_ID,
                row_idx=0,
                row_label_raw="Doanh thu thuan",
                row_label_canonical="net_revenue",
            ),
            fused_score=0.9,
            bm25_score=0.9,
            dense_score=0.0,
        ),
    )

    with patch(
        "financial_report_qa.planning.cell_grounding.choose_row_label",
        return_value="Doanh thu ao",
    ):
        llm_client = MagicMock(spec=LLMClient)
        res = ground_with_recovery(
            question="Tra cứu chỉ số ACB năm 2023.",
            entities=entities,
            retrieved=(TABLE_ID,),
            row_labels=("Doanh thu ao", "Doanh thu thuan"),
            fusion_rows=fusion_rows,
            release_dir=release_dir,
            execution_settings=_ALLOW_LOOKUP,
            llm_client=llm_client,
        )

    assert res.status == "accepted"
    assert res.low_confidence is False
    assert res.plan_source == "llm_cell_grounded_recovered"
    assert res.plan is not None
    assert res.plan.metric.raw_text == "Doanh thu thuan"
    assert res.grounding_score == 0.9

