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
            # Row 99 does not exist in the release: this candidate is the
            # decoy the LLM picks at Attempt 0, so binding it positionally
            # (plan.md §14) must fail exactly as matching its label does.
            table_id=TABLE_ID,
            row_idx=99,
            rank=1,
            snippet="",
            metadata=RowMetadata(
                table_id=TABLE_ID,
                row_idx=99,
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
            row_idx=99,
            rank=1,
            snippet="Doanh thu ao snippet",
            metadata=RowMetadata(
                table_id=TABLE_ID,
                row_idx=99,
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



def test_company_name_rows_are_never_offered_as_metric_rows(tmp_path: Path) -> None:
    """Grounding recovery walks the fusion-ranked row labels until one
    compiles. Issuer-name rows compile just fine (they sit in a numeric
    table) but hold no metric, so a real run accepted "▪ Tập đoàn Dệt May
    Việt Nam - Công ty mẹ" as the row for "Tổng cộng dự phòng phải trả"
    (plan.md §19 dev benchmark, question 175). They must be dropped before
    any tier -- LLM row choice included -- gets to pick one."""
    release_dir = _write_release(tmp_path)
    entities = QueryEntities(
        question="Tổng cộng dự phòng phải trả của Tập đoàn Dệt May Việt Nam năm 2023?",
        company_codes=("ACB",),
        periods=("2023",),
        metrics=(),
        metric_phrases=("Tổng cộng dự phòng phải trả",),
        operation="lookup",
        spans=(),
    )
    offered: list[tuple[str, ...]] = []

    def _capture(question: str, labels: object, *, client: object) -> str | None:
        offered.append(tuple(labels))  # type: ignore[arg-type]
        return "Doanh thu thuan"

    with patch(
        "financial_report_qa.planning.cell_grounding.choose_row_label", side_effect=_capture
    ):
        res = ground_with_recovery(
            question=entities.question,
            entities=entities,
            retrieved=(TABLE_ID,),
            row_labels=(
                "▪ Tập đoàn Dệt May Việt Nam - Công ty mẹ",
                "Doanh thu thuan",
            ),
            fusion_rows=(),
            release_dir=release_dir,
            execution_settings=_ALLOW_LOOKUP,
            llm_client=MagicMock(spec=LLMClient),
        )

    assert offered, "the LLM row-choice tier was never reached"
    for labels in offered:
        assert "▪ Tập đoàn Dệt May Việt Nam - Công ty mẹ" not in labels
        assert "Doanh thu thuan" in labels
    assert res.status == "accepted"


def _write_release_with_duplicate_label(tmp_path: Path) -> Path:
    """A release where one raw label names two rows that disagree -- the shape
    that made 71 of 88 wrong dev-benchmark answers land on a plausible number
    from the wrong row (plan.md §19 / v2-remaining-gaps)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from financial_report_qa.data.dataset_builder import CELL_SCHEMA, DOCUMENT_SCHEMA, TABLE_SCHEMA

    release_dir = _write_release(tmp_path)
    cells = [
        {
            "cell_id": "cell_" + character * 64,
            "table_id": TABLE_ID,
            "row_idx": row_idx,
            "col_idx": 1,
            "row_label_raw": "Doanh thu thuan",
            "row_label_canonical": None,
            "row_group_context_raw": None,
            "column_label_raw": "Năm 2023",
            "column_label_canonical": None,
            "value_raw": value,
            "value_numeric": Decimal(value),
            "period": "2023",
            "unit": "VND",
            "source_line_start": 5,
            "source_line_end": 5,
            "extraction_confidence": 0.9,
        }
        for character, row_idx, value in (("b", 1, "100"), ("c", 2, "900"))
    ]
    pq.write_table(
        pa.Table.from_pylist(cells, schema=CELL_SCHEMA), release_dir / "cells.parquet"
    )
    # `documents`/`tables` are rewritten identically so the release stays whole.
    for name, schema in (("documents", DOCUMENT_SCHEMA), ("tables", TABLE_SCHEMA)):
        table = pq.read_table(release_dir / f"{name}.parquet")
        pq.write_table(table.cast(schema), release_dir / f"{name}.parquet")
    return release_dir


def _duplicate_label_fusion_row(row_idx: int, rank: int) -> RowFusedCandidate:
    return RowFusedCandidate(
        row_id=f"{TABLE_ID}|row_{row_idx}",
        table_id=TABLE_ID,
        row_idx=row_idx,
        rank=rank,
        fused_score=1.0 / rank,
        snippet="Doanh thu thuan",
        metadata=RowMetadata(
            table_id=TABLE_ID,
            row_idx=row_idx,
            company_code="ACB",
            row_label_raw="Doanh thu thuan",
            row_label_canonical=None,
        ),
    )


def _duplicate_label_entities() -> QueryEntities:
    return QueryEntities(
        question="Doanh thu thuan của ACB năm 2023 là bao nhiêu?",
        company_codes=("ACB",),
        periods=("2023",),
        metrics=(),
        metric_phrases=("Doanh thu thuan",),
        operation="lookup",
        spans=(),
    )


def test_ground_with_recovery_uses_the_retrieved_row_position_to_break_a_label_tie(
    tmp_path: Path,
) -> None:
    """plan.md §9/§14: retrieval already ranked one of the two identically
    labelled rows first. Binding the plan to that position answers from it,
    where label matching can only report the pair as ambiguous."""
    release_dir = _write_release_with_duplicate_label(tmp_path)
    res = ground_with_recovery(
        question="Doanh thu thuan của ACB năm 2023 là bao nhiêu?",
        entities=_duplicate_label_entities(),
        retrieved=(TABLE_ID,),
        row_labels=("Doanh thu thuan",),
        fusion_rows=(
            _duplicate_label_fusion_row(row_idx=2, rank=1),
            _duplicate_label_fusion_row(row_idx=1, rank=2),
        ),
        release_dir=release_dir,
        execution_settings=_ALLOW_LOOKUP,
    )

    assert res.status == "accepted"
    assert res.compiled is not None
    assert res.compiled.answer == Decimal("900")
    assert "df1.loc[" in res.compiled.pandas_query
    assert res.plan is not None and res.plan.metric is not None
    assert res.plan.metric.is_position_bound


def test_ground_with_recovery_exposes_grounded_facts_for_the_accepted_answer(
    tmp_path: Path,
) -> None:
    """plan.md §9: the accepted result carries per-fact provenance, keyed by
    row index rather than by the label string."""
    release_dir = _write_release_with_duplicate_label(tmp_path)
    res = ground_with_recovery(
        question="Doanh thu thuan của ACB năm 2023 là bao nhiêu?",
        entities=_duplicate_label_entities(),
        retrieved=(TABLE_ID,),
        row_labels=("Doanh thu thuan",),
        fusion_rows=(_duplicate_label_fusion_row(row_idx=2, rank=1),),
        release_dir=release_dir,
        execution_settings=_ALLOW_LOOKUP,
    )

    assert res.status == "accepted"
    assert len(res.facts) == 1
    fact = res.facts[0]
    assert fact.fact_id == "F1"
    assert fact.table_id == TABLE_ID
    assert fact.row_index == 2
    assert fact.row_label == "Doanh thu thuan"
    assert fact.column == "Năm 2023"
    assert fact.period == 2023
    assert fact.raw_value == Decimal("900")
    assert fact.unit == "VND"


def test_ground_with_recovery_still_answers_when_no_position_can_be_bound(
    tmp_path: Path,
) -> None:
    """Binding is an improvement, not a new failure mode: with no fusion rows
    to bind against, grounding behaves exactly as it did before."""
    release_dir = _write_release(tmp_path)
    res = ground_with_recovery(
        question="Tra cứu Doanh thu thuan của ACB năm 2023.",
        entities=_duplicate_label_entities(),
        retrieved=(TABLE_ID,),
        row_labels=("Doanh thu thuan",),
        fusion_rows=(),
        release_dir=release_dir,
        execution_settings=_ALLOW_LOOKUP,
    )
    assert res.status == "accepted"
    assert res.compiled is not None and res.compiled.answer == Decimal("100")
    assert res.plan is not None and res.plan.metric is not None
    assert res.plan.metric.is_position_bound is False
