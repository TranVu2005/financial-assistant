"""Contract pins for the masked-PAL export path.

The most important invariant of the whole plan lives here: the numbered
cell-candidate list must be built through exactly ONE shared helper
(`exporter.build_question_cell_candidates`) by BOTH production paths --
`submission row-batches` at payload-generation time and
`_run_one_question`'s answering branch at export time -- fed identical
inputs (full un-narrowed `retrieved`). A list that differs between the two
moments shifts every `ProgramDecision.cells` index.
"""

from decimal import Decimal
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock

import pyarrow as pa
import pyarrow.parquet as pq

from financial_report_qa.data.dataset_builder import (
    CELL_SCHEMA,
    DOCUMENT_SCHEMA,
    PLACEMENT_SCHEMA,
    TABLE_SCHEMA,
)
from financial_report_qa.execution.program_contracts import BoundValue, ExecutedProgram
from financial_report_qa.planning.entity_parser import parse_query_entities
from financial_report_qa.planning.row_choice_batch import build_program_batch_payload
from financial_report_qa.retrieval.contracts import (
    MetricLabelObservation,
    TableDocument,
    TableMetadata,
)
from financial_report_qa.retrieval.index import build_bm25_index
from financial_report_qa.retrieval.live_query import (
    TableRetriever,
    retrieve_candidate_table_ids,
)
from financial_report_qa.retrieval.row_documents import RowMetadata
from financial_report_qa.retrieval.row_fusion import DEFAULT_ROW_CANDIDATE_COUNT, RowFusionService
from financial_report_qa.retrieval.row_fusion_contracts import (
    RowFusedCandidate,
    RowFusionTrace,
    RowFusionWeights,
)
from financial_report_qa.retrieval.service import RetrievalService
from financial_report_qa.submission.exporter import (
    build_item_from_executed,
    build_question_cell_candidates,
)

_TABLE_ID = "tbl_" + "1" * 64
_DOC_ID = "doc_" + "a" * 64
_CELL_ID = "cell_" + "a" * 64
_OTHER_TABLE_ID = "tbl_" + "b" * 64

_QUESTION = "Tra cứu doanh thu thuần của ACB năm 2023."


def _executed() -> ExecutedProgram:
    return ExecutedProgram(
        question_id=7,
        program="[NUM_0]",
        scale="none",
        bindings=(
            BoundValue(
                num_index=0,
                candidate_index=0,
                table_id=_TABLE_ID,
                row_idx=3,
                col_idx=2,
                row_path="Doanh thu thuần",
                row_label_raw="Doanh thu thuần",
                col_path="Năm_2023",
                period=2023,
                value=Decimal("5310"),
            ),
        ),
        answer=Decimal("5310"),
        pandas_query='df1[(df1.row_idx == 3)]["value"].iloc[0]',
        table_ids=(_TABLE_ID,),
    )


def test_the_item_carries_the_program_for_c8() -> None:
    item = build_item_from_executed(
        _executed(), retrieved=(_OTHER_TABLE_ID, _TABLE_ID), relevant_docs=("doc.txt",)
    )

    assert item.program == "[NUM_0]"
    assert item.answer == 5310.0


def test_relevant_tables_keep_retrieval_rank_order_not_the_executed_tables() -> None:
    # N1 + bất biến MRR5: nhánh retrieval không bị nhánh answering ghi đè.
    item = build_item_from_executed(
        _executed(), retrieved=(_OTHER_TABLE_ID, _TABLE_ID), relevant_docs=("doc.txt",)
    )

    assert item.relevant_tables == (_OTHER_TABLE_ID, _TABLE_ID)


def _write_release(tmp_path: Path) -> Path:
    """Small release: one table, two numeric rows (period 2023)."""
    release_dir = tmp_path / "release"
    release_dir.mkdir()
    documents = [
        {
            "doc_id": _DOC_ID,
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
            "table_id": _TABLE_ID,
            "doc_id": _DOC_ID,
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
            "cell_id": _CELL_ID,
            "table_id": _TABLE_ID,
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
            "cell_id": "cell_" + "d" * 64,
            "table_id": _TABLE_ID,
            "row_idx": 1,
            "col_idx": 1,
            "row_label_raw": "Gia von hang ban",
            "row_label_canonical": "cost_of_goods_sold",
            "row_group_context_raw": None,
            "column_label_raw": "Năm 2023",
            "column_label_canonical": None,
            "value_raw": "60",
            "value_numeric": Decimal("60"),
            "period": "2023",
            "unit": "VND",
            "source_line_start": 6,
            "source_line_end": 6,
            "extraction_confidence": 0.9,
        },
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
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.Table.from_pylist(
            [
                {
                    "table_id": cell["table_id"],
                    "row_idx": cell["row_idx"],
                    "col_idx": cell["col_idx"],
                    "cell_id": cell["cell_id"],
                }
                for cell in cells
            ],
            schema=PLACEMENT_SCHEMA,
        ),
        release_dir / "placements.parquet",
    )
    return release_dir


def _bm25_service() -> TableRetriever:
    document = TableDocument(
        table_id=_TABLE_ID,
        doc_id=_DOC_ID,
        text="company_code: ACB\nperiod: 2023\nDoanh thu thuần | 2023 | 100",
        metadata=TableMetadata(
            table_id=_TABLE_ID,
            doc_id=_DOC_ID,
            company_code="ACB",
            periods=("2023",),
            statement_type="income_statement",
            source_path="a.txt",
            line_start=1,
            line_end=3,
        ),
        metric_labels=(MetricLabelObservation(canonical="net_revenue", raw=None),),
    )
    # cast: RetrievalTrace structurally satisfies TableRetriever but mypy
    # cannot prove it against the _RankedResult protocol (same known pattern
    # as retrieval/cli.py's sweep-k wiring).
    index = build_bm25_index((document,), dataset_fingerprint="f" * 64)
    return cast(TableRetriever, RetrievalService(index))


def _fused_row(row_idx: int, rank: int, label: str) -> RowFusedCandidate:
    return RowFusedCandidate(
        row_id=f"{_TABLE_ID}|row_{row_idx}",
        table_id=_TABLE_ID,
        row_idx=row_idx,
        rank=rank,
        fused_score=1.0 - rank / 10,
        snippet=f"{label} | 2023",
        metadata=RowMetadata(
            table_id=_TABLE_ID,
            row_idx=row_idx,
            company_code="ACB",
            row_label_raw=label,
        ),
    )


def _row_fusion(rows: tuple[RowFusedCandidate, ...]) -> MagicMock:
    fusion = MagicMock(spec=RowFusionService)

    def _trace(query: str, **_: object) -> RowFusionTrace:
        return RowFusionTrace(
            query=query,
            weights=RowFusionWeights(bm25=1, dense=0),
            candidate_table_ids=(_TABLE_ID,),
            bm25_candidate_count=len(rows),
            dense_candidate_count=0,
            results=rows,
        )

    fusion.retrieve_rows.side_effect = _trace
    return fusion


_FUSION_ROWS = (_fused_row(0, 1, "Doanh thu thuan"), _fused_row(1, 2, "Gia von hang ban"))


def test_batch_time_and_export_time_candidate_lists_are_identical(tmp_path: Path) -> None:
    """The plan's most-important invariant, pinned across the two REAL
    production sequences (not merely helper purity):

    - batch time: exactly what ``submission/cli.py``'s row-batches branch
      does -- retrieval -> row fusion over the full retrieved list ->
      ``build_question_cell_candidates(release_dir, question, retrieved,
      fused)`` -> ``build_program_batch_payload``;
    - export time: exactly what ``exporter._run_one_question``'s answering
      branch does -- the same fusion over the same full retrieved list into
      the same shared helper.

    Both lists must be element-wise identical (index / table_id / row_idx /
    col_idx), and the JSONL payload must mirror that list field-by-field:
    an offline decision's ``cells`` positions are only meaningful against
    THIS ordering."""
    release_dir = _write_release(tmp_path)

    # ---- batch time (cli.py row-batches branch, verbatim) ----
    retrieved = retrieve_candidate_table_ids(_QUESTION, _bm25_service(), k=10)
    assert retrieved == (_TABLE_ID,), "fixture must retrieve the fixture table"
    fused = (
        _row_fusion(_FUSION_ROWS)
        .retrieve_rows(_QUESTION, candidate_table_ids=retrieved, k=DEFAULT_ROW_CANDIDATE_COUNT)
        .results
    )
    payload = build_program_batch_payload(
        1,
        _QUESTION,
        parse_query_entities(_QUESTION),
        build_question_cell_candidates(release_dir, _QUESTION, retrieved, fused),
    )

    # ---- export time (exporter._run_one_question branch, verbatim) ----
    fusion_rows = (
        _row_fusion(_FUSION_ROWS)
        .retrieve_rows(_QUESTION, candidate_table_ids=retrieved, k=DEFAULT_ROW_CANDIDATE_COUNT)
        .results
    )
    candidates_export = build_question_cell_candidates(
        release_dir, _QUESTION, retrieved, fusion_rows
    )

    # Element-wise identity on position fields, contiguous from 0.
    assert [(c.index, c.table_id, c.row_idx, c.col_idx) for c in candidates_export] == [
        (0, _TABLE_ID, 0, 1),
        (1, _TABLE_ID, 1, 1),
    ]
    # Rank order is the contract: the rank-1 row's cells come first.
    assert [c.row_label_raw for c in candidates_export] == ["Doanh thu thuan", "Gia von hang ban"]
    # The JSONL payload mirrors exactly this list, field by field.
    assert [c["index"] for c in payload["candidates"]] == list(range(len(candidates_export)))
    for payload_candidate, candidate in zip(payload["candidates"], candidates_export):
        assert payload_candidate["company_code"] == candidate.company_code
        assert payload_candidate["row_path"] == candidate.row_path
        assert payload_candidate["col_path"] == candidate.col_path
        assert payload_candidate["period"] == candidate.period
        assert payload_candidate["statement_type"] == candidate.statement_type
        assert payload_candidate["unit"] == candidate.unit


def test_both_production_paths_call_the_one_shared_helper_with_the_full_retrieved_list() -> None:
    """Source-level wiring guard complementing the runtime pin above: neither
    path may grow its own construction site or narrow the fusion input."""
    import financial_report_qa.submission.cli as cli_module
    import financial_report_qa.submission.exporter as exporter_module

    cli_source = Path(cli_module.__file__).read_text(encoding="utf-8")
    exporter_source = Path(exporter_module.__file__).read_text(encoding="utf-8")

    # cli's row-batches branch feeds the helper the RAW retrieved list.
    assert (
        "build_question_cell_candidates(\n"
        "                            args.release_dir, raw_question.question, retrieved, fused\n"
        "                        ),"
    ) in cli_source
    assert "answerable" not in cli_source, (
        "the batch generator must never see a scope-narrowed table set"
    )

    # exporter: exactly ONE construction site (definition + one call).
    assert exporter_source.count("build_question_cell_candidates(") == 2, (
        "the numbered list may be built in exactly one place"
    )
    # The answering branch fuses over `retrieved`; the ONLY narrowed fusion
    # left is the backstop hint inside `_decision_row_hint`.
    assert "candidate_table_ids=retrieved," in exporter_source
    assert exporter_source.count("candidate_table_ids=answerable") == 1
    hint_start = exporter_source.find("def _decision_row_hint(")
    hint_end = exporter_source.find("\ndef ", hint_start + 10)
    narrowed_region = exporter_source[hint_start : hint_end if hint_end != -1 else None]
    assert "candidate_table_ids=answerable" in narrowed_region
