from decimal import Decimal

import pandas as pd

from financial_report_qa.execution.program_contracts import BoundValue, ExecutedProgram
from financial_report_qa.planning.cell_candidates import build_cell_candidates
from financial_report_qa.retrieval.row_documents import RowMetadata
from financial_report_qa.retrieval.row_fusion_contracts import RowFusedCandidate
from financial_report_qa.submission.exporter import build_item_from_executed

_TABLE_ID = "tbl_" + "a" * 64
_OTHER_TABLE_ID = "tbl_" + "b" * 64


def _row_candidate(row_idx: int, rank: int, group: str | None) -> RowFusedCandidate:
    return RowFusedCandidate(
        row_id=f"{_TABLE_ID}|row_{row_idx}",
        table_id=_TABLE_ID,
        row_idx=row_idx,
        rank=rank,
        fused_score=1.0 - rank / 10,
        metadata=RowMetadata(
            table_id=_TABLE_ID,
            row_idx=row_idx,
            company_code="VCB",
            row_label_raw="Doanh thu thuần" if row_idx == 3 else "Giá vốn",
            row_group_context_raw=group,
        ),
        snippet="x",
    )


def _frame() -> pd.DataFrame:
    rows = []
    for row_idx, label in ((3, "Doanh thu thuần"), (4, "Giá vốn")):
        for col_idx, period in ((1, 2022), (2, 2023)):
            rows.append(
                {
                    "table_id": _TABLE_ID,
                    "company_code": "VCB",
                    "row_idx": row_idx,
                    "col_idx": col_idx,
                    "row_label_raw": label,
                    "row_label_canonical": None,
                    "column_label": f"Năm {period}",
                    "period": period,
                    "statement_type": "income_statement",
                    "unit": "triệu VND",
                    "value": 100.0 * row_idx + col_idx,
                }
            )
    return pd.DataFrame(rows)


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


def test_batch_time_and_export_time_candidate_lists_are_identical() -> None:
    frame = _frame()
    rows = (_row_candidate(4, 1, None), _row_candidate(3, 2, "Doanh thu"))

    first = build_cell_candidates(frame, rows, periods=("2023",))
    second = build_cell_candidates(frame, rows, periods=("2023",))

    assert [c.index for c in first] == [c.index for c in second]
    assert [(c.table_id, c.row_idx, c.col_idx) for c in first] == [
        (c.table_id, c.row_idx, c.col_idx) for c in second
    ]
