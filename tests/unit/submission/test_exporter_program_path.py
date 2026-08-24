from decimal import Decimal

from financial_report_qa.execution.program_contracts import BoundValue, ExecutedProgram
from financial_report_qa.submission.exporter import build_item_from_executed

_TABLE_ID = "tbl_" + "a" * 64
_OTHER_TABLE_ID = "tbl_" + "b" * 64


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
