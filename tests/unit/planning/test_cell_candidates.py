import pandas as pd

from financial_report_qa.planning.cell_candidates import build_cell_candidates
from financial_report_qa.retrieval.row_documents import RowMetadata
from financial_report_qa.retrieval.row_fusion_contracts import RowFusedCandidate

_TABLE_ID = "tbl_" + "a" * 64


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


def test_candidates_are_numbered_from_zero_in_row_rank_then_column_order() -> None:
    candidates = build_cell_candidates(
        _frame(), (_row_candidate(4, 1, None), _row_candidate(3, 2, "Doanh thu"))
    )

    assert [candidate.index for candidate in candidates] == [0, 1, 2, 3]
    # Dòng hạng 1 (row_idx 4) đứng trước dòng hạng 2 (row_idx 3).
    assert [candidate.row_idx for candidate in candidates] == [4, 4, 3, 3]
    assert [candidate.col_idx for candidate in candidates] == [1, 2, 1, 2]


def test_row_path_carries_the_group_prefix_when_there_is_one() -> None:
    candidates = build_cell_candidates(_frame(), (_row_candidate(3, 1, "Doanh thu"),))

    assert candidates[0].row_path == "Doanh thu > Doanh thu thuần"


def test_row_path_is_the_bare_label_without_a_group() -> None:
    candidates = build_cell_candidates(_frame(), (_row_candidate(4, 1, None),))

    assert candidates[0].row_path == "Giá vốn"


def test_col_path_comes_from_the_column_label() -> None:
    candidates = build_cell_candidates(_frame(), (_row_candidate(3, 1, None),))

    assert candidates[0].col_path == "Năm 2022"
    assert candidates[0].period == 2022


def test_periods_filter_narrows_the_columns() -> None:
    candidates = build_cell_candidates(
        _frame(), (_row_candidate(3, 1, None),), periods=("2023",)
    )

    assert [candidate.period for candidate in candidates] == [2023]


def test_an_empty_periods_filter_keeps_every_column() -> None:
    candidates = build_cell_candidates(_frame(), (_row_candidate(3, 1, None),), periods=())

    assert len(candidates) == 2


def test_no_candidate_carries_a_value() -> None:
    candidates = build_cell_candidates(_frame(), (_row_candidate(3, 1, None),))

    assert all(not hasattr(candidate, "value") for candidate in candidates)


def test_max_candidates_truncates_from_the_lowest_ranked_row() -> None:
    candidates = build_cell_candidates(
        _frame(),
        (_row_candidate(4, 1, None), _row_candidate(3, 2, None)),
        max_candidates=3,
    )

    assert len(candidates) == 3
    assert [candidate.index for candidate in candidates] == [0, 1, 2]
    assert candidates[-1].row_idx == 3


def test_a_row_candidate_with_no_cells_in_the_frame_is_skipped() -> None:
    candidates = build_cell_candidates(_frame(), (_row_candidate(99, 1, None),))

    assert candidates == ()
