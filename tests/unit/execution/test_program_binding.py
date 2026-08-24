from decimal import Decimal

import pandas as pd
import pytest

from financial_report_qa.core.errors import ProgramBindingError
from financial_report_qa.execution.masked_program import run_program
from financial_report_qa.execution.pandas_query import replay_pandas_query
from financial_report_qa.execution.program_binding import (
    bind_values,
    render_cell_lookup,
    render_program_pandas,
    values_by_position,
)
from financial_report_qa.execution.program_contracts import CellCandidate, ProgramDecision

_TABLE_ID = "tbl_" + "a" * 64


def _candidate(index: int, row_idx: int, col_idx: int, label: str, period: int) -> CellCandidate:
    return CellCandidate(
        index=index,
        table_id=_TABLE_ID,
        company_code="VCB",
        row_idx=row_idx,
        col_idx=col_idx,
        row_path=f"Doanh thu > {label}",
        row_label_raw=label,
        row_label_canonical="doanh_thu_thuan",
        col_path=f"Năm_{period}",
        period=period,
    )


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "table_id": _TABLE_ID,
                "company_code": "VCB",
                "row_idx": 3,
                "col_idx": 1,
                "row_label_raw": "Doanh thu thuần",
                "row_label_canonical": "doanh_thu_thuan",
                "column_label": "Năm 2022",
                "period": 2022,
                "unit": "triệu VND",
                "value": 4500.0,
            },
            {
                "table_id": _TABLE_ID,
                "company_code": "VCB",
                "row_idx": 3,
                "col_idx": 2,
                "row_label_raw": "Doanh thu thuần",
                "row_label_canonical": "doanh_thu_thuan",
                "column_label": "Năm 2023",
                "period": 2023,
                "unit": "triệu VND",
                "value": 5310.0,
            },
        ]
    )


def _candidates() -> tuple[CellCandidate, ...]:
    return (
        _candidate(0, 3, 1, "Doanh thu thuần", 2022),
        _candidate(1, 3, 2, "Doanh thu thuần", 2023),
    )


def test_placeholder_index_follows_the_order_of_cells_not_candidate_order() -> None:
    # `cells=[1, 0]` nghĩa là [NUM_0] là ứng viên 1, [NUM_1] là ứng viên 0.
    decision = ProgramDecision(question_id=7, cells=(1, 0), program="[NUM_0] - [NUM_1]")

    bindings = bind_values(decision, _candidates(), values_by_position(_frame()))

    assert [bound.num_index for bound in bindings] == [0, 1]
    assert [bound.candidate_index for bound in bindings] == [1, 0]
    assert [bound.value for bound in bindings] == [Decimal("5310.0"), Decimal("4500.0")]


def test_binding_rejects_an_index_outside_the_candidate_list() -> None:
    decision = ProgramDecision(question_id=7, cells=(9,), program="[NUM_0]")

    with pytest.raises(ProgramBindingError, match="candidate_index_out_of_range"):
        bind_values(decision, _candidates(), values_by_position(_frame()))


def test_binding_rejects_a_candidate_with_no_value_in_the_frame() -> None:
    decision = ProgramDecision(question_id=7, cells=(0,), program="[NUM_0]")
    orphan = (_candidate(0, 99, 99, "Doanh thu thuần", 2022),)

    with pytest.raises(ProgramBindingError):
        bind_values(decision, orphan, values_by_position(_frame()))


def test_cell_lookup_names_the_row_so_the_query_explains_itself() -> None:
    decision = ProgramDecision(question_id=7, cells=(1,), program="[NUM_0]")
    bound = bind_values(decision, _candidates(), values_by_position(_frame()))[0]

    rendered = render_cell_lookup(bound)

    assert "row_label_canonical" in rendered
    assert "df1.row_idx == 3" in rendered
    assert "df1.col_idx == 2" in rendered
    assert rendered.endswith('["value"].iloc[0]')


def test_both_renderings_of_the_same_program_agree() -> None:
    # Đây là bất biến trung tâm của Task 3: một cây AST, hai cách đọc.
    decision = ProgramDecision(
        question_id=7, cells=(1, 0), program="([NUM_0] - [NUM_1]) / [NUM_1]", scale="percent"
    )
    frame = _frame()
    bindings = bind_values(decision, _candidates(), values_by_position(frame))

    from financial_report_qa.execution.masked_program import apply_scale

    direct = apply_scale(run_program(decision.program, [b.value for b in bindings]), "percent")
    query = render_program_pandas(decision.program, bindings, "percent")
    replayed = replay_pandas_query(query, frame)

    assert replayed == pytest.approx(float(direct))


def test_scale_none_appends_no_suffix() -> None:
    decision = ProgramDecision(question_id=7, cells=(1,), program="[NUM_0]")
    bindings = bind_values(decision, _candidates(), values_by_position(_frame()))

    assert not render_program_pandas(decision.program, bindings, "none").endswith("100")


def test_abs_and_unary_minus_survive_the_round_trip() -> None:
    decision = ProgramDecision(question_id=7, cells=(0, 1), program="abs(-[NUM_0] + [NUM_1])")
    frame = _frame()
    bindings = bind_values(decision, _candidates(), values_by_position(frame))

    query = render_program_pandas(decision.program, bindings, "none")
    replayed = replay_pandas_query(query, frame)

    assert replayed == pytest.approx(810.0)


def _dirty_frame(values: tuple[object, object]) -> pd.DataFrame:
    """Same layout as `_frame`, with the `value` column forced to object dtype
    so a None stays None instead of pandas coercing it to NaN."""
    frame = _frame().copy()
    frame["value"] = pd.Series(values, dtype=object)
    return frame


def test_binding_rejects_a_nan_cell_instead_of_binding_it() -> None:
    # Decimal("nan") parses, so an unconditional map would bind NaN and poison
    # every computation downstream. The cell must surface as a binding error.
    decision = ProgramDecision(question_id=7, cells=(0,), program="[NUM_0]")
    frame = _dirty_frame((float("nan"), 5310.0))

    with pytest.raises(ProgramBindingError, match="no numeric value"):
        bind_values(decision, _candidates(), values_by_position(frame))


def test_binding_rejects_a_none_cell_instead_of_crashing_on_decimal_none() -> None:
    # `Decimal(str(None))` raises an uncaught decimal.InvalidOperation unless
    # the unusable row is filtered out before binding.
    decision = ProgramDecision(question_id=7, cells=(0,), program="[NUM_0]")
    frame = _dirty_frame((None, 5310.0))

    with pytest.raises(ProgramBindingError, match="no numeric value"):
        bind_values(decision, _candidates(), values_by_position(frame))


def test_values_by_position_omits_unusable_cells_entirely() -> None:
    # NaN/None/inf/unconvertible junk all parse-or-crash paths; none of them
    # may appear in the map, while usable cells keep their exact values.
    frame = _frame().copy()
    junk_rows = []
    for row_idx, junk in enumerate((float("nan"), None, float("inf"), "không phải số"), start=10):
        row = dict(frame.iloc[0])
        row["row_idx"] = row_idx
        row["value"] = junk
        junk_rows.append(row)
    dirty = pd.DataFrame([*frame.to_dict("records"), *junk_rows])

    values = values_by_position(dirty)

    assert set(values) == {(_TABLE_ID, 3, 1), (_TABLE_ID, 3, 2)}
    assert values[(_TABLE_ID, 3, 2)] == Decimal("5310.0")
