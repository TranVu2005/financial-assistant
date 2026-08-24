from decimal import Decimal

import pandas as pd

from financial_report_qa.execution.program_contracts import CellCandidate, ProgramDecision
from financial_report_qa.execution.program_pipeline import run_question

_TABLE_ID = "tbl_" + "a" * 64


def _candidates() -> tuple[CellCandidate, ...]:
    return tuple(
        CellCandidate(
            index=index,
            table_id=_TABLE_ID,
            company_code="VCB",
            row_idx=3,
            col_idx=index + 1,
            row_path="Doanh thu thuần",
            row_label_raw="Doanh thu thuần",
            col_path=f"Năm_{2022 + index}",
            period=2022 + index,
        )
        for index in range(2)
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
                "row_label_canonical": None,
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
                "row_label_canonical": None,
                "column_label": "Năm 2023",
                "period": 2023,
                "unit": "triệu VND",
                "value": 5310.0,
            },
        ]
    )


class _Decisions:
    """Trả về từng quyết định theo lần thử, đếm số lần được hỏi."""

    def __init__(self, *decisions: ProgramDecision) -> None:
        self._decisions = decisions
        self.attempts: list[int] = []

    def decide(self, question_id: int, attempt: int) -> ProgramDecision:
        self.attempts.append(attempt)
        return self._decisions[min(attempt, len(self._decisions) - 1)]


def _good() -> ProgramDecision:
    return ProgramDecision(
        question_id=7,
        cells=(1,),
        program="[NUM_0]",
        uses=({"num": 0, "row": "Doanh thu thuần", "col": "Năm 2023"},),  # type: ignore[arg-type]
    )


def _bad_literal() -> ProgramDecision:
    return ProgramDecision(question_id=7, cells=(1,), program="[NUM_0] * 100")


def test_a_clean_first_attempt_is_not_marked_regenerated() -> None:
    source = _Decisions(_good())

    result = run_question(7, _candidates(), _frame(), source)

    assert result.executed is not None
    assert result.executed.answer == Decimal("5310.0")
    assert result.executed.regenerated is False
    assert result.executed.low_confidence is False
    assert source.attempts == [0]


def test_a_bad_first_attempt_is_retried_exactly_once_and_recovers() -> None:
    source = _Decisions(_bad_literal(), _good())

    result = run_question(7, _candidates(), _frame(), source)

    assert result.executed is not None
    assert result.executed.regenerated is True
    assert result.executed.low_confidence is False
    assert source.attempts == [0, 1]


def test_two_bad_attempts_still_produce_an_answer_marked_low_confidence() -> None:
    # Bỏ trống chắc chắn 0 điểm; sai thì cũng chỉ có thể 0 điểm.
    bad_uses = ProgramDecision(
        question_id=7,
        cells=(1,),
        program="[NUM_0]",
        uses=({"num": 0, "row": "Giá vốn hàng bán", "col": "Năm 2023"},),  # type: ignore[arg-type]
    )
    source = _Decisions(bad_uses, bad_uses)

    result = run_question(7, _candidates(), _frame(), source)

    assert result.executed is not None
    assert result.executed.low_confidence is True
    assert result.executed.failure_code == "use_binding_mismatch"
    assert source.attempts == [0, 1]


def test_the_retry_never_runs_a_third_time() -> None:
    source = _Decisions(_bad_literal(), _bad_literal(), _good())

    result = run_question(7, _candidates(), _frame(), source)

    assert source.attempts == [0, 1]
    # Không bind được lần nào -> không có đáp án, nhưng phải nói rõ vì sao.
    assert result.executed is None
    assert result.failure_code == "numeric_literal_in_program"


def test_an_empty_candidate_list_fails_before_asking_the_model() -> None:
    source = _Decisions(_good())

    result = run_question(7, (), _frame(), source)

    assert result.executed is None
    assert result.failure_code == "no_cell_candidates"
    assert source.attempts == []


def test_a_fabricated_number_in_the_explanation_triggers_the_retry() -> None:
    source = _Decisions(_good(), _good())
    explanations = iter(["Doanh thu là 9999.", "Doanh thu là 5310."])

    result = run_question(
        7,
        _candidates(),
        _frame(),
        source,
        explanations=lambda executed: next(explanations),
    )

    assert result.executed is not None
    assert result.executed.regenerated is True
    assert result.executed.low_confidence is False


def test_division_by_zero_is_reported_by_its_code() -> None:
    zero_frame = _frame()
    zero_frame.loc[zero_frame["col_idx"] == 1, "value"] = 0.0
    decision = ProgramDecision(
        question_id=7,
        cells=(1, 0),
        program="[NUM_0] / [NUM_1]",
        uses=(
            {"num": 0, "row": "Doanh thu thuần", "col": "Năm 2023"},  # type: ignore[arg-type]
            {"num": 1, "row": "Doanh thu thuần", "col": "Năm 2022"},  # type: ignore[arg-type]
        ),
    )
    source = _Decisions(decision, decision)

    result = run_question(7, _candidates(), zero_frame, source)

    assert result.executed is None
    assert result.failure_code == "division_by_zero"
