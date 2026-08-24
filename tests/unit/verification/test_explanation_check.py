from decimal import Decimal

from financial_report_qa.execution.program_contracts import BoundValue, ExecutedProgram
from financial_report_qa.verification.explanation_check import (
    check_explanation,
    program_number_whitelist,
)

_TABLE_ID = "tbl_" + "a" * 64


def _bound(num_index: int, period: int, value: str) -> BoundValue:
    return BoundValue(
        num_index=num_index,
        candidate_index=num_index,
        table_id=_TABLE_ID,
        row_idx=3,
        col_idx=num_index + 1,
        row_path="Doanh thu thuần",
        row_label_raw="Doanh thu thuần",
        col_path=f"Năm_{period}",
        period=period,
        value=Decimal(value),
    )


def _executed() -> ExecutedProgram:
    return ExecutedProgram(
        question_id=7,
        program="([NUM_0] - [NUM_1]) / [NUM_1]",
        scale="percent",
        bindings=(_bound(0, 2023, "5310"), _bound(1, 2022, "4500")),
        answer=Decimal("18"),
        pandas_query='df1[(df1.row_idx == 3)]["value"].iloc[0]',
        table_ids=(_TABLE_ID,),
    )


def test_whitelist_holds_the_answer_the_values_and_the_periods() -> None:
    whitelist = program_number_whitelist(_executed())

    assert Decimal("18") in whitelist
    assert Decimal("5310") in whitelist
    assert Decimal("4500") in whitelist
    assert Decimal("2023") in whitelist
    assert Decimal("2022") in whitelist


def test_an_explanation_using_only_grounded_numbers_passes() -> None:
    text = "Doanh thu thuần tăng từ 4500 năm 2022 lên 5310 năm 2023, tức 18%."

    assert check_explanation(text, _executed()).allowed is True


def test_an_invented_number_is_rejected() -> None:
    text = "Doanh thu thuần tăng từ 4500 lên 5310, tương đương 810 tỷ và 20%."

    result = check_explanation(text, _executed())

    assert result.allowed is False
    assert "20" in result.disallowed_numbers


def test_an_explanation_with_no_numbers_passes() -> None:
    assert check_explanation("Doanh thu thuần tăng so với năm trước.", _executed()).allowed


def test_the_raw_unscaled_result_is_not_whitelisted() -> None:
    # 0.18 là kết quả trước khi áp `scale`; giải thích chỉ được nêu đáp án đã
    # scale, nếu không thì con số người chấm thấy khác con số bài nộp.
    result = check_explanation("Tỷ lệ là 0.18", _executed())

    assert result.allowed is False
