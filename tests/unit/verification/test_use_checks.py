from decimal import Decimal

from financial_report_qa.execution.program_contracts import BoundValue, UseClaim
from financial_report_qa.verification.use_checks import check_use_bindings

_TABLE_ID = "tbl_" + "a" * 64


def _bound(num_index: int, period: int, *, canonical: str | None = None) -> BoundValue:
    return BoundValue(
        num_index=num_index,
        candidate_index=num_index,
        table_id=_TABLE_ID,
        row_idx=3,
        col_idx=num_index + 1,
        row_path="Doanh thu > Doanh thu thuần",
        row_label_raw="Doanh thu thuần",
        row_label_canonical=canonical,
        col_path=f"Tổng_cộng_31/12/{period}",
        period=period,
        value=Decimal("100"),
    )


def test_matching_claims_pass() -> None:
    result = check_use_bindings(
        (UseClaim(num=0, row="Doanh thu thuần", col="Năm 2023"),), (_bound(0, 2023),)
    )

    assert result.matched is True
    assert result.mismatches == ()


def test_swapped_indices_are_caught() -> None:
    # Đây chính là kịch bản trượt chỉ số: `uses` giữ nguyên, `cells` hoán vị.
    bindings = (_bound(0, 2022), _bound(1, 2023))
    uses = (
        UseClaim(num=0, row="Doanh thu thuần", col="Năm 2023"),
        UseClaim(num=1, row="Doanh thu thuần", col="Năm 2022"),
    )

    result = check_use_bindings(uses, bindings)

    assert result.matched is False
    assert len(result.mismatches) == 2


def test_case_and_spacing_differences_still_match() -> None:
    result = check_use_bindings(
        (UseClaim(num=0, row="  DOANH   THU  THUẦN ", col="2023"),), (_bound(0, 2023),)
    )

    assert result.matched is True


def test_the_canonical_label_is_accepted() -> None:
    bound = _bound(0, 2023, canonical="doanh thu bán hàng và cung cấp dịch vụ")
    result = check_use_bindings(
        (UseClaim(num=0, row="Doanh thu bán hàng và cung cấp dịch vụ", col="2023"),), (bound,)
    )

    assert result.matched is True


def test_the_child_label_alone_matches_a_grouped_row_path() -> None:
    result = check_use_bindings(
        (UseClaim(num=0, row="Doanh thu thuần", col="2023"),), (_bound(0, 2023),)
    )

    assert result.matched is True


def test_a_wrong_row_label_is_caught() -> None:
    result = check_use_bindings(
        (UseClaim(num=0, row="Giá vốn hàng bán", col="2023"),), (_bound(0, 2023),)
    )

    assert result.matched is False


def test_a_punctuation_only_row_claim_fails_closed() -> None:
    # "--" chuẩn hoá thành chuỗi rỗng; không được tự khớp mọi dòng.
    result = check_use_bindings((UseClaim(num=0, row="--", col="2023"),), (_bound(0, 2023),))

    assert result.matched is False
    assert len(result.mismatches) == 1
    assert "[NUM_0]" in result.mismatches[0]


def test_a_wrong_year_is_caught() -> None:
    result = check_use_bindings(
        (UseClaim(num=0, row="Doanh thu thuần", col="Năm 2021"),), (_bound(0, 2023),)
    )

    assert result.matched is False


def test_a_column_claim_with_no_year_only_checks_the_row() -> None:
    result = check_use_bindings(
        (UseClaim(num=0, row="Doanh thu thuần", col="cột cuối"),), (_bound(0, 2023),)
    )

    assert result.matched is True


def test_missing_claims_are_a_mismatch() -> None:
    result = check_use_bindings((), (_bound(0, 2023),))

    assert result.matched is False


def test_a_claim_for_an_unknown_placeholder_is_a_mismatch() -> None:
    result = check_use_bindings(
        (UseClaim(num=5, row="Doanh thu thuần", col="2023"),), (_bound(0, 2023),)
    )

    assert result.matched is False
