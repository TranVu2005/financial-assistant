import pytest

from financial_report_qa.normalization.companies import (
    COMPANY_REGISTRY,
    company_name_codes_in_text,
    company_name_for_code,
    normalize_company,
    resolve_company_code,
    validate_company_codes,
)
from financial_report_qa.schemas.documents import DocumentRecord, stable_document_id


def _document(company_code: str = "VCB") -> DocumentRecord:
    digest = "a" * 64
    return DocumentRecord(
        doc_id=stable_document_id(digest),
        repo_id="org/vifinqa",
        revision="rev-1",
        relative_path=f"{company_code}/2024/Consolidated/report.txt",
        company_code=company_code,
        report_year=2024,
        statement_scope="consolidated",
        sha256=digest,
        file_size_bytes=120,
        encoding="utf-8",
        inventory_status="ready",
    )


def test_registry_is_validated_and_contains_all_seed_rows() -> None:
    assert len(COMPANY_REGISTRY) == 100
    assert len(set(COMPANY_REGISTRY)) == 100
    assert company_name_for_code(" stb ") == "Ngân hàng TMCP Sài Gòn Thương Tín"
    assert company_name_for_code("TCB") is None


def test_every_canonical_company_name_and_curated_alias_resolves() -> None:
    for ticker, record in COMPANY_REGISTRY.items():
        canonical = resolve_company_code(record.canonical_name)
        assert canonical.value == ticker, record.canonical_name
        assert canonical.issue_code is None

        for alias in record.aliases:
            resolved_alias = resolve_company_code(alias)
            assert resolved_alias.value == ticker, alias
            assert resolved_alias.issue_code is None


@pytest.mark.parametrize(
    ("raw", "ticker"),
    [
        ("vcb", "VCB"),
        ("Mã CK: VCB", "VCB"),
        ("Ngân hàng TMCP Ngoại thương Việt Nam", "VCB"),
        ("Ngan hang thuong mai co phan Ngoai thuong Viet Nam", "VCB"),
        ("BÁO CÁO TÀI CHÍNH - CTCP Tập đoàn Hòa Phát", "HPG"),
        ("CTCP Tập đoàn Đất Xanh", "DXG"),
        ("Báo cáo của CTCP Chứng khoán FPT", "FTS"),
        ("CTCP Nông nghiệp Quốc tế Hoàng Anh Gia Lai", "HNG"),
    ],
)
def test_resolve_company_code_supports_ticker_legal_name_and_historical_alias(
    raw: str, ticker: str
) -> None:
    decision = resolve_company_code(raw)

    assert decision.value == ticker
    assert decision.issue_code is None


@pytest.mark.parametrize(
    "raw",
    [
        None,
        "",
        "Báo cáo có tỷ lệ VCB tăng trong kỳ",
        "Ngân hàng Ngoại thương Việt",
    ],
)
def test_resolver_does_not_fuzzy_match_or_guess_ticker_substrings(raw: str | None) -> None:
    decision = resolve_company_code(raw)

    assert decision.value is None
    assert decision.issue_code is None


def test_resolver_reports_contradictory_ticker_and_legal_name() -> None:
    decision = resolve_company_code("Mã CK: VCB - Ngân hàng TMCP Quân đội")

    assert decision.value is None
    assert decision.issue_code == "company_conflict"


def test_normalize_company_preserves_inventory_authority() -> None:
    matching = normalize_company(
        _document("VCB"), "Ngân hàng TMCP Ngoại thương Việt Nam - BCTC 2024"
    )
    conflicting = normalize_company(_document("VCB"), "Mã CK: MBB")

    assert matching.value == "VCB"
    assert matching.issue_code is None
    assert conflicting.value == "VCB"
    assert conflicting.issue_code == "company_conflict"


@pytest.mark.parametrize("title", ["VND", "CONTENTS", "Trang", "Shares"])
def test_table_layout_titles_are_not_company_evidence(title: str) -> None:
    assert normalize_company(_document("HBC"), title).issue_code is None


def test_only_explicit_conflicting_ticker_is_a_table_company_conflict() -> None:
    decision = normalize_company(_document("VCB"), "Mã CK: MBB")

    assert decision.value == "VCB"
    assert decision.issue_code == "company_conflict"


def test_registry_is_open_set_for_valid_unlisted_tickers() -> None:
    decision = resolve_company_code("TCB")

    assert decision.value == "TCB"
    assert decision.issue_code is None
    assert validate_company_codes(["VCB", "tcb", "TCB", "HPG"]) == ("TCB",)


@pytest.mark.parametrize(
    ("question", "ticker"),
    [
        # Day 22 coverage follow-up: 34 real ViFinQA questions failed with
        # company_missing because the registry only matched each company's
        # exact >=3-word canonical name, not the shorter informal >=3-word
        # phrase real questions actually use. `code_stock.csv` (the source
        # ViFinQA itself generated questions from) confirms these referents,
        # including STB's erroneous "Sài Gòn Tài Lộc" label -- kept as an
        # alias, not "corrected", because the question corpus uses it verbatim.
        (
            "Chi phí dự phòng của Ngân hàng TMCP Sài Gòn Tài Lộc trong năm 2020 "
            "là bao nhiêu triệu đồng?",
            "STB",
        ),
        (
            "Trong giai đoạn 2022–2024 của Tổng công ty Phân bón Dầu khí Cà Mau, "
            "xét các năm có tỷ lệ lợi nhuận sau thuế trên doanh thu thuần lớn hơn 10%.",
            "DCM",
        ),
        (
            "Khoản chênh lệch số dư Quỹ đầu tư phát triển cuối năm 2015 giữa "
            "Tập đoàn Công nghiệp Cao su Việt Nam và Tổng công ty Phân bón và "
            "Hóa chất Dầu khí là bao nhiêu nghìn tỷ đồng?",
            "GVR",
        ),
        (
            "Tốc độ tăng trưởng tiền và các khoản tương đương tiền của Tổng Công "
            "ty Cảng Hàng không Việt Nam từ cuối năm 2021 đến cuối năm 2022.",
            "ACV",
        ),
        (
            "Chênh lệch khoản vay ngắn hạn từ các bên liên quan cuối năm 2025 của "
            "công ty mẹ Tổng Công ty Phát triển Đô thị Kinh Bắc so với công ty mẹ "
            "Tập đoàn VINGROUP là bao nhiêu tỷ đồng?",
            "KBC",
        ),
        (
            "Cuối năm 2024, dự phòng rủi ro cho vay khách hàng của ngân hàng mẹ "
            "Bắc Á chênh lệch bao nhiêu triệu đồng so với Saigonbank?",
            "BAB",
        ),
        (
            "Năm 2024, trạng thái tiền tệ nội bảng của công ty mẹ Ngân hàng Quốc "
            "Dân và Ngân hàng Kiên Long chênh lệch nhau mấy triệu đồng?",
            "NVB",
        ),
        (
            "Trong năm 2021, tỷ trọng chi phí lãi tiền gửi của Sài Gòn Công Thương "
            "và TMCP Quốc Dân chênh lệch nhau bao nhiêu phần trăm?",
            "SGB",
        ),
        (
            "Chênh lệch tổng phải thu ngắn hạn từ các bên liên quan tại thời điểm "
            "cuối năm 2018 giữa hai công ty mẹ Địa ốc Sài Gòn Thương Tín và Đầu "
            "tư Phát triển Xây dựng là bao nhiêu tỷ đồng?",
            "SCR",
        ),
    ],
)
def test_company_name_codes_in_text_finds_informal_multiword_names(
    question: str, ticker: str
) -> None:
    assert ticker in company_name_codes_in_text(question)


def test_company_name_codes_in_text_finds_multiple_distinct_companies() -> None:
    """The Day 22 fix must not just swap which single winner is picked --
    two genuinely distinct companies named with different-length aliases in
    the same sentence must both be returned."""
    question = (
        "Khoản chênh lệch số dư Quỹ đầu tư phát triển cuối năm 2015 giữa Tập "
        "đoàn Công nghiệp Cao su Việt Nam và Tổng công ty Phân bón và Hóa "
        "chất Dầu khí là bao nhiêu nghìn tỷ đồng?"
    )
    assert set(company_name_codes_in_text(question)) >= {"GVR", "DPM"}


def test_company_name_codes_in_text_does_not_confuse_subsidiary_with_parent() -> None:
    """A subsidiary's own full legal name must still resolve to the
    subsidiary, not fall through to a parent-group alias that happens to be
    a substring of it (the original reason `_contained_company_codes`
    prefers the longest match at a given text span)."""
    assert company_name_codes_in_text(
        "Báo cáo tài chính của CTCP Nông nghiệp Quốc tế Hoàng Anh Gia Lai năm 2023."
    ) == ("HNG",)
