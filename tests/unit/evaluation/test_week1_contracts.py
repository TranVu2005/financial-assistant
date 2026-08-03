from pathlib import Path

import pytest
from pydantic import ValidationError

from financial_report_qa.core.errors import Week1GateInputError
from financial_report_qa.evaluation.week1_contracts import (
    EXPECTED_TABLE_COLUMNS,
    ExpectedTable,
    PilotDocument,
    read_csv_rows,
    stable_annotation_id,
    write_csv_rows,
)

DOC_ID = f"doc_{'a' * 64}"


def test_stable_annotation_id_uses_exact_canonical_payload() -> None:
    first = stable_annotation_id(DOC_ID, 10, 20, "balance_sheet")
    second = stable_annotation_id(DOC_ID, 10, 20, "balance_sheet")
    changed = stable_annotation_id(DOC_ID, 10, 21, "balance_sheet")
    assert first == second
    assert first.startswith("ann_")
    assert len(first) == 68
    assert changed != first


def test_expected_table_requires_derived_id_sorted_periods_and_positive_shape() -> None:
    annotation_id = stable_annotation_id(DOC_ID, 10, 20, "balance_sheet")
    expected = ExpectedTable(
        annotation_schema_version="1",
        annotation_id=annotation_id,
        doc_id=DOC_ID,
        relative_path="VCB/2024/Consolidated/report.txt",
        statement_type="balance_sheet",
        line_start=10,
        line_end=20,
        row_count=5,
        column_count=3,
        unit_normalized="VND_million",
        expected_periods=("2023", "2024"),
        notes="",
    )
    assert expected.expected_periods == ("2023", "2024")

    invalid_payload = expected.model_dump()
    invalid_payload["expected_periods"] = ("2024", "2023")
    with pytest.raises(ValidationError, match="sorted and duplicate-free"):
        ExpectedTable.model_validate(invalid_payload)


def test_pilot_document_rejects_unsafe_relative_path() -> None:
    with pytest.raises(ValidationError, match="safe POSIX"):
        PilotDocument(
            annotation_schema_version="1",
            dataset_fingerprint="b" * 64,
            source_manifest_sha256="c" * 64,
            doc_id=DOC_ID,
            relative_path="../report.txt",
            company_code="VCB",
            report_year=2024,
            statement_scope="consolidated",
        )


def test_expected_table_csv_is_byte_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "expected-tables.csv"
    row = {
        "annotation_schema_version": "1",
        "annotation_id": stable_annotation_id(DOC_ID, 10, 20, "balance_sheet"),
        "doc_id": DOC_ID,
        "relative_path": "VCB/2024/Consolidated/report.txt",
        "statement_type": "balance_sheet",
        "line_start": "10",
        "line_end": "20",
        "row_count": "5",
        "column_count": "3",
        "unit_normalized": "VND_million",
        "expected_periods": "2023|2024",
        "notes": "Unicode: kiểm toán",
    }
    write_csv_rows(path, EXPECTED_TABLE_COLUMNS, (row,))
    first = path.read_bytes()
    write_csv_rows(path, EXPECTED_TABLE_COLUMNS, (row,), allow_identical=True)
    assert path.read_bytes() == first
    assert read_csv_rows(path, EXPECTED_TABLE_COLUMNS) == (row,)


@pytest.mark.parametrize(
    "raw",
    [
        b"wrong,header\n",
        b"annotation_schema_version,annotation_schema_version\n",
        b"annotation_schema_version\r\n",
        b"annotation_schema_version",
    ],
)
def test_csv_reader_rejects_contract_drift(tmp_path: Path, raw: bytes) -> None:
    path = tmp_path / "bad.csv"
    path.write_bytes(raw)
    with pytest.raises(Week1GateInputError):
        read_csv_rows(path, EXPECTED_TABLE_COLUMNS)
