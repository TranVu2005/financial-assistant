# mypy: ignore-errors
import json
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from financial_report_qa.data.dataset_builder import (
    CELL_SCHEMA,
    DOCUMENT_SCHEMA,
    ISSUE_SCHEMA,
    TABLE_SCHEMA,
)
from financial_report_qa.evaluation.normalization_audit import (
    SAMPLE_SCHEMA,
    AuditComparison,
    AuditSamplingConfig,
    LabelRecord,
    QualityGateError,
    build_issue_sample,
    compare_releases,
    enforce_quality_gate,
    evaluate_labels,
    load_and_validate_labels,
)


def build_fixture_release(path: Path, reverse: bool = False) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    manifest_data = {
        "schema_version": "1",
        "dataset_fingerprint": "release-1",
        "source_manifest_sha256": "abcdef" * 10 + "abcd",
        "document_count": 2,
        "table_count": 2,
        "cell_count": 4,
        "issue_count": 4,
    }
    (path / "manifest.json").write_text(json.dumps(manifest_data), encoding="utf-8")

    docs = [
        {
            "doc_id": "doc_" + "1" * 64,
            "repo_id": "repo",
            "revision": "rev",
            "relative_path": "VCB/2024/doc1.txt",
            "company_code": "VCB",
            "report_year": 2024,
            "statement_scope": "consolidated",
            "sha256": "a" * 64,
            "file_size_bytes": 100,
            "encoding": "utf-8",
            "inventory_status": "ready",
            "ruleset_version": "2026.08",
            "normalization_fingerprint": "b" * 64,
        },
        {
            "doc_id": "doc_" + "2" * 64,
            "repo_id": "repo",
            "revision": "rev",
            "relative_path": "BID/2024/doc2.txt",
            "company_code": "BID",
            "report_year": 2024,
            "statement_scope": "consolidated",
            "sha256": "c" * 64,
            "file_size_bytes": 200,
            "encoding": "utf-8",
            "inventory_status": "ready",
            "ruleset_version": "2026.08",
            "normalization_fingerprint": "d" * 64,
        },
    ]
    tables = [
        {
            "table_id": "tbl_" + "1" * 64,
            "doc_id": "doc_" + "1" * 64,
            "title_raw": "Báo cáo kết quả kinh doanh",
            "statement_type": "income_statement",
            "unit_raw": "triệu đồng",
            "unit_normalized": "VND_million",
            "line_start": 10,
            "line_end": 20,
            "row_count": 2,
            "column_count": 2,
            "quality_score": 1.0,
            "csv_path": "tables/tbl1.csv",
        },
        {
            "table_id": "tbl_" + "2" * 64,
            "doc_id": "doc_" + "2" * 64,
            "title_raw": "Bảng cân đối kế toán",
            "statement_type": "balance_sheet",
            "unit_raw": "tỷ đồng",
            "unit_normalized": "VND_billion",
            "line_start": 30,
            "line_end": 40,
            "row_count": 2,
            "column_count": 2,
            "quality_score": 1.0,
            "csv_path": "tables/tbl2.csv",
        },
    ]
    cells = [
        {
            "cell_id": "cell_" + "1" * 64,
            "table_id": "tbl_" + "1" * 64,
            "row_idx": 1,
            "col_idx": 1,
            "row_label_raw": "Doanh thu",
            "row_label_canonical": "revenue",
            "column_label_raw": "Năm 2024",
            "column_label_canonical": "2024",
            "value_raw": "1000",
            "value_numeric": None,
            "period": "2024",
            "unit": None,
            "source_line_start": 12,
            "source_line_end": 12,
            "extraction_confidence": 1.0,
        },
        {
            "cell_id": "cell_" + "2" * 64,
            "table_id": "tbl_" + "1" * 64,
            "row_idx": 2,
            "col_idx": 1,
            "row_label_raw": "Chi phí",
            "row_label_canonical": "cost",
            "column_label_raw": "Năm 2024",
            "column_label_canonical": "2024",
            "value_raw": "500",
            "value_numeric": None,
            "period": "2024",
            "unit": None,
            "source_line_start": 13,
            "source_line_end": 13,
            "extraction_confidence": 1.0,
        },
        {
            "cell_id": "cell_" + "3" * 64,
            "table_id": "tbl_" + "2" * 64,
            "row_idx": 1,
            "col_idx": 1,
            "row_label_raw": "Tài sản",
            "row_label_canonical": "assets",
            "column_label_raw": "Năm 2024",
            "column_label_canonical": "2024",
            "value_raw": "2000",
            "value_numeric": None,
            "period": "2024",
            "unit": None,
            "source_line_start": 32,
            "source_line_end": 32,
            "extraction_confidence": 1.0,
        },
        {
            "cell_id": "cell_" + "4" * 64,
            "table_id": "tbl_" + "2" * 64,
            "row_idx": 2,
            "col_idx": 1,
            "row_label_raw": "Nợ",
            "row_label_canonical": "liabilities",
            "column_label_raw": "Năm 2024",
            "column_label_canonical": "2024",
            "value_raw": "800",
            "value_numeric": None,
            "period": "2024",
            "unit": None,
            "source_line_start": 33,
            "source_line_end": 33,
            "extraction_confidence": 1.0,
        },
    ]
    issues = [
        {
            "code": "unit_unknown",
            "doc_id": "doc_" + "1" * 64,
            "table_id": "tbl_" + "1" * 64,
            "cell_id": "cell_" + "1" * 64,
            "field": "unit",
            "raw_value": "Năm 2024",
        },
        {
            "code": "unit_unknown",
            "doc_id": "doc_" + "1" * 64,
            "table_id": "tbl_" + "1" * 64,
            "cell_id": "cell_" + "2" * 64,
            "field": "unit",
            "raw_value": "Năm 2024",
        },
        {
            "code": "unit_unknown",
            "doc_id": "doc_" + "2" * 64,
            "table_id": "tbl_" + "2" * 64,
            "cell_id": "cell_" + "3" * 64,
            "field": "unit",
            "raw_value": "Năm 2024",
        },
        {
            "code": "metric_unknown",
            "doc_id": "doc_" + "2" * 64,
            "table_id": "tbl_" + "2" * 64,
            "cell_id": "cell_" + "4" * 64,
            "field": "metric",
            "raw_value": "Nợ",
        },
    ]

    if reverse:
        docs.reverse()
        tables.reverse()
        cells.reverse()
        issues.reverse()

    write_table = cast(Any, pq.write_table)
    write_table(pa.Table.from_pylist(docs, schema=DOCUMENT_SCHEMA), path / "documents.parquet")
    write_table(pa.Table.from_pylist(tables, schema=TABLE_SCHEMA), path / "tables.parquet")
    write_table(pa.Table.from_pylist(cells, schema=CELL_SCHEMA), path / "cells.parquet")
    write_table(pa.Table.from_pylist(issues, schema=ISSUE_SCHEMA), path / "issues.parquet")

    return path


def test_sample_is_independent_of_input_order(tmp_path: Path) -> None:
    a = build_fixture_release(tmp_path / "a", reverse=False)
    b = build_fixture_release(tmp_path / "b", reverse=True)
    config = AuditSamplingConfig(
        issue_limits={"unit_unknown": 3, "metric_unknown": 2},
        max_per_stratum=1,
        seed="normalization-audit-v1",
    )
    left = build_issue_sample(a, "release-1", config)
    right = build_issue_sample(b, "release-1", config)
    assert left.schema == SAMPLE_SCHEMA
    assert left.to_pylist() == right.to_pylist()


def test_stratified_sampling_respects_stratum_cap_and_retains_rare_issues(
    tmp_path: Path,
) -> None:
    path = build_fixture_release(tmp_path / "skewed")
    config = AuditSamplingConfig(
        issue_limits={"unit_unknown": 5, "metric_unknown": 5},
        max_per_stratum=1,
        seed="normalization-audit-v1",
    )
    sample = build_issue_sample(path, "release-1", config)
    rows = sample.to_pylist()
    counts_by_code: dict[str, int] = {}
    for r in rows:
        code = str(r["issue_code"])
        counts_by_code[code] = counts_by_code.get(code, 0) + 1

    # unit_unknown has 3 rows in fixture, but 2 share the exact same stratum_key
    # With max_per_stratum=1, only 1 from that stratum + 1 from the other = 2 total.
    assert counts_by_code.get("unit_unknown") == 2

    # metric_unknown has 1 row in fixture, limit is 5 -> all retained.
    assert counts_by_code.get("metric_unknown") == 1


@pytest.mark.parametrize(
    "csv_content,match_msg",
    [
        ("sample_id,label,cause_code\nunknown_id,true_issue,ocr_corruption\n", "unknown sample_id"),
        (
            "sample_id,label,cause_code\n{valid_id},true_issue,ocr_corruption\n{valid_id},false_positive,other\n",
            "duplicate sample_id",
        ),
        ("sample_id,label,cause_code\n{valid_id},bad_label,ocr_corruption\n", "invalid label"),
        ("sample_id,label,cause_code\n{valid_id},true_issue,bad_cause\n", "invalid cause_code"),
    ],
)
def test_load_and_validate_labels_rejects_invalid_inputs(
    tmp_path: Path, csv_content: str, match_msg: str
) -> None:
    path = build_fixture_release(tmp_path / "labels_test")
    config = AuditSamplingConfig()
    sample = build_issue_sample(path, "release-1", config)
    valid_id = sample.to_pylist()[0]["sample_id"]
    csv_path = tmp_path / "labels.csv"
    csv_path.write_text(csv_content.format(valid_id=valid_id), encoding="utf-8")

    with pytest.raises(ValueError, match=match_msg):
        load_and_validate_labels(sample, csv_path)


def test_evaluate_labels_computes_exact_metrics_and_coverage(tmp_path: Path) -> None:
    # Build a mock sample table with 4 rows of unit_unknown
    mock_sample = pa.Table.from_pylist(
        [
            {
                "sample_id": f"s{i}",
                "release_fingerprint": "fp",
                "issue_code": "unit_unknown",
                "field": "unit",
                "raw_value": "test",
                "doc_id": "doc_1",
                "table_id": "tbl_1",
                "cell_id": f"cell_{i}",
                "company_code": "VCB",
                "report_year": 2024,
                "statement_type": "income_statement",
                "table_title_raw": "",
                "table_unit_raw": "",
                "row_label_raw": "",
                "column_label_raw": "",
                "value_raw": "",
                "source_line_start": 1,
                "source_line_end": 1,
                "stratum_key": f"s{i}",
                "selection_rank": f"r{i}",
            }
            for i in range(1, 5)
        ],
        schema=SAMPLE_SCHEMA,
    )

    labels = (
        LabelRecord(sample_id="s1", label="true_issue", cause_code="ocr_corruption"),
        LabelRecord(sample_id="s2", label="false_positive", cause_code="year_header_as_unit"),
        LabelRecord(sample_id="s3", label="uncertain", cause_code="other"),
        # s4 is unlabeled
    )
    metrics_by_code = evaluate_labels(mock_sample, labels)
    m = metrics_by_code["unit_unknown"]
    assert m.sample_count == 4
    assert m.true_issue_count == 1
    assert m.false_positive_count == 1
    assert m.uncertain_count == 1
    assert m.unlabeled_count == 1
    assert m.conclusive_coverage == Decimal("0.5")
    assert m.false_positive_rate == Decimal("0.5")
    assert m.cause_counts["ocr_corruption"] == 1
    assert m.cause_counts["year_header_as_unit"] == 1
    assert m.cause_counts["other"] == 1


def test_compare_releases_fails_on_changed_canonical_table_ids(tmp_path: Path) -> None:
    before = build_fixture_release(tmp_path / "before")
    after = build_fixture_release(tmp_path / "after")

    # Modify table_id in after to cause a mismatch
    after_tables = pq.read_table(after / "tables.parquet").to_pylist()
    after_tables[0]["table_id"] = "tbl_" + "9" * 64
    pq.write_table(
        pa.Table.from_pylist(after_tables, schema=TABLE_SCHEMA), after / "tables.parquet"
    )

    # Sample and labels files
    sample_path = tmp_path / "sample.parquet"
    pq.write_table(pa.Table.from_pylist([], schema=SAMPLE_SCHEMA), sample_path)
    labels_path = tmp_path / "labels.csv"
    labels_path.write_text("sample_id,label,cause_code\n", encoding="utf-8")

    with pytest.raises(QualityGateError, match="canonical table IDs changed"):
        compare_releases(before, after, sample_path, labels_path)


def test_compare_releases_fails_on_unresolved_sample_context(tmp_path: Path) -> None:
    before = build_fixture_release(tmp_path / "before")
    after = build_fixture_release(tmp_path / "after")

    # Write sample with 1 item
    sample_row = {
        "sample_id": "s1",
        "release_fingerprint": "release-1",
        "issue_code": "unit_unknown",
        "field": "unit",
        "raw_value": "test",
        "doc_id": "doc_1",
        "table_id": "tbl_1",
        "cell_id": "cell_" + "1" * 64,
        "company_code": "VCB",
        "report_year": 2024,
        "statement_type": "income_statement",
        "table_title_raw": "",
        "table_unit_raw": "",
        "row_label_raw": "Doanh thu",
        "column_label_raw": "Năm 2024",
        "value_raw": "1000",
        "source_line_start": 1,
        "source_line_end": 1,
        "stratum_key": "s1",
        "selection_rank": "r1",
    }
    sample_path = tmp_path / "sample.parquet"
    pq.write_table(pa.Table.from_pylist([sample_row], schema=SAMPLE_SCHEMA), sample_path)

    # Empty labels file -> sample s1 is unresolved
    labels_path = tmp_path / "labels.csv"
    labels_path.write_text("sample_id,label,cause_code\n", encoding="utf-8")

    with pytest.raises(QualityGateError, match="unresolved sample context"):
        compare_releases(before, after, sample_path, labels_path)


def test_quality_gate_checks_table_count() -> None:
    comparison = AuditComparison(
        before_fingerprint="fp1",
        after_fingerprint="fp2",
        before_table_count=100,
        after_table_count=100,  # not 146011
        before_issue_count=10,
        after_issue_count=5,
        coverage=Decimal("0.95"),
        false_positive_rate=Decimal("0.02"),
        passed=True,
        errors=(),
    )
    with pytest.raises(QualityGateError, match="table count not equal to 146,011"):
        enforce_quality_gate(comparison)


def test_quality_gate_checks_coverage() -> None:
    comparison = AuditComparison(
        before_fingerprint="fp1",
        after_fingerprint="fp2",
        before_table_count=146011,
        after_table_count=146011,
        before_issue_count=10,
        after_issue_count=5,
        coverage=Decimal("0.85"),  # < 0.90
        false_positive_rate=Decimal("0.02"),
        passed=True,
        errors=(),
    )
    with pytest.raises(QualityGateError, match="coverage below 0.90"):
        enforce_quality_gate(comparison)


def test_quality_gate_checks_false_positive_rate() -> None:
    comparison = AuditComparison(
        before_fingerprint="fp1",
        after_fingerprint="fp2",
        before_table_count=146011,
        after_table_count=146011,
        before_issue_count=10,
        after_issue_count=5,
        coverage=Decimal("0.95"),
        false_positive_rate=Decimal("0.08"),  # > 0.05
        passed=True,
        errors=(),
    )
    with pytest.raises(QualityGateError, match="false-positive rate above 0.05"):
        enforce_quality_gate(comparison)


def test_quality_gate_success() -> None:
    comparison = AuditComparison(
        before_fingerprint="fp1",
        after_fingerprint="fp2",
        before_table_count=146011,
        after_table_count=146011,
        before_issue_count=10,
        after_issue_count=5,
        coverage=Decimal("0.95"),
        false_positive_rate=Decimal("0.02"),
        passed=True,
        errors=(),
    )
    enforce_quality_gate(comparison)


def test_compare_releases_fails_on_missing_source_context(tmp_path: Path) -> None:
    before = build_fixture_release(tmp_path / "before")
    after = build_fixture_release(tmp_path / "after")

    # Remove a cell from the after release
    after_cells = pq.read_table(after / "cells.parquet").to_pylist()
    del after_cells[0]
    pq.write_table(pa.Table.from_pylist(after_cells, schema=CELL_SCHEMA), after / "cells.parquet")

    sample_row = {
        "sample_id": "s1",
        "release_fingerprint": "release-1",
        "issue_code": "unit_unknown",
        "field": "unit",
        "raw_value": "test",
        "doc_id": "doc_1",
        "table_id": "tbl_1",
        "cell_id": "cell_" + "1" * 64,  # First cell from fixture
        "company_code": "VCB",
        "report_year": 2024,
        "statement_type": "income_statement",
        "table_title_raw": "",
        "table_unit_raw": "",
        "row_label_raw": "Doanh thu",
        "column_label_raw": "Năm 2024",
        "value_raw": "1000",
        "source_line_start": 1,
        "source_line_end": 1,
        "stratum_key": "s1",
        "selection_rank": "r1",
    }
    sample_path = tmp_path / "sample.parquet"
    pq.write_table(pa.Table.from_pylist([sample_row], schema=SAMPLE_SCHEMA), sample_path)

    labels_path = tmp_path / "labels.csv"
    labels_path.write_text(
        "sample_id,label,cause_code\ns1,true_issue,ocr_corruption\n", encoding="utf-8"
    )

    with pytest.raises(QualityGateError, match="missing or changed source context"):
        compare_releases(before, after, sample_path, labels_path)


def test_compare_releases_fails_on_changed_source_context(tmp_path: Path) -> None:
    before = build_fixture_release(tmp_path / "before")
    after = build_fixture_release(tmp_path / "after")

    # Change a cell's raw value in the after release
    after_cells = pq.read_table(after / "cells.parquet").to_pylist()
    after_cells[0]["value_raw"] = "9999"
    pq.write_table(pa.Table.from_pylist(after_cells, schema=CELL_SCHEMA), after / "cells.parquet")

    sample_row = {
        "sample_id": "s1",
        "release_fingerprint": "release-1",
        "issue_code": "unit_unknown",
        "field": "unit",
        "raw_value": "test",
        "doc_id": "doc_1",
        "table_id": "tbl_1",
        "cell_id": "cell_" + "1" * 64,  # First cell from fixture
        "company_code": "VCB",
        "report_year": 2024,
        "statement_type": "income_statement",
        "table_title_raw": "",
        "table_unit_raw": "",
        "row_label_raw": "Doanh thu",
        "column_label_raw": "Năm 2024",
        "value_raw": "1000",
        "source_line_start": 1,
        "source_line_end": 1,
        "stratum_key": "s1",
        "selection_rank": "r1",
    }
    sample_path = tmp_path / "sample.parquet"
    pq.write_table(pa.Table.from_pylist([sample_row], schema=SAMPLE_SCHEMA), sample_path)

    labels_path = tmp_path / "labels.csv"
    labels_path.write_text(
        "sample_id,label,cause_code\ns1,true_issue,ocr_corruption\n", encoding="utf-8"
    )

    with pytest.raises(QualityGateError, match="missing or changed source context"):
        compare_releases(before, after, sample_path, labels_path)
