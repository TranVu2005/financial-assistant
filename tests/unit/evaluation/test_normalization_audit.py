import json
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq

from financial_report_qa.data.dataset_builder import (
    CELL_SCHEMA,
    DOCUMENT_SCHEMA,
    ISSUE_SCHEMA,
    TABLE_SCHEMA,
)
from financial_report_qa.evaluation.normalization_audit import (
    SAMPLE_SCHEMA,
    AuditSamplingConfig,
    build_issue_sample,
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
