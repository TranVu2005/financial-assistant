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
from financial_report_qa.evaluation.normalization_audit_cli import main


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


def test_normalization_audit_sample_and_baseline_cli(tmp_path: Path) -> None:
    release_dir = build_fixture_release(tmp_path / "release")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "seed: normalization-audit-v1\nmax_per_stratum: 5\nissue_limits:\n"
        "  unit_unknown: 200\n  metric_unknown: 200\n",
        encoding="utf-8",
    )

    sample_path = tmp_path / "output" / "sample.parquet"
    assert (
        main(
            [
                "sample",
                "--release",
                str(release_dir),
                "--output",
                str(sample_path),
                "--config",
                str(config_path),
            ]
        )
        == 0
    )
    assert sample_path.is_file()
    first_bytes = sample_path.read_bytes()

    assert (
        main(
            [
                "sample",
                "--release",
                str(release_dir),
                "--output",
                str(sample_path),
                "--config",
                str(config_path),
            ]
        )
        == 0
    )
    second_bytes = sample_path.read_bytes()
    assert first_bytes == second_bytes

    labels_path = tmp_path / "labels.csv"
    read_table = cast(Any, pq.read_table)
    table = read_table(sample_path)
    rows = table.to_pylist()
    csv_lines = ["sample_id,label,cause_code,reviewer_note"]
    for r in rows:
        csv_lines.append(f"{r['sample_id']},true_issue,ocr_corruption,test note")
    labels_path.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

    report_dir = tmp_path / "reports"
    assert (
        main(
            [
                "baseline",
                "--sample",
                str(sample_path),
                "--labels",
                str(labels_path),
                "--output-dir",
                str(report_dir),
            ]
        )
        == 0
    )

    baseline_json = report_dir / "baseline.json"
    baseline_md = report_dir / "baseline.md"
    assert baseline_json.is_file()
    assert baseline_md.is_file()

    data = json.loads(baseline_json.read_text(encoding="utf-8"))
    assert data["release_fingerprint"] == "release-1"
    assert "metrics_by_issue" in data
    assert "unit_unknown" in data["metrics_by_issue"]
    unit_unknown_metrics = data["metrics_by_issue"]["unit_unknown"]
    assert float(unit_unknown_metrics["false_positive_rate"]) <= 0.10
    true_rate = float(unit_unknown_metrics["true_issue_count"]) / float(
        unit_unknown_metrics["sample_count"]
    )
    assert true_rate >= 0.90
    md_content = baseline_md.read_text(encoding="utf-8")
    assert "| `unit_unknown` | 3 | 3 | 0 | 0 | 0 | 1.0000 | 0.0000 |" in md_content


