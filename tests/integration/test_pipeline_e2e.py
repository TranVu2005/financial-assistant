import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

from financial_report_qa.data.dataset_builder import (
    CELL_SCHEMA,
    DOCUMENT_SCHEMA,
    ISSUE_SCHEMA,
    SOURCE_TABLE_OCCURRENCE_SCHEMA,
    TABLE_SCHEMA,
    DatasetBuildConfig,
    build_dataset,
)
from financial_report_qa.data.inventory import InventoryResult
from financial_report_qa.data.manifests import write_manifest
from financial_report_qa.schemas.documents import DocumentRecord, stable_document_id


def _setup_fixture(tmp_path: Path) -> tuple[Path, Path]:
    snapshot_root = tmp_path / "snapshot"
    snapshot_root.mkdir()

    # Doc 1: Valid income statement table
    doc1_rel = "VCB/2024/Consolidated/report.txt"
    doc1_path = snapshot_root / doc1_rel
    doc1_path.parent.mkdir(parents=True)
    doc1_content = (
        "Báo cáo kết quả hoạt động kinh doanh\n"
        "Đơn vị tính: triệu đồng\n"
        "<table>\n"
        "<tr><th>Chỉ tiêu</th><th>2023</th><th>2024</th></tr>\n"
        "<tr><td>Doanh thu thuần về bán hàng và cung cấp dịch vụ</td>"
        "<td>(1.500)</td><td>2.000,50</td></tr>\n"
        "<tr><td>Lợi nhuận kế toán trước thuế</td><td>500</td><td>N/A</td></tr>\n"
        "</table>\n"
    )
    doc1_path.write_text(doc1_content, encoding="utf-8")
    bytes1 = doc1_path.read_bytes()
    digest1 = hashlib.sha256(bytes1).hexdigest()

    doc1_rec = DocumentRecord(
        doc_id=stable_document_id(digest1),
        repo_id="org/vifinqa",
        revision="rev-1",
        relative_path=doc1_rel,
        company_code="VCB",
        report_year=2024,
        statement_scope="consolidated",
        sha256=digest1,
        file_size_bytes=len(bytes1),
        encoding="utf-8",
        inventory_status="ready",
    )

    # Doc 2: Second document
    doc2_rel = "TCB/2023/Separate/report.txt"
    doc2_path = snapshot_root / doc2_rel
    doc2_path.parent.mkdir(parents=True)
    doc2_content = (
        "Bảng cân đối kế toán\n"
        "ĐVT: tỷ đồng\n"
        "<table>\n"
        "<tr><th>Chỉ tiêu</th><th>Năm 2023</th></tr>\n"
        "<tr><td>Tổng cộng tài sản</td><td>10.000</td></tr>\n"
        "</table>\n"
    )
    doc2_path.write_text(doc2_content, encoding="utf-8")
    bytes2 = doc2_path.read_bytes()
    digest2 = hashlib.sha256(bytes2).hexdigest()

    doc2_rec = DocumentRecord(
        doc_id=stable_document_id(digest2),
        repo_id="org/vifinqa",
        revision="rev-1",
        relative_path=doc2_rel,
        company_code="TCB",
        report_year=2023,
        statement_scope="separate",
        sha256=digest2,
        file_size_bytes=len(bytes2),
        encoding="utf-8",
        inventory_status="ready",
    )

    manifest_path = tmp_path / "documents.jsonl"
    write_manifest(
        InventoryResult(documents=(doc1_rec, doc2_rec), issues=()), manifest_path
    )

    return snapshot_root, manifest_path


def test_e2e_pipeline_builds_reproducible_release(tmp_path: Path) -> None:
    snapshot_root, manifest_path = _setup_fixture(tmp_path)

    processed_run1 = tmp_path / "processed_run1"
    config1 = DatasetBuildConfig(
        snapshot_root=snapshot_root,
        manifest_path=manifest_path,
        processed_root=processed_run1,
    )
    res1 = build_dataset(config1)

    assert res1.document_count == 2
    assert res1.table_count == 2
    assert res1.cell_count == 13
    assert res1.issue_count >= 1  # e.g., N/A number_missing

    # Read Parquet files and verify schemas
    rel_path = res1.release_path
    docs_tbl = pq.read_table(rel_path / "documents.parquet")  # type: ignore[no-untyped-call]
    tbls_tbl = pq.read_table(rel_path / "tables.parquet")  # type: ignore[no-untyped-call]
    cells_tbl = pq.read_table(rel_path / "cells.parquet")  # type: ignore[no-untyped-call]
    issues_tbl = pq.read_table(rel_path / "issues.parquet")  # type: ignore[no-untyped-call]

    assert DOCUMENT_SCHEMA.equals(docs_tbl.schema)
    assert TABLE_SCHEMA.equals(tbls_tbl.schema)
    assert CELL_SCHEMA.equals(cells_tbl.schema)
    assert ISSUE_SCHEMA.equals(issues_tbl.schema)

    # Run second time for byte-level reproducibility test
    processed_run2 = tmp_path / "processed_run2"
    config2 = DatasetBuildConfig(
        snapshot_root=snapshot_root,
        manifest_path=manifest_path,
        processed_root=processed_run2,
    )
    res2 = build_dataset(config2)

    assert res1.dataset_fingerprint == res2.dataset_fingerprint

    filenames = (
        "documents.parquet",
        "tables.parquet",
        "cells.parquet",
        "issues.parquet",
        "source_table_occurrences.parquet",
        "manifest.json",
    )
    for filename in filenames:
        bytes_run1 = (res1.release_path / filename).read_bytes()
        bytes_run2 = (res2.release_path / filename).read_bytes()
        assert bytes_run1 == bytes_run2, f"{filename} is not byte-identical across runs"

    # Verify source_table_occurrences contract (Task 3)
    occurrences = pq.read_table(
        rel_path / "source_table_occurrences.parquet"
    )  # type: ignore[no-untyped-call]
    assert SOURCE_TABLE_OCCURRENCE_SCHEMA.equals(occurrences.schema)
    assert occurrences.num_rows == 2
    assert set(occurrences.column("status").to_pylist()) == {"canonical"}

    manifest = json.loads((rel_path / "manifest.json").read_text("utf-8"))
    assert manifest["source_table_occurrence_counts"] == {
        "total": 2,
        "canonical": 2,
        "rejected": 0,
        "duplicate": 0,
    }
