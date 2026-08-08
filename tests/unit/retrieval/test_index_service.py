import hashlib
import json
from pathlib import Path

import pytest

from financial_report_qa.retrieval.contracts import RetrievalFilters, TableDocument, TableMetadata
from financial_report_qa.retrieval.index import (
    build_bm25_index,
    load_bm25_index,
    save_bm25_index,
)
from financial_report_qa.retrieval.service import RetrievalService


def _documents() -> tuple[TableDocument, ...]:
    table_a = "tbl_" + "a" * 64
    table_b = "tbl_" + "b" * 64
    return (
        TableDocument(
            table_id=table_a,
            doc_id="doc_a",
            text="company_code: ACB\nperiod: 2024\nDoanh thu | 2024 | 100",
            metadata=TableMetadata(
                table_id=table_a,
                doc_id="doc_a",
                company_code="ACB",
                period="2024",
                statement_type="income",
                source_path="a.txt",
                line_start=1,
                line_end=3,
            ),
        ),
        TableDocument(
            table_id=table_b,
            doc_id="doc_b",
            text="company_code: VIC\nperiod: 2024\nDoanh thu | 2024 | 200",
            metadata=TableMetadata(
                table_id=table_b,
                doc_id="doc_b",
                company_code="VIC",
                period="2024",
                statement_type="income",
                source_path="v.txt",
                line_start=1,
                line_end=3,
            ),
        ),
    )


def test_filter_first_never_returns_ineligible_document() -> None:
    service = RetrievalService(build_bm25_index(_documents(), dataset_fingerprint="f" * 64))

    trace = service.retrieve("doanh thu", filters=RetrievalFilters(company_codes=("ACB",)), k=10)

    assert [item.table_id for item in trace.results] == ["tbl_" + "a" * 64]
    assert trace.eligible_count == 1
    assert trace.filter_decisions[0].field == "company_codes"
    assert trace.filter_decisions[0].matched_count_before_intersection == 1


def test_empty_query_tokens_return_empty_without_padding() -> None:
    service = RetrievalService(build_bm25_index(_documents(), dataset_fingerprint="f" * 64))

    trace = service.retrieve("", filters=RetrievalFilters(), k=10)

    assert trace.results == ()


def test_out_of_vocabulary_query_returns_empty_without_zero_score_ranking() -> None:
    service = RetrievalService(build_bm25_index(_documents(), dataset_fingerprint="f" * 64))

    trace = service.retrieve("khongtontaitrongindex", filters=RetrievalFilters(), k=10)

    assert trace.results == ()


def test_persisted_index_manifest_hashes_exact_emitted_artifacts(tmp_path: Path) -> None:
    release_lock_sha256 = "e" * 64
    index = build_bm25_index(
        _documents(),
        dataset_fingerprint="f" * 64,
        release_lock_sha256=release_lock_sha256,
    )
    output_dir = tmp_path / "index"

    save_bm25_index(index, output_dir)

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "bm25-index-v1"
    assert manifest["builder_version"] == "v1"
    assert manifest["tokenizer_version"] == "v1"
    assert manifest["dtype"] == "float32"
    assert manifest["release_lock_sha256"] == release_lock_sha256
    documents_bytes = (output_dir / "documents.jsonl").read_bytes()
    assert manifest["document_sha256"] == hashlib.sha256(documents_bytes).hexdigest()
    artifact_paths = {
        path.relative_to(output_dir).as_posix()
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    assert set(manifest["artifact_sha256"]) == artifact_paths
    for relative_path, expected_hash in manifest["artifact_sha256"].items():
        actual_hash = hashlib.sha256((output_dir / relative_path).read_bytes()).hexdigest()
        assert actual_hash == expected_hash


def test_loader_rejects_artifact_corruption_before_bm25_load(tmp_path: Path) -> None:
    index = build_bm25_index(
        _documents(), dataset_fingerprint="f" * 64, release_lock_sha256="e" * 64
    )
    output_dir = tmp_path / "index"
    save_bm25_index(index, output_dir)
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    bm25_artifact = next(
        relative_path
        for relative_path in manifest["artifact_sha256"]
        if relative_path.startswith("bm25s/")
    )
    artifact_path = output_dir / bm25_artifact
    artifact_path.write_bytes(artifact_path.read_bytes() + b"corrupt")

    with pytest.raises(ValueError, match="artifact hash"):
        load_bm25_index(output_dir, release_lock_sha256="e" * 64)


def test_loader_rejects_byte_changes_to_documents_jsonl(tmp_path: Path) -> None:
    index = build_bm25_index(_documents(), dataset_fingerprint="f" * 64)
    output_dir = tmp_path / "index"
    save_bm25_index(index, output_dir)
    documents_path = output_dir / "documents.jsonl"
    documents_path.write_bytes(documents_path.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="artifact hash"):
        load_bm25_index(output_dir)


def test_existing_corrupt_target_is_rejected(tmp_path: Path) -> None:
    index = build_bm25_index(_documents(), dataset_fingerprint="f" * 64)
    output_dir = tmp_path / "index"
    save_bm25_index(index, output_dir)
    documents_path = output_dir / "documents.jsonl"
    documents_path.write_bytes(documents_path.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="artifact hash"):
        save_bm25_index(index, output_dir)


def test_loader_rejects_release_lock_hash_mismatch(tmp_path: Path) -> None:
    index = build_bm25_index(
        _documents(), dataset_fingerprint="f" * 64, release_lock_sha256="e" * 64
    )
    output_dir = tmp_path / "index"
    save_bm25_index(index, output_dir)

    with pytest.raises(ValueError, match="release lock"):
        load_bm25_index(output_dir, release_lock_sha256="d" * 64)
