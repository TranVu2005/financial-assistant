import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from financial_report_qa.retrieval.contracts import (
    MetricLabelObservation,
    RetrievalFilters,
    TableDocument,
    TableMetadata,
)
from financial_report_qa.retrieval.index import (
    build_bm25_index,
    load_bm25_index,
    save_bm25_index,
    tokenize_text,
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
                periods=("2023", "2024"),
                statement_type="income",
                source_path="a.txt",
                line_start=1,
                line_end=3,
            ),
            metric_labels=(MetricLabelObservation(canonical="net_revenue", raw=None),),
        ),
        TableDocument(
            table_id=table_b,
            doc_id="doc_b",
            text="company_code: VIC\nperiod: 2024\nDoanh thu | 2024 | 200",
            metadata=TableMetadata(
                table_id=table_b,
                doc_id="doc_b",
                company_code="VIC",
                periods=("2024",),
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


def test_period_filter_ors_all_canonical_table_periods() -> None:
    service = RetrievalService(build_bm25_index(_documents(), dataset_fingerprint="f" * 64))

    trace = service.retrieve("doanh thu", filters=RetrievalFilters(periods=("2023", "2025")), k=10)

    assert [item.table_id for item in trace.results] == ["tbl_" + "a" * 64]
    assert trace.filter_decisions[0].matched_count_before_intersection == 1


def test_tokenize_text_uses_nfkc_casefold_and_regex_boundaries() -> None:
    expected = ("lợi", "nhuận", "vcb", "năm", "2023")

    assert tokenize_text("  LỢI NHUẬN—ＶＣＢ, năm 2023  ") == expected
    assert tokenize_text("Lợi nhuận VCB năm 2023") == expected


def test_empty_query_tokens_return_empty_without_padding() -> None:
    service = RetrievalService(build_bm25_index(_documents(), dataset_fingerprint="f" * 64))

    trace = service.retrieve("", filters=RetrievalFilters(), k=10)

    assert trace.results == ()


def test_out_of_vocabulary_query_returns_empty_without_zero_score_ranking() -> None:
    service = RetrievalService(build_bm25_index(_documents(), dataset_fingerprint="f" * 64))

    trace = service.retrieve("khongtontaitrongindex", filters=RetrievalFilters(), k=10)

    assert trace.results == ()


def test_retrieval_expands_metric_aliases_without_duplicate_query_tokens() -> None:
    target_id = "tbl_" + "c" * 64
    distractor_id = "tbl_" + "d" * 64
    profit_id = "tbl_" + "e" * 64
    documents = (
        TableDocument(
            table_id=target_id,
            doc_id="target",
            text="metrics: net revenue\nmetric aliases: doanh thu thuần",
            metadata=TableMetadata(
                table_id=target_id,
                doc_id="target",
                company_code="VGT",
                periods=(),
                source_path="target.txt",
                line_start=1,
                line_end=1,
            ),
            metric_labels=(MetricLabelObservation(canonical="net_revenue", raw="Doanh thu thuần"),),
        ),
        TableDocument(
            table_id=distractor_id,
            doc_id="distractor",
            text="doanh thu thuần doanh thu thuần operating profit",
            metadata=TableMetadata(
                table_id=distractor_id,
                doc_id="distractor",
                company_code="VGT",
                periods=(),
                source_path="distractor.txt",
                line_start=1,
                line_end=1,
            ),
            metric_labels=(MetricLabelObservation(canonical="operating_profit", raw=None),),
        ),
        TableDocument(
            table_id=profit_id,
            doc_id="profit",
            text="metrics: profit after tax",
            metadata=TableMetadata(
                table_id=profit_id,
                doc_id="profit",
                company_code="VGT",
                periods=(),
                source_path="profit.txt",
                line_start=1,
                line_end=1,
            ),
            metric_labels=(
                MetricLabelObservation(canonical="profit_after_tax", raw="profit after tax"),
            ),
        ),
    )
    service = RetrievalService(build_bm25_index(documents, dataset_fingerprint="f" * 64))

    trace = service.retrieve(
        "Doanh thu thuần",
        filters=RetrievalFilters(company_codes=("VGT",)),
        k=10,
    )

    assert trace.results[0].table_id == target_id, [
        (candidate.table_id, candidate.score) for candidate in trace.results
    ]
    assert trace.query_tokens.count("net") == 1
    assert trace.query_tokens.count("revenue") == 1
    assert trace.metric_expansions[0].canonical_metric == "net_revenue"

    boundary_trace = service.retrieve(
        "profit after taxation", filters=RetrievalFilters(company_codes=("VGT",)), k=10
    )
    assert boundary_trace.metric_expansions == ()

    oov_trace = service.retrieve(
        "unlistedmetric", filters=RetrievalFilters(company_codes=("VGT",)), k=10
    )
    assert oov_trace.empty_reason == "no_index_tokens"


def test_low_length_normalization_keeps_long_primary_table_above_short_fragment() -> None:
    note_id = "tbl_" + "a" * 64
    primary_id = "tbl_" + "b" * 64
    documents = (
        TableDocument(
            table_id=note_id,
            doc_id="note",
            text="title: total assets\nmetrics: total assets",
            metadata=TableMetadata(
                table_id=note_id,
                doc_id="note",
                company_code="HDB",
                periods=("2023",),
                source_path="note.txt",
                line_start=2000,
                line_end=2000,
            ),
        ),
        TableDocument(
            table_id=primary_id,
            doc_id="primary",
            text=(
                "title: total assets\nmetrics: total assets\nmetric aliases: total assets\n"
                + "context: detail"
            ),
            metadata=TableMetadata(
                table_id=primary_id,
                doc_id="primary",
                company_code="HDB",
                periods=("2023",),
                source_path="primary.txt",
                line_start=50,
                line_end=50,
            ),
        ),
    )
    service = RetrievalService(build_bm25_index(documents, dataset_fingerprint="f" * 64))

    trace = service.retrieve(
        "total assets",
        filters=RetrievalFilters(company_codes=("HDB",)),
        k=2,
    )

    assert [candidate.table_id for candidate in trace.results] == [primary_id, note_id]


def test_retrieval_rejects_nonfinite_score_outside_requested_top_k(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = build_bm25_index(_documents(), dataset_fingerprint="f" * 64)
    monkeypatch.setattr(
        index.retriever,
        "get_scores",
        lambda _tokens: np.asarray([10.0, np.nan], dtype=np.float32),
    )
    service = RetrievalService(index)

    with pytest.raises(ValueError, match="non-finite"):
        service.retrieve("doanh thu", filters=RetrievalFilters(), k=1)


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
    assert manifest["schema_version"] == "bm25-index-v3"
    assert manifest["builder_version"] == "v3"
    assert manifest["tokenizer_version"] == "v1"
    assert manifest["query_expansion_version"] == "v1"
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
    loaded = load_bm25_index(output_dir, release_lock_sha256=release_lock_sha256)
    assert loaded.documents[0].metric_labels == index.documents[0].metric_labels
    assert loaded.manifest.schema_version == "bm25-index-v3"
    assert loaded.manifest.query_expansion_version == "v1"


def test_loader_rejects_v2_manifest_before_reading_bm25_artifacts(tmp_path: Path) -> None:
    index = build_bm25_index(_documents(), dataset_fingerprint="f" * 64)
    output_dir = tmp_path / "index"
    save_bm25_index(index, output_dir)
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = "bm25-index-v2"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported BM25 index schema; rebuild the index"):
        load_bm25_index(output_dir)


def test_loader_rejects_a_v3_manifest_with_fields_from_a_different_build(
    tmp_path: Path,
) -> None:
    """A manifest declaring bm25-index-v3 but carrying fields this build's
    contract does not know (e.g. from another branch's builder) must fail
    with a message that names the fields and says to rebuild -- not a raw
    pydantic dump -- and must fail before any artifact bytes are read."""
    index = build_bm25_index(_documents(), dataset_fingerprint="f" * 64)
    output_dir = tmp_path / "index"
    save_bm25_index(index, output_dir)
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["release_table_count"] = 1
    manifest["release_tables_sha256"] = "a" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (output_dir / "documents.jsonl").write_bytes(b"corrupted")  # would break reads if reached

    with pytest.raises(ValueError, match="release_table_count, release_tables_sha256"):
        load_bm25_index(output_dir)


@pytest.mark.parametrize("manifest_payload", ([], None, "not-an-object"))
def test_loader_rejects_non_object_manifest_before_reading_artifacts(
    tmp_path: Path, manifest_payload: object
) -> None:
    index = build_bm25_index(_documents(), dataset_fingerprint="f" * 64)
    output_dir = tmp_path / "index"
    save_bm25_index(index, output_dir)
    (output_dir / "manifest.json").write_text(json.dumps(manifest_payload), encoding="utf-8")

    with pytest.raises(ValueError, match="BM25 index manifest must be a JSON object"):
        load_bm25_index(output_dir)


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
