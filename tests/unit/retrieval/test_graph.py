from __future__ import annotations

import json
from pathlib import Path

import pytest

from financial_report_qa.core.errors import GraphArtifactError
from financial_report_qa.retrieval.contracts import (
    MetricLabelObservation,
    TableDocument,
    TableMetadata,
)
from financial_report_qa.retrieval.graph import build_graph, load_graph, save_graph
from financial_report_qa.retrieval.graph_contracts import GraphEdge, GraphEvidence


def _table_id(token: str) -> str:
    return f"tbl_{token * 64}"


def _document(
    token: str,
    *,
    doc_id: str,
    company_code: str | None = "ACB",
    periods: tuple[str, ...] = ("2024",),
    statement_type: str | None = None,
    line_start: int = 1,
    metrics: tuple[str, ...] = (),
) -> TableDocument:
    return TableDocument(
        table_id=_table_id(token),
        doc_id=doc_id,
        text=f"table {token}",
        metadata=TableMetadata(
            table_id=_table_id(token),
            doc_id=doc_id,
            company_code=company_code,
            periods=periods,
            statement_type=statement_type,
            source_path=f"{token}.txt",
            line_start=line_start,
            line_end=line_start + 1,
        ),
        metric_labels=tuple(
            MetricLabelObservation(canonical=metric) for metric in sorted(set(metrics))
        ),
    )


def _documents() -> tuple[TableDocument, ...]:
    return (
        _document("a", doc_id="doc1", periods=("2024",), line_start=1, metrics=("net_revenue",)),
        _document("b", doc_id="doc1", periods=("2024",), line_start=5, metrics=("net_revenue",)),
        _document("c", doc_id="doc2", periods=("2025",), line_start=1, metrics=("total_assets",)),
        _document(
            "d",
            doc_id="doc1",
            periods=("2024",),
            statement_type="balance_sheet",
            line_start=10,
        ),
        _document("e", doc_id="doc1", periods=("2024",), statement_type="notes", line_start=20),
    )


def _spanning_years_documents() -> tuple[TableDocument, ...]:
    return (
        _document("a", doc_id="doc1", periods=("2023", "2024"), line_start=1),
        _document("b", doc_id="doc2", periods=("2024",), line_start=1),
    )


def test_build_graph_is_byte_identical_across_two_builds(tmp_path: Path) -> None:
    documents = _documents()
    first = build_graph(documents, dataset_fingerprint="f" * 64, release_lock_sha256="e" * 64)
    second = build_graph(documents, dataset_fingerprint="f" * 64, release_lock_sha256="e" * 64)
    target_a = tmp_path / "a"
    target_b = tmp_path / "b"
    save_graph(first, target_a)
    save_graph(second, target_b)
    assert (target_a / "buckets.jsonl").read_bytes() == (target_b / "buckets.jsonl").read_bytes()
    assert (target_a / "manifest.json").read_bytes() == (target_b / "manifest.json").read_bytes()


def test_no_relation_produces_a_self_loop() -> None:
    documents = _documents()
    graph = build_graph(documents, dataset_fingerprint="f" * 64, release_lock_sha256="e" * 64)
    for relation, buckets in graph.buckets.items():
        for key, positions in buckets.items():
            assert len(positions) == len(set(positions)), (relation, key)


def test_a_table_does_not_duplicate_its_own_position_within_one_bucket() -> None:
    """A table whose `periods` include two strings sharing a year (e.g.
    "2024" and "2024-12-31", real on the locked release for ~20,558 tables)
    or whose `metric_labels` carry the same canonical id under two raw
    labels (real for ~1,449 tables) must still contribute its position to
    each bucket exactly once."""
    duplicate_metric_document = TableDocument(
        table_id=_table_id("a"),
        doc_id="doc1",
        text="table a",
        metadata=TableMetadata(
            table_id=_table_id("a"),
            doc_id="doc1",
            company_code="ACB",
            periods=("2024", "2024-12-31"),
            source_path="a.txt",
            line_start=1,
            line_end=2,
        ),
        metric_labels=(
            MetricLabelObservation(canonical="net_revenue", raw="Doanh thu"),
            MetricLabelObservation(canonical="net_revenue", raw="Doanh thu thuần"),
        ),
    )
    graph = build_graph(
        (duplicate_metric_document,), dataset_fingerprint="f" * 64, release_lock_sha256="e" * 64
    )
    adjacent_bucket = graph.buckets["adjacent_period"][("ACB", "2024")]
    assert adjacent_bucket.count(0) == 1
    shared_metric_bucket = graph.buckets["shared_metric"][("ACB", "net_revenue")]
    assert shared_metric_bucket.count(0) == 1


def test_table_spanning_adjacent_years_is_not_its_own_neighbor() -> None:
    from financial_report_qa.retrieval.graph_service import TableGraphService

    documents = _spanning_years_documents()
    graph = build_graph(documents, dataset_fingerprint="f" * 64, release_lock_sha256="e" * 64)
    service = TableGraphService(graph)
    edges = service.neighbors(_table_id("a"), relation="adjacent_period")
    assert all(edge.dst_table_id != _table_id("a") for edge in edges)


def test_same_document_is_symmetric() -> None:
    from financial_report_qa.retrieval.graph_service import TableGraphService

    documents = _documents()
    graph = build_graph(documents, dataset_fingerprint="f" * 64, release_lock_sha256="e" * 64)
    service = TableGraphService(graph)
    forward = service.neighbors(_table_id("a"), relation="same_document")
    reverse = service.neighbors(_table_id("b"), relation="same_document")
    forward_ab = next(edge for edge in forward if edge.dst_table_id == _table_id("b"))
    reverse_ba = next(edge for edge in reverse if edge.dst_table_id == _table_id("a"))
    assert forward_ab.weight == reverse_ba.weight


def test_shared_metric_is_symmetric_with_equal_jaccard() -> None:
    from financial_report_qa.retrieval.graph_service import TableGraphService

    documents = _documents()
    graph = build_graph(documents, dataset_fingerprint="f" * 64, release_lock_sha256="e" * 64)
    service = TableGraphService(graph)
    forward = service.neighbors(_table_id("a"), relation="shared_metric")
    reverse = service.neighbors(_table_id("b"), relation="shared_metric")
    forward_ab = next(edge for edge in forward if edge.dst_table_id == _table_id("b"))
    reverse_ba = next(edge for edge in reverse if edge.dst_table_id == _table_id("a"))
    assert forward_ab.weight == reverse_ba.weight == 1.0


def test_adjacent_period_is_symmetric() -> None:
    from financial_report_qa.retrieval.graph_service import TableGraphService

    documents = (
        _document("a", doc_id="doc1", periods=("2023",), company_code="ACB"),
        _document("b", doc_id="doc2", periods=("2024",), company_code="ACB"),
    )
    graph = build_graph(documents, dataset_fingerprint="f" * 64, release_lock_sha256="e" * 64)
    service = TableGraphService(graph)
    forward = service.neighbors(_table_id("a"), relation="adjacent_period")
    reverse = service.neighbors(_table_id("b"), relation="adjacent_period")
    assert any(edge.dst_table_id == _table_id("b") for edge in forward)
    assert any(edge.dst_table_id == _table_id("a") for edge in reverse)


def test_explained_by_note_is_asymmetric() -> None:
    from financial_report_qa.retrieval.graph_service import TableGraphService

    documents = _documents()
    graph = build_graph(documents, dataset_fingerprint="f" * 64, release_lock_sha256="e" * 64)
    service = TableGraphService(graph)
    forward = service.neighbors(_table_id("d"), relation="explained_by_note")
    reverse = service.neighbors(_table_id("e"), relation="explained_by_note")
    assert any(edge.dst_table_id == _table_id("e") for edge in forward)
    assert reverse == ()


def test_notes_table_has_no_outgoing_explained_by_note() -> None:
    from financial_report_qa.retrieval.graph_service import TableGraphService

    documents = _documents()
    graph = build_graph(documents, dataset_fingerprint="f" * 64, release_lock_sha256="e" * 64)
    service = TableGraphService(graph)
    assert service.neighbors(_table_id("e"), relation="explained_by_note") == ()


def test_graph_edge_rejects_a_self_loop() -> None:
    with pytest.raises(ValueError, match="self-loop"):
        GraphEdge(
            src_table_id=_table_id("a"),
            dst_table_id=_table_id("a"),
            relation="same_document",
            weight=1.0,
            evidence=GraphEvidence(doc_id="doc1", line_gap=0),
        )


def test_graph_edge_rejects_evidence_missing_for_relation() -> None:
    with pytest.raises(ValueError, match="requires evidence fields"):
        GraphEdge(
            src_table_id=_table_id("a"),
            dst_table_id=_table_id("b"),
            relation="same_document",
            weight=1.0,
            evidence=GraphEvidence(),
        )


def test_save_graph_is_a_no_op_when_the_target_already_matches(tmp_path: Path) -> None:
    documents = _documents()
    target = tmp_path / "graph"
    save_graph(
        build_graph(documents, dataset_fingerprint="f" * 64, release_lock_sha256="e" * 64),
        target,
    )
    first_bytes = (target / "buckets.jsonl").read_bytes()

    save_graph(
        build_graph(documents, dataset_fingerprint="f" * 64, release_lock_sha256="e" * 64),
        target,
    )

    assert (target / "buckets.jsonl").read_bytes() == first_bytes


def test_save_graph_rejects_a_target_with_different_content(tmp_path: Path) -> None:
    target = tmp_path / "graph"
    save_graph(
        build_graph(_documents(), dataset_fingerprint="f" * 64, release_lock_sha256="e" * 64),
        target,
    )

    other_documents = (
        _document("1", doc_id="doc9", periods=("2024",)),
        _document("2", doc_id="doc9", periods=("2024",)),
    )
    other_graph = build_graph(
        other_documents, dataset_fingerprint="d" * 64, release_lock_sha256="e" * 64
    )

    with pytest.raises(GraphArtifactError, match="already exists with different content"):
        save_graph(other_graph, target)


def test_loader_rejects_a_v1_manifest_with_fields_from_a_different_build(
    tmp_path: Path,
) -> None:
    """A manifest declaring graph-v1 but carrying fields this build's contract
    does not know must fail with a message naming the fields, before any
    artifact bytes are read."""
    documents = _documents()
    graph = build_graph(documents, dataset_fingerprint="f" * 64, release_lock_sha256="e" * 64)
    output_dir = tmp_path / "graph"
    save_graph(graph, output_dir)
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["future_field"] = "unknown"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (output_dir / "buckets.jsonl").write_bytes(b"corrupted")  # would break reads if reached

    with pytest.raises(GraphArtifactError, match="future_field"):
        load_graph(output_dir, documents, release_lock_sha256="e" * 64)


def test_loader_rejects_a_release_lock_mismatch(tmp_path: Path) -> None:
    documents = _documents()
    graph = build_graph(documents, dataset_fingerprint="f" * 64, release_lock_sha256="e" * 64)
    output_dir = tmp_path / "graph"
    save_graph(graph, output_dir)

    with pytest.raises(GraphArtifactError, match="release lock"):
        load_graph(output_dir, documents, release_lock_sha256="d" * 64)


def test_build_graph_rejects_duplicate_table_ids() -> None:
    documents = (_document("a", doc_id="doc1"), _document("a", doc_id="doc2"))
    with pytest.raises(ValueError, match="unique table IDs"):
        build_graph(documents, dataset_fingerprint="f" * 64, release_lock_sha256="e" * 64)
