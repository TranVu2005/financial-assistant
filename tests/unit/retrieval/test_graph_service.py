from __future__ import annotations

from financial_report_qa.retrieval.contracts import (
    MetricLabelObservation,
    TableDocument,
    TableMetadata,
)
from financial_report_qa.retrieval.graph import build_graph
from financial_report_qa.retrieval.graph_service import TableGraphService


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


def _service(documents: tuple[TableDocument, ...]) -> TableGraphService:
    graph = build_graph(documents, dataset_fingerprint="f" * 64, release_lock_sha256="e" * 64)
    return TableGraphService(graph)


def test_neighbors_are_ordered_deterministically() -> None:
    # "b" and "c" are both line_gap=1 from "a" -> tied weight; "d" is far away.
    documents = (
        _document("a", doc_id="doc1", line_start=2),
        _document("b", doc_id="doc1", line_start=1),
        _document("c", doc_id="doc1", line_start=3),
        _document("d", doc_id="doc1", line_start=100),
    )
    service = _service(documents)
    edges = service.neighbors(_table_id("a"), relation="same_document")
    weights = [edge.weight for edge in edges]
    assert weights == sorted(weights, reverse=True)
    # Confirm the documented tie-break: identical weight then breaks on dst_table_id.
    tied = [edge for edge in edges if edge.weight == edges[0].weight]
    assert len(tied) == 2
    assert [edge.dst_table_id for edge in tied] == sorted(edge.dst_table_id for edge in tied)


def test_neighbors_respects_limit_without_changing_order() -> None:
    documents = tuple(
        _document(chr(ord("a") + i), doc_id="doc1", line_start=i + 1) for i in range(6)
    )
    service = _service(documents)
    full = service.neighbors(_table_id("a"), relation="same_document")
    limited = service.neighbors(_table_id("a"), relation="same_document", limit=2)
    assert limited == full[:2]


def test_shared_metric_weight_is_jaccard() -> None:
    documents = (
        _document("a", doc_id="doc1", metrics=("net_revenue", "total_assets")),
        _document("b", doc_id="doc2", metrics=("net_revenue",)),
    )
    service = _service(documents)
    edges = service.neighbors(_table_id("a"), relation="shared_metric")
    assert len(edges) == 1
    assert edges[0].weight == 1 / 2
    assert edges[0].evidence.shared_metrics == ("net_revenue",)


def test_same_document_weight_decays_with_line_gap() -> None:
    documents = (
        _document("a", doc_id="doc1", line_start=1),
        _document("b", doc_id="doc1", line_start=2),
        _document("c", doc_id="doc1", line_start=51),
    )
    service = _service(documents)
    edges = {
        edge.dst_table_id: edge
        for edge in service.neighbors(_table_id("a"), relation="same_document")
    }
    assert edges[_table_id("b")].weight > edges[_table_id("c")].weight
    assert edges[_table_id("b")].weight == 1.0 / (1 + 1)  # line_gap == 1
    assert edges[_table_id("c")].weight == 1.0 / (1 + 50)  # line_gap == 50


def test_evidence_records_the_reason_for_every_relation() -> None:
    documents = (
        _document(
            "a",
            doc_id="doc1",
            company_code="ACB",
            periods=("2024",),
            statement_type="balance_sheet",
            metrics=("total_assets",),
        ),
        _document(
            "b",
            doc_id="doc1",
            company_code="ACB",
            periods=("2024",),
            statement_type="notes",
        ),
        _document(
            "c",
            doc_id="doc2",
            company_code="ACB",
            periods=("2025",),
            statement_type="balance_sheet",
            metrics=("total_assets",),
        ),
        _document(
            "d",
            doc_id="doc3",
            company_code="ACB",
            periods=("2024",),
            statement_type="balance_sheet",
        ),
    )
    service = _service(documents)

    same_doc = next(edge for edge in service.neighbors(_table_id("a"), relation="same_document"))
    assert same_doc.evidence.doc_id == "doc1"
    assert same_doc.evidence.line_gap is not None

    shared_metric = next(
        edge for edge in service.neighbors(_table_id("a"), relation="shared_metric")
    )
    assert shared_metric.evidence.shared_metrics == ("total_assets",)

    adjacent = next(edge for edge in service.neighbors(_table_id("a"), relation="adjacent_period"))
    assert adjacent.evidence.period_pairs

    same_stmt = next(
        edge for edge in service.neighbors(_table_id("a"), relation="same_statement_type")
    )
    assert same_stmt.evidence.statement_type == "balance_sheet"

    explained = next(
        edge for edge in service.neighbors(_table_id("a"), relation="explained_by_note")
    )
    assert explained.evidence.src_statement_type == "balance_sheet"
    assert explained.evidence.statement_type == "notes"


def test_neighbors_for_unknown_table_id_raises() -> None:
    service = _service((_document("a", doc_id="doc1"),))
    try:
        service.neighbors(_table_id("9"))
    except ValueError as exc:
        assert "not present" in str(exc)
    else:
        raise AssertionError("expected ValueError")
