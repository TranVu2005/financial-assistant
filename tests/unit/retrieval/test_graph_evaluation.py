from __future__ import annotations

from pathlib import Path

from financial_report_qa.retrieval.contracts import (
    MetricLabelObservation,
    TableDocument,
    TableMetadata,
)
from financial_report_qa.retrieval.graph import build_graph
from financial_report_qa.retrieval.graph_evaluation import (
    deterministic_projection,
    evaluate_graph_coverage,
    write_day11_graph,
)


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
        _document("a", doc_id="doc1", line_start=1, metrics=("net_revenue",)),
        _document("b", doc_id="doc1", line_start=2, metrics=("net_revenue",)),
        _document("c", doc_id="doc2", line_start=1),  # isolated: unique doc, no metrics
    )


def test_coverage_counts_isolated_nodes() -> None:
    documents = _documents()
    graph = build_graph(documents, dataset_fingerprint="f" * 64, release_lock_sha256="e" * 64)
    report = evaluate_graph_coverage(graph, documents)
    same_document = report.by_relation["same_document"]
    assert same_document.isolated_nodes == 1  # only "c" has no same-document sibling
    assert report.nodes_with_no_edge_in_any_relation == 1


def test_degree_percentiles_are_nearest_rank() -> None:
    # 3 nodes sharing one same_document bucket -> each has degree 2.
    documents = (
        _document("a", doc_id="doc1", line_start=1),
        _document("b", doc_id="doc1", line_start=2),
        _document("c", doc_id="doc1", line_start=3),
    )
    graph = build_graph(documents, dataset_fingerprint="f" * 64, release_lock_sha256="e" * 64)
    report = evaluate_graph_coverage(graph, documents)
    same_document = report.by_relation["same_document"]
    assert same_document.degree_p50 == 2
    assert same_document.degree_p95 == 2
    assert same_document.degree_max == 2


def test_excluded_relations_are_reported_with_measured_counts() -> None:
    documents = _documents()
    graph = build_graph(documents, dataset_fingerprint="f" * 64, release_lock_sha256="e" * 64)
    report = evaluate_graph_coverage(graph, documents)
    names = {excluded.name for excluded in report.excluded_relations}
    assert names == {"same_company", "same_period"}
    for excluded in report.excluded_relations:
        assert excluded.measured_pair_count >= 0
        assert "eligible pool" in excluded.reason


def test_deterministic_projection_is_stable() -> None:
    documents = _documents()
    graph = build_graph(documents, dataset_fingerprint="f" * 64, release_lock_sha256="e" * 64)
    report = evaluate_graph_coverage(graph, documents)
    assert deterministic_projection(report) == deterministic_projection(report)


def test_write_day11_graph_round_trips_json_and_markdown(tmp_path: Path) -> None:
    documents = _documents()
    graph = build_graph(documents, dataset_fingerprint="f" * 64, release_lock_sha256="e" * 64)
    report = evaluate_graph_coverage(graph, documents)
    json_path, markdown_path = write_day11_graph(report, tmp_path)
    assert json_path.exists()
    assert markdown_path.exists()
    assert json_path.name == f"retrieval-day11-graph-{'f' * 12}.json"
    assert "Day 11 Graph Coverage" in markdown_path.read_text(encoding="utf-8")
