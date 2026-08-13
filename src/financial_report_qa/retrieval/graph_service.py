"""Deterministic neighbor queries over a built GTR-lite table graph.

`neighbors()` turns a bucket lookup into `GraphEdge`s on demand -- no fan-out
cap is applied here (`limit=None` by default). Bounding fan-out is Day 12's
job (`plan.md:777`); this module only guarantees that whatever `limit` Day
12 chooses is applied to a stable, reproducible ordering.
"""

from __future__ import annotations

from financial_report_qa.retrieval.graph import TableGraph
from financial_report_qa.retrieval.graph_contracts import (
    GraphEdge,
    GraphEvidence,
    GraphRelation,
)

_NOTE_STATEMENT_TYPE = "notes"
_TYPED_STATEMENT_TYPES = (
    "balance_sheet",
    "cash_flow_statement",
    "equity_changes",
    "income_statement",
)


def _years(metadata_periods: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(sorted({period[:4] for period in metadata_periods}))


class TableGraphService:
    """Resolve GTR-lite neighbors for one already-built `TableGraph`."""

    def __init__(self, graph: TableGraph) -> None:
        self._graph = graph
        self._position_by_table_id = {
            document.table_id: position for position, document in enumerate(graph.documents)
        }

    def neighbors(
        self,
        table_id: str,
        *,
        relation: GraphRelation | None = None,
        limit: int | None = None,
    ) -> tuple[GraphEdge, ...]:
        if table_id not in self._position_by_table_id:
            raise ValueError(f"table_id {table_id!r} is not present in this graph")
        if limit is not None and limit < 0:
            raise ValueError("limit must be non-negative")
        source = self._position_by_table_id[table_id]
        relations = (relation,) if relation is not None else tuple(self._graph.buckets)
        edges: list[GraphEdge] = []
        for one_relation in relations:
            edges.extend(self._relation_neighbors(source, one_relation))
        edges.sort(key=lambda edge: (-edge.weight, edge.dst_table_id))
        return tuple(edges) if limit is None else tuple(edges[:limit])

    def _relation_neighbors(self, source: int, relation: GraphRelation) -> list[GraphEdge]:
        if relation == "same_document":
            return self._same_document_neighbors(source)
        if relation == "shared_metric":
            return self._shared_metric_neighbors(source)
        if relation == "adjacent_period":
            return self._adjacent_period_neighbors(source)
        if relation == "same_statement_type":
            return self._same_statement_type_neighbors(source)
        if relation == "explained_by_note":
            return self._explained_by_note_neighbors(source)
        raise AssertionError(f"unhandled graph relation: {relation}")

    def _same_document_neighbors(self, source: int) -> list[GraphEdge]:
        documents = self._graph.documents
        src_metadata = documents[source].metadata
        bucket = self._graph.buckets.get("same_document", {}).get((src_metadata.doc_id,), ())
        edges: list[GraphEdge] = []
        for dst in bucket:
            if dst == source:
                continue
            line_gap = abs(documents[dst].metadata.line_start - src_metadata.line_start)
            evidence = GraphEvidence(doc_id=src_metadata.doc_id, line_gap=line_gap)
            edges.append(
                self._edge(
                    source, dst, "same_document", weight=1.0 / (1 + line_gap), evidence=evidence
                )
            )
        return edges

    def _shared_metric_neighbors(self, source: int) -> list[GraphEdge]:
        documents = self._graph.documents
        src_metadata = documents[source].metadata
        if src_metadata.company_code is None:
            return []
        src_metrics = {obs.canonical for obs in documents[source].metric_labels}
        bucket_by_metric = self._graph.buckets.get("shared_metric", {})
        candidates: set[int] = set()
        for metric in src_metrics:
            candidates.update(bucket_by_metric.get((src_metadata.company_code, metric), ()))
        candidates.discard(source)

        edges: list[GraphEdge] = []
        for dst in candidates:
            dst_metrics = {obs.canonical for obs in documents[dst].metric_labels}
            shared = src_metrics & dst_metrics
            union = src_metrics | dst_metrics
            if not shared or not union:
                continue
            jaccard = len(shared) / len(union)
            evidence = GraphEvidence(
                company_code=src_metadata.company_code, shared_metrics=tuple(sorted(shared))
            )
            edges.append(
                self._edge(source, dst, "shared_metric", weight=jaccard, evidence=evidence)
            )
        return edges

    def _adjacent_period_neighbors(self, source: int) -> list[GraphEdge]:
        src_metadata = self._graph.documents[source].metadata
        if src_metadata.company_code is None:
            return []
        bucket_by_year = self._graph.buckets.get("adjacent_period", {})
        period_pairs_by_dst: dict[int, set[tuple[str, str]]] = {}
        for src_year in _years(src_metadata.periods):
            for neighbor_year in (str(int(src_year) - 1), str(int(src_year) + 1)):
                for dst in bucket_by_year.get((src_metadata.company_code, neighbor_year), ()):
                    if dst == source:
                        continue
                    period_pairs_by_dst.setdefault(dst, set()).add((src_year, neighbor_year))

        edges: list[GraphEdge] = []
        for dst, pairs in period_pairs_by_dst.items():
            evidence = GraphEvidence(
                company_code=src_metadata.company_code, period_pairs=tuple(sorted(pairs))
            )
            edges.append(self._edge(source, dst, "adjacent_period", weight=1.0, evidence=evidence))
        return edges

    def _same_statement_type_neighbors(self, source: int) -> list[GraphEdge]:
        src_metadata = self._graph.documents[source].metadata
        if src_metadata.company_code is None or src_metadata.statement_type is None:
            return []
        bucket = self._graph.buckets.get("same_statement_type", {}).get(
            (src_metadata.company_code, src_metadata.statement_type), ()
        )
        evidence = GraphEvidence(
            company_code=src_metadata.company_code, statement_type=src_metadata.statement_type
        )
        return [
            self._edge(source, dst, "same_statement_type", weight=1.0, evidence=evidence)
            for dst in bucket
            if dst != source
        ]

    def _explained_by_note_neighbors(self, source: int) -> list[GraphEdge]:
        documents = self._graph.documents
        src_metadata = documents[source].metadata
        if src_metadata.statement_type not in _TYPED_STATEMENT_TYPES:
            return []
        bucket = self._graph.buckets.get("explained_by_note", {}).get((src_metadata.doc_id,), ())
        evidence = GraphEvidence(
            doc_id=src_metadata.doc_id,
            src_statement_type=src_metadata.statement_type,
            statement_type=_NOTE_STATEMENT_TYPE,
        )
        return [
            self._edge(source, dst, "explained_by_note", weight=1.0, evidence=evidence)
            for dst in bucket
            if dst != source and documents[dst].metadata.statement_type == _NOTE_STATEMENT_TYPE
        ]

    def _edge(
        self,
        source: int,
        dst: int,
        relation: GraphRelation,
        *,
        weight: float,
        evidence: GraphEvidence,
    ) -> GraphEdge:
        documents = self._graph.documents
        return GraphEdge(
            src_table_id=documents[source].table_id,
            dst_table_id=documents[dst].table_id,
            relation=relation,
            weight=weight,
            evidence=evidence,
        )
