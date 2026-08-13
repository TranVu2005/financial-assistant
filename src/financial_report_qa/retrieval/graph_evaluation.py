"""Deterministic Day 11 coverage reporting for the GTR-lite table graph.

Coverage is computed analytically from `TableGraph.buckets`, never by
materializing every `GraphEdge` through `TableGraphService`. The real
directed-edge counts on the locked 146,011-table release are large enough
(`same_document` ~12.0M, `adjacent_period` ~62.0M) that building pydantic
`GraphEdge` objects for all of them would be needlessly slow; per-bucket set
arithmetic gives the exact same counts and weight extrema in a single pass.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from pydantic import Field

from financial_report_qa.retrieval.contracts import Fingerprint, TableDocument, _FrozenModel
from financial_report_qa.retrieval.dense_artifacts import write_text_atomic
from financial_report_qa.retrieval.graph import TableGraph
from financial_report_qa.retrieval.graph_contracts import ExcludedRelation, GraphRelation
from financial_report_qa.retrieval.graph_service import _TYPED_STATEMENT_TYPES, _years

_NOTE_STATEMENT_TYPE = "notes"


class GraphRelationCoverage(_FrozenModel):
    bucket_count: int = Field(ge=0)
    membership_count: int = Field(ge=0)
    nodes_with_edges: int = Field(ge=0)
    isolated_nodes: int = Field(ge=0)
    directed_edge_count: int = Field(ge=0)
    degree_p50: int = Field(ge=0)
    degree_p95: int = Field(ge=0)
    degree_max: int = Field(ge=0)
    weight_min: float | None = None
    weight_max: float | None = None


class GraphCoverageReport(_FrozenModel):
    dataset_fingerprint: Fingerprint
    document_count: int = Field(ge=0)
    by_relation: dict[GraphRelation, GraphRelationCoverage]
    nodes_with_no_edge_in_any_relation: int = Field(ge=0)
    excluded_relations: tuple[ExcludedRelation, ...]


def _nearest_rank_percentile(sorted_values: list[int], percentile: int) -> int:
    if not sorted_values:
        return 0
    rank = max(1, math.ceil(percentile / 100 * len(sorted_values)))
    return sorted_values[min(rank, len(sorted_values)) - 1]


def evaluate_graph_coverage(
    graph: TableGraph, documents: tuple[TableDocument, ...]
) -> GraphCoverageReport:
    ordered = tuple(sorted(documents, key=lambda document: document.table_id))
    if ordered != graph.documents:
        raise ValueError("documents do not match the graph's document order")
    node_count = len(ordered)
    metrics_by_position = [
        frozenset(observation.canonical for observation in document.metric_labels)
        for document in ordered
    ]

    by_relation: dict[GraphRelation, GraphRelationCoverage] = {}
    isolated_everywhere = set(range(node_count))

    for relation in graph.manifest.relations:
        buckets = graph.buckets.get(relation, {})
        degree = [0] * node_count
        weight_min: float | None = None
        weight_max: float | None = None
        directed_edge_count = 0

        if relation == "same_document":
            for positions in buckets.values():
                size = len(positions)
                if size < 2:
                    continue
                directed_edge_count += size * (size - 1)
                for position in positions:
                    degree[position] += size - 1
                lines = sorted(ordered[position].metadata.line_start for position in positions)
                local_min_gap = min(b - a for a, b in zip(lines, lines[1:]))
                local_max_gap = lines[-1] - lines[0]
                candidate_max = 1.0 / (1 + local_min_gap)
                candidate_min = 1.0 / (1 + local_max_gap)
                weight_max = candidate_max if weight_max is None else max(weight_max, candidate_max)
                weight_min = candidate_min if weight_min is None else min(weight_min, candidate_min)

        elif relation == "same_statement_type":
            for positions in buckets.values():
                size = len(positions)
                if size < 2:
                    continue
                directed_edge_count += size * (size - 1)
                for position in positions:
                    degree[position] += size - 1
            if directed_edge_count:
                weight_min = weight_max = 1.0

        elif relation == "adjacent_period":
            for position, document in enumerate(ordered):
                company = document.metadata.company_code
                if company is None:
                    continue
                years = _years(document.metadata.periods)
                if not years:
                    continue
                neighbor_positions: set[int] = set()
                for year in years:
                    neighbor_positions |= set(buckets.get((company, str(int(year) - 1)), ()))
                    neighbor_positions |= set(buckets.get((company, str(int(year) + 1)), ()))
                neighbor_positions.discard(position)
                if neighbor_positions:
                    degree[position] = len(neighbor_positions)
                    directed_edge_count += len(neighbor_positions)
            if directed_edge_count:
                weight_min = weight_max = 1.0

        elif relation == "explained_by_note":
            for positions in buckets.values():
                note_positions = [
                    position
                    for position in positions
                    if ordered[position].metadata.statement_type == _NOTE_STATEMENT_TYPE
                ]
                typed_positions = [
                    position
                    for position in positions
                    if ordered[position].metadata.statement_type in _TYPED_STATEMENT_TYPES
                ]
                if not note_positions or not typed_positions:
                    continue
                for position in typed_positions:
                    degree[position] = len(note_positions)
                    directed_edge_count += len(note_positions)
            if directed_edge_count:
                weight_min = weight_max = 1.0

        elif relation == "shared_metric":
            for position, document in enumerate(ordered):
                company = document.metadata.company_code
                src_metrics = metrics_by_position[position]
                if company is None or not src_metrics:
                    continue
                candidates: set[int] = set()
                for metric in src_metrics:
                    candidates |= set(buckets.get((company, metric), ()))
                candidates.discard(position)
                local_weights = []
                for dst in candidates:
                    union = src_metrics | metrics_by_position[dst]
                    if not union:
                        continue
                    local_weights.append(len(src_metrics & metrics_by_position[dst]) / len(union))
                if local_weights:
                    degree[position] = len(local_weights)
                    directed_edge_count += len(local_weights)
                    local_max, local_min = max(local_weights), min(local_weights)
                    weight_max = local_max if weight_max is None else max(weight_max, local_max)
                    weight_min = local_min if weight_min is None else min(weight_min, local_min)

        else:
            raise AssertionError(f"unhandled graph relation: {relation}")

        isolated_this_relation = {position for position, value in enumerate(degree) if value == 0}
        isolated_everywhere &= isolated_this_relation
        nodes_with_edges = node_count - len(isolated_this_relation)
        sorted_degree = sorted(degree)
        by_relation[relation] = GraphRelationCoverage(
            bucket_count=graph.manifest.bucket_counts.get(relation, 0),
            membership_count=graph.manifest.membership_counts.get(relation, 0),
            nodes_with_edges=nodes_with_edges,
            isolated_nodes=len(isolated_this_relation),
            directed_edge_count=directed_edge_count,
            degree_p50=_nearest_rank_percentile(sorted_degree, 50),
            degree_p95=_nearest_rank_percentile(sorted_degree, 95),
            degree_max=sorted_degree[-1] if sorted_degree else 0,
            weight_min=weight_min,
            weight_max=weight_max,
        )

    return GraphCoverageReport(
        dataset_fingerprint=graph.manifest.dataset_fingerprint,
        document_count=node_count,
        by_relation=by_relation,
        nodes_with_no_edge_in_any_relation=len(isolated_everywhere),
        excluded_relations=graph.manifest.excluded_relations,
    )


def deterministic_projection(report: GraphCoverageReport) -> dict[str, object]:
    """Return the replay-relevant projection (currently the full report).

    Nothing in `GraphCoverageReport` is timing-derived or cache-state-derived
    today, matching `fusion_evaluation.deterministic_projection`. Kept as an
    explicit function so a future non-deterministic field has an obvious
    place to be scrubbed instead of silently leaking into a replay
    comparison.
    """
    projection = report.model_dump(mode="json")
    if not isinstance(projection, dict):
        raise ValueError("day 11 deterministic projection must be an object")
    return projection


def _render_markdown(report: GraphCoverageReport) -> str:
    lines = [
        "# Day 11 Graph Coverage",
        "",
        f"- Dataset fingerprint: `{report.dataset_fingerprint}`",
        f"- Documents (nodes): {report.document_count}",
        f"- Nodes with no edge in any relation: {report.nodes_with_no_edge_in_any_relation}",
        "",
        "## Coverage by relation",
        "",
        "| relation | buckets | membership | nodes w/ edges | isolated | directed edges | "
        "p50 | p95 | max | weight min | weight max |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for relation in sorted(report.by_relation):
        coverage = report.by_relation[relation]
        weight_min = "-" if coverage.weight_min is None else f"{coverage.weight_min:.6f}"
        weight_max = "-" if coverage.weight_max is None else f"{coverage.weight_max:.6f}"
        lines.append(
            f"| {relation} | {coverage.bucket_count} | {coverage.membership_count} | "
            f"{coverage.nodes_with_edges} | {coverage.isolated_nodes} | "
            f"{coverage.directed_edge_count} | {coverage.degree_p50} | {coverage.degree_p95} | "
            f"{coverage.degree_max} | {weight_min} | {weight_max} |"
        )
    lines.extend(("", "## Excluded relations", ""))
    lines.append("| relation | measured pair count | reason |")
    lines.append("| --- | ---: | --- |")
    for excluded in report.excluded_relations:
        lines.append(f"| {excluded.name} | {excluded.measured_pair_count} | {excluded.reason} |")
    return "\n".join(lines) + "\n"


def write_day11_graph(report: GraphCoverageReport, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = report.dataset_fingerprint[:12]
    json_path = output_dir / f"retrieval-day11-graph-{prefix}.json"
    markdown_path = output_dir / f"retrieval-day11-graph-{prefix}.md"
    write_text_atomic(
        json_path,
        json.dumps(report.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, indent=2)
        + "\n",
    )
    write_text_atomic(markdown_path, _render_markdown(report))
    return json_path, markdown_path
