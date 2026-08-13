"""Deterministic GTR-lite table-relation graph construction and persistence.

Edges are never materialized eagerly. `build_graph` derives one bucketed
adjacency index per relation (`relation -> key -> positions`) in a single
pass over the sorted, positioned document sequence -- the same "position
into a sorted sequence" scheme `dense_index.py` uses to bind FAISS rows to
table IDs. `graph_service.TableGraphService.neighbors` turns a bucket lookup
into `GraphEdge`s on demand; fan-out limits belong to Day 12, not here (see
`retrieval/graph_contracts.py` module docstring for the excluded-relation
rationale).
"""

from __future__ import annotations

import json
import shutil
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from financial_report_qa.core.errors import GraphArtifactError
from financial_report_qa.retrieval.contracts import TableDocument
from financial_report_qa.retrieval.dense_artifacts import (
    canonical_json_bytes,
    file_sha256,
    write_text_atomic,
)
from financial_report_qa.retrieval.dense_corpus import documents_sha256
from financial_report_qa.retrieval.graph_contracts import (
    GRAPH_RELATIONS,
    ExcludedRelation,
    GraphManifest,
    GraphRelation,
)

BucketKey = tuple[str, ...]
Buckets = dict[GraphRelation, dict[BucketKey, tuple[int, ...]]]


@dataclass(frozen=True)
class TableGraph:
    """Bucketed adjacency over a sorted, positioned document sequence."""

    documents: tuple[TableDocument, ...]
    buckets: Buckets
    manifest: GraphManifest


def _pair_count(bucket_sizes: list[int]) -> int:
    return sum(size * (size - 1) // 2 for size in bucket_sizes)


def build_graph(
    documents: tuple[TableDocument, ...],
    *,
    dataset_fingerprint: str,
    release_lock_sha256: str,
) -> TableGraph:
    ordered = tuple(sorted(documents, key=lambda document: document.table_id))
    if len({document.table_id for document in ordered}) != len(ordered):
        raise ValueError("graph build requires unique table IDs")

    growing: dict[GraphRelation, dict[BucketKey, list[int]]] = {
        relation: defaultdict(list) for relation in GRAPH_RELATIONS
    }
    # Measurement-only buckets for the two relations deliberately excluded
    # from the graph (see graph_contracts module docstring); never persisted.
    company_only: dict[str, list[int]] = defaultdict(list)
    company_year: dict[tuple[str, str], list[int]] = defaultdict(list)

    for position, document in enumerate(ordered):
        metadata = document.metadata
        growing["same_document"][(metadata.doc_id,)].append(position)
        if metadata.statement_type is not None:
            growing["explained_by_note"][(metadata.doc_id,)].append(position)

        if metadata.company_code is not None:
            company_only[metadata.company_code].append(position)
            # Distinct years, not distinct periods: "2024" and "2024-12-31" on
            # the same table must contribute this position to a bucket once,
            # not once per period string that happens to share a year.
            for year in {period[:4] for period in metadata.periods}:
                growing["adjacent_period"][(metadata.company_code, year)].append(position)
                company_year[(metadata.company_code, year)].append(position)
            if metadata.statement_type is not None:
                growing["same_statement_type"][
                    (metadata.company_code, metadata.statement_type)
                ].append(position)
            # Distinct canonical metrics: metric_labels can carry two
            # observations with the same canonical id but different raw
            # labels (dedup is on the (canonical, raw) pair, not canonical
            # alone), which must not double-count this position in a bucket.
            for metric in {observation.canonical for observation in document.metric_labels}:
                growing["shared_metric"][(metadata.company_code, metric)].append(position)

    buckets: Buckets = {
        relation: {key: tuple(positions) for key, positions in per_key.items()}
        for relation, per_key in growing.items()
    }

    excluded_relations = (
        ExcludedRelation(
            name="same_company",
            reason=(
                "every reviewed gold question already hard-filters company_codes before "
                "ranking (retrieval/filtering.py), so same_company edges would only connect "
                "tables already inside the eligible pool -- no new information"
            ),
            measured_pair_count=_pair_count([len(v) for v in company_only.values()]),
        ),
        ExcludedRelation(
            name="same_period",
            reason=(
                "every reviewed gold question already hard-filters periods before ranking "
                "(retrieval/filtering.py), so same_period edges would only connect tables "
                "already inside the eligible pool -- no new information"
            ),
            measured_pair_count=_pair_count([len(v) for v in company_year.values()]),
        ),
    )

    manifest = GraphManifest(
        dataset_fingerprint=dataset_fingerprint,
        release_lock_sha256=release_lock_sha256,
        document_count=len(ordered),
        document_sha256=documents_sha256(ordered),
        excluded_relations=excluded_relations,
        bucket_counts={relation: len(per_key) for relation, per_key in buckets.items()},
        membership_counts={
            relation: sum(len(positions) for positions in per_key.values())
            for relation, per_key in buckets.items()
        },
    )
    return TableGraph(ordered, buckets, manifest)


def _identity(manifest: GraphManifest) -> dict[str, object]:
    return manifest.model_dump(mode="json", exclude={"artifact_sha256"})


def _bucket_lines(buckets: Buckets) -> bytes:
    rows = bytearray()
    for relation in sorted(buckets):
        for key in sorted(buckets[relation]):
            rows.extend(
                canonical_json_bytes(
                    {
                        "relation": relation,
                        "key": list(key),
                        "positions": list(buckets[relation][key]),
                    }
                )
            )
    return bytes(rows)


def save_graph(graph: TableGraph, output_dir: Path) -> Path:
    """Publish atomically; reject an existing non-identical content-addressed target."""
    if output_dir.exists():
        # Compare manifests directly rather than recursing through load_graph:
        # load_graph requires its `documents` argument to match the *existing*
        # artifact's document set, which is exactly what may differ here.
        existing_payload = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
        existing_manifest = GraphManifest.model_validate(existing_payload)
        if _identity(existing_manifest) != _identity(graph.manifest):
            raise GraphArtifactError(
                f"Graph target already exists with different content: {output_dir}"
            )
        return output_dir
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        (temporary / "buckets.jsonl").write_bytes(_bucket_lines(graph.buckets))
        manifest = graph.manifest.model_copy(
            update={"artifact_sha256": {"buckets.jsonl": file_sha256(temporary / "buckets.jsonl")}}
        )
        write_text_atomic(
            temporary / "manifest.json",
            json.dumps(
                manifest.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True
            )
            + "\n",
        )
        temporary.replace(output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output_dir


def load_graph(
    output_dir: Path, documents: tuple[TableDocument, ...], *, release_lock_sha256: str
) -> TableGraph:
    payload = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise GraphArtifactError("Graph manifest must be a JSON object")
    if payload.get("schema_version") != "graph-v1":
        raise GraphArtifactError("unsupported graph schema; rebuild the graph")
    unknown_fields = sorted(set(payload) - set(GraphManifest.model_fields))
    if unknown_fields:
        raise GraphArtifactError(
            "Graph manifest declares graph-v1 but carries fields this build does not know "
            f"({', '.join(unknown_fields)}); it was written by a different build -- rebuild "
            "the graph"
        )
    try:
        manifest = GraphManifest.model_validate(payload)
    except ValidationError as exc:
        raise GraphArtifactError(f"Graph manifest is invalid; rebuild the graph: {exc}") from exc

    if manifest.release_lock_sha256 != release_lock_sha256:
        raise GraphArtifactError("Graph release lock hash does not match")

    buckets_path = output_dir / "buckets.jsonl"
    if set(manifest.artifact_sha256) != {"buckets.jsonl"} or manifest.artifact_sha256[
        "buckets.jsonl"
    ] != file_sha256(buckets_path):
        raise GraphArtifactError("Graph artifact hash does not match manifest")

    ordered = tuple(sorted(documents, key=lambda document: document.table_id))
    if manifest.document_count != len(ordered) or manifest.document_sha256 != documents_sha256(
        ordered
    ):
        raise GraphArtifactError("Graph documents do not match manifest")

    buckets: Buckets = {relation: {} for relation in manifest.relations}
    for line in buckets_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        relation = row["relation"]
        if relation not in buckets:
            raise GraphArtifactError(f"Graph bucket file references unknown relation {relation!r}")
        positions = tuple(row["positions"])
        if any(position < 0 or position >= len(ordered) for position in positions):
            raise GraphArtifactError("Graph bucket file references an out-of-range position")
        buckets[relation][tuple(row["key"])] = positions

    bucket_counts = {relation: len(per_key) for relation, per_key in buckets.items()}
    membership_counts = {
        relation: sum(len(positions) for positions in per_key.values())
        for relation, per_key in buckets.items()
    }
    if bucket_counts != dict(manifest.bucket_counts) or membership_counts != dict(
        manifest.membership_counts
    ):
        raise GraphArtifactError("Graph bucket contents do not match manifest counts")

    return TableGraph(ordered, buckets, manifest)
