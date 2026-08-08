"""Deterministic BM25 index construction and persistence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import bm25s  # type: ignore[import-untyped]

from financial_report_qa.retrieval.contracts import BM25IndexManifest, TableDocument

K1 = 1.5
B = 0.75
DELTA = 0.5
METHOD = "lucene"


@dataclass(frozen=True)
class BM25Index:
    documents: tuple[TableDocument, ...]
    retriever: bm25s.BM25
    manifest: BM25IndexManifest


def tokenize_query(query: str) -> list[str]:
    """Use the same deterministic tokenizer for index and query text."""
    tokens = bm25s.tokenize([query], return_ids=False, stopwords=None, show_progress=False)
    return list(tokens[0])


def _documents_hash(documents: tuple[TableDocument, ...]) -> str:
    payload = "\n".join(
        json.dumps(
            document.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        for document in documents
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_bm25_index(
    documents: tuple[TableDocument, ...], *, dataset_fingerprint: str
) -> BM25Index:
    """Create an in-memory, stable BM25 index ordered by table ID."""
    ordered = tuple(sorted(documents, key=lambda document: document.table_id))
    if len({document.table_id for document in ordered}) != len(ordered):
        raise ValueError("BM25 documents must have unique table IDs")
    retriever = bm25s.BM25(k1=K1, b=B, delta=DELTA, method=METHOD)
    corpus_tokens = bm25s.tokenize(
        [document.text for document in ordered],
        return_ids=False,
        stopwords=None,
        show_progress=False,
    )
    retriever.index(corpus_tokens, show_progress=False)
    return BM25Index(
        documents=ordered,
        retriever=retriever,
        manifest=BM25IndexManifest(
            dataset_fingerprint=dataset_fingerprint,
            document_count=len(ordered),
            document_sha256=_documents_hash(ordered),
            bm25s_version=bm25s.__version__,
            k1=K1,
            b=B,
            delta=DELTA,
            method=METHOD,
        ),
    )


def save_bm25_index(index: BM25Index, output_dir: Path) -> None:
    """Persist all artifacts needed to audit and reload an index."""
    output_dir.mkdir(parents=True, exist_ok=True)
    index.retriever.save(output_dir / "bm25s", corpus=None)
    (output_dir / "documents.jsonl").write_text(
        "\n".join(document.model_dump_json() for document in index.documents) + "\n",
        encoding="utf-8",
    )
    (output_dir / "manifest.json").write_text(
        index.manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )


def load_bm25_index(index_dir: Path) -> BM25Index:
    """Load a persisted index only when its document hash still matches the manifest."""
    documents = tuple(
        TableDocument.model_validate_json(line)
        for line in (index_dir / "documents.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    manifest = BM25IndexManifest.model_validate_json(
        (index_dir / "manifest.json").read_text(encoding="utf-8")
    )
    if manifest.document_count != len(documents) or manifest.document_sha256 != _documents_hash(
        documents
    ):
        raise ValueError("Persisted BM25 document artifact does not match manifest")
    return BM25Index(
        documents=documents, retriever=bm25s.BM25.load(index_dir / "bm25s"), manifest=manifest
    )
