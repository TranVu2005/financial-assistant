"""Deterministic BM25 index construction and persistence."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import bm25s  # type: ignore[import-untyped]

from financial_report_qa.retrieval.contracts import BM25IndexManifest, TableDocument

K1 = 1.5
B = 0.25
DELTA = 0.5
METHOD = "lucene"
DTYPE = "float32"
TOKEN_RE = re.compile(r"(?u)\b\w+\b")


@dataclass(frozen=True)
class BM25Index:
    documents: tuple[TableDocument, ...]
    retriever: bm25s.BM25
    manifest: BM25IndexManifest


def tokenize_text(text: str) -> tuple[str, ...]:
    """Tokenize with the pinned Day 8 NFKC/casefold/regex policy."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return tuple(TOKEN_RE.findall(normalized))


def tokenize_query(query: str) -> list[str]:
    """Compatibility wrapper using the same tokenizer as index documents."""
    return list(tokenize_text(query))


def _document_line(document: TableDocument) -> bytes:
    return (
        json.dumps(
            document.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _documents_hash(documents: tuple[TableDocument, ...]) -> str:
    digest = hashlib.sha256()
    for document in documents:
        digest.update(_document_line(document))
    return digest.hexdigest()


def _write_documents(path: Path, documents: tuple[TableDocument, ...]) -> None:
    with path.open("wb") as stream:
        for document in documents:
            stream.write(_document_line(document))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_hashes(index_dir: Path) -> dict[str, str]:
    return {
        path.relative_to(index_dir).as_posix(): _file_sha256(path)
        for path in sorted(index_dir.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }


def _manifest_identity(manifest: BM25IndexManifest) -> dict[str, object]:
    return manifest.model_dump(mode="json", exclude={"artifact_sha256"})


def _write_manifest(path: Path, manifest: BM25IndexManifest) -> None:
    path.write_text(
        json.dumps(
            manifest.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def build_bm25_index(
    documents: tuple[TableDocument, ...],
    *,
    dataset_fingerprint: str,
    release_lock_sha256: str | None = None,
) -> BM25Index:
    """Create an in-memory, stable BM25 index ordered by table ID."""
    ordered = tuple(sorted(documents, key=lambda document: document.table_id))
    if len({document.table_id for document in ordered}) != len(ordered):
        raise ValueError("BM25 documents must have unique table IDs")
    retriever = bm25s.BM25(k1=K1, b=B, delta=DELTA, method=METHOD, dtype=DTYPE)
    corpus_tokens = [tokenize_text(document.text) for document in ordered]
    # Collect unique tokens and sort them to ensure deterministic vocabulary ID assignment
    unique_tokens = sorted({token for doc in corpus_tokens for token in doc})
    vocab_dict = {token: i for i, token in enumerate(unique_tokens)}
    corpus_token_ids = [[vocab_dict[token] for token in doc] for doc in corpus_tokens]
    retriever.index((corpus_token_ids, vocab_dict), show_progress=False)
    return BM25Index(
        documents=ordered,
        retriever=retriever,
        manifest=BM25IndexManifest(
            dataset_fingerprint=dataset_fingerprint,
            release_lock_sha256=release_lock_sha256,
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
    """Publish atomically; reject an existing non-identical content-addressed target."""
    if output_dir.exists():
        existing = load_bm25_index(output_dir)
        if _manifest_identity(existing.manifest) != _manifest_identity(index.manifest):
            raise ValueError(f"Index target already exists with different content: {output_dir}")
        return
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        index.retriever.save(temporary / "bm25s", corpus=None)
        _write_documents(temporary / "documents.jsonl", index.documents)
        persisted_manifest = index.manifest.model_copy(
            update={"artifact_sha256": _artifact_hashes(temporary)}
        )
        _write_manifest(temporary / "manifest.json", persisted_manifest)
        temporary.replace(output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def load_bm25_index(
    index_dir: Path, *, release_lock_sha256: str | None = None
) -> BM25Index:
    """Verify every persisted artifact before loading executable BM25 state."""
    manifest_payload = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest_payload, dict):
        raise ValueError("BM25 index manifest must be a JSON object")
    if manifest_payload.get("schema_version") != "bm25-index-v3":
        raise ValueError("unsupported BM25 index schema; rebuild the index")
    manifest = BM25IndexManifest.model_validate(manifest_payload)
    if (
        release_lock_sha256 is not None
        and manifest.release_lock_sha256 != release_lock_sha256
    ):
        raise ValueError("Persisted BM25 index release lock hash does not match")
    actual_hashes = _artifact_hashes(index_dir)
    if set(actual_hashes) != set(manifest.artifact_sha256):
        raise ValueError("Persisted BM25 artifact set does not match manifest")
    for relative_path, expected_hash in manifest.artifact_sha256.items():
        if actual_hashes[relative_path] != expected_hash:
            raise ValueError(f"Persisted BM25 artifact hash mismatch: {relative_path}")
    if manifest.artifact_sha256.get("documents.jsonl") != manifest.document_sha256:
        raise ValueError("Persisted BM25 document hash does not match artifact manifest")
    documents = tuple(
        TableDocument.model_validate_json(line)
        for line in (index_dir / "documents.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if manifest.document_count != len(documents) or manifest.document_sha256 != _documents_hash(
        documents
    ):
        raise ValueError("Persisted BM25 document artifact does not match manifest")
    return BM25Index(
        documents=documents, retriever=bm25s.BM25.load(index_dir / "bm25s"), manifest=manifest
    )
