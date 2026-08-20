"""Deterministic row BM25 index construction and persistence."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import bm25s  # type: ignore[import-untyped]
from pydantic import Field, ValidationError

from financial_report_qa.retrieval.contracts import Fingerprint, NonEmptyString, _FrozenModel
from financial_report_qa.retrieval.index import (
    DELTA,
    DTYPE,
    K1,
    METHOD,
    B,
    _file_sha256,
    tokenize_text,
)
from financial_report_qa.retrieval.row_documents import RowDocument


class RowBM25IndexManifest(_FrozenModel):
    schema_version: Literal["row-bm25-index-v1"] = "row-bm25-index-v1"
    builder_version: Literal["v1"] = "v1"
    dtype: Literal["float32"] = "float32"
    dataset_fingerprint: Fingerprint
    release_lock_sha256: Fingerprint | None = None
    document_count: int = Field(ge=0)
    document_sha256: Fingerprint
    artifact_sha256: dict[str, Fingerprint] = Field(default_factory=dict)
    bm25s_version: NonEmptyString
    k1: float
    b: float
    delta: float
    method: NonEmptyString


@dataclass(frozen=True)
class RowBM25Index:
    documents: tuple[RowDocument, ...]
    retriever: bm25s.BM25
    manifest: RowBM25IndexManifest


def _document_line(document: RowDocument) -> bytes:
    return (
        json.dumps(
            document.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _documents_hash(documents: tuple[RowDocument, ...]) -> str:
    digest = hashlib.sha256()
    for document in documents:
        digest.update(_document_line(document))
    return digest.hexdigest()


def _write_documents(path: Path, documents: tuple[RowDocument, ...]) -> None:
    with path.open("wb") as stream:
        for document in documents:
            stream.write(_document_line(document))


def _artifact_hashes(index_dir: Path) -> dict[str, str]:
    return {
        path.relative_to(index_dir).as_posix(): _file_sha256(path)
        for path in sorted(index_dir.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }


def _manifest_identity(manifest: RowBM25IndexManifest) -> dict[str, object]:
    return manifest.model_dump(mode="json", exclude={"artifact_sha256"})


def _write_manifest(path: Path, manifest: RowBM25IndexManifest) -> None:
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


def build_row_bm25_index(
    documents: tuple[RowDocument, ...],
    *,
    dataset_fingerprint: str,
    release_lock_sha256: str | None = None,
) -> RowBM25Index:
    """Create an in-memory, stable Row BM25 index ordered by row ID."""
    ordered = tuple(sorted(documents, key=lambda document: document.row_id))
    if len({document.row_id for document in ordered}) != len(ordered):
        raise ValueError("Row BM25 documents must have unique row IDs")

    retriever = bm25s.BM25(k1=K1, b=B, delta=DELTA, method=METHOD, dtype=DTYPE)
    corpus_tokens = [tokenize_text(document.text) for document in ordered]
    unique_tokens = sorted({token for doc in corpus_tokens for token in doc})
    vocab_dict = {token: i for i, token in enumerate(unique_tokens)}
    corpus_token_ids = [[vocab_dict[token] for token in doc] for doc in corpus_tokens]
    retriever.index((corpus_token_ids, vocab_dict), show_progress=False)

    return RowBM25Index(
        documents=ordered,
        retriever=retriever,
        manifest=RowBM25IndexManifest(
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


def save_row_bm25_index(index: RowBM25Index, output_dir: Path) -> None:
    """Publish Row index atomically; reject an existing non-identical content-addressed target."""
    if output_dir.exists():
        existing = load_row_bm25_index(output_dir)
        if _manifest_identity(existing.manifest) != _manifest_identity(index.manifest):
            raise ValueError(
                f"Row index target already exists with different content: {output_dir}"
            )
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


def load_row_bm25_index(index_dir: Path, *, release_lock_sha256: str | None = None) -> RowBM25Index:
    """Verify every persisted row artifact before loading executable Row BM25 state."""
    manifest_payload = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest_payload, dict):
        raise ValueError("Row BM25 index manifest must be a JSON object")
    if manifest_payload.get("schema_version") != "row-bm25-index-v1":
        raise ValueError("unsupported Row BM25 index schema; rebuild the index")
    unknown_fields = sorted(set(manifest_payload) - set(RowBM25IndexManifest.model_fields))
    if unknown_fields:
        raise ValueError(
            "Row BM25 index manifest declares row-bm25-index-v1 but carries fields this build does "
            f"not know ({', '.join(unknown_fields)}); it was written by a different build — "
            "rebuild the index"
        )
    try:
        manifest = RowBM25IndexManifest.model_validate(manifest_payload)
    except ValidationError as exc:
        raise ValueError(f"Row BM25 index manifest is invalid; rebuild the index: {exc}") from exc
    if release_lock_sha256 is not None and manifest.release_lock_sha256 != release_lock_sha256:
        raise ValueError("Persisted Row BM25 index release lock hash does not match")
    actual_hashes = _artifact_hashes(index_dir)
    if set(actual_hashes) != set(manifest.artifact_sha256):
        raise ValueError("Persisted Row BM25 artifact set does not match manifest")
    for relative_path, expected_hash in manifest.artifact_sha256.items():
        if actual_hashes[relative_path] != expected_hash:
            raise ValueError(f"Persisted Row BM25 artifact hash mismatch: {relative_path}")
    if manifest.artifact_sha256.get("documents.jsonl") != manifest.document_sha256:
        raise ValueError("Persisted Row BM25 document hash does not match artifact manifest")
    documents = tuple(
        RowDocument.model_validate_json(line)
        for line in (index_dir / "documents.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if manifest.document_count != len(documents) or manifest.document_sha256 != _documents_hash(
        documents
    ):
        raise ValueError("Persisted Row BM25 document artifact does not match manifest")
    return RowBM25Index(
        documents=documents, retriever=bm25s.BM25.load(index_dir / "bm25s"), manifest=manifest
    )
