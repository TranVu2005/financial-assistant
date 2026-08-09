"""Deterministic shared corpus persistence for dense table retrieval."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from financial_report_qa.core.errors import DenseArtifactError
from financial_report_qa.retrieval.contracts import TableDocument
from financial_report_qa.retrieval.dense_artifacts import (
    canonical_json_bytes,
    file_sha256,
    write_text_atomic,
)
from financial_report_qa.retrieval.dense_contracts import DenseCorpusManifest


@dataclass(frozen=True)
class DenseCorpus:
    documents: tuple[TableDocument, ...]
    manifest: DenseCorpusManifest


def document_line(document: TableDocument) -> bytes:
    return canonical_json_bytes(document.model_dump(mode="json"))


def documents_sha256(documents: tuple[TableDocument, ...]) -> str:
    digest = hashlib.sha256()
    for document in documents:
        digest.update(document_line(document))
    return digest.hexdigest()


def build_dense_corpus(
    documents: tuple[TableDocument, ...], *, dataset_fingerprint: str, release_lock_sha256: str
) -> DenseCorpus:
    ordered = tuple(sorted(documents, key=lambda document: document.table_id))
    if len({document.table_id for document in ordered}) != len(ordered):
        raise ValueError("dense corpus requires unique table IDs")
    return DenseCorpus(
        ordered,
        DenseCorpusManifest(
            dataset_fingerprint=dataset_fingerprint,
            release_lock_sha256=release_lock_sha256,
            document_count=len(ordered),
            document_sha256=documents_sha256(ordered),
        ),
    )


def _manifest_text(manifest: DenseCorpusManifest) -> str:
    return (
        json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n"
    )


def save_dense_corpus(corpus: DenseCorpus, output_dir: Path) -> Path:
    if output_dir.exists():
        existing = load_dense_corpus(
            output_dir, release_lock_sha256=corpus.manifest.release_lock_sha256
        )
        if existing.manifest.model_dump(exclude={"artifact_sha256"}) != corpus.manifest.model_dump(
            exclude={"artifact_sha256"}
        ):
            raise DenseArtifactError(
                f"Dense corpus target already exists with different content: {output_dir}"
            )
        return output_dir
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        (temporary / "documents.jsonl").write_bytes(
            b"".join(document_line(document) for document in corpus.documents)
        )
        manifest = corpus.manifest.model_copy(
            update={
                "artifact_sha256": {"documents.jsonl": file_sha256(temporary / "documents.jsonl")}
            }
        )
        write_text_atomic(temporary / "manifest.json", _manifest_text(manifest))
        temporary.replace(output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output_dir


def load_dense_corpus(corpus_dir: Path, *, release_lock_sha256: str) -> DenseCorpus:
    payload = json.loads((corpus_dir / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise DenseArtifactError("Dense corpus manifest must be a JSON object")
    manifest = DenseCorpusManifest.model_validate(payload)
    if manifest.release_lock_sha256 != release_lock_sha256:
        raise DenseArtifactError("Dense corpus release lock hash does not match")
    documents_path = corpus_dir / "documents.jsonl"
    if set(manifest.artifact_sha256) != {"documents.jsonl"} or manifest.artifact_sha256[
        "documents.jsonl"
    ] != file_sha256(documents_path):
        raise DenseArtifactError("Dense corpus artifact hash does not match manifest")
    documents = tuple(
        TableDocument.model_validate_json(line)
        for line in documents_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if tuple(document.table_id for document in documents) != tuple(
        sorted(document.table_id for document in documents)
    ) or len({document.table_id for document in documents}) != len(documents):
        raise DenseArtifactError("Dense corpus documents must be sorted unique table IDs")
    if manifest.document_count != len(documents) or manifest.document_sha256 != documents_sha256(
        documents
    ):
        raise DenseArtifactError("Dense corpus documents do not match manifest")
    return DenseCorpus(documents, manifest)
