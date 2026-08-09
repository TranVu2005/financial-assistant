from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from financial_report_qa.retrieval.contracts import TableDocument, TableMetadata
from financial_report_qa.retrieval.dense_corpus import (
    build_dense_corpus,
    load_dense_corpus,
    save_dense_corpus,
)


def _documents() -> tuple[TableDocument, ...]:
    return tuple(
        TableDocument(
            table_id=f"tbl_{character * 64}",
            doc_id=f"doc_{character}",
            text=f"title: {character}",
            metadata=TableMetadata(
                table_id=f"tbl_{character * 64}",
                doc_id=f"doc_{character}",
                source_path=f"{character}.txt",
                line_start=1,
                line_end=1,
            ),
        )
        for character in ("a", "b")
    )


def test_dense_corpus_sorts_documents_and_persists_stable_rows(tmp_path: Path) -> None:
    corpus = build_dense_corpus(
        tuple(reversed(_documents())), dataset_fingerprint="f" * 64, release_lock_sha256="e" * 64
    )
    target = tmp_path / "corpus"
    save_dense_corpus(corpus, target)
    loaded = load_dense_corpus(target, release_lock_sha256="e" * 64)
    assert [item.table_id for item in loaded.documents] == ["tbl_" + "a" * 64, "tbl_" + "b" * 64]
    assert (
        loaded.manifest.artifact_sha256["documents.jsonl"]
        == hashlib.sha256((target / "documents.jsonl").read_bytes()).hexdigest()
    )


def test_dense_corpus_rejects_duplicate_table_ids() -> None:
    with pytest.raises(ValueError, match="unique table IDs"):
        build_dense_corpus(
            (_documents()[0], _documents()[0]),
            dataset_fingerprint="f" * 64,
            release_lock_sha256="e" * 64,
        )
