from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from financial_report_qa.retrieval.contracts import TableDocument, TableMetadata
from financial_report_qa.retrieval.dense_contracts import DenseEncoderSpec
from financial_report_qa.retrieval.dense_corpus import DenseCorpus, build_dense_corpus
from financial_report_qa.retrieval.dense_encoder import approved_encoder_spec, encoder_spec_sha256
from financial_report_qa.retrieval.dense_index import (
    build_dense_index,
    load_dense_index,
    save_dense_index,
)


def _corpus() -> DenseCorpus:
    docs = tuple(
        TableDocument(
            table_id=f"tbl_{token * 64}",
            doc_id=f"doc_{token}",
            text=token,
            metadata=TableMetadata(
                table_id=f"tbl_{token * 64}",
                doc_id=f"doc_{token}",
                source_path=f"{token}.txt",
                line_start=1,
                line_end=1,
            ),
        )
        for token in ("a", "b", "c")
    )
    return build_dense_corpus(docs, dataset_fingerprint="f" * 64, release_lock_sha256="e" * 64)


@dataclass
class FakeEncoder:
    spec: DenseEncoderSpec
    document_batches: list[int] = field(default_factory=list)

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        self.document_batches.append(len(texts))
        values = np.asarray(
            [[1.0, 0.0] if text == "a" else [0.0, 1.0] for text in texts], dtype=np.float32
        )
        return values

    def encode_query(self, text: str) -> np.ndarray:
        return np.asarray([1.0, 0.0], dtype=np.float32)


def test_dense_index_batches_and_persists_vectors(tmp_path: Path) -> None:
    encoder = FakeEncoder(
        approved_encoder_spec("multilingual-e5-small").model_copy(
            update={"dimension": 2, "batch_size": 2}
        )
    )
    corpus = _corpus()
    built = build_dense_index(corpus, encoder)
    target = tmp_path / "index"
    save_dense_index(built, target)
    loaded = load_dense_index(
        target,
        corpus,
        expected_encoder_spec_sha256=encoder_spec_sha256(encoder.spec),
        release_lock_sha256="e" * 64,
    )

    assert encoder.document_batches == [2, 1]
    assert loaded.faiss_index.ntotal == 3
    assert loaded.faiss_index.d == 2
