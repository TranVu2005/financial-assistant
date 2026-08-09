from __future__ import annotations

import pytest
from pydantic import ValidationError

from financial_report_qa.retrieval.contracts import TableMetadata
from financial_report_qa.retrieval.dense_artifacts import canonical_json_bytes, sha256_bytes
from financial_report_qa.retrieval.dense_contracts import (
    DenseEncoderSpec,
    DenseRetrievalCandidate,
)


def _metadata(character: str) -> TableMetadata:
    return TableMetadata(
        table_id=f"tbl_{character * 64}",
        doc_id=f"doc_{character}",
        source_path=f"{character}.txt",
        line_start=1,
        line_end=1,
    )


def test_encoder_spec_rejects_unpinned_or_invalid_contracts() -> None:
    """A mutable/latest encoder definition would make an index unreproducible."""
    valid = {
        "name": "bge-m3",
        "model_id": "BAAI/bge-m3",
        "revision": "5" * 40,
        "dimension": 1024,
        "max_sequence_length": 512,
        "query_prefix": "",
        "document_prefix": "",
        "pooling": "sentence_transformers",
        "normalize_embeddings": True,
        "dtype": "float32",
        "device": "cpu",
        "batch_size": 8,
    }

    assert DenseEncoderSpec.model_validate(valid).dimension == 1024
    with pytest.raises(ValidationError):
        DenseEncoderSpec.model_validate({**valid, "revision": "main"})
    with pytest.raises(ValidationError):
        DenseEncoderSpec.model_validate({**valid, "normalize_embeddings": False})


def test_dense_candidate_rejects_nonfinite_scores() -> None:
    """A NaN cosine score would make the stable ranking undefined."""
    with pytest.raises(ValidationError, match="finite"):
        DenseRetrievalCandidate(
            row_id=0,
            table_id="tbl_" + "a" * 64,
            score=float("nan"),
            rank=1,
            metadata=_metadata("a"),
            snippet="title: revenue",
        )


def test_canonical_json_hash_is_key_order_independent() -> None:
    """Equivalent identity payloads must have one reproducible SHA-256 value."""
    assert sha256_bytes(canonical_json_bytes({"b": 2, "a": 1})) == sha256_bytes(
        canonical_json_bytes({"a": 1, "b": 2})
    )
