from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from financial_report_qa.core.errors import DenseArtifactError
from financial_report_qa.retrieval.contracts import RetrievalFilters, TableDocument, TableMetadata
from financial_report_qa.retrieval.dense_cache import QueryEmbeddingCache
from financial_report_qa.retrieval.dense_contracts import DenseEncoderSpec
from financial_report_qa.retrieval.dense_corpus import build_dense_corpus
from financial_report_qa.retrieval.dense_encoder import approved_encoder_spec
from financial_report_qa.retrieval.dense_index import build_dense_index
from financial_report_qa.retrieval.dense_service import DenseRetrievalService


@dataclass
class Encoder:
    spec: DenseEncoderSpec
    query_calls: int = 0

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        self.query_calls += 1
        return np.asarray([1.0, 0.0], dtype=np.float32)


def _document(value: str, company: str) -> TableDocument:
    table_id = "tbl_" + value * 64
    return TableDocument(
        table_id=table_id,
        doc_id=value,
        text=value,
        metadata=TableMetadata(
            table_id=table_id,
            doc_id=value,
            company_code=company,
            source_path=f"{value}.txt",
            line_start=1,
            line_end=1,
        ),
    )


def _service(tmp_path: Path) -> tuple[DenseRetrievalService, Encoder]:
    documents = tuple(
        _document(value, company) for value, company in (("a", "VCB"), ("b", "VCB"), ("c", "ACB"))
    )
    corpus = build_dense_corpus(
        documents,
        dataset_fingerprint="f" * 64,
        release_lock_sha256="e" * 64,
    )
    encoder = Encoder(
        approved_encoder_spec("multilingual-e5-small").model_copy(update={"dimension": 2})
    )
    return (
        DenseRetrievalService(
            build_dense_index(corpus, encoder),
            encoder,
            QueryEmbeddingCache(tmp_path, encoder.spec),
        ),
        encoder,
    )


def test_dense_service_filters_before_stable_tie_ranking(tmp_path: Path) -> None:
    """A global search followed by filtering could return an ineligible ACB table."""
    service, _ = _service(tmp_path)
    trace = service.retrieve("doanh thu", filters=RetrievalFilters(company_codes=("VCB",)), k=2)
    assert [item.table_id for item in trace.results] == [
        "tbl_" + "a" * 64,
        "tbl_" + "b" * 64,
    ]


def test_dense_service_does_not_encode_when_filters_match_nothing(tmp_path: Path) -> None:
    """An empty eligibility set must not invoke the model or write a cache entry."""
    service, encoder = _service(tmp_path)

    trace = service.retrieve("doanh thu", filters=RetrievalFilters(company_codes=("TCB",)))

    assert trace.empty_reason == "no_eligible_documents"
    assert trace.eligible_count == 0
    assert trace.results == ()
    assert encoder.query_calls == 0


def test_dense_service_rejects_a_cache_for_a_different_encoder_spec(tmp_path: Path) -> None:
    """A cache from another encoder could otherwise return an incompatible query vector."""
    service, encoder = _service(tmp_path)
    mismatched_cache = QueryEmbeddingCache(
        tmp_path,
        encoder.spec.model_copy(update={"batch_size": encoder.spec.batch_size + 1}),
    )

    with pytest.raises(DenseArtifactError, match="cache"):
        DenseRetrievalService(service._index, encoder, mismatched_cache)
