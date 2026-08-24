"""Unit tests for `retrieval.cli._build_table_retriever` (Task 8): the shared
wiring that assembles the live table retriever for BOTH `submission export`
and `retrieval sweep-k`.

Everything runs offline: the dense corpus/index fixtures are built with a
mock 2-dimensional encoder and the reranker is a fake -- exactly like
`test_row_dense_wiring.py` / `test_live_query.py` -- so no test ever loads a
real sentence-transformers or cross-encoder model."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

import financial_report_qa.retrieval.cli as retrieval_cli
from financial_report_qa.core.errors import DenseArtifactError, DenseInputError
from financial_report_qa.evaluation.week1_release import ReleaseLock
from financial_report_qa.retrieval.contracts import (
    MetricLabelObservation,
    RetrievalFilters,
    TableDocument,
    TableMetadata,
)
from financial_report_qa.retrieval.dense_contracts import DenseEncoderSpec
from financial_report_qa.retrieval.dense_corpus import build_dense_corpus, save_dense_corpus
from financial_report_qa.retrieval.dense_encoder import approved_encoder_spec
from financial_report_qa.retrieval.dense_index import (
    build_dense_index,
    save_dense_index,
)
from financial_report_qa.retrieval.fusion import FusionService
from financial_report_qa.retrieval.index import BM25Index, build_bm25_index
from financial_report_qa.retrieval.live_query import retrieve_candidate_table_ids
from financial_report_qa.retrieval.release import ResolvedRetrievalRelease
from financial_report_qa.retrieval.reranker import approved_reranker_spec
from financial_report_qa.retrieval.service import RetrievalService

_FINGERPRINT = "37a61be7aebde1fbcfe3aca42e6ba4ff37ae87bdd1a9ba6696506bcd188e7d1f"
_LOCK_SHA = "2" * 64
_TABLE_ID = "tbl_" + "a" * 64
_DOC_ID = "doc_" + "a" * 64


class _MockEncoder:
    """DenseEncoder stand-in with a tiny deterministic 2-d space."""

    def __init__(self, spec: DenseEncoderSpec) -> None:
        self.spec = spec

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        return np.asarray([1.0, 0.0], dtype=np.float32)


class _FakeReranker:
    """Reranker stand-in: never loads a model, scores everything equal."""

    def __init__(self) -> None:
        self.spec = approved_reranker_spec("qwen3-reranker-4b")
        self.calls: list[tuple[str, int]] = []

    def score(self, query: str, documents: Sequence[str]) -> np.ndarray:
        self.calls.append((query, len(documents)))
        return np.zeros(len(documents), dtype=np.float32)


def _mock_e5_encoder() -> _MockEncoder:
    return _MockEncoder(
        approved_encoder_spec("multilingual-e5-small").model_copy(update={"dimension": 2})
    )


def _document() -> TableDocument:
    return TableDocument(
        table_id=_TABLE_ID,
        doc_id=_DOC_ID,
        text="company_code: DBC\nperiod: 2023\nTổng tài sản | 2023 | 500",
        metadata=TableMetadata(
            table_id=_TABLE_ID,
            doc_id=_DOC_ID,
            company_code="DBC",
            periods=("2023",),
            statement_type="balance_sheet",
            source_path="a.txt",
            line_start=1,
            line_end=2,
        ),
        metric_labels=(MetricLabelObservation(canonical="total_assets", raw=None),),
    )


def _release(dataset_fingerprint: str = _FINGERPRINT) -> ResolvedRetrievalRelease:
    lock = ReleaseLock(
        alias="dataset-pilot-v1",
        sampling_version="week1-pilot-v1",
        dataset_fingerprint=dataset_fingerprint,
        source_manifest_sha256="0" * 64,
        release_path="fixture/release",
        gate_result_path="fixture/gate.json",
        evaluation_inputs_sha256="1" * 64,
    )
    return ResolvedRetrievalRelease(
        lock=lock,
        dataset_fingerprint=dataset_fingerprint,
        release_dir=Path("unused"),
        gate_result_path=Path("unused/gate.json"),
        lock_path=Path("unused/lock.json"),
        manifest={},
        lock_sha256=_LOCK_SHA,
    )


def _args(**overrides: object) -> argparse.Namespace:
    defaults: dict[str, object] = {
        "dense_index": None,
        "table_dense_weight": 1.0,
        "rerank": False,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _bm25_index() -> BM25Index:
    return build_bm25_index(
        (_document(),), dataset_fingerprint=_FINGERPRINT, release_lock_sha256=_LOCK_SHA
    )


def _write_dense_artifacts(
    tmp_path: Path, *, dataset_fingerprint: str = _FINGERPRINT, lock_sha256: str = _LOCK_SHA
) -> tuple[Path, Path]:
    """Build + save a real (tiny) table dense corpus and index; returns
    (corpus_dir, index_dir) laid out per repo convention: `<fp>/corpus` next
    to nothing else -- the locator must find `<index>/../corpus`."""
    corpus = build_dense_corpus(
        (_document(),), dataset_fingerprint=dataset_fingerprint, release_lock_sha256=lock_sha256
    )
    corpus_dir = tmp_path / f"{dataset_fingerprint[:12]}" / "corpus"
    save_dense_corpus(corpus, corpus_dir)
    encoder = _mock_e5_encoder()
    index = build_dense_index(corpus, encoder)
    index_dir = tmp_path / f"{dataset_fingerprint[:12]}" / "dense-e5-mock"
    save_dense_index(index, index_dir)
    return corpus_dir, index_dir


@pytest.fixture()
def _redirect_table_query_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The helper hardcodes the repo-relative query cache default; tests must
    # not write .npy entries into the real data/ tree.
    monkeypatch.setattr(retrieval_cli, "_TABLE_DENSE_QUERY_CACHE_DEFAULT", tmp_path / "qcache")


def test_bm25_only_by_default_keeps_the_old_behavior() -> None:
    retriever, reranker = retrieval_cli._build_table_retriever(_args(), _release(), _bm25_index())

    assert isinstance(retriever, RetrievalService)
    assert reranker is None


def test_rerank_without_dense_index_is_rejected_before_any_model_loads() -> None:
    with pytest.raises(DenseInputError, match="--rerank cần --dense-index"):
        retrieval_cli._build_table_retriever(_args(rerank=True), _release(), _bm25_index())


def test_missing_corpus_lists_every_searched_location(tmp_path: Path) -> None:
    index_dir = tmp_path / "lonely-index"
    index_dir.mkdir()

    with pytest.raises(DenseArtifactError) as excinfo:
        retrieval_cli._build_table_retriever(
            _args(dense_index=index_dir), _release(), _bm25_index()
        )

    message = str(excinfo.value)
    assert str(index_dir / "corpus") in message
    assert str(index_dir.parent / "corpus") in message


def test_dense_index_fingerprint_mismatch_is_rejected(tmp_path: Path) -> None:
    _, index_dir = _write_dense_artifacts(tmp_path)
    mismatched_release = _release(dataset_fingerprint="f" * 64)

    with pytest.raises(DenseArtifactError, match="does not match --release-lock"):
        retrieval_cli._build_table_retriever(
            _args(dense_index=index_dir), mismatched_release, _bm25_index()
        )


def test_dense_index_encoder_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    """The manifest pins one encoder spec; loading with an injected encoder of
    a DIFFERENT spec must fail the identity check inside `load_dense_index`,
    never silently rank against foreign vectors."""
    corpus_dir, index_dir = _write_dense_artifacts(tmp_path)
    del corpus_dir  # located via convention
    foreign_spec = approved_encoder_spec("bge-m3").model_copy(update={"dimension": 2})

    with pytest.raises(DenseArtifactError, match="does not match"):
        retrieval_cli._build_table_retriever(
            _args(dense_index=index_dir),
            _release(),
            _bm25_index(),
            encoder=_MockEncoder(foreign_spec),
        )


def test_happy_path_wires_fusion_and_runs_the_full_live_path(
    tmp_path: Path, _redirect_table_query_cache: None
) -> None:
    _, index_dir = _write_dense_artifacts(tmp_path)
    encoder = _mock_e5_encoder()

    retriever, reranker = retrieval_cli._build_table_retriever(
        _args(dense_index=index_dir),
        _release(),
        _bm25_index(),
        encoder=encoder,
    )

    assert isinstance(retriever, FusionService)
    # Không có --rerank thì KHÔNG có reranker nào được trả ra -- dù cờ rerank
    # vắng mặt, ý định rank lại phải được nói rõ chứ không mặc định bật.
    assert reranker is None

    trace = retriever.retrieve(
        "Tổng tài sản của DBC năm 2023 là bao nhiêu?",
        filters=RetrievalFilters(),
        k=5,
    )
    assert tuple(candidate.table_id for candidate in trace.results) == (_TABLE_ID,)
    assert all(candidate.fused_score > 0 for candidate in trace.results)


def test_full_live_path_runs_fusion_then_rerank_offline(
    tmp_path: Path, _redirect_table_query_cache: None
) -> None:
    _, index_dir = _write_dense_artifacts(tmp_path)
    encoder = _mock_e5_encoder()
    fake_reranker = _FakeReranker()

    retriever, reranker = retrieval_cli._build_table_retriever(
        _args(dense_index=index_dir, rerank=True),
        _release(),
        _bm25_index(),
        encoder=encoder,
        reranker=fake_reranker,
    )

    retrieved = retrieve_candidate_table_ids(
        "Tổng tài sản của DBC năm 2023 là bao nhiêu?", retriever, k=1, reranker=reranker
    )
    assert retrieved == (_TABLE_ID,)
    # The reranker really scored the fused window, not BM25 output directly.
    assert fake_reranker.calls and fake_reranker.calls[0][1] >= 1


def test_sweep_k_parser_accepts_the_three_stack_flags() -> None:
    args = retrieval_cli._parser().parse_args(
        [
            "sweep-k",
            "--release-lock",
            "lock.json",
            "--bm25-index",
            "idx",
            "--gold",
            "gold.jsonl",
            "--output-stem",
            "out/sweep",
            "--dense-index",
            "dense-dir",
            "--table-dense-weight",
            "0.75",
            "--rerank",
        ]
    )
    assert args.dense_index == Path("dense-dir")
    assert args.table_dense_weight == pytest.approx(0.75)
    assert args.rerank is True

    defaults = retrieval_cli._parser().parse_args(
        [
            "sweep-k",
            "--release-lock",
            "lock.json",
            "--bm25-index",
            "idx",
            "--gold",
            "gold.jsonl",
            "--output-stem",
            "out/sweep",
        ]
    )
    assert defaults.dense_index is None
    assert defaults.table_dense_weight == pytest.approx(1.0)
    assert defaults.rerank is False
