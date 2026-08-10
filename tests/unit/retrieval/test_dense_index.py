from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import faiss
import numpy as np
import pytest
from pytest import MonkeyPatch

from financial_report_qa.core.errors import DenseArtifactError
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


def _encoder() -> FakeEncoder:
    return FakeEncoder(
        approved_encoder_spec("multilingual-e5-small").model_copy(
            update={"dimension": 2, "batch_size": 2}
        )
    )

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


def test_cuda_build_uses_gpu_then_returns_cpu_index(monkeypatch: MonkeyPatch) -> None:
    calls: list[str] = []
    gpu_batches: list[int] = []
    cpu_index: faiss.IndexFlatIP | None = None

    class GpuIndexProxy:
        def __init__(self, wrapped: faiss.IndexFlatIP) -> None:
            self.wrapped = wrapped

        def add(self, vectors: np.ndarray) -> None:
            gpu_batches.append(len(vectors))
            self.wrapped.add(vectors)

    def make_resources() -> object:
        calls.append("resources")
        return object()

    def move_to_gpu(resources: object, device: int, index: faiss.IndexFlatIP) -> GpuIndexProxy:
        nonlocal cpu_index
        calls.append(f"to_gpu:{device}")
        cpu_index = index
        return GpuIndexProxy(index)

    def move_to_cpu(index: GpuIndexProxy) -> faiss.IndexFlatIP:
        calls.append("to_cpu")
        assert cpu_index is not None
        return cpu_index

    monkeypatch.setattr(
        faiss, "StandardGpuResources", make_resources, raising=False
    )
    monkeypatch.setattr(
        faiss,
        "index_cpu_to_gpu",
        move_to_gpu,
        raising=False,
    )
    monkeypatch.setattr(
        faiss,
        "index_gpu_to_cpu",
        move_to_cpu,
        raising=False,
    )
    monkeypatch.setattr(faiss, "get_num_gpus", lambda: 1, raising=False)
    encoder = FakeEncoder(
        approved_encoder_spec("multilingual-e5-small").model_copy(
            update={"dimension": 2, "batch_size": 2}
        )
    )

    built = build_dense_index(_corpus(), encoder, faiss_device="cuda")

    assert calls == ["resources", "to_gpu:0", "to_cpu"]
    assert encoder.document_batches == [2, 1]
    assert gpu_batches == [2, 1]
    assert isinstance(built.faiss_index, faiss.IndexFlatIP)
    assert built.faiss_index.ntotal == 3


@pytest.mark.parametrize(
    "missing_api", ("StandardGpuResources", "index_cpu_to_gpu", "index_gpu_to_cpu")
)
def test_cuda_build_fails_closed_when_gpu_api_is_missing(
    monkeypatch: MonkeyPatch, missing_api: str
) -> None:
    def move_to_gpu(resources: object, device: int, index: faiss.IndexFlatIP) -> faiss.IndexFlatIP:
        return index

    def move_to_cpu(index: faiss.IndexFlatIP) -> faiss.IndexFlatIP:
        return index

    monkeypatch.setattr(faiss, "StandardGpuResources", object, raising=False)
    monkeypatch.setattr(faiss, "index_cpu_to_gpu", move_to_gpu, raising=False)
    monkeypatch.setattr(faiss, "index_gpu_to_cpu", move_to_cpu, raising=False)
    monkeypatch.setattr(faiss, "get_num_gpus", lambda: 1, raising=False)
    monkeypatch.delattr(faiss, missing_api, raising=False)

    with pytest.raises(DenseArtifactError, match=missing_api):
        build_dense_index(_corpus(), _encoder(), faiss_device="cuda")


def test_cuda_build_fails_closed_when_no_gpu_is_available(monkeypatch: MonkeyPatch) -> None:
    def move_to_gpu(resources: object, device: int, index: faiss.IndexFlatIP) -> faiss.IndexFlatIP:
        return index

    def move_to_cpu(index: faiss.IndexFlatIP) -> faiss.IndexFlatIP:
        return index

    monkeypatch.setattr(faiss, "StandardGpuResources", object, raising=False)
    monkeypatch.setattr(faiss, "index_cpu_to_gpu", move_to_gpu, raising=False)
    monkeypatch.setattr(faiss, "index_gpu_to_cpu", move_to_cpu, raising=False)
    monkeypatch.setattr(faiss, "get_num_gpus", lambda: 0, raising=False)

    with pytest.raises(DenseArtifactError, match=r"get_num_gpus\(\) returned 0"):
        build_dense_index(_corpus(), _encoder(), faiss_device="cuda")


def test_dense_index_reports_progress_after_each_batch() -> None:
    progress: list[tuple[int, int, float]] = []

    build_dense_index(
        _corpus(),
        _encoder(),
        progress=lambda encoded, total, elapsed: progress.append((encoded, total, elapsed)),
    )

    assert [(encoded, total) for encoded, total, _ in progress] == [(2, 3), (3, 3)]
    assert progress[0][2] >= 0.0
    assert progress[1][2] >= progress[0][2]
