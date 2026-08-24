from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from financial_report_qa.retrieval.dense_encoder import (
    SentenceTransformerDenseEncoder,
    approved_encoder_spec,
    encoder_spec_sha256,
)


def test_approved_encoder_specs_are_fully_pinned() -> None:
    """Changing an encoder revision or prefix must produce a different artifact identity."""
    bge = approved_encoder_spec("bge-m3")
    e5 = approved_encoder_spec("multilingual-e5-small")

    assert (bge.revision, bge.dimension, bge.query_prefix, bge.batch_size) == (
        "5617a9f61b028005a4858fdac845db406aefb181",
        1024,
        "",
        8,
    )
    assert (e5.revision, e5.dimension, e5.query_prefix, e5.document_prefix) == (
        "614241f622f53c4eeff9890bdc4f31cfecc418b3",
        384,
        "query: ",
        "passage: ",
    )
    assert encoder_spec_sha256(e5) != encoder_spec_sha256(
        e5.model_copy(update={"query_prefix": ""})
    )
    assert bge.device == "cpu"
    assert encoder_spec_sha256(bge) != encoder_spec_sha256(
        bge.model_copy(update={"device": "cuda"})
    )


def test_qwen3_embedding_4b_is_an_approved_pinned_spec() -> None:
    spec = approved_encoder_spec("qwen3-embedding-4b")

    assert spec.model_id == "Qwen/Qwen3-Embedding-4B"
    assert re.fullmatch(r"[0-9a-f]{40}", spec.revision)
    assert spec.dimension == 2560
    assert spec.max_sequence_length == 8192
    # N5: điểm số liên tục dùng để xếp hạng -- không bao giờ quantize.
    assert spec.dtype == "float32"
    assert spec.normalize_embeddings is True


def test_legacy_encoders_keep_their_512_sequence_length() -> None:
    assert approved_encoder_spec("bge-m3").max_sequence_length == 512
    assert approved_encoder_spec("multilingual-e5-small").max_sequence_length == 512


@dataclass
class _StubSentenceTransformerRecorder:
    """Captures how SentenceTransformerDenseEncoder drives the fake model."""

    init_kwargs: dict[str, Any] = field(default_factory=dict)
    half_calls: int = 0
    encode_kwargs: list[dict[str, Any]] = field(default_factory=list)


class _StubSentenceTransformer:
    """Fake SentenceTransformer: records calls, emits fp16 like a real halved model."""

    def __init__(
        self,
        recorder: _StubSentenceTransformerRecorder,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self._recorder = recorder
        recorder.init_kwargs = dict(kwargs)
        self.max_seq_length = 0

    def half(self) -> _StubSentenceTransformer:
        self._recorder.half_calls += 1
        return self

    def encode(self, texts: Any, **kwargs: Any) -> np.ndarray:
        self._recorder.encode_kwargs.append(dict(kwargs))
        return np.zeros((len(texts), 4), dtype=np.float16)


def _install_stub_modules(monkeypatch: pytest.MonkeyPatch) -> _StubSentenceTransformerRecorder:
    """Inject fake torch / sentence_transformers so no model is ever downloaded.

    The constructor imports both modules lazily inside its body, so replacing
    their sys.modules entries is enough to intercept every touchpoint.
    """
    recorder = _StubSentenceTransformerRecorder()

    def model_factory(*args: Any, **kwargs: Any) -> _StubSentenceTransformer:
        return _StubSentenceTransformer(recorder, *args, **kwargs)

    sentence_transformers = ModuleType("sentence_transformers")
    sentence_transformers.SentenceTransformer = model_factory  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", ModuleType("torch"))
    monkeypatch.setitem(sys.modules, "sentence_transformers", sentence_transformers)
    return recorder


def test_default_construction_keeps_exact_legacy_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No knobs set: no half() call and no batch_size kwarg ever reach the model."""
    recorder = _install_stub_modules(monkeypatch)

    encoder = SentenceTransformerDenseEncoder(approved_encoder_spec("multilingual-e5-small"))
    encoder.encode_documents(["bang can doi"])

    assert recorder.half_calls == 0
    assert len(recorder.encode_kwargs) == 1
    assert "batch_size" not in recorder.encode_kwargs[0]


def test_model_dtype_float16_halves_the_model_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """§5.4: fp16 compute lets Qwen3-4B fit a T4; applied once right after load."""
    recorder = _install_stub_modules(monkeypatch)

    encoder = SentenceTransformerDenseEncoder(
        approved_encoder_spec("multilingual-e5-small"),
        model_dtype="float16",
    )
    encoder.encode_documents(["bang can doi"])

    assert recorder.half_calls == 1


def test_encode_batch_size_is_forwarded_to_sentence_transformer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """batch_size must reach encode itself (ST's internal chunking default is untunable)."""
    recorder = _install_stub_modules(monkeypatch)

    encoder = SentenceTransformerDenseEncoder(
        approved_encoder_spec("multilingual-e5-small"),
        encode_batch_size=8,
    )
    encoder.encode_documents(("hang mot", "hang hai", "hang ba"))

    assert len(recorder.encode_kwargs) == 1
    assert recorder.encode_kwargs[0]["batch_size"] == 8


def test_fp16_compute_output_is_still_cast_to_float32(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """N5: returned/stored vectors stay float32 even when compute runs fp16."""
    _install_stub_modules(monkeypatch)

    encoder = SentenceTransformerDenseEncoder(
        approved_encoder_spec("multilingual-e5-small"),
        model_dtype="float16",
        encode_batch_size=8,
    )

    documents = encoder.encode_documents(["bang can doi"])
    query = encoder.encode_query("doanh thu")

    # Stub emits float16 exactly like a halved model would.
    assert documents.dtype == np.float32
    assert query.dtype == np.float32
