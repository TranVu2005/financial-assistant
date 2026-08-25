from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from types import ModuleType, SimpleNamespace
from typing import Any, Literal, cast

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
    to_calls: list[Any] = field(default_factory=list)
    encode_kwargs: list[dict[str, Any]] = field(default_factory=list)
    deterministic_flags: list[bool] = field(default_factory=list)


class _StubSentenceTransformer:
    """Fake SentenceTransformer: records calls, emits fp16 like a dtype-cast model.

    Rows carry deliberately off-unit norms (magnitudes 2..3, distinct
    directions): real ST normalizes INSIDE the compute dtype, so fp16/bf16
    models hand back vectors whose L2 drifts ~1e-3 -- exactly what broke
    ``QueryEmbeddingCache._validate`` live on Kaggle."""

    def __init__(
        self,
        recorder: _StubSentenceTransformerRecorder,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self._recorder = recorder
        recorder.init_kwargs = dict(kwargs)
        self.max_seq_length = 0

    def to(self, dtype: Any) -> _StubSentenceTransformer:
        self._recorder.to_calls.append(dtype)
        return self

    def encode(self, texts: Any, **kwargs: Any) -> np.ndarray:
        self._recorder.encode_kwargs.append(dict(kwargs))
        rows = len(texts)
        base = np.arange(1, 4 * rows + 1, dtype=np.float64).reshape(rows, 4)
        magnitudes = 2.0 + (np.arange(rows) % 2)  # 2.0, 3.0, 2.0, ...
        return (base * magnitudes[:, None]).astype(np.float16)


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
    # Dtype markers the encoder resolves via getattr(torch, knob); distinct
    # sentinels let tests assert exactly which dtype was cast to.
    torch_stub = ModuleType("torch")
    torch_stub.float32 = "torch.float32"  # type: ignore[attr-defined]
    torch_stub.float16 = "torch.float16"  # type: ignore[attr-defined]
    # CUDA-determinism surface: the constructor flips these globals only when
    # the EFFECTIVE device starts with "cuda", so the recorder can observe
    # exactly when that setup fires.
    torch_stub.backends = SimpleNamespace(  # type: ignore[attr-defined]
        cudnn=SimpleNamespace(deterministic=False, benchmark=False)
    )

    def use_deterministic_algorithms(flag: bool) -> None:
        recorder.deterministic_flags.append(bool(flag))

    torch_stub.use_deterministic_algorithms = use_deterministic_algorithms  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    monkeypatch.setitem(sys.modules, "sentence_transformers", sentence_transformers)
    return recorder


def test_default_construction_enforces_float32_compute_dtype(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No knobs set: model is cast to spec.dtype once; no batch_size kwarg.

    The legacy "never touch dtype" behavior asserted here previously WAS the
    bug: Qwen3-Embedding-4B ships bf16 weights, so skipping the cast leaked
    bf16 rounding into embeddings and failed QueryEmbeddingCache's unit-norm
    check (measured L2 norm 1.0010896921157837 vs atol=1e-5). Compute dtype is
    therefore now enforced to spec.dtype by default.
    """
    recorder = _install_stub_modules(monkeypatch)

    encoder = SentenceTransformerDenseEncoder(approved_encoder_spec("multilingual-e5-small"))
    encoder.encode_documents(["bang can doi"])

    # Dtype cast first, then the (no-op on the stub) placement .to(spec.device).
    assert recorder.to_calls == ["torch.float32", "cpu"]
    assert len(recorder.encode_kwargs) == 1
    assert "batch_size" not in recorder.encode_kwargs[0]


def test_model_dtype_float16_casts_the_model_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """§5.4: fp16 compute lets Qwen3-4B fit a T4; a single .to(fp16), never fp32."""
    recorder = _install_stub_modules(monkeypatch)

    encoder = SentenceTransformerDenseEncoder(
        approved_encoder_spec("multilingual-e5-small"),
        model_dtype="float16",
    )
    encoder.encode_documents(["bang can doi"])

    assert recorder.to_calls == ["torch.float16", "cpu"]


def test_device_override_places_the_model_on_the_requested_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Placement-only knob (same precedent as ``model_dtype``): ``device=
    "cuda:1"`` constructs SentenceTransformer on cuda:1 and the final
    ``.to`` lands the model there, while ``spec`` itself stays untouched --
    so ``encoder_spec_sha256`` and cache keys are identical across devices."""
    recorder = _install_stub_modules(monkeypatch)

    encoder = SentenceTransformerDenseEncoder(
        approved_encoder_spec("multilingual-e5-small"),
        device="cuda:1",
    )
    encoder.encode_documents(["bang can doi"])

    assert recorder.init_kwargs["device"] == "cuda:1"
    # Dtype cast first (spec default fp32 -- placement does not change compute),
    # then the model ends up exactly where the caller asked.
    assert recorder.to_calls == ["torch.float32", "cuda:1"]
    assert encoder.spec.device == "cpu"


def test_cuda_placement_fires_determinism_setup_for_overridden_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CUDA-determinism globals trigger on the EFFECTIVE device: a runtime
    override onto any cuda* device must pin them exactly like a spec-pinned
    "cuda" would, and a cpu run must never touch them."""
    recorder_override = _install_stub_modules(monkeypatch)
    SentenceTransformerDenseEncoder(
        approved_encoder_spec("multilingual-e5-small"), device="cuda:0"
    )
    assert recorder_override.deterministic_flags == [True]

    recorder_spec = _install_stub_modules(monkeypatch)
    cuda_spec = approved_encoder_spec("multilingual-e5-small").model_copy(
        update={"device": "cuda"}
    )
    encoder = SentenceTransformerDenseEncoder(cuda_spec)
    assert recorder_spec.deterministic_flags == [True]
    assert recorder_spec.init_kwargs["device"] == "cuda"
    assert encoder.spec.device == "cuda"

    recorder_cpu = _install_stub_modules(monkeypatch)
    SentenceTransformerDenseEncoder(approved_encoder_spec("multilingual-e5-small"))
    assert recorder_cpu.deterministic_flags == []


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

    # Stub emits float16 exactly like an fp16-cast model would.
    assert documents.dtype == np.float32
    assert query.dtype == np.float32


def _expected_unit_rows(row_count: int) -> np.ndarray:
    """Mirror the stub's raw output, renormalized independently in float64."""
    base = np.arange(1, 4 * row_count + 1, dtype=np.float64).reshape(row_count, 4)
    magnitudes = 2.0 + (np.arange(row_count) % 2)
    raw = (base * magnitudes[:, None]).astype(np.float16).astype(np.float32)
    return np.asarray(raw / np.linalg.norm(raw, axis=1, keepdims=True), dtype=np.float32)


@pytest.mark.parametrize("model_dtype", [None, "float16"])
def test_embeddings_are_renormalized_in_float32_to_exact_unit_norm(
    monkeypatch: pytest.MonkeyPatch, model_dtype: str | None
) -> None:
    """ST normalizes INSIDE the compute dtype, so fp16/bf16 models hand back
    vectors whose L2 sits ~1e-3 off unit -- QueryEmbeddingCache._validate
    (atol=1e-5) rejected the first query encode live on Kaggle T4. After the
    float32 cast the wrapper must renormalize rows in float32 through BOTH
    default-fp32 and model_dtype="float16" paths, keeping row order and
    direction (unit vectors match an independent recomputation). Stored
    float32 rounds each norm by ~6e-8, so 1e-6 here is still 16x inside the
    downstream atol=1e-5 contract."""
    _install_stub_modules(monkeypatch)

    encoder = SentenceTransformerDenseEncoder(
        approved_encoder_spec("multilingual-e5-small"),
        model_dtype=cast(Literal["float32", "float16"] | None, model_dtype),
    )
    documents = encoder.encode_documents(["hang mot", "hang hai", "hang ba"])
    query = encoder.encode_query("doanh thu")

    assert documents.dtype == np.float32
    assert query.dtype == np.float32
    norms = np.linalg.norm(documents, axis=1)
    np.testing.assert_allclose(norms, 1.0, rtol=0.0, atol=1e-6)
    assert abs(float(np.linalg.norm(query)) - 1.0) <= 1e-6
    # Direction AND order preserved: each row equals the independently
    # renormalized stub vector at the same position.
    expected = _expected_unit_rows(3)
    np.testing.assert_allclose(documents, expected, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(query, expected[0], rtol=0.0, atol=1e-6)
