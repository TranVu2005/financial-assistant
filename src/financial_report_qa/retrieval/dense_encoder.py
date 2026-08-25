"""Pinned dense encoders with a small fakeable interface."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any, Literal, Protocol

import numpy as np

from financial_report_qa.core.errors import DenseModelError
from financial_report_qa.retrieval.dense_artifacts import canonical_json_bytes
from financial_report_qa.retrieval.dense_contracts import DenseEncoderSpec, EncoderName

_SPECS = {
    "bge-m3": DenseEncoderSpec(
        name="bge-m3",
        model_id="BAAI/bge-m3",
        revision="5617a9f61b028005a4858fdac845db406aefb181",
        dimension=1024,
        query_prefix="",
        document_prefix="",
        batch_size=8,
    ),
    "multilingual-e5-small": DenseEncoderSpec(
        name="multilingual-e5-small",
        model_id="intfloat/multilingual-e5-small",
        revision="614241f622f53c4eeff9890bdc4f31cfecc418b3",
        dimension=384,
        query_prefix="query: ",
        document_prefix="passage: ",
        batch_size=32,
    ),
    "qwen3-embedding-4b": DenseEncoderSpec(
        name="qwen3-embedding-4b",
        model_id="Qwen/Qwen3-Embedding-4B",
        # Điền SHA commit thật của model trên Hugging Face trước khi chạy
        # build-dense-index; `ModelRevision` từ chối mọi chuỗi không phải
        # 40 ký tự hex, nên một revision sai sẽ hỏng ngay lúc import test.
        revision="5cf2132abc99cad020ac570b19d031efec650f2b",
        dimension=2560,
        max_sequence_length=8192,
        query_prefix="Instruct: Given a financial question, retrieve the "
        "table that contains the answer\nQuery: ",
        document_prefix="",
        batch_size=4,
    ),
}


def approved_encoder_spec(name: EncoderName) -> DenseEncoderSpec:
    return _SPECS[name]


def encoder_spec_sha256(spec: DenseEncoderSpec) -> str:
    return hashlib.sha256(canonical_json_bytes(spec.model_dump(mode="json"))).hexdigest()


class DenseEncoder(Protocol):
    spec: DenseEncoderSpec

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray: ...
    def encode_query(self, text: str) -> np.ndarray: ...


class SentenceTransformerDenseEncoder:
    """Wrapper around ``SentenceTransformer`` with a pinned compute dtype.

    Compute dtype is ENFORCED right after load (single ``Module.to`` over the
    whole model): by default it is ``spec.dtype`` (always ``"float32"``),
    because native checkpoint dtype leaks into embeddings otherwise —
    Qwen3-Embedding-4B ships bf16 weights, whose rounding produced unit-norm
    deviations large enough to fail :meth:`QueryEmbeddingCache._validate`.
    ``model_dtype="float16"`` remains an opt-in knob for constrained-VRAM
    COMPUTE (T4), per plan §5.4.

    ``device`` is a PLACEMENT-only knob (same precedent as ``model_dtype``):
    it moves the model to ``"cpu"``/``"cuda"``/``"cuda:0"``/``"cuda:1"``
    regardless of ``spec.device``, e.g. to share a 2-GPU box between the
    encoder and the reranker. It is deliberately NOT part of
    :class:`DenseEncoderSpec` either.

    All of ``model_dtype``, ``encode_batch_size`` and ``device`` are
    deliberately NOT part of :class:`DenseEncoderSpec` (hence absent from
    ``encoder_spec_sha256`` and the index manifests pinning it): §5.4
    sanctions fp16 COMPUTE so Qwen3-Embedding-4B fits a Colab T4 (~8GB fp16
    weights instead of ~16GB fp32 -> CUDA OOM), while constraint N5 ("không
    quantize") governs what is STORED — this wrapper still casts every output
    to np.float32, so shards and indexes stay float32 on disk. An fp16-encoded
    index paired with fp32-encoded queries (or vice versa) differs only within
    fp16 rounding, and retrieval consumes ranks (RRF), never absolute scores,
    so artifact identity is unaffected by these knobs.
    """

    def __init__(
        self,
        spec: DenseEncoderSpec,
        *,
        local_files_only: bool = False,
        model_dtype: Literal["float32", "float16"] | None = None,
        encode_batch_size: int | None = None,
        device: str | None = None,
    ) -> None:
        # Compute/placement-only knobs; excluded from spec identity (§5.4 vs N5).
        self._model_dtype = model_dtype
        self._encode_batch_size = encode_batch_size
        effective_device = device or spec.device
        try:
            import torch
            from sentence_transformers import SentenceTransformer

            if effective_device.startswith("cuda"):
                # Same-process, same-GPU determinism so A/B replay builds hash identically;
                # cuBLAS workspace must also be pinned via CUBLAS_WORKSPACE_CONFIG before
                # this process starts, which torch cannot set retroactively. Fires for any
                # effective cuda placement -- spec-pinned or runtime-overridden alike.
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                torch.use_deterministic_algorithms(True)

            self._model = SentenceTransformer(
                spec.model_id,
                revision=spec.revision,
                device=effective_device,
                trust_remote_code=False,
                local_files_only=local_files_only,
            )
            # Cast parameters AND buffers to the compute dtype right after load.
            # Module.to(dtype) converts tensor-by-tensor in place (low memory
            # peak -- no fp32 round-trip on GPU), so the fp16 T4 path stays
            # safe mid-load. Without this, native checkpoint dtype (Qwen3
            # ships bf16) leaks bf16 rounding into embeddings and breaks the
            # downstream unit-norm contract.
            knob = model_dtype or spec.dtype  # Literal["float32", "float16"]
            torch_dtype = getattr(torch, knob)  # only float32/float16 possible
            self._model.to(torch_dtype)
            # Placement last: the model ends up exactly where the caller asked,
            # whether that is spec.device or the runtime override.
            self._model.to(effective_device)
        except Exception as exc:
            raise DenseModelError(
                f"Pinned dense model is unavailable: {spec.model_id}@{spec.revision}"
            ) from exc
        self.spec = spec
        self._model.max_seq_length = spec.max_sequence_length

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        extra_kwargs: dict[str, Any] = {}
        if self._encode_batch_size is not None:
            # Forwarded so ST's internal chunking is tunable from outside; when the
            # knob is unset the call below stays byte-identical to the legacy one.
            extra_kwargs["batch_size"] = self._encode_batch_size
        values = self._model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            **extra_kwargs,
        )
        return np.asarray(values, dtype=np.float32)

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(tuple(f"{self.spec.document_prefix}{text}" for text in texts))

    def encode_query(self, text: str) -> np.ndarray:
        return np.asarray(self._encode((f"{self.spec.query_prefix}{text}",))[0], dtype=np.float32)
