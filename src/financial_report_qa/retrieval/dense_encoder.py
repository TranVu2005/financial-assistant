"""Pinned dense encoders with a small fakeable interface."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Protocol

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
    def __init__(self, spec: DenseEncoderSpec, *, local_files_only: bool = False) -> None:
        try:
            import torch
            from sentence_transformers import SentenceTransformer

            if spec.device == "cuda":
                # Same-process, same-GPU determinism so A/B replay builds hash identically;
                # cuBLAS workspace must also be pinned via CUBLAS_WORKSPACE_CONFIG before
                # this process starts, which torch cannot set retroactively.
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                torch.use_deterministic_algorithms(True)

            self._model = SentenceTransformer(
                spec.model_id,
                revision=spec.revision,
                device=spec.device,
                trust_remote_code=False,
                local_files_only=local_files_only,
            )
        except Exception as exc:
            raise DenseModelError(
                f"Pinned dense model is unavailable: {spec.model_id}@{spec.revision}"
            ) from exc
        self.spec = spec
        self._model.max_seq_length = spec.max_sequence_length

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        values = self._model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(values, dtype=np.float32)

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(tuple(f"{self.spec.document_prefix}{text}" for text in texts))

    def encode_query(self, text: str) -> np.ndarray:
        return np.asarray(self._encode((f"{self.spec.query_prefix}{text}",))[0], dtype=np.float32)
