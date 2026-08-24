"""Pinned cross-encoder reranking with a small fakeable interface.

Mirrors `dense_encoder.py`: a `Protocol` the pure ranking function depends
on, a pinned spec allowlist, and one real implementation that is only
imported when it is actually constructed. Tests drive the pure function with
a fake scorer and never load a model.

The reranker runs on the top-N output of weighted RRF, never on the corpus
(§5.3 of the target architecture): N = 50 candidates is what fits in the
local CPU budget. It also runs *sequentially* with the dense encoder --
embedding the whole corpus first, reranking afterwards -- so the two models
never contend for the same VRAM.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from financial_report_qa.core.errors import RerankInputError, RerankModelError
from financial_report_qa.retrieval.fusion_contracts import FusedCandidate
from financial_report_qa.retrieval.rerank_contracts import (
    DEFAULT_RERANK_DEPTH,
    RerankedCandidate,
    RerankerName,
    RerankerSpec,
    RerankTrace,
    reranker_spec_sha256,
)

_SPECS: dict[str, RerankerSpec] = {
    "qwen3-reranker-4b": RerankerSpec(
        name="qwen3-reranker-4b",
        model_id="Qwen/Qwen3-Reranker-4B",
        revision="22e683669bc0f0bd69640a1354a6d0aebcfeede5",
        max_sequence_length=2048,
        batch_size=4,
    ),
}


def approved_reranker_spec(name: RerankerName) -> RerankerSpec:
    """Return the pinned spec for one allowlisted reranker."""
    return _SPECS[name]


class Reranker(Protocol):
    spec: RerankerSpec

    def score(self, query: str, documents: Sequence[str]) -> np.ndarray: ...


def rerank_candidates(
    query: str,
    candidates: Sequence[FusedCandidate],
    reranker: Reranker,
    *,
    k: int,
    depth: int = DEFAULT_RERANK_DEPTH,
    question_id: str | None = None,
) -> RerankTrace:
    """Rescore the top-`depth` fused candidates and return the top `k`.

    Ties break on `table_id` so two candidates the model scores identically
    still come back in a stable order -- MRR5 is graded on position, and a
    run-to-run reshuffle would move the score without any change in evidence.
    """
    if k < 1:
        raise RerankInputError("k must be positive")
    if depth < 1:
        raise RerankInputError("depth must be positive")

    spec_hash = reranker_spec_sha256(reranker.spec)
    window = tuple(candidates)[:depth]
    if not window:
        return RerankTrace(
            question_id=question_id,
            query=query,
            reranker_spec_sha256=spec_hash,
            input_count=0,
            empty_reason="no_fused_candidates",
        )

    documents = tuple(candidate.snippet for candidate in window)
    scores = np.asarray(reranker.score(query, documents), dtype=np.float64).reshape(-1)
    if scores.shape[0] != len(window):
        raise RerankModelError("reranker returned a different number of scores than documents")
    if not bool(np.all(np.isfinite(scores))):
        raise RerankModelError("reranker returned a non-finite score")

    ordered = sorted(
        zip(window, scores.tolist(), strict=True),
        key=lambda item: (-item[1], item[0].table_id),
    )[:k]
    results = tuple(
        RerankedCandidate(
            table_id=candidate.table_id,
            rank=rank,
            rerank_score=float(score),
            fused_rank=candidate.rank,
            fused_score=candidate.fused_score,
            metadata=candidate.metadata,
            snippet=candidate.snippet,
        )
        for rank, (candidate, score) in enumerate(ordered, start=1)
    )
    return RerankTrace(
        question_id=question_id,
        query=query,
        reranker_spec_sha256=spec_hash,
        input_count=len(window),
        results=results,
    )


class Qwen3CrossEncoderReranker:
    """Real Qwen3 cross-encoder, loaded from a pinned revision."""

    def __init__(self, spec: RerankerSpec, *, local_files_only: bool = False) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self._torch = torch
            self._tokenizer = AutoTokenizer.from_pretrained(
                spec.model_id,
                revision=spec.revision,
                trust_remote_code=False,
                local_files_only=local_files_only,
            )
            self._model = AutoModelForSequenceClassification.from_pretrained(
                spec.model_id,
                revision=spec.revision,
                torch_dtype=torch.float32,  # N5: never quantize a ranking score
                trust_remote_code=False,
                local_files_only=local_files_only,
            )
        except Exception as exc:
            raise RerankModelError(
                f"Pinned reranker model is unavailable: {spec.model_id}@{spec.revision}"
            ) from exc
        self._model.eval()
        self._model.to(spec.device)
        self.spec = spec

    def score(self, query: str, documents: Sequence[str]) -> np.ndarray:
        values: list[float] = []
        batch = self.spec.batch_size
        for start in range(0, len(documents), batch):
            chunk = list(documents[start : start + batch])
            encoded = self._tokenizer(
                [query] * len(chunk),
                chunk,
                padding=True,
                truncation=True,
                max_length=self.spec.max_sequence_length,
                return_tensors="pt",
            ).to(self.spec.device)
            with self._torch.no_grad():
                logits = self._model(**encoded).logits
            values.extend(logits[:, -1].float().cpu().numpy().tolist())
        return np.asarray(values, dtype=np.float32)
