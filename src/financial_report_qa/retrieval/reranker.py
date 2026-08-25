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
from typing import Any, Literal, Protocol

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

#: Default judge instruction when a caller does not supply one; mirrors the
#: Qwen3-Reranker model card's retrieval-style instruction for this domain.
_DEFAULT_RERANK_INSTRUCTION = (
    "Given a financial question, retrieve the table content that answers it"
)

_JUDGE_SYSTEM_LINE = (
    "Judge whether the Document meets the requirements based on the Query and "
    "the Instruct below. Note that the answer need not be explicitly stated "
    "in the Document."
)


def _judge_prompt(query: str, document: str) -> str:
    """Official Qwen3-Reranker judge prompt for one (query, document) pair.

    The checkpoint is a plain causal LM: relevance is read off the final
    position as ``logit("yes") - logit("no")``, so the pair must be rendered
    through the exact chat template the reranker was trained on.
    """
    return (
        "<|im_start|>system\n"
        f"{_JUDGE_SYSTEM_LINE}<|im_end|>\n"
        "<|im_start|>user\n"
        "# Instruct\n"
        f"{_DEFAULT_RERANK_INSTRUCTION}\n"
        "\n"
        "# Query\n"
        f"{query}\n"
        "\n"
        "# Document\n"
        f"{document}<|im_end|>\n"
        "<|im_start|>assistant\n"
        "\n"
    )


def _single_token_id(tokenizer: Any, token: str) -> int:
    """Resolve ``yes``/``no`` to one vocabulary id, or fail loudly."""
    token_id = tokenizer.convert_tokens_to_ids(token)
    unk_id = getattr(tokenizer, "unk_token_id", None)
    if token_id is None or (unk_id is not None and token_id == unk_id):
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if len(encoded) != 1:
            raise RerankModelError(
                f"reranker judge token {token!r} must map to a single id, got {encoded}"
            )
        return int(encoded[0])
    return int(token_id)


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
    """Qwen3-Reranker scored through the OFFICIAL causal yes/no path.

    The pinned checkpoint (``Qwen/Qwen3-Reranker-4B``) ships **base
    ``Qwen3ForCausalLM`` weights only -- there is no trained classification
    head**. Loading it via ``AutoModelForSequenceClassification`` made
    transformers report ``score.weight | MISSING | newly initialized``, so
    every emitted score was a random projection. Per the model card, relevance
    is instead read off the final position of a causal forward pass over the
    judge chat template as ``logit("yes") - logit("no")``.

    ``model_dtype`` is a COMPUTE-only knob (same precedent as
    ``SentenceTransformerDenseEncoder.model_dtype``): it is deliberately NOT
    part of :class:`RerankerSpec`, so ``reranker_spec_sha256``, the index-free
    score cache keys and every artifact identity stay unchanged across dtypes.
    N5 still governs what is STORED: the SCORE/CONTRACT dtype stays
    ``spec.dtype`` (always ``"float32"``) and :meth:`score` casts every logit
    difference through ``.float()``, so emitted scores remain float32 no matter
    what the forward pass computes in.

    ``None`` (the default) keeps the historical fp32 load. ``"float16"`` or
    ``"bfloat16"`` lowers compute precision so the ~8GB bf16 checkpoint fits
    a 15GB T4 instead of OOMing at ~16GB fp32 -- at the honest cost that
    near-tie ranks may shift under fp16/bf16 rounding: validate against an
    fp32 run on a sample of questions before trusting a measurement taken at
    reduced precision.
    """

    def __init__(
        self,
        spec: RerankerSpec,
        *,
        local_files_only: bool = False,
        model_dtype: Literal["float32", "float16", "bfloat16"] | None = None,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._torch = torch
            # Compute-only knob: ``None`` keeps the fp32 spec contract; any
            # explicit value only lowers forward-pass precision (see class doc).
            target = getattr(torch, model_dtype) if model_dtype is not None else torch.float32
            load_kwargs: dict[str, Any] = {
                "revision": spec.revision,
                "trust_remote_code": False,
                "local_files_only": local_files_only,
            }
            self._tokenizer = AutoTokenizer.from_pretrained(
                spec.model_id,
                torch_dtype=target,
                **load_kwargs,
            )
            # The pinned revision is a Qwen3ForCausalLM checkpoint; loading it
            # through AutoModelForSequenceClassification left `score.weight`
            # MISSING in the transformers LOAD REPORT and every score was a
            # random projection. transformers >=5 renamed the kwarg to `dtype`;
            # older releases spell it `torch_dtype`.
            try:
                self._model = AutoModelForCausalLM.from_pretrained(
                    spec.model_id,
                    dtype=target,  # COMPUTE precision; N5 governs stored scores
                    **load_kwargs,
                )
            except TypeError:
                self._model = AutoModelForCausalLM.from_pretrained(
                    spec.model_id,
                    torch_dtype=target,  # COMPUTE precision; N5 governs stored scores
                    **load_kwargs,
                )
        except Exception as exc:
            raise RerankModelError(
                f"Pinned reranker model is unavailable: {spec.model_id}@{spec.revision}"
            ) from exc
        # Qwen3ForCausalLM inherits eval/to without annotations in the shipped
        # transformers types; the calls themselves are trivially safe.
        self._model.eval()  # type: ignore[no-untyped-call]
        self._model.to(spec.device)  # type: ignore[arg-type]
        self.spec = spec
        # Judge vocabulary ids are fixed once at load; they must resolve to a
        # single id each or every downstream score would be meaningless.
        self._yes_id = _single_token_id(self._tokenizer, "yes")
        self._no_id = _single_token_id(self._tokenizer, "no")

    def score(self, query: str, documents: Sequence[str]) -> np.ndarray:
        values: list[float] = []
        batch = self.spec.batch_size
        for start in range(0, len(documents), batch):
            chunk = list(documents[start : start + batch])
            prompts = [_judge_prompt(query, document) for document in chunk]
            encoded = self._tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=self.spec.max_sequence_length,
                return_tensors="pt",
            ).to(self.spec.device)
            with self._torch.no_grad():
                logits = self._model(**encoded).logits
            last_logits = logits[:, -1, :]
            chunk_scores = (
                (last_logits[:, self._yes_id] - last_logits[:, self._no_id]).float().cpu().numpy()
            )
            values.extend(np.asarray(chunk_scores, dtype=np.float64).reshape(-1).tolist())
        return np.asarray(values, dtype=np.float32)
