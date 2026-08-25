import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

from financial_report_qa.core.errors import RerankInputError, RerankModelError
from financial_report_qa.retrieval.contracts import TableMetadata
from financial_report_qa.retrieval.fusion_contracts import FusedCandidate
from financial_report_qa.retrieval.rerank_contracts import RerankerSpec
from financial_report_qa.retrieval.reranker import (
    Qwen3CrossEncoderReranker,
    approved_reranker_spec,
    rerank_candidates,
)


class _FakeReranker:
    """Cho điểm theo bảng tra cứu table_id -> score, đếm số lần được gọi."""

    def __init__(self, scores: dict[str, float]) -> None:
        self.spec = RerankerSpec(
            name="qwen3-reranker-4b",
            model_id="Qwen/Qwen3-Reranker-4B",
            revision="a" * 40,
            batch_size=4,
        )
        self._scores = scores
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def score(self, query: str, documents):  # type: ignore[no-untyped-def]
        self.calls.append((query, tuple(documents)))
        return np.asarray([self._scores[doc] for doc in documents], dtype=np.float32)


def _table_id(suffix: str) -> str:
    return "tbl_" + suffix * 64


def _candidate(suffix: str, *, fused_rank: int, fused_score: float) -> FusedCandidate:
    table_id = _table_id(suffix)
    return FusedCandidate(
        table_id=table_id,
        rank=fused_rank,
        fused_score=fused_score,
        contradiction_count=0,
        metadata=TableMetadata(
            table_id=table_id,
            doc_id="doc_" + "0" * 64,
            source_path="X/2023/x/x_extracted.txt",
            line_start=1,
            line_end=2,
        ),
        snippet=f"snippet-{suffix}",
    )


def test_reranking_reorders_by_cross_encoder_score_not_fused_rank() -> None:
    candidates = (
        _candidate("a", fused_rank=1, fused_score=0.9),
        _candidate("b", fused_rank=2, fused_score=0.5),
    )
    reranker = _FakeReranker({"snippet-a": 0.1, "snippet-b": 0.8})

    trace = rerank_candidates("q", candidates, reranker, k=2)

    assert [item.table_id for item in trace.results] == [_table_id("b"), _table_id("a")]
    assert [item.rank for item in trace.results] == [1, 2]
    # Gốc RRF được giữ lại nguyên vẹn để giải trình quyết định đảo thứ tự.
    assert [item.fused_rank for item in trace.results] == [2, 1]


def test_reranking_only_sees_the_top_depth_candidates() -> None:
    candidates = tuple(
        _candidate(chr(ord("a") + i), fused_rank=i + 1, fused_score=1.0 - i / 10) for i in range(5)
    )
    reranker = _FakeReranker({f"snippet-{chr(ord('a') + i)}": float(i) for i in range(5)})

    trace = rerank_candidates("q", candidates, reranker, k=2, depth=3)

    assert trace.input_count == 3
    assert len(reranker.calls) == 1
    assert reranker.calls[0][1] == ("snippet-a", "snippet-b", "snippet-c")


def test_ties_break_on_table_id_so_the_order_is_deterministic() -> None:
    candidates = (
        _candidate("b", fused_rank=1, fused_score=0.9),
        _candidate("a", fused_rank=2, fused_score=0.5),
    )
    reranker = _FakeReranker({"snippet-b": 0.5, "snippet-a": 0.5})

    trace = rerank_candidates("q", candidates, reranker, k=2)

    assert [item.table_id for item in trace.results] == [_table_id("a"), _table_id("b")]


def test_empty_candidate_list_returns_an_explicit_empty_reason() -> None:
    reranker = _FakeReranker({})

    trace = rerank_candidates("q", (), reranker, k=10)

    assert trace.results == ()
    assert trace.empty_reason == "no_fused_candidates"
    assert reranker.calls == []


def test_k_must_be_positive() -> None:
    with pytest.raises(RerankInputError):
        rerank_candidates("q", (), _FakeReranker({}), k=0)


def test_a_non_finite_model_score_is_rejected_loudly() -> None:
    class _NaNReranker(_FakeReranker):
        def score(self, query: str, documents):  # type: ignore[no-untyped-def]
            return np.asarray([float("nan")] * len(documents), dtype=np.float32)

    with pytest.raises(RerankModelError):
        rerank_candidates(
            "q", (_candidate("a", fused_rank=1, fused_score=0.9),), _NaNReranker({}), k=1
        )


def test_score_count_mismatch_is_rejected() -> None:
    class _ShortReranker(_FakeReranker):
        def score(self, query: str, documents):  # type: ignore[no-untyped-def]
            return np.asarray([0.5], dtype=np.float32)

    candidates = (
        _candidate("a", fused_rank=1, fused_score=0.9),
        _candidate("b", fused_rank=2, fused_score=0.5),
    )
    with pytest.raises(RerankModelError):
        rerank_candidates("q", candidates, _ShortReranker({}), k=2)


def test_approved_spec_is_pinned() -> None:
    spec = approved_reranker_spec("qwen3-reranker-4b")
    assert spec.model_id == "Qwen/Qwen3-Reranker-4B"
    assert spec.dtype == "float32"


# ---------------------------------------------------------------------------
# The REAL Qwen3CrossEncoderReranker, driven through stub torch/transformers
# via the same sys.modules seam as test_dense_encoder.py: nothing downloads,
# nothing imports a GPU stack. These tests pin the OFFICIAL causal yes/no
# inference path (logit("yes") - logit("no") at the last position over the
# judge chat template) -- loading the pinned checkpoint through
# AutoModelForSequenceClassification silently initialized a random head.
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM_LINE = (
    "Judge whether the Document meets the requirements based on the Query and "
    'the Instruct provided. Note that the answer can only be "yes" or "no".'
)
_INSTRUCTION = "Given a financial question, retrieve the table content that answers it"

_STUB_YES_ID = 5
_STUB_NO_ID = 2
_STUB_VOCAB = 8
# Sentinel mass parked on every non-final position (pads included): reading it
# at logits[:, -1, :] yields this wrong diff instead of the crafted one, so any
# padding-side regression is visible in the emitted scores.
_PAD_SENTINEL_YES_DIFF = -60.0


class _StubTensor:
    """Numpy-backed stand-in exposing only the indexing surface score() uses."""

    def __init__(self, array: np.ndarray) -> None:
        self._array = np.asarray(array)

    @property
    def array(self) -> np.ndarray:
        return self._array

    def __getitem__(self, key: Any) -> "_StubTensor":
        return _StubTensor(self._array[key])

    def __sub__(self, other: "_StubTensor") -> "_StubTensor":
        return _StubTensor(self._array - other._array)

    def float(self) -> "_StubTensor":
        return self

    def cpu(self) -> "_StubTensor":
        return self

    def numpy(self) -> np.ndarray:
        return self._array


class _StubEncoding:
    """Batch encoding with PER-ROW lengths whose ``.to`` mimics BatchEncoding.

    Rows get distinct lengths and PAD positions sit according to the
    tokenizer's ``padding_side``, so reading column -1 is correct for every
    row only under LEFT padding -- the exact production invariant.
    """

    def __init__(self, texts: list[str], padding_side: str) -> None:
        self.padding_side = padding_side
        self.lengths = [2 + (j % 3) for j in range(len(texts))]
        self.max_len = max(self.lengths, default=1)

    def to(self, device: Any) -> dict[str, _StubTensor]:
        input_ids = np.zeros((len(self.lengths), self.max_len), dtype=np.int64)
        attention_mask = np.zeros_like(input_ids)
        for row, length in enumerate(self.lengths):
            real = slice(self.max_len - length, None)
            if self.padding_side == "right":
                real = slice(None, length)
            input_ids[row, real] = 1
            attention_mask[row, real] = 1
        return {
            "input_ids": _StubTensor(input_ids),
            "attention_mask": _StubTensor(attention_mask),
        }


@dataclass
class _StubRerankRecorder:
    """Captures how Qwen3CrossEncoderReranker drives the stubbed libraries."""

    dtype_kwargs_seen: list[dict[str, Any]] = field(default_factory=list)
    model_load_kwargs: dict[str, Any] = field(default_factory=dict)
    tokenizer_load_kwargs: dict[str, Any] = field(default_factory=dict)
    tokenizer_calls: list[dict[str, Any]] = field(default_factory=list)
    device_moves: list[Any] = field(default_factory=list)
    next_row: int = 0


class _StubTokenizer:
    """Fake AutoTokenizer: fixed yes/no ids, records every tokenization call."""

    unk_token_id: int | None = None
    padding_side: str = "right"  # transformers' own default when not configured

    def __init__(
        self,
        recorder: _StubRerankRecorder,
        *,
        judge_ids: dict[str, int | None] | None = None,
        fallback_encode_ids: list[int] | None = None,
    ) -> None:
        self._recorder = recorder
        self._judge_ids = judge_ids or {"yes": _STUB_YES_ID, "no": _STUB_NO_ID}
        self._fallback_encode_ids = fallback_encode_ids or [_STUB_NO_ID]

    def convert_tokens_to_ids(self, token: str) -> int | None:
        return self._judge_ids.get(token)

    def encode(self, token: str, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return list(self._fallback_encode_ids)

    def __call__(self, texts: Any, **kwargs: Any) -> _StubEncoding:
        texts = list(texts)
        self._recorder.tokenizer_calls.append({"texts": texts, "kwargs": kwargs})
        return _StubEncoding(texts, self.padding_side)


class _StubCausalLM:
    """Fake Qwen3ForCausalLM: yes/no mass sits ONLY at each row's last real token.

    Every other position -- pads included -- carries sentinel mass whose yes-no
    diff is ``_PAD_SENTINEL_YES_DIFF``, so reading ``logits[:, -1, :]`` yields
    the crafted per-document diffs iff the batch was LEFT padded.
    """

    def __init__(self, recorder: _StubRerankRecorder) -> None:
        self._recorder = recorder

    def eval(self) -> "_StubCausalLM":
        return self

    def to(self, device: Any) -> "_StubCausalLM":
        self._recorder.device_moves.append(device)
        return self

    def __call__(self, **inputs: Any) -> SimpleNamespace:
        attention = inputs["attention_mask"].array
        batch, seq = attention.shape
        logits = np.zeros((batch, seq, _STUB_VOCAB), dtype=np.float32)
        logits[..., _STUB_YES_ID] = _PAD_SENTINEL_YES_DIFF / 2.0  # -30
        logits[..., _STUB_NO_ID] = -_PAD_SENTINEL_YES_DIFF / 2.0  # +30
        for row in range(batch):
            position = self._recorder.next_row
            self._recorder.next_row += 1
            last_real = int(np.flatnonzero(attention[row])[-1])
            # Distinct per-document masses so yes-no diffs are order-sensitive.
            logits[row, last_real, _STUB_YES_ID] = 10.0 + float(position)
            logits[row, last_real, _STUB_NO_ID] = 0.5 * float(position)
        return SimpleNamespace(logits=_StubTensor(logits))


def _install_stub_modules(
    monkeypatch: pytest.MonkeyPatch,
    *,
    reject_dtype_kwarg: bool = False,
    tokenizer: _StubTokenizer | None = None,
    drop_padding_side: bool = False,
) -> _StubRerankRecorder:
    """Inject fake torch / transformers; the constructor imports both lazily.

    ``drop_padding_side`` simulates a loader that silently ignores the
    ``padding_side`` kwarg (tokenizer keeps its "right" default) while still
    recording what the production code passed.
    """
    recorder = _StubRerankRecorder()
    tokenizer = tokenizer or _StubTokenizer(recorder)

    def tokenizer_from_pretrained(*args: Any, **kwargs: Any) -> _StubTokenizer:
        recorder.tokenizer_load_kwargs.update(kwargs)
        applied = dict(kwargs)
        if drop_padding_side:
            applied.pop("padding_side", None)
        tokenizer.padding_side = str(applied.get("padding_side", "right"))
        return tokenizer

    def causal_from_pretrained(*args: Any, **kwargs: Any) -> _StubCausalLM:
        recorder.dtype_kwargs_seen.append(
            {k: v for k, v in kwargs.items() if k in ("dtype", "torch_dtype")}
        )
        if reject_dtype_kwarg and "dtype" in kwargs:
            raise TypeError("from_pretrained() got an unexpected keyword argument 'dtype'")
        recorder.model_load_kwargs.update(kwargs)
        return _StubCausalLM(recorder)

    transformers_stub = ModuleType("transformers")
    transformers_stub.AutoModelForCausalLM = SimpleNamespace(  # type: ignore[attr-defined]
        from_pretrained=causal_from_pretrained
    )
    transformers_stub.AutoTokenizer = SimpleNamespace(  # type: ignore[attr-defined]
        from_pretrained=tokenizer_from_pretrained
    )

    torch_stub = ModuleType("torch")
    torch_stub.float32 = "torch.float32"  # type: ignore[attr-defined]
    torch_stub.float16 = "torch.float16"  # type: ignore[attr-defined]
    torch_stub.bfloat16 = "torch.bfloat16"  # type: ignore[attr-defined]

    @contextmanager
    def no_grad() -> Iterator[None]:
        yield

    torch_stub.no_grad = no_grad  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    monkeypatch.setitem(sys.modules, "transformers", transformers_stub)
    return recorder


def test_score_builds_the_pinned_card_judge_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    """One single-string judge prompt per (query, document), byte-exact against
    the PINNED revision's model card: judge system line, <Instruct>/<Query>/
    <Document> body, forced empty think block after ``assistant``."""
    recorder = _install_stub_modules(monkeypatch)
    reranker = Qwen3CrossEncoderReranker(approved_reranker_spec("qwen3-reranker-4b"))

    query, document = "doanh thu nam 2023 la bao nhieu", "| bang ke | 120 ty |"
    reranker.score(query, [document])

    call = recorder.tokenizer_calls[0]
    expected_prompt = (
        "<|im_start|>system\n"
        f"{_JUDGE_SYSTEM_LINE}<|im_end|>\n"
        "<|im_start|>user\n"
        f"<Instruct>: {_INSTRUCTION}\n"
        f"<Query>: {query}\n"
        f"<Document>: {document}<|im_end|>\n"
        "<|im_start|>assistant\n\n\n"
    )
    assert call["texts"] == [expected_prompt]
    # Tokenization follows the model-card recipe exactly.
    assert call["kwargs"] == {
        "padding": True,
        "truncation": True,
        "max_length": 2048,
        "return_tensors": "pt",
    }
    assert recorder.device_moves == ["cpu"]


def test_tokenizer_loads_left_padded_and_without_a_dtype_kwarg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LEFT padding is what makes logits[:, -1, :] hit each row's true final
    token; the compute-dtype shim belongs to the model load only."""
    recorder = _install_stub_modules(monkeypatch)
    Qwen3CrossEncoderReranker(approved_reranker_spec("qwen3-reranker-4b"))

    assert recorder.tokenizer_load_kwargs["padding_side"] == "left"
    assert "dtype" not in recorder.tokenizer_load_kwargs
    assert "torch_dtype" not in recorder.tokenizer_load_kwargs


def test_right_padding_would_read_pad_positions_and_corrupt_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression guard on the stub itself: when padding_side is dropped and
    the tokenizer keeps its right default, column -1 is a PAD for shorter rows
    and the sentinel diff leaks into every score -- proving the production
    ``padding_side="left"`` is load-bearing for score correctness."""
    recorder = _install_stub_modules(monkeypatch, drop_padding_side=True)
    reranker = Qwen3CrossEncoderReranker(approved_reranker_spec("qwen3-reranker-4b"))

    scores = reranker.score("q", tuple(f"doc {i}" for i in range(6)))

    expected = [(10.0 + p) - 0.5 * p for p in range(6)]
    assert recorder.tokenizer_load_kwargs["padding_side"] == "left"  # code asked for left
    assert not np.allclose(scores, expected, atol=1e-6)
    # The longest row of each batch still lands on its real final token.
    assert scores.tolist()[2] == pytest.approx(expected[2], abs=1e-6)


def test_two_batches_split_respects_spec_batch_size(monkeypatch: pytest.MonkeyPatch) -> None:
    """Six documents against batch_size=4 -> one full batch plus one remainder."""
    recorder = _install_stub_modules(monkeypatch)
    reranker = Qwen3CrossEncoderReranker(approved_reranker_spec("qwen3-reranker-4b"))

    documents = tuple(f"van ban {i}" for i in range(6))
    reranker.score("q", documents)

    assert [len(call["texts"]) for call in recorder.tokenizer_calls] == [4, 2]
    # Remainder batch keeps judge-prompt order; the generation suffix follows.
    assert "<Document>: van ban 4<|im_end|>\n" in recorder.tokenizer_calls[1]["texts"][0]
    assert "<Document>: van ban 5<|im_end|>\n" in recorder.tokenizer_calls[1]["texts"][1]


def test_scores_are_yes_minus_no_diffs_in_document_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """score() returns float32 yes-minus-no diffs, preserving batched order."""
    _install_stub_modules(monkeypatch)
    reranker = Qwen3CrossEncoderReranker(approved_reranker_spec("qwen3-reranker-4b"))

    scores = reranker.score("q", tuple(f"doc {i}" for i in range(6)))

    expected = [(10.0 + p) - 0.5 * p for p in range(6)]
    np.testing.assert_allclose(scores, expected, rtol=0, atol=1e-6)
    assert scores.dtype == np.float32
    assert scores.shape == (6,)


def test_dtype_kwarg_reaches_from_pretrained_with_modern_spelling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """transformers >=5: compute dtype is forwarded as ``dtype=``."""
    recorder = _install_stub_modules(monkeypatch)

    reranker = Qwen3CrossEncoderReranker(approved_reranker_spec("qwen3-reranker-4b"))
    assert reranker.spec.batch_size == 4

    warm = Qwen3CrossEncoderReranker(
        approved_reranker_spec("qwen3-reranker-4b"), model_dtype="bfloat16"
    )
    assert warm.spec.revision.startswith("22e6836")

    assert recorder.dtype_kwargs_seen[0] == {"dtype": "torch.float32"}
    assert recorder.dtype_kwargs_seen[-1] == {"dtype": "torch.bfloat16"}
    assert all("torch_dtype" not in seen for seen in recorder.dtype_kwargs_seen)


def test_dtype_kwarg_falls_back_to_torch_dtype_on_type_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """transformers <5 rejects ``dtype=`` and still receives ``torch_dtype=``."""
    recorder = _install_stub_modules(monkeypatch, reject_dtype_kwarg=True)

    Qwen3CrossEncoderReranker(approved_reranker_spec("qwen3-reranker-4b"))

    assert recorder.dtype_kwargs_seen == [
        {"dtype": "torch.float32"},
        {"torch_dtype": "torch.float32"},
    ]
    assert recorder.model_load_kwargs.get("revision", "").startswith("22e6836")
    assert recorder.model_load_kwargs["trust_remote_code"] is False
    assert recorder.model_load_kwargs["local_files_only"] is False


def test_unresolvable_judge_tokens_fail_loudly_at_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tokenizer that cannot map yes/no to one id must never yield a model."""
    broken = _StubTokenizer(
        _StubRerankRecorder(),
        judge_ids={"yes": None, "no": None},
        fallback_encode_ids=[7, 9],
    )
    _install_stub_modules(monkeypatch, tokenizer=broken)

    with pytest.raises(RerankModelError, match="single id"):
        Qwen3CrossEncoderReranker(approved_reranker_spec("qwen3-reranker-4b"))
