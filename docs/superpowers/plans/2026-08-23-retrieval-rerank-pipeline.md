# Retrieval + Reranker Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Đưa nhánh retrieval từ BM25-đơn-lẻ (`dense_weight=0.0`, không reranker, Recall@10 47.41%) lên đúng pipeline `Question → Metadata Filtering → Candidate Tables → BM25+Dense → RRF top-50 → Reranker → top-10 evidence tables`, và chọn `k*` bằng số đo F2 ∧ MRR5 trên tập gold.

**Architecture:** Không viết lại gì. `filtering.py::eligible_positions` (metadata filtering), `service.py::RetrievalService` (BM25), `dense_service.py::DenseRetrievalService` (dense), `fusion.py::FusionService` (RRF có sẵn, có test, chưa được nối vào đường live), `retrieval_scoring.py::sweep_k` (8 chỉ số, chưa có caller) đều đã tồn tại. Việc cần làm là: (1) thêm encoder Qwen3-Embedding-4B vào allowlist đã pin, (2) viết module reranker mới cùng khuôn `dense_encoder.py`, (3) tổng quát hoá `live_query.py` để đường live nhận được Fusion+Rerank thay vì chỉ BM25, (4) index offline trên Colab, (5) CLI `sweep-k` chốt `k*`.

**Tech Stack:** Python 3.11, pydantic v2 (frozen contracts), numpy, faiss (`IndexFlatIP`), `bm25s`, `sentence-transformers`, `transformers` (`AutoModelForSequenceClassification` cho cross-encoder), pytest, DuckDB/pyarrow (release Parquet).

## Global Constraints

Sao chép nguyên văn từ `docs/superpowers/specs/2026-08-23-target-architecture.md`:

- **Mọi model < 14B tham số** (áp dụng cả embedding lẫn reranker). Qwen3-Embedding-4B và Qwen3-Reranker-4B đều thoả.
- **N5 — Không quantize embedding/reranker.** `dtype: Literal["float32"]` giữ nguyên trong mọi spec. Lý do: đầu ra là điểm số liên tục dùng xếp hạng; nhiễu lượng tử hoá làm lệch thứ hạng ứng viên gần nhau và không Critic nào bắt được.
- **`revision` bắt buộc là SHA 40 ký tự hex** (`ModelRevision`) — index phải pin được model.
- **N1 — Hai nhánh độc lập.** Nhánh answering thất bại không được ghi đè đầu ra nhánh retrieval. Cụ thể: `retrieved` (dùng cho `relevant_docs`/`relevant_tables`) không bao giờ được thay bằng danh sách đã narrow theo scope cho answering (`exporter.py:238` giữ nguyên hai tên biến tách biệt).
- **Thứ tự retrieval-rank là bất biến chấm điểm.** Dashboard chấm MRR5, không chỉ set membership. Mọi hàm trả `table_ids` phải giữ đúng thứ tự điểm giảm dần; không được `ORDER BY table_id`.
- **Reranker chỉ xếp lại top-N của RRF (N = 50), không phải toàn corpus** — đây là điều làm nó chạy được trên máy local không GPU.
- **Encoder và reranker chạy tuần tự**, không đồng thời, để tránh tranh chấp VRAM.
- Compute: local RTX 3050 Laptop 6GB VRAM + Colab/Kaggle T4 16GB.
- Repo dùng `uv`; chạy test bằng `uv run pytest`. Lint: `uv run ruff check`. Type: `uv run mypy src`.

---

## File Structure

| File | Trạng thái | Trách nhiệm |
|---|---|---|
| `src/financial_report_qa/retrieval/dense_contracts.py` | Sửa | Thêm `"qwen3-embedding-4b"` vào `EncoderName`; nới `max_sequence_length` |
| `src/financial_report_qa/retrieval/dense_encoder.py` | Sửa | Thêm entry `_SPECS["qwen3-embedding-4b"]` (model_id + revision pinned) |
| `src/financial_report_qa/retrieval/rerank_contracts.py` | Tạo | `RerankerSpec`, `RerankedCandidate`, `RerankTrace` — contract đóng băng |
| `src/financial_report_qa/retrieval/reranker.py` | Tạo | `Reranker` Protocol, `Qwen3CrossEncoderReranker`, `rerank_candidates()` |
| `src/financial_report_qa/retrieval/live_query.py` | Sửa | Tổng quát hoá sang `TableRetriever` Protocol + reranker tuỳ chọn |
| `src/financial_report_qa/retrieval/sweep.py` | Tạo | Chạy pipeline trên gold, gọi `sweep_k`, sinh báo cáo JSON/Markdown |
| `src/financial_report_qa/retrieval/cli.py` | Sửa | Thêm subcommand `sweep-k` |
| `src/financial_report_qa/core/errors.py` | Sửa | Thêm `RerankError`, `RerankInputError`, `RerankModelError` |
| `src/financial_report_qa/submission/cli.py` | Sửa | Lắp `FusionService` + reranker, truyền xuống exporter |
| `src/financial_report_qa/submission/exporter.py` | Sửa | Nhận `TableRetriever` thay vì `RetrievalService` cụ thể |
| `notebooks/colab_index_qwen3_emb_4b.ipynb` | Tạo | Embed toàn corpus trên T4, checkpoint theo shard |
| `tests/unit/retrieval/test_rerank_contracts.py` | Tạo | Test contract |
| `tests/unit/retrieval/test_reranker.py` | Tạo | Test `rerank_candidates` với fake reranker |
| `tests/unit/retrieval/test_live_query.py` | Sửa | Test đường live có/không reranker |
| `tests/unit/retrieval/test_sweep.py` | Tạo | Test sinh báo cáo sweep |
| `tests/unit/retrieval/test_dense_encoder.py` | Sửa | Test spec Qwen3 nằm trong allowlist |

**Thứ tự phụ thuộc:** Task 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8. Task 7 (Colab index) độc lập, chạy nền song song từ đầu.

---

### Task 1: Encoder allowlist — thêm Qwen3-Embedding-4B

**Files:**
- Modify: `src/financial_report_qa/retrieval/dense_contracts.py:20`, `:32`
- Modify: `src/financial_report_qa/retrieval/dense_encoder.py:15-34`
- Test: `tests/unit/retrieval/test_dense_encoder.py`

**Interfaces:**
- Consumes: `DenseEncoderSpec` (đã có, `dense_contracts.py:25`)
- Produces: `approved_encoder_spec("qwen3-embedding-4b") -> DenseEncoderSpec` với `dimension=2560`, `max_sequence_length=8192`, `dtype="float32"`.

- [ ] **Step 1: Write the failing test**

Thêm vào `tests/unit/retrieval/test_dense_encoder.py`:

```python
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
```

Thêm `import re` ở đầu file nếu chưa có.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/retrieval/test_dense_encoder.py -v -k qwen3`
Expected: FAIL với `KeyError: 'qwen3-embedding-4b'`

- [ ] **Step 3: Nới contract**

Trong `src/financial_report_qa/retrieval/dense_contracts.py`, đổi dòng 20:

```python
EncoderName = Literal["bge-m3", "multilingual-e5-small", "qwen3-embedding-4b"]
```

và dòng 32:

```python
    max_sequence_length: Literal[512, 1024, 2048, 8192] = 512
```

- [ ] **Step 4: Thêm spec đã pin**

Trong `src/financial_report_qa/retrieval/dense_encoder.py`, thêm vào dict `_SPECS` (sau entry `multilingual-e5-small`):

```python
    "qwen3-embedding-4b": DenseEncoderSpec(
        name="qwen3-embedding-4b",
        model_id="Qwen/Qwen3-Embedding-4B",
        # Điền SHA commit thật của model trên Hugging Face trước khi chạy
        # build-dense-index; `ModelRevision` từ chối mọi chuỗi không phải
        # 40 ký tự hex, nên một revision sai sẽ hỏng ngay lúc import test.
        revision="0000000000000000000000000000000000000000",
        dimension=2560,
        max_sequence_length=8192,
        query_prefix="Instruct: Given a financial question, retrieve the "
        "table that contains the answer\nQuery: ",
        document_prefix="",
        batch_size=4,
    ),
```

**Trước khi commit:** lấy SHA thật bằng lệnh sau và thay vào `revision`:

```bash
uv run python -c "from huggingface_hub import HfApi; print(HfApi().model_info('Qwen/Qwen3-Embedding-4B').sha)"
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/unit/retrieval/test_dense_encoder.py -v`
Expected: PASS (mọi test, kể cả test cũ của bge-m3/e5)

- [ ] **Step 6: Chạy toàn bộ test dense để chắc contract nới không phá gì**

Run: `uv run pytest tests/unit/retrieval/ -v -k "dense or contract"`
Expected: PASS toàn bộ

- [ ] **Step 7: Commit**

```bash
git add src/financial_report_qa/retrieval/dense_contracts.py src/financial_report_qa/retrieval/dense_encoder.py tests/unit/retrieval/test_dense_encoder.py
git commit -m "feat(retrieval): pin Qwen3-Embedding-4B in the encoder allowlist"
```

---

### Task 2: Reranker contracts

**Files:**
- Create: `src/financial_report_qa/retrieval/rerank_contracts.py`
- Modify: `src/financial_report_qa/core/errors.py` (cuối file, sau `RetrievalArtifactError` cụm)
- Test: `tests/unit/retrieval/test_rerank_contracts.py`

**Interfaces:**
- Consumes: `_FrozenModel`, `TableId`, `NonEmptyString`, `Fingerprint`, `QuestionId`, `TableMetadata` từ `retrieval/contracts.py`; `ModelRevision` từ `retrieval/dense_contracts.py`.
- Produces:
  - `RerankerSpec(name, model_id, revision, max_sequence_length, dtype, device, batch_size)`
  - `reranker_spec_sha256(spec: RerankerSpec) -> str`
  - `RerankedCandidate(table_id, rank, rerank_score, fused_rank, fused_score, metadata, snippet)`
  - `RerankTrace(question_id, query, reranker_spec_sha256, input_count, results, empty_reason)`
  - `RerankEmptyReason = Literal["no_fused_candidates"]`

- [ ] **Step 1: Write the failing test**

Tạo `tests/unit/retrieval/test_rerank_contracts.py`:

```python
import pytest
from pydantic import ValidationError

from financial_report_qa.retrieval.contracts import TableMetadata
from financial_report_qa.retrieval.rerank_contracts import (
    RerankedCandidate,
    RerankerSpec,
    RerankTrace,
    reranker_spec_sha256,
)

_REVISION = "a" * 40
_TABLE_ID = "tbl_" + "b" * 64


def _metadata() -> TableMetadata:
    return TableMetadata(
        table_id=_TABLE_ID,
        doc_id="doc_" + "c" * 64,
        company_code="VCB",
        periods=("2023",),
        statement_type="balance_sheet",
        title="Bảng cân đối kế toán",
        source_path="VCB/2023/x/x_extracted.txt",
        line_start=1,
        line_end=10,
    )


def _spec(**overrides: object) -> RerankerSpec:
    defaults: dict[str, object] = {
        "name": "qwen3-reranker-4b",
        "model_id": "Qwen/Qwen3-Reranker-4B",
        "revision": _REVISION,
        "batch_size": 4,
    }
    return RerankerSpec(**{**defaults, **overrides})  # type: ignore[arg-type]


def test_spec_pins_float32_and_rejects_any_quantized_dtype() -> None:
    assert _spec().dtype == "float32"
    with pytest.raises(ValidationError):
        _spec(dtype="int8")


def test_spec_is_frozen() -> None:
    spec = _spec()
    with pytest.raises(ValidationError):
        spec.batch_size = 8  # type: ignore[misc]


def test_spec_rejects_a_revision_that_is_not_a_40_char_sha() -> None:
    with pytest.raises(ValidationError):
        _spec(revision="main")


def test_spec_sha256_is_stable_and_distinguishes_specs() -> None:
    first = reranker_spec_sha256(_spec())
    assert first == reranker_spec_sha256(_spec())
    assert first != reranker_spec_sha256(_spec(batch_size=8))


def test_candidate_rejects_a_non_finite_score() -> None:
    with pytest.raises(ValidationError):
        RerankedCandidate(
            table_id=_TABLE_ID,
            rank=1,
            rerank_score=float("inf"),
            fused_rank=1,
            fused_score=0.5,
            metadata=_metadata(),
            snippet="x",
        )


def test_trace_defaults_to_no_results() -> None:
    trace = RerankTrace(
        query="doanh thu thuần 2023",
        reranker_spec_sha256="d" * 64,
        input_count=0,
        empty_reason="no_fused_candidates",
    )
    assert trace.results == ()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/retrieval/test_rerank_contracts.py -v`
Expected: FAIL với `ModuleNotFoundError: No module named 'financial_report_qa.retrieval.rerank_contracts'`

- [ ] **Step 3: Thêm error types**

Trong `src/financial_report_qa/core/errors.py`, thêm sau lớp `FusionArtifactError`:

```python
class RerankError(RetrievalError):
    """Cross-encoder reranking failed."""


class RerankInputError(RerankError):
    """Caller supplied invalid rerank input."""


class RerankModelError(RerankError):
    """Pinned reranker model is unavailable or produced an unusable score."""
```

- [ ] **Step 4: Viết contract**

Tạo `src/financial_report_qa/retrieval/rerank_contracts.py`:

```python
"""Frozen contracts for cross-encoder reranking of fused table candidates.

The reranker never sees the whole corpus: it rescoring only the top-N output
of weighted RRF (`fusion.py`), which is what keeps it runnable on a CPU-only
local machine. Both the fused rank and the rerank score are kept on every
candidate so a reordering decision stays auditable after the fact.
"""

from __future__ import annotations

import hashlib
import math
from typing import Literal

from pydantic import Field, field_validator

from financial_report_qa.retrieval.contracts import (
    Fingerprint,
    NonEmptyString,
    QuestionId,
    TableId,
    TableMetadata,
    _FrozenModel,
)
from financial_report_qa.retrieval.dense_artifacts import canonical_json_bytes
from financial_report_qa.retrieval.dense_contracts import ModelRevision

RerankerName = Literal["qwen3-reranker-4b"]
RerankEmptyReason = Literal["no_fused_candidates"]

#: Số ứng viên RRF đưa vào reranker. Spec §5.3: rerank top-50, không phải
#: toàn corpus -- đây là điều làm bước này chạy được trên CPU local.
DEFAULT_RERANK_DEPTH = 50


class RerankerSpec(_FrozenModel):
    """Pinned cross-encoder behavior that identifies a rerank run."""

    name: RerankerName
    model_id: NonEmptyString
    revision: ModelRevision
    max_sequence_length: Literal[512, 1024, 2048, 8192] = 2048
    # N5: điểm số liên tục dùng để xếp hạng -- không bao giờ quantize.
    dtype: Literal["float32"] = "float32"
    device: Literal["cpu", "cuda"] = "cpu"
    batch_size: int = Field(gt=0)


def reranker_spec_sha256(spec: RerankerSpec) -> str:
    """Return the canonical identity digest for one pinned reranker spec."""
    return hashlib.sha256(canonical_json_bytes(spec.model_dump(mode="json"))).hexdigest()


class RerankedCandidate(_FrozenModel):
    """One candidate after cross-encoder rescoring, with its fused origin kept."""

    table_id: TableId
    rank: int = Field(ge=1)
    rerank_score: float
    fused_rank: int = Field(ge=1)
    fused_score: float
    metadata: TableMetadata
    snippet: str

    @field_validator("rerank_score", "fused_score")
    @classmethod
    def validate_finite_score(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("score must be finite")
        return value


class RerankTrace(_FrozenModel):
    """Auditable rerank outcome for one query."""

    question_id: QuestionId | None = None
    query: str
    reranker_spec_sha256: Fingerprint
    input_count: int = Field(ge=0)
    results: tuple[RerankedCandidate, ...] = ()
    empty_reason: RerankEmptyReason | None = None
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/unit/retrieval/test_rerank_contracts.py -v`
Expected: PASS (6 tests)

- [ ] **Step 6: Commit**

```bash
git add src/financial_report_qa/retrieval/rerank_contracts.py src/financial_report_qa/core/errors.py tests/unit/retrieval/test_rerank_contracts.py
git commit -m "feat(retrieval): add frozen contracts for cross-encoder reranking"
```

---

### Task 3: Reranker — hàm thuần + Protocol

**Files:**
- Create: `src/financial_report_qa/retrieval/reranker.py`
- Test: `tests/unit/retrieval/test_reranker.py`

**Interfaces:**
- Consumes: `RerankerSpec`, `RerankedCandidate`, `RerankTrace`, `reranker_spec_sha256`, `DEFAULT_RERANK_DEPTH` (Task 2); `FusedCandidate` từ `retrieval/fusion_contracts.py`.
- Produces:
  - `class Reranker(Protocol)` với `spec: RerankerSpec` và `score(query: str, documents: Sequence[str]) -> np.ndarray`
  - `approved_reranker_spec(name: RerankerName) -> RerankerSpec`
  - `rerank_candidates(query, candidates: Sequence[FusedCandidate], reranker: Reranker, *, k: int, depth: int = DEFAULT_RERANK_DEPTH, question_id: str | None = None) -> RerankTrace`
  - `class Qwen3CrossEncoderReranker` — cài đặt thật

- [ ] **Step 1: Write the failing test**

Tạo `tests/unit/retrieval/test_reranker.py`:

```python
import numpy as np
import pytest

from financial_report_qa.core.errors import RerankInputError, RerankModelError
from financial_report_qa.retrieval.contracts import TableMetadata
from financial_report_qa.retrieval.fusion_contracts import FusedCandidate
from financial_report_qa.retrieval.rerank_contracts import RerankerSpec
from financial_report_qa.retrieval.reranker import approved_reranker_spec, rerank_candidates


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
        _candidate(chr(ord("a") + i), fused_rank=i + 1, fused_score=1.0 - i / 10)
        for i in range(5)
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
        rerank_candidates("q", (_candidate("a", fused_rank=1, fused_score=0.9),),
                          _NaNReranker({}), k=1)


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/retrieval/test_reranker.py -v`
Expected: FAIL với `ModuleNotFoundError: No module named 'financial_report_qa.retrieval.reranker'`

- [ ] **Step 3: Viết module**

Tạo `src/financial_report_qa/retrieval/reranker.py`:

```python
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
        # Điền SHA commit thật trước khi chạy live (xem Task 3 Step 4).
        revision="0000000000000000000000000000000000000000",
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
```

- [ ] **Step 4: Điền revision thật cho reranker**

```bash
uv run python -c "from huggingface_hub import HfApi; print(HfApi().model_info('Qwen/Qwen3-Reranker-4B').sha)"
```

Thay giá trị in ra vào trường `revision` của `_SPECS["qwen3-reranker-4b"]`.

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/unit/retrieval/test_reranker.py -v`
Expected: PASS (8 tests)

- [ ] **Step 6: Lint + type**

Run: `uv run ruff check src/financial_report_qa/retrieval/reranker.py && uv run mypy src/financial_report_qa/retrieval/reranker.py`
Expected: không có lỗi

- [ ] **Step 7: Commit**

```bash
git add src/financial_report_qa/retrieval/reranker.py tests/unit/retrieval/test_reranker.py
git commit -m "feat(retrieval): add pinned Qwen3 cross-encoder reranker over fused candidates"
```

---

### Task 4: Nối Fusion + Rerank vào đường live (`live_query.py`)

**Files:**
- Modify: `src/financial_report_qa/retrieval/live_query.py` (viết lại toàn bộ, 28 dòng)
- Test: `tests/unit/retrieval/test_live_query.py`

**Interfaces:**
- Consumes: `rerank_candidates` (Task 3); `FusionService.retrieve` (`fusion.py:78`), `RetrievalService.retrieve` (`service.py`), `parse_query_entities`, `to_retrieval_filters`.
- Produces:
  - `class TableRetriever(Protocol)` — `retrieve(query, *, filters, k, question_id) -> trace có .results` với mỗi phần tử có `.table_id`
  - `retrieve_candidate_table_ids(question, service, *, k=10, reranker=None, rerank_depth=DEFAULT_RERANK_DEPTH) -> tuple[TableId, ...]`

Chữ ký cũ `retrieve_candidate_table_ids(question, service, *, k=10)` giữ nguyên tương thích — hai tham số mới đều có mặc định, nên `exporter.py:205`, `exporter.py:425`, `submission/cli.py:392` không phải sửa để tiếp tục chạy.

- [ ] **Step 1: Write the failing test**

Ghi đè `tests/unit/retrieval/test_live_query.py` bằng:

```python
import numpy as np
import pytest

from financial_report_qa.core.errors import RerankInputError
from financial_report_qa.retrieval.contracts import (
    RetrievalCandidate,
    RetrievalTrace,
    TableMetadata,
)
from financial_report_qa.retrieval.fusion_contracts import (
    FusedCandidate,
    FusionTrace,
    FusionWeights,
)
from financial_report_qa.retrieval.live_query import retrieve_candidate_table_ids
from financial_report_qa.retrieval.rerank_contracts import RerankerSpec
from financial_report_qa.planning.entity_parser import parse_query_entities

_QUESTION = "Doanh thu thuần của VCB năm 2023 là bao nhiêu?"


def _table_id(suffix: str) -> str:
    return "tbl_" + suffix * 64


def _metadata(suffix: str) -> TableMetadata:
    return TableMetadata(
        table_id=_table_id(suffix),
        doc_id="doc_" + "0" * 64,
        company_code="VCB",
        periods=("2023",),
        source_path="VCB/2023/x/x_extracted.txt",
        line_start=1,
        line_end=2,
    )


class _FakeBm25Service:
    def __init__(self, suffixes: tuple[str, ...]) -> None:
        self._suffixes = suffixes
        self.last_k: int | None = None
        self.last_filters = None

    def retrieve(self, query, *, filters, k=10, question_id=None):  # type: ignore[no-untyped-def]
        self.last_k = k
        self.last_filters = filters
        return RetrievalTrace(
            query=query,
            query_tokens=(),
            eligible_count=len(self._suffixes),
            filter_decisions=(),
            results=tuple(
                RetrievalCandidate(
                    table_id=_table_id(suffix),
                    score=1.0 - index / 10,
                    rank=index + 1,
                    metadata=_metadata(suffix),
                    snippet=f"snippet-{suffix}",
                    matched_tokens=("doanh",),
                )
                for index, suffix in enumerate(self._suffixes[:k])
            ),
        )


class _FakeFusionService:
    def __init__(self, suffixes: tuple[str, ...]) -> None:
        self._suffixes = suffixes
        self.last_k: int | None = None

    def retrieve(self, query, *, filters, k=10, question_id=None):  # type: ignore[no-untyped-def]
        self.last_k = k
        return FusionTrace(
            query=query,
            weights=FusionWeights(bm25=1, dense=1),
            entities=parse_query_entities(query),
            eligible_count=len(self._suffixes),
            bm25_candidate_count=len(self._suffixes),
            dense_candidate_count=len(self._suffixes),
            results=tuple(
                FusedCandidate(
                    table_id=_table_id(suffix),
                    rank=index + 1,
                    fused_score=1.0 - index / 10,
                    contradiction_count=0,
                    metadata=_metadata(suffix),
                    snippet=f"snippet-{suffix}",
                )
                for index, suffix in enumerate(self._suffixes[:k])
            ),
        )


class _ReversingReranker:
    """Cho điểm ngược lại thứ tự đầu vào, để thấy rõ reranker có tác dụng."""

    def __init__(self) -> None:
        self.spec = RerankerSpec(
            name="qwen3-reranker-4b",
            model_id="Qwen/Qwen3-Reranker-4B",
            revision="a" * 40,
            batch_size=4,
        )

    def score(self, query, documents):  # type: ignore[no-untyped-def]
        return np.asarray(range(len(documents)), dtype=np.float32)


def test_bm25_only_path_is_unchanged_when_no_reranker_is_supplied() -> None:
    service = _FakeBm25Service(("a", "b", "c"))

    table_ids = retrieve_candidate_table_ids(_QUESTION, service, k=2)

    assert table_ids == (_table_id("a"), _table_id("b"))
    assert service.last_k == 2


def test_filters_are_derived_from_the_question_text() -> None:
    service = _FakeBm25Service(("a",))

    retrieve_candidate_table_ids(_QUESTION, service, k=1)

    assert service.last_filters is not None
    assert service.last_filters.company_codes == ("VCB",)


def test_reranker_reorders_the_fused_top_and_truncates_to_k() -> None:
    service = _FakeFusionService(("a", "b", "c"))

    table_ids = retrieve_candidate_table_ids(
        _QUESTION, service, k=2, reranker=_ReversingReranker(), rerank_depth=3
    )

    assert table_ids == (_table_id("c"), _table_id("b"))


def test_reranking_asks_the_retriever_for_the_full_rerank_depth_not_just_k() -> None:
    service = _FakeFusionService(tuple("abcdefghij"))

    retrieve_candidate_table_ids(
        _QUESTION, service, k=2, reranker=_ReversingReranker(), rerank_depth=8
    )

    assert service.last_k == 8


def test_rerank_depth_below_k_is_rejected() -> None:
    with pytest.raises(RerankInputError):
        retrieve_candidate_table_ids(
            _QUESTION, _FakeFusionService(("a",)), k=10,
            reranker=_ReversingReranker(), rerank_depth=5,
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/retrieval/test_live_query.py -v`
Expected: FAIL — `TypeError: retrieve_candidate_table_ids() got an unexpected keyword argument 'reranker'`

- [ ] **Step 3: Viết lại `live_query.py`**

Ghi đè `src/financial_report_qa/retrieval/live_query.py`:

```python
"""Retrieval for a raw, never-before-seen question.

Full target pipeline (§5.1 of the 2026-08-23 target architecture):

    question -> entities -> metadata filters -> candidate tables
             -> BM25 + dense -> weighted RRF -> top-N
             -> cross-encoder rerank -> top-k

The metadata-filter step is not implemented here: every retriever already
filters first (`filtering.py::eligible_positions`, shared by BM25, dense and
fusion), so the only thing this module adds is deriving those filters from
raw question text -- `to_retrieval_filters` drops any field the entity parser
itself flagged as ambiguous, so an uncertain parse widens the candidate set
rather than silently emptying it.

`service` is deliberately typed as a Protocol, not as `RetrievalService`:
`FusionService` (BM25 + dense under one RRF) satisfies the same shape, so
switching the live path from BM25-only to fused retrieval is a wiring change
at the call site, not a change here.
"""

from __future__ import annotations

from typing import Protocol

from financial_report_qa.core.errors import RerankInputError
from financial_report_qa.planning.entity_contracts import to_retrieval_filters
from financial_report_qa.planning.entity_parser import parse_query_entities
from financial_report_qa.retrieval.contracts import RetrievalFilters, TableId
from financial_report_qa.retrieval.rerank_contracts import DEFAULT_RERANK_DEPTH
from financial_report_qa.retrieval.reranker import Reranker, rerank_candidates


class _RankedResult(Protocol):
    table_id: str


class _RetrievalTraceLike(Protocol):
    results: tuple[_RankedResult, ...]


class TableRetriever(Protocol):
    """Anything that ranks tables under metadata filters: BM25 or fusion."""

    def retrieve(
        self,
        query: str,
        *,
        filters: RetrievalFilters,
        k: int = 10,
        question_id: str | None = None,
    ) -> _RetrievalTraceLike: ...


def retrieve_candidate_table_ids(
    question: str,
    service: TableRetriever,
    *,
    k: int = 10,
    reranker: Reranker | None = None,
    rerank_depth: int = DEFAULT_RERANK_DEPTH,
) -> tuple[TableId, ...]:
    """Rank candidate tables for one raw question, in retrieval-rank order.

    With a `reranker`, the retriever is asked for `rerank_depth` candidates
    (not `k`) so the cross-encoder has a real pool to reorder; the reranked
    list is then cut to `k`.
    """
    if reranker is not None and rerank_depth < k:
        raise RerankInputError("rerank_depth must be at least k")

    entities = parse_query_entities(question)
    filters = to_retrieval_filters(entities)
    depth = rerank_depth if reranker is not None else k
    trace = service.retrieve(question, filters=filters, k=depth)

    if reranker is None:
        return tuple(candidate.table_id for candidate in trace.results)

    rerank_trace = rerank_candidates(
        question,
        trace.results,  # type: ignore[arg-type]
        reranker,
        k=k,
        depth=rerank_depth,
    )
    return tuple(candidate.table_id for candidate in rerank_trace.results)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/retrieval/test_live_query.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Chắc chắn không phá call site cũ**

Run: `uv run pytest tests/unit/submission/ tests/integration/ -v`
Expected: PASS toàn bộ (chữ ký cũ vẫn tương thích vì hai tham số mới có mặc định)

- [ ] **Step 6: Commit**

```bash
git add src/financial_report_qa/retrieval/live_query.py tests/unit/retrieval/test_live_query.py
git commit -m "feat(retrieval): let the live path take a fused retriever and an optional reranker"
```

---

### Task 5: Sweep k — chạy pipeline trên gold và chốt `k*`

**Files:**
- Create: `src/financial_report_qa/retrieval/sweep.py`
- Test: `tests/unit/retrieval/test_sweep.py`

**Interfaces:**
- Consumes: `sweep_k`, `macro_f2`, `macro_mrr5` (`retrieval_scoring.py`); `GoldRetrievalQuestion` (`contracts.py:78`); `retrieve_candidate_table_ids` (Task 4).
- Produces:
  - `SweepResult(k, f2, mrr5)` — dataclass frozen
  - `run_sweep(questions, service, *, ks, reranker=None, rerank_depth=DEFAULT_RERANK_DEPTH) -> tuple[SweepResult, ...]`
  - `recommend_k(results: Sequence[SweepResult]) -> int`
  - `render_sweep_markdown(results: Sequence[SweepResult], recommended_k: int) -> str`

- [ ] **Step 1: Write the failing test**

Tạo `tests/unit/retrieval/test_sweep.py`:

```python
import pytest

from financial_report_qa.retrieval.sweep import (
    SweepResult,
    recommend_k,
    render_sweep_markdown,
    run_sweep,
)
from tests.unit.retrieval.test_live_query import _FakeBm25Service, _table_id


class _GoldStub:
    def __init__(self, question: str, gold_table_ids: tuple[str, ...]) -> None:
        self.question = question
        self.gold_table_ids = gold_table_ids


def test_sweep_reports_f2_and_mrr5_at_every_k() -> None:
    service = _FakeBm25Service(("a", "b", "c"))
    questions = (_GoldStub("Doanh thu VCB 2023?", (_table_id("b"),)),)

    results = run_sweep(questions, service, ks=(1, 2, 3))

    assert [item.k for item in results] == [1, 2, 3]
    # gold nằm ở hạng 2 -> k=1 trượt hoàn toàn, k>=2 bắt được.
    assert results[0].f2 == pytest.approx(0.0)
    assert results[0].mrr5 == pytest.approx(0.0)
    assert results[1].mrr5 == pytest.approx(0.5)
    assert results[2].mrr5 == pytest.approx(0.5)


def test_precision_penalty_makes_f2_fall_as_k_grows_past_the_gold() -> None:
    service = _FakeBm25Service(tuple("abcdefghij"))
    questions = (_GoldStub("Doanh thu VCB 2023?", (_table_id("a"),)),)

    results = {item.k: item for item in run_sweep(questions, service, ks=(1, 10))}

    assert results[1].f2 > results[10].f2


def test_recommend_k_prefers_the_best_f2_and_breaks_ties_on_mrr5() -> None:
    results = (
        SweepResult(k=1, f2=0.40, mrr5=0.40),
        SweepResult(k=5, f2=0.60, mrr5=0.55),
        SweepResult(k=8, f2=0.60, mrr5=0.58),
        SweepResult(k=10, f2=0.55, mrr5=0.58),
    )

    assert recommend_k(results) == 8


def test_recommend_k_rejects_an_empty_sweep() -> None:
    with pytest.raises(ValueError):
        recommend_k(())


def test_markdown_marks_the_recommended_row() -> None:
    results = (SweepResult(k=5, f2=0.6, mrr5=0.5), SweepResult(k=10, f2=0.5, mrr5=0.5))

    rendered = render_sweep_markdown(results, recommended_k=5)

    assert "| 5 |" in rendered
    assert "**5**" in rendered or "<-- k*" in rendered
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/retrieval/test_sweep.py -v`
Expected: FAIL với `ModuleNotFoundError: No module named 'financial_report_qa.retrieval.sweep'`

- [ ] **Step 3: Viết module**

Tạo `src/financial_report_qa/retrieval/sweep.py`:

```python
"""Chọn k* cho nhánh retrieval bằng số đo, không bằng cảm tính.

`retrieval_scoring.sweep_k` đã cài đúng hai công thức của dashboard nhưng
chưa có caller: module này là caller đó. Nó chạy đúng pipeline live
(`retrieve_candidate_table_ids`, tức là đã gồm metadata filtering, fusion và
reranker nếu được truyền vào) trên tập gold, cắt danh sách đã xếp hạng ở
từng k, rồi báo cáo F2 macro và MRR5 macro song song.

`recommend_k` chọn F2 cao nhất và phá thế hoà bằng MRR5, KHÔNG tối đa hoá
một chỉ số đơn lẻ: F2 phạt precision khi k lớn, còn MRR5 gần như không đổi
khi k > 5, nên tối đa hoá riêng MRR5 sẽ đẩy k lên vô ích và mất điểm F2.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from financial_report_qa.retrieval.live_query import (
    TableRetriever,
    retrieve_candidate_table_ids,
)
from financial_report_qa.retrieval.rerank_contracts import DEFAULT_RERANK_DEPTH
from financial_report_qa.retrieval.reranker import Reranker
from financial_report_qa.retrieval.retrieval_scoring import sweep_k

DEFAULT_KS: tuple[int, ...] = (1, 2, 3, 5, 8, 10, 15)


class _GoldQuestion(Protocol):
    question: str
    gold_table_ids: tuple[str, ...]


@dataclass(frozen=True)
class SweepResult:
    """F2 macro và MRR5 macro tại một giá trị k."""

    k: int
    f2: float
    mrr5: float


def run_sweep(
    questions: Sequence[_GoldQuestion],
    service: TableRetriever,
    *,
    ks: Sequence[int] = DEFAULT_KS,
    reranker: Reranker | None = None,
    rerank_depth: int = DEFAULT_RERANK_DEPTH,
) -> tuple[SweepResult, ...]:
    """Chạy pipeline một lần ở k lớn nhất, rồi cắt cho mọi k nhỏ hơn.

    Chạy đúng một lần cho mỗi câu (ở `max(ks)`) thay vì một lần cho mỗi
    (câu, k): danh sách trả về đã ở đúng thứ tự retrieval-rank nên cắt ngắn
    cho k nhỏ hơn cho kết quả y hệt như truy hồi lại với k đó, với chi phí
    bằng 1/len(ks).
    """
    if not ks:
        raise ValueError("ks must not be empty")
    depth = max(ks)
    effective_rerank_depth = max(rerank_depth, depth)

    ranked: dict[int, list[str]] = {}
    gold: dict[int, list[str]] = {}
    for index, question in enumerate(questions):
        ranked[index] = list(
            retrieve_candidate_table_ids(
                question.question,
                service,
                k=depth,
                reranker=reranker,
                rerank_depth=effective_rerank_depth,
            )
        )
        gold[index] = list(question.gold_table_ids)

    scored = sweep_k(ranked, gold, ks=tuple(ks))
    return tuple(
        SweepResult(k=k, f2=scored[k]["f2"], mrr5=scored[k]["mrr5"]) for k in ks
    )


def recommend_k(results: Sequence[SweepResult]) -> int:
    """F2 cao nhất; hoà thì MRR5 cao hơn thắng; vẫn hoà thì k nhỏ hơn thắng."""
    if not results:
        raise ValueError("cannot recommend k from an empty sweep")
    best = max(results, key=lambda item: (item.f2, item.mrr5, -item.k))
    return best.k


def render_sweep_markdown(results: Sequence[SweepResult], recommended_k: int) -> str:
    """Bảng Markdown một dòng một k, đánh dấu k* đã chọn."""
    lines = ["| k | F2 macro | MRR5 macro | |", "|---|---|---|---|"]
    for item in results:
        marker = " <-- k*" if item.k == recommended_k else ""
        lines.append(f"| {item.k} | {item.f2:.4f} | {item.mrr5:.4f} |{marker} |")
    return "\n".join(lines) + "\n"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/retrieval/test_sweep.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/financial_report_qa/retrieval/sweep.py tests/unit/retrieval/test_sweep.py
git commit -m "feat(retrieval): sweep k over F2 and MRR5 to pick k* by measurement"
```

---

### Task 6: CLI `retrieval sweep-k`

**Files:**
- Modify: `src/financial_report_qa/retrieval/cli.py` (thêm subcommand + nhánh dispatch)
- Test: `tests/unit/retrieval/test_sweep_cli.py` (tạo)

**Interfaces:**
- Consumes: `run_sweep`, `recommend_k`, `render_sweep_markdown` (Task 5); `load_gold_questions(path, release, *, require_count, question_ids)` (`gold.py:89`); `resolve_retrieval_release`, `load_bm25_index`, `RetrievalService`.
- Produces: subcommand `sweep-k` ghi `<output>.json` + `<output>.md`, in `k*=<k>` ra stdout, exit code 0.

- [ ] **Step 1: Write the failing test**

Tạo `tests/unit/retrieval/test_sweep_cli.py`:

```python
import json
from pathlib import Path

from financial_report_qa.retrieval.sweep import SweepResult
from financial_report_qa.retrieval.cli import write_sweep_report


def test_report_writes_both_json_and_markdown(tmp_path: Path) -> None:
    results = (SweepResult(k=5, f2=0.61, mrr5=0.52), SweepResult(k=10, f2=0.55, mrr5=0.52))

    json_path, markdown_path = write_sweep_report(results, 5, tmp_path / "day8-sweep")

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["recommended_k"] == 5
    assert payload["results"] == [
        {"k": 5, "f2": 0.61, "mrr5": 0.52},
        {"k": 10, "f2": 0.55, "mrr5": 0.52},
    ]
    assert "| 5 |" in markdown_path.read_text(encoding="utf-8")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/retrieval/test_sweep_cli.py -v`
Expected: FAIL với `ImportError: cannot import name 'write_sweep_report'`

- [ ] **Step 3: Thêm import + helper vào `cli.py`**

Thêm vào khối import của `src/financial_report_qa/retrieval/cli.py` (sau import `retrieval.service`):

```python
from financial_report_qa.retrieval.sweep import (
    DEFAULT_KS,
    SweepResult,
    recommend_k,
    render_sweep_markdown,
    run_sweep,
)
```

Thêm hàm này ngay trước `def main(` (khoảng dòng 257):

```python
def write_sweep_report(
    results: Sequence[SweepResult], recommended_k: int, output_stem: Path
) -> tuple[Path, Path]:
    """Ghi báo cáo sweep ra <stem>.json và <stem>.md; trả về hai đường dẫn."""
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_stem.with_suffix(".json")
    markdown_path = output_stem.with_suffix(".md")
    payload = {
        "recommended_k": recommended_k,
        "results": [
            {"k": item.k, "f2": item.f2, "mrr5": item.mrr5} for item in results
        ],
    }
    write_text_atomic(json_path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    write_text_atomic(markdown_path, render_sweep_markdown(results, recommended_k))
    return json_path, markdown_path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/retrieval/test_sweep_cli.py -v`
Expected: PASS

- [ ] **Step 5: Thêm subcommand vào `_parser()`**

Trong `_parser()` của `src/financial_report_qa/retrieval/cli.py`, thêm sau khối `cleanup = commands.add_parser("cleanup-day9-data")`:

```python
    sweep = commands.add_parser("sweep-k")
    sweep.add_argument("--release-lock", type=Path, required=True)
    sweep.add_argument("--bm25-index", type=Path, required=True)
    sweep.add_argument("--gold", type=Path, required=True)
    sweep.add_argument("--output-stem", type=Path, required=True)
    sweep.add_argument(
        "--ks",
        type=int,
        nargs="+",
        default=list(DEFAULT_KS),
        help="Các giá trị k cần đo (mặc định 1 2 3 5 8 10 15).",
    )
```

- [ ] **Step 6: Thêm nhánh dispatch trong `main()`**

Trong `main()` của `src/financial_report_qa/retrieval/cli.py`, thêm trước dòng cuối `raise AssertionError(...)`:

```python
        if args.command == "sweep-k":
            release = resolve_retrieval_release(args.release_lock, repo_root=Path.cwd())
            index = load_bm25_index(args.bm25_index)
            if index.manifest.dataset_fingerprint != release.dataset_fingerprint:
                raise RetrievalArtifactError(
                    "--bm25-index dataset_fingerprint does not match --release-lock"
                )
            questions = load_gold_questions(args.gold, release)
            results = run_sweep(questions, RetrievalService(index), ks=tuple(args.ks))
            best = recommend_k(results)
            json_path, markdown_path = write_sweep_report(results, best, args.output_stem)
            print(render_sweep_markdown(results, best), end="")
            print(f"k*={best}")
            print(json_path)
            print(markdown_path)
            return 0
```

- [ ] **Step 7: Chạy thật trên gold hiện có**

```bash
uv run financial-report-qa retrieval sweep-k --release-lock data/qa/week1_pilot_422df141c935/dataset-pilot-v1.json --bm25-index data/indexes/bm25-v4/422df141c935d46bfd14302abec50f32380e6e4c012159f8ad0ae5560c8a446a --gold data/qa/retrieval-gold-v1.jsonl --output-stem artifacts/evaluations/sweep-bm25-baseline
```

Expected: bảng 7 dòng + `k*=<số>`. Đây là **baseline BM25-only** — ghi lại con số F2/MRR5 tại `k*` vào commit message; mọi cải thiện ở Task 8 phải đo lại đúng lệnh này để so sánh có nghĩa.

Nếu `load_gold_questions` báo lỗi thiếu số câu, thêm `require_count=len(...)` phù hợp hoặc truyền `--gold` đúng file 120 câu đã kiểm chứng (`data/qa/retrieval-gold-v1.jsonl`, 120 dòng).

- [ ] **Step 8: Commit**

```bash
git add src/financial_report_qa/retrieval/cli.py tests/unit/retrieval/test_sweep_cli.py artifacts/evaluations/sweep-bm25-baseline.json artifacts/evaluations/sweep-bm25-baseline.md
git commit -m "feat(retrieval): add sweep-k CLI and record the BM25-only baseline"
```

---

### Task 7: Index Qwen3-Embedding-4B offline trên Colab

**Files:**
- Create: `notebooks/colab_index_qwen3_emb_4b.ipynb`

Task này **độc lập với Task 2–6**, chạy nền song song từ đầu vì embed toàn corpus tốn nhiều giờ Colab. Nó chỉ phụ thuộc Task 1 (spec đã pin).

**Interfaces:**
- Consumes: `approved_encoder_spec("qwen3-embedding-4b")` (Task 1), `build_dense_corpus`/`save_dense_corpus` (`dense_corpus.py`), `build_dense_index`/`save_dense_index` (`dense_index.py`).
- Produces: thư mục index tại `data/indexes/dense-qwen3-4b/<dataset_fingerprint>/` tải về từ Colab, dùng được bởi `load_dense_index` ở local.

- [ ] **Step 1: Tạo notebook với 6 cell theo thứ tự**

Cell 1 — cài đặt (pin phiên bản để lần chạy sau tái lập được):

```python
!pip install -q "sentence-transformers==3.0.1" "transformers==4.44.2" "faiss-cpu==1.8.0"
!nvidia-smi
```

Cell 2 — mount + clone repo, cài package:

```python
from google.colab import drive
drive.mount('/content/drive')
!git clone https://github.com/TranVu2005/financial-assistant /content/repo
%cd /content/repo
!pip install -q -e .
```

Cell 3 — dựng corpus (chạy trên CPU, nhanh):

```python
from pathlib import Path
from financial_report_qa.retrieval.dense_corpus import build_dense_corpus, save_dense_corpus
from financial_report_qa.retrieval.release import resolve_retrieval_release

release = resolve_retrieval_release(Path("data/qa/week1_pilot_422df141c935/dataset-pilot-v1.json"),
                                    repo_root=Path.cwd())
corpus = build_dense_corpus(release)
save_dense_corpus(corpus, Path("/content/drive/MyDrive/vifinqa/dense-corpus"))
print(len(corpus.documents), "documents")
```

Cell 4 — embed theo shard, checkpoint từng shard (session timeout không mất việc):

```python
import numpy as np, torch
from pathlib import Path
from financial_report_qa.retrieval.dense_encoder import (
    SentenceTransformerDenseEncoder, approved_encoder_spec,
)

SHARD = 2048
out = Path("/content/drive/MyDrive/vifinqa/qwen3-shards"); out.mkdir(parents=True, exist_ok=True)

spec = approved_encoder_spec("qwen3-embedding-4b").model_copy(update={"device": "cuda"})
encoder = SentenceTransformerDenseEncoder(spec)
texts = [document.text for document in corpus.documents]

for start in range(0, len(texts), SHARD):
    target = out / f"shard_{start:07d}.npy"
    if target.exists():
        continue  # đã xong ở lần chạy trước -- bỏ qua, không tính lại
    vectors = encoder.encode_documents(texts[start:start + SHARD])
    np.save(target, vectors.astype(np.float32))
    print(target.name, vectors.shape, flush=True)
```

**Lưu ý N5:** cell này giữ `float32` khi ghi ra đĩa. Nếu VRAM T4 không đủ, giảm `SHARD` và `spec.batch_size` — **không** đổi `dtype`.

Cell 5 — ghép shard và dựng FAISS index:

```python
import numpy as np, faiss
from pathlib import Path
from financial_report_qa.retrieval.dense_index import build_dense_index, save_dense_index

shards = sorted(Path("/content/drive/MyDrive/vifinqa/qwen3-shards").glob("shard_*.npy"))
vectors = np.concatenate([np.load(path) for path in shards], axis=0).astype(np.float32)
assert vectors.shape[0] == len(corpus.documents), (vectors.shape[0], len(corpus.documents))

index = build_dense_index(corpus, vectors, spec.model_copy(update={"device": "cpu"}))
save_dense_index(index, Path("/content/drive/MyDrive/vifinqa/dense-qwen3-4b"))
```

Cell 6 — kiểm tra tính toàn vẹn trước khi tải về:

```python
from financial_report_qa.retrieval.dense_index import load_dense_index
from pathlib import Path

loaded = load_dense_index(Path("/content/drive/MyDrive/vifinqa/dense-qwen3-4b"))
print(loaded.manifest.document_count, loaded.manifest.dimension,
      loaded.manifest.encoder.name, loaded.manifest.dataset_fingerprint)
assert loaded.manifest.dimension == 2560
assert loaded.manifest.dataset_fingerprint == release.dataset_fingerprint
```

- [ ] **Step 2: Chạy notebook trên Colab T4 tới khi Cell 6 in ra đúng `dataset_fingerprint`**

Expected: `document_count` bằng `table_count` trong `data/processed/release_v2_*/manifest.json`; `dimension = 2560`.

- [ ] **Step 3: Tải index về local**

Đặt vào `data/indexes/dense-qwen3-4b/<dataset_fingerprint>/`. Kiểm tra local:

```bash
uv run python -c "from pathlib import Path; from financial_report_qa.retrieval.dense_index import load_dense_index; i = load_dense_index(Path('data/indexes/dense-qwen3-4b/422df141c935d46bfd14302abec50f32380e6e4c012159f8ad0ae5560c8a446a')); print(i.manifest.document_count, i.manifest.dimension)"
```

Expected: in ra số document và `2560`.

- [ ] **Step 4: Commit notebook (KHÔNG commit index)**

```bash
git add notebooks/colab_index_qwen3_emb_4b.ipynb
git commit -m "feat(notebooks): shard-checkpointed Qwen3-Embedding-4B corpus indexing on Colab"
```

`data/indexes/` không được commit (xem `data/README.md`: không commit index).

---

### Task 8: Nối cả pipeline vào submission và đo lại

**Files:**
- Modify: `src/financial_report_qa/submission/cli.py` (thêm cờ + lắp service)
- Modify: `src/financial_report_qa/submission/exporter.py:194`, `:410`, `:462` (đổi kiểu tham số `service`)
- Test: `tests/unit/submission/test_exporter_retriever_protocol.py` (tạo)

**Interfaces:**
- Consumes: `TableRetriever` (Task 4), `FusionService` (`fusion.py:65`), `DenseRetrievalService`, `Qwen3CrossEncoderReranker` + `approved_reranker_spec` (Task 3), `FusionWeights` (`fusion_contracts.py:22`).
- Produces: `submission export` chạy được ở ba chế độ: BM25-only (mặc định, giữ nguyên hành vi cũ), `--dense-index` (fusion), `--dense-index --rerank` (fusion + rerank).

- [ ] **Step 1: Write the failing test**

Tạo `tests/unit/submission/test_exporter_retriever_protocol.py`:

```python
from financial_report_qa.retrieval.live_query import TableRetriever
from financial_report_qa.retrieval.fusion import FusionService
from financial_report_qa.retrieval.service import RetrievalService


def test_both_retrievers_satisfy_the_protocol_the_exporter_depends_on() -> None:
    # Nếu một trong hai lệch chữ ký, mypy bắt được ở đây trước khi chạy
    # export 3 tiếng mới phát hiện.
    assert issubclass(RetrievalService, TableRetriever)  # type: ignore[misc]
    assert issubclass(FusionService, TableRetriever)  # type: ignore[misc]
```

Để `issubclass` chạy được với Protocol, thêm `@runtime_checkable` lên `TableRetriever` trong `live_query.py`:

```python
from typing import Protocol, runtime_checkable


@runtime_checkable
class TableRetriever(Protocol):
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/submission/test_exporter_retriever_protocol.py -v`
Expected: FAIL với `TypeError: Instance and class checks can only be used with @runtime_checkable protocols`

- [ ] **Step 3: Thêm `@runtime_checkable` và chạy lại**

Sửa `live_query.py` như Step 1 mô tả.

Run: `uv run pytest tests/unit/submission/test_exporter_retriever_protocol.py -v`
Expected: PASS

- [ ] **Step 4: Đổi kiểu tham số trong `exporter.py`**

Trong `src/financial_report_qa/submission/exporter.py`, đổi import (dòng 37 lân cận):

```python
from financial_report_qa.retrieval.live_query import TableRetriever, retrieve_candidate_table_ids
```

rồi đổi annotation `service: RetrievalService` thành `service: TableRetriever` ở cả ba chỗ: `_run_one_question` (dòng 194), và hai hàm ở dòng 410 và 462. Thêm tham số truyền xuống ở cả ba:

```python
    reranker: Reranker | None = None,
```

và tại mỗi lời gọi `retrieve_candidate_table_ids(...)` (dòng 205 và 425) thêm `reranker=reranker`.

Thêm import: `from financial_report_qa.retrieval.reranker import Reranker`.

- [ ] **Step 5: Verify không phá đường BM25 hiện tại**

Run: `uv run pytest tests/unit/submission/ tests/integration/ -v`
Expected: PASS toàn bộ

- [ ] **Step 6: Thêm cờ CLI**

Trong `_parser()` của `src/financial_report_qa/submission/cli.py`, thêm vào parser `export`:

```python
    export.add_argument(
        "--dense-index",
        type=Path,
        default=None,
        help="Bật fusion BM25+dense. Không truyền thì chạy BM25-only như cũ.",
    )
    export.add_argument(
        "--dense-weight", type=float, default=1.0,
        help="Trọng số nhánh dense trong RRF (bm25 luôn = 1.0).",
    )
    export.add_argument(
        "--rerank", action="store_true",
        help="Xếp lại top-50 của RRF bằng Qwen3-Reranker-4B. Cần --dense-index.",
    )
```

Trong nhánh `if args.command == "export":`, trước khi gọi exporter, lắp service:

```python
            retriever: TableRetriever = RetrievalService(index)
            reranker = None
            if args.dense_index is not None:
                dense_index = load_dense_index(args.dense_index)
                if dense_index.manifest.dataset_fingerprint != release.dataset_fingerprint:
                    raise SubmissionError(
                        "--dense-index dataset_fingerprint does not match --release-lock"
                    )
                encoder = SentenceTransformerDenseEncoder(dense_index.manifest.encoder)
                cache = QueryEmbeddingCache(
                    Path("data/indexes/dense-query-cache/qwen3-4b"),
                    encoder_spec_sha256=dense_index.manifest.encoder_spec_sha256,
                )
                retriever = FusionService(
                    RetrievalService(index),
                    DenseRetrievalService(dense_index, encoder, cache),
                    FusionWeights(bm25=1.0, dense=args.dense_weight),
                )
            if args.rerank:
                if args.dense_index is None:
                    raise SubmissionError("--rerank cần --dense-index")
                # Tuần tự, không song song: encoder đã embed xong corpus
                # offline nên chỉ reranker chiếm VRAM lúc chạy.
                reranker = Qwen3CrossEncoderReranker(
                    approved_reranker_spec("qwen3-reranker-4b")
                )
```

rồi truyền `retriever` (thay cho `service`) và `reranker=reranker` xuống hàm export.

Thêm các import tương ứng ở đầu `submission/cli.py`.

- [ ] **Step 7: Đo lại sweep với stack đầy đủ**

```bash
uv run financial-report-qa retrieval sweep-k --release-lock data/qa/week1_pilot_422df141c935/dataset-pilot-v1.json --bm25-index data/indexes/bm25-v4/422df141c935d46bfd14302abec50f32380e6e4c012159f8ad0ae5560c8a446a --gold data/qa/retrieval-gold-v1.jsonl --output-stem artifacts/evaluations/sweep-fusion-rerank
```

**Lưu ý:** subcommand `sweep-k` ở Task 6 mới chỉ dựng `RetrievalService`. Để đo được stack đầy đủ, thêm vào `sweep-k` đúng ba cờ `--dense-index`, `--dense-weight`, `--rerank` và cùng khối lắp service như Step 6 (trích ra hàm dùng chung `_build_table_retriever(args, release, index)` trong `retrieval/cli.py` để không lặp code giữa hai CLI).

So sánh với `artifacts/evaluations/sweep-bm25-baseline.json` của Task 6.

**Tiêu chí thành công (§12.7 của spec):** Recall@10 ≥ 75% trên gold (BM25 hiện tại 47.41%). Nếu F2 tại `k*` không cao hơn baseline, **dừng lại phân tích** — không nộp một stack đắt hơn mà không tốt hơn.

- [ ] **Step 8: Commit**

```bash
git add src/financial_report_qa/submission/cli.py src/financial_report_qa/submission/exporter.py src/financial_report_qa/retrieval/cli.py src/financial_report_qa/retrieval/live_query.py tests/unit/submission/test_exporter_retriever_protocol.py artifacts/evaluations/sweep-fusion-rerank.json artifacts/evaluations/sweep-fusion-rerank.md
git commit -m "feat(submission): wire BM25+dense fusion and cross-encoder rerank into the live path"
```

---

## Ngoài phạm vi

Theo §11 của spec chi phối:

- Reranker cho **row** retrieval (`row_fusion.py`) — vòng này chỉ rerank ở tầng bảng.
- Fine-tune (QLoRA) bất kỳ model nào.
- Re-ingest corpus (đổi `dataset_fingerprint` → mất mọi baseline đã pin).
- Serving engine (vLLM/SGLang) cho chạy local.
- Nhánh 2 (answering): `plan_source`, LLM batch, compiler — plan riêng.
