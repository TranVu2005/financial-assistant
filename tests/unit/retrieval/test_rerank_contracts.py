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
        spec.batch_size = 8


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
