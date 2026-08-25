"""Unit tests for `retrieval.cli._build_table_retriever` (Task 8): the shared
wiring that assembles the live table retriever for BOTH `submission export`
and `retrieval sweep-k`.

Everything runs offline: the dense corpus/index fixtures are built with a
mock 2-dimensional encoder and the reranker is a fake -- exactly like
`test_row_dense_wiring.py` / `test_live_query.py` -- so no test ever loads a
real sentence-transformers or cross-encoder model."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

import financial_report_qa.retrieval.cli as retrieval_cli
from financial_report_qa.core.errors import DenseArtifactError, DenseInputError
from financial_report_qa.evaluation.week1_release import ReleaseLock
from financial_report_qa.retrieval.contracts import (
    MetricLabelObservation,
    RetrievalFilters,
    TableDocument,
    TableMetadata,
)
from financial_report_qa.retrieval.dense_contracts import DenseEncoderSpec
from financial_report_qa.retrieval.dense_corpus import build_dense_corpus, save_dense_corpus
from financial_report_qa.retrieval.dense_encoder import approved_encoder_spec
from financial_report_qa.retrieval.dense_index import (
    build_dense_index,
    save_dense_index,
)
from financial_report_qa.retrieval.fusion import FusionService
from financial_report_qa.retrieval.index import BM25Index, build_bm25_index
from financial_report_qa.retrieval.live_query import retrieve_candidate_table_ids
from financial_report_qa.retrieval.release import ResolvedRetrievalRelease
from financial_report_qa.retrieval.reranker import approved_reranker_spec
from financial_report_qa.retrieval.service import RetrievalService

_FINGERPRINT = "37a61be7aebde1fbcfe3aca42e6ba4ff37ae87bdd1a9ba6696506bcd188e7d1f"
_LOCK_SHA = "2" * 64
_TABLE_ID = "tbl_" + "a" * 64
_DOC_ID = "doc_" + "a" * 64


class _MockEncoder:
    """DenseEncoder stand-in with a tiny deterministic 2-d space."""

    def __init__(self, spec: DenseEncoderSpec) -> None:
        self.spec = spec

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        return np.asarray([1.0, 0.0], dtype=np.float32)


class _FakeReranker:
    """Reranker stand-in: never loads a model, scores everything equal."""

    def __init__(self) -> None:
        self.spec = approved_reranker_spec("qwen3-reranker-4b")
        self.calls: list[tuple[str, int]] = []

    def score(self, query: str, documents: Sequence[str]) -> np.ndarray:
        self.calls.append((query, len(documents)))
        return np.zeros(len(documents), dtype=np.float32)


class _RecordingQwen3Reranker:
    """`Qwen3CrossEncoderReranker` stand-in that only records init kwargs.

    The real class imports torch/transformers inside ``__init__``; unit tests
    must never construct it, so the CLI wiring is observed through this fake
    monkeypatched over the module attribute the factory lambda resolves."""

    init_calls: ClassVar[list[dict[str, object]]] = []

    def __init__(
        self,
        spec: object,
        *,
        local_files_only: bool = False,
        model_dtype: str | None = None,
        device: str | None = None,
    ) -> None:
        self.spec = spec
        type(self).init_calls.append(
            {
                "local_files_only": local_files_only,
                "model_dtype": model_dtype,
                "device": device,
            }
        )


class _RecordingDenseEncoder:
    """`SentenceTransformerDenseEncoder` stand-in that only records init kwargs.

    The real class downloads weights inside ``__init__``; monkeypatching it
    over the module attribute exposes exactly which placement (`device`) and
    compute dtype kwargs production threads in for a given manifest spec."""

    init_calls: ClassVar[list[dict[str, object]]] = []

    def __init__(
        self,
        spec: object,
        *,
        local_files_only: bool = False,
        model_dtype: str | None = None,
        encode_batch_size: int | None = None,
        device: str | None = None,
    ) -> None:
        self.spec = spec
        type(self).init_calls.append(
            {
                "local_files_only": local_files_only,
                "model_dtype": model_dtype,
                "encode_batch_size": encode_batch_size,
                "device": device,
            }
        )


class _EagerCachedReranker:
    """`CachedReranker` stand-in that builds the factory eagerly.

    The real class defers model construction until a cache miss, which would
    hide the factory's kwargs from a test; building immediately exposes them.
    Never touches disk."""

    def __init__(
        self,
        cache_dir: Path,
        spec: object,
        *,
        factory: Callable[[], object],
    ) -> None:
        self.cache_dir = cache_dir
        self.spec = spec
        self.inner = factory()

    def score(self, query: str, documents: Sequence[str]) -> np.ndarray:
        return np.zeros(len(documents), dtype=np.float32)


def _mock_e5_encoder() -> _MockEncoder:
    return _MockEncoder(
        approved_encoder_spec("multilingual-e5-small").model_copy(update={"dimension": 2})
    )


def _document() -> TableDocument:
    return TableDocument(
        table_id=_TABLE_ID,
        doc_id=_DOC_ID,
        text="company_code: DBC\nperiod: 2023\nTổng tài sản | 2023 | 500",
        metadata=TableMetadata(
            table_id=_TABLE_ID,
            doc_id=_DOC_ID,
            company_code="DBC",
            periods=("2023",),
            statement_type="balance_sheet",
            source_path="a.txt",
            line_start=1,
            line_end=2,
        ),
        metric_labels=(MetricLabelObservation(canonical="total_assets", raw=None),),
    )


def _release(dataset_fingerprint: str = _FINGERPRINT) -> ResolvedRetrievalRelease:
    lock = ReleaseLock(
        alias="dataset-pilot-v1",
        sampling_version="week1-pilot-v1",
        dataset_fingerprint=dataset_fingerprint,
        source_manifest_sha256="0" * 64,
        release_path="fixture/release",
        gate_result_path="fixture/gate.json",
        evaluation_inputs_sha256="1" * 64,
    )
    return ResolvedRetrievalRelease(
        lock=lock,
        dataset_fingerprint=dataset_fingerprint,
        release_dir=Path("unused"),
        gate_result_path=Path("unused/gate.json"),
        lock_path=Path("unused/lock.json"),
        manifest={},
        lock_sha256=_LOCK_SHA,
    )


def _args(**overrides: object) -> argparse.Namespace:
    defaults: dict[str, object] = {
        "dense_index": None,
        "table_dense_weight": 1.0,
        "rerank": False,
        # Compute/placement knobs (device-placement feature): parser defaults
        # for every subcommand that owns --dense-index.
        "table_encoder_device": "cpu",
        "table_encoder_model_dtype": None,
        "rerank_device": "cpu",
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _bm25_index() -> BM25Index:
    return build_bm25_index(
        (_document(),), dataset_fingerprint=_FINGERPRINT, release_lock_sha256=_LOCK_SHA
    )


def _write_dense_artifacts(
    tmp_path: Path, *, dataset_fingerprint: str = _FINGERPRINT, lock_sha256: str = _LOCK_SHA
) -> tuple[Path, Path]:
    """Build + save a real (tiny) table dense corpus and index; returns
    (corpus_dir, index_dir) laid out per repo convention: `<fp>/corpus` next
    to nothing else -- the locator must find `<index>/../corpus`."""
    corpus = build_dense_corpus(
        (_document(),), dataset_fingerprint=dataset_fingerprint, release_lock_sha256=lock_sha256
    )
    corpus_dir = tmp_path / f"{dataset_fingerprint[:12]}" / "corpus"
    save_dense_corpus(corpus, corpus_dir)
    encoder = _mock_e5_encoder()
    index = build_dense_index(corpus, encoder)
    index_dir = tmp_path / f"{dataset_fingerprint[:12]}" / "dense-e5-mock"
    save_dense_index(index, index_dir)
    return corpus_dir, index_dir


@pytest.fixture()
def _redirect_table_query_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The helper hardcodes the repo-relative query cache default; tests must
    # not write .npy entries into the real data/ tree.
    monkeypatch.setattr(retrieval_cli, "_TABLE_DENSE_QUERY_CACHE_DEFAULT", tmp_path / "qcache")


def test_bm25_only_by_default_keeps_the_old_behavior() -> None:
    retriever, reranker = retrieval_cli._build_table_retriever(_args(), _release(), _bm25_index())

    assert isinstance(retriever, RetrievalService)
    assert reranker is None


def test_rerank_without_dense_index_is_rejected_before_any_model_loads() -> None:
    with pytest.raises(DenseInputError, match="--rerank cần --dense-index"):
        retrieval_cli._build_table_retriever(_args(rerank=True), _release(), _bm25_index())


def test_missing_corpus_lists_every_searched_location(tmp_path: Path) -> None:
    index_dir = tmp_path / "lonely-index"
    index_dir.mkdir()

    with pytest.raises(DenseArtifactError) as excinfo:
        retrieval_cli._build_table_retriever(
            _args(dense_index=index_dir), _release(), _bm25_index()
        )

    message = str(excinfo.value)
    assert str(index_dir / "corpus") in message
    assert str(index_dir.parent / "corpus") in message


def test_dense_index_fingerprint_mismatch_is_rejected(tmp_path: Path) -> None:
    _, index_dir = _write_dense_artifacts(tmp_path)
    mismatched_release = _release(dataset_fingerprint="f" * 64)

    with pytest.raises(DenseArtifactError, match="does not match --release-lock"):
        retrieval_cli._build_table_retriever(
            _args(dense_index=index_dir), mismatched_release, _bm25_index()
        )


def test_dense_index_encoder_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    """The manifest pins one encoder spec; loading with an injected encoder of
    a DIFFERENT spec must fail the identity check inside `load_dense_index`,
    never silently rank against foreign vectors."""
    corpus_dir, index_dir = _write_dense_artifacts(tmp_path)
    del corpus_dir  # located via convention
    foreign_spec = approved_encoder_spec("bge-m3").model_copy(update={"dimension": 2})

    with pytest.raises(DenseArtifactError, match="does not match"):
        retrieval_cli._build_table_retriever(
            _args(dense_index=index_dir),
            _release(),
            _bm25_index(),
            encoder=_MockEncoder(foreign_spec),
        )


def test_happy_path_wires_fusion_and_runs_the_full_live_path(
    tmp_path: Path, _redirect_table_query_cache: None
) -> None:
    _, index_dir = _write_dense_artifacts(tmp_path)
    encoder = _mock_e5_encoder()

    retriever, reranker = retrieval_cli._build_table_retriever(
        _args(dense_index=index_dir),
        _release(),
        _bm25_index(),
        encoder=encoder,
    )

    assert isinstance(retriever, FusionService)
    # Không có --rerank thì KHÔNG có reranker nào được trả ra -- dù cờ rerank
    # vắng mặt, ý định rank lại phải được nói rõ chứ không mặc định bật.
    assert reranker is None

    trace = retriever.retrieve(
        "Tổng tài sản của DBC năm 2023 là bao nhiêu?",
        filters=RetrievalFilters(),
        k=5,
    )
    assert tuple(candidate.table_id for candidate in trace.results) == (_TABLE_ID,)
    assert all(candidate.fused_score > 0 for candidate in trace.results)


def test_full_live_path_runs_fusion_then_rerank_offline(
    tmp_path: Path, _redirect_table_query_cache: None
) -> None:
    _, index_dir = _write_dense_artifacts(tmp_path)
    encoder = _mock_e5_encoder()
    fake_reranker = _FakeReranker()

    retriever, reranker = retrieval_cli._build_table_retriever(
        _args(dense_index=index_dir, rerank=True),
        _release(),
        _bm25_index(),
        encoder=encoder,
        reranker=fake_reranker,
    )

    retrieved = retrieve_candidate_table_ids(
        "Tổng tài sản của DBC năm 2023 là bao nhiêu?", retriever, k=1, reranker=reranker
    )
    assert retrieved == (_TABLE_ID,)
    # The reranker really scored the fused window, not BM25 output directly.
    assert fake_reranker.calls and fake_reranker.calls[0][1] >= 1


def test_sweep_k_parser_accepts_the_three_stack_flags() -> None:
    args = retrieval_cli._parser().parse_args(
        [
            "sweep-k",
            "--release-lock",
            "lock.json",
            "--bm25-index",
            "idx",
            "--gold",
            "gold.jsonl",
            "--output-stem",
            "out/sweep",
            "--dense-index",
            "dense-dir",
            "--table-dense-weight",
            "0.75",
            "--rerank",
        ]
    )
    assert args.dense_index == Path("dense-dir")
    assert args.table_dense_weight == pytest.approx(0.75)
    assert args.rerank is True

    defaults = retrieval_cli._parser().parse_args(
        [
            "sweep-k",
            "--release-lock",
            "lock.json",
            "--bm25-index",
            "idx",
            "--gold",
            "gold.jsonl",
            "--output-stem",
            "out/sweep",
        ]
    )
    assert defaults.dense_index is None
    assert defaults.table_dense_weight == pytest.approx(1.0)
    assert defaults.rerank is False
    assert defaults.rerank_dtype == "float32"


def test_rerank_dtype_bfloat16_reaches_the_pinned_reranker_factory(
    tmp_path: Path,
    _redirect_table_query_cache: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`--rerank-dtype bfloat16` phải tới đúng kwargs của reranker pinned qua
    factory trong `CachedReranker` -- compute-only knob, spec không đổi."""
    _, index_dir = _write_dense_artifacts(tmp_path)
    monkeypatch.setattr(retrieval_cli, "Qwen3CrossEncoderReranker", _RecordingQwen3Reranker)
    monkeypatch.setattr(retrieval_cli, "CachedReranker", _EagerCachedReranker)
    _RecordingQwen3Reranker.init_calls.clear()

    retriever, reranker = retrieval_cli._build_table_retriever(
        _args(dense_index=index_dir, rerank=True, rerank_dtype="bfloat16"),
        _release(),
        _bm25_index(),
        encoder=_mock_e5_encoder(),
    )

    assert isinstance(retriever, FusionService)
    assert isinstance(reranker, _EagerCachedReranker)
    assert _RecordingQwen3Reranker.init_calls == [
        {"local_files_only": False, "model_dtype": "bfloat16", "device": "cpu"}
    ]


@pytest.mark.parametrize("overrides", [{}, {"rerank_dtype": "float32"}])
def test_rerank_dtype_default_keeps_the_fp32_spec_contract(
    tmp_path: Path,
    _redirect_table_query_cache: None,
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, object],
) -> None:
    """Mặc định (cờ vắng mặt trên Namespace lẫn giá trị parser "float32") map
    về ``model_dtype=None`` -- class thật hiểu None là torch.float32, đúng hợp
    đồng fp32 của `RerankerSpec`; knob chỉ tính toán, không đụng spec."""
    _, index_dir = _write_dense_artifacts(tmp_path)
    monkeypatch.setattr(retrieval_cli, "Qwen3CrossEncoderReranker", _RecordingQwen3Reranker)
    monkeypatch.setattr(retrieval_cli, "CachedReranker", _EagerCachedReranker)
    _RecordingQwen3Reranker.init_calls.clear()

    retrieval_cli._build_table_retriever(
        _args(dense_index=index_dir, rerank=True, **overrides),
        _release(),
        _bm25_index(),
        encoder=_mock_e5_encoder(),
    )

    assert len(_RecordingQwen3Reranker.init_calls) == 1
    assert _RecordingQwen3Reranker.init_calls[0]["model_dtype"] is None


def test_all_three_rerank_subcommands_expose_the_compute_dtype_flag() -> None:
    """Cả ba subcommand có `--rerank` (`retrieval sweep-k`, `submission export`,
    `submission row-batches`) đều nhận `--rerank-dtype`, mặc định float32."""
    sweep = retrieval_cli._parser().parse_args(
        [
            "sweep-k",
            "--release-lock",
            "lock.json",
            "--bm25-index",
            "idx",
            "--gold",
            "gold.jsonl",
            "--output-stem",
            "out/sweep",
            "--rerank-dtype",
            "bfloat16",
        ]
    )
    assert sweep.rerank_dtype == "bfloat16"

    from financial_report_qa.submission.cli import _parser as submission_parser

    export_argv = [
        "export",
        "--release-lock",
        "lock.json",
        "--bm25-index",
        "idx",
        "--questions-path",
        "q.jsonl",
        "--execution-config",
        "cfg.yaml",
        "--output-zip",
        "out.zip",
        "--report-dir",
        "reports",
        "--program-decisions",
        "decisions.jsonl",
        "--rerank-dtype",
        "bfloat16",
    ]
    row_batches_argv = [
        "row-batches",
        "--release-lock",
        "lock.json",
        "--bm25-index",
        "idx",
        "--questions-path",
        "q.jsonl",
        "--output-dir",
        "batches",
        "--release-dir",
        "release",
        "--rerank-dtype",
        "bfloat16",
    ]
    for argv in (export_argv, row_batches_argv):
        parsed = submission_parser().parse_args(argv)
        assert parsed.rerank_dtype == "bfloat16"

        defaults = submission_parser().parse_args(argv[: argv.index("--rerank-dtype")])
        assert defaults.rerank_dtype == "float32"


def test_device_flags_thread_through_to_both_pinned_models(
    tmp_path: Path,
    _redirect_table_query_cache: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`--table-encoder-device` / `--table-encoder-model-dtype` /
    `--rerank-device` phải tới đúng kwargs khởi tạo của cả hai model pinned
    qua `_build_table_retriever` -- placement knob, spec không đổi."""
    _, index_dir = _write_dense_artifacts(tmp_path)
    monkeypatch.setattr(
        retrieval_cli, "SentenceTransformerDenseEncoder", _RecordingDenseEncoder
    )
    monkeypatch.setattr(retrieval_cli, "Qwen3CrossEncoderReranker", _RecordingQwen3Reranker)
    monkeypatch.setattr(retrieval_cli, "CachedReranker", _EagerCachedReranker)
    _RecordingDenseEncoder.init_calls.clear()
    _RecordingQwen3Reranker.init_calls.clear()

    retriever, reranker = retrieval_cli._build_table_retriever(
        _args(
            dense_index=index_dir,
            rerank=True,
            table_encoder_device="cuda:1",
            table_encoder_model_dtype="float16",
            rerank_device="cuda:0",
        ),
        _release(),
        _bm25_index(),
    )

    assert isinstance(retriever, FusionService)
    assert isinstance(reranker, _EagerCachedReranker)
    # Encoder: placement cuda:1 with the explicitly requested fp16 compute.
    assert _RecordingDenseEncoder.init_calls == [
        {
            "local_files_only": False,
            "model_dtype": "float16",
            "encode_batch_size": None,
            "device": "cuda:1",
        }
    ]
    # Reranker: factory lambda captures --rerank-device for the lazy load.
    assert _RecordingQwen3Reranker.init_calls == [
        {"local_files_only": False, "model_dtype": None, "device": "cuda:0"}
    ]


def test_cuda_encoder_device_infers_float16_unless_dtype_is_explicit(
    tmp_path: Path,
    _redirect_table_query_cache: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """fp32 4B cannot fit one T4: bỏ trống --table-encoder-model-dtype khi
    device là cuda* thì suy ra float16; ghi đè tường minh thì thắng."""
    _, index_dir = _write_dense_artifacts(tmp_path)
    monkeypatch.setattr(
        retrieval_cli, "SentenceTransformerDenseEncoder", _RecordingDenseEncoder
    )

    def build(**overrides: object) -> list[dict[str, object]]:
        del _RecordingDenseEncoder.init_calls[:]
        retrieval_cli._build_table_retriever(
            _args(dense_index=index_dir, **overrides), _release(), _bm25_index()
        )
        return list(_RecordingDenseEncoder.init_calls)

    inferred = build(table_encoder_device="cuda:1")
    assert inferred[0]["model_dtype"] == "float16"
    assert inferred[0]["device"] == "cuda:1"

    explicit = build(table_encoder_device="cuda:0", table_encoder_model_dtype="float32")
    assert explicit[0]["model_dtype"] == "float32"
    assert explicit[0]["device"] == "cuda:0"

    bare = build()
    assert bare[0]["model_dtype"] == "float32"
    assert bare[0]["device"] == "cpu"


@pytest.mark.parametrize("overrides", [{}, {"rerank": True}])
def test_placement_defaults_keep_todays_behavior_exactly(
    tmp_path: Path,
    _redirect_table_query_cache: None,
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, object],
) -> None:
    """Mặc định (cờ vắng mặt): encoder cpu + fp32 và reranker cpu -- đúng
    hành vi trước feature, chỉ là bây giờ truyền tường minh thay vì dựa vào
    default bên trong lớp model."""
    _, index_dir = _write_dense_artifacts(tmp_path)
    monkeypatch.setattr(
        retrieval_cli, "SentenceTransformerDenseEncoder", _RecordingDenseEncoder
    )
    monkeypatch.setattr(retrieval_cli, "Qwen3CrossEncoderReranker", _RecordingQwen3Reranker)
    monkeypatch.setattr(retrieval_cli, "CachedReranker", _EagerCachedReranker)
    _RecordingDenseEncoder.init_calls.clear()
    _RecordingQwen3Reranker.init_calls.clear()

    retriever, reranker = retrieval_cli._build_table_retriever(
        _args(dense_index=index_dir, **overrides),
        _release(),
        _bm25_index(),
    )

    assert isinstance(retriever, FusionService)
    assert _RecordingDenseEncoder.init_calls == [
        {
            "local_files_only": False,
            "model_dtype": "float32",
            "encode_batch_size": None,
            "device": "cpu",
        }
    ]
    if overrides.get("rerank"):
        assert isinstance(reranker, _EagerCachedReranker)
        assert _RecordingQwen3Reranker.init_calls == [
            {"local_files_only": False, "model_dtype": None, "device": "cpu"}
        ]
    else:
        assert reranker is None


def test_all_three_subcommands_expose_the_placement_flags() -> None:
    """Cả ba subcommand có `--dense-index` đều nhận ba cờ placement/dtype,
    mặc định cpu / None / cpu."""
    sweep = retrieval_cli._parser().parse_args(
        [
            "sweep-k",
            "--release-lock",
            "lock.json",
            "--bm25-index",
            "idx",
            "--gold",
            "gold.jsonl",
            "--output-stem",
            "out/sweep",
            "--dense-index",
            "dense-dir",
            "--table-encoder-device",
            "cuda:1",
            "--table-encoder-model-dtype",
            "float16",
            "--rerank-device",
            "cuda:0",
        ]
    )
    assert sweep.table_encoder_device == "cuda:1"
    assert sweep.table_encoder_model_dtype == "float16"
    assert sweep.rerank_device == "cuda:0"

    from financial_report_qa.submission.cli import _parser as submission_parser

    export_argv = [
        "export",
        "--release-lock",
        "lock.json",
        "--bm25-index",
        "idx",
        "--questions-path",
        "q.jsonl",
        "--execution-config",
        "cfg.yaml",
        "--output-zip",
        "out.zip",
        "--report-dir",
        "reports",
        "--program-decisions",
        "decisions.jsonl",
    ]
    row_batches_argv = [
        "row-batches",
        "--release-lock",
        "lock.json",
        "--bm25-index",
        "idx",
        "--questions-path",
        "q.jsonl",
        "--output-dir",
        "batches",
        "--release-dir",
        "release",
    ]
    for argv in (export_argv, row_batches_argv):
        placed = submission_parser().parse_args(
            argv
            + [
                "--table-encoder-device",
                "cuda:1",
                "--table-encoder-model-dtype",
                "float16",
                "--rerank-device",
                "cuda:0",
            ]
        )
        assert placed.table_encoder_device == "cuda:1"
        assert placed.table_encoder_model_dtype == "float16"
        assert placed.rerank_device == "cuda:0"

        defaults = submission_parser().parse_args(argv)
        assert defaults.table_encoder_device == "cpu"
        assert defaults.table_encoder_model_dtype is None
        assert defaults.rerank_device == "cpu"


@pytest.mark.parametrize(
    ("overrides", "flag_name"),
    [
        ({"table_encoder_device": "cudo:1"}, "--table-encoder-device"),
        ({"rerank": True, "rerank_device": "gpu:0"}, "--rerank-device"),
    ],
)
def test_malformed_device_flags_fail_loudly_before_any_model_load(
    tmp_path: Path,
    _redirect_table_query_cache: None,
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, object],
    flag_name: str,
) -> None:
    """Cờ device gõ sai ("cudo:1") phải fail ngay ở _build_table_retriever với
    DenseInputError gọi rõ tên cờ -- không để trôi xuống torch rồi văng lỗi
    opaque sâu bên trong model load."""
    _, index_dir = _write_dense_artifacts(tmp_path)
    monkeypatch.setattr(
        retrieval_cli, "SentenceTransformerDenseEncoder", _RecordingDenseEncoder
    )
    _RecordingDenseEncoder.init_calls.clear()

    with pytest.raises(DenseInputError, match=flag_name):
        retrieval_cli._build_table_retriever(
            _args(dense_index=index_dir, **overrides), _release(), _bm25_index()
        )

    # Không một model nào được dựng khi cờ sai.
    assert _RecordingDenseEncoder.init_calls == []
