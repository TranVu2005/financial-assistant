"""Command-line interface for reproducible Day 8 BM25 retrieval."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from json import JSONDecodeError
from pathlib import Path
from typing import Literal, cast

import faiss
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from financial_report_qa.core.errors import (
    DenseArtifactError,
    DenseInputError,
    DenseModelError,
    ExpansionArtifactError,
    ExpansionInputError,
    FusionArtifactError,
    FusionInputError,
    GraphArtifactError,
    GraphInputError,
    RerankError,
    RetrievalArtifactError,
    RetrievalInputError,
)
from financial_report_qa.retrieval.contracts import GoldRetrievalQuestion
from financial_report_qa.retrieval.data_cleanup import plan_day9_cleanup, quarantine_day9_cleanup
from financial_report_qa.retrieval.dense_artifacts import write_text_atomic
from financial_report_qa.retrieval.dense_cache import QueryEmbeddingCache
from financial_report_qa.retrieval.dense_contracts import DenseIndexManifest, EncoderName
from financial_report_qa.retrieval.dense_corpus import (
    DenseCorpus,
    build_dense_corpus,
    load_dense_corpus,
    save_dense_corpus,
)
from financial_report_qa.retrieval.dense_encoder import (
    DenseEncoder,
    SentenceTransformerDenseEncoder,
    approved_encoder_spec,
    encoder_spec_sha256,
)
from financial_report_qa.retrieval.dense_evaluation import (
    DenseEvaluationRun,
    build_day9_comparison,
    evaluate_cold_and_warm,
    write_day9_comparison,
)
from financial_report_qa.retrieval.dense_index import (
    build_dense_index,
    load_dense_index,
    save_dense_index,
)
from financial_report_qa.retrieval.dense_service import DenseRetrievalService
from financial_report_qa.retrieval.documents import build_table_documents
from financial_report_qa.retrieval.evaluation import (
    evaluate_retrieval,
    evaluate_retrieval_v2,
    write_report,
    write_report_v2,
)
from financial_report_qa.retrieval.fusion import FusionService
from financial_report_qa.retrieval.fusion_contracts import FusionWeights
from financial_report_qa.retrieval.fusion_evaluation import evaluate_fusion_grid, write_day10_fusion
from financial_report_qa.retrieval.gold import (
    REQUIRED_GOLD_QUESTION_COUNT,
    load_gold_questions,
)
from financial_report_qa.retrieval.index import (
    BM25Index,
    build_bm25_index,
    load_bm25_index,
    save_bm25_index,
)
from financial_report_qa.retrieval.live_query import TableRetriever
from financial_report_qa.retrieval.reference import (
    ReferenceVersion,
    load_bm25_reference_report,
    resolve_gold_reference,
)
from financial_report_qa.retrieval.release import (
    ResolvedRetrievalRelease,
    resolve_retrieval_release,
)
from financial_report_qa.retrieval.rerank_cache import CachedReranker
from financial_report_qa.retrieval.reranker import (
    Qwen3CrossEncoderReranker,
    Reranker,
    approved_reranker_spec,
)
from financial_report_qa.retrieval.row_dense_corpus import (
    RowDenseCorpus,
    build_row_dense_corpus,
    load_row_dense_corpus,
    save_row_dense_corpus,
)
from financial_report_qa.retrieval.row_dense_index import (
    build_row_dense_index,
    save_row_dense_index,
)
from financial_report_qa.retrieval.row_documents import build_row_documents
from financial_report_qa.retrieval.row_index import build_row_bm25_index, save_row_bm25_index
from financial_report_qa.retrieval.service import RetrievalService
from financial_report_qa.retrieval.sweep import (
    DEFAULT_KS,
    SweepResult,
    recommend_k,
    render_sweep_markdown,
    run_sweep,
)


class _DenseBuildObservation(BaseModel):
    """Operational evidence emitted by an explicit dense index build."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    encoder_name: EncoderName
    encoder_spec_sha256: str
    dataset_fingerprint: str
    build_seconds: float = Field(ge=0)
    index_byte_size: int = Field(ge=0)
    faiss_device: Literal["cpu", "cuda"]
    faiss_gpu_count: int = Field(ge=0)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="financial-report-qa retrieval")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in (
        "build-index",
        "validate-gold",
        "evaluate",
        "evaluate-v2",
        "build-dense-corpus",
    ):
        command = commands.add_parser(name)
        command.add_argument("--release-lock", type=Path, required=True)
        if name == "build-index":
            command.add_argument("--output-root", type=Path, required=True)
        elif name == "build-dense-corpus":
            command.add_argument("--output-root", type=Path, required=True)
        else:
            command.add_argument("--gold-path", type=Path, required=True)
            command.add_argument("--gold-version", choices=("gold30", "gold70"))
        if name in {"evaluate", "evaluate-v2"}:
            command.add_argument("--index-dir", type=Path, required=True)
            command.add_argument("--output-dir", type=Path, required=True)
        if name == "evaluate-v2":
            command.add_argument("--diagnostic-k", type=int, default=100)
            command.add_argument("--repo-root", type=Path)
    dense_index = commands.add_parser("build-dense-index")
    dense_index.add_argument("--release-lock", type=Path, required=True)
    dense_index.add_argument("--corpus-dir", type=Path, required=True)
    dense_index.add_argument(
        "--encoder", choices=("bge-m3", "multilingual-e5-small"), required=True
    )
    dense_index.add_argument("--output-root", type=Path, required=True)
    dense_index.add_argument("--observation-path", type=Path, required=True)
    dense_index.add_argument("--faiss-device", choices=("cpu", "cuda"), default="cpu")
    dense_index.add_argument("--encoder-device", choices=("cpu", "cuda"), default="cpu")
    dense_index.add_argument("--local-files-only", action="store_true")
    dense_evaluation = commands.add_parser("evaluate-dense")
    dense_evaluation.add_argument("--release-lock", type=Path, required=True)
    dense_evaluation.add_argument("--corpus-dir", type=Path, required=True)
    dense_evaluation.add_argument("--index-dir", type=Path, required=True)
    dense_evaluation.add_argument(
        "--encoder", choices=("bge-m3", "multilingual-e5-small"), required=True
    )
    dense_evaluation.add_argument("--encoder-device", choices=("cpu", "cuda"), default="cpu")
    dense_evaluation.add_argument("--gold-path", type=Path, required=True)
    dense_evaluation.add_argument("--gold-version", choices=("gold30", "gold70"))
    dense_evaluation.add_argument("--cache-dir", type=Path, required=True)
    dense_evaluation.add_argument("--observation-path", type=Path, required=True)
    dense_evaluation.add_argument("--output-path", type=Path, required=True)
    compare = commands.add_parser("compare-day9")
    compare.add_argument("--release-lock", type=Path, required=True)
    compare.add_argument("--bm25-report", type=Path, required=True)
    compare.add_argument("--bge-report", type=Path, required=True)
    compare.add_argument("--e5-report", type=Path, required=True)
    compare.add_argument("--output-dir", type=Path, required=True)
    fusion = commands.add_parser("evaluate-fusion")
    fusion.add_argument("--release-lock", type=Path, required=True)
    fusion.add_argument("--index-dir", type=Path, required=True)
    fusion.add_argument("--corpus-dir", type=Path, required=True)
    fusion.add_argument("--dense-index-dir", type=Path, required=True)
    fusion.add_argument("--encoder", choices=("bge-m3", "multilingual-e5-small"), required=True)
    fusion.add_argument("--encoder-device", choices=("cpu", "cuda"), default="cpu")
    fusion.add_argument("--gold-path", type=Path, required=True)
    fusion.add_argument("--gold-version", choices=("gold30", "gold70"))
    fusion.add_argument("--cache-dir", type=Path, required=True)
    fusion.add_argument("--bm25-report", type=Path, required=True)
    fusion.add_argument("--output-dir", type=Path, required=True)
    cleanup = commands.add_parser("cleanup-day9-data")
    cleanup.add_argument("--repo-root", type=Path, required=True)
    cleanup.add_argument("--quarantine-root", type=Path, required=True)
    cleanup.add_argument("--apply", action="store_true")
    sweep = commands.add_parser("sweep-k")
    sweep.add_argument("--release-lock", type=Path, required=True)
    sweep.add_argument("--bm25-index", type=Path, required=True)
    sweep.add_argument("--gold", type=Path, required=True)
    sweep.add_argument("--output-stem", type=Path, required=True)
    sweep.add_argument(
        "--gold-count",
        type=int,
        default=REQUIRED_GOLD_QUESTION_COUNT,
        help="Số câu gold bắt buộc phải đọc được. Mặc định là kích thước gold "
        "v1 (120); tập gold gán nhãn tay mới có kích thước khác thì truyền vào "
        "-- đếm sai là dấu hiệu file gold bị cắt, nên đây là kiểm tra chứ "
        "không phải tuỳ chọn tiện lợi.",
    )
    sweep.add_argument(
        "--ks",
        type=int,
        nargs="+",
        default=list(DEFAULT_KS),
        help="Các giá trị k cần đo (mặc định 1 2 3 5 8 10 15).",
    )
    sweep.add_argument(
        "--dense-index",
        type=Path,
        default=None,
        help="Bật fusion BM25+dense cho tầng bảng: thư mục dense index "
        "(manifest.json + index.faiss). Corpus đi kèm được tìm ở "
        "<dense-index>/corpus hoặc <thư mục cha>/corpus. Không truyền thì "
        "đo BM25-only như cũ.",
    )
    sweep.add_argument(
        "--table-dense-weight",
        type=float,
        default=1.0,
        help="Trọng số nhánh dense trong RRF của tầng bảng (bm25 luôn = 1.0).",
    )
    sweep.add_argument(
        "--rerank-cache-dir",
        type=Path,
        default=None,
        help="Nơi lưu điểm cross-encoder (mặc định data/indexes/"
        "rerank-score-cache). Dùng CHUNG giữa `row-batches` và `export` thì "
        "reranker chỉ tốn GPU đúng một lần; lần thứ hai chạy không cần model.",
    )
    sweep.add_argument(
        "--rerank",
        action="store_true",
        help="Xếp lại top-50 của RRF bằng Qwen3-Reranker-4B (pinned). Cần --dense-index "
        "và ~32GB RAM (encoder dense ~16GB fp32 vẫn thường trú khi reranker nạp thêm "
        "~16GB); Colab là nơi chạy phù hợp cho phép đo fused+rerank.",
    )
    sweep.add_argument(
        "--rerank-dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
        help="Compute-only: nạp reranker ở fp16/bf16 để giảm VRAM cho T4; điểm số "
        "vẫn float32 theo spec.",
    )
    return parser


def _load_dense_encoder(
    name: EncoderName, *, local_files_only: bool, device: Literal["cpu", "cuda"] = "cpu"
) -> SentenceTransformerDenseEncoder:
    spec = approved_encoder_spec(name)
    if device != "cpu":
        spec = spec.model_copy(update={"device": device})
    return SentenceTransformerDenseEncoder(spec, local_files_only=local_files_only)


def _checked_dense_corpus(
    corpus_dir: Path, *, release_fingerprint: str, lock_sha256: str
) -> DenseCorpus:
    corpus = load_dense_corpus(corpus_dir, release_lock_sha256=lock_sha256)
    if corpus.manifest.dataset_fingerprint != release_fingerprint:
        raise DenseArtifactError("Dense corpus fingerprint does not match release lock")
    return corpus


def _checked_row_dense_corpus(
    corpus_dir: Path, *, release_fingerprint: str, lock_sha256: str
) -> RowDenseCorpus:
    corpus = load_row_dense_corpus(corpus_dir, release_lock_sha256=lock_sha256)
    if corpus.manifest.dataset_fingerprint != release_fingerprint:
        raise DenseArtifactError("Row dense corpus fingerprint does not match release lock")
    return corpus


def _load_build_observation(
    path: Path,
    *,
    encoder_name: EncoderName,
    encoder_hash: str,
    dataset_fingerprint: str,
) -> _DenseBuildObservation:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise DenseArtifactError("Dense build observation must be a JSON object")
    try:
        observation = _DenseBuildObservation.model_validate(payload)
    except ValidationError as exc:
        raise DenseArtifactError(
            "Dense build observation is missing required FAISS device metadata; rebuild the index"
        ) from exc
    if (
        observation.encoder_name != encoder_name
        or observation.encoder_spec_sha256 != encoder_hash
        or observation.dataset_fingerprint != dataset_fingerprint
    ):
        raise DenseArtifactError("Dense build observation does not match requested artifacts")
    return observation


def _write_dense_run(run: DenseEvaluationRun, path: Path) -> None:
    write_text_atomic(
        path,
        json.dumps(run.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
    )


def _load_gold_for_cli(
    path: Path,
    release: ResolvedRetrievalRelease,
    version: str | None,
) -> tuple[GoldRetrievalQuestion, ...]:
    """Keep legacy fixture/custom gold behavior while locking named real snapshots."""
    if version is None:
        return load_gold_questions(path, release)
    resolved = resolve_gold_reference(path, version=cast(ReferenceVersion, version))
    return load_gold_questions(
        path,
        release,
        require_count=resolved.descriptor.question_count,
        question_ids=resolved.selected_question_ids,
    )


def write_sweep_report(
    results: Sequence[SweepResult], recommended_k: int, output_stem: Path
) -> tuple[Path, Path]:
    """Ghi báo cáo sweep ra <stem>.json và <stem>.md; trả về hai đường dẫn."""
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_stem.with_suffix(".json")
    markdown_path = output_stem.with_suffix(".md")
    payload = {
        "recommended_k": recommended_k,
        "results": [{"k": item.k, "f2": item.f2, "mrr5": item.mrr5} for item in results],
    }
    write_text_atomic(json_path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    write_text_atomic(markdown_path, render_sweep_markdown(results, recommended_k))
    return json_path, markdown_path


# Mặc định theo kế hoạch Task 8: QueryEmbeddingCache tự thêm thư mục con
# `<encoder_spec_sha256[:12]>` bên dưới root này.
_TABLE_DENSE_QUERY_CACHE_DEFAULT = Path("data/indexes/dense-query-cache/qwen3-4b")
_RERANK_CACHE_DEFAULT = Path("data/indexes/rerank-score-cache")


def _locate_table_dense_corpus(dense_index_dir: Path) -> Path:
    """Tìm corpus đi kèm một dense index tầng bảng.

    `load_dense_index` cần đúng corpus đã dùng lúc build (kiểm tra
    `document_sha256`), nhưng cờ CLI chỉ nhận thư mục index nên corpus được
    tìm theo quy ước bố trí: ngay trong thư mục index (`<index>/corpus`) hoặc
    cạnh nó dưới cùng một thư mục cha (`<cha>/corpus`).
    """
    candidates = (dense_index_dir / "corpus", dense_index_dir.parent / "corpus")
    for candidate in candidates:
        if (candidate / "manifest.json").is_file():
            return candidate
    searched = "; ".join(str(candidate) for candidate in candidates)
    raise DenseArtifactError(
        "--dense-index cần cả dense corpus nhưng không thấy manifest.json ở "
        f"{searched}. Hãy chạy `retrieval build-dense-corpus --output-root "
        "<thư mục cha của dense-index>` hoặc đặt 'corpus' vào trong thư mục "
        "dense-index."
    )


def _build_table_retriever(
    args: argparse.Namespace,
    release: ResolvedRetrievalRelease,
    index: BM25Index,
    *,
    encoder: DenseEncoder | None = None,
    reranker: Reranker | None = None,
) -> tuple[TableRetriever, Reranker | None]:
    """Lắp bộ retrieve tầng bảng cho đường live, dùng chung bởi
    `submission export` và `retrieval sweep-k` để hai CLI không tự lắp hai
    stack khác nhau cho cùng một câu hỏi.

    Ba chế độ:
    - mặc định (không cờ): BM25-only, hành vi cũ nguyên vẹn;
    - `--dense-index`: fusion BM25+dense bằng weighted RRF
      (`FusionWeights(bm25=1.0, dense=--table-dense-weight)`);
    - `--dense-index --rerank`: cộng thêm cross-encoder xếp lại top-50.

    `--rerank` không có `--dense-index` bị từ chối: reranker chỉ nhận ứng
    viên fused (`FusedCandidate`), chạy trên BM25 thuần là sai ngữ nghĩa.

    `encoder`/`reranker` là điểm tiêm cho test -- production để `None` và nhận
    encoder dựng từ chính manifest của dense index (nên spec hash luôn khớp
    index) và reranker pinned `qwen3-reranker-4b`.

    Về bộ nhớ: chỉ việc embed *corpus* là offline. Encoder dense vẫn phải
    thường trú lúc chạy để embed từng câu hỏi, nên khi bật `--rerank` cả hai
    model cùng nằm trong RAM (~16GB + ~16GB fp32, xem help của `--rerank`).
    `--rerank-dtype` hạ precision TÍNH TOÁN của reranker (fp16/bf16, không thuộc
    spec) để model 4B vừa VRAM T4; điểm trả ra vẫn float32.
    Rerank tuần tự *sau* fusion trong mỗi câu, nhưng đó là thứ tự thực thi --
    không phải là hai model thay phiên nhau chiếm chỗ.
    """
    bm25_service = RetrievalService(index)
    dense_index_dir: Path | None = getattr(args, "dense_index", None)
    wants_rerank = bool(getattr(args, "rerank", False))

    if dense_index_dir is None:
        if wants_rerank:
            raise DenseInputError("--rerank cần --dense-index: rerank chỉ chạy trên ứng viên fused")
        # cast: RetrievalTrace structurally satisfies TableRetriever but mypy
        # cannot prove it against the _RankedResult protocol (same known
        # pattern as the row_recall_evaluation / submission exporter callers).
        return cast(TableRetriever, bm25_service), None

    corpus = load_dense_corpus(
        _locate_table_dense_corpus(dense_index_dir), release_lock_sha256=release.lock_sha256
    )
    if corpus.manifest.dataset_fingerprint != release.dataset_fingerprint:
        raise DenseArtifactError(
            "--dense-index dataset_fingerprint does not match --release-lock"
        )

    # Encoder spec đến từ chính manifest của index: spec hash ghi trong
    # manifest và hash tính từ spec này phải trùng nhau qua `load_dense_index`
    # lẫn kiểm tra của `DenseRetrievalService`.
    manifest = DenseIndexManifest.model_validate(
        json.loads((dense_index_dir / "manifest.json").read_text(encoding="utf-8"))
    )
    if encoder is None:
        encoder = SentenceTransformerDenseEncoder(manifest.encoder)
    loaded_dense_index = load_dense_index(
        dense_index_dir,
        corpus,
        expected_encoder_spec_sha256=encoder_spec_sha256(encoder.spec),
        release_lock_sha256=release.lock_sha256,
    )
    cache = QueryEmbeddingCache(_TABLE_DENSE_QUERY_CACHE_DEFAULT, encoder.spec)
    dense_service = DenseRetrievalService(loaded_dense_index, encoder, cache)

    if wants_rerank:
        if reranker is None:
            spec = approved_reranker_spec("qwen3-reranker-4b")
            # `row-batches` and `export` rank under identical settings by
            # construction, so the second of the two runs scores exactly the
            # pairs the first already scored. Wrapping the model in the cache
            # makes that second pass free -- and, on a full hit, model-free:
            # `CachedReranker` only builds the 8GB cross-encoder on a miss.
            cache_dir: Path = (
                getattr(args, "rerank_cache_dir", None) or _RERANK_CACHE_DEFAULT
            )
            # Compute-only dtype (giống SentenceTransformerDenseEncoder.model_dtype):
            # không thuộc RerankerSpec nên spec hash lẫn cache key không đổi;
            # "float32"/vắng mặt -> None giữ đúng hợp đồng fp32 của spec.
            rerank_dtype: str | None = getattr(args, "rerank_dtype", None)
            model_dtype: Literal["float16", "bfloat16"] | None = (
                None
                if rerank_dtype in (None, "float32")
                else cast(Literal["float16", "bfloat16"], rerank_dtype)
            )
            reranker = CachedReranker(
                cache_dir,
                spec,
                factory=lambda: Qwen3CrossEncoderReranker(spec, model_dtype=model_dtype),
            )
    else:
        reranker = None

    fusion = FusionService(
        bm25_service,
        dense_service,
        FusionWeights(bm25=1.0, dense=getattr(args, "table_dense_weight", 1.0)),
    )
    return cast(TableRetriever, fusion), reranker


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "cleanup-day9-data":
            plan = plan_day9_cleanup(args.repo_root)
            for entry in plan.entries:
                print(
                    json.dumps(
                        {
                            "path": str(entry.path),
                            "reason": entry.reason,
                            "status": entry.status,
                            "byte_count": entry.byte_count,
                            "detail": entry.detail,
                        },
                        sort_keys=True,
                    )
                )
            if not args.apply:
                return 0
            for destination in quarantine_day9_cleanup(plan, args.quarantine_root):
                print(json.dumps({"action": "moved", "destination": str(destination)}))
            return 2 if any(entry.status == "blocked" for entry in plan.entries) else 0
        root = args.repo_root if getattr(args, "repo_root", None) is not None else Path.cwd()
        release = resolve_retrieval_release(args.release_lock, repo_root=root)
        if args.command == "build-index":
            documents = build_table_documents(
                release.release_dir / "documents.parquet",
                release.release_dir / "tables.parquet",
                release.release_dir / "cells.parquet",
            )
            index = build_bm25_index(
                documents,
                dataset_fingerprint=release.dataset_fingerprint,
                release_lock_sha256=release.lock_sha256,
            )
            target = args.output_root / release.dataset_fingerprint
            save_bm25_index(index, target)

            # Build and save row index
            row_docs = build_row_documents(
                release.release_dir / "documents.parquet",
                release.release_dir / "tables.parquet",
                release.release_dir / "cells.parquet",
            )
            row_index = build_row_bm25_index(
                row_docs,
                dataset_fingerprint=release.dataset_fingerprint,
                release_lock_sha256=release.lock_sha256,
            )
            row_target = args.output_root / f"{release.dataset_fingerprint}_row"
            save_row_bm25_index(row_index, row_target)

            print(target)
            print(row_target)
            return 0
        if args.command == "build-dense-corpus":
            documents = build_table_documents(
                release.release_dir / "documents.parquet",
                release.release_dir / "tables.parquet",
                release.release_dir / "cells.parquet",
            )
            corpus = build_dense_corpus(
                documents,
                dataset_fingerprint=release.dataset_fingerprint,
                release_lock_sha256=release.lock_sha256,
            )
            target = args.output_root / release.dataset_fingerprint / "corpus"
            save_dense_corpus(corpus, target)

            # Build and save row dense corpus
            row_docs = build_row_documents(
                release.release_dir / "documents.parquet",
                release.release_dir / "tables.parquet",
                release.release_dir / "cells.parquet",
            )
            row_corpus = build_row_dense_corpus(
                row_docs,
                dataset_fingerprint=release.dataset_fingerprint,
                release_lock_sha256=release.lock_sha256,
            )
            row_target = args.output_root / release.dataset_fingerprint / "row_corpus"
            save_row_dense_corpus(row_corpus, row_target)

            print(target)
            print(row_target)
            return 0
        if args.command == "build-dense-index":
            encoder_name = cast(EncoderName, args.encoder)
            faiss_device = cast(Literal["cpu", "cuda"], args.faiss_device)
            encoder_device = cast(Literal["cpu", "cuda"], args.encoder_device)
            encoder = _load_dense_encoder(
                encoder_name, local_files_only=args.local_files_only, device=encoder_device
            )
            corpus = _checked_dense_corpus(
                args.corpus_dir,
                release_fingerprint=release.dataset_fingerprint,
                lock_sha256=release.lock_sha256,
            )
            get_num_gpus = getattr(faiss, "get_num_gpus", None)
            faiss_gpu_count = get_num_gpus() if callable(get_num_gpus) else 0

            def report_progress(encoded: int, total: int, elapsed: float) -> None:
                rate = encoded / elapsed if elapsed > 0 else 0.0
                print(
                    f"dense-build: {encoded}/{total} vectors, {elapsed:.1f}s, {rate:.1f} vectors/s",
                    flush=True,
                )

            target = args.output_root / f"{encoder_name}-{encoder_spec_sha256(encoder.spec)[:12]}"
            if faiss_device == "cuda" and target.exists():
                raise DenseArtifactError(
                    "Refusing CUDA build into existing dense index target; "
                    "remove the target before rebuilding"
                )
            started = time.perf_counter()
            built = build_dense_index(
                corpus,
                encoder,
                faiss_device=faiss_device,
                progress=report_progress,
            )
            save_dense_index(built, target)

            # Build and save row dense index
            row_corpus_dir = args.corpus_dir.parent / "row_corpus"
            row_corpus = _checked_row_dense_corpus(
                row_corpus_dir,
                release_fingerprint=release.dataset_fingerprint,
                lock_sha256=release.lock_sha256,
            )
            row_built = build_row_dense_index(
                row_corpus,
                encoder,
                faiss_device=faiss_device,
                progress=report_progress,
            )
            row_target = target.parent / f"{target.name}_row"
            save_row_dense_index(row_built, row_target)

            print("dense-build: complete", flush=True)
            observation = _DenseBuildObservation(
                encoder_name=encoder_name,
                encoder_spec_sha256=encoder_spec_sha256(encoder.spec),
                dataset_fingerprint=release.dataset_fingerprint,
                build_seconds=time.perf_counter() - started,
                index_byte_size=(target / "index.faiss").stat().st_size,
                faiss_device=faiss_device,
                faiss_gpu_count=faiss_gpu_count,
            )
            write_text_atomic(
                args.observation_path,
                json.dumps(
                    observation.model_dump(mode="json"),
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
            )
            print(target)
            return 0
        if args.command == "validate-gold":
            gold = _load_gold_for_cli(args.gold_path, release, args.gold_version)
            print(f"validated {len(gold)} reviewed retrieval questions")
            return 0
        if args.command == "evaluate-dense":
            gold = _load_gold_for_cli(args.gold_path, release, args.gold_version)
            encoder_name = cast(EncoderName, args.encoder)
            encoder_device = cast(Literal["cpu", "cuda"], args.encoder_device)
            encoder = _load_dense_encoder(
                encoder_name, local_files_only=True, device=encoder_device
            )
            encoder_hash = encoder_spec_sha256(encoder.spec)
            corpus = _checked_dense_corpus(
                args.corpus_dir,
                release_fingerprint=release.dataset_fingerprint,
                lock_sha256=release.lock_sha256,
            )
            dense_index = load_dense_index(
                args.index_dir,
                corpus,
                expected_encoder_spec_sha256=encoder_hash,
                release_lock_sha256=release.lock_sha256,
            )
            observation = _load_build_observation(
                args.observation_path,
                encoder_name=encoder_name,
                encoder_hash=encoder_hash,
                dataset_fingerprint=release.dataset_fingerprint,
            )
            if observation.index_byte_size != dense_index.manifest.index_byte_size:
                raise DenseArtifactError("Dense build observation index size does not match index")
            service = DenseRetrievalService(
                dense_index,
                encoder,
                QueryEmbeddingCache(args.cache_dir, encoder.spec),
            )
            run = evaluate_cold_and_warm(
                service,
                gold,
                build_seconds=observation.build_seconds,
                index_byte_size=observation.index_byte_size,
            )
            _write_dense_run(run, args.output_path)
            print(args.output_path)
            return 0
        if args.command == "compare-day9":
            try:
                bm25_report = load_bm25_reference_report(args.bm25_report).report
            except ValueError as exc:
                raise DenseArtifactError("BM25 reference artifact is invalid") from exc
            bge_run = DenseEvaluationRun.model_validate_json(
                args.bge_report.read_text(encoding="utf-8")
            )
            e5_run = DenseEvaluationRun.model_validate_json(
                args.e5_report.read_text(encoding="utf-8")
            )
            try:
                comparison = build_day9_comparison(bm25_report, bge_run, e5_run)
            except ValueError as exc:
                raise DenseArtifactError("Day 9 comparison inputs are invalid") from exc
            json_path, markdown_path = write_day9_comparison(comparison, args.output_dir)
            print(json_path)
            print(markdown_path)
            return 0
        if args.command == "evaluate-fusion":
            gold = _load_gold_for_cli(args.gold_path, release, args.gold_version)
            try:
                bm25_index = load_bm25_index(
                    args.index_dir, release_lock_sha256=release.lock_sha256
                )
            except ValueError as exc:
                raise RetrievalArtifactError(f"BM25 index artifact is invalid: {exc}") from exc
            if bm25_index.manifest.dataset_fingerprint != release.dataset_fingerprint:
                raise RetrievalArtifactError("BM25 index fingerprint does not match release lock")
            bm25_service = RetrievalService(bm25_index)

            encoder_name = cast(EncoderName, args.encoder)
            encoder_device = cast(Literal["cpu", "cuda"], args.encoder_device)
            encoder = _load_dense_encoder(
                encoder_name, local_files_only=True, device=encoder_device
            )
            encoder_hash = encoder_spec_sha256(encoder.spec)
            corpus = _checked_dense_corpus(
                args.corpus_dir,
                release_fingerprint=release.dataset_fingerprint,
                lock_sha256=release.lock_sha256,
            )
            dense_index = load_dense_index(
                args.dense_index_dir,
                corpus,
                expected_encoder_spec_sha256=encoder_hash,
                release_lock_sha256=release.lock_sha256,
            )
            dense_service = DenseRetrievalService(
                dense_index, encoder, QueryEmbeddingCache(args.cache_dir, encoder.spec)
            )
            try:
                bm25_report = load_bm25_reference_report(args.bm25_report).report
                grid_report = evaluate_fusion_grid(bm25_service, dense_service, gold, bm25_report)
            except ValueError as exc:
                raise FusionArtifactError("Fusion grid evaluation inputs are invalid") from exc
            json_path, markdown_path = write_day10_fusion(grid_report, args.output_dir)
            print(json_path)
            print(markdown_path)
            return 0
        if args.command == "sweep-k":
            release = resolve_retrieval_release(args.release_lock, repo_root=Path.cwd())
            index = load_bm25_index(args.bm25_index)
            if index.manifest.dataset_fingerprint != release.dataset_fingerprint:
                raise RetrievalArtifactError(
                    "--bm25-index dataset_fingerprint does not match --release-lock"
                )
            questions = load_gold_questions(
                args.gold, release, require_count=args.gold_count
            )
            retriever, reranker = _build_table_retriever(args, release, index)
            results = run_sweep(questions, retriever, ks=tuple(args.ks), reranker=reranker)
            best = recommend_k(results)
            json_path, markdown_path = write_sweep_report(results, best, args.output_stem)
            print(render_sweep_markdown(results, best), end="")
            print(f"k*={best}")
            print(json_path)
            print(markdown_path)
            return 0
        try:
            gold = _load_gold_for_cli(args.gold_path, release, args.gold_version)
            index = load_bm25_index(args.index_dir, release_lock_sha256=release.lock_sha256)
        except ValueError as exc:
            raise RetrievalArtifactError(f"BM25 index artifact is invalid: {exc}") from exc
        if index.manifest.dataset_fingerprint != release.dataset_fingerprint:
            raise RetrievalArtifactError("BM25 index fingerprint does not match release lock")
        if args.command == "evaluate-v2":
            try:
                report_v2 = evaluate_retrieval_v2(
                    RetrievalService(index), gold, diagnostic_k=args.diagnostic_k
                )
            except ValueError as exc:
                raise RetrievalArtifactError("V2 diagnostic ranking is invalid") from exc
            json_path, markdown_path = write_report_v2(report_v2, args.output_dir)
        else:
            legacy_report = evaluate_retrieval(RetrievalService(index), gold)
            json_path, markdown_path = write_report(legacy_report, args.output_dir)
        print(json_path)
        print(markdown_path)
        return 0
    except (
        RetrievalInputError,
        RetrievalArtifactError,
        DenseInputError,
        DenseArtifactError,
        DenseModelError,
        FusionInputError,
        FusionArtifactError,
        GraphInputError,
        GraphArtifactError,
        ExpansionInputError,
        ExpansionArtifactError,
        RerankError,
        ValidationError,
        JSONDecodeError,
        RuntimeError,
        OSError,
    ) as exc:
        print(f"retrieval error: {exc}", file=sys.stderr)
        return 2
