"""Command-line interface for the Day 22 submission bundle (export/validate)."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from decimal import Decimal
from json import JSONDecodeError
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import ValidationError

if TYPE_CHECKING:
    from financial_report_qa.retrieval.dense_encoder import DenseEncoder
    from financial_report_qa.retrieval.release import ResolvedRetrievalRelease
    from financial_report_qa.retrieval.row_dense_service import RowDenseRetrievalService
    from financial_report_qa.retrieval.row_fusion import RowFusionService

from financial_report_qa.core.config import load_execution_settings
from financial_report_qa.core.errors import (
    ExecutionError,
    PlanningArtifactError,
    PlanningInputError,
    RetrievalError,
    SubmissionError,
)
from financial_report_qa.planning.entity_parser import parse_query_entities
from financial_report_qa.planning.program_decisions import load_program_decisions
from financial_report_qa.planning.row_choice_batch import build_program_batch_payload
from financial_report_qa.retrieval.cli import _build_table_retriever
from financial_report_qa.retrieval.index import load_bm25_index
from financial_report_qa.retrieval.live_query import (
    retrieve_candidate_table_ids,
)
from financial_report_qa.retrieval.release import resolve_retrieval_release
from financial_report_qa.retrieval.row_fusion import DEFAULT_ROW_CANDIDATE_COUNT
from financial_report_qa.submission.compliance import check_bundle
from financial_report_qa.submission.contracts import SubmissionExportReport
from financial_report_qa.submission.exporter import (
    build_question_cell_candidates,
    export_submission,
    load_raw_questions,
    write_export_report,
    write_submission_zip,
)
from financial_report_qa.submission.retrieval_fingerprint import (
    RetrievalFingerprint,
    assert_fingerprint_matches,
    write_retrieval_fingerprint,
)
from financial_report_qa.submission.validator import validate_submission_zip


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="financial-report-qa submission")
    commands = parser.add_subparsers(dest="command", required=True)

    export = commands.add_parser("export")
    export.add_argument("--release-lock", type=Path, required=True)
    export.add_argument("--bm25-index", type=Path, required=True)
    export.add_argument(
        "--questions-path",
        type=Path,
        required=True,
        help="Official question file, one JSON object per line: "
        '{"id": int, "question": str} (e.g. data/raw/ViFinQA/questions/questions.jsonl).',
    )
    export.add_argument("--execution-config", type=Path, nargs="+", required=True)
    export.add_argument("--output-zip", type=Path, required=True)
    export.add_argument("--report-dir", type=Path, required=True)
    export.add_argument("--k", type=int, default=10)
    export.add_argument(
        "--row-dense-corpus",
        type=Path,
        default=None,
        help="Row dense corpus dir from `retrieval build-dense-corpus` (the "
        "'row_corpus' sibling of the table 'corpus' dir). Requires "
        "--row-dense-index and --dense-encoder too; row fusion falls back to "
        "bm25+fuzzy+alias only (dense weight 0) when any of the three is omitted.",
    )
    export.add_argument(
        "--row-dense-index",
        type=Path,
        default=None,
        help="Row dense FAISS index dir from `retrieval build-dense-index` "
        "(the '..._row' sibling of the table dense index dir).",
    )
    export.add_argument(
        "--dense-encoder",
        choices=("bge-m3", "multilingual-e5-small"),
        default=None,
        help="Encoder the --row-dense-index was built with (must match its manifest).",
    )
    export.add_argument(
        "--dense-cache-dir",
        type=Path,
        default=None,
        help="Query embedding cache dir (default: --row-dense-index parent / 'query-cache').",
    )
    export.add_argument(
        "--dense-local-files-only",
        action="store_true",
        help="Never fetch the dense encoder from the network; require a local HF cache hit.",
    )
    export.add_argument(
        "--dense-weight",
        type=float,
        default=0.0,
        help="Row-fusion weight for the dense branch (default 0.0, i.e. off, even when "
        "--row-dense-corpus/--row-dense-index/--dense-encoder are all given; do not "
        "confuse with --table-dense-weight, the table-retrieval RRF weight). plan.md §20's "
        "row-recall benchmark (58 gold questions, dense weight 0.5) measured dense making "
        "Row Recall@3/@5 *worse* than bm25+fuzzy+alias alone (74.1%%->67.2%%, 82.8%%->74.1%%; "
        "@1/@10 unchanged) -- re-measure with `retrieval.row_recall_evaluation` before "
        "raising this above 0.",
    )
    export.add_argument(
        "--program-decisions",
        type=Path,
        required=True,
        help=(
            "File JSONL quyết định masked-PAL (spec 2026-08-24 §4.3) sinh "
            "offline bởi lệnh `submission row-batches` -- lệnh này giờ chỉ "
            "sinh payload masked-PAL (yêu cầu --release-dir), không còn chế "
            "độ nào khác. Bắt buộc: đây là đường answering duy nhất."
        ),
    )
    export.add_argument(
        "--assert-payload-fingerprint",
        type=Path,
        default=None,
        help=(
            "Đường dẫn tới retrieval-fingerprint.json do `submission "
            "row-batches` ghi cạnh các file batch. Trước khi chạy câu nào, "
            "lượt export này phải khớp cài đặt retrieval lúc sinh payload "
            "(k, rows_per_question, reranker, --dense-index, release lock); "
            "lệch trường nào bị từ chối ngay, nêu rõ tên trường -- vì lệch đó "
            "làm dịch mọi chỉ số ProgramDecision.cells. Không truyền thì "
            "không khẳng định gì (hành vi cũ)."
        ),
    )
    export.add_argument(
        "--dense-index",
        type=Path,
        default=None,
        help="Bật fusion BM25+dense cho TẦNG BẢNG (khác --row-dense-index của "
        "row fusion): thư mục dense index (manifest.json + index.faiss). Corpus "
        "đi kèm được tìm ở <dense-index>/corpus hoặc <thư mục cha>/corpus. Không "
        "truyền thì tầng bảng chạy BM25-only như cũ.",
    )
    export.add_argument(
        "--table-dense-weight",
        type=float,
        default=1.0,
        help="Trọng số nhánh dense trong RRF của tầng bảng (bm25 luôn = 1.0). "
        "Khác --dense-weight bên trên, vốn là trọng số dense của ROW fusion.",
    )
    export.add_argument(
        "--rerank-cache-dir",
        type=Path,
        default=None,
        help="Nơi lưu điểm cross-encoder (mặc định data/indexes/"
        "rerank-score-cache). Dùng CHUNG giữa `row-batches` và `export` thì "
        "reranker chỉ tốn GPU đúng một lần; lần thứ hai chạy không cần model.",
    )
    export.add_argument(
        "--rerank",
        action="store_true",
        help="Xếp lại top-50 của RRF tầng bảng bằng Qwen3-Reranker-4B (pinned). "
        "Cần --dense-index và ~32GB RAM (encoder dense ~16GB fp32 vẫn thường trú "
        "khi reranker nạp thêm ~16GB); Colab là nơi chạy phù hợp cho phép đo "
        "fused+rerank.",
    )
    export.add_argument(
        "--rerank-dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
        help="Compute-only: nạp reranker ở fp16/bf16 để giảm VRAM cho T4; điểm số "
        "vẫn float32 theo spec.",
    )
    export.add_argument(
        "--table-encoder-device",
        default="cpu",
        help="Compute-only: đặt encoder dense tầng bảng lên cpu/cuda/cuda:0/cuda:1 "
        "(không thuộc spec).",
    )
    export.add_argument(
        "--table-encoder-model-dtype",
        choices=("float32", "float16"),
        default=None,
        help="Compute-only dtype của encoder dense tầng bảng; bỏ trống thì float16 "
        "khi --table-encoder-device là cuda*, ngược lại float32.",
    )
    export.add_argument(
        "--rerank-device",
        default="cpu",
        help="Compute-only: đặt cross-encoder lên cpu/cuda/cuda:0/cuda:1 (không thuộc spec).",
    )

    batches = commands.add_parser(
        "row-batches",
        help=(
            "Chạy retrieval + row fusion cho mọi câu hỏi và ghi payload ứng "
            "viên Ô ra JSONL để LLM sinh chương trình masked offline (spec "
            "2026-08-24 §4.3)."
        ),
    )
    batches.add_argument("--release-lock", type=Path, required=True)
    batches.add_argument("--bm25-index", type=Path, required=True)
    batches.add_argument("--questions-path", type=Path, required=True)
    batches.add_argument("--output-dir", type=Path, required=True)
    batches.add_argument(
        "--release-dir",
        type=Path,
        required=True,
        help="Thư mục release Parquet, để dựng cell frame cho ứng viên Ô.",
    )
    batches.add_argument("--k", type=int, default=10, help="Số bảng ứng viên mỗi câu.")
    batches.add_argument(
        "--rows-per-question",
        type=int,
        # Must match `export`'s row-fusion `k` (DEFAULT_ROW_CANDIDATE_COUNT):
        # decision files built from these batches reference candidate indices
        # up to this count - 1, and `export` must retrieve at least as many
        # row candidates or those indices will look out of range.
        default=DEFAULT_ROW_CANDIDATE_COUNT,
        help="Số dòng ứng viên mỗi câu.",
    )
    batches.add_argument("--batch-size", type=int, default=64, help="Số câu mỗi file batch.")
    # Ba cờ dưới đây PHẢI khớp `export`: `ProgramDecision.cells` là vị trí
    # trong danh sách ô ứng viên, mà danh sách đó dựng từ `retrieved`. Sinh
    # payload bằng BM25 rồi export bằng fusion+rerank sẽ dịch mọi chỉ số.
    # `retrieval-fingerprint.json` ghi lại chúng và
    # `export --assert-payload-fingerprint` chặn nếu lệch.
    batches.add_argument(
        "--dense-index",
        type=Path,
        default=None,
        help="Bật fusion BM25+dense cho tầng bảng lúc sinh payload. Phải "
        "trùng --dense-index của `export`.",
    )
    batches.add_argument(
        "--table-dense-weight",
        type=float,
        default=1.0,
        help="Trọng số nhánh dense trong RRF tầng bảng (bm25 luôn = 1.0). "
        "Phải trùng `export`.",
    )
    batches.add_argument(
        "--rerank-cache-dir",
        type=Path,
        default=None,
        help="Nơi lưu điểm cross-encoder (mặc định data/indexes/"
        "rerank-score-cache). Dùng CHUNG giữa `row-batches` và `export` thì "
        "reranker chỉ tốn GPU đúng một lần; lần thứ hai chạy không cần model.",
    )
    batches.add_argument(
        "--rerank",
        action="store_true",
        help="Xếp lại top-50 của RRF bằng Qwen3-Reranker-4B. Cần "
        "--dense-index. Phải trùng `export`.",
    )
    batches.add_argument(
        "--rerank-dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
        help="Compute-only: nạp reranker ở fp16/bf16 để giảm VRAM cho T4; điểm "
        "số vẫn float32 theo spec. Phải trùng `export`.",
    )
    batches.add_argument(
        "--table-encoder-device",
        default="cpu",
        help="Compute-only: đặt encoder dense tầng bảng lên cpu/cuda/cuda:0/"
        "cuda:1 (không thuộc spec). Phải trùng `export`.",
    )
    batches.add_argument(
        "--table-encoder-model-dtype",
        choices=("float32", "float16"),
        default=None,
        help="Compute-only dtype của encoder dense tầng bảng; bỏ trống thì "
        "float16 khi --table-encoder-device là cuda*, ngược lại float32. "
        "Phải trùng `export`.",
    )
    batches.add_argument(
        "--rerank-device",
        default="cpu",
        help="Compute-only: đặt cross-encoder lên cpu/cuda/cuda:0/cuda:1 "
        "(không thuộc spec). Phải trùng `export`.",
    )

    validate = commands.add_parser("validate")
    validate.add_argument("--zip-path", type=Path, required=True)
    validate.add_argument(
        "--report-path",
        type=Path,
        required=True,
        help="A submission-export-*.json written by `export` -- the validator "
        "checks the ZIP against exactly the ids that report marked 'answered', "
        "not the full official question file (Day 22 plan §2 decision G: "
        "coverage may legitimately be partial).",
    )
    validate.add_argument("--tolerance", type=str, default="0.01")

    return parser


def _load_row_dense_service(
    args: argparse.Namespace,
    release: ResolvedRetrievalRelease,
    *,
    encoder: DenseEncoder | None = None,
) -> RowDenseRetrievalService | None:
    """Load the row dense retrieval branch from `--row-dense-corpus`/
    `--row-dense-index`/`--dense-encoder`, or return `None` if any of the
    three is missing or fails to load (row fusion then just runs without
    the dense branch -- dense weight 0 in `RowFusionWeights` degrades
    exactly to the pre-dense-wiring bm25+fuzzy+alias behavior).

    `encoder` is an injection seam for tests -- production always leaves it
    `None` and gets the real pinned `SentenceTransformerDenseEncoder`.

    Reads `args.row_dense_corpus`/`--row-dense-index`/`--dense-encoder` via
    `getattr` with a `None` default: the `row-batches` command has no CLI
    flags for these (dense row retrieval is not exposed there), so it always
    degrades to the bm25+fuzzy+alias-only branch below rather than crashing
    on a missing attribute."""
    row_dense_corpus = getattr(args, "row_dense_corpus", None)
    row_dense_index = getattr(args, "row_dense_index", None)
    dense_encoder = getattr(args, "dense_encoder", None)
    if row_dense_corpus is None or row_dense_index is None or dense_encoder is None:
        return None

    from financial_report_qa.retrieval.dense_cache import QueryEmbeddingCache
    from financial_report_qa.retrieval.dense_encoder import (
        SentenceTransformerDenseEncoder,
        approved_encoder_spec,
        encoder_spec_sha256,
    )
    from financial_report_qa.retrieval.row_dense_corpus import load_row_dense_corpus
    from financial_report_qa.retrieval.row_dense_index import load_row_dense_index
    from financial_report_qa.retrieval.row_dense_service import RowDenseRetrievalService

    try:
        row_corpus = load_row_dense_corpus(
            row_dense_corpus, release_lock_sha256=release.lock_sha256
        )
        if row_corpus.manifest.dataset_fingerprint != release.dataset_fingerprint:
            print(
                "Warning: --row-dense-corpus dataset_fingerprint does not match "
                "--release-lock; skipping row dense retrieval",
                file=sys.stderr,
            )
            return None
        if encoder is None:
            spec = approved_encoder_spec(dense_encoder)
            encoder = SentenceTransformerDenseEncoder(
                spec, local_files_only=getattr(args, "dense_local_files_only", False)
            )
        loaded_row_dense_index = load_row_dense_index(
            row_dense_index,
            row_corpus,
            expected_encoder_spec_sha256=encoder_spec_sha256(encoder.spec),
            release_lock_sha256=release.lock_sha256,
        )
        cache_dir = getattr(args, "dense_cache_dir", None) or (
            row_dense_index.parent / "query-cache"
        )
        cache = QueryEmbeddingCache(cache_dir, encoder.spec)
        return RowDenseRetrievalService(loaded_row_dense_index, encoder, cache)
    except Exception as e:
        print(f"Warning: Failed to load row dense index: {e}", file=sys.stderr)
        return None


def _build_row_fusion(
    args: argparse.Namespace, release: ResolvedRetrievalRelease
) -> RowFusionService | None:
    """Load the row BM25 index and assemble the shared `RowFusionService`
    (bm25 + optional dense + fuzzy + alias), or return `None` if the row
    index directory is missing or fails to load.

    Shared by the `export` and `row-batches` commands so both see exactly
    the same fused row candidates for a given question -- a divergence here
    would make a downstream LLM's `chosen_index` point at the wrong row.
    `row-batches` has no CLI flags for dense retrieval, so `getattr(args,
    "dense_weight", 0.0)` degrades it to the pre-dense-wiring
    bm25+fuzzy+alias behavior, identical to what `export` gets when its own
    `--dense-weight` is left at its 0.0 default."""
    from financial_report_qa.retrieval.row_fusion import RowFusionService
    from financial_report_qa.retrieval.row_fusion_contracts import RowFusionWeights
    from financial_report_qa.retrieval.row_index import load_row_bm25_index
    from financial_report_qa.retrieval.row_lexical import (
        RowAliasRetrievalService,
        RowFuzzyRetrievalService,
    )
    from financial_report_qa.retrieval.row_service import RowRetrievalService

    row_index_dir = args.bm25_index.parent / f"{args.bm25_index.name}_row"
    if not row_index_dir.is_dir():
        return None

    try:
        row_index = load_row_bm25_index(row_index_dir, release_lock_sha256=release.lock_sha256)
        row_service = RowRetrievalService(row_index)
        # Dense (plan.md §7) is opt-in via --row-dense-corpus/
        # --row-dense-index/--dense-encoder; loading it still
        # defaults to weight 0.0 (--dense-weight) -- plan.md §20's
        # benchmark measured 0.5 making Row Recall@3/@5 worse,
        # not better, than bm25+fuzzy+alias alone.
        row_dense_service = _load_row_dense_service(args, release)
        return RowFusionService(
            bm25=row_service,
            dense=row_dense_service,
            weights=RowFusionWeights(
                bm25=1.0, dense=getattr(args, "dense_weight", 0.0), fuzzy=0.3, alias=0.2
            ),
            fuzzy=RowFuzzyRetrievalService(row_index),
            alias=RowAliasRetrievalService(row_index),
        )
    except Exception as e:
        print(f"Warning: Failed to load row BM25 index: {e}", file=sys.stderr)
        return None


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "export":
            root = Path.cwd()
            release = resolve_retrieval_release(args.release_lock, repo_root=root)
            index = load_bm25_index(args.bm25_index)
            if index.manifest.dataset_fingerprint != release.dataset_fingerprint:
                raise SubmissionError(
                    "--bm25-index dataset_fingerprint does not match --release-lock"
                )
            try:
                retriever, reranker = _build_table_retriever(args, release, index)
            except RetrievalError as exc:
                # Lỗi lắp bộ retrieve tầng bảng (fingerprint lệch, corpus thiếu,
                # model không tải được) là lỗi đầu vào của lệnh export: báo đúng
                # kiểu `submission error` thay vì để traceback sổ ra.
                raise SubmissionError(str(exc)) from exc
            execution_settings = load_execution_settings(args.execution_config)
            questions = load_raw_questions(args.questions_path)

            # Load row BM25 index and initialize row fusion if available
            row_fusion = _build_row_fusion(args, release)
            # Masked-PAL quyết định (spec 2026-08-24 §4.3): bắt buộc -- đây
            # là đường answering duy nhất còn lại.
            program_decisions = load_program_decisions(args.program_decisions)

            # Final review 2026-08-24: `ProgramDecision.cells` là chỉ số trong
            # danh sách ô ứng viên, phụ thuộc đúng các cài đặt retrieval dưới
            # đây. So với sidecar lúc sinh payload TRƯỚC khi chạy câu nào --
            # lệch thì chặn, nêu rõ trường, thay vì dịch im lặng mọi chỉ số.
            # `export` luôn fusion DEFAULT_ROW_CANDIDATE_COUNT dòng (hằng số
            # trong exporter), và reranker/dense-index đến từ _build_table_retriever.
            current_fingerprint = RetrievalFingerprint(
                k=args.k,
                rows_per_question=DEFAULT_ROW_CANDIDATE_COUNT,
                reranker_enabled=reranker is not None,
                dense_index=args.dense_index.name if args.dense_index is not None else None,
                release_lock=release.lock_path.name,
                release_lock_sha256=release.lock_sha256,
            )
            if args.assert_payload_fingerprint is not None:
                assert_fingerprint_matches(args.assert_payload_fingerprint, current_fingerprint)

            report, items, csv_rows = export_submission(
                questions,
                retriever,
                release.release_dir,
                execution_settings=execution_settings,
                dataset_fingerprint=release.dataset_fingerprint,
                k=args.k,
                reranker=reranker,
                row_fusion=row_fusion,
                program_decisions=program_decisions,
            )
            # Write the per-question coverage report BEFORE the compliance gate
            # (Important 5, 2026-08-21 final review): it writes only to
            # --report-dir, never to the ZIP, so writing it first cannot ship a
            # violating bundle. A failed run used to `return 2` before this
            # call ever ran, leaving nothing but compliance-violations.json on
            # disk -- exactly the per-question outcomes/stage/code data needed
            # to debug the violation (and the input the `validate` subcommand
            # requires) was silently discarded.
            json_path, markdown_path = write_export_report(report, args.report_dir)
            # Chốt chặn cứng (design §5.3): thể lệ ghi "Các câu hỏi vi phạm quy định
            # này sẽ không được tính điểm", và mục VIII liệt kê hardcode đáp án là căn
            # cứ loại đội thi. Không bao giờ ghi ra ZIP một bundle vi phạm.
            violations = check_bundle(
                items, csv_rows, timeout_seconds=execution_settings.timeout_seconds
            )
            if violations:
                args.report_dir.mkdir(parents=True, exist_ok=True)
                (args.report_dir / "compliance-violations.json").write_text(
                    json.dumps(
                        [
                            {"id": v.question_id, "code": v.code, "detail": v.detail}
                            for v in violations
                        ],
                        ensure_ascii=False,
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                affected = len({v.question_id for v in violations})
                print(
                    f"COMPLIANCE FAIL: {len(violations)} vi phạm trên {affected} câu. "
                    f"Chi tiết: {args.report_dir / 'compliance-violations.json'}"
                )
                return 2
            sha256 = write_submission_zip(items, csv_rows, args.output_zip)
            print(args.output_zip)
            print(f"sha256:{sha256}")
            print(json_path)
            print(markdown_path)
            print(f"answered {report.answered_count}/{report.question_count}")
            return 0
        if args.command == "row-batches":
            root = Path.cwd()
            release = resolve_retrieval_release(args.release_lock, repo_root=root)
            index = load_bm25_index(args.bm25_index)
            if index.manifest.dataset_fingerprint != release.dataset_fingerprint:
                raise SubmissionError(
                    "--bm25-index dataset_fingerprint does not match --release-lock"
                )
            # Cùng builder với `export`: mọi cấu hình retrieval đi qua đúng
            # một chỗ, nên hai lệnh không thể lệch nhau vì code khác nhau.
            service, batch_reranker = _build_table_retriever(args, release, index)
            questions = load_raw_questions(args.questions_path)
            row_fusion = _build_row_fusion(args, release)
            if row_fusion is None:
                raise SubmissionError(
                    f"không tìm thấy row index tại {args.bm25_index.parent}/"
                    f"{args.bm25_index.name}_row -- không thể sinh ứng viên dòng"
                )

            args.output_dir.mkdir(parents=True, exist_ok=True)
            # Final review 2026-08-24: ghi cạnh payload các cài đặt retrieval
            # đã dùng, để `export --assert-payload-fingerprint` chặn mọi lượt
            # chạy sinh lại danh sách ô ứng viên khác lúc batch (mọi chỉ số
            # ProgramDecision.cells sẽ dịch).
            write_retrieval_fingerprint(
                args.output_dir,
                RetrievalFingerprint(
                    k=args.k,
                    rows_per_question=args.rows_per_question,
                    reranker_enabled=batch_reranker is not None,
                    dense_index=(
                        args.dense_index.name if args.dense_index is not None else None
                    ),
                    release_lock=release.lock_path.name,
                    release_lock_sha256=release.lock_sha256,
                ),
            )
            written = 0
            for batch_number, start in enumerate(range(0, len(questions), args.batch_size)):
                chunk = questions[start : start + args.batch_size]
                lines: list[str] = []
                for raw_question in chunk:
                    retrieved = retrieve_candidate_table_ids(
                        raw_question.question, service, k=args.k, reranker=batch_reranker
                    )
                    fused = row_fusion.retrieve_rows(
                        raw_question.question,
                        candidate_table_ids=retrieved,
                        k=args.rows_per_question,
                    ).results
                    # Đường duy nhất (spec 2026-08-24 §4.3): payload ứng viên
                    # Ô qua cùng helper dựng danh sách đánh số với lúc export.
                    payload = build_program_batch_payload(
                        raw_question.id,
                        raw_question.question,
                        parse_query_entities(raw_question.question),
                        build_question_cell_candidates(
                            args.release_dir, raw_question.question, retrieved, fused
                        ),
                    )
                    lines.append(json.dumps(payload, ensure_ascii=False))
                    written += 1
                target = args.output_dir / f"batch_{batch_number:03d}.jsonl"
                target.write_text("\n".join(lines) + "\n", encoding="utf-8")
            print(f"đã ghi {written} câu vào {args.output_dir}")
            return 0
        if args.command == "validate":
            report_payload = json.loads(args.report_path.read_text(encoding="utf-8"))
            export_report = SubmissionExportReport.model_validate(report_payload)
            # Every question id, not just the answered ones: plan.md §2.4 rule
            # 1 requires the ZIP's id set to match the official question set
            # exactly, and the Day 23 backstop tier exists precisely to fill
            # the gap for questions no reasoning tier answered. Filtering to
            # `answered` here predates that tier and made `id_set_mismatch`
            # fire on every submission that uses it -- i.e. every real one.
            expected_ids = [outcome.id for outcome in export_report.outcomes]
            validation = validate_submission_zip(
                args.zip_path, expected_ids, tolerance=Decimal(args.tolerance)
            )
            for issue in validation.issues:
                print(f"{issue.code}: {issue.message}", file=sys.stderr)
            print(f"valid={validation.valid} items={validation.item_count}")
            return 0 if validation.valid else 1
        raise AssertionError("argparse accepted an unknown submission command")
    except (
        PlanningInputError,
        PlanningArtifactError,
        ValidationError,
        JSONDecodeError,
        OSError,
        ExecutionError,
        SubmissionError,
    ) as exc:
        print(f"submission error: {exc}", file=sys.stderr)
        return 2
