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

from financial_report_qa.core.config import load_execution_settings, load_llm_settings
from financial_report_qa.core.errors import (
    ExecutionError,
    PlanningArtifactError,
    PlanningInputError,
    SubmissionError,
)
from financial_report_qa.planning.llm_client import LLMClient
from financial_report_qa.retrieval.index import load_bm25_index
from financial_report_qa.retrieval.release import resolve_retrieval_release
from financial_report_qa.retrieval.service import RetrievalService
from financial_report_qa.submission.compliance import check_bundle
from financial_report_qa.submission.contracts import SubmissionExportReport
from financial_report_qa.submission.exporter import (
    export_submission,
    load_raw_questions,
    write_export_report,
    write_submission_zip,
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
    export.add_argument(
        "--llm-config",
        type=Path,
        nargs="+",
        default=None,
        help="One or more YAML files layering the `llm:` block (e.g. configs/base.yaml "
        "configs/local_rtx3050.yaml). When given, a rule-planner abstain falls back to "
        "the LLM planner (plan_router.route_plan, ADR 0006 A1) against this endpoint -- "
        "the rule planner still always runs first and is never overridden once it "
        "succeeds. Omit to keep the rule-planner-only behavior.",
    )
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
        "--row-dense-corpus/--row-dense-index/--dense-encoder are all given). plan.md §20's "
        "row-recall benchmark (58 gold questions, dense weight 0.5) measured dense making "
        "Row Recall@3/@5 *worse* than bm25+fuzzy+alias alone (74.1%%->67.2%%, 82.8%%->74.1%%; "
        "@1/@10 unchanged) -- re-measure with `retrieval.row_recall_evaluation` before "
        "raising this above 0.",
    )
    export.add_argument(
        "--allow-inferred-scope",
        action="store_true",
        help=(
            "Ship answers whose statement_scope came from "
            "`execution.default_statement_scope` rather than the question itself "
            "(ADR 0010 B1 normally blocks these). The organizers score "
            "correct/TOTAL questions, so such an answer costs exactly what an "
            "abstention costs while retaining a chance of being right. The "
            "`scope_inferred` issue is still recorded on every affected "
            "AnswerPackage. Off by default: for internal quality measurement a "
            "scope-guessed answer really is untrustworthy."
        ),
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
    `None` and gets the real pinned `SentenceTransformerDenseEncoder`."""
    if args.row_dense_corpus is None or args.row_dense_index is None or args.dense_encoder is None:
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
            args.row_dense_corpus, release_lock_sha256=release.lock_sha256
        )
        if row_corpus.manifest.dataset_fingerprint != release.dataset_fingerprint:
            print(
                "Warning: --row-dense-corpus dataset_fingerprint does not match "
                "--release-lock; skipping row dense retrieval",
                file=sys.stderr,
            )
            return None
        if encoder is None:
            spec = approved_encoder_spec(args.dense_encoder)
            encoder = SentenceTransformerDenseEncoder(
                spec, local_files_only=args.dense_local_files_only
            )
        row_dense_index = load_row_dense_index(
            args.row_dense_index,
            row_corpus,
            expected_encoder_spec_sha256=encoder_spec_sha256(encoder.spec),
            release_lock_sha256=release.lock_sha256,
        )
        cache_dir = args.dense_cache_dir or (args.row_dense_index.parent / "query-cache")
        cache = QueryEmbeddingCache(cache_dir, encoder.spec)
        return RowDenseRetrievalService(row_dense_index, encoder, cache)
    except Exception as e:
        print(f"Warning: Failed to load row dense index: {e}", file=sys.stderr)
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
            service = RetrievalService(index)
            execution_settings = load_execution_settings(args.execution_config)
            questions = load_raw_questions(args.questions_path)

            # Load row BM25 index and initialize row fusion if available
            row_index_dir = args.bm25_index.parent / f"{args.bm25_index.name}_row"
            row_fusion = None
            if row_index_dir.is_dir():
                from financial_report_qa.retrieval.row_fusion import RowFusionService
                from financial_report_qa.retrieval.row_fusion_contracts import RowFusionWeights
                from financial_report_qa.retrieval.row_index import load_row_bm25_index
                from financial_report_qa.retrieval.row_lexical import (
                    RowAliasRetrievalService,
                    RowFuzzyRetrievalService,
                )
                from financial_report_qa.retrieval.row_service import RowRetrievalService

                try:
                    row_index = load_row_bm25_index(
                        row_index_dir, release_lock_sha256=release.lock_sha256
                    )
                    row_service = RowRetrievalService(row_index)
                    # Dense (plan.md §7) is opt-in via --row-dense-corpus/
                    # --row-dense-index/--dense-encoder; loading it still
                    # defaults to weight 0.0 (--dense-weight) -- plan.md §20's
                    # benchmark measured 0.5 making Row Recall@3/@5 worse,
                    # not better, than bm25+fuzzy+alias alone.
                    row_dense_service = _load_row_dense_service(args, release)
                    row_fusion = RowFusionService(
                        bm25=row_service,
                        dense=row_dense_service,
                        weights=RowFusionWeights(
                            bm25=1.0, dense=args.dense_weight, fuzzy=0.3, alias=0.2
                        ),
                        fuzzy=RowFuzzyRetrievalService(row_index),
                        alias=RowAliasRetrievalService(row_index),
                    )
                except Exception as e:
                    print(f"Warning: Failed to load row BM25 index: {e}", file=sys.stderr)

            if args.llm_config is not None:
                llm_settings = load_llm_settings(args.llm_config)
                with LLMClient(llm_settings) as llm_client:
                    report, items, csv_rows = export_submission(
                        questions,
                        service,
                        release.release_dir,
                        execution_settings=execution_settings,
                        dataset_fingerprint=release.dataset_fingerprint,
                        k=args.k,
                        llm_client=llm_client,
                        row_fusion=row_fusion,
                        allow_inferred_scope=args.allow_inferred_scope,
                    )
            else:
                report, items, csv_rows = export_submission(
                    questions,
                    service,
                    release.release_dir,
                    execution_settings=execution_settings,
                    dataset_fingerprint=release.dataset_fingerprint,
                    k=args.k,
                    row_fusion=row_fusion,
                    allow_inferred_scope=args.allow_inferred_scope,
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
