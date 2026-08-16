"""Command-line interface for the Day 22 submission bundle (export/validate)."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from decimal import Decimal
from json import JSONDecodeError
from pathlib import Path

from pydantic import ValidationError

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
                    allow_inferred_scope=args.allow_inferred_scope,
                )
            sha256 = write_submission_zip(items, csv_rows, args.output_zip)
            json_path, markdown_path = write_export_report(report, args.report_dir)
            print(args.output_zip)
            print(f"sha256:{sha256}")
            print(json_path)
            print(markdown_path)
            print(f"answered {report.answered_count}/{report.question_count}")
            return 0
        if args.command == "validate":
            report_payload = json.loads(args.report_path.read_text(encoding="utf-8"))
            export_report = SubmissionExportReport.model_validate(report_payload)
            expected_ids = [
                outcome.id for outcome in export_report.outcomes if outcome.status == "answered"
            ]
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
