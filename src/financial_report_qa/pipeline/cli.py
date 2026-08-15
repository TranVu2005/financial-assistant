"""Command-line interface for the Day 21 E2E pipeline (retrieval in the loop)."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from json import JSONDecodeError
from pathlib import Path

from pydantic import ValidationError

from financial_report_qa.core.config import load_execution_settings
from financial_report_qa.core.errors import (
    ExecutionError,
    PlanningArtifactError,
    PlanningInputError,
)
from financial_report_qa.pipeline.evaluation import (
    evaluate_scope_policies,
    load_rankings,
    run_e2e_pipeline,
    write_pipeline_report,
    write_scope_policy_report,
)
from financial_report_qa.retrieval.gold import load_gold_questions
from financial_report_qa.retrieval.release import resolve_retrieval_release
from financial_report_qa.verification.evaluation import load_answer_gold


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="financial-report-qa pipeline")
    commands = parser.add_subparsers(dest="command", required=True)

    run_e2e = commands.add_parser("run-e2e")
    run_e2e.add_argument("--release-lock", type=Path, required=True)
    run_e2e.add_argument("--gold-path", type=Path, required=True)
    run_e2e.add_argument(
        "--rankings-path",
        type=Path,
        required=True,
        help="A saved `retrieval-v2-*.json` report (e.g. "
        "artifacts/evaluations/day14/v2/retrieval-v2-bm25-v4-<fp>.json) -- "
        "no retrieval index is rebuilt here; this is the actual candidate-"
        "table source per question, not gold_table_ids (Day 21 plan §1.1).",
    )
    run_e2e.add_argument("--output-dir", type=Path, required=True)
    run_e2e.add_argument(
        "--execution-config",
        type=Path,
        nargs="+",
        default=None,
        help="One or more YAML files layering the `execution:` block (e.g. "
        "configs/base.yaml configs/local_rtx3050.yaml).",
    )
    run_e2e.add_argument(
        "--answer-gold-path",
        type=Path,
        default=None,
        help="Optional data/qa/answer-gold-v1.jsonl for accuracy scoring.",
    )

    scope_policy_report = commands.add_parser("scope-policy-report")
    scope_policy_report.add_argument("--release-lock", type=Path, required=True)
    scope_policy_report.add_argument("--gold-path", type=Path, required=True)
    scope_policy_report.add_argument("--rankings-path", type=Path, required=True)
    scope_policy_report.add_argument("--output-dir", type=Path, required=True)
    scope_policy_report.add_argument(
        "--execution-config",
        type=Path,
        nargs="+",
        default=None,
        help="One or more YAML files layering the `execution:` block. Its "
        "`default_statement_scope` (if any) is overridden per policy.",
    )
    scope_policy_report.add_argument("--answer-gold-path", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        root = Path.cwd()
        release = resolve_retrieval_release(args.release_lock, repo_root=root)
        if args.command == "run-e2e":
            if args.execution_config is None:
                raise PlanningInputError(
                    "--execution-config is required (identifies the operation whitelist)"
                )
            execution_settings = load_execution_settings(args.execution_config)
            gold = load_gold_questions(args.gold_path, release)
            rankings, rankings_sha256 = load_rankings(args.rankings_path)
            answer_gold = (
                load_answer_gold(args.answer_gold_path)
                if args.answer_gold_path is not None
                else None
            )
            report = run_e2e_pipeline(
                gold,
                rankings,
                release.release_dir,
                execution_settings=execution_settings,
                rankings_source=str(args.rankings_path),
                rankings_sha256=rankings_sha256,
                answer_gold=answer_gold,
            )
            json_path, markdown_path = write_pipeline_report(report, args.output_dir)
            print(json_path)
            print(markdown_path)
            return 0
        if args.command == "scope-policy-report":
            if args.execution_config is None:
                raise PlanningInputError(
                    "--execution-config is required (identifies the operation whitelist)"
                )
            execution_settings = load_execution_settings(args.execution_config)
            gold = load_gold_questions(args.gold_path, release)
            rankings, rankings_sha256 = load_rankings(args.rankings_path)
            answer_gold = (
                load_answer_gold(args.answer_gold_path)
                if args.answer_gold_path is not None
                else None
            )
            policy_report = evaluate_scope_policies(
                gold,
                rankings,
                release.release_dir,
                base_execution_settings=execution_settings,
                rankings_source=str(args.rankings_path),
                rankings_sha256=rankings_sha256,
                answer_gold=answer_gold,
            )
            json_path, markdown_path = write_scope_policy_report(policy_report, args.output_dir)
            print(json_path)
            print(markdown_path)
            return 0
        raise AssertionError("argparse accepted an unknown pipeline command")
    except (
        PlanningInputError,
        PlanningArtifactError,
        ValidationError,
        JSONDecodeError,
        OSError,
        ExecutionError,
    ) as exc:
        print(f"pipeline error: {exc}", file=sys.stderr)
        return 2
