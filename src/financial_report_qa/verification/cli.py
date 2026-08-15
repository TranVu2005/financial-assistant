"""Command-line interface for the Day 20 answer verifier."""

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
from financial_report_qa.retrieval.gold import load_gold_questions
from financial_report_qa.retrieval.release import resolve_retrieval_release
from financial_report_qa.verification.evaluation import (
    evaluate_answer_packages_on_gold,
    load_answer_gold,
    write_answer_verification_report,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="financial-report-qa verification")
    commands = parser.add_subparsers(dest="command", required=True)

    verify_answers = commands.add_parser("verify-answers")
    verify_answers.add_argument("--release-lock", type=Path, required=True)
    verify_answers.add_argument("--gold-path", type=Path, required=True)
    verify_answers.add_argument("--output-dir", type=Path, required=True)
    verify_answers.add_argument(
        "--execution-config",
        type=Path,
        nargs="+",
        default=None,
        help="One or more YAML files layering the `execution:` block (e.g. "
        "configs/base.yaml configs/local_rtx3050.yaml).",
    )
    verify_answers.add_argument(
        "--answer-gold-path",
        type=Path,
        default=None,
        help="Optional data/qa/answer-gold-v1.jsonl for accuracy scoring (task 20.9).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        root = Path.cwd()
        release = resolve_retrieval_release(args.release_lock, repo_root=root)
        if args.command == "verify-answers":
            if args.execution_config is None:
                raise PlanningInputError(
                    "--execution-config is required (identifies the operation whitelist)"
                )
            execution_settings = load_execution_settings(args.execution_config)
            gold = load_gold_questions(args.gold_path, release)
            answer_gold = (
                load_answer_gold(args.answer_gold_path)
                if args.answer_gold_path is not None
                else None
            )
            report = evaluate_answer_packages_on_gold(
                gold,
                release.release_dir,
                execution_settings=execution_settings,
                answer_gold=answer_gold,
            )
            json_path, markdown_path = write_answer_verification_report(report, args.output_dir)
            print(json_path)
            print(markdown_path)
            return 0
        raise AssertionError("argparse accepted an unknown verification command")
    except (
        PlanningInputError,
        PlanningArtifactError,
        ValidationError,
        JSONDecodeError,
        OSError,
        ExecutionError,
    ) as exc:
        print(f"verification error: {exc}", file=sys.stderr)
        return 2
