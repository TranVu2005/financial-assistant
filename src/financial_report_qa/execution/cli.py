"""Command-line interface for the Day 18 deterministic compiler."""

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
from financial_report_qa.execution.evaluation import (
    evaluate_compiled_plans_on_gold,
    write_compiled_plan_report,
)
from financial_report_qa.retrieval.gold import load_gold_questions
from financial_report_qa.retrieval.release import resolve_retrieval_release


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="financial-report-qa execution")
    commands = parser.add_subparsers(dest="command", required=True)

    compile_plans = commands.add_parser("compile-plans")
    compile_plans.add_argument("--release-lock", type=Path, required=True)
    compile_plans.add_argument("--gold-path", type=Path, required=True)
    compile_plans.add_argument("--output-dir", type=Path, required=True)
    compile_plans.add_argument(
        "--execution-config",
        type=Path,
        nargs="+",
        default=None,
        help="One or more YAML files layering the `execution:` block (e.g. "
        "configs/base.yaml configs/local_rtx3050.yaml).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        root = Path.cwd()
        release = resolve_retrieval_release(args.release_lock, repo_root=root)
        if args.command == "compile-plans":
            if args.execution_config is None:
                raise PlanningInputError(
                    "--execution-config is required (identifies the operation whitelist)"
                )
            execution_settings = load_execution_settings(args.execution_config)
            gold = load_gold_questions(args.gold_path, release)
            report = evaluate_compiled_plans_on_gold(
                gold, release.release_dir, execution_settings=execution_settings
            )
            json_path, markdown_path = write_compiled_plan_report(report, args.output_dir)
            print(json_path)
            print(markdown_path)
            return 0
        raise AssertionError("argparse accepted an unknown execution command")
    except (
        PlanningInputError,
        PlanningArtifactError,
        ValidationError,
        JSONDecodeError,
        OSError,
        ExecutionError,
    ) as exc:
        print(f"execution error: {exc}", file=sys.stderr)
        return 2
