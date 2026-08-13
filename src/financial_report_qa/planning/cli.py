"""Command-line interface for the Day 10 deterministic entity parser."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from json import JSONDecodeError
from pathlib import Path

from pydantic import ValidationError

from financial_report_qa.core.errors import PlanningArtifactError, PlanningInputError
from financial_report_qa.planning.entity_cases import (
    entity_case_set_sha256,
    generate_entity_cases,
    load_entity_cases,
    write_entity_cases,
)
from financial_report_qa.planning.entity_evaluation import (
    evaluate_entity_cases,
    evaluate_entity_parser_on_gold,
    write_entity_case_report,
    write_held_out_report,
)
from financial_report_qa.retrieval.gold import load_gold_questions
from financial_report_qa.retrieval.release import resolve_retrieval_release


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="financial-report-qa planning")
    commands = parser.add_subparsers(dest="command", required=True)

    generate_cases = commands.add_parser("generate-entity-cases")
    generate_cases.add_argument("--release-lock", type=Path, required=True)
    generate_cases.add_argument("--output-path", type=Path, required=True)

    evaluate_cases = commands.add_parser("evaluate-entities")
    evaluate_cases.add_argument("--release-lock", type=Path, required=True)
    evaluate_cases.add_argument("--case-path", type=Path, required=True)
    evaluate_cases.add_argument("--output-dir", type=Path, required=True)
    evaluate_cases.add_argument(
        "--gold-path",
        type=Path,
        default=None,
        help="Optional retrieval-gold-v1 path for a one-shot held-out report.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        root = Path.cwd()
        release = resolve_retrieval_release(args.release_lock, repo_root=root)
        if args.command == "generate-entity-cases":
            cases = generate_entity_cases(release)
            write_entity_cases(cases, args.output_path)
            print(entity_case_set_sha256(cases))
            print(args.output_path)
            return 0
        if args.command == "evaluate-entities":
            cases = load_entity_cases(args.case_path)
            case_set_sha256 = entity_case_set_sha256(cases)
            case_report = evaluate_entity_cases(cases, case_set_sha256=case_set_sha256)
            json_path, markdown_path = write_entity_case_report(case_report, args.output_dir)
            print(json_path)
            print(markdown_path)
            if args.gold_path is not None:
                gold = load_gold_questions(args.gold_path, release)
                held_out_report = evaluate_entity_parser_on_gold(gold)
                held_out_json, held_out_markdown = write_held_out_report(
                    held_out_report, args.output_dir
                )
                print(held_out_json)
                print(held_out_markdown)
            return 0
        raise AssertionError("argparse accepted an unknown planning command")
    except (
        PlanningInputError,
        PlanningArtifactError,
        ValidationError,
        JSONDecodeError,
        ValueError,
        OSError,
    ) as exc:
        print(f"planning error: {exc}", file=sys.stderr)
        return 2
