"""Command-line interface for the Day 10 deterministic entity parser."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from json import JSONDecodeError
from pathlib import Path

from pydantic import ValidationError

from financial_report_qa.core.config import load_llm_settings
from financial_report_qa.core.errors import LLMError, PlanningArtifactError, PlanningInputError
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
from financial_report_qa.planning.llm_client import LLMClient
from financial_report_qa.planning.llm_evaluation import (
    ReplayCacheClient,
    evaluate_llm_plan_cases,
    evaluate_router_abstain_cases,
    load_replay_cache,
    write_llm_plan_case_report,
    write_router_abstain_report,
)
from financial_report_qa.planning.plan_cases import load_plan_cases, plan_case_set_sha256
from financial_report_qa.planning.plan_evaluation import (
    evaluate_plan_cases,
    evaluate_rule_planner_on_gold,
    write_held_out_plan_report,
    write_plan_case_report,
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

    evaluate_plans = commands.add_parser("evaluate-plans")
    evaluate_plans.add_argument("--release-lock", type=Path, required=True)
    evaluate_plans.add_argument("--case-path", type=Path, required=True)
    evaluate_plans.add_argument("--output-dir", type=Path, required=True)
    evaluate_plans.add_argument(
        "--gold-path",
        type=Path,
        default=None,
        help="Optional retrieval-gold-v1 path for a one-shot descriptive held-out report.",
    )

    evaluate_llm_plans = commands.add_parser("evaluate-llm-plans")
    evaluate_llm_plans.add_argument("--release-lock", type=Path, required=True)
    evaluate_llm_plans.add_argument("--case-path", type=Path, required=True)
    evaluate_llm_plans.add_argument("--output-dir", type=Path, required=True)
    evaluate_llm_plans.add_argument(
        "--cache-path",
        type=Path,
        required=True,
        help="Replay-cache JSONL (Day 17 plan §1.8): read on every run; appended to only "
        "with --live.",
    )
    evaluate_llm_plans.add_argument(
        "--llm-config",
        type=Path,
        nargs="+",
        default=None,
        help="One or more YAML files layering the `llm:` block (e.g. configs/base.yaml "
        "configs/local_rtx3050.yaml) — identifies which model's cache to use. Always "
        "required, whether or not --live is set.",
    )
    evaluate_llm_plans.add_argument(
        "--live",
        action="store_true",
        help="Fall through to a real LLM server on a cache miss and record new responses "
        "to --cache-path. Without this flag, a cache miss abstains (llm_unavailable) "
        "instead of ever reaching a network.",
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
        if args.command == "evaluate-plans":
            plan_cases = load_plan_cases(args.case_path)
            plan_case_sha256 = plan_case_set_sha256(plan_cases)
            plan_case_report = evaluate_plan_cases(plan_cases, case_set_sha256=plan_case_sha256)
            plan_json_path, plan_markdown_path = write_plan_case_report(
                plan_case_report, args.output_dir
            )
            print(plan_json_path)
            print(plan_markdown_path)
            if args.gold_path is not None:
                gold = load_gold_questions(args.gold_path, release)
                plan_held_out_report = evaluate_rule_planner_on_gold(gold)
                plan_held_out_json, plan_held_out_markdown = write_held_out_plan_report(
                    plan_held_out_report, args.output_dir
                )
                print(plan_held_out_json)
                print(plan_held_out_markdown)
            return 0
        if args.command == "evaluate-llm-plans":
            if args.llm_config is None:
                raise PlanningInputError(
                    "--llm-config is required (identifies the model whose cache to use), "
                    "whether or not --live is set"
                )
            llm_settings = load_llm_settings(args.llm_config)
            model_identity = f"{llm_settings.model}@t{llm_settings.temperature}"
            underlying = LLMClient(llm_settings) if args.live else None
            client = ReplayCacheClient(
                cache=load_replay_cache(args.cache_path),
                model_identity=model_identity,
                underlying=underlying,
                record_path=args.cache_path,
            )
            llm_plan_cases = load_plan_cases(args.case_path)
            llm_case_sha256 = plan_case_set_sha256(llm_plan_cases)

            llm_report = evaluate_llm_plan_cases(
                llm_plan_cases, client=client, case_set_sha256=llm_case_sha256
            )
            llm_json_path, llm_markdown_path = write_llm_plan_case_report(
                llm_report, args.output_dir
            )
            print(llm_json_path)
            print(llm_markdown_path)

            router_report = evaluate_router_abstain_cases(
                llm_plan_cases, client=client, case_set_sha256=llm_case_sha256
            )
            router_json_path, router_markdown_path = write_router_abstain_report(
                router_report, args.output_dir
            )
            print(router_json_path)
            print(router_markdown_path)
            return 0
        raise AssertionError("argparse accepted an unknown planning command")
    except (
        PlanningInputError,
        PlanningArtifactError,
        ValidationError,
        JSONDecodeError,
        ValueError,
        OSError,
        LLMError,
    ) as exc:
        print(f"planning error: {exc}", file=sys.stderr)
        return 2
