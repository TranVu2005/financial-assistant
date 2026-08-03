"""Command line entrypoint for Week 1 Quality Gate evaluation."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from financial_report_qa.core.errors import Week1GateError, Week1GateInputError
from financial_report_qa.evaluation.week1_dataset import load_gate_dataset
from financial_report_qa.evaluation.week1_evaluator import (
    evaluate_week1_gate,
    publish_gate_artifacts,
)
from financial_report_qa.evaluation.week1_sampling import prepare_pilot


def build_parser() -> argparse.ArgumentParser:
    """Build the week1-gate CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="financial-report-qa week1-gate",
        description="Prepare pilot annotations or evaluate Week 1 Quality Gate.",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # prepare subcommand
    prep_parser = subparsers.add_parser(
        "prepare",
        help="Select 20x3 pilot documents and prepare annotation directory.",
    )
    prep_parser.add_argument(
        "--manifest-path",
        type=Path,
        required=True,
        help="Path to source manifest documents.jsonl",
    )
    prep_parser.add_argument(
        "--release-path",
        type=Path,
        required=True,
        help="Path to normalized dataset release directory",
    )
    prep_parser.add_argument(
        "--annotation-root",
        type=Path,
        required=True,
        help="Target directory to initialize pilot annotation templates",
    )

    # evaluate subcommand
    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate Week 1 Quality Gate against pilot annotations.",
    )
    eval_parser.add_argument(
        "--manifest-path",
        type=Path,
        required=True,
        help="Path to source manifest documents.jsonl",
    )
    eval_parser.add_argument(
        "--release-path",
        type=Path,
        required=True,
        help="Path to normalized dataset release directory",
    )
    eval_parser.add_argument(
        "--corpus-dir",
        type=Path,
        required=True,
        help="Path to raw source text report corpus directory",
    )
    eval_parser.add_argument(
        "--annotation-dir",
        type=Path,
        required=True,
        help="Path to populated pilot annotation directory",
    )
    eval_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Target directory to publish gate evaluation reports",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Main CLI entrypoint for week1-gate."""
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 2

    try:
        dataset = load_gate_dataset(args.manifest_path, args.release_path)

        if args.subcommand == "prepare":
            prepare_pilot(dataset, args.annotation_root)
            print(f"Successfully prepared pilot annotation workspace at {args.annotation_root}")
            return 0

        if args.subcommand == "evaluate":
            result, assessments, cell_audits = evaluate_week1_gate(
                dataset, args.corpus_dir, args.annotation_dir
            )
            publish_gate_artifacts(result, cell_audits, args.output_dir)
            if result.passed:
                print(f"Week 1 Quality Gate PASSED. Artifacts published to {args.output_dir}")
                return 0
            else:
                print(f"Week 1 Quality Gate FAILED. Artifacts published to {args.output_dir}")
                return 1

    except Week1GateInputError as exc:
        print(f"Input Error: {exc}", file=sys.stderr)
        return 2
    except Week1GateError as exc:
        print(f"Gate Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Unexpected Error: {exc}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
