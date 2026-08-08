"""Command line entrypoint for Week 1 Quality Gate evaluation."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from financial_report_qa.core.errors import Week1GateError, Week1GateInputError
from financial_report_qa.evaluation.week1_annotations import (
    finalize_review_worksheet,
    generate_review_worksheet,
    load_annotation_bundle,
)
from financial_report_qa.evaluation.week1_contracts import (
    CELL_AUDIT_COLUMNS,
    write_csv_rows,
)
from financial_report_qa.evaluation.week1_dataset import GateDataset, load_gate_dataset
from financial_report_qa.evaluation.week1_evaluator import (
    evaluate_week1_gate,
    publish_gate_artifacts,
)
from financial_report_qa.evaluation.week1_matching import assess_table_matching
from financial_report_qa.evaluation.week1_provenance import generate_cell_audits
from financial_report_qa.evaluation.week1_release import publish_release_lock
from financial_report_qa.evaluation.week1_sampling import prepare_pilot, select_audit_cells


def sample_cells_workflow(
    dataset: GateDataset,
    corpus_dir: Path,
    annotation_dir: Path,
) -> None:
    """Run automated provenance and sample 30 cells for manual audit."""
    cell_audit_csv_path = annotation_dir / "cell-audit.csv"
    if cell_audit_csv_path.is_file():
        raise Week1GateInputError(f"cell-audit.csv already exists at {cell_audit_csv_path}")

    bundle = load_annotation_bundle(dataset, annotation_dir, require_expected_tables=True)

    pilot_doc_ids = {doc.doc_id for doc in bundle.pilot_documents}
    pilot_extracted_tables = tuple(
        tbl for tbl in dataset.tables_by_id.values() if tbl.doc_id in pilot_doc_ids
    )

    _, matched_tables = assess_table_matching(bundle.expected_tables, pilot_extracted_tables)
    all_cell_audits = generate_cell_audits(
        dataset, corpus_dir, bundle.expected_tables, matched_tables
    )

    sampled_cells = select_audit_cells(all_cell_audits, sample_size=30)

    audit_rows = [ca.model_dump(mode="json") for ca in sampled_cells]
    for r in audit_rows:
        if r.get("verified") is None:
            r["verified"] = ""
    write_csv_rows(cell_audit_csv_path, CELL_AUDIT_COLUMNS, audit_rows)


def build_parser() -> argparse.ArgumentParser:
    """Build the week1-gate CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="financial-report-qa week1-gate",
        description=(
            "Prepare pilot annotations, sample audit cells, or evaluate Week 1 Quality Gate."
        ),
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

    # prepare-review subcommand
    prep_review_parser = subparsers.add_parser(
        "prepare-review",
        help="Generate advisory table review worksheet.",
    )
    prep_review_parser.add_argument(
        "--manifest-path",
        type=Path,
        required=True,
        help="Path to source manifest documents.jsonl",
    )
    prep_review_parser.add_argument(
        "--release-path",
        type=Path,
        required=True,
        help="Path to normalized dataset release directory",
    )
    prep_review_parser.add_argument(
        "--corpus-dir",
        type=Path,
        required=True,
        help="Path to raw source text report corpus directory",
    )
    prep_review_parser.add_argument(
        "--annotation-dir",
        type=Path,
        required=True,
        help="Path to pilot annotation directory",
    )
    prep_review_parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Output CSV path for the review worksheet",
    )

    # finalize-tables subcommand
    finalize_parser = subparsers.add_parser(
        "finalize-tables",
        help="Finalize expected-tables.csv from completed table-review.csv.",
    )
    finalize_parser.add_argument(
        "--manifest-path",
        type=Path,
        required=True,
        help="Path to source manifest documents.jsonl",
    )
    finalize_parser.add_argument(
        "--release-path",
        type=Path,
        required=True,
        help="Path to normalized dataset release directory",
    )
    finalize_parser.add_argument(
        "--annotation-dir",
        type=Path,
        required=True,
        help="Path to pilot annotation directory",
    )
    finalize_parser.add_argument(
        "--review-path",
        type=Path,
        required=True,
        help="Path to completed table-review.csv review worksheet",
    )

    # sample-cells subcommand
    sample_parser = subparsers.add_parser(
        "sample-cells",
        help="Extract deterministic 30-cell audit template from populated expected-tables.csv.",
    )
    sample_parser.add_argument(
        "--manifest-path",
        type=Path,
        required=True,
        help="Path to source manifest documents.jsonl",
    )
    sample_parser.add_argument(
        "--release-path",
        type=Path,
        required=True,
        help="Path to normalized dataset release directory",
    )
    sample_parser.add_argument(
        "--corpus-dir",
        type=Path,
        required=True,
        help="Path to raw source text report corpus directory",
    )
    sample_parser.add_argument(
        "--annotation-dir",
        type=Path,
        required=True,
        help="Path to populated pilot annotation directory",
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
        "--report-root",
        dest="output_dir",
        type=Path,
        required=True,
        help="Target directory to publish gate evaluation reports",
    )

    # lock-release subcommand
    lock_parser = subparsers.add_parser(
        "lock-release",
        help="Publish an immutable dataset-pilot-v1.json release lock from a passing gate result.",
    )
    lock_parser.add_argument(
        "--release-path",
        type=Path,
        required=True,
        help="Path to normalized dataset release directory",
    )
    lock_parser.add_argument(
        "--gate-result-path",
        type=Path,
        required=True,
        help="Path to canonical gate-result.json from a passing evaluate run",
    )
    lock_parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Output path for dataset-pilot-v1.json release lock",
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
        if args.subcommand == "lock-release":
            # lock-release does not need the full GateDataset
            publish_release_lock(
                args.release_path,
                args.gate_result_path,
                args.output_path,
            )
            print(f"Successfully published release lock to {args.output_path}")
            return 0

        dataset = load_gate_dataset(args.manifest_path, args.release_path)

        if args.subcommand == "prepare":
            prepare_pilot(dataset, args.annotation_root)
            print(f"Successfully prepared pilot annotation workspace at {args.annotation_root}")
            return 0

        if args.subcommand == "prepare-review":
            generate_review_worksheet(
                dataset, args.corpus_dir, args.annotation_dir, args.output_path
            )
            print(f"Successfully generated review worksheet at {args.output_path}")
            return 0

        if args.subcommand == "finalize-tables":
            finalize_review_worksheet(dataset, args.annotation_dir, args.review_path)
            print(f"Successfully finalized expected tables in {args.annotation_dir}")
            return 0

        if args.subcommand == "sample-cells":
            sample_cells_workflow(dataset, args.corpus_dir, args.annotation_dir)
            print(f"Successfully generated cell audit sample in {args.annotation_dir}")
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
