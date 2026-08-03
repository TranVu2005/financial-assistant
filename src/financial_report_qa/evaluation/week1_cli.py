"""Command line entrypoint for Week 1 Quality Gate evaluation."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from financial_report_qa.core.errors import Week1GateError, Week1GateInputError
from financial_report_qa.evaluation.week1_contracts import (
    CELL_AUDIT_COLUMNS,
    EXPECTED_TABLE_COLUMNS,
    PILOT_DOCUMENT_COLUMNS,
    ExpectedTable,
    PilotDocument,
    PilotMetadata,
    read_csv_rows,
    write_csv_rows,
)
from financial_report_qa.evaluation.week1_dataset import GateDataset, load_gate_dataset
from financial_report_qa.evaluation.week1_evaluator import (
    evaluate_week1_gate,
    publish_gate_artifacts,
)
from financial_report_qa.evaluation.week1_matching import assess_table_matching
from financial_report_qa.evaluation.week1_provenance import generate_cell_audits
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

    metadata_path = annotation_dir / "pilot-metadata.json"
    if not metadata_path.is_file():
        raise Week1GateInputError(f"Missing pilot-metadata.json in {annotation_dir}")

    meta = PilotMetadata.model_validate_json(metadata_path.read_bytes())
    if meta.dataset_fingerprint != dataset.dataset_fingerprint:
        raise Week1GateInputError("dataset fingerprint mismatch")
    if meta.source_manifest_sha256 != dataset.source_manifest_sha256:
        raise Week1GateInputError("source manifest fingerprint mismatch")

    exp_csv_path = annotation_dir / "expected-tables.csv"
    if not exp_csv_path.is_file():
        raise Week1GateInputError(f"Missing expected-tables.csv in {annotation_dir}")

    exp_rows = read_csv_rows(exp_csv_path, EXPECTED_TABLE_COLUMNS)
    if not exp_rows:
        raise Week1GateInputError(f"expected-tables.csv is empty in {annotation_dir}")

    expected_tables: list[ExpectedTable] = []
    for r in exp_rows:
        periods_tuple = tuple(p.strip() for p in r["expected_periods"].split(";") if p.strip())
        expected_tables.append(
            ExpectedTable(
                annotation_schema_version="1",
                annotation_id=r["annotation_id"],
                doc_id=r["doc_id"],
                relative_path=r["relative_path"],
                statement_type=r["statement_type"],  # type: ignore[arg-type]
                line_start=int(r["line_start"]),
                line_end=int(r["line_end"]),
                row_count=int(r["row_count"]),
                column_count=int(r["column_count"]),
                unit_normalized=r["unit_normalized"],
                expected_periods=periods_tuple,
                notes=r.get("notes", ""),
            )
        )

    docs_csv_path = annotation_dir / "pilot-documents.csv"
    doc_rows = read_csv_rows(docs_csv_path, PILOT_DOCUMENT_COLUMNS)
    pilot_docs = tuple(PilotDocument.model_validate(row) for row in doc_rows)

    pilot_doc_ids = {doc.doc_id for doc in pilot_docs}
    pilot_extracted_tables = tuple(
        tbl for tbl in dataset.tables_by_id.values() if tbl.doc_id in pilot_doc_ids
    )

    expected_tables_tuple = tuple(expected_tables)
    _, matched_tables = assess_table_matching(expected_tables_tuple, pilot_extracted_tables)
    all_cell_audits = generate_cell_audits(
        dataset, corpus_dir, expected_tables_tuple, matched_tables
    )

    sample_size = min(30, len(all_cell_audits)) if len(all_cell_audits) > 0 else 30
    sampled_cells = select_audit_cells(all_cell_audits, sample_size=sample_size)

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
