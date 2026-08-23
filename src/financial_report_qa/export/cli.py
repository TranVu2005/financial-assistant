"""Command-line interface for exporting normalized CSVs and synced text."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from financial_report_qa.core.errors import FinancialReportQAError
from financial_report_qa.export.csv_export import export_normalized_csvs
from financial_report_qa.export.synced_text import export_synced_text


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="financial-report-qa export-tables")
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--csv-output-dir", type=Path, required=True)
    parser.add_argument(
        "--text-output-dir",
        type=Path,
        default=Path("data/interim/synced_text"),
        help="Mirror directory for the rewritten synced-text documents.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Export normalized CSVs, then rewrite synced text, from one release."""
    args = _parser().parse_args(argv)
    try:
        csv_manifest = export_normalized_csvs(args.release_dir, args.csv_output_dir)
        text_manifest = export_synced_text(
            args.release_dir,
            args.snapshot_root,
            csv_manifest,
            args.text_output_dir,
        )
    except FinancialReportQAError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        f"exported {csv_manifest.table_count} tables; "
        f"synced {text_manifest.document_count} documents"
    )
    return 0
