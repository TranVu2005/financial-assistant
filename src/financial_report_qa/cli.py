"""Product command-line dispatcher."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence

from financial_report_qa.data.download import main as download_main
from financial_report_qa.data.inventory import main as inventory_main

DownloadMain = Callable[[Sequence[str] | None], int]
InventoryMain = Callable[[Sequence[str] | None], int]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="financial-report-qa",
        description="Auditable Vietnamese financial-report question answering.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "download-data",
        add_help=False,
        help="Dry-run or download a Hugging Face dataset snapshot.",
    )
    commands.add_parser(
        "inventory-data",
        add_help=False,
        help="Inventory a ViFinQA TXT snapshot and write its manifest.",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    download_main_fn: DownloadMain = download_main,
    inventory_main_fn: InventoryMain = inventory_main,
) -> int:
    """Dispatch a product command while preserving the child command arguments."""
    raw_args = list(sys.argv[1:] if argv is None else argv)
    parsed, remaining = _parser().parse_known_args(raw_args)
    if parsed.command == "download-data":
        return download_main_fn(remaining)
    if parsed.command == "inventory-data":
        return inventory_main_fn(remaining)
    raise AssertionError("argparse accepted an unknown command")


if __name__ == "__main__":
    raise SystemExit(main())
