from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from financial_report_qa.core.errors import FinancialReportQAError
from financial_report_qa.data.inventory import build_inventory
from financial_report_qa.ingestion import extract_document


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke-test ViFinQA TXT ingestion.")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--repeat-sample", type=int, default=10)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.repeat_sample < 0:
        print("error: --repeat-sample must be non-negative", file=sys.stderr)
        return 2
    try:
        inventory = build_inventory(
            args.root,
            repo_id=args.repo_id,
            revision=args.revision,
        )
        ready = tuple(
            item for item in inventory.documents if item.inventory_status == "ready"
        )
        table_count = 0
        cell_count = 0
        placement_count = 0
        rejection_count = 0
        html_free_count = 0
        html_free_with_tables = 0
        for index, document in enumerate(ready):
            result = extract_document(args.root, document)
            if index < args.repeat_sample:
                repeated = extract_document(args.root, document)
                if repeated != result:
                    raise ValueError(
                        f"non-deterministic extraction: {document.relative_path}"
                    )
            table_count += len(result.tables)
            cell_count += sum(len(item.cells) for item in result.tables)
            placement_count += sum(len(item.placements) for item in result.tables)
            rejection_count += len(result.rejected)
            html_free = all(block.kind != "table" for block in result.blocks)
            if html_free:
                html_free_count += 1
                if result.tables:
                    html_free_with_tables += 1
    except (FinancialReportQAError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    print(f"Discovered:            {len(inventory.documents) + len(inventory.issues)}")
    print(f"Ready:                 {len(ready)}")
    print(f"Tables:                {table_count}")
    print(f"Cells:                 {cell_count}")
    print(f"Placements:            {placement_count}")
    print(f"Rejections:            {rejection_count}")
    print(f"HTML-free documents:   {html_free_count}")
    print(f"HTML-free with tables: {html_free_with_tables}")
    print(f"Repeated sample:       {min(len(ready), args.repeat_sample)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
