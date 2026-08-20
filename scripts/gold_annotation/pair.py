"""Targeted two-period lookups for difference/growth questions.

Reads a JSON list on stdin, each item {"id", "code", "years", "needle",
"scope"?, "limit"?}, and prints the matching rows per year so the annotator
can read both endpoints of the comparison in one pass.

Imports nothing from `financial_report_qa` (ADR 0009 A2).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from review import normalize_row
from srcread import fold, read_rows, statement_dirs


def show(code: str, year: int, needle: str, scope: str | None, limit: int) -> None:
    print(f"  -- {code} {year} [{scope or 'any'}] ~ {needle}")
    found = 0
    for directory in statement_dirs(code, year):
        if scope and scope not in directory.name:
            continue
        for path in directory.glob("*_extracted.txt"):
            for row in read_rows(path):
                label, values, headers = normalize_row(row)
                if not values or fold(needle) not in fold(label):
                    continue
                if found >= limit:
                    return
                found += 1
                tag = directory.name.split("_")[-1][:4]
                print(f"     {tag}:{row.line} {label[:52]!r}")
                print(
                    "        hdr="
                    + repr([c[:20] for c in headers[:4]])
                    + " vals="
                    + repr([c[:22] for c in values[:4]])
                )
    if not found:
        print("     (none)")


def main() -> None:
    for item in json.loads(sys.stdin.read()):
        print(f"### {item['id']}")
        for year in item["years"]:
            show(
                item["code"],
                int(year),
                item["needle"],
                item.get("scope"),
                int(item.get("limit", 2)),
            )
        print()


if __name__ == "__main__":
    main()
