"""Append verified gold annotations, re-reading each value from source.

Every record is checked against the raw report text before it is stored: the
named row must exist at the named line and the raw value string must appear
in that row, so a typo fails loudly instead of becoming a wrong gold label.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from review import normalize_row
from srcread import ROOT, fold, read_rows

OUT = Path(__file__).resolve().parents[2] / "data" / "qa" / "dev-benchmark-v1.gold.jsonl"
NL = chr(10)
VID = "vifinqa_id"
GOLD_ROWS = "gold_rows"


def verify(record: dict) -> None:
    if record.get("status") != "annotated":
        return
    path = ROOT / record["gold_table"]["relative_path"]
    if not path.is_file():
        raise SystemExit(f"id {record[VID]}: no such report {path}")
    line = record["gold_table"]["line"]
    rows = [r for r in read_rows(path) if r.line == line]
    if not rows:
        raise SystemExit(f"id {record[VID]}: no table row at line {line}")
    wanted = [fold(x) for x in record["gold_rows"]]
    # A row's metric name may sit in the first cell or, when the line
    # numbering takes that cell, in the next one - accept either spelling.
    matched = [
        r for r in rows
        if fold(r.label) in wanted
        or fold(normalize_row(r)[0]) in wanted
        or any(fold(cell) in wanted for cell in r.cells)
    ]
    if len(matched) < len(wanted):
        labels = [r.label for r in rows][:12]
        raise SystemExit(f"id {record[VID]}: rows {record[GOLD_ROWS]} not at line {line}; saw {labels}")
    for value in record["gold_values"]:
        raw = value["raw"]
        if not any(raw in cell for r in matched for cell in r.cells):
            raise SystemExit(f"id {record[VID]}: raw {raw} not in the named row(s)")


def main() -> None:
    payload = json.loads(sys.stdin.read())
    records = payload if isinstance(payload, list) else [payload]
    existing = {}
    if OUT.exists():
        for line in OUT.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                existing[item[VID]] = item
    for record in records:
        verify(record)
        existing[record[VID]] = record
    body = NL.join(json.dumps(existing[k], ensure_ascii=False) for k in sorted(existing))
    OUT.write_text(body + NL, encoding="utf-8")
    print(f"stored {len(records)} record(s); file now holds {len(existing)}")


if __name__ == "__main__":
    main()
