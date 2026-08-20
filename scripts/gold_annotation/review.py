"""Compact bulk-review sheet for gold annotation.

For each benchmark question, prints the resolved issuer/year/scope/unit and the
best candidate rows from the raw report, each already narrowed to the column
for the period the question asks about. The annotator reads this sheet and
decides; nothing here writes a label.

Imports nothing from `financial_report_qa` (ADR 0009 A2).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from auto import label_key, metric_phrase, phrase_variants, requested_unit, source_unit_of
from propose import BENCH, company_of, content_tokens, scope_of, years_of
from srcread import ROOT, fold, parse_number, read_rows, statement_dirs

GOLD = Path(__file__).resolve().parents[2] / "data" / "qa" / "dev-benchmark-v1.gold.jsonl"


def already_done() -> set[int]:
    if not GOLD.exists():
        return set()
    out = set()
    for line in GOLD.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.add(json.loads(line)["vifinqa_id"])
    return out


NUMBERING = re.compile(r"^[IVXLC0-9]+\s*[.)-]?\s*$")


def normalize_row(row):
    """(label, values, headers) with each value paired to its own column.

    Three OCR layouts occur and all three have to line up, or balance-sheet
    rows stay invisible to a label search: cells aligned 1:1 with the header;
    the row missing its label cell; and extra leading cells where the line
    numbering ("IV.", "10") sits in its own column ahead of the metric name.
    The label is then the first cell that is neither a number nor bare
    numbering.
    """
    cells = [c for c in row.cells]
    header = [c for c in row.header]
    if not cells:
        return "", [], []
    if header and len(header) == len(cells):
        pairs = list(zip(header, cells))
    elif header and len(header) - 1 == len(cells):
        pairs = list(zip(header[1:], cells))
    elif header and len(cells) > len(header):
        pairs = list(zip(header, cells[len(cells) - len(header):]))
    else:
        pairs = [("", c) for c in cells]
    label_index = 0
    for index, (_head, cell) in enumerate(pairs[:3]):
        text = cell.strip()
        if text and parse_number(text) is None and not NUMBERING.match(text):
            label_index = index
            break
    label = pairs[label_index][1].strip()
    rest = pairs[label_index + 1:]
    return label, [c for _h, c in rest], [h for h, _c in rest]


def period_columns(row, year: int, report_year: int):
    """(header cell, raw value) pairs for the period the question asks about."""
    _label, values, headers = normalize_row(row)
    out = []
    for index, cell in enumerate(headers):
        key = fold(cell)
        current = year == report_year and ("so cuoi nam" in key or "nam nay" in key)
        prior = year == report_year - 1 and ("so dau nam" in key or "nam truoc" in key)
        if str(year) not in cell and not current and not prior:
            continue
        raw = values[index]
        if parse_number(raw) is None:
            continue
        out.append((cell, raw))
    if not out:
        numeric = [c for c in values if parse_number(c) is not None]
        out = [("(no period column matched)", c) for c in numeric[:3]]
    return out


def candidates(record: dict, limit: int = 3):
    question = record["question"]
    codes = company_of(question)
    years = years_of(question)
    scope = scope_of(question)
    if len(codes) != 1 or not years:
        # Multi-company questions are multi-hop by construction; scanning
        # every issuer's reports for them costs minutes and answers nothing,
        # so the sheet just flags them for the multi-hop pass.
        return codes, years, scope, []
    if len(years) > 2:
        return codes, years, scope, []
    tokens = content_tokens(question, codes)
    wants = phrase_variants(metric_phrase(question))
    scored = []
    for year in years:
        for directory in statement_dirs(codes[0], year):
            if scope and scope not in directory.name:
                continue
            for path in directory.glob("*_extracted.txt"):
                for row in read_rows(path):
                    if len(row.cells) < 2:
                        continue
                    label, _values, _headers = normalize_row(row)
                    key = label_key(label)
                    hits = sum(1 for t in tokens if t in key) if key else 0
                    if key and key in wants:
                        score = 100.0
                    elif hits >= 2:
                        score = hits / (1 + 0.08 * len(key.split()))
                    else:
                        # OCR often leaves the metric name in a later cell and
                        # the first cell empty or a line code, so fall back to
                        # the whole row's text at a lower score.
                        joined = label_key(" ".join(row.cells[:3]))
                        row_hits = sum(1 for t in tokens if t in joined)
                        if row_hits < max(2, len(tokens) - 1):
                            continue
                        score = 0.5 * row_hits / (1 + 0.08 * len(joined.split()))
                    # A row whose table actually has a column for the asked
                    # period is far likelier to be the intended one than a
                    # segment/maturity note that merely repeats the label.
                    has_period = any(
                        str(year) in cell
                        or "so cuoi nam" in fold(cell)
                        or "nam nay" in fold(cell)
                        for cell in row.header
                    )
                    score += 2.0 if has_period else -1.0
                    scored.append((score, path, row, year))
    scored.sort(key=lambda item: -item[0])
    seen = set()
    out = []
    for score, path, row, year in scored:
        marker = (path.name, row.line, row.label)
        if marker in seen:
            continue
        seen.add(marker)
        out.append((score, path, row, year))
        if len(out) >= limit:
            break
    return codes, years, scope, out


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    limit = 15
    for arg in sys.argv[1:]:
        if arg.startswith("--limit="):
            limit = int(arg.split("=", 1)[1])
    ids = {int(a) for a in args}
    done = already_done()
    shown = 0
    for line in BENCH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        qid = record["vifinqa_id"]
        if ids and qid not in ids:
            continue
        if not ids and qid in done:
            continue
        codes, years, scope, rows = candidates(record)
        unit = requested_unit(record["question"])
        head = "### " + str(qid) + " | " + str(codes) + " " + str(years)
        head += " scope=" + str(scope) + " ask=" + str(unit)
        print(head)
        print("Q: " + record["question"])
        if not rows:
            print("    !! no candidate row -- raw text fallback:")
            tokens = content_tokens(record["question"], codes)
            needle = " ".join(sorted(tokens, key=len, reverse=True)[:3])
            shown_lines = 0
            for year in years:
                for directory in statement_dirs(codes[0] if codes else "", year):
                    if scope and scope not in directory.name:
                        continue
                    for path in directory.glob("*_extracted.txt"):
                        for lineno, raw in enumerate(
                            path.read_text(encoding="utf-8").splitlines(), start=1
                        ):
                            folded = fold(raw)
                            if all(t in folded for t in needle.split()) and shown_lines < 3:
                                where = folded.find(needle.split()[0])
                                start = max(0, where - 80)
                                tag = directory.name.split("_")[-1][:4]
                                print("      " + tag + ":" + str(lineno) + " ..."
                                      + raw[start:start + 320] + "...")
                                shown_lines += 1
        for score, path, row, year in rows:
            tag = path.parent.name.split("_")[-1][:4]
            label, values, headers = normalize_row(row)
            print("  - " + tag + ":" + str(row.line) + " " + repr(label[:88]))
            print("      hdr=" + repr([c[:24] for c in headers[:6]]))
            print("      cells=" + repr([c[:22] for c in values[:6]]))
            for cell, raw in period_columns(row, year, year)[:2]:
                unit_hint = source_unit_of(cell, " ".join(row.header))
                print("      -> col=" + repr(cell[:32]) + " raw=" + raw + " [" + unit_hint + "]")
        print()
        shown += 1
        if shown >= limit:
            break


if __name__ == "__main__":
    main()
