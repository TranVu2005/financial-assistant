"""Auto-propose gold annotations by row-label match; the annotator reviews."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from srcread import ROOT, fold, parse_number, read_rows, statement_dirs
from propose import BENCH, company_of, scope_of, years_of

UNIT_WORDS = [
    ("nghin ty", "VND_trillion"),
    ("ty dong", "VND_billion"),
    ("trieu dong", "VND_million"),
    ("phan tram", "percent"),
]


def requested_unit(question):
    key = fold(question)
    for word, unit in UNIT_WORDS:
        if word in key:
            return unit
    if chr(37) in question:
        return "percent"
    return None


def metric_phrase(question):
    text = question.strip()
    for cut in (" của ", " cua "):
        if cut in text:
            return text.split(cut)[0].strip()
    for cut in (" là bao nhiêu", " trong năm", " năm "):
        if cut in text:
            return text.split(cut)[0].strip()
    return text


def source_unit_of(header_cell, fallback):
    key = fold(header_cell + " " + fallback)
    if "trieu dong" in key or "trieu vnd" in key:
        return "VND_million"
    if "ty dong" in key or "ty vnd" in key:
        return "VND_billion"
    if "nghin dong" in key:
        return "VND_thousand"
    if "vnd" in key or "dong" in key:
        return "VND"
    return "unknown"


def pick_column(row, header, year, report_year):
    offset = len(row.cells) - len(header)
    hits = []
    for index, cell in enumerate(header):
        key = fold(cell)
        if not key:
            continue
        if str(year) in cell:
            hits.append((index, cell, "year-in-header"))
        elif year == report_year and ("so cuoi nam" in key or "nam nay" in key):
            hits.append((index, cell, "current-period-label"))
    return hits, offset


LEAD_NUMBER = re.compile(r'^[IVX0-9]+[.)]?([.)]?[0-9]+)*[.)]?\s+')
LEAD_MARK = re.compile(r'^[-–•▪+*]\s*')
DROP_LEADING = ('so du ', 'tong gia tri ', 'tong so ', 'tong ', 'gia tri ', 'muc ', 'khoan ')


def label_key(text):
    cleaned = LEAD_MARK.sub('', text.strip())
    cleaned = LEAD_NUMBER.sub('', cleaned)
    return fold(cleaned)


def phrase_variants(phrase):
    base = fold(phrase)
    out = [base]
    for prefix in DROP_LEADING:
        if base.startswith(prefix):
            out.append(base[len(prefix):])
    trimmed = re.sub(r' (nam|cuoi nam|dau nam) [0-9]{4}$', '', base)
    if trimmed != base:
        out.append(trimmed)
    return [v for v in dict.fromkeys(out) if v]


def auto(record):
    question = record["question"]
    codes = company_of(question)
    years = years_of(question)
    scope = scope_of(question)
    phrase = metric_phrase(question)
    out = {"vifinqa_id": record["vifinqa_id"], "question": question, "company": codes,
           "years": years, "scope": scope, "requested_unit": requested_unit(question),
           "phrase": phrase, "matches": []}
    if len(codes) != 1 or len(years) != 1:
        out["problem"] = "company/year not unique"
        return out
    wants = phrase_variants(phrase)
    year = years[0]
    for directory in statement_dirs(codes[0], year):
        if scope and scope not in directory.name:
            continue
        for path in directory.glob("*_extracted.txt"):
            for row in read_rows(path):
                label = label_key(row.label)
                if not label or len(row.cells) < 2:
                    continue
                exact = label in wants
                near = any(w in label or label in w for w in wants if len(w) > 12)
                if not (exact or near):
                    continue
                hits, offset = pick_column(row, row.header, year, year)
                for index, cell, why in hits:
                    value_index = index - offset if offset else index
                    if not (0 <= value_index < len(row.cells)):
                        continue
                    raw = row.cells[value_index]
                    numeric = parse_number(raw)
                    if numeric is None:
                        continue
                    out["matches"].append({
                        "exact": exact, "why": why,
                        "relative_path": str(path.relative_to(ROOT)).replace(chr(92), "/"),
                        "line": row.line, "row": row.label, "column": cell,
                        "raw": raw, "numeric": numeric,
                        "source_unit": source_unit_of(cell, " ".join(row.header)),
                    })
    return out


def main():
    ids = {int(x) for x in sys.argv[1:]}
    for line in open(BENCH, encoding="utf-8"):
        record = json.loads(line)
        if ids and record["vifinqa_id"] not in ids:
            continue
        result = auto(record)
        exact = [m for m in result["matches"] if m["exact"]]
        chosen = exact or result["matches"]
        tag = "EXACT" if exact else ("LOOSE" if result["matches"] else "NONE")
        head = "### " + str(result["vifinqa_id"]) + " [" + tag + "] " + repr(result["phrase"])
        head += " " + str(result["company"]) + " " + str(result["years"])
        head += " " + str(result["scope"]) + " -> " + str(result["requested_unit"])
        print(head)
        seen = set()
        for m in chosen[:4]:
            key = (m["relative_path"], m["line"], m["row"], m["column"], m["raw"])
            if key in seen:
                continue
            seen.add(key)
            name = m["relative_path"].split("/")[-1][:46]
            print("    " + name + ":" + str(m["line"]) + " | " + repr(m["row"][:54]))
            print("        col=" + repr(m["column"][:38]) + " raw=" + m["raw"]
                  + " unit=" + m["source_unit"] + " (" + m["why"] + ")")


if __name__ == "__main__":
    main()
