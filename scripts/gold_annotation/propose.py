"""Surface gold-annotation candidates for dev-benchmark questions.

Reads only the raw ViFinQA text and `code_stock.csv`; imports nothing from
`financial_report_qa` (ADR 0009 A2 -- gold must not come from the pipeline
being scored). It proposes; a human/annotator decides.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from srcread import ROOT, fold, read_rows, statement_dirs  # noqa: E402

BENCH = Path("data/qa/dev-benchmark-v1.jsonl")
STOP = set(
    fold(
        "cua nam la bao nhieu ty dong trieu nghin phan tram cong ty me rieng hop nhat "
        "tai ngay thang cuoi dau ky trong cho va den tinh theo bang duoc gi thi so voi "
        "giua muc chiem bao gom hay"
    ).split()
)


def tickers() -> dict[str, str]:
    out: dict[str, str] = {}
    with open("data/raw/ViFinQA/code_stock.csv", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if len(row) >= 2 and row[0] != "code":
                out[row[0].strip()] = row[1].strip()
    return out


TICKERS = tickers()


def company_of(question: str) -> list[str]:
    """Longest issuer *name* wins over any bare ticker in the text.

    "CTCP Chung khoan FPT" is FTS, not FPT: matching the bare token first
    silently annotates the wrong company's report.
    """
    key = fold(question)
    # The registry writes "CTCP …" while questions often spell it out as
    # "Công ty Cổ phần …"; without expanding the abbreviation the issuer is
    # simply not found (e.g. DNH, "CTCP Thủy điện Đa Nhim - Hàm Thuận - Đa Mi").
    for long_form, short_form in (("cong ty co phan", "ctcp"), ("cong ty cp", "ctcp"),
                                 ("ngan hang thuong mai co phan", "ngan hang tmcp")):
        key = key.replace(long_form, short_form)
    named = sorted(
        ((len(fold(name)), code) for code, name in TICKERS.items() if fold(name).replace('cong ty co phan', 'ctcp') in key),
        reverse=True,
    )
    if named:
        longest = named[0][0]
        return sorted({code for length, code in named if length == longest})
    parenthesised = [t for t in re.findall(r"\(([A-Z0-9]{2,10})\)", question) if t in TICKERS]
    if parenthesised:
        return sorted(set(parenthesised))
    return sorted({t for t in re.findall(r"(?<![A-Z0-9])[A-Z0-9]{2,10}(?![A-Z0-9])", question) if t in TICKERS})


def years_of(question: str) -> list[int]:
    return sorted({int(y) for y in re.findall(r"\b(20[0-2][0-9])\b", question)})


def scope_of(question: str) -> str | None:
    key = fold(question)
    if "cong ty me" in key or "rieng" in key:
        return "separate"
    if "hop nhat" in key:
        return "consolidated"
    return None


def content_tokens(question: str, codes: list[str]) -> list[str]:
    company_words: set[str] = set()
    for code in codes:
        company_words |= set(fold(TICKERS[code]).split())
        company_words.add(fold(code))
    tokens = [
        t
        for t in fold(question).split()
        if len(t) >= 3 and t not in STOP and t not in company_words and not t.isdigit()
    ]
    return tokens


UNIT_RE = re.compile(r"(don vi tinh|don vi|dvt)[^a-z0-9]{0,4}([a-z0-9 ]{0,30})")


def unit_hint(path, line: int) -> str:
    """Nearest 'Đơn vị tính: ...' above the row, plus any unit word in the header."""
    text = path.read_text(encoding="utf-8").splitlines()
    for offset in range(line - 1, max(-1, line - 25), -1):
        candidate = fold(text[offset])
        match = UNIT_RE.search(candidate)
        if match:
            return text[offset].strip()[:110]
    return ""


def propose(record: dict, *, top: int = 3, per_file: int = 2) -> None:
    question = record["question"]
    codes = company_of(question)
    years = years_of(question)
    scope = scope_of(question)
    tokens = content_tokens(question, codes)
    print(f"### id {record['vifinqa_id']} | baseline {record['source_status']}/{record['source_code']}")
    print(f"Q: {question}")
    print(f"   company={codes} years={years} scope={scope or 'unspecified'} terms={tokens}")
    if not codes or not years:
        print("   !! cannot resolve company/year from the question text")
        print()
        return
    scored: list[tuple[float, str, object]] = []
    for code in codes:
        for year in years:
            for directory in statement_dirs(code, year):
                if scope and scope not in directory.name:
                    continue
                files = list(directory.glob("*_extracted.txt"))
                if not files:
                    continue
                path = files[0]
                best: list[tuple[float, object]] = []
                for row in read_rows(path):
                    if len(row.cells) < 2:
                        continue
                    label_key = fold(row.label)
                    if not label_key:
                        continue
                    hits = sum(1 for t in tokens if t in label_key)
                    if not hits:
                        continue
                    score = hits / (1 + 0.05 * len(label_key.split()))
                    best.append((score, row))
                best.sort(key=lambda pair: -pair[0])
                for score, row in best[:per_file]:
                    scored.append((score, str(path.relative_to(ROOT)), row))
    scored.sort(key=lambda triple: -triple[0])
    if not scored:
        print("   !! no row label matched any question term")
    for score, rel, row in scored[:top]:
        hint = unit_hint(ROOT / rel, row.line)
        if hint and not any(w in fold(hint) for w in ("don vi", "trieu", "ty dong", "nghin")):
            hint = ""
        print(f"   [{score:.2f}] {rel}:{row.line} (p{row.page})")
        print(f"        row: {row.label!r}")
        print(f"        hdr: {row.header[:6]}")
        print(f"        val: {row.cells[1:6]}")
        if hint:
            print(f"        unit-context: {hint}")
    print()


if __name__ == "__main__":
    ids = {int(x) for x in sys.argv[1:]}
    for line in open(BENCH, encoding="utf-8"):
        record = json.loads(line)
        if not ids or record["vifinqa_id"] in ids:
            propose(record)
