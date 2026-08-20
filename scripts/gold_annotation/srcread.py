"""Standalone reader for ViFinQA `_extracted.txt` reports.

Deliberately imports nothing from `financial_report_qa`: gold labels must be
read from the source text independently of the pipeline being scored
(ADR 0009 decision A2).
"""
from __future__ import annotations

import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path("data/raw/ViFinQA/financial_statements")
_CELL_RE = re.compile(r"<td[^>]*>(.*?)</td>", re.DOTALL)
_ROW_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")


def fold(text: str) -> str:
    folded = unicodedata.normalize("NFKD", text.casefold()).replace("đ", "d")
    stripped = "".join(c for c in folded if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^a-z0-9]+", " ", stripped).split())


@dataclass
class Row:
    line: int
    page: int
    cells: list[str]
    table_line: int
    header: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return self.cells[0] if self.cells else ""


def statement_dirs(company: str, year: int) -> list[Path]:
    base = ROOT / company / str(year)
    return sorted(p for p in base.iterdir() if p.is_dir()) if base.is_dir() else []


_ROW_CACHE: dict[str, list] = {}


def read_rows(path: Path) -> list[Row]:
    """Parsed rows for one report, cached: a single annotation batch
    re-reads the same files dozens of times otherwise."""
    key = str(path)
    cached = _ROW_CACHE.get(key)
    if cached is not None:
        return cached
    rows: list[Row] = []
    page = 0
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        page_marker = re.match(r"=====\s*PAGE\s+(\d+)\s*=====", raw)
        if page_marker:
            page = int(page_marker.group(1))
            continue
        if "<tr" not in raw:
            continue
        table_rows = _ROW_RE.findall(raw)
        header: list[str] = []
        for index, tr in enumerate(table_rows):
            cells = [" ".join(_TAG_RE.sub("", c).split()) for c in _CELL_RE.findall(tr)]
            if index == 0:
                header = cells
            rows.append(Row(line=lineno, page=page, cells=cells, table_line=lineno, header=header))
    _ROW_CACHE[key] = rows
    return rows


def parse_number(text: str) -> float | None:
    """Vietnamese formatting: '.' thousands, ',' decimals, (…) negative."""
    cleaned = text.strip().replace("\u00a0", " ")
    negative = cleaned.startswith("(") and cleaned.endswith(")")
    cleaned = cleaned.strip("()").replace(" ", "")
    if not re.fullmatch(r"-?\d{1,3}(\.\d{3})*(,\d+)?|-?\d+(,\d+)?", cleaned):
        return None
    cleaned = cleaned.replace(".", "").replace(",", ".")
    try:
        value = float(cleaned)
    except ValueError:
        return None
    return -value if negative else value


def search(company: str, year: int, terms: list[str], *, scope: str | None = None) -> None:
    for directory in statement_dirs(company, year):
        if scope and scope not in directory.name:
            continue
        files = list(directory.glob("*_extracted.txt"))
        if not files:
            continue
        path = files[0]
        print(f"--- {path.relative_to(ROOT)} ---")
        for row in read_rows(path):
            label_key = fold(row.label)
            if all(fold(t) in label_key for t in terms):
                values = " | ".join(row.cells[1:8])
                print(f"  line {row.line} (p{row.page}) [{row.label}] -> {values}")
                print(f"      header: {row.header[:8]}")




def grep(company: str, year: int, phrase: str, *, scope=None, width: int = 260) -> None:
    """Full-text search inside a company-year report (rows or prose)."""
    needle = fold(phrase)
    for directory in statement_dirs(company, year):
        if scope and scope not in directory.name:
            continue
        for path in directory.glob('*_extracted.txt'):
            for lineno, raw in enumerate(path.read_text(encoding='utf-8').splitlines(), start=1):
                if needle in fold(raw):
                    where = fold(raw).find(needle)
                    start = max(0, where - 60)
                    print(f'  {path.name}:{lineno}: ...{raw[start:start + width]}...')
if __name__ == "__main__":
    company, year = sys.argv[1], int(sys.argv[2])
    scope = None
    terms = sys.argv[3:]
    if terms and terms[0] in ("consolidated", "separate"):
        scope, terms = terms[0], terms[1:]
    search(company, year, terms, scope=scope)
