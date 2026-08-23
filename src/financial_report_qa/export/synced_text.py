"""Synced-text rewrite: collapse exported table spans into CSV links.

Reads each verified source TXT back through
``financial_report_qa.ingestion.txt_reader.read_document``, replaces every
exported table's inclusive 1-based line span with a single
``[TABLE: {table_id} -> {csv_relpath}]`` line, and mirrors the rewritten
document under the output directory while preserving every untouched byte of
prose, page markers, and original line endings.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import duckdb

from financial_report_qa.core.errors import ExportError
from financial_report_qa.export.csv_export import CsvExportManifest
from financial_report_qa.ingestion.provenance import SourceLine
from financial_report_qa.ingestion.txt_reader import read_document
from financial_report_qa.schemas.documents import DocumentRecord

_DEFAULT_OUTPUT_DIR = Path("data/interim/synced_text")
_LINK_TEXT = "[TABLE: {table_id} -> {csv_relpath}]"


@dataclass(frozen=True)
class TableExportEntry:
    """One table span scheduled for replacement inside its source document."""

    line_start: int
    line_end: int
    table_id: str
    csv_relpath: str


@dataclass(frozen=True)
class SyncedTextManifest:
    """Result summary of one ``export_synced_text`` run."""

    output_dir: Path
    document_count: int
    table_count: int


def build_synced_text(
    snapshot_root: Path, document: DocumentRecord, tables: list[TableExportEntry]
) -> str:
    """Rewrite one document with each table span collapsed to a single link.

    Spans are 1-based and inclusive; each must satisfy
    ``1 <= line_start <= line_end <= line_count`` and no two spans may overlap,
    otherwise ``ExportError`` is raised. Spans are applied from the largest
    ``line_end`` downward so earlier offsets never shift. Untouched lines keep
    their exact original text and line endings; every replacement line ends
    with ``"\\n"``.
    """
    decoded = read_document(snapshot_root, document)
    lines: list[SourceLine] = list(decoded.lines)
    line_count = len(lines)

    previous_end = 0
    for entry in sorted(tables, key=lambda item: (item.line_start, item.line_end)):
        if not 1 <= entry.line_start <= entry.line_end <= line_count:
            raise ExportError(
                f"table span outside document bounds: table_id={entry.table_id} "
                f"span=[{entry.line_start}, {entry.line_end}] line_count={line_count}"
            )
        if entry.line_start <= previous_end:
            raise ExportError(
                f"overlapping table spans: table_id={entry.table_id} "
                f"span=[{entry.line_start}, {entry.line_end}] "
                f"previous_line_end={previous_end}"
            )
        previous_end = entry.line_end

    for entry in sorted(tables, key=lambda item: item.line_end, reverse=True):
        lines[entry.line_start - 1 : entry.line_end] = [
            SourceLine(
                number=entry.line_start,
                text=_LINK_TEXT.format(table_id=entry.table_id, csv_relpath=entry.csv_relpath),
                line_ending="\n",
            )
        ]
    return "".join(line.text + line.line_ending for line in lines)


def export_synced_text(
    release_dir: Path,
    snapshot_root: Path,
    csv_manifest: CsvExportManifest,
    output_dir: Path | None = None,
) -> SyncedTextManifest:
    """Mirror every document owning at least one exported CSV, link-rewritten.

    Reads ``csv_manifest.manifest_path`` to map ``table_id`` to its exported
    bare CSV file name, selects from the release the documents that own at
    least one such table, rewrites each source text via ``build_synced_text``
    (link target ``(csv_manifest.output_dir / file_name).as_posix()``), and
    writes UTF-8 mirrors to ``output_dir / relative_path`` atomically --
    defaulting to ``data/interim/synced_text`` when ``output_dir`` is omitted.
    Documents are processed in ascending ``relative_path`` order; missing or
    unverifiable source files propagate their read errors unchanged.
    """
    resolved_output = _DEFAULT_OUTPUT_DIR if output_dir is None else output_dir
    links = _read_csv_links(csv_manifest.manifest_path)
    if not links:
        return SyncedTextManifest(output_dir=resolved_output, document_count=0, table_count=0)

    rows = _fetch_document_rows(release_dir, sorted(links))
    ordered = _group_entries(rows, csv_manifest.output_dir, links)

    for record, entries in ordered:
        text = build_synced_text(snapshot_root, record, entries)
        target = resolved_output.joinpath(*PurePosixPath(record.relative_path).parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        _write_bytes_atomically(target, text.encode("utf-8"))

    return SyncedTextManifest(
        output_dir=resolved_output,
        document_count=len(ordered),
        table_count=sum(len(entries) for _, entries in ordered),
    )


def _read_csv_links(manifest_path: Path) -> dict[str, str]:
    """Map ``table_id`` to its bare CSV file name from the JSONL manifest."""
    if not manifest_path.is_file():
        raise ExportError(f"CSV export manifest not found: {manifest_path}")
    links: dict[str, str] = {}
    for number, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise ExportError(
                f"invalid JSON in CSV export manifest line {number}: {manifest_path}"
            ) from error
        if not isinstance(payload, dict):
            raise ExportError(
                f"CSV export manifest line {number} is not an object: {manifest_path}"
            )
        table_id = payload.get("table_id")
        csv_path = payload.get("csv_path")
        if not isinstance(table_id, str) or not isinstance(csv_path, str) or not csv_path:
            raise ExportError(
                f"CSV export manifest line {number} lacks table_id/csv_path: {manifest_path}"
            )
        if table_id in links:
            raise ExportError(f"duplicate table_id in CSV export manifest: {table_id}")
        links[table_id] = csv_path
    return links


def _fetch_document_rows(release_dir: Path, table_ids: list[str]) -> list[tuple[Any, ...]]:
    """Join documents x tables for the given table ids in relative_path order."""
    placeholders = ", ".join("?" for _ in table_ids)
    connection = duckdb.connect(":memory:")
    try:
        return connection.execute(
            f"""
            SELECT
                d.doc_id, d.repo_id, d.revision, d.relative_path, d.company_code,
                d.report_year, d.statement_scope, d.sha256, d.file_size_bytes,
                d.encoding, d.inventory_status,
                t.table_id, t.line_start, t.line_end
            FROM read_parquet(?) AS t
            JOIN read_parquet(?) AS d USING (doc_id)
            WHERE t.table_id IN ({placeholders})
            ORDER BY d.relative_path, t.table_id
            """,
            [
                str(release_dir / "tables.parquet"),
                str(release_dir / "documents.parquet"),
                *table_ids,
            ],
        ).fetchall()
    except duckdb.Error as error:
        raise ExportError(f"cannot read release parquet in {release_dir}: {error}") from error
    finally:
        connection.close()


def _group_entries(
    rows: list[tuple[Any, ...]], csv_output_dir: Path, links: dict[str, str]
) -> list[tuple[DocumentRecord, list[TableExportEntry]]]:
    """Rebuild DocumentRecords plus per-document entries from parquet rows.

    Only fields present on ``DocumentRecord`` are selected -- the parquet's
    extra ``ruleset_version`` / ``normalization_fingerprint`` columns are
    dropped because the model forbids extras. Row order (ascending
    ``relative_path``) defines processing order.
    """
    ordered: list[tuple[DocumentRecord, list[TableExportEntry]]] = []
    positions: dict[str, int] = {}
    for row in rows:
        (
            doc_id,
            repo_id,
            revision,
            relative_path,
            company_code,
            report_year,
            statement_scope,
            sha256,
            file_size_bytes,
            encoding,
            inventory_status,
            table_id,
            line_start,
            line_end,
        ) = row
        record = DocumentRecord.model_validate(
            {
                "doc_id": str(doc_id),
                "repo_id": str(repo_id),
                "revision": str(revision),
                "relative_path": str(relative_path),
                "company_code": str(company_code),
                "report_year": int(report_year),
                "statement_scope": str(statement_scope),
                "sha256": str(sha256),
                "file_size_bytes": int(file_size_bytes),
                "encoding": None if encoding is None else str(encoding),
                "inventory_status": str(inventory_status),
            }
        )
        entry = TableExportEntry(
            line_start=int(line_start),
            line_end=int(line_end),
            table_id=str(table_id),
            csv_relpath=(csv_output_dir / links[str(table_id)]).as_posix(),
        )
        position = positions.get(record.doc_id)
        if position is None:
            positions[record.doc_id] = len(ordered)
            ordered.append((record, [entry]))
        else:
            ordered[position][1].append(entry)
    return ordered


def _write_bytes_atomically(target: Path, payload: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary_path.replace(target)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
