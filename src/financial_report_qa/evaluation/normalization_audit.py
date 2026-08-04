"""Normalization issue audit and remediation evaluation module."""

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field

SAMPLE_SCHEMA = pa.schema(
    [
        ("sample_id", pa.string()),
        ("release_fingerprint", pa.string()),
        ("issue_code", pa.string()),
        ("field", pa.string()),
        ("raw_value", pa.string()),
        ("doc_id", pa.string()),
        ("table_id", pa.string()),
        ("cell_id", pa.string()),
        ("company_code", pa.string()),
        ("report_year", pa.int32()),
        ("statement_type", pa.string()),
        ("table_title_raw", pa.string()),
        ("table_unit_raw", pa.string()),
        ("row_label_raw", pa.string()),
        ("column_label_raw", pa.string()),
        ("value_raw", pa.string()),
        ("source_line_start", pa.int32()),
        ("source_line_end", pa.int32()),
        ("stratum_key", pa.string()),
        ("selection_rank", pa.string()),
    ]
)


class AuditSamplingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    issue_limits: dict[str, int] = Field(default_factory=dict)
    max_per_stratum: int = Field(default=5, ge=1)
    seed: str = Field(default="normalization-audit-v1", min_length=1)


def build_issue_sample(
    release_path: Path,
    release_fingerprint: str,
    config: AuditSamplingConfig,
) -> pa.Table:
    """Deterministically sample normalization issues from a dataset release."""
    if not release_path.is_dir():
        raise ValueError(f"Release path is not a directory: {release_path}")
    manifest_path = release_path / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("Release manifest.json is missing")

    try:
        manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise ValueError(f"Invalid release manifest.json: {e}") from e

    manifest_fp = str(manifest_data.get("dataset_fingerprint", ""))
    if manifest_fp != release_fingerprint:
        raise ValueError(
            f"Release fingerprint mismatch: expected {release_fingerprint}, got {manifest_fp}"
        )

    for fn in ("documents.parquet", "tables.parquet", "cells.parquet", "issues.parquet"):
        if not (release_path / fn).is_file():
            raise ValueError(f"Missing required release file: {fn}")

    read_table = cast(Any, pq.read_table)
    doc_rows = read_table(release_path / "documents.parquet").to_pylist()
    tbl_rows = read_table(release_path / "tables.parquet").to_pylist()
    cell_rows = read_table(release_path / "cells.parquet").to_pylist()
    issue_rows = read_table(release_path / "issues.parquet").to_pylist()

    docs_by_id = {str(r["doc_id"]): r for r in doc_rows}
    tables_by_id = {str(r["table_id"]): r for r in tbl_rows}
    cells_by_id = {str(r["cell_id"]): r for r in cell_rows}

    seen_sample_ids: set[str] = set()
    candidates_by_issue: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for iss in issue_rows:
        doc_id = str(iss["doc_id"])
        if doc_id not in docs_by_id:
            raise ValueError(f"Unresolved doc_id: {doc_id}")

        table_id = str(iss["table_id"]) if iss.get("table_id") is not None else None
        if table_id is not None and table_id not in tables_by_id:
            raise ValueError(f"Unresolved table_id: {table_id}")

        cell_id = str(iss["cell_id"]) if iss.get("cell_id") is not None else None
        if cell_id is not None and cell_id not in cells_by_id:
            raise ValueError(f"Unresolved cell_id: {cell_id}")

        doc = docs_by_id[doc_id]
        company_code = str(doc.get("company_code", ""))
        report_year = int(doc["report_year"])

        statement_type: str | None = None
        table_title_raw: str | None = None
        table_unit_raw: str | None = None
        if table_id is not None:
            tbl = tables_by_id[table_id]
            val = tbl.get("statement_type")
            statement_type = str(val) if val is not None else None
            val = tbl.get("title_raw")
            table_title_raw = str(val) if val is not None else None
            val = tbl.get("unit_raw")
            table_unit_raw = str(val) if val is not None else None

        row_label_raw: str | None = None
        column_label_raw: str | None = None
        value_raw: str | None = None
        source_line_start: int | None = None
        source_line_end: int | None = None

        if cell_id is not None:
            c = cells_by_id[cell_id]
            val = c.get("row_label_raw")
            row_label_raw = str(val) if val is not None else None
            val = c.get("column_label_raw")
            column_label_raw = str(val) if val is not None else None
            val = c.get("value_raw")
            value_raw = str(val) if val is not None else None
            val = c.get("source_line_start")
            source_line_start = int(val) if val is not None else None
            val = c.get("source_line_end")
            source_line_end = int(val) if val is not None else None
        elif table_id is not None:
            tbl = tables_by_id[table_id]
            val = tbl.get("line_start")
            source_line_start = int(val) if val is not None else None
            val = tbl.get("line_end")
            source_line_end = int(val) if val is not None else None

        issue_code = str(iss["code"])
        field = str(iss["field"])
        raw_val = iss.get("raw_value")
        raw_value = str(raw_val) if raw_val is not None else None

        # Hash sample_id
        payload = (
            f"{release_fingerprint}|{issue_code}|{doc_id}|{table_id or ''}|"
            f"{cell_id or ''}|{field}|{raw_value or ''}"
        )
        sample_id = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        if sample_id in seen_sample_ids:
            raise ValueError(f"Duplicate sample_id: {sample_id}")
        seen_sample_ids.add(sample_id)

        # Hash selection_rank
        rank_payload = f"{config.seed}|{sample_id}"
        selection_rank = hashlib.sha256(rank_payload.encode("utf-8")).hexdigest()

        # Stratum key
        norm_raw = raw_value.strip().casefold() if raw_value is not None else ""
        stratum_key = f"{issue_code}|{company_code}|{report_year}|{statement_type or ''}|{norm_raw}"

        row_record = {
            "sample_id": sample_id,
            "release_fingerprint": release_fingerprint,
            "issue_code": issue_code,
            "field": field,
            "raw_value": raw_value,
            "doc_id": doc_id,
            "table_id": table_id,
            "cell_id": cell_id,
            "company_code": company_code,
            "report_year": report_year,
            "statement_type": statement_type,
            "table_title_raw": table_title_raw,
            "table_unit_raw": table_unit_raw,
            "row_label_raw": row_label_raw,
            "column_label_raw": column_label_raw,
            "value_raw": value_raw,
            "source_line_start": source_line_start,
            "source_line_end": source_line_end,
            "stratum_key": stratum_key,
            "selection_rank": selection_rank,
        }
        candidates_by_issue[issue_code].append(row_record)

    selected_rows: list[dict[str, Any]] = []

    for issue_code, items in sorted(candidates_by_issue.items()):
        strata: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for it in items:
            strata[str(it["stratum_key"])].append(it)

        stratum_retained: list[dict[str, Any]] = []
        for s_key, s_items in sorted(strata.items()):
            s_items.sort(key=lambda r: (str(r["selection_rank"]), str(r["sample_id"])))
            stratum_retained.extend(s_items[: config.max_per_stratum])

        stratum_retained.sort(
            key=lambda r: (str(r["selection_rank"]), str(r["sample_id"]))
        )

        limit = config.issue_limits.get(issue_code)
        if limit is not None:
            selected_rows.extend(stratum_retained[:limit])
        else:
            selected_rows.extend(stratum_retained)

    selected_rows.sort(
        key=lambda r: (str(r["issue_code"]), str(r["selection_rank"]), str(r["sample_id"]))
    )
    return pa.Table.from_pylist(selected_rows, schema=SAMPLE_SCHEMA)
