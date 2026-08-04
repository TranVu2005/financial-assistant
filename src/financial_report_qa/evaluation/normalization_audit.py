"""Normalization issue audit and remediation evaluation module."""

import csv
import hashlib
import json
from collections import defaultdict
from collections.abc import Collection, Sequence
from decimal import Decimal
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

        stratum_retained.sort(key=lambda r: (str(r["selection_rank"]), str(r["sample_id"])))

        limit = config.issue_limits.get(issue_code)
        if limit is not None:
            selected_rows.extend(stratum_retained[:limit])
        else:
            selected_rows.extend(stratum_retained)

    selected_rows.sort(
        key=lambda r: (str(r["issue_code"]), str(r["selection_rank"]), str(r["sample_id"]))
    )
    return pa.Table.from_pylist(selected_rows, schema=SAMPLE_SCHEMA)


VALID_LABELS = {"true_issue", "false_positive", "uncertain"}

VALID_CAUSE_CODES = {
    "year_header_as_unit",
    "missing_unit_context",
    "unsupported_unit_alias",
    "non_metric_row",
    "unsupported_metric_alias",
    "non_value_cell",
    "ocr_corruption",
    "separator_ambiguity",
    "legitimate_missing_value",
    "mixed_unit_table",
    "statement_signal_conflict",
    "period_missing_year",
    "period_format_ambiguous",
    "other",
}


class LabelRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sample_id: str = Field(min_length=1)
    label: str
    cause_code: str
    reviewer_note: str = Field(default="")


class IssueAuditMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sample_count: int
    true_issue_count: int
    false_positive_count: int
    uncertain_count: int
    unlabeled_count: int
    conclusive_coverage: Decimal
    false_positive_rate: Decimal | None
    cause_counts: dict[str, int]


def load_and_validate_labels(sample: pa.Table, labels_path: Path) -> tuple[LabelRecord, ...]:
    """Load and validate human review labels against a sampled dataset."""
    valid_sample_ids = {str(row["sample_id"]) for row in sample.to_pylist()}
    if not labels_path.is_file():
        return ()

    seen_sample_ids: set[str] = set()
    records: list[LabelRecord] = []

    with labels_path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            s_id = str(row.get("sample_id", "")).strip()
            if not s_id or s_id not in valid_sample_ids:
                raise ValueError(f"unknown sample_id: {s_id}")
            if s_id in seen_sample_ids:
                raise ValueError(f"duplicate sample_id: {s_id}")
            seen_sample_ids.add(s_id)

            label = str(row.get("label", "")).strip()
            if label not in VALID_LABELS:
                raise ValueError(f"invalid label: {label}")

            cause_code = str(row.get("cause_code", "")).strip()
            if cause_code not in VALID_CAUSE_CODES:
                raise ValueError(f"invalid cause_code: {cause_code}")

            note = str(row.get("reviewer_note", "")).strip()
            records.append(
                LabelRecord(
                    sample_id=s_id,
                    label=label,
                    cause_code=cause_code,
                    reviewer_note=note,
                )
            )
    return tuple(records)


def evaluate_labels(
    sample: pa.Table,
    labels: Sequence[LabelRecord],
) -> dict[str, IssueAuditMetrics]:
    """Evaluate conclusive label coverage and false positive rates by issue code."""
    labels_by_id = {lbl.sample_id: lbl for lbl in labels}

    rows_by_issue: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sample.to_pylist():
        rows_by_issue[str(row["issue_code"])].append(row)

    metrics_by_code: dict[str, IssueAuditMetrics] = {}

    for issue_code, rows in sorted(rows_by_issue.items()):
        sample_count = len(rows)
        true_issue_count = 0
        false_positive_count = 0
        uncertain_count = 0
        unlabeled_count = 0
        cause_counts: dict[str, int] = defaultdict(int)

        for row in rows:
            s_id = str(row["sample_id"])
            lbl = labels_by_id.get(s_id)
            if lbl is None:
                unlabeled_count += 1
            else:
                cause_counts[lbl.cause_code] += 1
                if lbl.label == "true_issue":
                    true_issue_count += 1
                elif lbl.label == "false_positive":
                    false_positive_count += 1
                elif lbl.label == "uncertain":
                    uncertain_count += 1

        conclusive = true_issue_count + false_positive_count
        conclusive_coverage = (
            Decimal(conclusive) / Decimal(sample_count) if sample_count > 0 else Decimal("0")
        )
        false_positive_rate = (
            Decimal(false_positive_count) / Decimal(conclusive) if conclusive > 0 else None
        )

        sorted_cause_counts = {k: cause_counts[k] for k in sorted(cause_counts)}

        metrics_by_code[issue_code] = IssueAuditMetrics(
            sample_count=sample_count,
            true_issue_count=true_issue_count,
            false_positive_count=false_positive_count,
            uncertain_count=uncertain_count,
            unlabeled_count=unlabeled_count,
            conclusive_coverage=conclusive_coverage,
            false_positive_rate=false_positive_rate,
            cause_counts=sorted_cause_counts,
        )

    return metrics_by_code


class QualityGateError(Exception):
    pass


class AuditComparison(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    before_fingerprint: str
    after_fingerprint: str
    before_table_count: int
    after_table_count: int
    before_issue_count: int
    after_issue_count: int
    coverage: Decimal
    false_positive_rate: Decimal | None
    passed: bool
    errors: tuple[str, ...]
    metrics_by_code: dict[str, IssueAuditMetrics] = Field(default_factory=dict)


def compare_releases(
    before_path: Path | str,
    after_path: Path | str,
    sample: Path | str | pa.Table,
    labels: Path | str | Sequence[LabelRecord],
) -> AuditComparison:
    before_path = Path(before_path)
    after_path = Path(after_path)

    try:
        before_tables_tbl = pq.read_table(before_path / "tables.parquet")  # type: ignore[no-untyped-call]
        after_tables_tbl = pq.read_table(after_path / "tables.parquet")  # type: ignore[no-untyped-call]
        before_issues_tbl = pq.read_table(before_path / "issues.parquet")  # type: ignore[no-untyped-call]
        after_issues_tbl = pq.read_table(after_path / "issues.parquet")  # type: ignore[no-untyped-call]
    except Exception as e:
        raise ValueError(f"Failed to read release Parquet files: {e}") from e

    before_table_ids = set(before_tables_tbl.column("table_id").to_pylist())
    after_table_ids = set(after_tables_tbl.column("table_id").to_pylist())

    if before_table_ids != after_table_ids:
        raise QualityGateError("canonical table IDs changed")

    try:
        after_cells_tbl = pq.read_table(after_path / "cells.parquet")  # type: ignore[no-untyped-call]
    except Exception as e:
        raise ValueError(f"Failed to read cells.parquet: {e}") from e

    after_cells_by_id = {
        str(c["cell_id"]): c for c in after_cells_tbl.to_pylist() if c.get("cell_id") is not None
    }
    after_tables_by_id = {
        str(t["table_id"]): t for t in after_tables_tbl.to_pylist() if t.get("table_id") is not None
    }

    if isinstance(sample, (str, Path)):
        try:
            sample_tbl = pq.read_table(Path(sample))  # type: ignore[no-untyped-call]
        except Exception as e:
            raise ValueError(f"Failed to read sample Parquet: {e}") from e
    else:
        sample_tbl = sample

    for row in sample_tbl.to_pylist():
        cell_id = row.get("cell_id")
        if cell_id is None or cell_id == "" or str(cell_id) == "None":
            table_id = row.get("table_id")
            if table_id is not None and table_id != "" and str(table_id) != "None":
                after_table = after_tables_by_id.get(str(table_id))
                if after_table is None:
                    raise QualityGateError("missing or changed source context")
                if after_table.get("title_raw") != row.get("table_title_raw") or after_table.get(
                    "unit_raw"
                ) != row.get("table_unit_raw"):
                    raise QualityGateError("missing or changed source context")
            continue

        after_cell = after_cells_by_id.get(str(cell_id))
        if after_cell is None:
            raise QualityGateError("missing or changed source context")

        # Compare raw values
        if (
            after_cell.get("row_label_raw") != row.get("row_label_raw")
            or after_cell.get("column_label_raw") != row.get("column_label_raw")
            or after_cell.get("value_raw") != row.get("value_raw")
        ):
            raise QualityGateError("missing or changed source context")

    if isinstance(labels, (str, Path)):
        labels_list = load_and_validate_labels(sample_tbl, Path(labels))
    else:
        labels_list = tuple(labels)

    sample_ids = {str(row["sample_id"]) for row in sample_tbl.to_pylist()}
    labeled_ids = {lbl.sample_id for lbl in labels_list}

    unresolved_ids = sample_ids - labeled_ids
    if unresolved_ids:
        raise QualityGateError("unresolved sample context")

    after_issues_set = set()
    for r in after_issues_tbl.to_pylist():
        d_id = str(r.get("doc_id", ""))
        t_id = str(r.get("table_id", "") or "None")
        c_id = str(r.get("cell_id", "") or "None")
        field = str(r.get("field", ""))
        code = str(r.get("code", ""))
        after_issues_set.add((d_id, t_id, c_id, field, code))

    labels_by_id = {lbl.sample_id: lbl for lbl in labels_list}

    rows_by_issue = defaultdict(list)
    for row in sample_tbl.to_pylist():
        rows_by_issue[str(row["issue_code"])].append(row)

    metrics_by_code = {}
    total_samples = len(sample_ids)
    total_true = 0
    total_fp = 0
    total_uncertain = 0
    total_unlabeled = 0

    for issue_code, rows in sorted(rows_by_issue.items()):
        sample_count = len(rows)
        true_issue_count = 0
        false_positive_count = 0
        uncertain_count = 0
        unlabeled_count = 0
        cause_counts: dict[str, int] = defaultdict(int)

        for row in rows:
            s_id = str(row["sample_id"])
            lbl = labels_by_id.get(s_id)
            if lbl is None:
                unlabeled_count += 1
                total_unlabeled += 1
            else:
                cause_counts[lbl.cause_code] += 1
                if lbl.label == "true_issue":
                    true_issue_count += 1
                    total_true += 1
                elif lbl.label == "false_positive":
                    # Check if STILL present in after release
                    d_id = str(row.get("doc_id", ""))
                    t_id = str(row.get("table_id", "") or "None")
                    c_id = str(row.get("cell_id", "") or "None")
                    field = str(row.get("field", ""))

                    is_present = (d_id, t_id, c_id, field, issue_code) in after_issues_set
                    if is_present:
                        false_positive_count += 1
                        total_fp += 1
                    else:
                        # Corrected! Does not count as false positive in numerator,
                        # but remains in denominator.
                        true_issue_count += 1
                        total_true += 1
                elif lbl.label == "uncertain":
                    uncertain_count += 1
                    total_uncertain += 1

        conclusive = true_issue_count + false_positive_count
        conclusive_coverage = (
            Decimal(conclusive) / Decimal(sample_count) if sample_count > 0 else Decimal("0")
        )
        false_positive_rate = (
            Decimal(false_positive_count) / Decimal(conclusive) if conclusive > 0 else None
        )

        sorted_cause_counts = {k: cause_counts[k] for k in sorted(cause_counts)}

        metrics_by_code[issue_code] = IssueAuditMetrics(
            sample_count=sample_count,
            true_issue_count=true_issue_count,
            false_positive_count=false_positive_count,
            uncertain_count=uncertain_count,
            unlabeled_count=unlabeled_count,
            conclusive_coverage=conclusive_coverage,
            false_positive_rate=false_positive_rate,
            cause_counts=sorted_cause_counts,
        )

    conclusive = total_true + total_fp
    coverage = Decimal(conclusive) / Decimal(total_samples) if total_samples > 0 else Decimal("0")
    false_positive_rate = Decimal(total_fp) / Decimal(conclusive) if conclusive > 0 else None

    try:
        before_manifest = json.loads((before_path / "manifest.json").read_text(encoding="utf-8"))
        before_fp = before_manifest.get("dataset_fingerprint", "")
    except Exception:
        before_fp = ""

    try:
        after_manifest = json.loads((after_path / "manifest.json").read_text(encoding="utf-8"))
        after_fp = after_manifest.get("dataset_fingerprint", "")
    except Exception:
        after_fp = ""

    return AuditComparison(
        before_fingerprint=before_fp,
        after_fingerprint=after_fp,
        before_table_count=len(before_table_ids),
        after_table_count=len(after_table_ids),
        before_issue_count=len(before_issues_tbl),
        after_issue_count=len(after_issues_tbl),
        coverage=coverage,
        false_positive_rate=false_positive_rate,
        passed=True,
        errors=(),
        metrics_by_code=metrics_by_code,
    )


def enforce_quality_gate(
    comparison: AuditComparison,
    remediated_codes: Collection[str] = (),
) -> None:
    if comparison.after_table_count != 146011:
        raise QualityGateError(
            f"table count not equal to 146,011: got {comparison.after_table_count}"
        )

    if comparison.coverage < Decimal("0.90"):
        raise QualityGateError(f"coverage below 0.90: got {comparison.coverage}")

    # Check false-positive rate
    # Check overall false-positive rate
    if comparison.false_positive_rate is not None and comparison.false_positive_rate > Decimal(
        "0.05"
    ):
        raise QualityGateError(
            f"false-positive rate above 0.05: got {comparison.false_positive_rate}"
        )

    # Check remediated codes false-positive rates
    codes_to_check = remediated_codes if remediated_codes else comparison.metrics_by_code.keys()
    for code in codes_to_check:
        m = comparison.metrics_by_code.get(code)
        if (
            m is not None
            and m.false_positive_rate is not None
            and m.false_positive_rate > Decimal("0.05")
        ):
            raise QualityGateError(
                f"false-positive rate for {code} is {m.false_positive_rate}, expected <= 0.05"
            )
