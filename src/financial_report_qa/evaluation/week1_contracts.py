"""Week 1 quality gate contracts, validation, stable IDs, and CSV/JSON I/O."""

import csv
import hashlib
import os
from collections.abc import Iterable, Sequence
from io import StringIO
from pathlib import Path
from typing import Any, Literal

import orjson
from pydantic import BaseModel, ConfigDict, Field, field_validator

from financial_report_qa.core.errors import Week1GateInputError

SAMPLING_VERSION = "week1-pilot-v1"
ANNOTATION_SCHEMA_VERSION = "1"

StatementType = Literal[
    "balance_sheet", "income_statement", "cash_flow_statement"
]

GateFailureCode = Literal[
    "missing_table",
    "span_mismatch",
    "shape_mismatch",
    "statement_mismatch",
    "unit_mismatch",
    "period_mismatch",
    "no_numeric_value",
    "invalid_provenance",
    "manual_provenance_failure",
    "unclosed_html_table",
    "nested_html_table",
    "unsupported_html_structure",
    "invalid_span_value",
    "span_collision",
    "expansion_limit_exceeded",
    "ragged_structured_rows",
    "insufficient_structural_evidence",
    "empty_extracted_table",
    "company_conflict",
    "period_incomplete",
    "period_ambiguous",
    "period_invalid",
    "statement_conflict",
    "metric_unknown",
    "number_missing",
    "number_ambiguous",
    "number_invalid",
    "unit_unknown",
    "unit_conflict",
]

PILOT_DOCUMENT_COLUMNS = (
    "annotation_schema_version",
    "dataset_fingerprint",
    "source_manifest_sha256",
    "doc_id",
    "relative_path",
    "company_code",
    "report_year",
    "statement_scope",
)

EXPECTED_TABLE_COLUMNS = (
    "annotation_schema_version",
    "annotation_id",
    "doc_id",
    "relative_path",
    "statement_type",
    "line_start",
    "line_end",
    "row_count",
    "column_count",
    "unit_normalized",
    "expected_periods",
    "notes",
)

CELL_AUDIT_COLUMNS = (
    "annotation_schema_version",
    "sampling_version",
    "cell_id",
    "doc_id",
    "relative_path",
    "company_code",
    "report_year",
    "annotation_id",
    "statement_type",
    "table_id",
    "row_idx",
    "col_idx",
    "row_label_raw",
    "column_label_raw",
    "value_raw",
    "value_numeric",
    "period",
    "unit",
    "source_line_start",
    "source_line_end",
    "source_excerpt",
    "verified",
    "review_notes",
)

PARETO_CSV_COLUMNS = (
    "rank",
    "code",
    "count",
    "share",
    "cumulative_share",
)


def stable_annotation_id(
    doc_id: str,
    line_start: int,
    line_end: int,
    statement_type: StatementType,
) -> str:
    payload = f"{doc_id}\n{line_start}\n{line_end}\n{statement_type}".encode()
    return f"ann_{hashlib.sha256(payload).hexdigest()}"


class PilotDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    annotation_schema_version: Literal["1"]
    dataset_fingerprint: str
    source_manifest_sha256: str
    doc_id: str
    relative_path: str
    company_code: str
    report_year: int
    statement_scope: str

    @field_validator("relative_path")
    @classmethod
    def validate_relative_path(cls, v: str) -> str:
        if not v or v.startswith("/") or v.startswith("\\") or ".." in v.split("/"):
            raise ValueError("relative_path must be a safe POSIX path")
        return v


class PilotMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    annotation_schema_version: Literal["1"]
    sampling_version: str
    dataset_fingerprint: str
    source_manifest_sha256: str
    document_count: int
    company_count: int
    pilot_documents_sha256: str


class ExpectedTable(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    annotation_schema_version: Literal["1"]
    annotation_id: str
    doc_id: str
    relative_path: str
    statement_type: StatementType
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    row_count: int = Field(ge=1)
    column_count: int = Field(ge=1)
    unit_normalized: str
    expected_periods: tuple[str, ...]
    notes: str = ""

    @field_validator("line_end")
    @classmethod
    def validate_line_range(cls, v: int, info: Any) -> int:
        if "line_start" in info.data and v < info.data["line_start"]:
            raise ValueError("line_end must be greater than or equal to line_start")
        return v

    @field_validator("expected_periods")
    @classmethod
    def validate_periods(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        if not v:
            raise ValueError("expected_periods cannot be empty")
        if list(v) != sorted(set(v)):
            raise ValueError("expected_periods must be sorted and duplicate-free")
        return v


class CellAudit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    annotation_schema_version: Literal["1"]
    sampling_version: str
    cell_id: str
    doc_id: str
    relative_path: str
    company_code: str
    report_year: int
    annotation_id: str
    statement_type: StatementType
    table_id: str
    row_idx: int = Field(ge=0)
    col_idx: int = Field(ge=0)
    row_label_raw: str
    column_label_raw: str
    value_raw: str
    value_numeric: float | None
    period: str
    unit: str
    source_line_start: int = Field(ge=1)
    source_line_end: int = Field(ge=1)
    source_excerpt: str
    verified: bool | None
    review_notes: str = ""


class FailureEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    code: GateFailureCode
    doc_id: str
    annotation_id: str | None = None
    table_id: str | None = None
    cell_id: str | None = None


class TableAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    annotation: ExpectedTable
    table_id: str | None
    overlap_numerator: int
    overlap_denominator: int
    failures: tuple[FailureEvent, ...]
    usable: bool


class GateCheck(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    passed: bool
    numerator: int
    denominator: int
    threshold_percent: int


class ParetoRow(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    rank: int
    code: GateFailureCode
    count: int
    share: str
    cumulative_share: str


class GateResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sampling_version: str
    annotation_schema_version: Literal["1"]
    dataset_fingerprint: str
    source_manifest_sha256: str
    pilot_documents_sha256: str
    expected_tables_sha256: str
    cell_audit_sha256: str
    evaluation_inputs_sha256: str
    document_count: int
    annotated_table_count: int
    matched_table_count: int
    usable_table_count: int
    checks: tuple[GateCheck, ...]
    statement_metrics: dict[str, dict[str, Any]]
    stratum_metrics: dict[str, dict[str, Any]]
    pareto_rows: tuple[ParetoRow, ...]
    passed: bool


def read_csv_rows(
    path: Path, expected_columns: Sequence[str]
) -> tuple[dict[str, str], ...]:
    if not path.is_file():
        raise Week1GateInputError(f"CSV file not found: {path.name}")
    raw_bytes = path.read_bytes()
    try:
        raw_text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as e:
        raise Week1GateInputError(f"CSV file {path.name} is not valid UTF-8: {e}") from e

    if "\r" in raw_text:
        raise Week1GateInputError(f"CSV file {path.name} contains invalid CRLF/CR line endings")
    if raw_text and not raw_text.endswith("\n"):
        raise Week1GateInputError(f"CSV file {path.name} missing trailing LF newline")

    stream = StringIO(raw_text)
    reader = csv.DictReader(stream, lineterminator="\n")
    if reader.fieldnames is None:
        raise Week1GateInputError(f"CSV file {path.name} is empty")
    if tuple(reader.fieldnames) != tuple(expected_columns):
        raise Week1GateInputError(
            f"CSV file {path.name} header mismatch: "
            f"expected {expected_columns}, got {reader.fieldnames}"
        )

    rows: list[dict[str, str]] = []
    for row in reader:
        rows.append(dict(row))
    return tuple(rows)


def write_csv_rows(
    path: Path,
    expected_columns: Sequence[str],
    rows: Iterable[dict[str, str]],
    *,
    allow_identical: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = StringIO()
    writer = csv.DictWriter(
        stream,
        fieldnames=list(expected_columns),
        lineterminator="\n",
        extrasaction="raise",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    content_bytes = stream.getvalue().encode("utf-8")

    if path.exists():
        if allow_identical and path.read_bytes() == content_bytes:
            return
        if not allow_identical:
            raise Week1GateInputError(f"Refusing to overwrite existing file: {path.name}")

    tmp_path = path.parent / f".tmp_{path.name}"
    try:
        with open(tmp_path, "wb") as f:
            f.write(content_bytes)
            f.flush()
            os.fsync(f.fileno())
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def write_canonical_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content_bytes = orjson.dumps(
        payload, option=orjson.OPT_SORT_KEYS | orjson.OPT_APPEND_NEWLINE
    )
    tmp_path = path.parent / f".tmp_{path.name}"
    try:
        with open(tmp_path, "wb") as f:
            f.write(content_bytes)
            f.flush()
            os.fsync(f.fileno())
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
