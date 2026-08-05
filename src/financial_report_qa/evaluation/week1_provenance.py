"""Provenance audit and cell verification logic for Week 1 Quality Gate."""

import html
import re
from pathlib import Path

from financial_report_qa.evaluation.week1_contracts import (
    SAMPLING_VERSION,
    CellAudit,
    ExpectedTable,
    FailureEvent,
    GateFailureCode,
    TableAssessment,
)
from financial_report_qa.evaluation.week1_dataset import GateDataset
from financial_report_qa.ingestion.provenance import stable_cell_id
from financial_report_qa.schemas import CellRecord, TableRecord

_BR_TAG_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
_NUMERIC_VALUE_RE = re.compile(r"\(?[+-]?[0-9][0-9., ]*%?\)?$")
_AUXILIARY_COLUMN_SIGNALS = ("stt", "số thứ tự", "mã số", "code", "thuyết minh", "note")


def _normalized_text(value: str) -> str:
    return " ".join(value.casefold().split())


def _is_eligible_value_cell(cell: CellAudit) -> bool:
    raw = cell.value_raw.strip()
    if not raw or not cell.row_label_raw.strip() or not cell.column_label_raw.strip():
        return False
    if _normalized_text(raw) == _normalized_text(cell.row_label_raw):
        return False
    column_label = _normalized_text(cell.column_label_raw)
    if any(signal in column_label for signal in _AUXILIARY_COLUMN_SIGNALS):
        return False
    return cell.value_numeric is not None or _NUMERIC_VALUE_RE.fullmatch(raw) is not None


def audit_cell_provenance(
    cell: CellRecord,
    doc_lines: tuple[str, ...],
) -> tuple[bool, str, list[GateFailureCode]]:
    """Audit line provenance and source excerpt for a single extracted cell."""
    failures: list[GateFailureCode] = []

    if cell.source_line_start < 1 or cell.source_line_end < cell.source_line_start:
        failures.append("invalid_provenance")
        return False, "", failures

    if cell.source_line_end > len(doc_lines):
        failures.append("invalid_provenance")
        return False, "", failures

    excerpt_lines = doc_lines[cell.source_line_start - 1 : cell.source_line_end]
    source_excerpt = "\n".join(excerpt_lines)

    # Verification checks
    if cell.value_numeric is None and cell.value_raw.strip() != "":
        # Non-numeric value where raw is present
        pass

    comparable_excerpt = html.unescape(_BR_TAG_RE.sub("\n", source_excerpt))
    if cell.value_raw and cell.value_raw not in comparable_excerpt:
        failures.append("invalid_provenance")

    verified = len(failures) == 0
    return verified, source_excerpt, failures


def generate_cell_audits(
    dataset: GateDataset,
    corpus_dir: Path,
    expected_tables: tuple[ExpectedTable, ...],
    matched_tables: dict[str, TableRecord],
) -> tuple[CellAudit, ...]:
    """Generate CellAudit records for all cells in matched tables."""
    audits: list[CellAudit] = []

    # Cache doc lines
    doc_lines_cache: dict[str, tuple[str, ...]] = {}

    for exp in expected_tables:
        table = matched_tables.get(exp.annotation_id)
        if table is None:
            continue

        doc = dataset.documents_by_id.get(exp.doc_id)
        if doc is None:
            continue

        if exp.doc_id not in doc_lines_cache:
            doc_path = corpus_dir / doc.relative_path
            if doc_path.is_file():
                text = doc_path.read_text(encoding="utf-8", errors="replace")
                doc_lines_cache[exp.doc_id] = tuple(text.splitlines())
            else:
                doc_lines_cache[exp.doc_id] = ()

        lines = doc_lines_cache[exp.doc_id]
        cells = dataset.cells_by_table_id.get(table.table_id, ())
        seen_coordinates: set[tuple[int, int]] = set()

        for cell in cells:
            verified, excerpt, failures = audit_cell_provenance(cell, lines)
            if cell.cell_id != stable_cell_id(cell.table_id, cell.row_idx, cell.col_idx):
                failures.append("invalid_provenance")
            if cell.source_line_start < table.line_start or cell.source_line_end > table.line_end:
                failures.append("invalid_provenance")
            coordinate = (cell.row_idx, cell.col_idx)
            if coordinate in seen_coordinates:
                failures.append("invalid_provenance")
            seen_coordinates.add(coordinate)
            verified = not failures

            audits.append(
                CellAudit(
                    annotation_schema_version="1",
                    sampling_version=SAMPLING_VERSION,
                    cell_id=cell.cell_id,
                    doc_id=doc.doc_id,
                    relative_path=doc.relative_path,
                    company_code=doc.company_code,
                    report_year=doc.report_year,
                    annotation_id=exp.annotation_id,
                    statement_type=exp.statement_type,
                    table_id=table.table_id,
                    row_idx=cell.row_idx,
                    col_idx=cell.col_idx,
                    row_label_raw=cell.row_label_raw or "",
                    column_label_raw=cell.column_label_raw or "",
                    value_raw=cell.value_raw,
                    value_numeric=float(cell.value_numeric)
                    if cell.value_numeric is not None
                    else None,
                    period=cell.period or "",
                    unit=cell.unit or "",
                    source_line_start=cell.source_line_start,
                    source_line_end=cell.source_line_end,
                    source_excerpt=excerpt,
                    verified=verified,
                    review_notes="",
                )
            )

    audits.sort(key=lambda a: (a.annotation_id, a.row_idx, a.col_idx, a.cell_id))
    return tuple(audits)


def evaluate_table_usability(
    assessments: tuple[TableAssessment, ...],
    matched_tables: dict[str, TableRecord],
    cell_audits: tuple[CellAudit, ...],
) -> tuple[TableAssessment, ...]:
    """Evaluate detailed failures for matched tables and produce final assessments."""
    audits_by_ann: dict[str, list[CellAudit]] = {}
    for ca in cell_audits:
        audits_by_ann.setdefault(ca.annotation_id, []).append(ca)

    final_assessments: list[TableAssessment] = []

    for ta in assessments:
        if ta.table_id is None or ta.annotation.annotation_id not in matched_tables:
            final_assessments.append(ta)
            continue

        matched_tbl = matched_tables[ta.annotation.annotation_id]
        failures = list(ta.failures)

        # Check shape mismatch
        if (
            matched_tbl.row_count != ta.annotation.row_count
            or matched_tbl.column_count != ta.annotation.column_count
        ):
            failures.append(
                FailureEvent(
                    code="shape_mismatch",
                    doc_id=ta.annotation.doc_id,
                    annotation_id=ta.annotation.annotation_id,
                    table_id=matched_tbl.table_id,
                )
            )

        # Check statement_type mismatch
        if (matched_tbl.statement_type or "").lower() != ta.annotation.statement_type.lower():
            failures.append(
                FailureEvent(
                    code="statement_mismatch",
                    doc_id=ta.annotation.doc_id,
                    annotation_id=ta.annotation.annotation_id,
                    table_id=matched_tbl.table_id,
                )
            )

        # Check unit mismatch only when the annotation declares an expected unit.
        if ta.annotation.unit_normalized and (
            (matched_tbl.unit_normalized or "").lower() != ta.annotation.unit_normalized.lower()
        ):
            failures.append(
                FailureEvent(
                    code="unit_mismatch",
                    doc_id=ta.annotation.doc_id,
                    annotation_id=ta.annotation.annotation_id,
                    table_id=matched_tbl.table_id,
                )
            )

        # Check cell-level automated provenance and normalization completeness.
        ann_audits = audits_by_ann.get(ta.annotation.annotation_id, [])
        value_cells = [ca for ca in ann_audits if _is_eligible_value_cell(ca)]
        non_numeric_cells = [ca for ca in value_cells if ca.value_numeric is None]
        if non_numeric_cells:
            failures.append(
                FailureEvent(
                    code="no_numeric_value",
                    doc_id=ta.annotation.doc_id,
                    annotation_id=ta.annotation.annotation_id,
                    table_id=matched_tbl.table_id,
                )
            )

        expected_periods = set(ta.annotation.expected_periods)
        period_mismatch_cells = [
            ca
            for ca in value_cells
            if expected_periods and (not ca.period or ca.period not in expected_periods)
        ]
        if period_mismatch_cells:
            failures.append(
                FailureEvent(
                    code="period_mismatch",
                    doc_id=ta.annotation.doc_id,
                    annotation_id=ta.annotation.annotation_id,
                    table_id=matched_tbl.table_id,
                )
            )

        unverified_cells = [ca for ca in ann_audits if ca.verified is False]
        if unverified_cells:
            failures.append(
                FailureEvent(
                    code="invalid_provenance",
                    doc_id=ta.annotation.doc_id,
                    annotation_id=ta.annotation.annotation_id,
                    table_id=matched_tbl.table_id,
                )
            )

        usable = len(failures) == 0
        final_assessments.append(
            TableAssessment(
                annotation=ta.annotation,
                table_id=ta.table_id,
                overlap_numerator=ta.overlap_numerator,
                overlap_denominator=ta.overlap_denominator,
                failures=tuple(failures),
                usable=usable,
            )
        )

    final_assessments.sort(key=lambda a: a.annotation.annotation_id)
    return tuple(final_assessments)