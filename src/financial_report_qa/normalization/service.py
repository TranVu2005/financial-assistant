import hashlib
from typing import Any

import orjson

from financial_report_qa.core.errors import NormalizationError
from financial_report_qa.ingestion.provenance import ExtractedTable, ExtractionResult
from financial_report_qa.normalization._shared import (
    RULESET_VERSION,
    issue_sort_key,
    normalized_key,
)
from financial_report_qa.normalization.companies import normalize_company
from financial_report_qa.normalization.metrics import normalize_metric
from financial_report_qa.normalization.numbers import (
    NumberContext,
    is_numeric_candidate,
    parse_number,
)
from financial_report_qa.normalization.periods import has_period_evidence, normalize_period
from financial_report_qa.normalization.statements import normalize_statement_type
from financial_report_qa.normalization.units import (
    has_unit_evidence,
    is_monetary_unit,
    normalize_unit,
    resolve_unit,
)
from financial_report_qa.schemas.documents import DocumentRecord
from financial_report_qa.schemas.normalization import (
    NormalizationField,
    NormalizationIssue,
    NormalizationIssueCode,
    NormalizedDocument,
)
from financial_report_qa.schemas.tables import CellRecord


def _issue(
    *,
    code: NormalizationIssueCode,
    field: NormalizationField,
    document: DocumentRecord,
    table_id: str | None,
    cell_id: str | None,
    raw_value: str | None,
) -> NormalizationIssue:
    return NormalizationIssue(
        code=code,
        field=field,
        doc_id=document.doc_id,
        table_id=table_id,
        cell_id=cell_id,
        raw_value=raw_value,
    )


def normalize_extraction(document: DocumentRecord, result: ExtractionResult) -> NormalizedDocument:
    if document.doc_id != result.doc_id:
        raise NormalizationError("document and extraction IDs must match")

    issues: list[NormalizationIssue] = []
    normalized_tables: list[ExtractedTable] = []

    for extracted in result.tables:
        table_rec = extracted.table
        table_id = table_rec.table_id

        # Company normalization
        company_dec = normalize_company(document, table_rec.title_raw)
        if company_dec.issue_code is not None:
            issues.append(
                _issue(
                    code=company_dec.issue_code,
                    field="company",
                    document=document,
                    table_id=table_id,
                    cell_id=None,
                    raw_value=table_rec.title_raw,
                )
            )

        # Statement type normalization
        stmt_dec = normalize_statement_type(table_rec.title_raw)
        if stmt_dec.issue_code is not None:
            issues.append(
                _issue(
                    code=stmt_dec.issue_code,
                    field="statement_type",
                    document=document,
                    table_id=table_id,
                    cell_id=None,
                    raw_value=table_rec.title_raw,
                )
            )

        # Table unit normalization
        tbl_unit_dec = normalize_unit(table_rec.unit_raw)
        if tbl_unit_dec.issue_code is not None and has_unit_evidence(table_rec.unit_raw):
            issues.append(
                _issue(
                    code=tbl_unit_dec.issue_code,
                    field="unit",
                    document=document,
                    table_id=table_id,
                    cell_id=None,
                    raw_value=table_rec.unit_raw,
                )
            )

        # Group cells by row_idx to normalize row labels (metrics)
        rows: dict[int, list[CellRecord]] = {}
        for cell in extracted.cells:
            rows.setdefault(cell.row_idx, []).append(cell)

        row_metric_decisions: dict[int, tuple[str | None, NormalizationIssueCode | None]] = {}
        for row_idx, cell_list in rows.items():
            first_label = cell_list[0].row_label_raw
            if first_label is not None:
                m_dec = normalize_metric(
                    first_label, group_context=cell_list[0].row_group_context_raw
                )
                row_metric_decisions[row_idx] = (m_dec.value, m_dec.issue_code)
                if m_dec.issue_code is not None:
                    row_has_numeric_value = any(
                        candidate.value_raw is not None
                        and candidate.value_raw != candidate.row_label_raw
                        and is_numeric_candidate(candidate.value_raw)
                        for candidate in cell_list
                    )
                    metric_issue_is_actionable = m_dec.issue_code != "metric_unknown" or (
                        stmt_dec.value
                        in {"income_statement", "balance_sheet", "cash_flow_statement"}
                        and row_has_numeric_value
                    )
                    if metric_issue_is_actionable:
                        # Emit one metric issue per logical row for lowest (col_idx, cell_id)
                        target_cell = min(cell_list, key=lambda c: (c.col_idx, c.cell_id))
                        issues.append(
                            _issue(
                                code=m_dec.issue_code,
                                field="metric",
                                document=document,
                                table_id=table_id,
                                cell_id=target_cell.cell_id,
                                raw_value=first_label,
                            )
                        )
            else:
                row_metric_decisions[row_idx] = (None, None)

        # Cache period decisions per column label
        col_period_decisions: dict[str, tuple[str | None, NormalizationIssueCode | None]] = {}
        for cell in extracted.cells:
            if cell.column_label_raw and cell.column_label_raw not in col_period_decisions:
                if has_period_evidence(cell.column_label_raw):
                    p_dec = normalize_period(cell.column_label_raw, document.report_year)
                    col_period_decisions[cell.column_label_raw] = (p_dec.value, p_dec.issue_code)
                    if p_dec.issue_code is not None:
                        issues.append(
                            _issue(
                                code=p_dec.issue_code,
                                field="period",
                                document=document,
                                table_id=table_id,
                                cell_id=cell.cell_id,
                                raw_value=cell.column_label_raw,
                            )
                        )
                else:
                    col_period_decisions[cell.column_label_raw] = (None, None)

        # Cache column-level unit evidence once. Cell hints remain more specific
        # and are resolved per cell below.
        col_unit_decisions = {}
        for cell in extracted.cells:
            raw_column = cell.column_label_raw
            if raw_column and raw_column not in col_unit_decisions:
                if has_unit_evidence(raw_column):
                    col_unit_decisions[raw_column] = resolve_unit(
                        cell_hint=None,
                        column_raw=raw_column,
                        table_raw=table_rec.unit_raw,
                    )

        # Process cells
        normalized_cells: list[CellRecord] = []
        for cell in extracted.cells:
            metric_val, _ = row_metric_decisions.get(cell.row_idx, (None, None))
            p_val, p_issue = col_period_decisions.get(cell.column_label_raw or "", (None, None))

            # Determine if cell is a value candidate
            is_value_candidate = (
                cell.row_label_raw is not None
                and cell.value_raw != cell.row_label_raw
                and (
                    not cell.column_label_raw
                    or normalized_key(cell.column_label_raw)
                    not in {"stt", "mã số", "thuyết minh", "code", "note", "notes"}
                )
            )

            val_num = None
            unit_val = None

            if is_value_candidate and cell.value_raw is not None:
                has_unit_ctx = (
                    has_unit_evidence(cell.value_raw)
                    or has_unit_evidence(cell.column_label_raw)
                    or has_unit_evidence(table_rec.unit_raw)
                )
                provisional_unit_dec = None
                if has_unit_ctx:
                    provisional_unit_dec = col_unit_decisions.get(cell.column_label_raw or "")
                    if provisional_unit_dec is None:
                        provisional_unit_dec = resolve_unit(
                            cell_hint=None,
                            column_raw=None,
                            table_raw=table_rec.unit_raw,
                        )
                number_context: NumberContext = "unknown"
                if provisional_unit_dec is not None:
                    if provisional_unit_dec.value == "percent":
                        number_context = "percent"
                    elif is_monetary_unit(provisional_unit_dec.value):
                        number_context = "monetary"

                num_dec = None
                if is_numeric_candidate(cell.value_raw):
                    num_dec = parse_number(cell.value_raw, context=number_context)
                    if num_dec.value is not None:
                        val_num = num_dec.value
                    elif num_dec.issue_code is not None:
                        if num_dec.issue_code == "number_missing" and metric_val is None:
                            pass
                        else:
                            issues.append(
                                _issue(
                                    code=num_dec.issue_code,
                                    field="number",
                                    document=document,
                                    table_id=table_id,
                                    cell_id=cell.cell_id,
                                    raw_value=cell.value_raw,
                                )
                            )

                cell_hint = num_dec.unit_hint if num_dec is not None else None
                has_unit_ctx = cell_hint is not None or has_unit_ctx
                if has_unit_ctx:
                    if cell_hint is not None:
                        unit_dec = resolve_unit(
                            cell_hint=cell_hint,
                            column_raw=cell.column_label_raw,
                            table_raw=table_rec.unit_raw,
                        )
                    else:
                        unit_dec = col_unit_decisions.get(
                            cell.column_label_raw or "",
                            resolve_unit(
                                cell_hint=None,
                                column_raw=None,
                                table_raw=table_rec.unit_raw,
                            ),
                        )
                    if unit_dec.value is not None:
                        unit_val = unit_dec.value
                    elif unit_dec.issue_code is not None and (
                        num_dec is not None
                        and num_dec.value is not None
                        and (
                            cell_hint is not None
                            or has_unit_evidence(cell.column_label_raw)
                            or not has_unit_evidence(table_rec.unit_raw)
                        )
                    ):
                        issues.append(
                            _issue(
                                code=unit_dec.issue_code,
                                field="unit",
                                document=document,
                                table_id=table_id,
                                cell_id=cell.cell_id,
                                raw_value=cell.value_raw,
                            )
                        )

            updated_cell = cell.model_copy(
                update={
                    "row_label_canonical": metric_val,
                    "column_label_canonical": p_val,
                    "value_numeric": val_num,
                    "period": p_val,
                    "unit": unit_val,
                }
            )
            normalized_cells.append(updated_cell)

        stmt_val = stmt_dec.value
        if stmt_val is None:
            import unicodedata

            def clean_label(s: str | None) -> str:
                if not s:
                    return ""
                return " ".join(unicodedata.normalize("NFKC", s).lower().split())

            labels = {clean_label(c.row_label_raw) for c in normalized_cells if c.row_label_raw}

            if any(
                "lưu chuyển tiền" in lbl
                or "net cash flows" in lbl
                or "khấu hao" in lbl
                or "depreciation" in lbl
                for lbl in labels
            ):
                stmt_val = "cash_flow_statement"
            elif any(
                "lợi nhuận gộp" in lbl
                or "lợi nhuận trước thuế" in lbl
                or "lợi nhuận sau thuế" in lbl
                or "doanh thu thuần" in lbl
                or "thu nhập lãi thuần" in lbl
                or "doanh thu bán hàng" in lbl
                for lbl in labels
            ):
                stmt_val = "income_statement"
            elif any(
                "tài sản ngắn hạn" in lbl
                or "tài sản dài hạn" in lbl
                or "nợ phải trả" in lbl
                or "vốn chủ sở hữu" in lbl
                or "tổng cộng tài sản" in lbl
                or "tổng tài sản" in lbl
                or "total assets" in lbl
                for lbl in labels
            ):
                stmt_val = "balance_sheet"

        updated_table_rec = table_rec.model_copy(
            update={
                "statement_type": stmt_val,
                "unit_normalized": tbl_unit_dec.value,
            }
        )
        updated_extracted = ExtractedTable(
            table=updated_table_rec,
            cells=tuple(normalized_cells),
            placements=extracted.placements,
            evidence=extracted.evidence,
        )
        normalized_tables.append(updated_extracted)

    # De-duplicate issues and sort
    unique_issues_dict: dict[tuple[Any, ...], NormalizationIssue] = {}
    for issue in issues:
        key = (
            issue.doc_id,
            issue.table_id,
            issue.cell_id,
            issue.field,
            issue.code,
            issue.raw_value,
        )
        if key not in unique_issues_dict:
            unique_issues_dict[key] = issue

    sorted_issues = tuple(sorted(unique_issues_dict.values(), key=issue_sort_key))

    normalized_extraction = ExtractionResult(
        doc_id=document.doc_id,
        blocks=result.blocks,
        tables=tuple(normalized_tables),
        rejected=result.rejected,
    )

    payload = {
        "doc_id": document.doc_id,
        "extraction": normalized_extraction.model_dump(mode="json"),
        "issues": [issue.model_dump(mode="json") for issue in sorted_issues],
        "ruleset_version": RULESET_VERSION,
    }
    fingerprint = hashlib.sha256(orjson.dumps(payload, option=orjson.OPT_SORT_KEYS)).hexdigest()

    return NormalizedDocument(
        document=document,
        extraction=normalized_extraction,
        issues=sorted_issues,
        ruleset_version=RULESET_VERSION,
        normalization_fingerprint=fingerprint,
    )
