"""Table matching and completeness evaluation logic for Week 1 Quality Gate."""

import hashlib

from financial_report_qa.evaluation.week1_contracts import (
    ExpectedTable,
    FailureEvent,
    TableAssessment,
)
from financial_report_qa.schemas import TableRecord


def derive_table_match_key(
    doc_id: str,
    statement_type: str | None,
    line_start: int,
    line_end: int,
) -> str:
    """Return a stable candidate match key for an extracted table."""
    norm_scope = (statement_type or "unknown").strip().lower()
    payload = f"{doc_id}\n{norm_scope}\n{line_start}\n{line_end}".encode()
    return f"tblmatch_{hashlib.sha256(payload).hexdigest()}"


def assess_table_matching(
    expected_tables: tuple[ExpectedTable, ...],
    extracted_tables: tuple[TableRecord, ...],
) -> tuple[tuple[TableAssessment, ...], dict[str, TableRecord]]:
    """Match expected annotated tables against extracted candidates and produce assessments."""
    {tbl.table_id: tbl for tbl in extracted_tables}

    # Group extracted candidate tables by doc_id
    extracted_by_doc: dict[str, list[TableRecord]] = {}
    for tbl in extracted_tables:
        extracted_by_doc.setdefault(tbl.doc_id, []).append(tbl)

    assessments: list[TableAssessment] = []
    matched_extracted: dict[str, TableRecord] = {}

    for exp in expected_tables:
        candidates = extracted_by_doc.get(exp.doc_id, [])

        # Priority 1: Exact span match (line_start and line_end match)
        exact_matches = [
            c for c in candidates if c.line_start == exp.line_start and c.line_end == exp.line_end
        ]

        matched_candidate: TableRecord | None = None
        if len(exact_matches) == 1:
            matched_candidate = exact_matches[0]
        elif len(exact_matches) > 1:
            # Tie-break by statement_type match if exact span has multiple candidates
            type_matches = [
                c
                for c in exact_matches
                if (c.statement_type or "").lower() == exp.statement_type.lower()
            ]
            if len(type_matches) == 1:
                matched_candidate = type_matches[0]
            else:
                exact_matches.sort(key=lambda c: c.table_id)
                matched_candidate = exact_matches[0]

        # Priority 2: Overlapping span match (if no exact span match)
        if matched_candidate is None:
            overlaps = [
                c
                for c in candidates
                if max(c.line_start, exp.line_start) <= min(c.line_end, exp.line_end)
            ]
            if len(overlaps) == 1:
                matched_candidate = overlaps[0]
            elif len(overlaps) > 1:
                # Pick maximum overlap length
                def overlap_len(c: TableRecord) -> int:
                    return min(c.line_end, exp.line_end) - max(c.line_start, exp.line_start) + 1

                overlaps.sort(key=lambda c: (-overlap_len(c), c.table_id))
                matched_candidate = overlaps[0]

        if matched_candidate is not None:
            matched_id = matched_candidate.table_id
            matched_extracted[exp.annotation_id] = matched_candidate

            # Compute overlap
            overlap_num = max(
                0,
                min(matched_candidate.line_end, exp.line_end)
                - max(matched_candidate.line_start, exp.line_start)
                + 1,
            )
            overlap_den = (
                max(matched_candidate.line_end, exp.line_end)
                - min(matched_candidate.line_start, exp.line_start)
                + 1
            )

            failures: list[FailureEvent] = []
            if (
                matched_candidate.line_start != exp.line_start
                or matched_candidate.line_end != exp.line_end
            ):
                failures.append(
                    FailureEvent(
                        code="span_mismatch",
                        doc_id=exp.doc_id,
                        annotation_id=exp.annotation_id,
                        table_id=matched_id,
                    )
                )

            assessments.append(
                TableAssessment(
                    annotation=exp,
                    table_id=matched_id,
                    overlap_numerator=overlap_num,
                    overlap_denominator=overlap_den,
                    failures=tuple(failures),
                    usable=len(failures) == 0,
                )
            )
        else:
            assessments.append(
                TableAssessment(
                    annotation=exp,
                    table_id=None,
                    overlap_numerator=0,
                    overlap_denominator=exp.line_end - exp.line_start + 1,
                    failures=(
                        FailureEvent(
                            code="missing_table",
                            doc_id=exp.doc_id,
                            annotation_id=exp.annotation_id,
                        ),
                    ),
                    usable=False,
                )
            )

    assessments.sort(key=lambda a: a.annotation.annotation_id)
    return tuple(assessments), matched_extracted
