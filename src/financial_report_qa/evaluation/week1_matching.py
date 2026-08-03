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
    # Group extracted candidate tables by doc_id
    extracted_by_doc: dict[str, list[TableRecord]] = {}
    for tbl in extracted_tables:
        extracted_by_doc.setdefault(tbl.doc_id, []).append(tbl)

    # Group expected tables by doc_id
    expected_by_doc: dict[str, list[ExpectedTable]] = {}
    for exp in expected_tables:
        expected_by_doc.setdefault(exp.doc_id, []).append(exp)

    matched_extracted: dict[str, TableRecord] = {}

    for doc_id, doc_expected in expected_by_doc.items():
        candidates = extracted_by_doc.get(doc_id, [])
        if not candidates:
            continue

        pairs: list[tuple[tuple[int, float, int], str, str, ExpectedTable, TableRecord]] = []
        for exp in doc_expected:
            exp_span = exp.line_end - exp.line_start + 1
            for c in candidates:
                overlap_num = max(
                    0, min(c.line_end, exp.line_end) - max(c.line_start, exp.line_start) + 1
                )
                if overlap_num > 0:
                    is_exact = int(
                        c.line_start == exp.line_start and c.line_end == exp.line_end
                    )
                    overlap_ratio = overlap_num / exp_span
                    dist = -abs(c.line_start - exp.line_start) - abs(c.line_end - exp.line_end)
                    score = (is_exact, overlap_ratio, dist)
                    pairs.append((score, c.table_id, exp.annotation_id, exp, c))

        # Sort descending by score, tie-break ascending by table_id and annotation_id
        pairs.sort(key=lambda p: (-p[0][0], -p[0][1], -p[0][2], p[1], p[2]))

        assigned_exp: set[str] = set()
        assigned_cand: set[str] = set()

        for _, _, _, exp, c in pairs:
            if exp.annotation_id not in assigned_exp and c.table_id not in assigned_cand:
                assigned_exp.add(exp.annotation_id)
                assigned_cand.add(c.table_id)
                matched_extracted[exp.annotation_id] = c

    # Build final assessments in order of expected_tables input
    assessments: list[TableAssessment] = []
    for exp in expected_tables:
        matched_candidate = matched_extracted.get(exp.annotation_id)

        if matched_candidate is not None:
            matched_id = matched_candidate.table_id
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
