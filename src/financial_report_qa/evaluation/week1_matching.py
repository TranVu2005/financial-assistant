"""Table matching and completeness evaluation logic for Week 1 Quality Gate."""

import hashlib
from fractions import Fraction

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
        candidates = sorted(extracted_by_doc.get(doc_id, []), key=lambda c: c.table_id)
        if not candidates:
            continue

        ordered_expected = sorted(doc_expected, key=lambda exp: exp.annotation_id)
        pair_scores: dict[tuple[str, str], tuple[int, Fraction, int]] = {}
        for exp in doc_expected:
            exp_span = exp.line_end - exp.line_start + 1
            for c in candidates:
                overlap_num = max(
                    0, min(c.line_end, exp.line_end) - max(c.line_start, exp.line_start) + 1
                )
                if overlap_num > 0:
                    is_exact = int(c.line_start == exp.line_start and c.line_end == exp.line_end)
                    overlap_ratio = Fraction(overlap_num, exp_span)
                    dist = -abs(c.line_start - exp.line_start) - abs(c.line_end - exp.line_end)
                    pair_scores[(exp.annotation_id, c.table_id)] = (
                        is_exact,
                        overlap_ratio,
                        dist,
                    )

        best_score: tuple[int, Fraction, int] | None = None
        best_table_ids: tuple[str, ...] | None = None
        best_assignment: dict[str, TableRecord] = {}

        def search(
            index: int,
            used_table_ids: set[str],
            assignment: dict[str, TableRecord],
            total_score: tuple[int, Fraction, int],
        ) -> None:
            nonlocal best_assignment, best_score, best_table_ids

            if index == len(ordered_expected):
                table_ids = tuple(
                    assignment[exp.annotation_id].table_id
                    for exp in ordered_expected
                    if exp.annotation_id in assignment
                )
                if (
                    best_score is None
                    or total_score > best_score
                    or (total_score == best_score and table_ids < (best_table_ids or ()))
                ):
                    best_score = total_score
                    best_table_ids = table_ids
                    best_assignment = dict(assignment)
                return

            exp = ordered_expected[index]
            search(index + 1, used_table_ids, assignment, total_score)

            for candidate in candidates:
                if candidate.table_id in used_table_ids:
                    continue
                score = pair_scores.get((exp.annotation_id, candidate.table_id))
                if score is None:
                    continue
                used_table_ids.add(candidate.table_id)
                assignment[exp.annotation_id] = candidate
                search(
                    index + 1,
                    used_table_ids,
                    assignment,
                    (
                        total_score[0] + score[0],
                        total_score[1] + score[1],
                        total_score[2] + score[2],
                    ),
                )
                assignment.pop(exp.annotation_id)
                used_table_ids.remove(candidate.table_id)

        search(0, set(), {}, (0, Fraction(0, 1), 0))
        matched_extracted.update(best_assignment)

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
            overlap_den = exp.line_end - exp.line_start + 1

            failures: list[FailureEvent] = []
            if overlap_num * 100 < overlap_den * 80:
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
