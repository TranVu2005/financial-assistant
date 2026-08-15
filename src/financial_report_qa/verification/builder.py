"""Day 20 `build_answer_package` orchestrator (ADR 0009).

Ties `checks.py` + `templates.py` together the way `execution/compiler.py`
ties `locator`/`operations`/`pandas_query` together for Day 18: one function,
never a guessed or half-verified package. Every evidence cell must resolve
to a citation through `citation_lookup` (Day 20 plan Sec 1.6 measured 100%
of evidence cells on gold70 have complete provenance in the release) -- a
missing entry is a hard error, not a placeholder.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from financial_report_qa.execution.contracts import CompiledQuery
from financial_report_qa.planning.plan_contracts import FinancialQueryPlan
from financial_report_qa.verification import checks
from financial_report_qa.verification.contracts import AnswerPackage, Citation, is_blocking_issue
from financial_report_qa.verification.templates import render_answer, render_sentence


def _build_citations(
    compiled: CompiledQuery, citation_lookup: Mapping[str, Mapping[str, object]]
) -> tuple[Citation, ...]:
    citations: list[Citation] = []
    for cell in compiled.evidence:
        for cell_id in cell.cell_ids:
            provenance = citation_lookup.get(cell_id)
            if provenance is None:
                raise ValueError(f"no citation provenance supplied for cell_id {cell_id!r}")
            citations.append(
                Citation.model_validate(
                    {
                        "cell_id": cell_id,
                        "table_id": cell.table_id,
                        "doc_relative_path": provenance["doc_relative_path"],
                        "source_line_start": provenance["source_line_start"],
                        "source_line_end": provenance["source_line_end"],
                        "table_title": provenance["table_title"],
                        "value": cell.value,
                        "unit": cell.unit,
                    }
                )
            )
    return tuple(citations)


def build_answer_package(
    *,
    question_id: str,
    question: str,
    plan: FinancialQueryPlan,
    compiled: CompiledQuery,
    retrieved_table_ids: frozenset[str],
    citation_lookup: Mapping[str, Mapping[str, object]] | None = None,
) -> AnswerPackage:
    """Compile a verified `AnswerPackage` from one answered `CompiledQuery`.

    Raises `ValueError` if `compiled.status != "answered"` -- there is
    nothing to verify or cite for an error result (it is already a typed
    `CompiledQuery` error, handled upstream).
    """
    if compiled.status != "answered":
        raise ValueError("cannot build an AnswerPackage from a non-answered CompiledQuery")
    assert compiled.answer is not None and compiled.unit is not None
    citation_lookup = citation_lookup if citation_lookup is not None else {}

    display, display_precision = render_answer(plan, compiled)
    answer_text = render_sentence(plan, compiled, display)

    issues = [
        issue
        for issue in (
            checks.check_recompute_mismatch(plan, compiled),
            checks.check_unit_not_presentable(plan, compiled),
            checks.check_evidence_outside_retrieval(compiled, retrieved_table_ids),
            checks.check_display_roundtrip_mismatch(
                compiled.answer, display, display_precision=display_precision
            ),
            checks.check_period_inferred_warning(compiled),
            checks.check_scope_inferred(compiled),
        )
        if issue is not None
    ]
    status: Literal["verified", "rejected"] = (
        "rejected" if any(is_blocking_issue(issue.code) for issue in issues) else "verified"
    )

    return AnswerPackage.model_validate(
        {
            "question_id": question_id,
            "question": question,
            "operation": compiled.operation,
            "answer": compiled.answer,
            "unit": compiled.unit,
            "display": display,
            "display_precision": display_precision,
            "answer_text": answer_text,
            "evidence": _build_citations(compiled, citation_lookup),
            "retrieved_table_ids": tuple(sorted(retrieved_table_ids)),
            "pandas_query": compiled.pandas_query,
            "period_inferred": any(cell.period_inferred for cell in compiled.evidence),
            "verification_status": status,
            "verification_issues": tuple(issues),
        }
    )
