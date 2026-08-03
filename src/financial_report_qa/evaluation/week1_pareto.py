"""Pareto analysis calculation and ranking logic for Week 1 Quality Gate failures."""

from collections import Counter
from decimal import ROUND_HALF_UP, Decimal

from financial_report_qa.evaluation.week1_contracts import (
    GateFailureCode,
    ParetoRow,
    TableAssessment,
)


def compute_pareto_analysis(
    assessments: tuple[TableAssessment, ...],
) -> tuple[ParetoRow, ...]:
    """Compute deterministic Pareto error distribution across table assessments."""
    counts: Counter[GateFailureCode] = Counter()
    for ta in assessments:
        for f in ta.failures:
            counts[f.code] += 1

    if not counts:
        return ()

    total_failures = sum(counts.values())

    # Sort primarily by count descending, secondarily by code alphabetically for stability
    sorted_failures = sorted(counts.items(), key=lambda item: (-item[1], item[0]))

    rows: list[ParetoRow] = []
    cumulative_count = 0

    for idx, (code, count) in enumerate(sorted_failures, start=1):
        cumulative_count += count

        share_dec = (Decimal(count) * Decimal(100) / Decimal(total_failures)).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
        cum_dec = (Decimal(cumulative_count) * Decimal(100) / Decimal(total_failures)).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

        rows.append(
            ParetoRow(
                rank=idx,
                code=code,
                count=count,
                share=f"{share_dec:.2f}%",
                cumulative_share=f"{cum_dec:.2f}%",
            )
        )

    return tuple(rows)
