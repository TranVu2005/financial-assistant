from collections.abc import Collection
from dataclasses import dataclass

from financial_report_qa.schemas.normalization import NormalizationIssueCode
from financial_report_qa.schemas.tables import CellRecord

BLOCKING_ISSUES = {"unit_conflict", "number_ambiguous", "period_ambiguous"}
MONETARY_UNITS = {"VND", "VND_thousand", "VND_million", "VND_billion"}


@dataclass(frozen=True)
class CellEligibility:
    searchable: bool
    comparable: bool
    calculable: bool
    blocking_reasons: tuple[str, ...]


def classify_cell_eligibility(
    cell: CellRecord, issue_codes: Collection[NormalizationIssueCode]
) -> CellEligibility:
    is_comparable_base = (
        cell.row_label_canonical is not None
        and cell.period is not None
        and cell.value_numeric is not None
    )

    blocking_reasons = tuple(issue for issue in issue_codes if issue in BLOCKING_ISSUES)
    has_blocking = len(blocking_reasons) > 0
    is_monetary = cell.unit in MONETARY_UNITS

    calculable = is_comparable_base and is_monetary and not has_blocking
    comparable = is_comparable_base and not calculable

    has_labels = cell.row_label_canonical is not None and cell.period is not None
    non_empty_raw = bool(cell.value_raw and cell.value_raw.strip())
    searchable = has_labels and non_empty_raw and cell.value_numeric is None

    return CellEligibility(
        searchable=searchable,
        comparable=comparable,
        calculable=calculable,
        blocking_reasons=blocking_reasons,
    )
