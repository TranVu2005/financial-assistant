"""Stable Pydantic contracts shared across product modules."""

from financial_report_qa.schemas.documents import DocumentRecord, stable_document_id
from financial_report_qa.schemas.tables import CellRecord, TableRecord, stable_table_id

__all__ = (
    "CellRecord",
    "DocumentRecord",
    "NormalizationIssue",
    "NormalizedDocument",
    "TableRecord",
    "stable_document_id",
    "stable_table_id",
)


def __getattr__(name: str) -> object:
    if name in ("NormalizationIssue", "NormalizedDocument"):
        import financial_report_qa.schemas.normalization as norm

        return getattr(norm, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
