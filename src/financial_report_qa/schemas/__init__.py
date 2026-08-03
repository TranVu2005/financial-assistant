"""Stable Pydantic contracts shared across product modules."""

from financial_report_qa.schemas.documents import DocumentRecord, stable_document_id
from financial_report_qa.schemas.tables import CellRecord, TableRecord, stable_table_id

__all__ = (
    "CellRecord",
    "DocumentRecord",
    "TableRecord",
    "stable_document_id",
    "stable_table_id",
)
