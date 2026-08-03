"""Source parsing and provenance-preserving table extraction."""

from financial_report_qa.ingestion.provenance import (
    DecodedDocument,
    DetectionResult,
    ExtractionResult,
    stable_cell_id,
)
from financial_report_qa.ingestion.table_detector import detect_table_candidates
from financial_report_qa.ingestion.table_extractor import extract_candidates, extract_document
from financial_report_qa.ingestion.txt_reader import read_document

__all__ = (
    "DecodedDocument",
    "DetectionResult",
    "ExtractionResult",
    "detect_table_candidates",
    "extract_candidates",
    "extract_document",
    "read_document",
    "stable_cell_id",
)
