import pytest
from pydantic import ValidationError

from financial_report_qa.ingestion.provenance import ExtractionResult
from financial_report_qa.schemas.documents import DocumentRecord, stable_document_id
from financial_report_qa.schemas.normalization import NormalizationIssue, NormalizedDocument


def _document() -> DocumentRecord:
    digest = "a" * 64
    return DocumentRecord(
        doc_id=stable_document_id(digest),
        repo_id="org/vifinqa",
        revision="rev-1",
        relative_path="VCB/2024/Consolidated/report.txt",
        company_code="VCB",
        report_year=2024,
        statement_scope="consolidated",
        sha256=digest,
        file_size_bytes=1,
        encoding="utf-8",
        inventory_status="ready",
    )


def test_normalized_document_is_frozen_and_requires_matching_document() -> None:
    document = _document()
    extraction = ExtractionResult(doc_id=document.doc_id, blocks=(), tables=(), rejected=())
    normalized = NormalizedDocument(
        document=document,
        extraction=extraction,
        issues=(),
        ruleset_version="2026.08.1",
        normalization_fingerprint="b" * 64,
    )

    assert normalized.document.doc_id == normalized.extraction.doc_id
    with pytest.raises(ValidationError, match="frozen"):
        setattr(normalized, "ruleset_version", "changed")


def test_normalized_document_rejects_mismatched_doc_id() -> None:
    document = _document()
    extraction = ExtractionResult(doc_id=f"doc_{'c' * 64}", blocks=(), tables=(), rejected=())

    with pytest.raises(ValidationError, match="document and extraction IDs must match"):
        NormalizedDocument(
            document=document,
            extraction=extraction,
            issues=(),
            ruleset_version="2026.08.1",
            normalization_fingerprint="b" * 64,
        )


def test_issue_rejects_unknown_fields_and_noncanonical_ids() -> None:
    with pytest.raises(ValidationError):
        NormalizationIssue(
            code="metric_unknown",
            doc_id="bad",
            table_id=None,
            cell_id=None,
            field="metric",
            raw_value="Doanh thu",
        )
