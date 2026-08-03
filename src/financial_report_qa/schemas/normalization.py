from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from financial_report_qa.ingestion.provenance import ExtractionResult
from financial_report_qa.schemas.documents import DocumentRecord

NormalizationIssueCode = Literal[
    "company_conflict",
    "period_incomplete",
    "period_ambiguous",
    "period_invalid",
    "statement_conflict",
    "metric_unknown",
    "number_missing",
    "number_ambiguous",
    "number_invalid",
    "unit_unknown",
    "unit_conflict",
]

NormalizationField = Literal[
    "company", "period", "statement_type", "metric", "number", "unit"
]

ISSUE_FIELD_BY_CODE: dict[NormalizationIssueCode, NormalizationField] = {
    "company_conflict": "company",
    "period_incomplete": "period",
    "period_ambiguous": "period",
    "period_invalid": "period",
    "statement_conflict": "statement_type",
    "metric_unknown": "metric",
    "number_missing": "number",
    "number_ambiguous": "number",
    "number_invalid": "number",
    "unit_unknown": "unit",
    "unit_conflict": "unit",
}


class NormalizationIssue(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    code: NormalizationIssueCode
    doc_id: str = Field(pattern=r"^doc_[0-9a-f]{64}$")
    table_id: str | None = Field(default=None, pattern=r"^tbl_[0-9a-f]{64}$")
    cell_id: str | None = Field(default=None, pattern=r"^cell_[0-9a-f]{64}$")
    field: NormalizationField
    raw_value: str | None

    @model_validator(mode="after")
    def validate_code_field_pair(self) -> Self:
        expected = ISSUE_FIELD_BY_CODE[self.code]
        if self.field != expected:
            raise ValueError(f"issue code {self.code} requires field {expected}")
        return self


class NormalizedDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    document: DocumentRecord
    extraction: ExtractionResult
    issues: tuple[NormalizationIssue, ...]
    ruleset_version: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    normalization_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_document_identity(self) -> Self:
        if self.document.doc_id != self.extraction.doc_id:
            raise ValueError("document and extraction IDs must match")
        return self
