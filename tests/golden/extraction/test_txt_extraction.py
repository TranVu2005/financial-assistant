from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from financial_report_qa.ingestion import ExtractionResult, extract_document
from financial_report_qa.schemas.documents import DocumentRecord, stable_document_id

CASES = (
    "unicode_continuation",
    "structured_fallback",
    "explanatory_letter",
)


def semantic_projection(result: ExtractionResult) -> dict[str, object]:
    return {
        "table_count": len(result.tables),
        "rejection_codes": [item.reason for item in result.rejected],
        "tables": [
            {
                "line_start": item.table.line_start,
                "line_end": item.table.line_end,
                "row_count": item.table.row_count,
                "column_count": item.table.column_count,
                "values": [cell.value_raw for cell in item.cells],
                "evidence": list(item.evidence),
            }
            for item in result.tables
        ],
    }


@pytest.mark.parametrize("case_name", CASES)
def test_fixture_matches_hand_reviewed_golden_and_is_deterministic(
    tmp_path: Path,
    case_name: str,
) -> None:
    base = Path(__file__).parent
    content = (base / "fixtures" / f"{case_name}.txt").read_bytes()
    relative = f"AAA/2024/AAA_consolidated/{case_name}.txt"
    source = tmp_path / Path(relative)
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    document = DocumentRecord(
        doc_id=stable_document_id(digest),
        repo_id="org/vifinqa",
        revision="abc123",
        relative_path=relative,
        company_code="AAA",
        report_year=2024,
        statement_scope="consolidated",
        sha256=digest,
        file_size_bytes=len(content),
        encoding="utf-8",
        inventory_status="ready",
        notes=(),
    )
    expected = json.loads((base / "expected" / f"{case_name}.json").read_text(encoding="utf-8"))

    first = extract_document(tmp_path, document)
    second = extract_document(tmp_path, document)

    assert semantic_projection(first) == expected
    assert second == first
