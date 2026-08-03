from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from financial_report_qa.data.inventory import (
    InventoryIssue,
    InventoryResult,
    _parse_vifinqa_path,
)


def test_parse_vifinqa_path_preserves_unicode_and_extracts_metadata(tmp_path: Path) -> None:
    root = tmp_path / "financial_statements"
    path = root / "vcb" / "2024" / "BÃ¡o cÃ¡o CONSOLIDATED" / "báº£ng cÃ¢n Ä‘á»‘i.TXT"

    metadata = _parse_vifinqa_path(path, root)

    assert metadata.relative_path == (
        "vcb/2024/BÃ¡o cÃ¡o CONSOLIDATED/báº£ng cÃ¢n Ä‘á»‘i.TXT"
    )
    assert metadata.company_code == "VCB"
    assert metadata.report_year == 2024
    assert metadata.statement_scope == "consolidated"


@pytest.mark.parametrize(
    ("relative", "message"),
    [
        ("VCB/2024/file.txt", "exactly ticker/year/document/file"),
        ("v!/2024/report/file.txt", "ticker"),
        ("VCB/year/report/file.txt", "year"),
        ("VCB/2101/report/file.txt", "year"),
    ],
)
def test_parse_vifinqa_path_rejects_invalid_hierarchy(
    tmp_path: Path,
    relative: str,
    message: str,
) -> None:
    root = tmp_path / "financial_statements"

    with pytest.raises(ValueError, match=message):
        _parse_vifinqa_path(root / Path(relative), root)


def test_inventory_models_are_frozen_and_forbid_unknown_fields() -> None:
    issue = InventoryIssue(
        relative_path="bad/year/report/file.txt",
        reason="invalid year directory",
        file_size_bytes=4,
        sha256="a" * 64,
    )
    result = InventoryResult(documents=(), issues=(issue,))

    with pytest.raises(ValidationError):
        InventoryIssue.model_validate({**issue.model_dump(), "unexpected": True})
    with pytest.raises(ValidationError, match="frozen"):
        setattr(result, "issues", ())
