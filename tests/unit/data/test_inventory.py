from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from pydantic import ValidationError

from financial_report_qa.data.inventory import (
    InventoryIssue,
    InventoryResult,
    _parse_vifinqa_path,
    build_inventory,
    main,
)


def _write_report(root: Path, relative: str, content: bytes) -> Path:
    path = root / Path(relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_build_inventory_hashes_unicode_utf8_and_preserves_source(tmp_path: Path) -> None:
    root = tmp_path / "financial_statements"
    content = "Báº£ng cÃ¢n Ä‘á»‘i káº¿ toÃ¡n".encode()
    source = _write_report(root, "VCB/2024/Consolidated/BÃ¡o cÃ¡o.TXT", content)
    before = source.read_bytes()

    result = build_inventory(root, repo_id="org/vifinqa", revision="abc123")

    assert result.issues == ()
    assert len(result.documents) == 1
    record = result.documents[0]
    assert record.relative_path == "VCB/2024/Consolidated/BÃ¡o cÃ¡o.TXT"
    assert record.sha256 == hashlib.sha256(content).hexdigest()
    assert record.file_size_bytes == len(content)
    assert record.encoding == "utf-8"
    assert record.inventory_status == "ready"
    assert source.read_bytes() == before


def test_build_inventory_distinguishes_bom_empty_duplicate_and_issue(tmp_path: Path) -> None:
    root = tmp_path / "financial_statements"
    shared = b"doanh thu"
    _write_report(root, "AAA/2023/Separate/a.txt", shared)
    _write_report(root, "AAA/2023/Separate/b.txt", shared)
    _write_report(root, "AAA/2023/Separate/bom.txt", b"\xef\xbb\xbf" + "ná»£".encode())
    _write_report(root, "AAA/2023/Separate/empty.txt", b"")
    _write_report(root, "AAA/2023/Separate/bad.txt", b"\xff\xfe")
    _write_report(root, "bad/year/path.txt", b"valid bytes")

    result = build_inventory(root, repo_id="org/vifinqa", revision="abc123")
    by_path = {record.relative_path: record for record in result.documents}

    assert by_path["AAA/2023/Separate/a.txt"].inventory_status == "ready"
    duplicate = by_path["AAA/2023/Separate/b.txt"]
    assert duplicate.inventory_status == "duplicate"
    assert duplicate.notes == ("duplicate_of=AAA/2023/Separate/a.txt",)
    assert by_path["AAA/2023/Separate/bom.txt"].encoding == "utf-8-sig"
    assert by_path["AAA/2023/Separate/empty.txt"].inventory_status == "empty"
    assert {issue.relative_path for issue in result.issues} == {
        "AAA/2023/Separate/bad.txt",
        "bad/year/path.txt",
    }


def test_build_inventory_is_deterministic_and_ignores_non_txt(tmp_path: Path) -> None:
    root = tmp_path / "financial_statements"
    _write_report(root, "ZZZ/2022/Aggregated/z.txt", b"z")
    _write_report(root, "AAA/2022/Other/a.txt", b"a")
    _write_report(root, "AAA/2022/Other/ignored.csv", b"csv")

    first = build_inventory(root, repo_id="org/vifinqa", revision="abc123")
    second = build_inventory(root, repo_id="org/vifinqa", revision="abc123")

    assert first == second
    assert [record.relative_path for record in first.documents] == [
        "AAA/2022/Other/a.txt",
        "ZZZ/2022/Aggregated/z.txt",
    ]


@pytest.mark.parametrize("root_state", ["missing", "file"])
def test_build_inventory_rejects_non_directory_root(
    tmp_path: Path,
    root_state: str,
) -> None:
    root = tmp_path / "financial_statements"
    if root_state == "file":
        root.write_text("not a directory", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="inventory root"):
        build_inventory(root, repo_id="org/vifinqa", revision="abc123")


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


def test_parse_vifinqa_path_accepts_uppercase_txt_extension(tmp_path: Path) -> None:
    root = tmp_path / "financial_statements"

    metadata = _parse_vifinqa_path(root / "VCB" / "2024" / "report" / "source.TXT", root)

    assert metadata.relative_path == "VCB/2024/report/source.TXT"


@pytest.mark.parametrize("filename", ["source.pdf", "source"])
def test_parse_vifinqa_path_rejects_non_txt_files(tmp_path: Path, filename: str) -> None:
    root = tmp_path / "financial_statements"

    with pytest.raises(ValueError, match="TXT"):
        _parse_vifinqa_path(root / "VCB" / "2024" / "report" / filename, root)


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


def test_inventory_main_writes_manifest_and_prints_counts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = tmp_path / "financial_statements"
    _write_report(root, "AAA/2024/Consolidated/ready.txt", b"ready")
    _write_report(root, "AAA/2024/Consolidated/empty.txt", b"")
    manifest = tmp_path / "manifests" / "documents.jsonl"

    exit_code = main(
        [
            "--root", str(root),
            "--repo-id", "org/vifinqa",
            "--revision", "abc123",
            "--manifest", str(manifest),
        ]
    )

    assert exit_code == 0
    assert manifest.exists()
    output = capsys.readouterr().out
    assert "Documents: 2" in output
    assert "Ready:     1" in output
    assert "Empty:     1" in output
    assert "Duplicate: 0" in output
    assert "Issues:    0" in output
    assert f"Manifest:  {manifest.resolve()}" in output


def test_inventory_main_reports_expected_failure_without_traceback(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        [
            "--root", str(tmp_path / "missing"),
            "--repo-id", "org/vifinqa",
            "--revision", "abc123",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "inventory root" in captured.err
    assert "Traceback" not in captured.err


def test_inventory_main_returns_2_for_missing_required_argument(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(["--root", "data/raw/vifinqa", "--repo-id", "org/vifinqa"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "the following arguments are required: --revision" in captured.err
    assert "Traceback" not in captured.err
