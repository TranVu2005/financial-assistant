"""Security tests for the Day 22 submission validator (plan.md §2.4 rule 8):
the validator must reject a malicious ZIP before ever reading its contents as
trusted paths -- ZIP Slip (`../`), absolute/drive paths, symlink entries, and
duplicate entry names.
"""

from __future__ import annotations

import json
import stat
import zipfile
from pathlib import Path

from financial_report_qa.submission.validator import validate_submission_zip


def _minimal_valid_payload() -> bytes:
    return json.dumps([]).encode("utf-8")


def test_validator_rejects_zip_slip_entry(tmp_path: Path) -> None:
    zip_path = tmp_path / "evil.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("submission.json", _minimal_valid_payload())
        archive.writestr("../../etc/evil.csv", "x")

    report = validate_submission_zip(zip_path, expected_ids=[])
    assert report.valid is False
    assert any(issue.code == "unsafe_archive_entry" for issue in report.issues)


def test_validator_rejects_absolute_path_entry(tmp_path: Path) -> None:
    zip_path = tmp_path / "evil.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("submission.json", _minimal_valid_payload())
        archive.writestr("/etc/evil.csv", "x")

    report = validate_submission_zip(zip_path, expected_ids=[])
    assert report.valid is False
    assert any(issue.code == "unsafe_archive_entry" for issue in report.issues)


def test_validator_rejects_backslash_path_entry(tmp_path: Path) -> None:
    """A Windows-style separator could smuggle an escape past a naive
    POSIX-only traversal check."""
    zip_path = tmp_path / "evil.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("submission.json", _minimal_valid_payload())
        archive.writestr("data\\..\\..\\evil.csv", "x")

    report = validate_submission_zip(zip_path, expected_ids=[])
    assert report.valid is False
    assert any(issue.code == "unsafe_archive_entry" for issue in report.issues)


def test_validator_rejects_drive_path_entry(tmp_path: Path) -> None:
    zip_path = tmp_path / "evil.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("submission.json", _minimal_valid_payload())
        archive.writestr("C:/evil.csv", "x")

    report = validate_submission_zip(zip_path, expected_ids=[])
    assert report.valid is False
    assert any(issue.code == "unsafe_archive_entry" for issue in report.issues)


def test_validator_rejects_symlink_entry(tmp_path: Path) -> None:
    """A symlink entry inside the archive (unix mode bit in `external_attr`)
    could point CSV reads at an arbitrary file on the validating machine --
    checked from ZIP metadata, so this does not depend on the host OS being
    able to create real symlinks."""
    zip_path = tmp_path / "evil.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("submission.json", _minimal_valid_payload())
        info = zipfile.ZipInfo("data/link.csv")
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(info, "/etc/passwd")

    report = validate_submission_zip(zip_path, expected_ids=[])
    assert report.valid is False
    assert any(issue.code == "unsafe_archive_entry" for issue in report.issues)


def test_validator_rejects_duplicate_entry_names(tmp_path: Path) -> None:
    """Two entries with the same name is a classic zip-parser confusion
    vector (different tools may read different copies)."""
    zip_path = tmp_path / "evil.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("submission.json", _minimal_valid_payload())
        archive.writestr("data/dup.csv", "first")
        archive.writestr("data/dup.csv", "second")

    report = validate_submission_zip(zip_path, expected_ids=[])
    assert report.valid is False
    assert any(issue.code == "unsafe_archive_entry" for issue in report.issues)


def test_validator_rejects_entry_outside_data_dir(tmp_path: Path) -> None:
    zip_path = tmp_path / "evil.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("submission.json", _minimal_valid_payload())
        archive.writestr("notes.txt", "should not be here")

    report = validate_submission_zip(zip_path, expected_ids=[])
    assert report.valid is False
    assert any(issue.code == "entry_outside_data_dir" for issue in report.issues)


def test_validator_rejects_extra_root_json(tmp_path: Path) -> None:
    """A second root-level JSON file must not silently be ignored -- plan.md
    §2.4 rule 8 requires exactly one JSON at root."""
    zip_path = tmp_path / "evil.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("submission.json", _minimal_valid_payload())
        archive.writestr("hidden.json", _minimal_valid_payload())

    report = validate_submission_zip(zip_path, expected_ids=[])
    assert report.valid is False
    assert any(issue.code == "root_json_missing" for issue in report.issues)
