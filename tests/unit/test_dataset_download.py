"""Tests for resumable Hugging Face dataset downloads."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from huggingface_hub.file_download import DryRunFileInfo

from financial_report_qa.data.download import (
    DownloadRequest,
    InsufficientDiskSpaceError,
    build_download_plan,
    download_dataset,
    main,
)


def _file_info(
    filename: str,
    *,
    size: int,
    will_download: bool,
    commit_hash: str = "a" * 40,
) -> DryRunFileInfo:
    return DryRunFileInfo(
        commit_hash=commit_hash,
        file_size=size,
        filename=filename,
        local_path=filename,
        is_cached=not will_download,
        will_download=will_download,
    )


def test_build_download_plan_counts_only_files_not_already_downloaded(tmp_path: Path) -> None:
    """A missing will_download filter would overstate the required free space on resume."""
    infos = [
        _file_info("AAA/2024/report.txt", size=2_048, will_download=True),
        _file_info("FPT/2024/report.txt", size=1_024, will_download=False),
    ]

    def fake_snapshot_download(**kwargs: Any) -> list[DryRunFileInfo]:
        assert kwargs["dry_run"] is True
        return infos

    request = DownloadRequest(target_dir=tmp_path / "data")
    plan = build_download_plan(request, snapshot_download_fn=fake_snapshot_download)

    assert plan.resolved_revision == "a" * 40
    assert plan.file_count == 2
    assert plan.total_bytes == 3_072
    assert plan.files_to_download == 1
    assert plan.bytes_to_download == 2_048


def test_download_stops_before_network_transfer_when_disk_space_is_insufficient(
    tmp_path: Path,
) -> None:
    """Removing the capacity gate would start a download that cannot finish."""
    calls: list[bool] = []

    def fake_snapshot_download(**kwargs: Any) -> list[DryRunFileInfo] | str:
        calls.append(kwargs["dry_run"])
        if kwargs["dry_run"]:
            return [_file_info("AAA/2024/report.txt", size=2_048, will_download=True)]
        raise AssertionError("transfer must not start")

    request = DownloadRequest(
        target_dir=tmp_path / "data",
        manifest_path=tmp_path / "manifest.json",
        reserve_bytes=1_024,
    )

    with pytest.raises(InsufficientDiskSpaceError) as error:
        download_dataset(
            request,
            snapshot_download_fn=fake_snapshot_download,
            free_space_fn=lambda _: 2_500,
        )

    assert error.value.required_bytes == 3_072
    assert error.value.available_bytes == 2_500
    assert calls == [True]
    assert not request.manifest_path.exists()


def test_download_pins_revision_preserves_unicode_and_writes_manifest(tmp_path: Path) -> None:
    """Using the moving branch or rewriting names would make a download irreproducible."""
    unicode_name = "FPT/2024/Báo cáo tài chính_extracted.txt"
    commit_hash = "b" * 40
    calls: list[dict[str, Any]] = []

    def fake_snapshot_download(**kwargs: Any) -> list[DryRunFileInfo] | str:
        calls.append(kwargs)
        if kwargs["dry_run"]:
            return [
                _file_info(
                    unicode_name,
                    size=15,
                    will_download=True,
                    commit_hash=commit_hash,
                )
            ]

        target = Path(kwargs["local_dir"])
        downloaded_file = target / unicode_name
        downloaded_file.parent.mkdir(parents=True, exist_ok=True)
        downloaded_file.write_text("bảng tài chính", encoding="utf-8")
        return str(target)

    manifest_path = tmp_path / "manifests" / "tinix.json"
    request = DownloadRequest(
        target_dir=tmp_path / "data" / "raw" / "ocr_annual_financials",
        manifest_path=manifest_path,
        reserve_bytes=1_000,
    )

    result = download_dataset(
        request,
        snapshot_download_fn=fake_snapshot_download,
        free_space_fn=lambda _: 10_000,
    )

    assert calls[0]["revision"] == "main"
    assert calls[0]["dry_run"] is True
    assert calls[1]["revision"] == commit_hash
    assert calls[1]["dry_run"] is False
    assert (request.target_dir / unicode_name).read_text(encoding="utf-8") == "bảng tài chính"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["repo_id"] == "tinixai/ocr_annual_financials"
    assert manifest["requested_revision"] == "main"
    assert manifest["resolved_revision"] == commit_hash
    assert manifest["file_count"] == 1
    assert manifest["total_bytes"] == 15
    assert manifest["status"] == "complete"
    assert result.resolved_revision == commit_hash


def test_dry_run_does_not_start_transfer_or_write_manifest(tmp_path: Path) -> None:
    """A dry-run flag must never trigger the expensive transfer as a side effect."""
    calls: list[bool] = []

    def fake_snapshot_download(**kwargs: Any) -> list[DryRunFileInfo] | str:
        calls.append(kwargs["dry_run"])
        return [_file_info("AAA/2024/report.txt", size=50, will_download=True)]

    request = DownloadRequest(
        target_dir=tmp_path / "data",
        manifest_path=tmp_path / "manifest.json",
    )

    result = download_dataset(
        request,
        dry_run=True,
        snapshot_download_fn=fake_snapshot_download,
    )

    assert calls == [True]
    assert result.bytes_to_download == 50
    assert not request.manifest_path.exists()


def test_cli_defaults_to_dry_run_until_download_flag_is_present(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Changing the safe CLI default could accidentally start a 194 GB transfer."""
    calls: list[bool] = []

    def fake_snapshot_download(**kwargs: Any) -> list[DryRunFileInfo] | str:
        calls.append(kwargs["dry_run"])
        return [_file_info("AAA/2024/report.txt", size=50, will_download=True)]

    exit_code = main(
        ["--target", str(tmp_path / "data")],
        snapshot_download_fn=fake_snapshot_download,
    )

    assert exit_code == 0
    assert calls == [True]
    assert "Dry run only" in capsys.readouterr().out
