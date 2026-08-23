"""Tests for the ``export-tables`` command-line interface."""

from __future__ import annotations

from pathlib import Path

import pytest

import financial_report_qa.export.cli as export_cli
from financial_report_qa.core.errors import ExportError, FinancialReportQAError
from financial_report_qa.export.csv_export import CsvExportManifest
from financial_report_qa.export.synced_text import SyncedTextManifest

_CSV_MANIFEST = CsvExportManifest(
    output_dir=Path("out/csv"),
    manifest_path=Path("out/csv/manifest.jsonl"),
    table_count=3,
    entries=(),
)


def test_main_exports_then_syncs_and_prints_summary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CSV export runs first; its manifest feeds synced text; stdout summarizes counts."""
    calls: list[tuple[object, ...]] = []
    text_manifest = SyncedTextManifest(
        output_dir=Path("data/interim/synced_text"), document_count=2, table_count=3
    )

    def fake_csv(release_dir: Path, output_dir: Path) -> CsvExportManifest:
        calls.append(("csv", release_dir, output_dir))
        return _CSV_MANIFEST

    def fake_text(
        release_dir: Path,
        snapshot_root: Path,
        csv_manifest: CsvExportManifest,
        output_dir: Path | None,
    ) -> SyncedTextManifest:
        calls.append(("text", release_dir, snapshot_root, csv_manifest, output_dir))
        return text_manifest

    monkeypatch.setattr(export_cli, "export_normalized_csvs", fake_csv)
    monkeypatch.setattr(export_cli, "export_synced_text", fake_text)

    exit_code = export_cli.main(
        [
            "--release-dir",
            "release",
            "--snapshot-root",
            "snapshot",
            "--csv-output-dir",
            "out/csv",
        ]
    )

    assert exit_code == 0
    assert calls == [
        ("csv", Path("release"), Path("out/csv")),
        (
            "text",
            Path("release"),
            Path("snapshot"),
            _CSV_MANIFEST,
            Path("data/interim/synced_text"),
        ),
    ]
    assert capsys.readouterr().out == "exported 3 tables; synced 2 documents\n"


def test_main_honors_explicit_text_output_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    received: dict[str, Path | None] = {}

    def fake_csv(release_dir: Path, output_dir: Path) -> CsvExportManifest:
        return _CSV_MANIFEST

    def fake_text(
        release_dir: Path,
        snapshot_root: Path,
        csv_manifest: CsvExportManifest,
        output_dir: Path | None,
    ) -> SyncedTextManifest:
        received["output_dir"] = output_dir
        return SyncedTextManifest(
            output_dir=output_dir or Path("."), document_count=0, table_count=0
        )

    monkeypatch.setattr(export_cli, "export_normalized_csvs", fake_csv)
    monkeypatch.setattr(export_cli, "export_synced_text", fake_text)

    exit_code = export_cli.main(
        [
            "--release-dir",
            "release",
            "--snapshot-root",
            "snapshot",
            "--csv-output-dir",
            "out/csv",
            "--text-output-dir",
            "custom/text",
        ]
    )

    assert exit_code == 0
    assert received["output_dir"] == Path("custom/text")


def test_main_reports_base_class_failures_on_stderr(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Any FinancialReportQAError becomes `error: …` on stderr with exit code 1."""

    def failing_csv(release_dir: Path, output_dir: Path) -> CsvExportManifest:
        raise FinancialReportQAError("release manifest missing")

    monkeypatch.setattr(export_cli, "export_normalized_csvs", failing_csv)

    exit_code = export_cli.main(
        [
            "--release-dir",
            "release",
            "--snapshot-root",
            "snapshot",
            "--csv-output-dir",
            "out/csv",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert captured.err.startswith("error: ")
    assert "release manifest missing" in captured.err


def test_main_reports_synced_text_failures_on_stderr(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Failures raised by the second stage take the same boundary path."""

    def fake_csv(release_dir: Path, output_dir: Path) -> CsvExportManifest:
        return _CSV_MANIFEST

    def failing_text(
        release_dir: Path,
        snapshot_root: Path,
        csv_manifest: CsvExportManifest,
        output_dir: Path | None,
    ) -> SyncedTextManifest:
        raise ExportError("source document unreadable")

    monkeypatch.setattr(export_cli, "export_normalized_csvs", fake_csv)
    monkeypatch.setattr(export_cli, "export_synced_text", failing_text)

    exit_code = export_cli.main(
        [
            "--release-dir",
            "release",
            "--snapshot-root",
            "snapshot",
            "--csv-output-dir",
            "out/csv",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.err.startswith("error: ")
    assert "source document unreadable" in captured.err


def test_main_reports_missing_release_dir_as_error_line(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An unreadable release dir surfaces as ExportError on stderr, not a traceback."""
    missing_release = tmp_path / "missing"

    exit_code = export_cli.main(
        [
            "--release-dir",
            str(missing_release),
            "--snapshot-root",
            str(tmp_path / "snapshot"),
            "--csv-output-dir",
            str(tmp_path / "out"),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert captured.err.startswith("error: ")
    assert "cannot read release parquet" in captured.err
    assert str(missing_release) in captured.err
