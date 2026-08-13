"""Fixture-only end-to-end coverage for the offline Day 11 graph CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from pytest import MonkeyPatch

from financial_report_qa.cli import main
from financial_report_qa.evaluation.week1_release import ReleaseLock
from financial_report_qa.retrieval.release import ResolvedRetrievalRelease

_FINGERPRINT = "37a61be7aebde1fbcfe3aca42e6ba4ff37ae87bdd1a9ba6696506bcd188e7d1f"
_TABLE_A = f"tbl_{'a' * 64}"
_TABLE_B = f"tbl_{'b' * 64}"


def _fixture_release(tmp_path: Path) -> ResolvedRetrievalRelease:
    release_dir = tmp_path / "release"
    release_dir.mkdir()
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.table(
            {
                "doc_id": ["doc_a"],
                "company_code": ["DBC"],
                "report_year": [2023],
                "relative_path": ["DBC/a.txt"],
            }
        ),
        release_dir / "documents.parquet",
    )
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.table(
            {
                "table_id": [_TABLE_A, _TABLE_B],
                "doc_id": ["doc_a", "doc_a"],
                "title_raw": ["Bảng cân đối kế toán", "Thuyết minh"],
                "statement_type": ["balance_sheet", "notes"],
                "unit_normalized": [None, None],
                "line_start": [1, 20],
                "line_end": [2, 21],
            }
        ),
        release_dir / "tables.parquet",
    )
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.table(
            {
                "table_id": [_TABLE_A, _TABLE_B],
                "row_label_canonical": ["total_assets", "total_assets"],
                "row_label_raw": ["Tổng tài sản", "Tổng tài sản"],
                "period": ["2023", "2023"],
                "unit": ["VND", "VND"],
            }
        ),
        release_dir / "cells.parquet",
    )
    lock = ReleaseLock(
        alias="dataset-pilot-v1",
        sampling_version="week1-pilot-v1",
        dataset_fingerprint=_FINGERPRINT,
        source_manifest_sha256="0" * 64,
        release_path="fixture/release",
        gate_result_path="fixture/gate.json",
        evaluation_inputs_sha256="1" * 64,
    )
    return ResolvedRetrievalRelease(
        lock=lock,
        dataset_fingerprint=_FINGERPRINT,
        release_dir=release_dir,
        gate_result_path=tmp_path / "gate.json",
        lock_path=tmp_path / "lock.json",
        manifest={},
        lock_sha256="2" * 64,
    )


def _patch_release_resolver(monkeypatch: MonkeyPatch, release: ResolvedRetrievalRelease) -> None:
    import financial_report_qa.retrieval.cli as retrieval_cli

    monkeypatch.setattr(
        retrieval_cli, "resolve_retrieval_release", lambda *_args, **_kwargs: release
    )


def test_build_graph_cli_is_byte_identical_across_two_builds(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    release = _fixture_release(tmp_path)
    _patch_release_resolver(monkeypatch, release)

    root_a = tmp_path / "graph-a"
    root_b = tmp_path / "graph-b"
    assert (
        main(
            [
                "retrieval",
                "build-graph",
                "--release-lock",
                "fixture-lock",
                "--output-root",
                str(root_a),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "retrieval",
                "build-graph",
                "--release-lock",
                "fixture-lock",
                "--output-root",
                str(root_b),
            ]
        )
        == 0
    )

    dir_a = root_a / _FINGERPRINT
    dir_b = root_b / _FINGERPRINT
    assert (dir_a / "buckets.jsonl").read_bytes() == (dir_b / "buckets.jsonl").read_bytes()
    assert (dir_a / "manifest.json").read_bytes() == (dir_b / "manifest.json").read_bytes()


def test_evaluate_graph_cli_produces_expected_reports_and_replays(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    release = _fixture_release(tmp_path)
    _patch_release_resolver(monkeypatch, release)

    graph_root = tmp_path / "graph"
    assert (
        main(
            [
                "retrieval",
                "build-graph",
                "--release-lock",
                "fixture-lock",
                "--output-root",
                str(graph_root),
            ]
        )
        == 0
    )
    graph_dir = graph_root / _FINGERPRINT

    output_dir = tmp_path / "day11"
    assert (
        main(
            [
                "retrieval",
                "evaluate-graph",
                "--release-lock",
                "fixture-lock",
                "--graph-dir",
                str(graph_dir),
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )
    json_reports = list(output_dir.glob("retrieval-day11-graph-*.json"))
    md_reports = list(output_dir.glob("retrieval-day11-graph-*.md"))
    assert len(json_reports) == 1
    assert len(md_reports) == 1
    first_bytes = json_reports[0].read_bytes()
    report = json.loads(first_bytes)
    assert report["document_count"] == 2
    assert report["by_relation"]["same_document"]["directed_edge_count"] == 2
    assert report["by_relation"]["explained_by_note"]["directed_edge_count"] == 1
    excluded = {row["name"] for row in report["excluded_relations"]}
    assert excluded == {"same_company", "same_period"}

    output_dir_replay = tmp_path / "day11-replay"
    assert (
        main(
            [
                "retrieval",
                "evaluate-graph",
                "--release-lock",
                "fixture-lock",
                "--graph-dir",
                str(graph_dir),
                "--output-dir",
                str(output_dir_replay),
            ]
        )
        == 0
    )
    replayed = list(output_dir_replay.glob("retrieval-day11-graph-*.json"))
    assert replayed[0].read_bytes() == first_bytes


def test_evaluate_graph_cli_fails_closed_on_missing_graph(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    release = _fixture_release(tmp_path)
    _patch_release_resolver(monkeypatch, release)

    exit_code = main(
        [
            "retrieval",
            "evaluate-graph",
            "--release-lock",
            "fixture-lock",
            "--graph-dir",
            str(tmp_path / "missing-graph"),
            "--output-dir",
            str(tmp_path / "day11"),
        ]
    )
    assert exit_code == 2
