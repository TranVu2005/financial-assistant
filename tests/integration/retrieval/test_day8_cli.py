"""Fixture-only end-to-end coverage for the Day 8 retrieval CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from pytest import CaptureFixture, MonkeyPatch

from financial_report_qa.cli import main
from financial_report_qa.evaluation.week1_release import ReleaseLock
from financial_report_qa.retrieval.contracts import RetrievalFilters
from financial_report_qa.retrieval.gold import stable_question_id
from financial_report_qa.retrieval.release import ResolvedRetrievalRelease

_FINGERPRINT = "c" * 64
_TABLE_IDS = tuple(f"tbl_{value * 64}" for value in ("a", "b", "d"))


def _fixture_release(tmp_path: Path) -> ResolvedRetrievalRelease:
    release_dir = tmp_path / "release"
    release_dir.mkdir()
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.table(
            {
                "doc_id": ["doc_a", "doc_b", "doc_d"],
                "company_code": ["VCB", "VCB", "VCB"],
                "report_year": [2024, 2024, 2024],
                "relative_path": ["VCB/a.txt", "VCB/b.txt", "VCB/d.txt"],
            }
        ),
        release_dir / "documents.parquet",
    )
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.table(
            {
                "table_id": list(_TABLE_IDS),
                "doc_id": ["doc_a", "doc_b", "doc_d"],
                "title_raw": ["Doanh thu", "Chi phi", "Loi nhuan"],
                "statement_type": ["income", "income", "income"],
                "unit_normalized": [None, None, None],
                "line_start": [1, 3, 5],
                "line_end": [2, 4, 6],
            }
        ),
        release_dir / "tables.parquet",
    )
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.table(
            {
                "table_id": list(_TABLE_IDS),
                "row_label_canonical": ["Doanh thu", "Chi phi", "Loi nhuan"],
                "row_label_raw": ["Doanh thu", "Chi phi", "Loi nhuan"],
                "period": ["2024", "2024", "2024"],
                "unit": ["VND", "VND", "VND"],
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


def _write_fixture_gold(path: Path, *, include_failure: bool = False) -> None:
    filters = RetrievalFilters(company_codes=("VCB",))
    records: list[dict[str, object]] = []
    for number in range(70):
        question = (
            "zzzz-no-index-token"
            if include_failure and number == 0
            else f"Doanh thu VCB cau hoi {number:02d}?"
        )
        gold_ids = (_TABLE_IDS[0],)
        records.append(
            {
                "question_id": stable_question_id(question, filters, gold_ids, _FINGERPRINT),
                "question": question,
                "intent": ("lookup", "compare", "growth")[number % 3],
                "filters": filters.model_dump(mode="json"),
                "gold_table_ids": list(gold_ids),
                "reviewed_by": "fixture-reviewer",
                "reviewed_at": "2026-08-08T00:00:00+00:00",
                "gold_evidence": [
                    {
                        "table_id": _TABLE_IDS[0],
                        "relative_path": "VCB/a.txt",
                        "line_start": 1,
                        "line_end": 2,
                        "verified": True,
                    }
                ],
                "dataset_fingerprint": _FINGERPRINT,
            }
        )
    path.write_text(
        "\n".join(
            json.dumps(record, sort_keys=True)
            for record in sorted(records, key=lambda item: str(item["question_id"]))
        )
        + "\n",
        encoding="utf-8",
    )


def _patch_release_resolver(monkeypatch: MonkeyPatch, release: ResolvedRetrievalRelease) -> None:
    import financial_report_qa.retrieval.cli as retrieval_cli

    monkeypatch.setattr(
        retrieval_cli, "resolve_retrieval_release", lambda *_args, **_kwargs: release
    )


def test_retrieval_cli_fixture_lifecycle_is_replayable_and_fails_closed_when_corrupt(
    tmp_path: Path, monkeypatch: MonkeyPatch, capsys: CaptureFixture[str]
) -> None:
    """Exercise build, validation, evaluation, replay, and corrupted-index exit handling."""
    release = _fixture_release(tmp_path)
    _patch_release_resolver(monkeypatch, release)
    gold_path = tmp_path / "retrieval-gold-v1.jsonl"
    _write_fixture_gold(gold_path)
    index_root = tmp_path / "indexes"

    assert (
        main(
            [
                "retrieval",
                "build-index",
                "--release-lock",
                "fixture-lock",
                "--output-root",
                str(index_root),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "retrieval",
                "validate-gold",
                "--release-lock",
                "fixture-lock",
                "--gold-path",
                str(gold_path),
            ]
        )
        == 0
    )

    index_dir = index_root / _FINGERPRINT
    manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "bm25-index-v3"
    assert manifest["query_expansion_version"] == "v1"
    first_output = tmp_path / "evaluations-a"
    second_output = tmp_path / "evaluations-b"
    evaluate_args = [
        "retrieval",
        "evaluate",
        "--release-lock",
        "fixture-lock",
        "--index-dir",
        str(index_dir),
        "--gold-path",
        str(gold_path),
    ]
    assert main([*evaluate_args, "--output-dir", str(first_output)]) == 0
    assert main([*evaluate_args, "--output-dir", str(second_output)]) == 0

    json_name = f"retrieval-day8-{_FINGERPRINT[:12]}.json"
    markdown_name = f"retrieval-day8-{_FINGERPRINT[:12]}.md"
    first_report = json.loads((first_output / json_name).read_text(encoding="utf-8"))
    first_markdown = (first_output / markdown_name).read_text(encoding="utf-8")
    assert f"Precision@10: {first_report['macro']['precision']:.6f}" in first_markdown
    assert (first_output / json_name).read_bytes() == (second_output / json_name).read_bytes()
    assert (first_output / markdown_name).read_bytes() == (
        second_output / markdown_name
    ).read_bytes()

    (index_dir / "manifest.json").write_text("not-json", encoding="utf-8")
    assert main([*evaluate_args, "--output-dir", str(tmp_path / "corrupt-output")]) == 2
    assert "retrieval error:" in capsys.readouterr().err


def test_retrieval_cli_evaluate_v2_and_failure_export_are_offline_and_replayable(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture[str],
) -> None:
    release = _fixture_release(tmp_path)
    _patch_release_resolver(monkeypatch, release)
    gold_path = tmp_path / "retrieval-gold-v1.jsonl"
    _write_fixture_gold(gold_path, include_failure=True)
    index_root = tmp_path / "indexes"
    assert (
        main(
            [
                "retrieval",
                "build-index",
                "--release-lock",
                "fixture-lock",
                "--output-root",
                str(index_root),
            ]
        )
        == 0
    )

    index_dir = index_root / _FINGERPRINT
    first_output = tmp_path / "v2-a"
    second_output = tmp_path / "v2-b"
    evaluate_v2_args = [
        "retrieval",
        "evaluate-v2",
        "--release-lock",
        "fixture-lock",
        "--index-dir",
        str(index_dir),
        "--gold-path",
        str(gold_path),
    ]
    assert main([*evaluate_v2_args, "--output-dir", str(first_output)]) == 0
    assert main([*evaluate_v2_args, "--output-dir", str(second_output)]) == 0

    v2_json = first_output / f"retrieval-v2-{_FINGERPRINT[:12]}.json"
    assert (
        v2_json.read_bytes()
        == (second_output / f"retrieval-v2-{_FINGERPRINT[:12]}.json").read_bytes()
    )
    v2_payload = json.loads(v2_json.read_text(encoding="utf-8"))
    assert v2_payload["question_count"] == len(v2_payload["per_question"]) == 70
    assert "by_statement_filter" in v2_payload

    legacy_output = tmp_path / "legacy"
    assert (
        main(
            [
                "retrieval",
                "evaluate",
                "--release-lock",
                "fixture-lock",
                "--index-dir",
                str(index_dir),
                "--gold-path",
                str(gold_path),
                "--output-dir",
                str(legacy_output),
            ]
        )
        == 0
    )
    system_output = tmp_path / "system-v2"
    assert (
        main(
            [
                "retrieval",
                "derive-v2",
                "--release-lock",
                "fixture-lock",
                "--gold-path",
                str(gold_path),
                "--source-report",
                str(legacy_output / f"retrieval-day8-{_FINGERPRINT[:12]}.json"),
                "--source-kind",
                "legacy",
                "--system-name",
                "fixture-bm25",
                "--output-dir",
                str(system_output),
            ]
        )
        == 0
    )
    system_payload = json.loads(
        (system_output / f"retrieval-v2-fixture-bm25-{_FINGERPRINT[:12]}.json").read_text(
            encoding="utf-8"
        )
    )
    assert system_payload["question_count"] == len(system_payload["per_question"]) == 70
    assert set(system_payload["by_intent"]) == {"compare", "growth", "lookup"}

    annotations = tmp_path / "failure-annotations.jsonl"
    failure_rows = [
        {
            "question_id": item["question_id"],
            "root_cause": "ranking_only",
            "note": "Fixture gold is absent from this deterministic local ranking.",
        }
        for item in v2_payload["per_question"]
        if item["failure"] != "none"
    ]
    annotations.write_text(
        "\n".join(json.dumps(item, sort_keys=True) for item in failure_rows) + "\n",
        encoding="utf-8",
    )
    failure_output = tmp_path / "failures"
    assert (
        main(
            [
                "retrieval",
                "export-failures",
                "--evaluation-report",
                str(v2_json),
                "--annotations",
                str(annotations),
                "--output-dir",
                str(failure_output),
            ]
        )
        == 0
    )
    failure_payload = json.loads(
        (failure_output / f"failures-{_FINGERPRINT[:12]}.json").read_text(encoding="utf-8")
    )
    assert failure_payload["failure_count"] == len(failure_rows)
    assert failure_payload["evaluated_question_count"] == 70

    annotations.write_text("\n", encoding="utf-8")
    assert (
        main(
            [
                "retrieval",
                "export-failures",
                "--evaluation-report",
                str(v2_json),
                "--annotations",
                str(annotations),
                "--output-dir",
                str(tmp_path / "invalid-failures"),
            ]
        )
        == 2
    )
    assert "retrieval error:" in capsys.readouterr().err
