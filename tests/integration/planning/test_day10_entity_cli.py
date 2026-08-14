"""End-to-end coverage for the offline Day 10 entity-parser CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from pytest import MonkeyPatch

from financial_report_qa.cli import main
from financial_report_qa.evaluation.week1_release import ReleaseLock
from financial_report_qa.retrieval.contracts import RetrievalFilters
from financial_report_qa.retrieval.gold import stable_question_id
from financial_report_qa.retrieval.release import ResolvedRetrievalRelease

_FINGERPRINT = "37a61be7aebde1fbcfe3aca42e6ba4ff37ae87bdd1a9ba6696506bcd188e7d1f"


def _fixture_release(tmp_path: Path) -> ResolvedRetrievalRelease:
    release_dir = tmp_path / "release"
    release_dir.mkdir()
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.table(
            {
                "doc_id": ["doc_a", "doc_b"],
                "company_code": ["DBC", "ACB"],
                "report_year": [2023, 2023],
                "relative_path": ["DBC/a.txt", "ACB/b.txt"],
            }
        ),
        release_dir / "documents.parquet",
    )
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.table(
            {
                "table_id": [f"tbl_{'a' * 64}", f"tbl_{'b' * 64}"],
                "doc_id": ["doc_a", "doc_b"],
                "title_raw": ["Bảng cân đối kế toán", "Báo cáo kết quả kinh doanh"],
                "statement_type": ["balance_sheet", "income_statement"],
                "unit_normalized": [None, None],
                "line_start": [1, 1],
                "line_end": [2, 2],
            }
        ),
        release_dir / "tables.parquet",
    )
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.table(
            {
                "table_id": [f"tbl_{'a' * 64}"] * 7,
                "row_label_canonical": ["total_assets"] * 7,
                "row_label_raw": ["Tổng tài sản"] * 7,
                "period": [
                    "2022",
                    "2023",
                    "2024",
                    "2023-Q4",
                    "2022-Q1",
                    "2023-12-31",
                    "2022-12-31",
                ],
                "unit": ["VND"] * 7,
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
    import financial_report_qa.planning.cli as planning_cli

    monkeypatch.setattr(
        planning_cli, "resolve_retrieval_release", lambda *_args, **_kwargs: release
    )


def _write_fixture_gold(path: Path) -> None:
    filters = RetrievalFilters(company_codes=("DBC",))
    records: list[dict[str, object]] = []
    for number in range(70):
        question = f"Tra cứu tổng tài sản của DBC năm 2023 câu {number:02d}?"
        gold_ids = (f"tbl_{'a' * 64}",)
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
                        "table_id": gold_ids[0],
                        "relative_path": "DBC/a.txt",
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
            json.dumps(record, sort_keys=True, ensure_ascii=False)
            for record in sorted(records, key=lambda record: str(record["question_id"]))
        )
        + "\n",
        encoding="utf-8",
    )


def test_entity_cli_generate_and_evaluate_is_replayable(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    release = _fixture_release(tmp_path)
    _patch_release_resolver(monkeypatch, release)
    cases_path = tmp_path / "entity-cases.jsonl"

    assert (
        main(
            [
                "planning",
                "generate-entity-cases",
                "--release-lock",
                "fixture-lock",
                "--output-path",
                str(cases_path),
            ]
        )
        == 0
    )
    assert cases_path.is_file()
    first_bytes = cases_path.read_bytes()

    second_path = tmp_path / "entity-cases-2.jsonl"
    assert (
        main(
            [
                "planning",
                "generate-entity-cases",
                "--release-lock",
                "fixture-lock",
                "--output-path",
                str(second_path),
            ]
        )
        == 0
    )
    assert second_path.read_bytes() == first_bytes

    output_dir = tmp_path / "artifacts"
    assert (
        main(
            [
                "planning",
                "evaluate-entities",
                "--release-lock",
                "fixture-lock",
                "--case-path",
                str(cases_path),
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )
    json_reports = list(output_dir.glob("entity-cases-*.json"))
    assert len(json_reports) == 1
    report = json.loads(json_reports[0].read_text(encoding="utf-8"))
    assert report["exact_match_rate"] == 1.0
    assert report["failures"] == []


def test_entity_cli_evaluate_with_held_out_gold(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    release = _fixture_release(tmp_path)
    _patch_release_resolver(monkeypatch, release)
    cases_path = tmp_path / "entity-cases.jsonl"
    assert (
        main(
            [
                "planning",
                "generate-entity-cases",
                "--release-lock",
                "fixture-lock",
                "--output-path",
                str(cases_path),
            ]
        )
        == 0
    )
    gold_path = tmp_path / "gold.jsonl"
    _write_fixture_gold(gold_path)
    output_dir = tmp_path / "artifacts"

    assert (
        main(
            [
                "planning",
                "evaluate-entities",
                "--release-lock",
                "fixture-lock",
                "--case-path",
                str(cases_path),
                "--output-dir",
                str(output_dir),
                "--gold-path",
                str(gold_path),
            ]
        )
        == 0
    )
    assert list(output_dir.glob("entity-held-out-*.json"))
    assert list(output_dir.glob("entity-held-out-*.md"))


def test_entity_cli_fails_closed_on_missing_case_file(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    release = _fixture_release(tmp_path)
    _patch_release_resolver(monkeypatch, release)
    exit_code = main(
        [
            "planning",
            "evaluate-entities",
            "--release-lock",
            "fixture-lock",
            "--case-path",
            str(tmp_path / "missing.jsonl"),
            "--output-dir",
            str(tmp_path / "artifacts"),
        ]
    )
    assert exit_code == 2


def _write_fixture_plan_cases(release: ResolvedRetrievalRelease, path: Path) -> None:
    from financial_report_qa.planning.entity_cases import generate_entity_cases
    from financial_report_qa.planning.plan_cases import generate_plan_cases, write_plan_cases

    entity_cases = generate_entity_cases(release)
    write_plan_cases(generate_plan_cases(entity_cases), path)


def test_plan_cli_evaluate_is_replayable(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    release = _fixture_release(tmp_path)
    _patch_release_resolver(monkeypatch, release)
    plan_cases_path = tmp_path / "plan-cases.jsonl"
    _write_fixture_plan_cases(release, plan_cases_path)
    output_dir = tmp_path / "artifacts"

    assert (
        main(
            [
                "planning",
                "evaluate-plans",
                "--release-lock",
                "fixture-lock",
                "--case-path",
                str(plan_cases_path),
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )
    json_reports = list(output_dir.glob("plan-cases-*.json"))
    assert len(json_reports) == 1
    report = json.loads(json_reports[0].read_text(encoding="utf-8"))
    assert report["false_plan_rate"] == 0.0


def test_plan_cli_evaluate_with_held_out_gold(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    release = _fixture_release(tmp_path)
    _patch_release_resolver(monkeypatch, release)
    plan_cases_path = tmp_path / "plan-cases.jsonl"
    _write_fixture_plan_cases(release, plan_cases_path)
    gold_path = tmp_path / "gold.jsonl"
    _write_fixture_gold(gold_path)
    output_dir = tmp_path / "artifacts"

    assert (
        main(
            [
                "planning",
                "evaluate-plans",
                "--release-lock",
                "fixture-lock",
                "--case-path",
                str(plan_cases_path),
                "--output-dir",
                str(output_dir),
                "--gold-path",
                str(gold_path),
            ]
        )
        == 0
    )
    assert list(output_dir.glob("plan-held-out-*.json"))
    assert list(output_dir.glob("plan-held-out-*.md"))


def test_plan_cli_fails_closed_on_missing_case_file(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    release = _fixture_release(tmp_path)
    _patch_release_resolver(monkeypatch, release)
    exit_code = main(
        [
            "planning",
            "evaluate-plans",
            "--release-lock",
            "fixture-lock",
            "--case-path",
            str(tmp_path / "missing.jsonl"),
            "--output-dir",
            str(tmp_path / "artifacts"),
        ]
    )
    assert exit_code == 2
