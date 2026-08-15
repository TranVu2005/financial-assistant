"""Tests for the Day 18 `compile-plans` CLI."""

from __future__ import annotations

import json
import os
from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from pytest import MonkeyPatch

from financial_report_qa.data.dataset_builder import CELL_SCHEMA, DOCUMENT_SCHEMA, TABLE_SCHEMA
from financial_report_qa.evaluation.week1_release import ReleaseLock
from financial_report_qa.execution.cli import main
from financial_report_qa.retrieval.contracts import RetrievalFilters
from financial_report_qa.retrieval.gold import stable_question_id
from financial_report_qa.retrieval.release import ResolvedRetrievalRelease

_FINGERPRINT = "37a61be7aebde1fbcfe3aca42e6ba4ff37ae87bdd1a9ba6696506bcd188e7d1f"
_TABLE_ID = "tbl_" + "a" * 64
_DOC_ID = "doc_" + "a" * 64


def _fixture_release(tmp_path: Path) -> ResolvedRetrievalRelease:
    release_dir = tmp_path / "release"
    release_dir.mkdir()
    documents = [
        {
            "doc_id": _DOC_ID,
            "repo_id": "repo",
            "revision": "1",
            "relative_path": "DBC/2023/report.txt",
            "company_code": "DBC",
            "report_year": 2023,
            "statement_scope": "consolidated",
            "sha256": "0" * 64,
            "file_size_bytes": 10,
            "encoding": "utf-8",
            "inventory_status": "ready",
            "ruleset_version": "1",
            "normalization_fingerprint": "0" * 64,
        }
    ]
    tables = [
        {
            "table_id": _TABLE_ID,
            "doc_id": _DOC_ID,
            "source_ordinal": 0,
            "title_raw": "Bảng cân đối kế toán",
            "statement_type": "balance_sheet",
            "unit_raw": "VND",
            "unit_normalized": "vnd",
            "line_start": 1,
            "line_end": 2,
            "row_count": 1,
            "column_count": 1,
            "quality_score": 0.9,
            "csv_path": None,
        }
    ]
    cells = [
        {
            "cell_id": "cell_" + "a" * 64,
            "table_id": _TABLE_ID,
            "row_idx": 1,
            "col_idx": 1,
            "row_label_raw": "Tổng tài sản",
            "row_label_canonical": "total_assets",
            "row_group_context_raw": None,
            "column_label_raw": "Năm 2023",
            "column_label_canonical": None,
            "value_raw": "500",
            "value_numeric": Decimal("500"),
            "period": "2023",
            "unit": "VND",
            "source_line_start": 2,
            "source_line_end": 2,
            "extraction_confidence": 0.9,
        }
    ]
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.Table.from_pylist(documents, schema=DOCUMENT_SCHEMA), release_dir / "documents.parquet"
    )
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.Table.from_pylist(tables, schema=TABLE_SCHEMA), release_dir / "tables.parquet"
    )
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.Table.from_pylist(cells, schema=CELL_SCHEMA), release_dir / "cells.parquet"
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
    import financial_report_qa.execution.cli as execution_cli

    monkeypatch.setattr(
        execution_cli, "resolve_retrieval_release", lambda *_args, **_kwargs: release
    )


def _write_fixture_gold(path: Path) -> None:
    filters = RetrievalFilters(company_codes=("DBC",))
    records: list[dict[str, object]] = []
    for number in range(70):
        question = f"Tra cứu tổng tài sản của DBC năm 2023 câu {number:02d}?"
        gold_ids = (_TABLE_ID,)
        records.append(
            {
                "question_id": stable_question_id(question, filters, gold_ids, _FINGERPRINT),
                "question": question,
                "intent": "lookup",
                "filters": filters.model_dump(mode="json"),
                "gold_table_ids": list(gold_ids),
                "reviewed_by": "fixture-reviewer",
                "reviewed_at": "2026-08-08T00:00:00+00:00",
                "gold_evidence": [
                    {
                        "table_id": gold_ids[0],
                        "relative_path": "DBC/2023/report.txt",
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


def _write_execution_config(path: Path) -> None:
    path.write_text(
        "execution:\n"
        "  timeout_seconds: 5\n"
        "  max_rows: 100000\n"
        "  allow_operations:\n"
        "    - lookup\n"
        "    - difference\n"
        "    - growth_rate\n"
        "    - compare\n"
        "    - compare_companies\n"
        "    - ratio\n"
        "    - average\n"
        "    - sum\n"
        "    - rank\n",
        encoding="utf-8",
    )


def test_compile_plans_cli_writes_report(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    release = _fixture_release(tmp_path)
    _patch_release_resolver(monkeypatch, release)
    gold_path = tmp_path / "gold.jsonl"
    _write_fixture_gold(gold_path)
    config_path = tmp_path / "execution.yaml"
    _write_execution_config(config_path)
    output_dir = tmp_path / "out"

    original_cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        exit_code = main(
            [
                "compile-plans",
                "--release-lock",
                str(release.lock_path),
                "--gold-path",
                str(gold_path),
                "--output-dir",
                str(output_dir),
                "--execution-config",
                str(config_path),
            ]
        )
    finally:
        os.chdir(original_cwd)

    assert exit_code == 0
    assert len(list(output_dir.glob("compiled-plans-*.json"))) == 1
    assert len(list(output_dir.glob("compiled-plans-*.md"))) == 1


def test_compile_plans_cli_requires_execution_config(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    release = _fixture_release(tmp_path)
    _patch_release_resolver(monkeypatch, release)
    gold_path = tmp_path / "gold.jsonl"
    _write_fixture_gold(gold_path)
    output_dir = tmp_path / "out"

    original_cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        exit_code = main(
            [
                "compile-plans",
                "--release-lock",
                str(release.lock_path),
                "--gold-path",
                str(gold_path),
                "--output-dir",
                str(output_dir),
            ]
        )
    finally:
        os.chdir(original_cwd)

    assert exit_code == 2
