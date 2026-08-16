"""Tests for the Day 22 `submission export`/`submission validate` CLI."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
from pytest import MonkeyPatch

from financial_report_qa.core.config import LLMSettings
from financial_report_qa.data.dataset_builder import CELL_SCHEMA, DOCUMENT_SCHEMA, TABLE_SCHEMA
from financial_report_qa.evaluation.week1_release import ReleaseLock
from financial_report_qa.planning.llm_client import LLMClient
from financial_report_qa.retrieval.contracts import (
    MetricLabelObservation,
    TableDocument,
    TableMetadata,
)
from financial_report_qa.retrieval.index import build_bm25_index, save_bm25_index
from financial_report_qa.retrieval.release import ResolvedRetrievalRelease
from financial_report_qa.submission.cli import main

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
            "relative_path": "DBC/2023/DBC_report_extracted.txt",
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
    import financial_report_qa.submission.cli as submission_cli

    monkeypatch.setattr(submission_cli, "resolve_retrieval_release", lambda *_a, **_k: release)


def _write_bm25_index(index_dir: Path) -> None:
    document = TableDocument(
        table_id=_TABLE_ID,
        doc_id=_DOC_ID,
        text="company_code: DBC\nperiod: 2023\nTổng tài sản | 2023 | 500",
        metadata=TableMetadata(
            table_id=_TABLE_ID,
            doc_id=_DOC_ID,
            company_code="DBC",
            periods=("2023",),
            statement_type="balance_sheet",
            source_path="a.txt",
            line_start=1,
            line_end=3,
        ),
        metric_labels=(MetricLabelObservation(canonical="total_assets", raw=None),),
    )
    index = build_bm25_index((document,), dataset_fingerprint=_FINGERPRINT)
    save_bm25_index(index, index_dir)


def _write_execution_config(path: Path) -> None:
    path.write_text(
        "execution:\n  timeout_seconds: 5\n  max_rows: 100000\n  allow_operations:\n    - lookup\n",
        encoding="utf-8",
    )


def _write_questions(path: Path) -> None:
    path.write_text(
        '{"id": 1, "question": "Tổng tài sản của DBC năm 2023 là bao nhiêu?"}\n',
        encoding="utf-8",
    )


def test_export_then_validate_roundtrip(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    release = _fixture_release(tmp_path)
    _patch_release_resolver(monkeypatch, release)
    index_dir = tmp_path / "index"
    _write_bm25_index(index_dir)
    config_path = tmp_path / "execution.yaml"
    _write_execution_config(config_path)
    questions_path = tmp_path / "questions.jsonl"
    _write_questions(questions_path)
    output_zip = tmp_path / "submission.zip"
    report_dir = tmp_path / "report"

    exit_code = main(
        [
            "export",
            "--release-lock",
            "lock.json",
            "--bm25-index",
            str(index_dir),
            "--questions-path",
            str(questions_path),
            "--execution-config",
            str(config_path),
            "--output-zip",
            str(output_zip),
            "--report-dir",
            str(report_dir),
        ]
    )
    assert exit_code == 0
    assert output_zip.exists()
    report_files = list(report_dir.glob("submission-export-*.json"))
    assert len(report_files) == 1
    report_payload = json.loads(report_files[0].read_text(encoding="utf-8"))
    assert report_payload["answered_count"] == 1

    exit_code = main(
        ["validate", "--zip-path", str(output_zip), "--report-path", str(report_files[0])]
    )
    assert exit_code == 0


def _write_llm_config(path: Path) -> None:
    path.write_text(
        "llm:\n"
        "  base_url: http://127.0.0.1:8080/v1\n"
        "  model: test-model\n"
        "  timeout_seconds: 5\n"
        "  max_output_tokens: 160\n"
        "  temperature: 0.0\n"
        "  context_length: 4096\n"
        "  json_schema_constrained: true\n",
        encoding="utf-8",
    )


def test_export_with_llm_config_answers_a_rule_planner_abstain(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """`--llm-config` must route a rule-planner abstain through the LLM
    fallback (never a live server here -- `LLMClient` is monkeypatched to use
    an in-memory `httpx.MockTransport`, the same pattern test_plan_router.py
    uses)."""
    release = _fixture_release(tmp_path)
    _patch_release_resolver(monkeypatch, release)
    index_dir = tmp_path / "index"
    _write_bm25_index(index_dir)
    config_path = tmp_path / "execution.yaml"
    _write_execution_config(config_path)
    llm_config_path = tmp_path / "llm.yaml"
    _write_llm_config(llm_config_path)
    questions_path = tmp_path / "questions.jsonl"
    questions_path.write_text(
        '{"id": 1, "question": "Tổng lợi thế cạnh tranh của DBC năm 2023 là bao nhiêu?"}\n',
        encoding="utf-8",
    )
    output_zip = tmp_path / "submission.zip"
    report_dir = tmp_path / "report"

    valid_plan = json.dumps(
        {
            "operation": "lookup",
            "companies": ["DBC"],
            "periods": ["2023"],
            "metric": {"canonical": "total_assets"},
        }
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": valid_plan}}]})

    import financial_report_qa.submission.cli as submission_cli

    def fake_llm_client(settings: LLMSettings, **kwargs: object) -> LLMClient:
        return LLMClient(settings, transport=httpx.MockTransport(handler), max_retries=1)

    monkeypatch.setattr(submission_cli, "LLMClient", fake_llm_client)

    exit_code = main(
        [
            "export",
            "--release-lock",
            "lock.json",
            "--bm25-index",
            str(index_dir),
            "--questions-path",
            str(questions_path),
            "--execution-config",
            str(config_path),
            "--llm-config",
            str(llm_config_path),
            "--output-zip",
            str(output_zip),
            "--report-dir",
            str(report_dir),
        ]
    )
    assert exit_code == 0
    report_files = list(report_dir.glob("submission-export-*.json"))
    report_payload = json.loads(report_files[0].read_text(encoding="utf-8"))
    assert report_payload["answered_count"] == 1
    assert report_payload["outcomes"][0]["plan_source"] == "llm"


def test_export_rejects_mismatched_index_fingerprint(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    release = _fixture_release(tmp_path)
    _patch_release_resolver(monkeypatch, release)
    index_dir = tmp_path / "index"
    document = TableDocument(
        table_id=_TABLE_ID,
        doc_id=_DOC_ID,
        text="dummy",
        metadata=TableMetadata(
            table_id=_TABLE_ID,
            doc_id=_DOC_ID,
            source_path="a.txt",
            line_start=1,
            line_end=3,
        ),
    )
    save_bm25_index(build_bm25_index((document,), dataset_fingerprint="0" * 64), index_dir)
    config_path = tmp_path / "execution.yaml"
    _write_execution_config(config_path)
    questions_path = tmp_path / "questions.jsonl"
    _write_questions(questions_path)

    exit_code = main(
        [
            "export",
            "--release-lock",
            "lock.json",
            "--bm25-index",
            str(index_dir),
            "--questions-path",
            str(questions_path),
            "--execution-config",
            str(config_path),
            "--output-zip",
            str(tmp_path / "out.zip"),
            "--report-dir",
            str(tmp_path / "report"),
        ]
    )
    assert exit_code == 2
