"""Fixture-only end-to-end coverage for the offline Day 10 fusion CLI."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pytest import MonkeyPatch

from financial_report_qa.cli import main
from financial_report_qa.evaluation.week1_release import ReleaseLock
from financial_report_qa.retrieval.contracts import RetrievalFilters
from financial_report_qa.retrieval.dense_contracts import DenseEncoderSpec, EncoderName
from financial_report_qa.retrieval.dense_encoder import approved_encoder_spec
from financial_report_qa.retrieval.gold import stable_question_id
from financial_report_qa.retrieval.release import ResolvedRetrievalRelease

_FINGERPRINT = "422df141c935d46bfd14302abec50f32380e6e4c012159f8ad0ae5560c8a446a"
_TABLE_ID = f"tbl_{'a' * 64}"


@dataclass
class _FakeEncoder:
    spec: DenseEncoderSpec

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        return np.asarray([1.0, 0.0], dtype=np.float32)


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
                "table_id": [_TABLE_ID],
                "doc_id": ["doc_a"],
                "title_raw": ["Bảng cân đối kế toán"],
                "statement_type": ["balance_sheet"],
                "unit_normalized": [None],
                "line_start": [1],
                "line_end": [2],
            }
        ),
        release_dir / "tables.parquet",
    )
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.table(
            {
                "table_id": [_TABLE_ID],
                "row_label_canonical": ["total_assets"],
                "row_label_raw": ["Tổng tài sản"],
                "period": ["2023"],
                "unit": ["VND"],
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


def _write_fixture_gold(path: Path) -> None:
    filters = RetrievalFilters(company_codes=("DBC",))
    records: list[dict[str, object]] = []
    for number in range(120):
        question = f"Doanh thu DBC cau hoi {number:02d}?"
        gold_ids = (_TABLE_ID,)
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
                        "table_id": _TABLE_ID,
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
            json.dumps(record, sort_keys=True)
            for record in sorted(records, key=lambda record: str(record["question_id"]))
        )
        + "\n",
        encoding="utf-8",
    )


def _write_bm25_reference(path: Path) -> None:
    source = (
        Path(__file__).parents[3]
        / "artifacts/evaluations/day14/bm25/retrieval-day8-422df141c935.json"
    )
    path.write_bytes(source.read_bytes())


def _patch_release_resolver(monkeypatch: MonkeyPatch, release: ResolvedRetrievalRelease) -> None:
    import financial_report_qa.retrieval.cli as retrieval_cli

    monkeypatch.setattr(
        retrieval_cli, "resolve_retrieval_release", lambda *_args, **_kwargs: release
    )


def _patch_fake_encoder_loader(monkeypatch: MonkeyPatch) -> None:
    import financial_report_qa.retrieval.cli as retrieval_cli

    def load(name: EncoderName, *, local_files_only: bool, device: str = "cpu") -> _FakeEncoder:
        return _FakeEncoder(
            approved_encoder_spec(name).model_copy(update={"dimension": 2, "device": device})
        )

    monkeypatch.setattr(retrieval_cli, "_load_dense_encoder", load, raising=False)


def test_evaluate_fusion_cli_is_network_free_and_replayable(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    release = _fixture_release(tmp_path)
    _patch_release_resolver(monkeypatch, release)
    _patch_fake_encoder_loader(monkeypatch)

    gold_path = tmp_path / "retrieval-gold-v1.jsonl"
    _write_fixture_gold(gold_path)
    bm25_report_path = tmp_path / "bm25.json"
    _write_bm25_reference(bm25_report_path)

    bm25_index_root = tmp_path / "bm25-index"
    assert (
        main(
            [
                "retrieval",
                "build-index",
                "--release-lock",
                "fixture-lock",
                "--output-root",
                str(bm25_index_root),
            ]
        )
        == 0
    )
    bm25_index_dir = next(bm25_index_root.glob("*"))

    corpus_root = tmp_path / "dense-corpus"
    assert (
        main(
            [
                "retrieval",
                "build-dense-corpus",
                "--release-lock",
                "fixture-lock",
                "--output-root",
                str(corpus_root),
            ]
        )
        == 0
    )
    corpus_dir = corpus_root / _FINGERPRINT / "corpus"

    index_root = tmp_path / "dense-index"
    observation_path = tmp_path / "observation.json"
    assert (
        main(
            [
                "retrieval",
                "build-dense-index",
                "--release-lock",
                "fixture-lock",
                "--corpus-dir",
                str(corpus_dir),
                "--encoder",
                "multilingual-e5-small",
                "--output-root",
                str(index_root),
                "--observation-path",
                str(observation_path),
                "--faiss-device",
                "cpu",
                "--local-files-only",
            ]
        )
        == 0
    )
    dense_index_dir = next(index_root.glob("multilingual-e5-small-*"))

    output_dir = tmp_path / "day10"
    cache_dir = tmp_path / "cache"
    exit_code = main(
        [
            "retrieval",
            "evaluate-fusion",
            "--release-lock",
            "fixture-lock",
            "--index-dir",
            str(bm25_index_dir),
            "--corpus-dir",
            str(corpus_dir),
            "--dense-index-dir",
            str(dense_index_dir),
            "--encoder",
            "multilingual-e5-small",
            "--gold-path",
            str(gold_path),
            "--cache-dir",
            str(cache_dir),
            "--bm25-report",
            str(bm25_report_path),
            "--output-dir",
            str(output_dir),
        ]
    )
    assert exit_code == 0
    json_reports = list(output_dir.glob("retrieval-day10-fusion-*.json"))
    assert len(json_reports) == 1
    first_bytes = json_reports[0].read_bytes()
    report = json.loads(first_bytes)
    assert len(report["grid"]) == 6
    assert report["default_system"] in ("bm25-v3", "fusion")

    output_dir_replay = tmp_path / "day10-replay"
    assert (
        main(
            [
                "retrieval",
                "evaluate-fusion",
                "--release-lock",
                "fixture-lock",
                "--index-dir",
                str(bm25_index_dir),
                "--corpus-dir",
                str(corpus_dir),
                "--dense-index-dir",
                str(dense_index_dir),
                "--encoder",
                "multilingual-e5-small",
                "--gold-path",
                str(gold_path),
                "--cache-dir",
                str(cache_dir),
                "--bm25-report",
                str(bm25_report_path),
                "--output-dir",
                str(output_dir_replay),
            ]
        )
        == 0
    )
    replayed = list(output_dir_replay.glob("retrieval-day10-fusion-*.json"))
    assert replayed[0].read_bytes() == first_bytes


def test_evaluate_fusion_cli_fails_closed_on_missing_bm25_report(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    release = _fixture_release(tmp_path)
    _patch_release_resolver(monkeypatch, release)
    _patch_fake_encoder_loader(monkeypatch)
    gold_path = tmp_path / "retrieval-gold-v1.jsonl"
    _write_fixture_gold(gold_path)

    exit_code = main(
        [
            "retrieval",
            "evaluate-fusion",
            "--release-lock",
            "fixture-lock",
            "--index-dir",
            str(tmp_path / "missing-bm25-index"),
            "--corpus-dir",
            str(tmp_path / "missing-corpus"),
            "--dense-index-dir",
            str(tmp_path / "missing-dense-index"),
            "--encoder",
            "multilingual-e5-small",
            "--gold-path",
            str(gold_path),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--bm25-report",
            str(tmp_path / "missing-bm25.json"),
            "--output-dir",
            str(tmp_path / "day10"),
        ]
    )
    assert exit_code == 2
