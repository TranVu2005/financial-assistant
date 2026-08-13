"""Fixture-only end-to-end coverage for the offline Day 9 dense CLI."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pytest import CaptureFixture, MonkeyPatch

from financial_report_qa.cli import main
from financial_report_qa.evaluation.week1_release import ReleaseLock
from financial_report_qa.retrieval.contracts import RetrievalFilters
from financial_report_qa.retrieval.dense_contracts import DenseEncoderSpec, EncoderName
from financial_report_qa.retrieval.dense_encoder import approved_encoder_spec
from financial_report_qa.retrieval.evaluation import RetrievalEvaluationReport, RetrievalMetrics
from financial_report_qa.retrieval.gold import stable_question_id
from financial_report_qa.retrieval.release import ResolvedRetrievalRelease

_FINGERPRINT = "422df141c935d46bfd14302abec50f32380e6e4c012159f8ad0ae5560c8a446a"
_TABLE_IDS = tuple(f"tbl_{value * 64}" for value in ("a", "b", "d"))


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


def _write_fixture_gold(path: Path) -> None:
    filters = RetrievalFilters(company_codes=("VCB",))
    records: list[dict[str, object]] = []
    for number in range(70):
        question = f"Doanh thu VCB cau hoi {number:02d}?"
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
            for record in sorted(records, key=lambda record: str(record["question_id"]))
        )
        + "\n",
        encoding="utf-8",
    )


def _write_bm25_reference(path: Path) -> None:
    metrics = RetrievalMetrics(
        true_positive=105,
        precision=0.1499999999999999,
        recall=0.880952380952381,
        f2=0.4224545295973871,
    )
    report = RetrievalEvaluationReport(
        dataset_fingerprint=_FINGERPRINT,
        question_count=70,
        macro=metrics,
        by_intent={"compare": metrics, "growth": metrics, "lookup": metrics},
        per_question=(),
    )
    path.write_text(json.dumps(report.model_dump(mode="json")), encoding="utf-8")


def _patch_release_resolver(monkeypatch: MonkeyPatch, release: ResolvedRetrievalRelease) -> None:
    import financial_report_qa.retrieval.cli as retrieval_cli

    monkeypatch.setattr(
        retrieval_cli, "resolve_retrieval_release", lambda *_args, **_kwargs: release
    )


def _patch_fake_encoder_loader(monkeypatch: MonkeyPatch) -> list[str]:
    import financial_report_qa.retrieval.cli as retrieval_cli

    loaded: list[str] = []

    def load(
        name: EncoderName, *, local_files_only: bool, device: str = "cpu"
    ) -> _FakeEncoder:
        loaded.append(name)
        return _FakeEncoder(
            approved_encoder_spec(name).model_copy(update={"dimension": 2, "device": device})
        )

    monkeypatch.setattr(retrieval_cli, "_load_dense_encoder", load, raising=False)
    return loaded


def test_day9_dense_cli_fixture_lifecycle_is_network_free_and_replayable(
    tmp_path: Path, monkeypatch: MonkeyPatch, capsys: CaptureFixture[str]
) -> None:
    """Commands must use only explicit local artifacts and fail closed on index corruption."""
    release = _fixture_release(tmp_path)
    _patch_release_resolver(monkeypatch, release)
    loaded = _patch_fake_encoder_loader(monkeypatch)
    gold_path = tmp_path / "retrieval-gold-v1.jsonl"
    _write_fixture_gold(gold_path)
    bm25_path = tmp_path / "bm25.json"
    _write_bm25_reference(bm25_path)
    corpus_root = tmp_path / "corpora"
    index_root = tmp_path / "indexes"
    corpus_dir = corpus_root / _FINGERPRINT / "corpus"

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
    for encoder, label in (("bge-m3", "bge"), ("multilingual-e5-small", "e5")):
        observation = tmp_path / f"{label}-observation.json"
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
                    encoder,
                    "--output-root",
                    str(index_root),
                    "--observation-path",
                    str(observation),
                    "--faiss-device",
                    "cpu",
                    "--local-files-only",
                ]
            )
            == 0
        )
        observed = json.loads(observation.read_text(encoding="utf-8"))
        assert observed["faiss_device"] == "cpu"
        assert observed["faiss_gpu_count"] == 0
        index_dir = next(index_root.glob(f"{encoder}-*"))
        report_path = tmp_path / f"{label}-report.json"
        assert (
            main(
                [
                    "retrieval",
                    "evaluate-dense",
                    "--release-lock",
                    "fixture-lock",
                    "--corpus-dir",
                    str(corpus_dir),
                    "--index-dir",
                    str(index_dir),
                    "--encoder",
                    encoder,
                    "--gold-path",
                    str(gold_path),
                    "--cache-dir",
                    str(tmp_path / "cache" / label),
                    "--observation-path",
                    str(observation),
                    "--output-path",
                    str(report_path),
                ]
            )
            == 0
        )

    assert (
        main(
            [
                "retrieval",
                "compare-day9",
                "--release-lock",
                "fixture-lock",
                "--bm25-report",
                str(bm25_path),
                "--bge-report",
                str(tmp_path / "bge-report.json"),
                "--e5-report",
                str(tmp_path / "e5-report.json"),
                "--output-dir",
                str(tmp_path / "comparison"),
            ]
        )
        == 0
    )
    comparison_path = tmp_path / "comparison" / f"retrieval-day9-dense-{_FINGERPRINT[:12]}.json"
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    assert list(comparison["systems"]) == ["bge-m3", "bm25-v3", "multilingual-e5-small"]
    assert loaded == ["bge-m3", "bge-m3", "multilingual-e5-small", "multilingual-e5-small"]

    bge_index = next(index_root.glob("bge-m3-*"))
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
                "bge-m3",
                "--output-root",
                str(index_root),
                "--observation-path",
                str(tmp_path / "bge-observation.json"),
                "--faiss-device",
                "cuda",
                "--local-files-only",
            ]
        )
        == 2
    )
    (bge_index / "index.faiss").write_bytes((bge_index / "index.faiss").read_bytes() + b"x")
    assert (
        main(
            [
                "retrieval",
                "evaluate-dense",
                "--release-lock",
                "fixture-lock",
                "--corpus-dir",
                str(corpus_dir),
                "--index-dir",
                str(bge_index),
                "--encoder",
                "bge-m3",
                "--gold-path",
                str(gold_path),
                "--cache-dir",
                str(tmp_path / "cache" / "bge"),
                "--observation-path",
                str(tmp_path / "bge-observation.json"),
                "--output-path",
                str(tmp_path / "corrupt.json"),
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert "retrieval error:" in captured.err
    assert "Refusing CUDA build into existing dense index target" in captured.err
    assert "dense-build: 3/3" in captured.out
    assert "dense-build: complete" in captured.out


def test_cleanup_day9_data_cli_is_dry_by_default_and_quarantines_on_apply(
    tmp_path: Path, capsys: CaptureFixture[str]
) -> None:
    """Removing --apply gating would mutate the candidate during a dry run."""
    candidate = tmp_path / "data/interim/week1_gate_attempts"
    candidate.mkdir(parents=True)
    (candidate / "attempt.json").write_text("fixture", encoding="utf-8")
    blocked = tmp_path / "data/interim/week1_gate_replay"
    blocked.mkdir(parents=True)
    (blocked / "manifest.json").write_text("{}", encoding="utf-8")
    (tmp_path / "plan.md").write_text("week1_gate_replay remains locked", encoding="utf-8")
    quarantine_root = tmp_path / "data/quarantine/day9-cleanup"

    assert (
        main(
            [
                "retrieval",
                "cleanup-day9-data",
                "--repo-root",
                str(tmp_path),
                "--quarantine-root",
                str(quarantine_root),
            ]
        )
        == 0
    )
    dry_run = capsys.readouterr().out
    assert '"status": "approved"' in dry_run
    assert candidate.exists()

    assert (
        main(
            [
                "retrieval",
                "cleanup-day9-data",
                "--repo-root",
                str(tmp_path),
                "--quarantine-root",
                str(quarantine_root),
                "--apply",
            ]
        )
        == 2
    )
    applied = capsys.readouterr().out
    assert '"action": "moved"' in applied
    assert not candidate.exists()
    assert blocked.exists()
    assert any(path.name == "week1_gate_attempts" for path in quarantine_root.rglob("*"))
