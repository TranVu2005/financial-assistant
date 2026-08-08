import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from financial_report_qa.core.errors import RetrievalReleaseError
from financial_report_qa.retrieval.release import EXPECTED_FINGERPRINT, resolve_retrieval_release


def _write_release(tmp_path: Path, *, fingerprint: str = EXPECTED_FINGERPRINT) -> Path:
    release = tmp_path / "data" / "release"
    release.mkdir(parents=True)
    for name in ("documents.parquet", "tables.parquet", "cells.parquet"):
        pq.write_table(pa.table({"id": [name]}), release / name)
    (release / "manifest.json").write_text(
        json.dumps(
            {
                "dataset_fingerprint": fingerprint,
                "source_manifest_sha256": "a" * 64,
                "document_count": 1,
                "table_count": 1,
                "cell_count": 1,
            }
        ),
        encoding="utf-8",
    )
    gate = tmp_path / "data" / "gate.json"
    gate.write_text(
        json.dumps({"passed": True, "dataset_fingerprint": fingerprint}), encoding="utf-8"
    )
    lock = tmp_path / "data" / "qa" / "lock.json"
    lock.parent.mkdir(parents=True)
    lock.write_text(
        json.dumps(
            {
                "alias": "dataset-pilot-v1",
                "sampling_version": "week1-pilot-v1",
                "dataset_fingerprint": fingerprint,
                "source_manifest_sha256": "a" * 64,
                "release_path": "data/release",
                "gate_result_path": "data/gate.json",
                "evaluation_inputs_sha256": "b" * 64,
            }
        ),
        encoding="utf-8",
    )
    return lock


def test_resolve_release_accepts_valid_lock(tmp_path: Path) -> None:
    resolved = resolve_retrieval_release(_write_release(tmp_path), repo_root=tmp_path)

    assert resolved.dataset_fingerprint == EXPECTED_FINGERPRINT
    assert resolved.release_dir == tmp_path / "data" / "release"


def test_resolve_release_rejects_fingerprint_mismatch(tmp_path: Path) -> None:
    lock = _write_release(tmp_path, fingerprint="0" * 64)

    with pytest.raises(RetrievalReleaseError, match="fingerprint"):
        resolve_retrieval_release(lock, repo_root=tmp_path)


def test_resolve_release_rejects_path_escape(tmp_path: Path) -> None:
    lock = _write_release(tmp_path)
    data = json.loads(lock.read_text(encoding="utf-8"))
    data["release_path"] = "../outside"
    lock.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(RetrievalReleaseError, match="relative"):
        resolve_retrieval_release(lock, repo_root=tmp_path)
