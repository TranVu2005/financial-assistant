"""Unit tests for week1_release.py: ReleaseLock and publish_release_lock."""

import json
from pathlib import Path

import pytest

from financial_report_qa.core.errors import Week1GateError, Week1GateInputError
from financial_report_qa.evaluation.week1_release import ReleaseLock, publish_release_lock

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FINGERPRINT = "37a61be7aebde1fbcfe3aca42e6ba4ff37ae87bdd1a9ba6696506bcd188e7d1f"
MANIFEST_SHA = "924d165211c63bbfc718b790f217ec356f80236e21fa0d8aa2acb497e186a5cf"
INPUTS_SHA = "abc123def456abc123def456abc123def456abc123def456abc123def456abc1"


def _write_release(tmp_path: Path, fingerprint: str = FINGERPRINT) -> Path:
    release_dir = tmp_path / "release"
    release_dir.mkdir()
    manifest = {
        "dataset_fingerprint": fingerprint,
        "source_manifest_sha256": MANIFEST_SHA,
    }
    (release_dir / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return release_dir


def _write_gate_result(
    tmp_path: Path,
    *,
    passed: bool = True,
    fingerprint: str = FINGERPRINT,
    sampling_version: str = "week1-pilot-v1",
) -> Path:
    gate_dir = tmp_path / "gate"
    gate_dir.mkdir(exist_ok=True)
    result = {
        "passed": passed,
        "dataset_fingerprint": fingerprint,
        "sampling_version": sampling_version,
        "source_manifest_sha256": MANIFEST_SHA,
        "evaluation_inputs_sha256": INPUTS_SHA,
    }
    gate_path = gate_dir / "gate-result.json"
    gate_path.write_text(json.dumps(result), encoding="utf-8")
    return gate_path


# ---------------------------------------------------------------------------
# ReleaseLock model
# ---------------------------------------------------------------------------


def test_release_lock_accepts_valid_payload() -> None:
    lock = ReleaseLock(
        alias="dataset-pilot-v1",
        sampling_version="week1-pilot-v1",
        dataset_fingerprint=FINGERPRINT,
        source_manifest_sha256=MANIFEST_SHA,
        release_path="data/processed/release_v2_37a61be7aebd",
        gate_result_path="data/interim/week1_gate/37a61be7aebd/gate-result.json",
        evaluation_inputs_sha256=INPUTS_SHA,
    )
    assert lock.alias == "dataset-pilot-v1"
    assert lock.dataset_fingerprint == FINGERPRINT


def test_release_lock_rejects_path_with_traversal_in_release_path() -> None:
    with pytest.raises(Exception):
        ReleaseLock(
            alias="dataset-pilot-v1",
            sampling_version="week1-pilot-v1",
            dataset_fingerprint=FINGERPRINT,
            source_manifest_sha256=MANIFEST_SHA,
            release_path="data/../../../etc/shadow",
            gate_result_path="data/interim/gate/gate-result.json",
            evaluation_inputs_sha256=INPUTS_SHA,
        )


# ---------------------------------------------------------------------------
# publish_release_lock
# ---------------------------------------------------------------------------


def test_publish_release_lock_creates_valid_lock(tmp_path: Path) -> None:
    release_dir = _write_release(tmp_path)
    gate_path = _write_gate_result(tmp_path)
    output_path = tmp_path / "dataset-pilot-v1.json"

    lock = publish_release_lock(release_dir, gate_path, output_path)

    assert lock.passed if hasattr(lock, "passed") else True
    assert lock.alias == "dataset-pilot-v1"
    assert lock.dataset_fingerprint == FINGERPRINT
    assert lock.evaluation_inputs_sha256 == INPUTS_SHA
    assert output_path.is_file()

    # Validate the written JSON
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["alias"] == "dataset-pilot-v1"
    assert written["dataset_fingerprint"] == FINGERPRINT


def test_publish_release_lock_idempotent_on_same_content(tmp_path: Path) -> None:
    release_dir = _write_release(tmp_path)
    gate_path = _write_gate_result(tmp_path)
    output_path = tmp_path / "dataset-pilot-v1.json"

    lock1 = publish_release_lock(release_dir, gate_path, output_path)
    # Second call with same content must not raise
    lock2 = publish_release_lock(release_dir, gate_path, output_path)
    assert lock1 == lock2


def test_publish_release_lock_rejects_existing_different_content(tmp_path: Path) -> None:
    release_dir = _write_release(tmp_path)
    gate_path = _write_gate_result(tmp_path)
    output_path = tmp_path / "dataset-pilot-v1.json"

    # Write a different existing lock
    output_path.write_text(
        json.dumps({"alias": "dataset-pilot-v1", "foo": "bar"}), encoding="utf-8"
    )

    with pytest.raises(Week1GateInputError, match="already exists with different content"):
        publish_release_lock(release_dir, gate_path, output_path)


def test_publish_release_lock_rejects_failed_gate(tmp_path: Path) -> None:
    release_dir = _write_release(tmp_path)
    gate_path = _write_gate_result(tmp_path, passed=False)
    output_path = tmp_path / "dataset-pilot-v1.json"

    with pytest.raises(Week1GateInputError, match="did not pass"):
        publish_release_lock(release_dir, gate_path, output_path)


def test_publish_release_lock_rejects_fingerprint_mismatch(tmp_path: Path) -> None:
    release_dir = _write_release(tmp_path, fingerprint="aaaaaa")
    gate_path = _write_gate_result(tmp_path, fingerprint="bbbbbb")
    output_path = tmp_path / "dataset-pilot-v1.json"

    with pytest.raises(Week1GateError, match="Fingerprint mismatch"):
        publish_release_lock(release_dir, gate_path, output_path)


def test_publish_release_lock_rejects_missing_release_manifest(tmp_path: Path) -> None:
    release_dir = tmp_path / "nonexistent_release"
    release_dir.mkdir()
    gate_path = _write_gate_result(tmp_path)
    output_path = tmp_path / "dataset-pilot-v1.json"

    with pytest.raises(Week1GateInputError, match="manifest not found"):
        publish_release_lock(release_dir, gate_path, output_path)


def test_publish_release_lock_rejects_missing_gate_result(tmp_path: Path) -> None:
    release_dir = _write_release(tmp_path)
    output_path = tmp_path / "dataset-pilot-v1.json"

    with pytest.raises(Week1GateInputError, match="Gate result not found"):
        publish_release_lock(release_dir, tmp_path / "nonexistent.json", output_path)


def test_publish_release_lock_rejects_wrong_sampling_version(tmp_path: Path) -> None:
    release_dir = _write_release(tmp_path)
    gate_path = _write_gate_result(tmp_path, sampling_version="week2-pilot-v1")
    output_path = tmp_path / "dataset-pilot-v1.json"

    with pytest.raises(Week1GateInputError, match="sampling_version"):
        publish_release_lock(release_dir, gate_path, output_path)
