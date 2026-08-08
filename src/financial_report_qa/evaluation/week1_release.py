"""Release lock publication for Week 1 Quality Gate dataset."""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

from financial_report_qa.core.errors import Week1GateError, Week1GateInputError


class ReleaseLock(BaseModel):
    """Immutable binding of alias, dataset fingerprint, manifest hash and canonical gate result."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    alias: Literal["dataset-pilot-v1"]
    sampling_version: Literal["week1-pilot-v1"]
    dataset_fingerprint: str
    source_manifest_sha256: str
    release_path: str
    gate_result_path: str
    evaluation_inputs_sha256: str

    @field_validator("release_path", "gate_result_path")
    @classmethod
    def validate_safe_path(cls, v: str) -> str:
        """Reject empty values and path traversal components."""
        if not v:
            raise ValueError("path must not be empty")
        parts = v.replace("\\", "/").split("/")
        if any(part == ".." for part in parts):
            raise ValueError(f"path must not contain '..': {v!r}")
        return v



def publish_release_lock(
    release_path: Path,
    gate_result_path: Path,
    output_path: Path,
) -> ReleaseLock:
    """Read gate result and publish an immutable release lock JSON.

    Raises:
        Week1GateInputError: release or gate result not found, or gate did not pass.
        Week1GateError: fingerprint mismatch between release and gate result.
    """
    # Read release manifest
    manifest_path = release_path / "manifest.json"
    if not manifest_path.is_file():
        raise Week1GateInputError(
            f"Release manifest not found: {manifest_path}"
        )
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    release_fingerprint: str = manifest.get("dataset_fingerprint", "")
    source_manifest_sha256: str = manifest.get("source_manifest_sha256", "")

    if not release_fingerprint:
        raise Week1GateInputError("Release manifest is missing 'dataset_fingerprint'")

    # Read gate result
    if not gate_result_path.is_file():
        raise Week1GateInputError(f"Gate result not found: {gate_result_path}")
    with gate_result_path.open("r", encoding="utf-8") as f:
        gate_result = json.load(f)

    if not gate_result.get("passed", False):
        raise Week1GateInputError(
            "Cannot create release lock: the gate result did not pass"
        )

    gate_fingerprint: str = gate_result.get("dataset_fingerprint", "")
    if gate_fingerprint != release_fingerprint:
        raise Week1GateError(
            f"Fingerprint mismatch: release has {release_fingerprint!r} "
            f"but gate result has {gate_fingerprint!r}"
        )

    evaluation_inputs_sha256: str = gate_result.get("evaluation_inputs_sha256", "")
    gate_sampling_version: str = gate_result.get("sampling_version", "")

    if gate_sampling_version != "week1-pilot-v1":
        raise Week1GateInputError(
            f"Expected sampling_version 'week1-pilot-v1', got {gate_sampling_version!r}"
        )

    # Build relative paths for portability
    try:
        rel_release = _safe_relative(release_path)
    except ValueError:
        rel_release = release_path.as_posix()
    try:
        rel_gate = _safe_relative(gate_result_path)
    except ValueError:
        rel_gate = gate_result_path.as_posix()

    lock = ReleaseLock(
        alias="dataset-pilot-v1",
        sampling_version="week1-pilot-v1",
        dataset_fingerprint=release_fingerprint,
        source_manifest_sha256=source_manifest_sha256,
        release_path=rel_release,
        gate_result_path=rel_gate,
        evaluation_inputs_sha256=evaluation_inputs_sha256,
    )

    # Guard against overwriting a non-identical existing lock
    if output_path.is_file():
        existing = json.loads(output_path.read_bytes())
        new_dict = json.loads(lock.model_dump_json())
        if existing != new_dict:
            raise Week1GateInputError(
                f"Output path {output_path} already exists with different content; "
                "delete it manually before publishing a new lock"
            )
        return lock

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(lock.model_dump_json(indent=2).encode("utf-8") + b"\n")
    return lock


def _safe_relative(path: Path) -> str:
    """Return a POSIX-style relative path from CWD, or raise ValueError."""
    try:
        rel = path.resolve().relative_to(Path.cwd().resolve())
        return rel.as_posix()
    except ValueError:
        raise ValueError(f"Path {path} is not relative to cwd {Path.cwd()}")
