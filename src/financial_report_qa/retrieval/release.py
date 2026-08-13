"""Resolution and verification of the immutable Week 1 retrieval release."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq
from pydantic import ValidationError

from financial_report_qa.core.errors import RetrievalReleaseError
from financial_report_qa.evaluation.week1_release import ReleaseLock

EXPECTED_FINGERPRINT = "422df141c935d46bfd14302abec50f32380e6e4c012159f8ad0ae5560c8a446a"
REQUIRED_PARQUETS = ("documents.parquet", "tables.parquet", "cells.parquet")


@dataclass(frozen=True)
class ResolvedRetrievalRelease:
    lock: ReleaseLock
    dataset_fingerprint: str
    release_dir: Path
    gate_result_path: Path
    lock_path: Path
    manifest: dict[str, object]
    lock_sha256: str


def _resolve_repo_path(repo_root: Path, relative_path: str, *, label: str) -> Path:
    path = Path(relative_path)
    if path.is_absolute() or ".." in path.parts:
        raise RetrievalReleaseError(f"{label} must be a safe repository-relative path")
    resolved = (repo_root / path).resolve()
    try:
        resolved.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise RetrievalReleaseError(f"{label} escapes repository root") from exc
    return resolved


def resolve_retrieval_release(lock_path: Path, *, repo_root: Path) -> ResolvedRetrievalRelease:
    """Resolve the approved Week 1 release and validate its binding invariants."""
    root = repo_root.resolve()
    resolved_lock = lock_path.resolve()
    try:
        resolved_lock.relative_to(root)
    except ValueError as exc:
        raise RetrievalReleaseError(
            "Retrieval release lock must stay within repository root"
        ) from exc
    if not resolved_lock.is_file():
        raise RetrievalReleaseError(f"Retrieval release lock not found: {resolved_lock}")
    try:
        lock_bytes = resolved_lock.read_bytes()
        lock = ReleaseLock.model_validate_json(lock_bytes)
    except (OSError, ValidationError) as exc:
        raise RetrievalReleaseError(
            f"Invalid retrieval release lock or repository-relative path: {resolved_lock}"
        ) from exc
    if lock.dataset_fingerprint != EXPECTED_FINGERPRINT:
        raise RetrievalReleaseError(
            "Release lock fingerprint does not match the approved Day 8 fingerprint"
        )

    release_dir = _resolve_repo_path(root, lock.release_path, label="release_path")
    gate_result_path = _resolve_repo_path(root, lock.gate_result_path, label="gate_result_path")
    manifest_path = release_dir / "manifest.json"
    if not manifest_path.is_file() or not gate_result_path.is_file():
        raise RetrievalReleaseError("Release manifest or passed gate result is missing")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        gate_result = json.loads(gate_result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RetrievalReleaseError("Release manifest or gate result is invalid JSON") from exc
    if manifest.get("dataset_fingerprint") != EXPECTED_FINGERPRINT:
        raise RetrievalReleaseError(
            "Release manifest fingerprint does not match the approved fingerprint"
        )
    if (
        not gate_result.get("passed")
        or gate_result.get("dataset_fingerprint") != EXPECTED_FINGERPRINT
    ):
        raise RetrievalReleaseError("Week 1 gate result is not a passed approved release")
    if manifest.get("source_manifest_sha256") != lock.source_manifest_sha256:
        raise RetrievalReleaseError(
            "Release manifest source manifest hash does not match release lock"
        )
    expected_counts = {
        "documents.parquet": manifest.get("document_count"),
        "tables.parquet": manifest.get("table_count"),
        "cells.parquet": manifest.get("cell_count"),
    }
    for filename in REQUIRED_PARQUETS:
        parquet_path = release_dir / filename
        if not parquet_path.is_file():
            raise RetrievalReleaseError(f"Required release artifact is missing: {filename}")
        try:
            row_count = pq.read_metadata(parquet_path).num_rows  # type: ignore[no-untyped-call]
            if row_count <= 0:
                raise RetrievalReleaseError(f"Release artifact has no rows: {filename}")
            if (
                not isinstance(expected_counts[filename], int)
                or row_count != expected_counts[filename]
            ):
                raise RetrievalReleaseError(f"Release manifest count differs from {filename}")
        except RetrievalReleaseError:
            raise
        except Exception as exc:
            raise RetrievalReleaseError(f"Invalid Parquet release artifact: {filename}") from exc
    return ResolvedRetrievalRelease(
        lock=lock,
        dataset_fingerprint=EXPECTED_FINGERPRINT,
        release_dir=release_dir,
        gate_result_path=gate_result_path,
        lock_path=resolved_lock,
        manifest=manifest,
        lock_sha256=hashlib.sha256(lock_bytes).hexdigest(),
    )
