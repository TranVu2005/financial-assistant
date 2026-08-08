"""Resolution and verification of the immutable Week 1 retrieval release."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq
from pydantic import ValidationError

from financial_report_qa.core.errors import RetrievalReleaseError
from financial_report_qa.evaluation.week1_release import ReleaseLock

EXPECTED_FINGERPRINT = "37a61be7aebde1fbcfe3aca42e6ba4ff37ae87bdd1a9ba6696506bcd188e7d1f"
REQUIRED_PARQUETS = ("documents.parquet", "tables.parquet", "cells.parquet")


@dataclass(frozen=True)
class ResolvedRetrievalRelease:
    dataset_fingerprint: str
    release_dir: Path
    gate_result_path: Path
    lock_path: Path


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
    if not resolved_lock.is_file():
        raise RetrievalReleaseError(f"Retrieval release lock not found: {resolved_lock}")
    try:
        lock = ReleaseLock.model_validate_json(resolved_lock.read_text(encoding="utf-8"))
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
    for filename in REQUIRED_PARQUETS:
        parquet_path = release_dir / filename
        if not parquet_path.is_file():
            raise RetrievalReleaseError(f"Required release artifact is missing: {filename}")
        try:
            if pq.read_metadata(parquet_path).num_rows <= 0:  # type: ignore[no-untyped-call]
                raise RetrievalReleaseError(f"Release artifact has no rows: {filename}")
        except RetrievalReleaseError:
            raise
        except Exception as exc:
            raise RetrievalReleaseError(f"Invalid Parquet release artifact: {filename}") from exc
    return ResolvedRetrievalRelease(
        dataset_fingerprint=EXPECTED_FINGERPRINT,
        release_dir=release_dir,
        gate_result_path=gate_result_path,
        lock_path=resolved_lock,
    )
