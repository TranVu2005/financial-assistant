"""Fail-closed planning and reversible quarantine for Day 9 data artifacts."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

_CANDIDATE_PATHS = (
    Path("data/interim/week1_gate_attempts"),
    Path("data/interim/week1_gate_replay"),
    Path("data/processed/release_v2_37a61be7aeba"),
    Path("data/processed/release_v2_7868718f2547"),
    Path("data/processed/release_v2_7fc5d5d57bf6"),
    Path("data/processed/v2_remediated"),
)
_CANDIDATE_DATA_PATHS = tuple(path.relative_to("data") for path in _CANDIDATE_PATHS)
_SKIPPED_ARTIFACT_DIRECTORIES = {
    ".cache",
    ".git",
    ".mypy_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
}


@dataclass(frozen=True)
class CleanupEntry:
    """One immutable cleanup decision for a candidate path."""

    path: Path
    reason: str
    status: Literal["approved", "blocked", "missing"]
    byte_count: int
    detail: str


@dataclass(frozen=True)
class CleanupPlan:
    """A timestamped, inspectable collection of cleanup decisions."""

    entries: list[CleanupEntry]
    generated_at: str


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _artifact_references(repo_root: Path, candidate: Path) -> list[Path]:
    """Return any readable repository text artifact mentioning a candidate name."""
    references: list[Path] = []
    raw_root = repo_root / "data/raw"
    quarantine_root = repo_root / "data/quarantine"
    for directory, directories, filenames in os.walk(repo_root, topdown=True):
        current = Path(directory)
        directories[:] = [
            name
            for name in directories
            if name not in _SKIPPED_ARTIFACT_DIRECTORIES
            and not (current / name).is_symlink()
            and not _inside((current / name).resolve(strict=False), candidate)
            and not _inside((current / name).resolve(strict=False), raw_root)
            and not _inside((current / name).resolve(strict=False), quarantine_root)
        ]
        for filename in filenames:
            path = current / filename
            if path.is_symlink() or _inside(path.resolve(strict=False), candidate):
                continue
            try:
                with path.open("rb") as artifact:
                    sample = artifact.read(4096)
            except (OSError, UnicodeDecodeError):
                raise OSError(f"cannot read reference artifact {path}") from None
            if b"\0" in sample:
                continue
            try:
                sample.decode("utf-8")
                content = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                raise OSError(f"cannot read reference artifact {path}") from None
            if candidate.name in content:
                references.append(path)
    return references


def _candidate_bytes_and_manifests(candidate: Path, data_root: Path) -> int:
    """Read candidate manifests and count bytes, rejecting inaccessible tree members."""
    total = 0
    errors: list[OSError] = []

    def onerror(error: OSError) -> None:
        errors.append(error)

    for directory, _directories, filenames in os.walk(candidate, onerror=onerror):
        current = Path(directory)
        try:
            resolved_directory = current.resolve(strict=True)
        except OSError as exc:
            raise OSError(f"cannot resolve candidate directory {current}: {exc}") from exc
        if not _inside(resolved_directory, data_root):
            raise OSError(f"candidate member resolves outside data/: {current}")
        for filename in filenames:
            path = current / filename
            try:
                resolved_path = path.resolve(strict=True)
                if not _inside(resolved_path, data_root):
                    raise OSError(f"candidate member resolves outside data/: {path}")
                total += path.stat().st_size
                if path.suffix.lower() == ".json":
                    path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                raise OSError(f"cannot read candidate {path}: {exc}") from exc
    if errors:
        raise OSError(f"cannot inspect candidate {candidate}: {errors[0]}")
    return total


def plan_day9_cleanup(repo_root: Path) -> CleanupPlan:
    """Inspect only fixed Day 9 candidates and produce fail-closed decisions."""
    resolved_root = repo_root.resolve(strict=True)
    data_root = (resolved_root / "data").resolve(strict=False)
    entries: list[CleanupEntry] = []
    for relative_path in _CANDIDATE_PATHS:
        candidate = resolved_root / relative_path
        try:
            if candidate.is_symlink():
                entries.append(
                    CleanupEntry(
                        candidate,
                        "symlink candidate is unsafe",
                        "blocked",
                        0,
                        "candidate must be an immutable repository-relative path",
                    )
                )
                continue
            if not candidate.exists():
                entries.append(
                    CleanupEntry(candidate, "candidate absent", "missing", 0, "path does not exist")
                )
                continue
            resolved_candidate = candidate.resolve(strict=True)
            if not _inside(resolved_candidate, data_root):
                entries.append(
                    CleanupEntry(
                        candidate,
                        "outside protected data root",
                        "blocked",
                        0,
                        f"resolves outside {data_root}",
                    )
                )
                continue
            byte_count = _candidate_bytes_and_manifests(resolved_candidate, data_root)
            references = _artifact_references(resolved_root, resolved_candidate)
            if references:
                entries.append(
                    CleanupEntry(
                        candidate,
                        "referenced by repository artifact",
                        "blocked",
                        byte_count,
                        ", ".join(
                            path.relative_to(resolved_root).as_posix() for path in references
                        ),
                    )
                )
                continue
            entries.append(
                CleanupEntry(
                    candidate,
                    "rebuildable interim data or superseded release",
                    "approved",
                    byte_count,
                    "no protected reference found",
                )
            )
        except (OSError, UnicodeDecodeError) as exc:
            entries.append(
                CleanupEntry(candidate, "unreadable candidate", "blocked", 0, str(exc))
            )
    return CleanupPlan(
        entries=entries,
        generated_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    )


def _data_root_for(source: Path) -> Path:
    for parent in (source, *source.parents):
        if parent.name == "data":
            return parent
    raise ValueError(f"cleanup source is not inside data/: {source}")


def quarantine_day9_cleanup(plan: CleanupPlan, quarantine_root: Path) -> list[Path]:
    """Move approved entries into a timestamped, data-local quarantine directory."""
    approved = [entry for entry in plan.entries if entry.status == "approved"]
    validated: list[tuple[Path, Path, Path]] = []
    resolved_quarantine = quarantine_root.resolve(strict=False)
    for entry in approved:
        if entry.path.is_symlink():
            raise ValueError(f"cleanup source must not be a symlink: {entry.path}")
        source = entry.path.resolve(strict=True)
        data_root = _data_root_for(source)
        if not _inside(source, data_root) or not _inside(resolved_quarantine, data_root):
            raise ValueError("cleanup sources and quarantine root must resolve inside data/")
        destination_relative = source.relative_to(data_root)
        if destination_relative not in _CANDIDATE_DATA_PATHS:
            raise ValueError(f"cleanup source is not an approved candidate path: {source}")
        validated.append((source, data_root, destination_relative))
    if not validated:
        return []

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    timestamp_root = resolved_quarantine / timestamp
    counter = 1
    while timestamp_root.exists():
        timestamp_root = resolved_quarantine / f"{timestamp}-{counter}"
        counter += 1
    destinations: list[Path] = []
    for _source, data_root, relative_destination in validated:
        destination = (timestamp_root / relative_destination).resolve(strict=False)
        if not _inside(destination, data_root):
            raise ValueError("cleanup destination resolves outside data/")
        if destination.exists():
            raise ValueError(f"cleanup destination already exists: {destination}")
        destinations.append(destination)

    timestamp_root.mkdir(parents=True, exist_ok=False)
    for (
        source, _data_root, _relative_destination
    ), destination in zip(validated, destinations, strict=True):
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))
    return destinations
