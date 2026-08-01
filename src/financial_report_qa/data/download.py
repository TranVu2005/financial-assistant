"""Resumable, revision-pinned downloads for Hugging Face datasets."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download
from huggingface_hub.file_download import DryRunFileInfo

DEFAULT_REPO_ID = "tinixai/ocr_annual_financials"
DEFAULT_TARGET_DIR = Path("data/raw/ocr_annual_financials")
GIB = 1024**3

SnapshotDownloadFn = Callable[..., list[DryRunFileInfo] | str]
FreeSpaceFn = Callable[[Path], int]


@dataclass(frozen=True)
class DownloadRequest:
    """User-selected source, destination, and safety settings."""

    target_dir: Path
    repo_id: str = DEFAULT_REPO_ID
    revision: str = "main"
    manifest_path: Path | None = None
    include_patterns: tuple[str, ...] = ()
    exclude_patterns: tuple[str, ...] = ()
    max_workers: int = 8
    reserve_bytes: int = 20 * GIB

    def __post_init__(self) -> None:
        if not self.repo_id.strip():
            raise ValueError("repo_id must not be empty")
        if not self.revision.strip():
            raise ValueError("revision must not be empty")
        if self.max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        if self.reserve_bytes < 0:
            raise ValueError("reserve_bytes must not be negative")

    @property
    def resolved_manifest_path(self) -> Path:
        """Return the explicit manifest path or keep it beside the raw snapshot."""
        return self.manifest_path or self.target_dir / "download_manifest.json"


@dataclass(frozen=True)
class DownloadPlan:
    """Immutable result of the Hub dry run used by the capacity gate."""

    repo_id: str
    requested_revision: str
    resolved_revision: str
    target_dir: Path
    file_count: int
    total_bytes: int
    files_to_download: int
    bytes_to_download: int


class DatasetDownloadError(RuntimeError):
    """Base exception for downloader validation failures."""


class InsufficientDiskSpaceError(DatasetDownloadError):
    """Raised before transfer when the destination cannot fit the snapshot."""

    def __init__(self, *, required_bytes: int, available_bytes: int) -> None:
        self.required_bytes = required_bytes
        self.available_bytes = available_bytes
        super().__init__(
            "Insufficient disk space: "
            f"need {_format_bytes(required_bytes)}, "
            f"have {_format_bytes(available_bytes)} available"
        )


def _snapshot_arguments(
    request: DownloadRequest,
    *,
    revision: str,
    dry_run: bool,
) -> dict[str, Any]:
    return {
        "repo_id": request.repo_id,
        "repo_type": "dataset",
        "revision": revision,
        "local_dir": str(request.target_dir),
        "allow_patterns": list(request.include_patterns) or None,
        "ignore_patterns": list(request.exclude_patterns) or None,
        "max_workers": request.max_workers,
        "dry_run": dry_run,
    }


def build_download_plan(
    request: DownloadRequest,
    *,
    snapshot_download_fn: SnapshotDownloadFn = snapshot_download,
) -> DownloadPlan:
    """Resolve the moving revision and calculate the remaining transfer size."""
    raw_plan = snapshot_download_fn(
        **_snapshot_arguments(request, revision=request.revision, dry_run=True)
    )
    if isinstance(raw_plan, str) or not raw_plan:
        raise DatasetDownloadError("Hugging Face returned an empty dry-run plan")

    revisions = {item.commit_hash for item in raw_plan}
    if len(revisions) != 1:
        raise DatasetDownloadError("Dry-run files did not resolve to one dataset revision")

    pending = [item for item in raw_plan if item.will_download]
    return DownloadPlan(
        repo_id=request.repo_id,
        requested_revision=request.revision,
        resolved_revision=revisions.pop(),
        target_dir=request.target_dir,
        file_count=len(raw_plan),
        total_bytes=sum(item.file_size for item in raw_plan),
        files_to_download=len(pending),
        bytes_to_download=sum(item.file_size for item in pending),
    )


def _available_bytes(path: Path) -> int:
    return shutil.disk_usage(path).free


def _write_manifest(request: DownloadRequest, plan: DownloadPlan) -> None:
    manifest = {
        **asdict(plan),
        "target_dir": str(plan.target_dir.resolve()),
        "status": "complete",
        "completed_at": datetime.now(UTC).isoformat(),
    }
    path = request.resolved_manifest_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def download_dataset(
    request: DownloadRequest,
    *,
    dry_run: bool = False,
    snapshot_download_fn: SnapshotDownloadFn = snapshot_download,
    free_space_fn: FreeSpaceFn = _available_bytes,
) -> DownloadPlan:
    """Plan and optionally download a complete dataset snapshot."""
    plan = build_download_plan(request, snapshot_download_fn=snapshot_download_fn)
    if dry_run:
        return plan

    request.target_dir.mkdir(parents=True, exist_ok=True)
    available_bytes = free_space_fn(request.target_dir)
    required_bytes = plan.bytes_to_download + request.reserve_bytes
    if available_bytes < required_bytes:
        raise InsufficientDiskSpaceError(
            required_bytes=required_bytes,
            available_bytes=available_bytes,
        )

    snapshot_download_fn(
        **_snapshot_arguments(
            request,
            revision=plan.resolved_revision,
            dry_run=False,
        )
    )
    _write_manifest(request, plan)
    return plan


def _format_bytes(value: int) -> str:
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.1f} {unit}"
        size /= 1024
    raise AssertionError("unreachable")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download a revision-pinned Hugging Face dataset into data/raw.",
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET_DIR)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--include", action="append", default=[])
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--reserve-gb", type=float, default=20.0)
    parser.add_argument(
        "--download",
        action="store_true",
        help="Perform the transfer. Without this flag, only a dry run is performed.",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    snapshot_download_fn: SnapshotDownloadFn = snapshot_download,
    free_space_fn: FreeSpaceFn = _available_bytes,
) -> int:
    """Run the downloader command-line interface."""
    args = _parser().parse_args(argv)
    try:
        request = DownloadRequest(
            repo_id=args.repo_id,
            revision=args.revision,
            target_dir=args.target,
            manifest_path=args.manifest,
            include_patterns=tuple(args.include),
            exclude_patterns=tuple(args.exclude),
            max_workers=args.workers,
            reserve_bytes=int(args.reserve_gb * GIB),
        )
        plan = download_dataset(
            request,
            dry_run=not args.download,
            snapshot_download_fn=snapshot_download_fn,
            free_space_fn=free_space_fn,
        )
    except (DatasetDownloadError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    print(f"Repository: {plan.repo_id}")
    print(f"Revision:   {plan.resolved_revision}")
    print(f"Files:      {plan.file_count} total, {plan.files_to_download} pending")
    print(
        "Size:       "
        f"{_format_bytes(plan.total_bytes)} total, "
        f"{_format_bytes(plan.bytes_to_download)} pending"
    )
    print(f"Target:     {plan.target_dir.resolve()}")
    if args.download:
        print(f"Manifest:   {request.resolved_manifest_path.resolve()}")
    else:
        print("Dry run only. Re-run with --download to start the transfer.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
