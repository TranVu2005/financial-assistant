"""Day 22 offline submission validator (plan.md §2.4 rule 9): opens the
packaged ZIP fresh -- never the exporter's in-memory objects -- so a bug in
`exporter.py` cannot mark its own bundle valid. Guards path traversal/symlink
entries before reading anything, validates `SubmissionItem` schema, checks
the id set, and replays every `pandas_query` from the packaged CSV through
the same AST whitelist interpreter `execution/compiler.py` uses (Day 19
hardening), never `eval`/`exec`.
"""

from __future__ import annotations

import io
import json
import stat
import zipfile
from collections.abc import Sequence
from decimal import Decimal
from pathlib import Path, PurePosixPath, PureWindowsPath

import pandas as pd
from pydantic import ValidationError

from financial_report_qa.execution.sandbox import replay_in_sandbox
from financial_report_qa.submission.contracts import (
    SubmissionItem,
    ValidationIssue,
    ValidationReport,
)

_DEFAULT_TOLERANCE = Decimal("0.01")
_DEFAULT_REPLAY_TIMEOUT_SECONDS = 5.0


class _ArchiveUnsafeError(ValueError):
    pass


def _safe_member_names(archive: zipfile.ZipFile) -> tuple[str, ...]:
    """Reject ZIP Slip (`../`), absolute paths, drive paths, backslashes, and
    symlink entries (unix mode bit in `external_attr`) before anything is
    read or extracted."""
    names: list[str] = []
    for info in archive.infolist():
        name = info.filename
        if not name or name.endswith("/"):
            raise _ArchiveUnsafeError(f"unexpected directory entry: {name!r}")
        if "\\" in name or PureWindowsPath(name).drive:
            raise _ArchiveUnsafeError(f"unsafe archive entry name: {name!r}")
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts:
            raise _ArchiveUnsafeError(f"unsafe archive entry name: {name!r}")
        mode = info.external_attr >> 16
        if stat.S_ISLNK(mode):
            raise _ArchiveUnsafeError(f"archive entry is a symlink: {name!r}")
        names.append(name)
    if len(set(names)) != len(names):
        raise _ArchiveUnsafeError("archive contains duplicate entry names")
    return tuple(names)


def validate_submission_zip(
    zip_path: Path,
    expected_ids: Sequence[int],
    *,
    tolerance: Decimal = _DEFAULT_TOLERANCE,
    replay_timeout_seconds: float = _DEFAULT_REPLAY_TIMEOUT_SECONDS,
) -> ValidationReport:
    issues: list[ValidationIssue] = []
    try:
        with zipfile.ZipFile(zip_path) as archive:
            try:
                names = _safe_member_names(archive)
            except _ArchiveUnsafeError as exc:
                return ValidationReport(
                    valid=False,
                    item_count=0,
                    issues=(ValidationIssue(code="unsafe_archive_entry", message=str(exc)),),
                )

            root_json_names = [name for name in names if "/" not in name and name.endswith(".json")]
            if root_json_names != ["submission.json"]:
                issues.append(
                    ValidationIssue(
                        code="root_json_missing",
                        message="archive must contain exactly one root file: submission.json",
                    )
                )
                return ValidationReport(valid=False, item_count=0, issues=tuple(issues))

            data_names = [name for name in names if name != "submission.json"]
            for name in data_names:
                if not name.startswith("data/"):
                    issues.append(
                        ValidationIssue(
                            code="entry_outside_data_dir",
                            message=f"non-root entry must live under data/: {name!r}",
                        )
                    )

            try:
                raw_items = json.loads(archive.read("submission.json"))
            except json.JSONDecodeError as exc:
                issues.append(ValidationIssue(code="root_json_invalid", message=str(exc)))
                return ValidationReport(valid=False, item_count=0, issues=tuple(issues))
            if not isinstance(raw_items, list):
                issues.append(
                    ValidationIssue(
                        code="root_json_invalid", message="submission.json must be a JSON array"
                    )
                )
                return ValidationReport(valid=False, item_count=0, issues=tuple(issues))

            items: list[SubmissionItem] = []
            for raw in raw_items:
                try:
                    items.append(SubmissionItem.model_validate(raw))
                except ValidationError as exc:
                    issues.append(ValidationIssue(code="schema_invalid", message=str(exc)))

            ids = [item.id for item in items]
            if len(set(ids)) != len(ids):
                issues.append(
                    ValidationIssue(
                        code="duplicate_id", message="submission.json has duplicate ids"
                    )
                )
            if sorted(ids) != sorted(expected_ids):
                missing = sorted(set(expected_ids) - set(ids))
                extra = sorted(set(ids) - set(expected_ids))
                issues.append(
                    ValidationIssue(
                        code="id_set_mismatch",
                        message=f"missing={missing} extra={extra}",
                    )
                )

            referenced_csvs: set[str] = set()
            for item in items:
                for evidence in item.evidence:
                    referenced_csvs.add(evidence.csv_path)
                    if evidence.csv_path not in names:
                        issues.append(
                            ValidationIssue(
                                code="csv_missing",
                                message=f"id={item.id}: {evidence.csv_path!r} not found in archive",
                            )
                        )
                        continue
                    frame = pd.read_csv(io.BytesIO(archive.read(evidence.csv_path)))
                    sandbox_result = replay_in_sandbox(
                        item.pandas_query, frame, timeout_seconds=replay_timeout_seconds
                    )
                    if sandbox_result.error_code is not None:
                        issues.append(
                            ValidationIssue(
                                code="replay_failed",
                                message=f"id={item.id}: {sandbox_result.error_message}",
                            )
                        )
                        continue
                    replayed = sandbox_result.value
                    assert replayed is not None
                    if abs(replayed - Decimal(str(item.answer))) > tolerance:
                        issues.append(
                            ValidationIssue(
                                code="answer_mismatch",
                                message=(
                                    f"id={item.id}: replayed={replayed} "
                                    f"answer={item.answer} tolerance={tolerance}"
                                ),
                            )
                        )

            orphan_csvs = sorted(set(data_names) - referenced_csvs)
            if orphan_csvs:
                issues.append(
                    ValidationIssue(code="orphan_csv", message=f"unreferenced CSVs: {orphan_csvs}")
                )
    except zipfile.BadZipFile as exc:
        return ValidationReport(
            valid=False, item_count=0, issues=(ValidationIssue(code="not_a_zip", message=str(exc)),)
        )

    return ValidationReport(valid=not issues, item_count=len(items), issues=tuple(issues))
