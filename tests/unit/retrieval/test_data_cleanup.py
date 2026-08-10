"""Fail-closed Day 9 data-cleanup policy tests."""

from __future__ import annotations

import stat
from pathlib import Path

import pytest

from financial_report_qa.retrieval.cli import main as retrieval_main
from financial_report_qa.retrieval.data_cleanup import (
    CleanupEntry,
    plan_day9_cleanup,
    quarantine_day9_cleanup,
)


def _write(path: Path, content: str = "fixture") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _entry_by_name(repo_root: Path, name: str) -> CleanupEntry:
    return next(entry for entry in plan_day9_cleanup(repo_root).entries if entry.path.name == name)


def test_plan_lists_rebuildable_candidates_but_never_protected_paths(tmp_path: Path) -> None:
    """Removing a policy candidate must not make protected data eligible."""
    _write(tmp_path / "data/interim/week1_gate_attempts/attempt.json")
    _write(tmp_path / "data/processed/release_v2_7868718f2547/manifest.json", "{}")
    _write(tmp_path / "data/raw/source.txt")
    _write(tmp_path / "data/manifests/inventory.json")
    _write(tmp_path / "data/qa/retrieval-gold-v1.jsonl")
    _write(tmp_path / "data/processed/release_v2_37a61be7aebd/manifest.json", "{}")

    plan = plan_day9_cleanup(tmp_path)
    entries = {entry.path.relative_to(tmp_path).as_posix(): entry for entry in plan.entries}

    assert entries["data/interim/week1_gate_attempts"].status == "approved"
    assert entries["data/processed/release_v2_7868718f2547"].status == "approved"
    assert "data/raw" not in entries
    assert "data/manifests" not in entries
    assert "data/qa" not in entries
    assert "data/processed/release_v2_37a61be7aebd" not in entries


def test_plan_blocks_candidate_referenced_by_day9_report(tmp_path: Path) -> None:
    """Dropping a report/reference scan must make an in-use release unsafe."""
    _write(tmp_path / "data/processed/release_v2_7fc5d5d57bf6/manifest.json", "{}")
    report = _write(
        tmp_path / "plan.md",
        "Uses release_v2_7fc5d5d57bf6 for the locked comparison.",
    )

    entry = _entry_by_name(tmp_path, "release_v2_7fc5d5d57bf6")

    assert entry.status == "blocked"
    assert report.relative_to(tmp_path).as_posix() in entry.detail


def test_plan_blocks_candidate_referenced_by_nested_source_artifact(tmp_path: Path) -> None:
    """Skipping nested text artifacts could approve a release still named by source evidence."""
    _write(tmp_path / "data/processed/release_v2_37a61be7aeba/manifest.json", "{}")
    reference = _write(
        tmp_path / "src/locked_day9_plan.txt",
        "release_v2_37a61be7aeba remains required by the locked replay.",
    )

    entry = _entry_by_name(tmp_path, "release_v2_37a61be7aeba")

    assert entry.status == "blocked"
    assert reference.relative_to(tmp_path).as_posix() in entry.detail


@pytest.mark.parametrize(
    ("self_declaration", "candidate_name"),
    [
        (
            "src/financial_report_qa/retrieval/data_cleanup.py",
            "release_v2_37a61be7aeba",
        ),
        ("tests/unit/retrieval/test_data_cleanup.py", "release_v2_7868718f2547"),
    ],
)
def test_plan_ignores_cleanup_policy_self_declarations(
    tmp_path: Path,
    self_declaration: str,
    candidate_name: str,
) -> None:
    """Scanning cleanup policy source/tests as user references blocks every real candidate."""
    _write(tmp_path / f"data/processed/{candidate_name}/manifest.json", "{}")
    _write(
        tmp_path / self_declaration,
        f"_CANDIDATE_PATHS = (Path('data/processed/{candidate_name}'),)",
    )

    entry = _entry_by_name(tmp_path, candidate_name)

    assert entry.status == "approved"


def test_plan_blocks_when_nested_reference_artifact_is_not_utf8(tmp_path: Path) -> None:
    """Ignoring a decode error could approve a candidate named by unreadable evidence."""
    _write(tmp_path / "data/processed/release_v2_7868718f2547/manifest.json", "{}")
    artifact = tmp_path / "scripts/locked_reference.md"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"release_v2_7868718f2547" + (b"x" * 4096) + b"\xff")

    entry = _entry_by_name(tmp_path, "release_v2_7868718f2547")

    assert entry.status == "blocked"
    assert "cannot read reference artifact" in entry.detail


def test_plan_blocks_when_reference_artifact_permissions_prevent_reading(tmp_path: Path) -> None:
    """Ignoring ACL read failures could approve a candidate still named by locked evidence."""
    _write(tmp_path / "data/processed/v2_remediated/manifest.json", "{}")
    artifact = _write(
        tmp_path / "scripts/locked_reference.md",
        "v2_remediated remains required by the locked replay.",
    )
    artifact.chmod(0)
    try:
        try:
            artifact.read_text(encoding="utf-8")
        except OSError:
            pass
        else:
            pytest.skip("OS cannot enforce unreadable-file ACL for this test process")

        entry = _entry_by_name(tmp_path, "v2_remediated")
    finally:
        artifact.chmod(stat.S_IRUSR | stat.S_IWUSR)

    assert entry.status == "blocked"
    assert "cannot read reference artifact" in entry.detail


def test_plan_rejects_symlink_candidate_to_protected_raw_data(tmp_path: Path) -> None:
    """Following a candidate symlink could quarantine raw source provenance."""
    raw = _write(tmp_path / "data/raw/source.txt")
    candidate = tmp_path / "data/interim/week1_gate_attempts"
    candidate.parent.mkdir(parents=True)
    try:
        candidate.symlink_to(raw.parent, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks unavailable in this environment: {exc}")

    plan = plan_day9_cleanup(tmp_path)
    entry = _entry_by_name(tmp_path, "week1_gate_attempts")
    moved = quarantine_day9_cleanup(plan, tmp_path / "data/quarantine/day9-cleanup")

    assert entry.status == "blocked"
    assert moved == []
    assert raw.exists()


def test_broken_symlink_candidate_is_blocked_and_apply_returns_two(tmp_path: Path) -> None:
    """Checking existence before symlink identity would mislabel a broken link as missing."""
    candidate = tmp_path / "data/interim/week1_gate_replay"
    candidate.parent.mkdir(parents=True)
    try:
        candidate.symlink_to(tmp_path / "missing-target", target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks unavailable in this environment: {exc}")

    entry = _entry_by_name(tmp_path, "week1_gate_replay")
    exit_code = retrieval_main(
        [
            "cleanup-day9-data",
            "--repo-root",
            str(tmp_path),
            "--quarantine-root",
            str(tmp_path / "data/quarantine/day9-cleanup"),
            "--apply",
        ]
    )

    assert entry.status == "blocked"
    assert exit_code == 2
    assert candidate.is_symlink()


def test_plan_blocks_unreadable_candidate_without_quarantining_it(tmp_path: Path) -> None:
    """Treating an unreadable manifest as approved would move uncertain data."""
    candidate = tmp_path / "data/interim/week1_gate_replay"
    candidate.mkdir(parents=True)
    (candidate / "manifest.json").write_bytes(b"\xff\xfe")

    plan = plan_day9_cleanup(tmp_path)
    entry = _entry_by_name(tmp_path, "week1_gate_replay")
    moved = quarantine_day9_cleanup(plan, tmp_path / "data/quarantine/day9-cleanup")

    assert entry.status == "blocked"
    assert "unreadable" in entry.detail
    assert moved == []
    assert candidate.exists()


def test_quarantine_moves_only_approved_candidates_under_timestamped_data_root(
    tmp_path: Path,
) -> None:
    """A missing approval check would move protected or blocked source data."""
    candidate = _write(tmp_path / "data/interim/week1_gate_attempts/attempt.json")
    _write(tmp_path / "data/raw/source.txt")

    plan = plan_day9_cleanup(tmp_path)
    moved = quarantine_day9_cleanup(plan, tmp_path / "data/quarantine/day9-cleanup")

    assert len(moved) == 1
    assert moved[0].is_relative_to(tmp_path / "data/quarantine/day9-cleanup")
    assert moved[0].parts[-2:] == ("interim", "week1_gate_attempts")
    assert not candidate.parent.exists()
    assert (tmp_path / "data/raw/source.txt").exists()
