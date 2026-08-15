"""Day 19 plan §3 task 19.9: architecture tests that keep ADR 0008's
boundaries enforced by CI, not just by convention.

Two invariants, both direct responses to measured regressions:

- `replay_pandas_query` must only be imported from `sandbox.py` (ADR 0008
  decision B2) -- otherwise a future caller could bypass the exception
  conversion / timing budget `sandbox.py` exists to provide.
- Every `ExecutionSettings` field must be referenced somewhere in `src/`
  (Day 19 plan §1.9 -- `timeout_seconds`/`max_rows` were dead code for two
  days running before this was caught by inspection, not by a test).
"""

from __future__ import annotations

from pathlib import Path

from financial_report_qa.core.config import ExecutionSettings

_SRC_ROOT = Path(__file__).resolve().parents[3] / "src" / "financial_report_qa"


def _iter_src_files() -> list[Path]:
    return sorted(_SRC_ROOT.rglob("*.py"))


def test_replay_pandas_query_only_imported_by_sandbox() -> None:
    offenders = []
    for path in _iter_src_files():
        if path.name in {"pandas_query.py", "sandbox.py"}:
            continue
        text = path.read_text(encoding="utf-8")
        if "replay_pandas_query" in text:
            offenders.append(path.relative_to(_SRC_ROOT))
    assert not offenders, f"replay_pandas_query referenced outside sandbox.py: {offenders}"


def test_every_execution_settings_field_is_referenced_in_src() -> None:
    config_path = (_SRC_ROOT / "core" / "config.py").resolve()
    combined = "\n".join(
        path.read_text(encoding="utf-8")
        for path in _iter_src_files()
        if path.resolve() != config_path
    )
    unreferenced = [field for field in ExecutionSettings.model_fields if field not in combined]
    assert not unreferenced, (
        f"ExecutionSettings fields never read outside config.py: {unreferenced}"
    )
