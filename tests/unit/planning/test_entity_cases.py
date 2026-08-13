"""Unit tests for the deterministic, release-driven entity case generator."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from financial_report_qa.evaluation.week1_release import ReleaseLock
from financial_report_qa.planning.entity_cases import (
    TEMPLATE_INVENTORY,
    entity_case_set_sha256,
    generate_entity_cases,
    load_entity_cases,
    write_entity_cases,
)
from financial_report_qa.retrieval.release import ResolvedRetrievalRelease

_FINGERPRINT = "37a61be7aebde1fbcfe3aca42e6ba4ff37ae87bdd1a9ba6696506bcd188e7d1f"


def _fixture_release(tmp_path: Path) -> ResolvedRetrievalRelease:
    """Two real registry companies with enough period/statement diversity."""
    release_dir = tmp_path / "release"
    release_dir.mkdir()
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.table({"doc_id": ["doc_a", "doc_b"], "company_code": ["DBC", "ACB"]}),
        release_dir / "documents.parquet",
    )
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.table(
            {
                "period": [
                    "2022",
                    "2023",
                    "2024",
                    "2023-Q4",
                    "2022-Q1",
                    "2023-12-31",
                    "2022-12-31",
                    None,
                ]
            }
        ),
        release_dir / "cells.parquet",
    )
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.table(
            {
                "statement_type": [
                    "balance_sheet",
                    "income_statement",
                    "cash_flow_statement",
                    "notes",
                ]
            }
        ),
        release_dir / "tables.parquet",
    )
    lock = ReleaseLock(
        alias="dataset-pilot-v1",
        sampling_version="week1-pilot-v1",
        dataset_fingerprint=_FINGERPRINT,
        source_manifest_sha256="0" * 64,
        release_path="fixture/release",
        gate_result_path="fixture/gate.json",
        evaluation_inputs_sha256="1" * 64,
    )
    return ResolvedRetrievalRelease(
        lock=lock,
        dataset_fingerprint=_FINGERPRINT,
        release_dir=release_dir,
        gate_result_path=tmp_path / "gate.json",
        lock_path=tmp_path / "lock.json",
        manifest={},
        lock_sha256="2" * 64,
    )


def test_generation_is_deterministic_across_runs(tmp_path: Path) -> None:
    release = _fixture_release(tmp_path)
    first = generate_entity_cases(release)
    second = generate_entity_cases(release)
    assert entity_case_set_sha256(first) == entity_case_set_sha256(second)
    assert [case.case_id for case in first] == [case.case_id for case in second]


def test_case_count_matches_templates_times_companies(tmp_path: Path) -> None:
    release = _fixture_release(tmp_path)
    cases = generate_entity_cases(release)
    assert len(cases) == len(TEMPLATE_INVENTORY) * 2  # two companies in the fixture


def test_case_ids_are_unique(tmp_path: Path) -> None:
    release = _fixture_release(tmp_path)
    cases = generate_entity_cases(release)
    assert len({case.case_id for case in cases}) == len(cases)


def test_generated_cases_cover_every_reachable_ambiguity_code(tmp_path: Path) -> None:
    release = _fixture_release(tmp_path)
    cases = generate_entity_cases(release)
    observed = {code for case in cases for code in case.expected_ambiguity}
    assert observed == {
        "company_missing",
        "company_conflict",
        "period_relative_unresolved",
        "period_incomplete",
        "metric_unknown",
        "period_missing",
    }


def test_generation_fails_closed_on_a_release_missing_value_diversity(tmp_path: Path) -> None:
    release_dir = tmp_path / "thin_release"
    release_dir.mkdir()
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.table({"doc_id": ["doc_a"], "company_code": ["DBC"]}), release_dir / "documents.parquet"
    )
    pq.write_table(pa.table({"period": [None]}), release_dir / "cells.parquet")  # type: ignore[no-untyped-call]
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.table({"statement_type": [None]}), release_dir / "tables.parquet"
    )
    lock = ReleaseLock(
        alias="dataset-pilot-v1",
        sampling_version="week1-pilot-v1",
        dataset_fingerprint=_FINGERPRINT,
        source_manifest_sha256="0" * 64,
        release_path="fixture/release",
        gate_result_path="fixture/gate.json",
        evaluation_inputs_sha256="1" * 64,
    )
    release = ResolvedRetrievalRelease(
        lock=lock,
        dataset_fingerprint=_FINGERPRINT,
        release_dir=release_dir,
        gate_result_path=tmp_path / "gate.json",
        lock_path=tmp_path / "lock.json",
        manifest={},
        lock_sha256="2" * 64,
    )
    with pytest.raises(ValueError, match="value diversity"):
        generate_entity_cases(release)


def test_write_and_load_round_trip_preserves_case_set(tmp_path: Path) -> None:
    release = _fixture_release(tmp_path)
    cases = generate_entity_cases(release)
    path = tmp_path / "cases.jsonl"
    write_entity_cases(cases, path)
    loaded = load_entity_cases(path)
    assert loaded == cases
