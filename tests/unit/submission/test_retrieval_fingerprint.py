"""Unit tests for the retrieval-fingerprint sidecar (final review
2026-08-24): batch-time settings are pinned next to the payload files and
`export` can refuse to run when its own settings would produce a different
numbered cell-candidate list."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from financial_report_qa.core.errors import SubmissionError
from financial_report_qa.submission.retrieval_fingerprint import (
    SIDECAR_FILENAME,
    RetrievalFingerprint,
    assert_fingerprint_matches,
    load_retrieval_fingerprint,
    write_retrieval_fingerprint,
)

_SIDECAR_VALUES: dict[str, object] = {
    "k": 10,
    "rows_per_question": 20,
    "reranker_enabled": False,
    "dense_index": None,
    "release_lock": "retrieval-release.lock.json",
    "release_lock_sha256": "a" * 64,
}


def _fingerprint(**overrides: object) -> RetrievalFingerprint:
    values = dict(_SIDECAR_VALUES)
    values.update(overrides)
    return RetrievalFingerprint.model_validate(values)


def test_write_then_match_passes(tmp_path: Path) -> None:
    sidecar = write_retrieval_fingerprint(tmp_path, _fingerprint())
    assert sidecar == tmp_path / SIDECAR_FILENAME
    assert json.loads(sidecar.read_text(encoding="utf-8")) == _SIDECAR_VALUES
    # Identical settings: the guard is silent.
    assert_fingerprint_matches(sidecar, _fingerprint())


def test_load_rejects_a_non_object_payload(tmp_path: Path) -> None:
    sidecar = tmp_path / SIDECAR_FILENAME
    sidecar.write_text("[1, 2, 3]\n", encoding="utf-8")
    with pytest.raises(SubmissionError, match="JSON object"):
        load_retrieval_fingerprint(sidecar)


@pytest.mark.parametrize(
    ("field", "other_value"),
    [
        ("k", 12),
        ("rows_per_question", 30),
        ("reranker_enabled", True),
        ("dense_index", "dense-faiss"),
        ("release_lock", "other.lock.json"),
        ("release_lock_sha256", "b" * 64),
    ],
)
def test_each_field_mismatch_fails_naming_the_field(
    tmp_path: Path, field: str, other_value: object
) -> None:
    sidecar_path = write_retrieval_fingerprint(tmp_path, _fingerprint())
    with pytest.raises(SubmissionError) as excinfo:
        assert_fingerprint_matches(sidecar_path, _fingerprint(**{field: other_value}))
    message = str(excinfo.value)
    # The error must name the differing field with BOTH values...
    assert f"{field}: payloads=" in message
    assert repr(other_value) in message
    # ...and no other field may be implicated (match on segment starts so a
    # name like "k" cannot false-positive inside "release_lock:").
    named = set(re.findall(r"(?:^|: |; )(\w+): payloads=", message))
    assert named == {field}


def test_every_field_mismatch_is_reported_in_one_error(tmp_path: Path) -> None:
    sidecar_path = write_retrieval_fingerprint(tmp_path, _fingerprint())
    with pytest.raises(SubmissionError) as excinfo:
        assert_fingerprint_matches(
            sidecar_path,
            _fingerprint(k=12, rows_per_question=30, reranker_enabled=True),
        )
    message = str(excinfo.value)
    for field in ("k", "rows_per_question", "reranker_enabled"):
        assert f"{field}:" in message


def _batch_parser_defaults(argv: list[str]) -> object:
    """Parse a `row-batches` command line through the real production parser."""
    from financial_report_qa.submission.cli import _parser

    return _parser().parse_args(argv)


_COMMON = [
    "--release-lock",
    "lock.json",
    "--bm25-index",
    "idx",
    "--questions-path",
    "q.jsonl",
]


def test_row_batches_exposes_the_same_retrieval_flags_as_export() -> None:
    """`ProgramDecision.cells` are positions in the candidate list, so the two
    commands must be *able* to be configured identically. A flag present on one
    side only is what silently shifts every index."""
    batches = _batch_parser_defaults(
        [
            "row-batches",
            *_COMMON,
            "--output-dir",
            "out",
            "--release-dir",
            "rel",
            "--dense-index",
            "data/indexes/dense-qwen3-4b/fp",
            "--table-dense-weight",
            "0.3",
            "--rerank",
        ]
    )

    assert batches.dense_index.name == "fp"  # type: ignore[attr-defined]
    assert batches.table_dense_weight == 0.3  # type: ignore[attr-defined]
    assert batches.rerank is True  # type: ignore[attr-defined]


def test_batch_and_export_defaults_agree_on_every_fingerprint_field() -> None:
    """Same flags on both sides must yield an identical fingerprint.

    `export` hardcodes `rows_per_question=DEFAULT_ROW_CANDIDATE_COUNT` while
    `row-batches` takes it from `--rows-per-question`; this pins their defaults
    together so a change to either one fails here instead of in a three-hour
    export.
    """
    from financial_report_qa.retrieval.row_fusion import DEFAULT_ROW_CANDIDATE_COUNT

    batches = _batch_parser_defaults(
        ["row-batches", *_COMMON, "--output-dir", "out", "--release-dir", "rel"]
    )
    export = _batch_parser_defaults(
        [
            "export",
            *_COMMON,
            "--execution-config",
            "configs/base.yaml",
            "--output-zip",
            "out.zip",
            "--report-dir",
            "rep",
            "--program-decisions",
            "decisions.jsonl",
        ]
    )

    assert batches.k == export.k  # type: ignore[attr-defined]
    assert batches.rows_per_question == DEFAULT_ROW_CANDIDATE_COUNT  # type: ignore[attr-defined]
    assert batches.table_dense_weight == export.table_dense_weight  # type: ignore[attr-defined]
    assert batches.rerank == export.rerank is False  # type: ignore[attr-defined]
    assert batches.dense_index == export.dense_index is None  # type: ignore[attr-defined]
