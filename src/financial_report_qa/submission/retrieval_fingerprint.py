"""Retrieval-fingerprint sidecar: pin offline payload batches to the
retrieval settings that produced them (final review 2026-08-24, Important).

`ProgramDecision.cells` are positions in the numbered cell-candidate list,
and that list depends on how retrieval was configured when the payloads were
generated: the table-tier candidate width (`k`), whether a reranker or a
dense branch reordered candidates, the row-fusion width
(`rows_per_question`), and which release lock the parquet came from.

`submission row-batches` writes `retrieval-fingerprint.json` next to its
batch files; `submission export --assert-payload-fingerprint <path>` loads it
and refuses to answer a single question if its own settings would produce a
different candidate order -- regenerating payloads with different flags and
then reusing an old decisions file would otherwise silently shift every
`ProgramDecision.cells` index instead of failing loudly here.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from financial_report_qa.core.errors import SubmissionError
from financial_report_qa.retrieval.contracts import _FrozenModel

SIDECAR_FILENAME = "retrieval-fingerprint.json"

_FIELDS = (
    "k",
    "rows_per_question",
    "reranker_enabled",
    "dense_index",
    "release_lock",
    "release_lock_sha256",
)


class RetrievalFingerprint(_FrozenModel):
    """The retrieval-relevant settings behind one payload generation.

    `dense_index` is the basename of export's `--dense-index` directory (or
    None: `row-batches`, which has no such flag, always records None).
    `reranker_enabled` is False at batch time for the same reason -- recorded
    anyway so a future batch-time reranker cannot drift silently either.
    """

    k: int
    rows_per_question: int
    reranker_enabled: bool
    dense_index: str | None
    release_lock: str
    release_lock_sha256: str

    def differences(self, other: RetrievalFingerprint) -> dict[str, tuple[object, object]]:
        """Field name -> (self value, other value) for every differing field."""
        differences: dict[str, tuple[object, object]] = {}
        for name in _FIELDS:
            mine = getattr(self, name)
            theirs = getattr(other, name)
            if mine != theirs:
                differences[name] = (mine, theirs)
        return differences


def write_retrieval_fingerprint(output_dir: Path, fingerprint: RetrievalFingerprint) -> Path:
    """Write the sidecar next to the batch files and return its path."""
    path = output_dir / SIDECAR_FILENAME
    path.write_text(
        json.dumps(
            fingerprint.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def load_retrieval_fingerprint(sidecar_path: Path) -> RetrievalFingerprint:
    payload = _load_payload(sidecar_path)
    if not isinstance(payload, dict):
        raise SubmissionError(
            f"payload fingerprint sidecar {sidecar_path} must be a JSON object, "
            f"got {type(payload).__name__}"
        )
    try:
        return RetrievalFingerprint.model_validate(payload)
    except ValidationError as exc:
        raise SubmissionError(
            f"payload fingerprint sidecar {sidecar_path} is not a valid "
            f"{RetrievalFingerprint.__name__}: {exc}"
        ) from exc


def _load_payload(sidecar_path: Path) -> object:
    try:
        return json.loads(sidecar_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SubmissionError(
            f"cannot read payload fingerprint sidecar {sidecar_path}: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise SubmissionError(
            f"payload fingerprint sidecar {sidecar_path} is not valid JSON: {exc}"
        ) from exc


def assert_fingerprint_matches(sidecar_path: Path, current: RetrievalFingerprint) -> None:
    """Load the batch-time sidecar and fail before any question executes when
    the current run's settings differ in any field. The error names every
    differing field with both values."""
    sidecar = load_retrieval_fingerprint(sidecar_path)
    differences = sidecar.differences(current)
    if differences:
        detail = "; ".join(
            f"{name}: payloads={expected!r} vs export={actual!r}"
            for name, (expected, actual) in differences.items()
        )
        raise SubmissionError(
            f"--assert-payload-fingerprint mismatch against {sidecar_path}: "
            f"{detail}. The payloads were generated under different retrieval "
            "settings; reusing the decisions file would shift every "
            "ProgramDecision.cells index. Regenerate the batches or align the "
            "export flags."
        )
