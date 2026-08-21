"""Day 23 corpus-aware metric grounding fallback.

`entity_parser`/`rule_planner` are deliberately pure and corpus-free (see
`rule_planner.py`'s module docstring). ADR 0004 considered and rejected an
open-vocabulary fallback that fuzzy-matches `row_label_raw` against the
whole corpus (Option B): with no scoping, several near-matching rows makes
picking one a guess, violating the project's "never guess" invariant.

This module is a narrower mechanism than Option B ever proposed: it is
scoped to only *this question's own retrieved candidate tables* (never the
whole corpus), and it accepts a match only when it is the single unambiguous
`row_label_raw` the question text names. It is the one place `raw_text`
metric selection (ADR 0004 Option C) is grounded in a specific question's
own evidence rather than a static, hand-curated alias table -- see
docs/plans/day23-coverage-and-evidence-table.md for the corpus measurements
that motivate it (82.4% of tables have zero canonically-labeled cells).
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

import duckdb

from financial_report_qa.normalization._shared import normalized_key, sanitize_selector_text
from financial_report_qa.planning.entity_contracts import QueryEntities
from financial_report_qa.planning.rule_planner import RulePlanResult, build_plan
from financial_report_qa.retrieval.contracts import TableId

# Mirrors the company-alias >=3-word safety threshold (Day 22 coverage
# follow-up, plan.md commit note "Thêm 21 alias >=3 từ an toàn"). Measured
# (day23 plan §1, "top raw labels without canonical mapping"): the highest-
# frequency 1-2 token raw labels are generic accounting boilerplate ("Số dư",
# "Cộng", "Khác", "TỔNG CỘNG") that recur across unrelated line items --
# too short to trust as a self-sufficient metric name on their own.
_MIN_LABEL_TOKENS = 3

_MATERIALIZE = """
CREATE TABLE labels AS
SELECT DISTINCT table_id, row_label_raw, column_label_raw
FROM read_parquet(?) AS c
WHERE c.col_idx > 0
  AND c.value_numeric IS NOT NULL
  AND c.row_label_raw IS NOT NULL
"""

_QUERY = """
SELECT DISTINCT row_label_raw
FROM labels
WHERE table_id IN (SELECT UNNEST(?))
"""

_COLUMN_QUERY = """
SELECT DISTINCT row_label_raw, column_label_raw
FROM labels
WHERE table_id IN (SELECT UNNEST(?))
  AND column_label_raw IS NOT NULL
"""


def _hardened_connection() -> duckdb.DuckDBPyConnection:
    """Same hardening as `execution.cell_frame` (ADR 0008 decision F1)."""
    connection = duckdb.connect(":memory:")
    connection.execute("SET autoinstall_known_extensions = false")
    connection.execute("SET autoload_known_extensions = false")
    return connection


@lru_cache(maxsize=2)
def _label_connection(cells_path: str) -> duckdb.DuckDBPyConnection:
    """One connection per release, holding the numeric row labels in memory.

    `ground_raw_metric` runs once per question, so re-opening a connection and
    re-scanning `cells.parquet` (6.2M rows) per call meant ~500 full scans over
    a 1,012-question export. Materializing the 977,686 distinct
    `(table_id, row_label_raw)` pairs once turns each subsequent lookup into an
    in-memory filter. Releases are immutable by contract (plan.md §3), so a
    process-lifetime cache cannot go stale under normal use.
    """
    connection = _hardened_connection()
    connection.execute(_MATERIALIZE, [cells_path])
    return connection


def _candidate_raw_labels(release_dir: Path, table_ids: Sequence[str]) -> tuple[str, ...]:
    if not table_ids:
        return ()
    connection = _label_connection(str(release_dir / "cells.parquet"))
    rows = connection.execute(_QUERY, [list(table_ids)]).fetchall()
    labels = (row[0] for row in rows)
    return tuple(sorted(labels, key=lambda label: (normalized_key(label), label)))


def _stable_unique_labels(labels: Sequence[str]) -> tuple[str, ...]:
    """Return sanitized, unique labels in deterministic semantic order.

    `sanitize_selector_text`, not just `.strip()`: a real corpus header can
    concatenate two source lines with an embedded newline (`cell_frame.py`'s
    own docstring -- extraction joins a header with the row above and its
    unit row), and this menu feeds straight into a `MetricSelector`, whose
    `raw_text`/`column_text` fields forbid control characters outright. A
    live full-export run crashed on exactly this: a chosen column header
    containing a literal newline raised a `ValidationError` at
    `MetricSelector` construction.
    """
    sanitized = (sanitize_selector_text(label) for label in labels if label and label.strip())
    unique = dict.fromkeys(label for label in sanitized if label)
    return tuple(sorted(unique, key=lambda label: (normalized_key(label), label)))


def candidate_row_labels(release_dir: Path, table_ids: Sequence[str]) -> tuple[str, ...]:
    """Distinct raw row labels that genuinely carry a numeric value in these
    tables, in a stable order.

    The menu the Day 25 LLM cell-grounding tier chooses from
    (`llm_cell_grounding.choose_row_label`): because the model can only index
    into this list, it cannot invent a metric name -- the exact failure mode
    measured at 23.4% in Day 22.
    """
    return _stable_unique_labels(_candidate_raw_labels(release_dir, table_ids))


def ground_raw_metric(
    question: str, candidate_table_ids: Sequence[str], release_dir: Path
) -> str | None:
    """Return the one raw row label this question unambiguously names among
    its own candidate tables' numeric cells, or `None` if zero or more than
    one distinct label matches.

    Grouped by `normalized_key` so the same metric name recurring (in a
    different casing) across candidate tables counts as one identity, not an
    ambiguity. Never picks among multiple genuinely distinct matches.
    """
    normalized_question = normalized_key(question)

    groups: dict[str, str] = {}
    for label in _candidate_raw_labels(release_dir, candidate_table_ids):
        key = normalized_key(label)
        groups.setdefault(key, label)

    matched_keys = {
        key
        for key in groups
        if len(key.split()) >= _MIN_LABEL_TOKENS and key in normalized_question
    }
    if len(matched_keys) != 1:
        return None
    # Stripped, not just normalized: the result feeds `QueryEntities.metrics`,
    # whose validator rejects any value differing from its own `.strip()` --
    # corpus labels can carry OCR leading/trailing whitespace noise.
    return groups[next(iter(matched_keys))].strip()


def plan_with_raw_grounding_fallback(
    entities: QueryEntities,
    *,
    candidate_table_ids: tuple[TableId, ...],
    known_table_ids: frozenset[str],
    release_dir: Path,
) -> tuple[RulePlanResult, bool]:
    """Try the rule planner; if -- and only if -- it abstains with
    `entity_ambiguous` purely because `metric_unknown` was the sole
    ambiguity code, retry once with a metric grounded in this question's own
    candidate tables (`ground_raw_metric`).

    Returns `(result, grounded)`. `grounded` is True only when the retry is
    what produced the returned plan, so callers can mark `plan_source`
    distinctly (Day 23 plan Step 1) without this module reaching into
    `submission`/`pipeline` internals.
    """
    result = build_plan(
        entities, candidate_table_ids=candidate_table_ids, known_table_ids=known_table_ids
    )
    if result.plan is not None or entities.ambiguity != ("metric_unknown",):
        return result, False

    raw_metric = ground_raw_metric(entities.question, candidate_table_ids, release_dir)
    if raw_metric is None:
        return result, False

    grounded_entities = QueryEntities.model_validate(
        {
            **entities.model_dump(mode="python"),
            "metrics": (raw_metric,),
            "ambiguity": (),
        }
    )
    retried = build_plan(
        grounded_entities,
        candidate_table_ids=candidate_table_ids,
        known_table_ids=known_table_ids,
    )
    return retried, retried.plan is not None


def candidate_column_labels(
    release_dir: Path, table_ids: Sequence[str], row_label: str
) -> tuple[str, ...]:
    """Real column headers that carry a number on `row_label`, in stable order.

    The menu `llm_cell_grounding.choose_column_label` indexes into. Scoped to
    the one row so the model cannot pick a header that row does not have, and
    matched through `normalized_key` because the row label arrives straight
    from the corpus via `choose_row_label` and can differ in casing or spacing
    from the stored value.
    """
    if not table_ids:
        return ()
    connection = _label_connection(str(release_dir / "cells.parquet"))
    rows = connection.execute(_COLUMN_QUERY, [list(table_ids)]).fetchall()
    target = normalized_key(row_label)
    columns = tuple(
        column.strip()
        for stored_row, column in rows
        if normalized_key(stored_row) == target and column and column.strip()
    )
    return _stable_unique_labels(columns)
