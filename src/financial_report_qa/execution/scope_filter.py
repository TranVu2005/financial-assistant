"""Day 21 statement-scope resolution and filtering (ADR 0010 decision A1).

Separate ("riêng"/"công ty mẹ") and consolidated ("hợp nhất") reports of the
same company/period/metric carry genuinely different values (Day 21 plan
§1.4: 92.8% of two-scope groups disagree). This module resolves which scope a
plan should be filtered to -- its own `statement_scope` if stated, else
`ExecutionSettings.default_statement_scope` -- and narrows a plan's
`candidate_table_ids` to tables whose document has that scope.
"""

from __future__ import annotations

from pathlib import Path

import duckdb

from financial_report_qa.planning.entity_contracts import StatementScope


def resolve_statement_scope(
    *,
    plan_scope: StatementScope | None,
    default_scope: StatementScope | None,
) -> tuple[StatementScope | None, bool]:
    """Return `(effective_scope, was_inferred)`.

    `was_inferred` is True only when `plan_scope` was unset and
    `default_scope` filled it in -- ADR 0010 decision B1's signal that the
    candidate frame, not the question, chose the statement scope.
    """
    if plan_scope is not None:
        return plan_scope, False
    if default_scope is not None:
        return default_scope, True
    return None, False


def filter_table_ids_by_scope(
    release_dir: Path, table_ids: tuple[str, ...], scope: StatementScope
) -> tuple[str, ...]:
    """Keep only the ids among `table_ids` whose document has `scope`.

    Order-preserving. Mirrors `cell_frame._hardened_connection` (ADR 0008 F1):
    extension autoinstall/autoload disabled so this lookup cannot reach the
    network either.
    """
    connection = duckdb.connect(":memory:")
    connection.execute("SET autoinstall_known_extensions = false")
    connection.execute("SET autoload_known_extensions = false")
    try:
        rows = connection.execute(
            """
            SELECT t.table_id
            FROM read_parquet(?) AS t
            JOIN read_parquet(?) AS d USING (doc_id)
            WHERE t.table_id IN (SELECT UNNEST(?)) AND d.statement_scope = ?
            """,
            [
                str(release_dir / "tables.parquet"),
                str(release_dir / "documents.parquet"),
                list(table_ids),
                scope,
            ],
        ).fetchall()
    finally:
        connection.close()
    kept = frozenset(row[0] for row in rows)
    return tuple(table_id for table_id in table_ids if table_id in kept)
