"""Citation provenance lookup shared by the submission answering path.

The Day 20 gold-measurement harness that used to live here
(`evaluate_answer_packages_on_gold` and its report writers) measured the
deleted planner tiers and went away with them (spec 2026-08-23 §8). What
remains is the live half: resolving citation provenance for evidence cell
ids, used by `submission/exporter.py` and `submission/citation_summary.py`.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import duckdb


def build_citation_lookup(
    release_dir: Path, cell_ids: Sequence[str]
) -> dict[str, dict[str, object]]:
    """Resolve citation provenance fields for a batch of cell ids.

    Mirrors `execution/cell_frame.py::_hardened_connection` (ADR 0008 F1):
    disable extension autoinstall/autoload so this read-only lookup cannot
    reach the network either.
    """
    if not cell_ids:
        return {}
    connection = duckdb.connect(":memory:")
    connection.execute("SET autoinstall_known_extensions = false")
    connection.execute("SET autoload_known_extensions = false")
    try:
        rows = connection.execute(
            """
            SELECT c.cell_id, d.relative_path, c.source_line_start, c.source_line_end, t.title_raw
            FROM read_parquet(?) AS c
            JOIN read_parquet(?) AS t USING (table_id)
            JOIN read_parquet(?) AS d USING (doc_id)
            WHERE c.cell_id IN (SELECT UNNEST(?))
            """,
            [
                str(release_dir / "cells.parquet"),
                str(release_dir / "tables.parquet"),
                str(release_dir / "documents.parquet"),
                list(cell_ids),
            ],
        ).fetchall()
    finally:
        connection.close()
    return {
        cell_id: {
            "doc_relative_path": relative_path,
            "source_line_start": line_start,
            "source_line_end": line_end,
            "table_title": title,
        }
        for cell_id, relative_path, line_start, line_end, title in rows
    }
