"""Shared `relevant_docs`/`relevant_tables` derivation, used by both
`exporter.py` (the answered path) and `backstop_answer.py` (the tier-4
fallback path).

Extracted out of `exporter.py` (Task 6/Important 6 of the 2026-08-21 final
review): the two modules previously reimplemented the same
table_ids -> (docs, tables) mapping with subtly different logic --
`backstop_answer.py` used `str(...).rsplit("/", 1)[-1]` (diverges from
`PurePosixPath(...).name` on a backslash-containing path) and called
`build_citation_lookup` once per candidate table (many DuckDB round-trips)
instead of batching. Only the `exporter.py` copy was covered by the MRR5
rank-order invariant test, even though ~82% of submitted items come from the
backstop tier -- i.e. the untested copy determined most of the retrieval
score. Living here, both modules import the same function and both are
covered by the same test.

This module sits below both callers (`build_cell_frame`,
`build_citation_lookup`) so importing it from either `exporter.py` or
`backstop_answer.py` cannot create an import cycle.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path, PurePosixPath

from financial_report_qa.execution.cell_frame import build_cell_frame
from financial_report_qa.verification.evaluation import build_citation_lookup


def relevant_docs_and_tables(
    retrieved_table_ids: Sequence[str], release_dir: Path
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Table list reported for retrieval scoring.

    The dashboard grades 8 retrieval metrics (TABLES/DOCS x
    Precision/Recall/F2/MRR5), INDEPENDENTLY of Answer/Execution Accuracy.

    Mandatory invariant: the order of elements in the returned tuple MUST
    match the order of `retrieved_table_ids` (retrieval-rank, highest score
    first) -- the dashboard grades MRR5 (rank of the first correct result
    in the top 5), not just set membership. This function therefore must
    NOT get its table list from `build_cell_frame()` (it `ORDER BY
    table_id` -- alphabetical, not rank) and iterate in that order; it may
    only use it to look up citation info for a single already-known
    table_id.
    """
    if not retrieved_table_ids:
        return ((), ())

    ordered_table_ids = tuple(dict.fromkeys(retrieved_table_ids))
    frame = build_cell_frame(release_dir, ordered_table_ids)

    # Any one cell per table is enough to look up citation info (doc + source
    # line are both stable within a table_id) -- no need for the whole cell.
    first_cell_by_table: dict[str, str] = {}
    for record in frame.to_dict(orient="records"):
        table_id = str(record["table_id"])
        first_cell_by_table.setdefault(table_id, str(record["cell_id"]))

    lookup = build_citation_lookup(release_dir, tuple(first_cell_by_table.values()))

    docs: dict[str, None] = {}
    tables: dict[str, None] = {}
    for table_id in ordered_table_ids:  # <-- rank order, NOT frame order
        cell_id = first_cell_by_table.get(table_id)
        if cell_id is None:
            continue  # retrieved but has no numeric cell left (rare, safe to skip)
        provenance = lookup[cell_id]
        report_id = PurePosixPath(str(provenance["doc_relative_path"])).name
        if report_id.endswith(".txt"):
            report_id = report_id[: -len(".txt")]
        docs.setdefault(report_id, None)
        tables.setdefault(f"{report_id}|{provenance['source_line_start']}", None)
    return tuple(docs), tuple(tables)
