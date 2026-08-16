"""Day 23 last-resort tier: render a candidate table's real content as
compact text for the grounded LLM fallback prompt (`llm_planner.build_plan_grounded`).

Corpus-aware, not pure -- same architectural boundary as
`raw_metric_grounding.py`: `entity_parser`/`rule_planner` stay pure, this
module is the one place a question's own retrieved tables are read for
anything beyond their ids. Day 22 measured that a vocabulary-free LLM prompt
(no real row labels shown) caused 23.4% of plans to invent plausible-sounding
metric names instead of copying real ones -- ADR 0004 Option C's original
intent ("LLM phải sao chép nguyên văn từ bảng ứng viên nó thấy") was never
actually wired into the Day 17 prompt.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from financial_report_qa.data.table_frame import load_table_frame

_MAX_ROWS_PER_TABLE = 80


def render_table_context(release_dir: Path, table_ids: Sequence[str]) -> str:
    """Render each candidate table's real grid as readable text, labeled with
    its source document and title."""
    blocks: list[str] = []
    for table_id in table_ids:
        frame = load_table_frame(release_dir, table_id)
        grid = frame.grid.head(_MAX_ROWS_PER_TABLE)
        grid_text = grid.to_string(index=False, header=False) if not grid.empty else "(bảng trống)"
        header = f"--- Bảng {table_id} ({frame.relative_path})"
        if frame.title_raw:
            header += f" — {frame.title_raw}"
        header += " ---"
        blocks.append(f"{header}\n{grid_text}")
    return "\n\n".join(blocks)
