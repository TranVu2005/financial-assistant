"""Day 23 absolute last-resort tier: guarantees a contract-valid
`SubmissionItem` for literally any question, once every reasoning tier
(rule planner, raw grounding, typed LLM planner, grounded LLM fallback) has
failed.

Why this exists at all: plan.md §2.4 rule 1 requires the submission's id set
to exactly match the official question set -- a single missing id fails the
*entire* ZIP's contract validation, not just that one question. The
official Dashboard scoring (Answer/Execution Accuracy, macro-averaged over
the full 1.012-question set, not just attempted ones) already gives a wrong
numeric answer the same 0 credit as a missing one -- there is no scoring
downside to filling every remaining gap, only a hard requirement to do so.

This tier's only job is contract validity, never correctness: it never
fabricates a row. The packaged CSV is always a real slice of
`build_cell_frame(release_dir, table_ids)` -- the full source table(s), not a
row synthesized backwards from the declared answer (2026-08-21 compliance
design BI-1/BI-2). `answer` is best-effort and frequently wrong; that is an
accepted trade-off (Answer Accuracy scores wrong and missing answers
identically at 0 -- there is no additional penalty for guessing).

For the 42/1.012 `no_candidate_tables` questions (retrieval returned no
candidates at all), an arbitrary corpus-wide table is still used to produce
a best-effort answer/CSV/pandas_query -- but it is never reported as a
retrieval result: `relevant_docs`/`relevant_tables` are emitted as empty
tuples for that path, per spec §6.1 ("khong duoc emit bang tuy y").
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import duckdb
import pandas as pd

from financial_report_qa.execution.cell_frame import build_cell_frame
from financial_report_qa.submission.contracts import (
    RawQuestion,
    SubmissionEvidence,
    SubmissionItem,
)
from financial_report_qa.verification.evaluation import build_citation_lookup

CsvRow = Mapping[str, object]

_ANY_TABLE_QUERY = """
SELECT c.table_id
FROM read_parquet(?) AS c
WHERE c.col_idx > 0
  AND c.value_numeric IS NOT NULL
  AND c.row_label_raw IS NOT NULL
  AND c.period IS NOT NULL
LIMIT 1
"""


def _any_corpus_table_id(release_dir: Path) -> str:
    """Absolute floor for the 42/1.012 `no_candidate_tables` questions:
    retrieval returned nothing, so there is no candidate list to draw from.
    Picks *a table*, never a synthesized row -- the caller still routes it
    through the normal `build_cell_frame` full-table path. Raises only for a
    genuinely empty release."""
    connection = duckdb.connect(":memory:")
    connection.execute("SET autoinstall_known_extensions = false")
    connection.execute("SET autoload_known_extensions = false")
    try:
        frame = connection.execute(
            _ANY_TABLE_QUERY, [str(release_dir / "cells.parquet")]
        ).fetchdf()
    finally:
        connection.close()
    if frame.empty:
        raise RuntimeError(f"no numeric cell exists anywhere in release: {release_dir}")
    return str(frame.iloc[0]["table_id"])


def _uniquely_addressable_row(frame: pd.DataFrame) -> pd.Series:
    """Chọn một ô mà predicate ngữ nghĩa định vị được duy nhất trong bảng.

    Đo trên corpus: `(row_label_raw, column_label, period)` trong phạm vi một
    bảng còn nhập nhằng 4.37%. Ưu tiên ô không nhập nhằng để `pandas_query`
    replay được mà không cần tie-break vị trí.
    """
    usable = frame[frame["period"].notna() & frame["row_label_raw"].notna()]
    if usable.empty:
        raise RuntimeError("bảng ứng viên không có ô nào định vị được")
    counts = usable.groupby(["row_label_raw", "column_label", "period"], dropna=False)[
        "value"
    ].transform("nunique")
    unique = usable[counts == 1]
    return (unique if not unique.empty else usable).iloc[0]


def build_backstop_item(
    raw_question: RawQuestion,
    candidate_table_ids: Sequence[str],
    release_dir: Path,
) -> tuple[SubmissionItem, tuple[CsvRow, ...]]:
    """Tầng cuối: luôn trả về `SubmissionItem` hợp lệ và HỢP QUY.

    Khác bản trước ở hai điểm quyết định:

    1. CSV là trọn bảng nguồn (`build_cell_frame`), không phải một dòng dựng
       ngược từ đáp án. Bản cũ vi phạm quy định "không được gán cứng, mã hóa
       hoặc lưu sẵn kết quả" -- 826/1012 câu đi qua đây.
    2. `relevant_tables` là MỌI bảng đã retrieve, không phải một bảng suy ra
       từ ô được chọn. Thể lệ chấm truy hồi (50% điểm, F2 macro) độc lập với
       việc trả lời đúng hay sai, nên vứt bỏ danh sách đã retrieve là mất
       điểm không cần thiết.

    Nếu retrieval không trả về bảng nào (42/1012 câu `no_candidate_tables`),
    rơi về một bảng bất kỳ trong toàn kho (`_any_corpus_table_id`) để vẫn có
    answer/CSV/pandas_query, nhưng KHÔNG báo bảng đó là "relevant": spec
    §6.1 cấm emit một bảng tuỳ ý như thể nó liên quan. Trong nhánh này
    `relevant_docs`/`relevant_tables` luôn là tuple rỗng `()` -- hợp lệ vì
    hai trường này không có ràng buộc độ dài tối thiểu trong contracts.py.

    Đáp án vẫn là best-effort và thường sai -- đó là đánh đổi chấp nhận được
    (Answer Accuracy tính trên tổng số câu, sai và bỏ trống đều bằng 0).
    """
    is_no_candidate_fallback = not candidate_table_ids
    if candidate_table_ids:
        table_ids = tuple(dict.fromkeys(candidate_table_ids))
    else:
        table_ids = (_any_corpus_table_id(release_dir),)

    frame = build_cell_frame(release_dir, table_ids)
    chosen = _uniquely_addressable_row(frame)
    table_id = str(chosen["table_id"])

    # CSV thu về đúng bảng chứa ô đã chọn: predicate ngữ nghĩa chỉ duy nhất
    # trong phạm vi một bảng (4.37% nhập nhằng), không duy nhất giữa 10 bảng.
    table_frame = frame[frame["table_id"] == table_id]
    rows: tuple[CsvRow, ...] = tuple(
        {
            "table_id": record["table_id"],
            "row_idx": record["row_idx"],
            "col_idx": record["col_idx"],
            "company_code": record["company_code"],
            "row_label_canonical": record["row_label_canonical"],
            "row_label_raw": record["row_label_raw"],
            "column_label": record["column_label"],
            "period": record["period"],
            "value": record["value"],
        }
        for record in table_frame.to_dict(orient="records")
    )

    clauses = [
        f"(df1.row_label_raw == {json.dumps(str(chosen['row_label_raw']), ensure_ascii=False)})",
        f"(df1.period == {int(chosen['period'])})",
    ]
    if chosen["column_label"] is not None and not pd.isna(chosen["column_label"]):
        clauses.append(
            f"(df1.column_label == {json.dumps(str(chosen['column_label']), ensure_ascii=False)})"
        )
    query = f'df1[{" & ".join(clauses)}]["value"].iloc[0]'

    # relevant_docs/relevant_tables cover EVERY retrieved candidate table, not
    # just the one the chosen cell came from -- retrieval is scored
    # independently at 50% weight. But when there were no candidate tables
    # at all, `table_ids` holds only the arbitrary corpus-wide fallback
    # table (_any_corpus_table_id), and that table is NOT a retrieval
    # result -- reporting it as relevant would violate spec §6.1. Skip the
    # citation lookup entirely for that path and emit empty tuples.
    docs: dict[str, None] = {}
    tables: dict[str, None] = {}
    if not is_no_candidate_fallback:
        for candidate_table_id in table_ids:
            candidate_frame = frame[frame["table_id"] == candidate_table_id]
            if candidate_frame.empty:
                continue
            cell_id = str(candidate_frame.iloc[0]["cell_id"])
            provenance = build_citation_lookup(release_dir, [cell_id])[cell_id]
            report_id = str(provenance["doc_relative_path"]).rsplit("/", 1)[-1]
            if report_id.endswith(".txt"):
                report_id = report_id[: -len(".txt")]
            docs.setdefault(report_id, None)
            tables.setdefault(f"{report_id}|{provenance['source_line_start']}", None)

    item = SubmissionItem.model_validate(
        {
            "id": raw_question.id,
            "question": raw_question.question,
            "answer": float(chosen["value"]),
            "relevant_docs": tuple(docs),
            "relevant_tables": tuple(tables),
            "evidence": (
                SubmissionEvidence(
                    variable="df1", csv_path=f"data/q{raw_question.id:06d}_df1.csv"
                ),
            ),
            "pandas_query": query,
        }
    )
    return item, rows
