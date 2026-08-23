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
from financial_report_qa.submission.citation_summary import relevant_docs_and_tables
from financial_report_qa.submission.contracts import (
    RawQuestion,
    SubmissionEvidence,
    SubmissionItem,
)

CsvRow = Mapping[str, object]

#: Minor 2 (2026-08-21 final review round 2): this is the corpus-wide
#: fallback used when retrieval returned zero candidates at all. It must
#: only ever hand back a table `_uniquely_addressable_row` can actually use
#: -- i.e. one with >= 2 usable numeric cells, the same invariant that
#: function itself enforces (`len(table_frame) < 2` -> `None`). Without the
#: `HAVING COUNT(*) >= 2` guard, an unlucky `LIMIT 1` pick could land on a
#: singleton-cell table and fall straight into the last-resort `RuntimeError`
#: below -- a case that RuntimeError was never meant to cover, since a
#: perfectly usable table exists elsewhere in the corpus. `ORDER BY
#: c.table_id` makes the pick deterministic/reproducible across runs.
_ANY_TABLE_QUERY = """
SELECT c.table_id
FROM read_parquet(?) AS c
WHERE c.col_idx > 0
  AND c.value_numeric IS NOT NULL
  AND c.row_label_raw IS NOT NULL
  AND c.period IS NOT NULL
GROUP BY c.table_id
HAVING COUNT(*) >= 2
ORDER BY c.table_id
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


def _uniquely_addressable_row(table_frame: pd.DataFrame) -> pd.Series | None:
    """Chọn một ô mà predicate ngữ nghĩa định vị được duy nhất trong MỘT
    bảng (`table_frame` đã lọc theo đúng một `table_id`).

    Đo trên corpus: `(row_label_raw, column_label, period)` trong phạm vi một
    bảng còn nhập nhằng 4.37%. Ưu tiên ô không nhập nhằng để `pandas_query`
    replay được mà không cần tie-break vị trí.

    Trả về `None` -- thay vì raise -- khi bảng này không dùng được, để caller
    thử bảng ứng viên tiếp theo trong danh sách xếp hạng:

    - Bảng có < 2 ô numeric: CSV đóng gói sẽ chỉ có 1 dòng, và dòng đó CHÍNH
      LÀ `answer` -- tái tạo đúng hình dạng hardcode (`result =
      df["answer"].iloc[0]`) mà cả kế hoạch tồn tại để loại bỏ (Critical 1:
      đo trên corpus thật, 2.446/130.518 bảng chỉ có 1 ô numeric).
    - Bảng có >= 2 ô nhưng không ô nào có đủ `period` + `row_label_raw` để
      định vị.
    """
    if len(table_frame) < 2:
        return None
    usable = table_frame[table_frame["period"].notna() & table_frame["row_label_raw"].notna()]
    if usable.empty:
        return None
    counts = usable.groupby(["row_label_raw", "column_label", "period"], dropna=False)[
        "value"
    ].transform("nunique")
    unique = usable[counts == 1]
    return (unique if not unique.empty else usable).iloc[0]


def _preferred_addressable_row(
    table_frame: pd.DataFrame, row_idx: int, period: int | None
) -> pd.Series | None:
    """Ô tại đúng dòng mà quyết định offline đã chọn, hoặc `None`.

    Tầng này xử lý 823/1012 câu, và trước đây nó bỏ qua hoàn toàn lựa chọn của
    LLM: `_uniquely_addressable_row` lấy dòng đầu tiên định vị được trong bảng
    ứng viên đầu tiên, không nhìn chỉ tiêu, không nhìn kỳ. Đo trên đúng 823 câu
    đó: 404 câu có ô tại dòng LLM chọn **đúng kỳ được hỏi**, 304 câu nữa có ô
    tại dòng đó nhưng khác kỳ. Trả lời bằng một dòng tùy tiện là vứt bỏ toàn bộ
    thông tin ấy.

    Ưu tiên đúng kỳ; không có thì lấy kỳ gần nhất **trên cùng dòng**. Sai kỳ và
    bỏ trống đều bằng 0 điểm (Answer Accuracy tính trên tổng số câu), nên cùng
    một dòng ở kỳ khác vẫn tốt hơn hẳn một dòng không liên quan.

    Trả `None` khi bảng có < 2 ô numeric (Critical 1: CSV một dòng tái tạo đúng
    hình dạng hardcode) hoặc dòng đó không có ô nào định vị được -- caller quay
    về hành vi cũ.
    """
    if len(table_frame) < 2:
        return None
    usable = table_frame[
        (table_frame["row_idx"] == row_idx)
        & table_frame["period"].notna()
        & table_frame["row_label_raw"].notna()
        & table_frame["value"].notna()
    ]
    if usable.empty:
        return None
    if period is not None:
        exact = usable[usable["period"] == period]
        if not exact.empty:
            return exact.iloc[0]
        nearest = (usable["period"].astype(int) - period).abs().idxmin()
        return usable.loc[nearest]
    return usable.iloc[0]


def build_backstop_item(
    raw_question: RawQuestion,
    candidate_table_ids: Sequence[str],
    release_dir: Path,
    *,
    preferred_row: tuple[str, int] | None = None,
    preferred_period: int | None = None,
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
    HOẶC mọi bảng ứng viên đều không dùng được (Critical 2: bảng chỉ có 1 ô
    numeric, hoặc không ô nào định vị được), rơi về một bảng bất kỳ trong
    toàn kho (`_any_corpus_table_id`) để vẫn có answer/CSV/pandas_query,
    nhưng KHÔNG báo bảng đó là "relevant": spec §6.1 cấm emit một bảng tuỳ ý
    như thể nó liên quan. Trong nhánh này `relevant_docs`/`relevant_tables`
    luôn là tuple rỗng `()` -- hợp lệ vì hai trường này không có ràng buộc
    độ dài tối thiểu trong contracts.py.

    `RuntimeError` chỉ còn được raise khi bảng dự phòng toàn kho
    (`_any_corpus_table_id`) cũng không dùng được -- tức là cả release
    không có nổi một bảng chứa >= 2 ô numeric định vị được, một trường hợp
    gần như không xảy ra. Trước đây hàm này raise ngay khi 10 bảng ứng viên
    đầu tiên không dùng được, dù kho vẫn còn hàng chục nghìn bảng khác --
    lỗi đó làm sập TOÀN BỘ lượt export 1012 câu (Critical 2).

    Đáp án vẫn là best-effort và thường sai -- đó là đánh đổi chấp nhận được
    (Answer Accuracy tính trên tổng số câu, sai và bỏ trống đều bằng 0).
    """
    ranked_table_ids = tuple(dict.fromkeys(candidate_table_ids))

    chosen: pd.Series | None = None
    chosen_table_id: str | None = None
    frame: pd.DataFrame | None = None
    used_preferred = False
    if ranked_table_ids:
        frame = build_cell_frame(release_dir, ranked_table_ids)
        # Dòng quyết định offline đã chọn đi trước mọi thứ khác.
        if preferred_row is not None and preferred_row[0] in ranked_table_ids:
            preferred_table_id, preferred_row_idx = preferred_row
            row = _preferred_addressable_row(
                frame[frame["table_id"] == preferred_table_id],
                preferred_row_idx,
                preferred_period,
            )
            if row is not None:
                chosen = row
                chosen_table_id = preferred_table_id
                used_preferred = True
    if chosen is None and ranked_table_ids:
        assert frame is not None
        for candidate_id in ranked_table_ids:
            row = _uniquely_addressable_row(frame[frame["table_id"] == candidate_id])
            if row is not None:
                chosen = row
                chosen_table_id = candidate_id
                break

    is_no_candidate_fallback = chosen is None
    if is_no_candidate_fallback:
        fallback_table_id = _any_corpus_table_id(release_dir)
        frame = build_cell_frame(release_dir, (fallback_table_id,))
        chosen = _uniquely_addressable_row(frame[frame["table_id"] == fallback_table_id])
        if chosen is None:
            raise RuntimeError(
                "bảng dự phòng toàn kho cũng không có ô nào định vị được: "
                f"{fallback_table_id}"
            )
        chosen_table_id = fallback_table_id
        table_ids = (fallback_table_id,)
    else:
        table_ids = ranked_table_ids

    assert frame is not None and chosen_table_id is not None
    table_id = chosen_table_id

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
    if used_preferred:
        # `_uniquely_addressable_row` chỉ trả về ô có `(row_label_raw,
        # column_label, period)` duy nhất trong bảng, nên query của nó không
        # cần vị trí. Dòng do quyết định chọn không có bảo đảm đó (4.37% dòng
        # trùng nhãn), nên phải ghim `row_idx` -- nếu không `.iloc[0]` có thể
        # replay ra một ô khác ô đã đóng gói, và validator sẽ bác cả bài nộp.
        clauses.append(f"(df1.row_idx == {int(chosen['row_idx'])})")
    if chosen["column_label"] is not None and not pd.isna(chosen["column_label"]):
        clauses.append(
            f"(df1.column_label == {json.dumps(str(chosen['column_label']), ensure_ascii=False)})"
        )
    query = f'df1[{" & ".join(clauses)}]["value"].iloc[0]'

    # relevant_docs/relevant_tables cover EVERY retrieved candidate table, not
    # just the one the chosen cell came from -- retrieval is scored
    # independently at 50% weight. But when there were no usable candidate
    # tables at all, `table_ids` holds only the arbitrary corpus-wide
    # fallback table (_any_corpus_table_id), and that table is NOT a
    # retrieval result -- reporting it as relevant would violate spec §6.1.
    # Skip the lookup entirely for that path and emit empty tuples.
    #
    # Shared with `exporter.py`'s answered path (Important 6, 2026-08-21
    # final review): this used to be a separate, subtly divergent
    # reimplementation with no test coverage of its own, despite handling
    # ~82% of submitted items.
    relevant_docs, relevant_tables = (
        ((), ()) if is_no_candidate_fallback else relevant_docs_and_tables(table_ids, release_dir)
    )

    item = SubmissionItem.model_validate(
        {
            "id": raw_question.id,
            "question": raw_question.question,
            "answer": float(chosen["value"]),
            "relevant_docs": relevant_docs,
            "relevant_tables": relevant_tables,
            "evidence": (
                SubmissionEvidence(
                    variable="df1", csv_path=f"data/q{raw_question.id:06d}_df1.csv"
                ),
            ),
            "pandas_query": query,
        }
    )
    return item, rows
