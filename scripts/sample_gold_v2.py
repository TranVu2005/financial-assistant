"""Lấy mẫu phân tầng câu hỏi ViFinQA thật và dựng phiếu gán nhãn thủ công.

Tập gold v1 hỏng vì 87% câu hỏi do máy sinh từ chính chuỗi nhãn dòng mà BM25
index (`scripts/day21_expand_gold.py`), khiến BM25 đạt Recall@10 = 99% trên
nhóm đó — một con số của bộ sinh câu hỏi, không phải của retrieval.

Script này đi hướng ngược lại và giữ đúng một ranh giới:

  * Câu hỏi lấy **nguyên văn** từ `questions.jsonl` chính thức — không sinh,
    không viết lại, không chuẩn hoá.
  * Tài liệu ứng viên chọn bằng **metadata** (company_code + report_year +
    statement_scope) từ `documents.parquet`. Không mở BM25/dense/fusion.
  * Phiếu liệt kê **TOÀN BỘ** bảng trong các tài liệu đó, theo thứ tự dòng.
    Không xếp hạng, không cắt bớt, không lọc theo bất kỳ tín hiệu retrieval nào.

Ranh giới đó là điều làm số đo sau này có nghĩa: nếu ứng viên đưa cho người
gán nhãn là top-N của một retriever, thì recall của tập gold bị chặn trên bởi
chính retriever đó và mọi so sánh trở nên vòng tròn.

Phân tầng theo ba trục, phân bổ **tỷ lệ thuận** với tổng thể 1012 câu để mẫu
giữ nguyên hình dạng của bài thật:

  A. Từ vựng chỉ tiêu: có/không chứa 1 trong 13 metric mà gold v1 dùng.
  B. Số công ty trong câu: 1 hay nhiều.
  C. Độ dài câu: tam phân vị tính trên chính tổng thể.

Chạy:

    uv run python scripts/sample_gold_v2.py \
        --release-dir data/processed/release_v2_422df141c935 \
        --questions data/raw/ViFinQA/questions/questions.jsonl \
        --output-dir data/qa/gold-v2-review \
        --sample-size 60
"""

from __future__ import annotations

import argparse
import json
import random
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import duckdb

from financial_report_qa.planning.entity_parser import parse_query_entities

#: 13 chuỗi `CANONICAL_TO_VI` của `day21_expand_gold.py`. Dùng ở đây CHỈ để
#: phân tầng — biết một câu có nằm trong vùng từ vựng dễ của gold v1 hay không
#: — không dùng để chọn bảng.
_GOLD_V1_METRICS = (
    "doanh thu thuần",
    "tổng tài sản",
    "lợi nhuận sau thuế",
    "tiền và các khoản tương đương tiền",
    "lưu chuyển tiền thuần từ hoạt động kinh doanh",
    "tổng nợ phải trả",
    "vốn chủ sở hữu",
    "lợi nhuận sau thuế chưa phân phối",
    "giá vốn hàng bán",
    "lợi nhuận gộp",
    "vay và nợ thuê tài chính ngắn hạn",
    "vay và nợ thuê tài chính dài hạn",
    "hàng tồn kho",
)

#: Số nhãn dòng in kèm mỗi bảng trong phiếu. Đủ để nhận ra một bảng thuyết
#: minh mà không biến phiếu thành bản sao của cả tài liệu.
_ROW_LABEL_PREVIEW = 8


def _normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split()).casefold()


def _stratum(question: str, company_count: int, terciles: tuple[int, int]) -> tuple[str, str, str]:
    """Ba nhãn tầng cho một câu hỏi. Thuần, không phụ thuộc thứ tự gọi."""
    lowered = _normalize(question)
    vocabulary = "v1metric" if any(m in lowered for m in _GOLD_V1_METRICS) else "longtail"
    companies = "single" if company_count <= 1 else "multi"
    words = len(question.split())
    if words <= terciles[0]:
        length = "short"
    elif words <= terciles[1]:
        length = "mid"
    else:
        length = "long"
    return vocabulary, companies, length


def _terciles(questions: list[dict[str, Any]]) -> tuple[int, int]:
    lengths = sorted(len(row["question"].split()) for row in questions)
    return lengths[len(lengths) // 3], lengths[2 * len(lengths) // 3]


def _allocate(counts: Counter[tuple[str, str, str]], sample_size: int) -> dict[Any, int]:
    """Phân bổ tỷ lệ thuận, phần dư lớn nhất được thêm trước.

    Tỷ lệ thuận chứ không đều nhau: mục đích của mẫu là giữ nguyên hình dạng
    tổng thể, nên một tầng chiếm 40% câu hỏi phải chiếm ~40% mẫu.
    """
    total = sum(counts.values())
    exact = {key: sample_size * value / total for key, value in counts.items()}
    allocation = {key: int(value) for key, value in exact.items()}
    remaining = sample_size - sum(allocation.values())
    for key, _ in sorted(
        exact.items(), key=lambda item: (-(item[1] - int(item[1])), item[0])
    )[:remaining]:
        allocation[key] += 1
    return allocation


def _documents_for(
    connection: duckdb.DuckDBPyConnection,
    release_dir: Path,
    company_codes: tuple[str, ...],
    periods: tuple[str, ...],
    scope: str | None,
) -> list[dict[str, Any]]:
    """Tài liệu ứng viên, chọn thuần bằng metadata. Không chạm retrieval.

    Không có công ty hoặc không có kỳ thì trả rỗng: phiếu sẽ ghi rõ để người
    gán nhãn tự tra, thay vì đoán bừa một tập tài liệu.
    """
    if not company_codes or not periods:
        return []
    years = sorted({int(period[:4]) for period in periods if period[:4].isdigit()})
    if not years:
        return []
    clause = "AND statement_scope = ?" if scope else ""
    parameters: list[Any] = [
        str(release_dir / "documents.parquet"),
        list(company_codes),
        years,
    ]
    if scope:
        parameters.append(scope)
    rows = connection.execute(
        f"""
        SELECT doc_id, relative_path, company_code, report_year, statement_scope
        FROM read_parquet(?)
        WHERE company_code IN (SELECT UNNEST(?))
          AND report_year IN (SELECT UNNEST(?))
          {clause}
        ORDER BY company_code, report_year, statement_scope
        """,
        parameters,
    ).fetchall()
    return [
        {
            "doc_id": row[0],
            "relative_path": row[1],
            "company_code": row[2],
            "report_year": int(row[3]),
            "statement_scope": row[4],
        }
        for row in rows
    ]


def _tables_for(
    connection: duckdb.DuckDBPyConnection, release_dir: Path, doc_ids: list[str]
) -> dict[str, list[dict[str, Any]]]:
    """MỌI bảng của các tài liệu đã chọn, theo thứ tự dòng trong tài liệu.

    `tables.parquet` có hai cột chết trong release này -- `source_ordinal`
    bằng 0 ở cả 146.011 bảng và `csv_path` null ở cả 146.011 -- nên thứ tự
    lấy từ `line_start`, còn tên file CSV lấy từ manifest của bước export.

    Không xếp hạng và không cắt bớt — đó là điều giữ cho recall của tập gold
    không bị chặn trên bởi một retriever nào.
    """
    if not doc_ids:
        return {}
    rows = connection.execute(
        """
        SELECT t.doc_id, t.table_id, t.title_raw, t.statement_type,
               COALESCE(t.unit_normalized, t.unit_raw) AS unit,
               t.row_count, t.column_count, t.line_start, t.line_end
        FROM read_parquet(?) AS t
        WHERE t.doc_id IN (SELECT UNNEST(?))
          AND t.table_id IN (
              SELECT DISTINCT table_id FROM read_parquet(?)
              WHERE value_numeric IS NOT NULL AND col_idx > 0
          )
        ORDER BY t.doc_id, t.line_start
        """,
        [
            str(release_dir / "tables.parquet"),
            doc_ids,
            str(release_dir / "cells.parquet"),
        ],
    ).fetchall()
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[0])].append(
            {
                "table_id": row[1],
                "title": row[2],
                "statement_type": row[3],
                "unit": row[4],
                "row_count": int(row[5]),
                "column_count": int(row[6]),
                "line_start": int(row[7]),
                "line_end": int(row[8]),
            }
        )
    return grouped


def _row_labels_for(
    connection: duckdb.DuckDBPyConnection, release_dir: Path, doc_ids: list[str]
) -> dict[str, list[str]]:
    """Vài nhãn dòng đầu của mỗi bảng, để người gán nhãn nhận diện nhanh."""
    if not doc_ids:
        return {}
    rows = connection.execute(
        """
        SELECT table_id, row_label_raw
        FROM (
            SELECT c.table_id AS table_id, c.row_label_raw AS row_label_raw,
                   ROW_NUMBER() OVER (
                       PARTITION BY c.table_id ORDER BY c.row_idx
                   ) AS position
            FROM read_parquet(?) AS c
            JOIN read_parquet(?) AS t USING (table_id)
            WHERE t.doc_id IN (SELECT UNNEST(?))
              AND c.row_label_raw IS NOT NULL
              AND c.col_idx = 0
        )
        WHERE position <= ?
        ORDER BY table_id, position
        """,
        [
            str(release_dir / "cells.parquet"),
            str(release_dir / "tables.parquet"),
            doc_ids,
            _ROW_LABEL_PREVIEW,
        ],
    ).fetchall()
    labels: dict[str, list[str]] = defaultdict(list)
    for table_id, label in rows:
        text = str(label).strip()
        if text:
            labels[str(table_id)].append(text)
    return labels


def _load_csv_names(manifest_path: Path) -> dict[str, str]:
    """table_id -> tên file CSV đã export, để người gán nhãn mở ra đối chiếu.

    Vắng manifest thì trả rỗng: phiếu vẫn dùng được, chỉ mất một tiện ích.
    """
    if not manifest_path.is_file():
        return {}
    names: dict[str, str] = {}
    with manifest_path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            payload = json.loads(line)
            names[str(payload["table_id"])] = str(payload["csv_path"])
    return names


def _render_worksheet(
    record: dict[str, Any],
    documents: list[dict[str, Any]],
    tables: dict[str, list[dict[str, Any]]],
    labels: dict[str, list[str]],
    csv_names: dict[str, str],
) -> str:
    lines = [
        f"# Câu {record['id']}",
        "",
        f"> {record['question']}",
        "",
        "## Máy đã tách được",
        "",
        f"- Công ty: `{', '.join(record['company_codes']) or '(không tách được)'}`",
        f"- Kỳ: `{', '.join(record['periods']) or '(không tách được)'}`",
        f"- Phạm vi: `{record['statement_scope'] or '(không nêu)'}`",
        f"- Cụm chỉ tiêu: `{', '.join(record['metric_phrases']) or '(không có)'}`",
        f"- Đơn vị được hỏi: `{record['requested_unit'] or '(không nêu)'}`",
        "",
        "## Cách điền",
        "",
        "Tìm bảng **thực sự chứa số cần để trả lời**, chép `table_id` của nó vào",
        f"`answers.jsonl`, dòng có `\"id\": {record['id']}`. Nhiều bảng thì ghi nhiều.",
        "Không tìm thấy thì để rỗng và ghi `note`.",
        "",
        "**Cách nhanh nhất — đọc báo cáo, đừng quét bảng:** mở file synced-text ở",
        "mục dưới, Ctrl-F cụm từ trong câu hỏi (thuyết minh thường nhắc nguyên văn),",
        "đọc số dòng chỗ tìm thấy, rồi tra số dòng đó trong *Chỉ mục dòng → table_id*.",
        "Danh sách bảng đầy đủ ở cuối chỉ là phương án dự phòng.",
        "",
    ]
    if not documents:
        lines += [
            "## ⚠ Không xác định được tài liệu bằng metadata",
            "",
            "Máy không tách được công ty hoặc kỳ từ câu hỏi này, nên không có",
            "danh sách bảng. Tự tra trong `data/raw/` rồi điền `table_id`, hoặc",
            "để rỗng và ghi `note` để câu này bị loại khỏi tập gold.",
            "",
        ]
        return "\n".join(lines)

    total = sum(len(tables.get(document["doc_id"], ())) for document in documents)
    lines += [
        f"## Tài liệu cần mở ({len(documents)})",
        "",
        "Chọn bằng metadata (công ty + năm + phạm vi) — không dùng retrieval.",
        "",
    ]
    for document in documents:
        mirror = f"data/interim/synced_text/{document['relative_path']}"
        lines += [
            f"- `{mirror}`",
            f"  (gốc: `data/raw/{document['relative_path']}`)",
        ]
    lines += ["", "## Chỉ mục dòng → table_id", ""]
    for document in documents:
        entries = tables.get(document["doc_id"], [])
        if not entries:
            continue
        lines += [f"**{document['relative_path']}**", "", "```"]
        for entry in entries:
            lines.append(
                f"dòng {entry['line_start']:>6}  {entry['table_id']}  "
                f"{csv_names.get(str(entry['table_id']), '(chưa export CSV)')}"
            )
        lines += ["```", ""]

    lines += [
        f"## Danh sách bảng đầy đủ — {total} bảng (dự phòng)",
        "",
        "Mọi bảng có ít nhất một ô số của các tài liệu trên, theo thứ tự dòng.",
        "Không xếp hạng, không cắt bớt: đó là điều giữ cho recall của tập gold",
        "không bị chặn trên bởi một retriever nào.",
        "",
    ]
    for document in documents:
        entries = tables.get(document["doc_id"], [])
        lines += [
            f"### {document['relative_path']}",
            "",
            f"`{document['company_code']}` · {document['report_year']} · "
            f"{document['statement_scope']} · {len(entries)} bảng",
            "",
        ]
        for entry in entries:
            title = (entry["title"] or "").strip() or "(không có tiêu đề)"
            preview = " | ".join(labels.get(str(entry["table_id"]), [])) or "(không có nhãn dòng)"
            lines += [
                f"- **dòng {entry['line_start']}** · {title}",
                f"  - `{entry['table_id']}`",
                f"  - CSV: `{csv_names.get(str(entry['table_id']), '(chưa export)')}`",
                f"  - {entry['statement_type'] or 'không phân loại'} · "
                f"{entry['row_count']}×{entry['column_count']} · "
                f"đơn vị `{entry['unit'] or 'không rõ'}`",
                f"  - {preview}",
            ]
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=60)
    parser.add_argument(
        "--csv-manifest",
        type=Path,
        default=Path("data/interim/normalized_table_csv_export/manifest.jsonl"),
        help="Manifest của bước export CSV, để in kèm tên file CSV mỗi bảng.",
    )
    parser.add_argument("--seed", type=int, default=20260825)
    arguments = parser.parse_args()

    questions = [
        json.loads(line)
        for line in arguments.questions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    terciles = _terciles(questions)

    entities_by_id = {
        int(row["id"]): parse_query_entities(row["question"]) for row in questions
    }
    strata: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in questions:
        entities = entities_by_id[int(row["id"])]
        key = _stratum(row["question"], len(entities.company_codes), terciles)
        strata[key].append(row)

    counts: Counter[tuple[str, str, str]] = Counter(
        {key: len(value) for key, value in strata.items()}
    )
    allocation = _allocate(counts, arguments.sample_size)

    rng = random.Random(arguments.seed)
    sampled: list[dict[str, Any]] = []
    for key in sorted(strata):
        pool = sorted(strata[key], key=lambda row: int(row["id"]))
        sampled.extend(rng.sample(pool, min(allocation.get(key, 0), len(pool))))
    sampled.sort(key=lambda row: int(row["id"]))

    print(f"tổng thể {len(questions)} câu; tam phân vị độ dài = {terciles}")
    print(f"{'tầng':34s} {'tổng thể':>9s} {'%':>6s} {'mẫu':>5s}")
    for key in sorted(counts):
        share = 100 * counts[key] / len(questions)
        print(f"{'/'.join(key):34s} {counts[key]:9d} {share:5.1f}% {allocation.get(key, 0):5d}")
    print(f"{'TỔNG':34s} {len(questions):9d} {100.0:5.1f}% {len(sampled):5d}")

    csv_names = _load_csv_names(arguments.csv_manifest)
    if not csv_names:
        print(f"⚠ không đọc được {arguments.csv_manifest} — phiếu sẽ không có tên CSV")

    connection = duckdb.connect(":memory:")
    connection.execute("SET autoinstall_known_extensions = false")
    connection.execute("SET autoload_known_extensions = false")

    worksheet_dir = arguments.output_dir / "questions"
    worksheet_dir.mkdir(parents=True, exist_ok=True)
    answers: list[str] = []
    unresolved = 0

    for row in sampled:
        entities = entities_by_id[int(row["id"])]
        scope = entities.statement_scope
        documents = _documents_for(
            connection,
            arguments.release_dir,
            entities.company_codes,
            entities.periods,
            scope,
        )
        doc_ids = [document["doc_id"] for document in documents]
        tables = _tables_for(connection, arguments.release_dir, doc_ids)
        labels = _row_labels_for(connection, arguments.release_dir, doc_ids)
        if not documents:
            unresolved += 1

        record = {
            "id": int(row["id"]),
            "question": row["question"],
            "company_codes": list(entities.company_codes),
            "periods": list(entities.periods),
            "statement_scope": scope,
            "metric_phrases": list(entities.metric_phrases),
            "requested_unit": entities.requested_unit,
        }
        (worksheet_dir / f"q{int(row['id']):05d}.md").write_text(
            _render_worksheet(record, documents, tables, labels, csv_names),
            encoding="utf-8",
        )
        answers.append(
            json.dumps(
                {
                    "id": int(row["id"]),
                    "question": row["question"],
                    "gold_table_ids": [],
                    "note": "",
                },
                ensure_ascii=False,
            )
        )

    connection.close()
    (arguments.output_dir / "answers.jsonl").write_text(
        "\n".join(answers) + "\n", encoding="utf-8"
    )
    print()
    print(f"phiếu   : {worksheet_dir} ({len(sampled)} file)")
    print(f"điền vào: {arguments.output_dir / 'answers.jsonl'}")
    if unresolved:
        print(f"⚠ {unresolved} câu không xác định được tài liệu bằng metadata — xem phiếu")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
