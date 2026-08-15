# Ngày 22 — Live retrieval + Dashboard submission export

> **Đã thực hiện xong (2026-08-15).** Kết quả chạy thật trên toàn bộ 1.012 câu hỏi:
> 32/1.012 trả lời được (3,16 %), `submission.zip` vượt validator offline (`valid=True items=32`).
> Xem [plan.md § Ngày 24](../../plan.md) và [README.md § Ngày 22](../../README.md) để biết số đo
> đầy đủ và các quyết định giản lược (A-G) đã áp dụng đúng như dưới đây.

## 0. Bối cảnh

Người dùng cung cấp bộ câu hỏi thi thật: `data/raw/ViFinQA/questions/questions.jsonl`
(1.012 câu, `id` 1..1.012 liên tục, không trùng, không có đáp án — xác nhận bằng đo trực
tiếp, không suy đoán). Corpus nguồn của nó (`data/raw/ViFinQA/financial_statements/`, 1.973
file `*_extracted.txt`) khớp gần như tuyệt đối với corpus của release đã khóa
(`data/processed/release_v2_422df141c935/documents.parquet` có 1.971 dòng — lệch 2 file,
không phải corpus khác). Vậy release + BM25 index (`data/indexes/bm25-v4/...`, cùng
fingerprint) đã sẵn sàng dùng cho bộ câu hỏi thật này mà không cần ingest lại.

Đây là công việc thay thế/đứng trước Ngày 22 (Streamlit UI)/Ngày 23 (Audit view) trong
`plan.md`, theo yêu cầu ưu tiên rõ ràng của người dùng: "đủ chạy ra file zip submission
chưa, ưu tiên cái này trước giao diện". Phạm vi trùng Task G / Ngày 24 của kế hoạch gốc,
cộng thêm một phần chưa từng được lên kế hoạch: **retrieval tại thời điểm suy luận cho câu
hỏi chưa từng thấy** — mọi harness trước đây (Day 8-21) chỉ đánh giá BM25 trên câu hỏi đã có
nhãn gold (`GoldRetrievalQuestion.filters` được gán sẵn), chưa từng chạy trên câu hỏi thô.

## 1. Đo trước khi viết code

- `RetrievalService.retrieve(query: str, *, filters: RetrievalFilters, k)` đã nhận `query`
  thô, nhưng `filters` là bắt buộc; mọi call site hiện tại đều lấy `question.filters` từ
  `GoldRetrievalQuestion` (đã gán nhãn) — không có nơi nào dựng `RetrievalFilters` từ text
  thô. Tuy nhiên `planning/entity_contracts.py::to_retrieval_filters(entities)` **đã tồn
  tại** (Day 10) và làm đúng việc này: chiếu `QueryEntities` (từ
  `parse_query_entities(question: str)`, không phụ thuộc gold) sang `RetrievalFilters`, bỏ
  field nào có ambiguity flag. Không cần code truy hồi mới — chỉ cần nối
  `parse_query_entities -> to_retrieval_filters -> RetrievalService.retrieve`.
- `execution/compiler.py::_dispatch` đã tính đúng `replay_rows` (danh sách dict
  `company_code/row_label_canonical/row_label_raw/period/value`) dùng để dựng khung dữ liệu
  sandbox replay — đây **chính là** DataFrame `pandas_query` (biến `df1`) đã thao tác, đúng
  yêu cầu contract §2.4 quy tắc 7 ("CSV trong ZIP là đúng DataFrame đầu vào mà compiler đã
  dùng"). Hiện `compile_plan` tính rồi vứt đi sau khi replay xong — không lộ ra
  `CompiledQuery`. Không dựng lại logic này ở exporter (tránh trùng lặp/lệch), mà thêm field
  `replay_rows` vào `CompiledQuery`.
- `verification/evaluation.py::build_citation_lookup` đã map `cell_id -> (doc_relative_path,
  source_line_start/end, table_title)` qua `cells.parquet JOIN tables.parquet JOIN
  documents.parquet` — đây là chuỗi provenance Day 20 đã đo 100% đầy đủ trên gold70. Tái
  dùng trực tiếp cho `relevant_docs`/`relevant_tables`.
- `AnswerPackage.question_id` bị ràng buộc `^retq_[0-9a-f]{64}$` (contract Day 8). Câu hỏi
  thật có `id` integer 1..1012, không khớp pattern này. Quyết định: sinh `question_id` tổng
  hợp `retq_<sha256("submission:<id>")>` chỉ dùng nội bộ pipeline (không xuất ra
  submission.json — schema submission dùng đúng field `id` integer của người dùng).

## 2. Quyết định thiết kế

**A. `ReplayRow` + `CompiledQuery.replay_rows`.** Thêm model đóng băng
`ReplayRow(company_code, row_label_canonical, row_label_raw, period, value)` và field
`replay_rows: tuple[ReplayRow, ...] = ()` vào `CompiledQuery`, điền từ `_dispatch`'s
`replay_rows` khi `status == "answered"`. Default rỗng giữ tương thích ngược cho mọi test
hiện có dựng `CompiledQuery` trực tiếp.

**B. `retrieval/live_query.py`.** Một hàm mỏng
`retrieve_candidate_table_ids(question: str, service: RetrievalService, *, k: int) ->
tuple[str, ...]` = `parse_query_entities` → `to_retrieval_filters` →
`service.retrieve(...).results` → table_ids theo rank. Không có logic mới ngoài việc nối ba
hàm đã kiểm chứng; test chỉ cần xác nhận thứ tự/đầu ra đúng, không re-test BM25 hay entity
parser.

**C. `relevant_tables` dùng line span cấp CELL, không cấp bảng.** Contract §2.4 quy tắc 5 mô
tả `<report_id>|<line_start>` là dòng bắt đầu "của bảng", có xử lý continuation/duplicate
occurrence qua `placements.parquet`/`source_table_occurrences.parquet`. Việc dựng đúng
occurrence-được-dùng-thực-sự đòi hỏi thêm một tầng ánh xạ chưa tồn tại trong codebase và
không có ca lỗi nào được đo cho thấy nó cần thiết ngay. Quyết định: dùng
`source_line_start`/`source_line_end` **của chính cell_id trong evidence**
(`build_citation_lookup`, đã kiểm chứng 100% đầy đủ ở Day 20) — chính xác hơn dòng-đầu-bảng
vì trỏ thẳng vào occurrence thực sự cung cấp giá trị, dedupe theo `(report_id, line_start)`
trên mọi cell trong evidence. Ghi lại làm nợ đã biết: nếu occurrence-merge sai lệch với dòng
bắt đầu bảng bị giám khảo yêu cầu chính xác tuyệt đối theo nghĩa "bảng", đây là nơi cần quay
lại, không phải nợ ẩn.

**D. Câu hỏi không trả lời được không vào `submission.json`.** Theo contract §2.4: chỉ
`answered` đã verify mới thành `SubmissionItem`; `abstained`/`error` chặn phát hành. Với
1.012 câu thật — không có gold để đo trước — chấp nhận là các câu này sẽ **thiếu** khỏi
`submission.json` cho tới khi coverage được cải thiện (nợ kỹ thuật của toàn hệ thống, không
phải bug của module này). Exporter phải xuất một **báo cáo coverage riêng** (ngoài ZIP, JSON
+ Markdown) liệt kê từng câu bị bỏ qua và giai đoạn/lý do (retrieval rỗng, planning abstain,
execution error, verification rejected) — dùng lại đúng phân loại 5-stage của
`pipeline/contracts.py::PipelineStage` cho nhất quán, nhưng **không** dùng lại
`run_e2e_pipeline` (nó đòi `GoldRetrievalQuestion` với `gold_table_ids` bắt buộc non-empty —
không có cho câu hỏi thật).

**E. `k=10` mặc định cho truy hồi trực tiếp.** Khớp `k` đã dùng xuyên suốt Day 8-21 (Recall@10
là ngưỡng cổng Day 14). Có thể override qua `--k` CLI.

**F. Đóng gói ZIP xác định (deterministic).** Entry sort theo tên (root JSON trước, rồi
`data/*.csv` theo thứ tự bảng chữ cái), `zipfile.ZipInfo.date_time` cố định (không dùng
thời gian hệ thống — nếu không, cùng input sẽ sinh SHA-256 khác nhau giữa hai lần chạy, vi
phạm contract §2.4 quy tắc 8), `compress_type=ZIP_DEFLATED` cố định.

**G. Validator độc lập, không tin exporter.** `submission/validator.py` mở lại ZIP đã đóng
gói (không dùng lại object Python trong bộ nhớ của exporter), giải nén vào thư mục tạm với
guard chống ZIP Slip/symlink/absolute path, kiểm schema 7-field (`extra="forbid"`), kiểm tập
`id` khớp *chính xác* file câu hỏi gốc truyền vào (không thiếu/thừa/trùng — nhưng vì D ở
trên, submission thật sự sẽ **luôn thiếu id** so với 1.012 câu gốc cho tới khi coverage đạt
100%; validator vì vậy nhận `expected_ids` là tập con "đã trả lời", không phải toàn bộ file
câu hỏi — ghi rõ trong CLI help để không tự huyễn hoặc "đã đủ 1012/1012"), replay
`pandas_query` từ CSV đã đóng gói qua `execution.pandas_query.replay_pandas_query` (đúng
whitelist AST đã kiểm chứng Day 19) và so khớp `answer` theo dung sai cấu hình.

## 3. Việc không làm ở đây

- Không xây `data/official/test_questions.json` giả — dùng thẳng
  `data/raw/ViFinQA/questions/questions.jsonl` làm input thật, tên biến CLI generic
  (`--questions-path`) để không khóa cứng đường dẫn.
- Không tối ưu hiệu năng chạy 1.012 câu (Ngày 27). Chấp nhận runtime dài ở lần chạy thật đầu
  tiên; nếu quá chậm để thực dụng, ghi lại làm nợ thay vì âm thầm cắt bớt câu hỏi.
- Không cố đạt coverage cao — báo cáo coverage thật, dù thấp, là kết quả hợp lệ của bước
  này. Cổng chất lượng (≥ 0,85 accuracy, ...) là việc của các ngày sau khi có nhãn.
