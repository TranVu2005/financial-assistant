# ADR 0003: Loại graph expansion khỏi hệ thống mặc định

- **Trạng thái:** Accepted
- **Ngày:** 2026-08-14
- **Quyết định:** Bỏ graph expansion (Day 11/12) khỏi đường truy hồi chính. Giữ code, index và
  báo cáo làm phụ lục có thể bật lại nếu bằng chứng thay đổi.

## Bối cảnh

[plan.md § Ngày 14](../../plan.md) đặt quy tắc quyết định trước khi nhìn số: giữ graph expansion
**chỉ khi** tồn tại một điểm lưới `alpha > 0` thắng control `alpha = 0` ở F2@R hoặc ở Recall@10
của phân khúc `three_or_more` (nhóm gold ≥ 3 bảng, phân khúc yếu nhất), với biên ≥ 0,01.

Quy tắc này được đặt ra vì hai lo ngại về tính đủ điều kiện của bằng chứng trước đó (Ngày 12,
Ngày 13):

1. Gold 70 câu (kể cả sau khi mở rộng ở Ngày 13) có **0 câu** mà bằng chứng bắt buộc là bảng
   `notes` được nối tới seed qua quan hệ `explained_by_note` — quan hệ này chưa từng được đánh
   giá với dữ liệu thật.
2. Quan hệ `shared_metric` tính Jaccard trên `metric_labels`, vốn trước Ngày 14 chỉ phủ **17,2 %**
   số bảng (do 94 % nhãn dòng bị vứt khỏi document — xem [ADR liên quan trong
   plan.md](../../plan.md) mục cảnh báo Ngày 14). Sau khi Task 14.2 khôi phục nhãn `raw`, quan hệ
   này có nền dữ liệu hoàn toàn khác.

## Số đo (release `422df141c935…`, sau Task 14.2/14.3, gold 70 câu)

Lưới 13 điểm cố định (`retrieval/expansion_contracts.py::PRE_REGISTERED_EXPANSION_GRID`), seed
BM25 top-50, fan-out 25:

| alpha | expand_non_seeds | relations | F2 | Recall@10 |
| ---: | :---: | ---: | ---: | ---: |
| **0.0 (control)** | — | — | **0.437709** | **0.914286** |
| 0.25 | false | 1 | 0.280339 | 0.602381 |
| 0.25 | true | 1 | 0.275237 | 0.588095 |
| 0.5 | false | 1 | 0.267300 | 0.573810 |
| 0.5 | true | 1 | 0.258230 | 0.552381 |
| 1.0 | false | 1 | 0.236843 | 0.504762 |
| 1.0 | true | 1 | 0.223804 | 0.476190 |
| 0.25 | false | 5 | 0.307066 | 0.640476 |
| 0.25 | true | 5 | 0.259534 | 0.551190 |
| 0.5 | false | 5 | 0.303098 | 0.633333 |
| 0.5 | true | 5 | 0.259534 | 0.551190 |
| 1.0 | false | 5 | 0.293635 | 0.619048 |
| 1.0 | true | 5 | 0.254040 | 0.544048 |

**Mọi điểm `alpha > 0` đều tệ hơn control ở cả F2 và Recall@10** — không phải chênh lệch nhỏ
trong biên sai số, mà là khoảng cách 0,13–0,20 tuyệt đối. Đây là kết quả dứt khoát hơn cả Ngày 12
(4/30 câu còn headroom) và Ngày 13 (chênh 0,003 giữa best và control): sau khi sửa lỗi nhãn dòng,
graph expansion không chỉ "không đóng góp" mà **chủ động làm giảm chất lượng** truy hồi ở mọi cấu
hình đã đăng ký trước.

Cơ chế: mở rộng một-hop pha trộn RRF của các bảng lân cận (cùng tài liệu, cùng kỳ liền kề, cùng
loại báo cáo, chia sẻ metric) vào kết quả BM25 vốn đã mạnh hơn nhiều sau Task 14.2 — với ngân sách
top-10 cố định, mọi bảng lân cận được chèn thêm đều đẩy một bảng BM25-xếp-hạng-cao ra ngoài top-10.

## Quyết định

Không tồn tại điểm nào thỏa quy tắc đã đăng ký trước → **bỏ graph expansion khỏi hệ thống mặc
định**, dùng BM25 v4 (`alpha = 0`, tương đương không expansion) làm baseline chính cho Tuần 3.

## Hệ quả

- `retrieval/expansion.py`, `graph.py`, `graph_service.py` và index `graph-day14-a/` **giữ
  nguyên trong repo** — không xóa. Đây là hạ tầng đã kiểm chứng đúng (byte-identical A/B build,
  0 self-loop ngoài chủ đích, coverage đo được), chỉ là chưa có bằng chứng lợi ích trên gold hiện
  tại.
- Nếu Tuần 3 bổ sung câu hỏi cần bảng `notes` liên quan (điều mà gold 70 câu vẫn thiếu — xem
  [plan.md § Ngày 13](../../plan.md)), lưới 13 điểm có thể chạy lại nguyên trạng để đánh giá lại
  quyết định này mà không cần code mới.
- `service.py`/pipeline Tuần 3 dùng thẳng `RetrievalService` (BM25) hoặc `FusionService`, không
  wrap qua `GraphExpansionService`.
- Không xóa `data/indexes/graph-day14-a/` — vẫn nhỏ (buckets.jsonl) và là bằng chứng tái lập được
  cho quyết định này.
