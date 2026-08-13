# ADR 0002: Định nghĩa metric retrieval cho cổng Ngày 14

- **Trạng thái:** Accepted
- **Ngày:** 2026-08-14
- **Quyết định:** Option B

## Bối cảnh

Đánh giá Day 8--12 cố định `k = 10` và, cho mỗi câu hỏi có `TP` bảng gold
trong top-10 và `R = |gold|`, tính:

```text
Precision@10 = TP / 10
Recall@10    = TP / R
F2@10        = 5 * Precision@10 * Recall@10 / (4 * Precision@10 + Recall@10)
```

Các metric này vẫn đúng để so sánh lịch sử: chúng phạt mọi bảng ngoài gold
trong ngân sách 10 kết quả. Tuy nhiên chúng không thể dùng cổng `F2 >= 0.80`
cho tập gold hiện tại. Khi thu hồi hoàn hảo (`TP = R`), trần F2@10 của một câu
có `R` bảng gold là:

```text
F2@10_max(R) = 5R / (4R + 10)
```

Tập 30 câu hiện có 12 câu `R = 1` và 18 câu `R = 2`. Vì vậy các trần là
`0.357142857142857` cho lookup (10 câu một bảng),
`0.535714285714286` cho compare và growth (mỗi intent gồm 9 câu hai bảng,
1 câu một bảng), và macro `0.476190476190476`. Ngay cả retrieval hoàn hảo
không thể đạt `F2@10 >= 0.80`. BM25 v3 đạt F2@10 `0.431216931216931`, tương
đương 90.6% trần macro hiện tại; đây không phải bằng chứng rằng cổng 0.80 có
thể đạt được bằng đổi model.

## Các phương án đã xét

### A. Giữ F2@10 làm cổng và hạ ngưỡng

Giữ toàn bộ phép tính cũ nhưng thay `0.80` bằng một ngưỡng dưới trần của tập
gold. Cách này vẫn buộc precision dùng mẫu số 10, nên kết quả cổng phụ thuộc
mạnh vào phân bố cardinality của gold khi tập được mở rộng từ 30 lên 70 câu.

### B. Giữ metric lịch sử, thêm Precision@R/F2@R làm cổng

Với `R = |gold|`, tính:

```text
Precision@R = TP trong top-R / R
F2@R        = 5 * Precision@R * Recall@10 / (4 * Precision@R + Recall@10)
```

`Precision@R` và `F2@R` đo chất lượng trong đúng ngân sách bằng số bảng cần
thiết cho từng câu, còn `Recall@10` giữ yêu cầu bằng chứng vẫn xuất hiện trong
ngân sách 10. Báo cáo tiếp tục xuất Precision@10, Recall@10 và F2@10 để so sánh
Day 8--12.

### C. Chỉ dùng Recall@10 làm cổng

Loại bỏ thành phần precision khỏi cổng sẽ tránh trần do mẫu số 10, nhưng không
phát hiện được việc đưa bằng chứng gold xuống thấp trong top-10 hoặc trả về quá
nhiều bảng không liên quan.

## Quyết định

Chọn **Option B**.

- Cổng Ngày 14 là **F2@R >= 0.80** và **Recall@10 >= 0.90**.
- `Precision@10`, `Recall@10` và `F2@10` là metric lịch sử bắt buộc phải giữ
  trong báo cáo, không thay đổi `score_at_10` hay định dạng report Day 8--12.
- `Precision@R` và `F2@R` được bổ sung bằng model/report phiên bản mới, để các
  artifact cũ vẫn đọc lại được.

## Hệ quả

Day 13 phải xuất song song hai bộ metric và ghi rõ nhãn `@10` hoặc `@R`; không
được diễn giải F2@R như một giá trị thay thế lịch sử cho F2@10. Day 14 đánh giá
ablation BM25/dense/fusion/graph bằng cả hai bộ; chỉ F2@R và Recall@10 được dùng
để qua cổng. Nếu không đạt, ưu tiên normalization/aliases trước đổi model và
dựa trên failure cases đã phân loại nguyên nhân.
