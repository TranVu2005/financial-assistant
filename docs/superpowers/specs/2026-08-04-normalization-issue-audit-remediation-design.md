# Normalization Issue Audit and Remediation Design

## Goal

Xác định nguyên nhân thực tế của các normalization issue, sửa các rule tạo false positive, và chứng minh bằng mẫu kiểm định cố định rằng false-positive rate của từng issue được sửa không vượt quá 5%.

## Scope

Trong phạm vi:

- Audit `unit_unknown`, `metric_unknown`, `number_invalid`, `number_missing`, `unit_conflict`, `statement_conflict`, `number_ambiguous`, `period_incomplete`, và `period_ambiguous`.
- Lấy mẫu có thể tái tạo từ release Parquet hiện có.
- Gán nhãn thủ công cho sample bằng taxonomy cố định.
- Sửa các normalization rule có bằng chứng false positive.
- Rebuild dataset và tạo báo cáo before/after.
- Giữ nguyên 146.011 canonical tables và toàn bộ canonical table IDs.

Ngoài phạm vi:

- Dùng ML/LLM classifier trong runtime normalization.
- Loại bảng hoặc cell để giảm issue count.
- Thay đổi extraction, continuation merge, duplicate handling hoặc source-occurrence audit.
- Cố đạt một tỷ lệ giảm issue tổng tùy ý khi issue còn lại là lỗi dữ liệu thật.

## Current Baseline

Release hiện tại ghi nhận 5.890.516 issue:

| Issue | Count |
|---|---:|
| `unit_unknown` | 2.196.770 |
| `metric_unknown` | 1.227.314 |
| `number_missing` | 855.141 |
| `number_invalid` | 485.961 |
| `unit_conflict` | 1.661 |
| `number_ambiguous` | 735 |
| `period_incomplete` | 178 |
| `statement_conflict` | 18 |
| `period_ambiguous` | 17 |

Các count này là số issue record, không phải số bảng lỗi độc lập. Một table hoặc cell có thể tạo nhiều issue.

## Approach

Áp dụng sampling-first và rule-specific fixes:

1. Đo baseline và phân phối theo issue code, raw value, table, company, year và statement type.
2. Sinh sample phân tầng, deterministic, đủ context để đánh giá thủ công.
3. Gán nhãn nguyên nhân và đo false-positive rate trước khi sửa.
4. Chỉ sửa rule khi sample chỉ ra pattern false positive cụ thể.
5. Viết regression test từ case thật đã được gán nhãn.
6. Rebuild và đánh giá lại cùng tập sample.

Cách này được chọn thay cho frequency-first vì pattern xuất hiện nhiều chưa chắc là false positive, và thay cho classifier vì runtime cần deterministic và audit được.

## Audit Artifacts

### Sample dataset

Tạo `data/qa/normalization_issue_sample.parquet` với một record cho mỗi issue được chọn. Các field bắt buộc:

- `sample_id`: SHA-256 ổn định từ release fingerprint và issue identity.
- `release_fingerprint`.
- `issue_code`, `field`, `raw_value`.
- `doc_id`, `table_id`, `cell_id`.
- `company_code`, `report_year`, `statement_type`.
- `table_title_raw`, `table_unit_raw`.
- `row_label_raw`, `column_label_raw`, `value_raw`.
- `source_line_start`, `source_line_end`.
- `stratum_key` và `selection_rank`.

Sampling dùng seed cố định và hash ranking, không phụ thuộc thứ tự đọc Parquet. Mỗi issue code được lấy mẫu riêng; issue hiếm lấy toàn bộ, issue phổ biến lấy mẫu phân tầng để tránh một raw value hoặc một công ty chiếm toàn bộ sample.

### Human labels

Tạo `data/qa/normalization_issue_labels.csv` với:

- `sample_id`.
- `label`: `true_issue`, `false_positive`, hoặc `uncertain`.
- `cause_code`.
- `reviewer_note`.

Taxonomy `cause_code`:

- `year_header_as_unit`.
- `missing_unit_context`.
- `unsupported_unit_alias`.
- `non_metric_row`.
- `unsupported_metric_alias`.
- `non_value_cell`.
- `ocr_corruption`.
- `separator_ambiguity`.
- `legitimate_missing_value`.
- `mixed_unit_table`.
- `statement_signal_conflict`.
- `period_missing_year`.
- `period_format_ambiguous`.
- `other`.

Nhãn `uncertain` không được tính là true issue hoặc false positive; báo cáo phải hiển thị riêng tỷ lệ coverage của các nhãn kết luận được.

### Before/after report

Tạo JSON và Markdown report dưới `artifacts/normalization-audit/` gồm:

- Release fingerprint trước và sau.
- Issue counts trước và sau theo code.
- Sample size và label coverage.
- False-positive rate với numerator/denominator rõ ràng.
- Phân phối cause code.
- Danh sách rule thay đổi.
- Canonical table count và table-ID invariant.
- Các issue còn lại chưa xử lý.

## Normalization Changes

Các module được phép thay đổi:

- `normalization/units.py`: unit evidence và alias normalization.
- `normalization/metrics.py`: metric aliases và nhận diện non-metric rows.
- `normalization/numbers.py`: missing marker, OCR-safe parsing và ambiguity rules.
- `normalization/periods.py`: incomplete/ambiguous period rules.
- `normalization/statements.py`: conflict rules.
- `normalization/service.py`: xác định đúng value candidate và vị trí phát issue.

Mỗi thay đổi phải gắn với ít nhất một `cause_code`, một sample đã review và một regression test. Không được suppress issue chỉ dựa trên mục tiêu giảm count.

## Data Flow

```text
release parquet
  -> deterministic stratified sampler
  -> human labels
  -> baseline metrics by issue/cause
  -> targeted normalization fixes + regression tests
  -> rebuilt release
  -> replay same sample identities/context
  -> before/after report + invariants
```

Nếu một sample không còn tạo issue sau fix, evaluator vẫn giữ sample trong denominator và đánh giá kết quả mới từ cùng source cell. Sample không được thay mới để làm đẹp kết quả.

## Validation and Error Handling

- Dừng audit nếu release fingerprint không khớp giữa sample và artifact đầu vào.
- Dừng evaluation khi có `sample_id` trùng, label không thuộc enum, hoặc label thiếu sample tương ứng.
- Báo lỗi nếu sample không resolve được về document/table/cell nguồn.
- Báo riêng sample bị thay đổi context sau rebuild; không âm thầm loại khỏi denominator.
- Không publish kết quả nếu canonical table count khác 146.011 hoặc tập canonical table IDs thay đổi.

## Testing Strategy

- Unit test hash sampling độc lập với input order.
- Unit test stratification và cap cho frequent raw values/companies.
- Unit test schema và label validation.
- Regression test cho từng normalization rule sửa từ sample thật đã rút gọn.
- Integration test sample -> labels -> report trên fixture nhỏ.
- Rebuild test xác nhận table count và canonical table IDs không đổi.
- Full test suite trước khi chấp nhận release mới.

## Acceptance Criteria

- Mỗi issue code được sửa có sample review với label coverage tối thiểu 90%.
- False-positive rate sau fix không vượt quá 5% cho từng issue code được sửa; denominator chỉ gồm `true_issue` và `false_positive`.
- Báo cáo luôn hiển thị số `uncertain` và coverage, không che giấu mẫu chưa kết luận.
- Canonical `tables.parquet` vẫn có đúng 146.011 rows.
- Tập canonical table IDs trước và sau giống nhau hoàn toàn.
- Không có bảng/cell nào bị loại chỉ để giảm issue.
- Mỗi rule thay đổi có regression test và nguồn sample/cause tương ứng.
- Toàn bộ test thực thi được đều pass; Windows symlink skip được ghi nhận riêng nếu môi trường thiếu quyền.

## Delivery Sequence

1. Audit sampler và schema.
2. Human-label validation và baseline report.
3. Unit remediation.
4. Metric/value-candidate remediation.
5. Number/period/statement remediation theo bằng chứng sample.
6. Rebuild, replay sample và quality gate.
