# ADR-0001: Chọn modular monolith

- Trạng thái: Accepted
- Ngày: 2026-08-02

## Bối cảnh

Sản phẩm cần nhiều năng lực liên tiếp: tải dữ liệu, trích bảng, chuẩn hóa, retrieval, lập kế hoạch,
tính toán và đánh giá. Nhóm phát triển nhỏ, pipeline chưa ổn định và chưa có bằng chứng rằng các
thành phần cần scale hoặc triển khai độc lập.

## Quyết định

Đóng gói toàn bộ application code trong namespace `financial_report_qa`, chia thành các module có
trách nhiệm riêng và interface có kiểu. CLI, test, data, config và tài liệu nằm ngoài package.

## Hệ quả

- Local development, test và deployment đơn giản hơn microservices.
- Refactor xuyên module vẫn thực hiện được trong một repository và một transaction thay đổi.
- Cần giữ kỷ luật dependency giữa module để tránh tạo monolith rối.
- Chỉ tách service khi có ownership riêng hoặc nhu cầu scale/deploy độc lập được đo lường.

## Phương án không chọn

- Microservices: tăng chi phí network, container, observability và vận hành quá sớm.
- Tập hợp script ở repository root: nhanh cho thử nghiệm nhưng khó đóng gói, test và quản lý API.
