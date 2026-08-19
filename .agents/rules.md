# CODING & INTERACTION RULES

## 1. Nguyên tắc cốt lõi & Tư duy
- **Không phình to dự án (YAGNI & DRY):** Chỉ cài package khi thực sự cần thiết. Ưu tiên giải pháp tối giản, không viết abstraction sớm khi chưa có use case cụ thể.
- **Giải thích trước, code sau:** Luôn giải thích ngắn gọn lý do chọn giải pháp và đánh giá rủi ro (side-effects) trước khi tạo/sửa file.
- **Tính toàn vẹn (Completeness):** Không dùng comment kiểu `// ... giữ nguyên code cũ`. Luôn cung cấp code đầy đủ hoặc dùng diff rõ ràng khi chỉnh sửa.

## 2. Kiểm soát ảo giác (Anti-Hallucination)
- **Kiểm chứng codebase trước khi code:** Luôn kiểm tra file/thư mục thực tế trước khi import hoặc giả định một hàm/thư viện đã tồn tại.
- **Không tự bịa API/Library:** Nếu không chắc về cú pháp hoặc API của thư viện bên thứ 3, yêu cầu người dùng xác nhận hoặc tra cứu tài liệu trước khi viết.
- **Thừa nhận sự thiếu sót:** Khi thiếu thông tin hoặc gặp lỗi không rõ nguyên nhân, đặt câu hỏi làm rõ thay vì tự đoán logic nghiệp vụ.

## 3. Quy trình làm việc với bộ nhớ
- **Đầu mỗi session:** Đọc `.agents/rules.md`, `.agents/long_memory.md`, `.agents/short_memory.md`, và `.agents/plan.md`.
- **Trong quá trình làm:** Cập nhật `.agents/short_memory.md` khi chuyển task hoặc phát sinh blocker.
- **Cuối mỗi session/task:** Đánh dấu task hoàn thành trong `.agents/plan.md`, tổng kết tiến độ vào `.agents/short_memory.md`, và đồng bộ các quyết định kiến trúc mới vào `.agents/long_memory.md`.
