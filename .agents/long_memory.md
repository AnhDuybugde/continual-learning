# LONG-TERM PROJECT MEMORY

## 1. Tổng quan dự án
- **Tên dự án:** CONTINUAL-LEARNING
- **Mục tiêu cốt lõi:** Xây dựng benchmark công bằng, tái lập cho online continual time-series forecasting có delayed feedback và thử nghiệm DPST-Core.
- **Đối tượng sử dụng:** Nhà nghiên cứu và kỹ sư ML cần kiểm tra causal timing, fairness và continual adaptation.

## 2. Tech Stack & Công cụ
- **Ngôn ngữ:** Python.
- **ML/data:** PyTorch, NumPy, pandas; chỉ bổ sung dependency khi cần và phải ghi version.
- **Mô hình chung:** Shared causal TCN + AdamW cho OGD, ER và DPST.
- **Dữ liệu:** ETTh1 trước; các dataset khác chỉ sau smoke/fairness gate.

## 3. Quyết định kiến trúc quan trọng (ADR)
- *2026-08-19* - **Causal prequential protocol:** Forecast phải được ghi trước khi resolved sample được dùng để update; horizon `H` chỉ release khi đủ toàn bộ `H` target raw steps.
- *2026-08-19* - **Split bất biến:** 20% chronological train, 5% validation, 75% online; scaler chỉ fit trên train.
- *2026-08-19* - **DPST-Core giới hạn:** Chỉ adaptive learning rate `eta` và replay weight `lambda`; không thêm controller neural, drift detector, Fisher/NatSR hay relevance replay.

## 4. Ràng buộc bất biến
- Không sửa/xóa dữ liệu gốc.
- Không chọn hyperparameter từ online segment.
- Tất cả randomness phải có seed; không che giấu failure/NaN.
- Không chạy full benchmark trước khi smoke và leakage/fairness tests pass.
- Mọi run phải lưu config, environment/commit, data hash, metrics, predictions/targets, timing/memory, logs, checkpoint và failure info nếu lỗi.
