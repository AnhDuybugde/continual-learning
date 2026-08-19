# SHORT-TERM SESSION MEMORY

## 1. Trạng thái phiên hiện tại
- **Ngày cập nhật:** 2026-08-19
- **Task đang làm dở:** Audit repository và xây dựng benchmark causal cho online continual time-series forecasting cùng DPST-Core.
- **Đã hoàn thành:** Đọc `.agents/*`, `baseline_benchmark_todo.md`, `dpst_method_todo.md`; cập nhật execution plan; pipeline OGD/ER/DPST và ETTh1 smoke đã pass.
- **File đang chỉnh sửa / liên quan:**
  - `.agents/plan.md`
  - `.agents/short_memory.md`
  - `src/prompt/baseline_benchmark_todo.md`
  - `src/prompt/dpst_method_todo.md`

## 2. Blockers & Vấn đề tồn đọng
- [ ] Repository chưa có `.git`; khởi tạo remote sau khi kiểm tra staged scope.
- [ ] Cần thêm peak memory instrumentation, automatic plots/report generator và parquet engine trước final benchmark.
- [ ] External baselines chưa được kiểm tra package/official commit; chưa tích hợp.

## 3. Ghi chú nhanh cho session tiếp theo
- Giữ dữ liệu gốc bất biến.
- Không chạy full benchmark; chỉ smoke sau khi leakage/fairness tests pass.
- Mọi method phải dùng cùng split, scaler, timestamps, lookback, horizon và evaluated samples.
