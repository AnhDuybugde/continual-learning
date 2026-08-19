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
- [x] Git initialized, commit `55d875c`, and pushed to `origin/master`; working tree was clean after push.
- [ ] Cần thêm peak memory instrumentation, automatic plots/report generator và parquet engine trước final benchmark.
- [x] Added `requirements.lock.txt` for the captured smoke runtime: NumPy 2.2.6, pandas 2.3.3, PyTorch 2.6.0+cu124.
- [ ] External audit found official/source-linked candidates; exact commit pinning and adapters are pending.
- [ ] NatSR provenance is currently blocked because the paper points to an anonymous repository that has not yet been verified as a stable official source.

## 3. Ghi chú nhanh cho session tiếp theo
- Giữ dữ liệu gốc bất biến.
- Không chạy full benchmark; chỉ smoke sau khi leakage/fairness tests pass.
- Mọi method phải dùng cùng split, scaler, timestamps, lookback, horizon và evaluated samples.
- External audit is recorded in `reports/external_baseline_audit.md`; do not run full benchmark until its gate passes.
