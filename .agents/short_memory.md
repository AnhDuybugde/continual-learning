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
- [x] Added prediction/update timing, CPU RSS/CUDA memory instrumentation, and artifact-only trajectory plot generator.
- [ ] No parquet engine is installed; CSV/NPZ/JSONL remain source of truth.
- [x] Added `requirements.lock.txt` for the captured smoke runtime: NumPy 2.2.6, pandas 2.3.3, PyTorch 2.6.0+cu124.
- [x] Pinned official/current HEADs: DER++/Mammoth `e75a491c69fd729edeb01431afb753d9157d9a81`, FSNet `c776afc623fa6384a6a559121aacadd2bbea5968`, OneNet `65eed9d6c878133a4d81d9c381c69e742ad47fd0`.
- [~] Causal adapter contract added in `src/continual_forecasting/external_adapters.py`; contract tests pass. FSNet official entry imports; OneNet probe is blocked by missing `wandb` and legacy protocol coupling. NatSR remains `unavailable` because its source is anonymous/unverified.

## 3. Ghi chú nhanh cho session tiếp theo
- Giữ dữ liệu gốc bất biến.
- Không chạy full benchmark; chỉ smoke sau khi leakage/fairness tests pass.
- Mọi method phải dùng cùng split, scaler, timestamps, lookback, horizon và evaluated samples.
- External audit is recorded in `reports/external_baseline_audit.md`; do not run full benchmark until its gate passes.
- `curren_report.md` is now the replace-on-each-run current status report.
- Official sources are kept only in ignored `.external/`; no external source or raw data is committed.
