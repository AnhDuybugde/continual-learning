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
- [x] `src/continual_forecasting/external_adapters.py` now runs pinned official FSNet and OneNet-TCN forward/update paths through the shared queue for 1,999 samples. OneNet reproduces official branch update, decision MLP bias update, and long-term sigmoid weight update; `lambda`/gate moves from 0.50025 onward. NatSR remains `unavailable` because its source is anonymous/unverified.

## 3. Ghi chú nhanh cho session tiếp theo
- Giữ dữ liệu gốc bất biến.
- Không chạy full benchmark; chỉ smoke sau khi leakage/fairness tests pass.
- Mọi method phải dùng cùng split, scaler, timestamps, lookback, horizon và evaluated samples.
- External audit is recorded in `reports/external_baseline_audit.md`; do not run full benchmark until its gate passes.
- `curren_report.md` is now the replace-on-each-run current status report.
- Official sources are kept only in ignored `.external/`; no external source or raw data is committed.
- Milestone 6 artifacts: `artifacts/dpst_milestone/milestone_results.json` and `artifacts/dpst_milestone/H1/DERPP/results.json`.
- H=1 three-seed means: ER `0.089916/0.014234`, DPST-eta `0.091432/0.014659`, DPST-lambda `0.089082/0.013976`, DPST-full `0.089573/0.014052`, DER++ `0.089703/0.014161` (MAE/MSE).
- H=24 seed-0 smoke: ER `0.313632/0.157221`, DPST `0.306701/0.152023`, DER++ `0.293207/0.140646`, 176 resolved samples, release IDs `4355..4530`.
- Verification complete: deterministic pipeline 9/9, external adapters 3/3,
  and long smoke integration 1/1 with extended timeout; the default 120s
  aggregate command was only too short.
