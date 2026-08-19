# CONTINUAL-LEARNING BENCHMARK & DPST PLAN

## Milestone 1 — Audit và nền tảng
- [x] Đọc `.agents/*`, hai đặc tả benchmark/DPST và kiểm tra phạm vi dữ liệu.
- [x] Xác nhận repository chưa có Git metadata; chưa khởi tạo/push trước khi có kết quả kiểm chứng.
- [x] Khảo sát ETTh1: schema là `date` + 7 numeric columns, target smoke `OT`, hourly timestamp.
- [ ] Chốt fully pinned dependency lockfile; current exact runtime is recorded in the smoke environment manifest.

## Milestone 2 — Pipeline chung causal
- [x] Dataset interface và chronological split 20% train / 5% validation / 75% online.
- [x] Scaler fit trên train segment và manifest/hash bất biến.
- [x] Direct multi-horizon windowing với lookback `L=60`.
- [x] Pending queue, feedback release sau đủ `H` raw steps, và prequential ordering.
- [x] Metrics, logging, checkpoint/restore, failure capture và report artifact helpers.

## Milestone 3 — Shared methods
- [x] Shared causal TCN backbone + common optimizer/update interface.
- [x] OGD trên resolved sample.
- [x] Reservoir replay và ER với ngân sách chung.
- [x] DPST-Core: adaptive `eta`, adaptive `lambda`, alignment và forgetting diagnostics.
- [ ] DER++ sau khi smoke pipeline nền tảng đã ổn định.

## Milestone 4 — Tests và smoke validation
- [x] Unit/integration tests: split/scaler leakage, queue/horizon, reservoir, checkpoint, metrics.
- [x] Regression test `DPST(beta_eta=beta_lambda=0) == ER fixed` trong tolerance (controller state/update invariants).
- [x] Smoke ETTh1, `H=1`, `L=60`, một seed, 1.999 resolved samples, hai LR candidates.
- [x] Smoke report và failure report; không chạy full benchmark.

## Milestone 5 — External baselines và full benchmark (blocked)
- [ ] Kiểm tra package/repository chính thức và pin exact commit cho DER++, FSNet, OneNet, NatSR.
- [ ] Chỉ tích hợp sau smoke và fairness/leakage tests pass.
- [ ] Full benchmark/report chưa được phép chạy ở giai đoạn hiện tại.
- [ ] Khởi tạo Git, commit, verify rồi push remote `https://github.com/AnhDuybugde/continual-learning`.
