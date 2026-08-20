# CONTINUAL-LEARNING BENCHMARK & DPST PLAN

## Milestone 1 — Audit và nền tảng
- [x] Đọc `.agents/*`, hai đặc tả benchmark/DPST và kiểm tra phạm vi dữ liệu.
- [x] Xác nhận repository chưa có Git metadata; chưa khởi tạo/push trước khi có kết quả kiểm chứng.
- [x] Khảo sát ETTh1: schema là `date` + 7 numeric columns, target smoke `OT`, hourly timestamp.
- [x] Chốt dependency lockfile cho runtime smoke; external repositories vẫn cần environment isolation.

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
- [x] DER++ regression objective, causal adapter, replay prediction storage và ETTh1 smoke.

## Milestone 4 — Tests và smoke validation
- [x] Unit/integration tests: split/scaler leakage, queue/horizon, reservoir, checkpoint, metrics.
- [x] Regression test `DPST(beta_eta=beta_lambda=0) == ER fixed` trajectory trong `rtol=1e-6`, `atol=1e-8`.
- [x] Smoke ETTh1, `H=1`, `L=60`, một seed, 1.999 resolved samples, hai LR candidates.
- [x] Smoke report và failure report; không chạy full benchmark.

## Milestone 5 — External baselines và full benchmark (blocked)
- [x] Audit nguồn chính thức/nguồn paper-linked cho DER++, FSNet, OneNet, NatSR; ghi blocker provenance của NatSR.
- [x] Pin exact commit cho DER++, FSNet, OneNet; DER++ adapter đã smoke PASS.
- [x] Official FSNet/OneNet causal wrappers chạy được 1,999-sample smoke; OneNet đã tái hiện adaptive decision bias và long-term sigmoid weight update theo official `onenet_tcn` loop.
- [x] Cài và khóa `wandb==0.28.2`; isolated import probe của cả FSNet và OneNet PASS.
- [x] Hoàn thiện OneNet adaptive decision/weight update, rerun 1,999-sample smoke và full tests; không chạy full benchmark.
- [ ] Full benchmark/report chưa được phép chạy ở giai đoạn hiện tại.
- [ ] Khởi tạo Git, commit, verify rồi push remote `https://github.com/AnhDuybugde/continual-learning`.

## Milestone 6 — DPST pilot confirmation (engineering only)
- [x] Deterministic DPST trajectory-equivalence gate rechecked on CPU.
- [x] Add reusable DPST beta controls and three-seed H=1 ablation runner.
- [x] Run ER, DPST-eta, DPST-lambda, DPST-full, and DER++ on ETTh1 H=1.
- [x] Run H=24 delayed-feedback smoke and verify release/update counts.
- [x] Update current, benchmark, DPST, and failure reports with artifact paths.
- [ ] Promote to multi-dataset/full benchmark; remains blocked until pilot evidence and fairness review are accepted.

## Milestone 7 — Fixed replay-weight audit (pilot only)
- [x] Add fixed replay-weight runner with controller betas disabled.
- [x] Run λ ∈ {0, 0.25, 0.5, 0.75, 1.0} over ETTh1 H=1 and seeds 0,1,2.
- [x] Re-run official FSNet and OneNet TCN smoke on the common causal boundary.
- [x] Confirm fixed λ=1.0 beats prior DPST-full pilot means on MAE and MSE.
- [ ] Redesign/simplify DPST replay controller before broader benchmark; full benchmark remains blocked.
