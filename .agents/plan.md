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
- [~] Official FSNet/OneNet causal wrappers chạy được 1,999-sample engineering smoke; OneNet vẫn chưa đạt method gate vì wrapper hiện dùng fixed 0.5/0.5 blend thay vì đầy đủ adaptive decision update.
- [x] Cài và khóa `wandb==0.28.2`; isolated import probe của cả FSNet và OneNet PASS.
- [ ] Hoàn thiện OneNet adaptive decision/weight update theo official loop rồi rerun external smoke; không gọi fixed-blend smoke là OneNet method PASS.
- [ ] Full benchmark/report chưa được phép chạy ở giai đoạn hiện tại.
- [ ] Khởi tạo Git, commit, verify rồi push remote `https://github.com/AnhDuybugde/continual-learning`.
