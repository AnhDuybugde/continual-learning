# Current Report

> File này là report trạng thái hiện tại của repository. Mỗi lần chạy benchmark,
> preflight hoặc smoke test mới, hãy replace toàn bộ nội dung file này bằng kết
> quả mới nhất; không append kết quả cũ vào đây. Lịch sử chi tiết giữ trong
> `reports/` và Git.

## Version

- Report date: 2026-08-19
- Repository: `continual-learning`
- Git commit: `7616bc4`
- Branch: `master`
- Working tree: clean
- Data gốc: không sửa hoặc commit

## Current gate

| Gate | Status | Ghi chú |
|---|---|---|
| Common causal pipeline | PASS | ETTh1 smoke đã chạy thành công |
| Leakage/fairness smoke tests | PASS | Unit và integration smoke pass |
| Dependency capture | PASS | Runtime smoke đã được khóa |
| External baseline readiness | BLOCKED | NatSR provenance chưa xác minh; chưa pin exact commit/adapters |
| Full benchmark | NOT RUN | Chưa được phép chạy |

## Smoke result

- Dataset: ETTh1
- Split: 20% train / 5% validation / 75% online
- Lookback: `L=60`
- Horizon: `H=1`
- Seed: cố định
- Resolved samples: `1,999`
- Methods: OGD, ER, DPST-Core
- Device: CPU
- Selected learning rate: `0.001`

| Method | MAE | MSE | MASE | Updates |
|---|---:|---:|---:|---:|
| OGD | 0.109258 | 0.019369 | 1.230648 | 1,999 |
| ER | 0.091804 | 0.014680 | 1.034047 | 1,999 |
| DPST-Core | 0.088667 | 0.014002 | 0.998713 | 1,999 |

## Verification

- Pipeline unit tests: `8/8 PASS`.
- Smoke integration test: `1/1 PASS`.
- Checkpoint/restore, pending queue, horizon release, scaler leakage và metric recomputation: PASS.
- Không ghi nhận NaN/Inf trong smoke.

## Artifacts

- Smoke report: `reports/benchmark_smoke.md`
- DPST report: `reports/dpst_report.md`
- Failure log/report: `reports/failures.md`
- External baseline audit: `reports/external_baseline_audit.md`
- Smoke artifacts: `artifacts/smoke_etth1/`
- Dependency lock: `requirements.lock.txt`

## Blockers and next action

1. Verify official provenance and exact commit for NatSR.
2. Pin exact commits and implement causal adapters for DER++, FSNet, OneNet and NatSR.
3. Add peak-memory instrumentation, automatic plots/report generation and parquet support before final benchmark.
4. Re-run fairness/leakage tests after each adapter.
5. Only after all gates pass, consider broader benchmark execution.

## Reproduction commands

```powershell
python -m unittest discover -s tests -q
python scripts/run_smoke.py --data data/all_six_datasets/ETT-small/ETTh1.csv --output artifacts/smoke_etth1 --prefix 2000 --device cpu
```
