# Current Report

> File này là report trạng thái hiện tại của repository. Mỗi lần chạy benchmark,
> preflight hoặc smoke test mới, hãy replace toàn bộ nội dung file này bằng kết
> quả mới nhất; không append kết quả cũ vào đây. Lịch sử chi tiết giữ trong
> `reports/` và Git.

## Version

- Report date: 2026-08-20
- Repository: `continual-learning`
- Git commit at start of milestone: `ef22c8b`
- Branch: `master`
- Working tree: modified (adapter/report changes; pre-existing root `curren_report.md` deletion preserved)
- Data gốc: không sửa hoặc commit

## Current gate

| Gate | Status | Ghi chú |
|---|---|---|
| Common causal pipeline | PASS | ETTh1 smoke đã chạy thành công |
| Leakage/fairness smoke tests | PASS | Unit và integration smoke pass |
| Dependency capture | PASS | Runtime smoke đã được khóa |
| External baseline readiness | PASS (except NatSR) | FSNet and OneNet official causal method smoke PASS; NatSR unavailable |
| Full benchmark | NOT RUN | Chưa được phép chạy |

## Smoke result

- Dataset: ETTh1
- Split: 20% train / 5% validation / 75% online
- Lookback: `L=60`
- Horizon: `H=1`
- Seed: cố định
- Resolved samples: `1,999`
- Methods: OGD, ER, DPST-Core, DER++
- Device: CPU
- Selected learning rate: `0.001`

| Method | MAE | MSE | MASE | Updates |
|---|---:|---:|---:|---:|
| OGD | 0.109258 | 0.019369 | 1.230648 | 1,999 |
| ER | 0.091804 | 0.014680 | 1.034047 | 1,999 |
| DPST-Core | 0.088667 | 0.014002 | 0.998713 | 1,999 |
| DER++ | 0.091031 | 0.014569 | 1.025342 | 1,999 |
| OneNet adaptive | 0.488431 | 7.505736 | 5.501539 | 1,999 |

OneNet adaptive is now included directly in this Smoke result table. Its metrics are prequential over the same 1,999 resolved ETTh1 samples and come from `artifacts/external_smoke/OneNet/smoke.json`; the method uses the official adaptive two-branch decision/update mechanism.

## Official External Baseline Smoke

| Method | Official commit | Device | Resolved | Release timing | Finite | Runtime | Status |
|---|---|---|---:|---|---|---:|---|
| FSNet | `c776afc623fa6384a6a559121aacadd2bbea5968` | CUDA | 1,999/1,999 | issue `4355..6353` → resolve `4356..6354` | yes | 178.89s | PASS |
| OneNet adaptive | `65eed9d6c878133a4d81d9c381c69e742ad47fd0` | CUDA | 1,999/1,999 | issue `4355..6353` → resolve `4356..6354` | yes | 173.15s | PASS |

OneNet used the official `onenet_tcn` update sequence: branch forecaster
update, decision-MLP short-term bias update, and long-term sigmoid weight
update. The gate moved from `0.50025`; `branch_loss`, `decision_loss`, and
`weight_loss` are stored in
`artifacts/external_smoke/OneNet/smoke.json`. For the OT protocol, the final
official time-branch channel is used as the fixed scalar target projection.

## Verification

- Pipeline and integration tests: `13/13 PASS`.
- Smoke integration test with DER++: `1/1 PASS`.
- Checkpoint/restore, pending queue, horizon release, scaler leakage và metric recomputation: PASS.
- Không ghi nhận NaN/Inf trong smoke.
- External adapter contract tests: `3/3 PASS`.
- Official source probe: FSNet and OneNet import PASS; `wandb==0.28.2` is locked.
- Latest smoke rerun: `1/1 PASS`; exact artifacts are under `artifacts/smoke_etth1/`.
- Official external smoke: FSNet `1999/1999 PASS` (178.89s); OneNet adaptive `1999/1999 PASS` (173.15s). OneNet gate moves from 0.50025 and updates over the stream; decision and weight losses are recorded in `artifacts/external_smoke/OneNet/smoke.json`.

## Artifacts

- Smoke report: `reports/benchmark_smoke.md`
- DPST report: `reports/dpst_report.md`
- Failure log/report: `reports/failures.md`
- External baseline audit: `reports/external_baseline_audit.md`
- Smoke artifacts: `artifacts/smoke_etth1/`
- Dependency lock: `requirements.lock.txt`
- Plots: `reports/figures/`

## Blockers and next action

1. Continue to the next predeclared benchmark milestone. Full benchmark remains gated separately; NatSR stays unavailable.
2. Keep NatSR marked `unavailable` unless official provenance becomes verifiable.
3. Parquet remains optional; JSONL/CSV/NPZ are source of truth.
4. Re-run fairness/leakage tests after each adapter.
5. Only after all gates pass, consider broader benchmark execution.

## Reproduction commands

```powershell
python -m unittest discover -s tests -q
python scripts/run_smoke.py --data data/all_six_datasets/ETT-small/ETTh1.csv --output artifacts/smoke_etth1 --prefix 2000 --device cpu
```
