# Current Report

> File này là report trạng thái hiện tại của repository. Mỗi lần chạy benchmark,
> preflight hoặc smoke test mới, hãy replace toàn bộ nội dung file này bằng kết
> quả mới nhất; không append kết quả cũ vào đây. Lịch sử chi tiết giữ trong
> `reports/` và Git.

## Version

- Report date: 2026-08-20
- Repository: `continual-learning`
- Git commit at start of milestone: `bb6dbe0`
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
| DPST trajectory equivalence | PASS | CPU deterministic test, `rtol=1e-6`, `atol=1e-8` |
| Three-seed H=1 ablation | PASS (pilot) | ETTh1, 1,999 resolved samples per run; not confirmatory |
| H=24 delayed-feedback smoke | PASS (engineering) | 176 resolved samples; release IDs `4355..4530` |
| Fixed replay-weight comparison | PASS (pilot) | 15/15 ETTh1 H=1 runs; fixed λ=1.0 best mean |

## Smoke result

- Dataset: ETTh1
- Split: 20% train / 5% validation / 75% online
- Lookback: `L=60`
- Horizon: `H=1`
- Seed: `7`
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
| FSNet fair warm-start | 0.200855 | 0.085754 | 2.262366 | 1,999 |
| OneNet adaptive (fair warm-start) | 0.202376 | 0.083336 | 2.279503 | 1,999 |

OneNet adaptive is now included directly in this Smoke result table. The old random-initialization result was invalidated; this row is prequential over the same 1,999 resolved ETTh1 samples after train-only warm-start and validation LR selection. Artifact: `artifacts/external_smoke/OneNet/smoke.json`.

## DPST ablation milestone (engineering/pilot)

All H=1 runs use ETTh1, the same 20%/5%/75% chronological split, train-only
scaler, `L=60`, validation-only LR selection, CUDA, and 1,999 resolved samples.
These artifacts are excluded from confirmatory statistics.

| Method | MAE mean ± std | MSE mean ± std | Seeds |
|---|---:|---:|---:|
| ER | 0.089916 ± 0.002777 | 0.014234 ± 0.000926 | 0,1,2 |
| DPST-eta | 0.091432 ± 0.003321 | 0.014659 ± 0.000997 | 0,1,2 |
| DPST-lambda | 0.089082 ± 0.002516 | 0.013976 ± 0.000840 | 0,1,2 |
| DPST-full | 0.089573 ± 0.003113 | 0.014052 ± 0.000954 | 0,1,2 |
| DER++ | 0.089703 ± 0.002542 | 0.014161 ± 0.000891 | 0,1,2 |

DPST-full beats ER and DER++ on mean MAE/MSE, and beats ER in every seed.
However, eta-only loses to ER in every seed, while lambda-only is the strongest
ablation; this supports a pilot component signal, not a claim that both
components independently help.

Artifacts: `artifacts/dpst_milestone/milestone_results.json` and
`artifacts/dpst_milestone/H1/DERPP/results.json`.

## H=24 delayed-feedback smoke

Seed 0, prefix 200, with 176 resolved samples and release IDs `4355..4530` for
ER, DPST, and DER++. Results: ER MAE/MSE `0.313632/0.157221`, DPST
`0.306701/0.152023`, DER++ `0.293207/0.140646`. This is an engineering smoke,
not a multi-seed or confirmatory result.

## Official External Baseline Smoke

| Method | Official commit | Device | Resolved | Release timing | Finite | Runtime | Status |
|---|---|---|---:|---|---|---:|---|
| FSNet fair warm-start | `c776afc623fa6384a6a559121aacadd2bbea5968` | CUDA | 1,999/1,999 | issue `4355..6353` → resolve `4356..6354` | yes | 217.10s | PASS |
| OneNet adaptive (fair warm-start) | `65eed9d6c878133a4d81d9c381c69e742ad47fd0` | CUDA | 1,999/1,999 | issue `4355..6353` → resolve `4356..6354` | yes | 70.37s online | PASS |

OneNet used the official `onenet_tcn` update sequence: branch forecaster
update, decision-MLP short-term bias update, and long-term sigmoid weight
update. The gate moved from `0.50025`; `branch_loss`, `decision_loss`, and
`weight_loss` are stored in
`artifacts/external_smoke/OneNet/smoke.json`. For the OT protocol, the final
official time-branch channel is used as the fixed scalar target projection.
The fair run warm-started on the train split and selected `lr=3e-4` using
validation only (`MSE=0.441491`; `lr=1e-3` scored `0.707949`).
FSNet fair warm-start selected `lr=3e-4` using validation only (`MSE=0.892541`;
`lr=1e-3` scored `1.368886`) and achieved MAE/MSE `0.200855/0.085754`.

## Verification

- Deterministic pipeline tests: `9/9 PASS`; external adapter tests: `3/3 PASS`.
- Full suite components now pass: deterministic pipeline `9/9`, external
  adapters `3/3`, and long smoke integration `1/1` (the latter required a
  longer timeout than the default command).
- Smoke integration test with DER++: `1/1 PASS`.
- Checkpoint/restore, pending queue, horizon release, scaler leakage và metric recomputation: PASS.
- Không ghi nhận NaN/Inf trong smoke.
- External adapter contract tests: `3/3 PASS`.
- Official source probe: FSNet and OneNet import PASS; `wandb==0.28.2` is locked.
- Latest smoke rerun: `1/1 PASS`; exact artifacts are under `artifacts/smoke_etth1/`.
- Official external smoke: OneNet fair warm-start `1999/1999 PASS` (70.37s online). The prior random-initialization result is invalidated; the fair row uses train-only warm-start and validation LR selection.
- DPST equivalence: PASS. H=1 ablation: 12/12 runs finite with 1,999/1,999 updates. H=24 smoke: 3/3 methods finite with 176/176 updates.
- Fixed replay-weight milestone: 15/15 runs finite; fixed λ=1.0 was best on
  both mean MAE and mean MSE.
- Official external TCN rerun: FSNet and OneNet both `1,999/1,999` finite.

## Artifacts

- Smoke report: `reports/benchmark_smoke.md`
- DPST report: `reports/dpst_report.md`
- Failure log/report: `reports/failures.md`
- External baseline audit: `reports/external_baseline_audit.md`
- Smoke artifacts: `artifacts/smoke_etth1/`
- Dependency lock: `requirements.lock.txt`
- Plots: `reports/figures/`

## Blockers and next action

1. DPST v2 has diagnostic headroom but is not implemented: the full-prefix
   fixed-trajectory oracle found changing block winners and positive gain.
   Full benchmark remains gated; NatSR stays unavailable.
2. Keep NatSR marked `unavailable` unless official provenance becomes verifiable.
3. Parquet remains optional; JSONL/CSV/NPZ are source of truth.
4. Re-run fairness/leakage tests after each adapter.
5. DPST-v1 controller is rejected; a new controller may be designed against
   the blockwise oracle evidence. Do not run full benchmark yet.

## Fixed replay-weight milestone

ETTh1 H=1, same split/scaler/LR selection/online prefix, three seeds, and
1,999 resolved samples per run. Controllers were disabled (`beta_eta=0`,
`beta_lambda=0`) while fixed replay weight λ was varied.

| Fixed λ | MAE mean ± std | MSE mean ± std |
|---:|---:|---:|
| 0.00 | 0.097534 ± 0.004224 | 0.016304 ± 0.001400 |
| 0.25 | 0.091718 ± 0.003259 | 0.014706 ± 0.001032 |
| 0.50 | 0.089916 ± 0.002777 | 0.014234 ± 0.000926 |
| 0.75 | 0.089168 ± 0.002565 | 0.014021 ± 0.000877 |
| 1.00 | **0.089062 ± 0.002591** | **0.013963 ± 0.000862** |

DPST-full from the prior pilot was MAE `0.089573` and MSE `0.014052`, so it
did not beat fixed λ=1.0. Artifact:
`artifacts/dpst_milestone/fixed_lambda/results.json`.
The recorded DPST-full trajectories reached λ=`0.9999999979` at the final
update for all three seeds, so the current controller is effectively
saturating at λ=1.0 rather than demonstrating an adaptive advantage.

## Full-prefix fixed-trajectory blockwise oracle

This artifact-only analysis reuses the existing 15 fixed-λ runs (ETTh1 H=1,
1,999 evaluated samples per seed) and partitions each trajectory into 20
blocks of 100 samples. MAE and MSE winners are computed separately. This is
counterfactual across independently trained fixed-λ trajectories, so it is a
headroom diagnostic rather than confirmatory evidence.

| Seed | MAE λ=1 win rate | MAE winner changes | MAE oracle gain | MSE λ=1 win rate | MSE winner changes | MSE oracle gain |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 50.0% | 15 | 0.0011274 | 50.0% | 12 | 0.0002811 |
| 1 | 30.0% | 12 | 0.0007428 | 50.0% | 13 | 0.0001561 |
| 2 | 55.0% | 15 | 0.0012376 | 45.0% | 12 | 0.0002365 |
| Aggregate | 45.0% | — | 0.0010359 (1.163%) | 48.33% | — | 0.0002246 (1.608%) |

The oracle therefore shows meaningful replay-weight headroom: λ=1 is not
blockwise optimal on this stream, even though it is the best fixed setting in
the aggregate end-to-end table. The prior 500-sample local counterfactual
diagnostic tested a different estimand (one-step audit loss on an ER-0.5
reference trajectory) and does not invalidate this full-prefix result.

Artifact: `artifacts/dpst_milestone/fixed_lambda/blockwise_oracle.json`.

## Blockwise oracle diagnostic

An artifact-only diagnostic evaluated counterfactual one-step updates with
λ ∈ {0, 0.5, 1.0} on a 500-sample ETTh1 H=1 prefix (490 resolved samples,
10 blocks of 50). Each decision used the current resolved sample and a
separate audit replay batch sampled only from previously resolved items;
audit items were excluded from the training replay draw. The best fixed
candidate was λ=1.0 (mean audit loss `0.0161349`). The blockwise oracle
selected λ=1.0 in all 10 blocks and did not improve over the best fixed
candidate (oracle mean `0.0162006`, gain `-0.0000657`). Therefore the
diagnostic does not show potential benefit for DPST v2 on this stream; do
not code v2 yet. Artifact: `artifacts/dpst_milestone/oracle_diagnostic.json`.

## Reproduction commands

```powershell
python -m unittest discover -s tests -q
python scripts/run_smoke.py --data data/all_six_datasets/ETT-small/ETTh1.csv --output artifacts/smoke_etth1 --prefix 2000 --device cpu
python scripts/run_dpst_milestone.py --device cuda --prefix 2000 --h24-prefix 200 --seeds 0 1 2
python scripts/run_derpp_three_seeds.py --device cuda --prefix 2000 --seeds 0 1 2
python scripts/run_fixed_lambda_milestone.py --device cuda --prefix 2000 --seeds 0 1 2
python scripts/run_external_smoke.py --prefix 1999 --methods FSNet OneNet
python scripts/run_dpst_oracle_diagnostic.py --device cuda --prefix 500 --seed 7 --block-size 50
python scripts/analyze_fixed_lambda_blocks.py --block-size 100
```
