# External Baseline Audit

Date: 2026-08-19

This is an implementation-readiness audit only. No external baseline was
installed and no full benchmark was run.

| Baseline | Source identified | Official-source status | Current decision |
|---|---|---|---|
| DER++ | [Mammoth](https://github.com/aimagelab/mammoth), `derpp` strategy | Official framework repository; HEAD checked 2026-08-19 | **Smoke PASS**, pinned HEAD `e75a491c69fd729edeb01431afb753d9157d9a81` |
| FSNet | [Salesforce FSNet](https://github.com/salesforce/fsnet) | Official repository linked by the paper/authors; HEAD checked 2026-08-19 | Pinned HEAD `c776afc623fa6384a6a559121aacadd2bbea5968`; causal contract PASS, official entry import PASS |
| OneNet | [yfzhang114/OneNet](https://github.com/yfzhang114/OneNet) | Official PyTorch implementation identified by the paper; HEAD checked 2026-08-19 | Pinned HEAD `65eed9d6c878133a4d81d9c381c69e742ad47fd0`; causal contract PASS, official import PASS, adaptive smoke PASS |
| NatSR | Repository linked from the paper: `anonymous.4open.science/r/NatSR` | Paper-linked anonymous source; official provenance not verified | **unavailable**, not used and not blocking other baselines |

## Fairness checks still required

Each adapter must preserve the common ETTh1 split, train-only scaler, `L=60`,
horizon, pending feedback release rule, evaluated sample IDs, and forecast-before-
update ordering. The legacy repositories must not silently use labels before
their release time. Their own preprocessing and optimizer defaults cannot be
copied into the confirmatory benchmark without an explicit protocol mapping.

## Environment lock

The current smoke environment is captured in `requirements.lock.txt` and in the
smoke artifact environment manifest: Python 3.10.9, NumPy 2.2.6, pandas 2.3.3,
and PyTorch 2.6.0+cu124. The external repositories require compatibility
testing in isolated environments; the current benchmark environment must not
be downgraded in place.

## Gate

The common pipeline gate remains PASS. DER++, FSNet, and OneNet are now
smoke-verified through the shared causal boundary. NatSR is unavailable.
Full benchmark execution remains prohibited at this stage.

## Causal adapter verification (2026-08-19)

- `src/continual_forecasting/external_adapters.py` defines the shared causal boundary: prediction has no target argument, and targets are consumed only by `observe_resolved`.
- Contract tests verify prediction-before-update, resolve-only target use, finite loss, and checkpoint restore.
- FSNet official entry imports successfully at the pinned commit.
- OneNet official entry now imports after installing and locking `wandb==0.28.2`; isolated loading is required because of its legacy top-level module namespace.
- The pinned official model paths ran 1,999 resolved samples with finite values and correct `H=1` release timing. FSNet includes its official post-update gradient-store hook.
- The OneNet wrapper now reproduces the official `onenet_tcn` sequence: branch forecaster update, short-term decision-MLP bias update, and long-term sigmoid weight update on detached branch forecasts. The ETTh1 OT adapter uses the final official time-branch channel as the scalar target projection; this mapping is fixed before online evaluation and is not tuned online.
- Therefore FSNet and OneNet causal adapter smoke gates are **PASS**. The fair FSNet run selected `lr=3e-4` and achieved MAE/MSE `0.200855/0.085754` in 217.10 seconds for 1,999 resolved samples; the final OneNet run used the corrected decision-gradient path and fair warm-start. The OT projection is an architectural compatibility note, not a post-hoc result adjustment.
- The final common CUDA rerun completed with FSNet `0.200855/0.085754` and
  OneNet `0.202376/0.083336` MAE/MSE, each `1,999/1,999` finite. This confirms
  causal protocol fairness, but the official architectures are not compute-
  matched to the native shared-TCN methods.
