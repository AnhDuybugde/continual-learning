# External Baseline Audit

Date: 2026-08-19

This is an implementation-readiness audit only. No external baseline was
installed and no full benchmark was run.

| Baseline | Source identified | Official-source status | Current decision |
|---|---|---|---|
| DER++ | [Mammoth](https://github.com/aimagelab/mammoth), `derpp` strategy | Official framework repository; HEAD checked 2026-08-19 | **Smoke PASS**, pinned HEAD `e75a491c69fd729edeb01431afb753d9157d9a81` |
| FSNet | [Salesforce FSNet](https://github.com/salesforce/fsnet) | Official repository linked by the paper/authors; HEAD checked 2026-08-19 | Candidate, pinned HEAD `c776afc623fa6384a6a559121aacadd2bbea5968`; causal adapter pending |
| OneNet | [yfzhang114/OneNet](https://github.com/yfzhang114/OneNet) | Official PyTorch implementation identified by the paper; HEAD checked 2026-08-19 | Candidate, pinned HEAD `65eed9d6c878133a4d81d9c381c69e742ad47fd0`; causal adapter pending |
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

The common pipeline gate remains PASS. DER++ is now smoke-verified. FSNet and
OneNet remain pending causal adapters; NatSR is unavailable. Full benchmark
execution remains prohibited at this stage.
