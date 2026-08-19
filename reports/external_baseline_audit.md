# External Baseline Audit

Date: 2026-08-19

This is an implementation-readiness audit only. No external baseline was
installed and no full benchmark was run.

| Baseline | Source identified | Official-source status | Current decision |
|---|---|---|---|
| DER++ | [Mammoth](https://github.com/aimagelab/mammoth), `derpp` strategy | Official framework repository for the implementation | Candidate after adapter and exact commit pinning |
| FSNet | [Salesforce FSNet](https://github.com/salesforce/fsnet) | Official repository linked by the paper/authors | Candidate, but its README requires Python 3.7.3/PyTorch 1.8.0 and must be isolated from the current environment |
| OneNet | [yfzhang114/OneNet](https://github.com/yfzhang114/OneNet) | Official PyTorch implementation identified by the paper | Candidate after exact commit pinning and causal-delay adapter audit |
| NatSR | Repository linked from the paper: `anonymous.4open.science/r/NatSR` | Paper-linked source, but anonymous and not yet verified as a stable official repository/commit | BLOCKED for integration until provenance and reproducibility are verified |

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

The common pipeline gate remains PASS. The external-baseline gate is **BLOCKED**
until NatSR provenance is verified and each candidate has an exact commit plus
a causal adapter. Full benchmark execution remains prohibited at this stage.
