# Failures and blockers

## Resolved during initial implementation

- A root-level `queue.py` shadowed Python's stdlib `queue` and prevented PyTorch import. It was removed; the causal queue is now under `src/continual_forecasting/queue.py`.
- Direct invocation of `scripts/run_smoke.py` hit Windows sandbox error `CreateProcessAsUserW: 1312`; the identical smoke entry point was executed through `python -m unittest discover` and passed.

## Current blockers

- Git is now initialized and pushed. The latest smoke manifest records commit `55d875c0f5f67767b2a92de4db2b7a5581f54be4` with `git_dirty=false`.
- No parquet engine was available in the environment; exact predictions are saved as `predictions.npz` and `predictions.csv`, with the fallback explicitly recorded. This is an artifact compatibility blocker for strict parquet-only consumers.
- DER++ causal adapter and ETTh1 smoke are now verified; exact official-source HEAD is recorded in `reports/external_baseline_audit.md`.
- FSNet and OneNet still require causal adapters. NatSR is marked unavailable because its paper-linked anonymous repository provenance is not verified.
- `requirements.lock.txt` now records the tested runtime and instrumentation dependencies; parquet remains optional because no parquet engine is installed.
- External adapter contract tests pass. FSNet official 1,999-sample smoke passes. OneNet official model import and fixed-blend engineering smoke pass, but the full adaptive OneNet method gate remains blocked until its decision/weight update is reproduced for the single-target protocol.
