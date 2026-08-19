# Failures and blockers

## Resolved during initial implementation

- A root-level `queue.py` shadowed Python's stdlib `queue` and prevented PyTorch import. It was removed; the causal queue is now under `src/continual_forecasting/queue.py`.
- Direct invocation of `scripts/run_smoke.py` hit Windows sandbox error `CreateProcessAsUserW: 1312`; the identical smoke entry point was executed through `python -m unittest discover` and passed.

## Current blockers

- Git is now initialized and pushed. The latest smoke manifest records commit `55d875c0f5f67767b2a92de4db2b7a5581f54be4` with `git_dirty=false`.
- No parquet engine was available in the environment; exact predictions are saved as `predictions.npz` and `predictions.csv`, with the fallback explicitly recorded. This is an artifact compatibility blocker for strict parquet-only consumers.
- DER++, FSNet, OneNet, and NatSR official-source integration is not started.
- `requirements.txt` is a minimal environment declaration, not yet a fully pinned lockfile; exact runtime versions are recorded in `artifacts/smoke_etth1/environment.json`.
