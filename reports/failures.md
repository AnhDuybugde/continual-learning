# Failures and blockers

## Resolved during initial implementation

- A root-level `queue.py` shadowed Python's stdlib `queue` and prevented PyTorch import. It was removed; the causal queue is now under `src/continual_forecasting/queue.py`.
- Direct invocation of `scripts/run_smoke.py` hit Windows sandbox error `CreateProcessAsUserW: 1312`; the identical smoke entry point was executed through `python -m unittest discover` and passed.

## Current blockers

- The repository has no `.git` metadata yet, so commit/environment records contain no commit hash. Git initialization and remote push are intentionally deferred until this stage is reviewed/verified.
- No parquet engine was available in the environment; exact predictions are saved as `predictions.npz` and `predictions.csv`, with the fallback explicitly recorded. This is an artifact compatibility blocker for strict parquet-only consumers.
- DER++, FSNet, OneNet, and NatSR official-source integration is not started.
