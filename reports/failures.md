# Failures and blockers

## Resolved during initial implementation

- A root-level `queue.py` shadowed Python's stdlib `queue` and prevented PyTorch import. It was removed; the causal queue is now under `src/continual_forecasting/queue.py`.
- Direct invocation of `scripts/run_smoke.py` hit Windows sandbox error `CreateProcessAsUserW: 1312`; the identical smoke entry point was executed through `python -m unittest discover` and passed.

## Current blockers

- Git is now initialized and pushed. The latest smoke manifest records commit `55d875c0f5f67767b2a92de4db2b7a5581f54be4` with `git_dirty=false`.
- No parquet engine was available in the environment; exact predictions are saved as `predictions.npz` and `predictions.csv`, with the fallback explicitly recorded. This is an artifact compatibility blocker for strict parquet-only consumers.
- DER++ causal adapter and ETTh1 smoke are now verified; exact official-source HEAD is recorded in `reports/external_baseline_audit.md`.
- FSNet and OneNet causal adapters are verified. NatSR is marked unavailable because its paper-linked anonymous repository provenance is not verified.
- `requirements.lock.txt` now records the tested runtime and instrumentation dependencies; parquet remains optional because no parquet engine is installed.
- External adapter contract tests pass. FSNet and OneNet official 1,999-sample causal smoke pass. OneNet's adaptive decision and long-term weight trajectories are now recorded; OT projection remains documented as the only architectural compatibility note.
- The default 120-second full-test command timed out, but the long smoke test
  passes 1/1 with an extended timeout; deterministic pipeline (9/9) and
  external adapter (3/3) suites also pass. This was a command-timeout issue,
  not a test failure.
- DPST H=1 three-seed ablation and H=24 smoke completed with finite losses and
  full resolved counts. Eta-only did not improve ER; this is a scientific
  limitation of the current pilot signal, not a technical failure.
