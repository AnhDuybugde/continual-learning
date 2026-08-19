# TODO: Fair and Fast Online-Continual Forecasting Benchmark

## 0. Goal and non-negotiable rules

Build one reproducible benchmark for `OGD`, `ER`, `DER++`, `FSNet`, `OneNet`, optional `NatSR`, and `DPST`. Start with a smoke run, then run the full experiment only after all leakage/fairness tests pass.

- [ ] Use one chronological `20% train / 5% validation / 75% online` split for every method.
- [ ] Fit scaler on the first 20% only; reuse it unchanged afterward.
- [ ] Tune only on the 5% validation segment. Never select configs from the 75% online results.
- [ ] Enforce the same lookback, horizon, target channels, data order, metrics, feedback delay, seeds, and compute/memory accounting.
- [ ] Predict before a sample is trained; for horizon `H`, release its complete label only after `H` raw time steps.
- [ ] Never silently skip, shorten, shuffle, or repair a failed run. Record the failure.

## 1. Repository and output contract

- [ ] Create a clean Python project with pinned dependencies and record Python/CUDA/PyTorch versions.
- [ ] Store the current git commit and dirty-state flag in every run.
- [ ] Use this output layout:

```text
artifacts/<run_id>/
  config.yaml
  environment.json
  data_manifest.json
  metrics.json
  online_metrics.jsonl
  predictions.parquet
  timing.json
  logs/run.log
  logs/failure.txt          # only on failure
  checkpoints/             # best warm-up checkpoint only
reports/
  benchmark_smoke.md
  benchmark_final.md
  failures.md
```

- [ ] Make each command resume-safe: completed runs are not overwritten; incomplete runs resume or receive a new `run_id`.
- [ ] Generate a report even when some methods fail.

## 2. Data pipeline

- [ ] Support ETTh1, ETTh2, ETTm1, ETTm2 first; add ECL, Traffic, Weather and user datasets through one dataset interface.
- [ ] Verify timestamps are sorted, unique or explicitly aggregated, and have the expected sampling interval.
- [ ] Hash raw files and save row count, feature count, time range, missing-value policy, split indices, lookback, horizon, and target columns in `data_manifest.json`.
- [ ] Construct direct multi-horizon samples: input `X_t=[x_{t-L+1},...,x_t]`, target `Y_t=[x_{t+1},...,x_{t+H}]`.
- [ ] Implement a shared causal online driver:
  1. receive the new raw observation;
  2. score any forecast whose full `H`-step target has just become available;
  3. update the method using that resolved sample only;
  4. issue the next forecast and add it to a pending queue.
- [ ] Unit-test that no target value with timestamp greater than the current simulated time enters an update.
- [ ] Compute MASE denominator from the warm-up training segment only. Guard against zero denominator.

## 3. Model integration order

Use installed, maintained packages when they expose the required method. Otherwise use the official repository pinned to an exact commit. Do not copy code from unofficial tutorials.

- [ ] Implement a shared TCN backbone and update interface: `predict`, `observe_resolved`, `state_dict`, `resource_usage`.
- [ ] `OGD`: shared TCN + one online optimizer update on each resolved sample; no replay.
- [ ] `ER`: same TCN/optimizer + reservoir buffer; train on current sample plus replay minibatch.
- [ ] `DER++`: same TCN; adapt the official DER++ objective to regression and document exactly what prediction/target is stored.
- [ ] `FSNet`: prefer official `salesforce/fsnet`; pin commit and adapt only its data/update interface.
- [ ] `OneNet`: prefer official `yfzhang114/OneNet`; pin commit and preserve its cross-time/cross-variable design.
- [ ] `NatSR` optional: use only official executable code matching the paper. If unavailable, incompatible, or irreproducible, mark it unavailable; do not create an approximate version and call it NatSR.
- [ ] `DPST`: implement from `dpst_method_todo.md` using the same shared TCN and buffer budget as ER.
- [ ] Record for every method: source/package version, commit, trainable parameters, model-state bytes, replay bytes, optimizer-state bytes, and any unavoidable architectural difference.

## 4. Fair hyperparameter protocol

- [ ] Warm-start every method on the 20% train segment; use the 5% segment as a chronological validation stream.
- [ ] Use official architecture defaults unless they are incompatible with the dataset.
- [ ] Tune the online learning rate for every gradient-based method over the same small grid: `{1e-4, 3e-4, 1e-3, 3e-3}`.
- [ ] Fix common buffer size and replay batch size across ER, DER++, NatSR, and DPST whenever the method permits it.
- [ ] Initially keep method-specific parameters at official defaults. If a default fails, permit one predeclared small grid and record the exception before seeing online-test results.
- [ ] Give methods the same maximum number of validation configurations. Stop all configs by the same validation-step or wall-time rule.
- [ ] Select by validation MASE; break ties by lower validation MAE, then lower runtime.
- [ ] After selection, reset model/optimizer/buffer state and run the 75% online stream once per seed.

## 5. Fast smoke run

Smoke results are for debugging only and must not appear as final scientific results.

- [ ] Dataset: ETTh1; horizon `H=1`; lookback `L=60`; one seed.
- [ ] Limit online evaluation to a clearly recorded prefix (for example 2,000 resolved samples).
- [ ] Run in this order: OGD -> ER -> DPST -> DER++ -> FSNet -> OneNet -> NatSR if available.
- [ ] Tune only two learning-rate candidates in smoke mode.
- [ ] Save MASE, MAE, MSE, runtime, peak CPU/GPU memory, prediction count, update count, and buffer count.
- [ ] Assert finite predictions/losses, identical evaluated timestamps, identical number of resolved targets, and no leakage.
- [ ] Write `reports/benchmark_smoke.md` with pass/fail status and the exact command needed to reproduce each failure.

## 6. Full experiment

- [ ] Use at least ETTh1/2, ETTm1/2, ECL, Traffic, and Weather.
- [ ] Use horizons `{1, 24, 48}` where supported; document any dataset exception.
- [ ] Use 3 fixed seeds and report mean plus standard deviation.
- [ ] Primary metric: online/prequential MASE. Also save MAE, MSE, rolling MASE, worst-window MASE, runtime, peak memory, and model/update counts.
- [ ] Use the same rolling-window length in raw time units across methods for a dataset.
- [ ] Keep immediate-feedback reproduction, if needed for an external repository, outside the main causal table and label it non-comparable.

## 7. Required tests and fairness audit

- [ ] Split/scaler leakage tests.
- [ ] Horizon-delay and pending-queue tests for `H=1,24,48`.
- [ ] Same sample/timestamp order across all methods.
- [ ] `OGD == ER` when replay batch size is zero.
- [ ] Reservoir buffer never exceeds capacity and is deterministic for a fixed seed.
- [ ] Metrics recomputed from `predictions.parquet` match `metrics.json`.
- [ ] Failed/non-convergent runs remain in the report and are not replaced post hoc.
- [ ] Compare effective memory and parameter counts; discuss deviations rather than claiming exact equality when FSNet/OneNet require custom architectures.

## 8. Final report checklist

- [ ] Write `reports/benchmark_final.md` for a reader who did not run the code.
- [ ] Include experiment question, data/splits, causal timing, preprocessing, model sources/commits, selected validation configs, and hardware.
- [ ] Provide result tables by dataset/horizon with mean +/- std and clearly mark unavailable/failed runs.
- [ ] Include rolling-error plots or links to generated figures, compute/memory table, and a concise fairness audit.
- [ ] Add a debugging section: first failing step, stack trace path, config path, suspected cause, and whether retry is safe.
- [ ] Add limitations: architectural mismatches, unavailable official code, reduced smoke prefix, and any deviation from paper protocols.
- [ ] End with exact reproduction commands and a machine-readable manifest of all included `run_id`s.

## Definition of done

- [ ] All leakage/fairness tests pass.
- [ ] Smoke report exists before any full run.
- [ ] Every reported number is traceable to config, prediction, log, environment, and data hash.
- [ ] Final report is generated automatically and remains useful when one or more external baselines fail.
