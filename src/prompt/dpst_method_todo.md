# TODO: Implement DPST-Core for Delayed Online Continual Forecasting

## 0. Method definition

DPST is a lightweight continual-learning wrapper around a deep forecasting backbone. Version 1 uses a shared TCN with AdamW, a causal pending queue, reservoir replay, adaptive learning-rate control, and adaptive replay-weight control. Do not add a neural controller, drift classifier, NatSR/Fisher machinery, or relevance-aware replay in the first implementation.

The two controlled variables are:

- `eta_t`: online learning rate / plasticity.
- `lambda_t`: replay-loss weight / stability.

DPST must support direct multi-horizon forecasting and real feedback delay: a forecast issued at raw time `s` can be scored/trained only when time `s + H` is observed.

## 1. Components and state

- [ ] `backbone`: the same TCN implementation used by OGD/ER.
- [ ] `optimizer`: AdamW; DPST changes its current learning rate but preserves optimizer moments.
- [ ] `pending_queue`: stores issue time, input window, target interval, prediction, and model step.
- [ ] `replay_buffer`: reservoir buffer storing `(X, Y, ref_loss, insert_step, sample_id)`.
- [ ] Controller state:
  - `log_eta`, bounded by configured `eta_min/eta_max`;
  - `replay_logit`, mapped to `lambda_t=lambda_max*sigmoid(replay_logit)`;
  - `alignment_ema`;
  - previous actual descent direction `p_prev` as a list of parameter-shaped tensors;
  - controller update count.
- [ ] Log every controller state transition; never reconstruct it only from checkpoints.

## 2. Configuration defaults

Expose all values in YAML. Use these only as initial smoke defaults, then tune on the 5% validation stream:

```yaml
lookback: 60
horizon: 1
optimizer: adamw
eta_init: 0.001
eta_min: 0.00001
eta_max: 0.01
lambda_init: 0.5
lambda_max: 1.0
beta_eta: 0.01
beta_lambda: 0.01
alignment_ema_decay: 0.9
forget_budget: 0.10
forget_score_clip: 5.0
buffer_size: 500
replay_batch_size: 8
controller_eps: 1.0e-8
```

- [ ] Tune `eta_init` on the same common grid as other baselines.
- [ ] For DPST-specific validation, use at most a small predeclared grid for `(beta_eta, beta_lambda, forget_budget)`; do not tune from the 75% online stream.
- [ ] Save chosen values and rejected validation results.

## 3. Causal online loop

At each raw time `t`, perform the following order:

- [ ] Receive the newly observed vector `x_t`.
- [ ] Resolve the pending forecast issued at `s=t-H`, if one exists. Its full target is `Y_s=[x_{s+1},...,x_{s+H}]`.
- [ ] Record its prequential prediction and loss before training on that sample.
- [ ] Call `observe_resolved(X_s, Y_s)` to update controller/model/buffer.
- [ ] Build the newest valid lookback `X_t`; forecast `Y_t`; enqueue the unresolved forecast.
- [ ] During the first `H` online steps, issue forecasts but do not invent labels or updates.

Unit-test the loop with a tiny numbered series so every input/target/release timestamp can be inspected exactly.

## 4. Controller update and model update

Implement per-sample loss (`reduction='none'`) and a scalar mean loss for backpropagation.

### 4.1 Fresh and replay gradients

- [ ] Compute `g_new = grad(loss_new, theta)` on the newly resolved sample.
- [ ] Sample replay batch `R_t` with the method RNG only; compute `loss_rep` and `g_rep`.
- [ ] If the buffer is empty, set `lambda_t=0` for the model update and skip the forgetting update.
- [ ] Do not call `optimizer.step()` until both controller updates below are complete.

### 4.2 Plasticity controller

Let `p_prev` be the previous *descent direction*, defined from the actual parameter change:

```text
p_prev = (theta_before - theta_after) / max(eta_used, eps)
```

Compute bounded alignment:

```text
c_t = dot(g_new, p_prev) / (norm(g_new) * norm(p_prev) + eps)
c_t = clip(c_t, -1, 1)
alignment_ema = rho * alignment_ema + (1-rho) * c_t
log_eta = clip(log_eta + beta_eta * alignment_ema,
               log(eta_min), log(eta_max))
eta_t = exp(log_eta)
```

- [ ] On the first resolved sample, keep `eta_init` because `p_prev` is absent.
- [ ] Detach all controller signals; DPST-Core does not backpropagate through time or through the controller.
- [ ] Apply `eta_t` to every optimizer parameter group before the model update.

### 4.3 Stability controller

For replay sample `i`, compare current pre-update loss with the stored reference loss:

```text
f_i = relu(loss_now_i - ref_loss_i) / (ref_loss_i + eps)
F_t = mean(f_i)
F_t = clip(F_t, 0, forget_score_clip)
replay_logit = clip(replay_logit + beta_lambda * (F_t - forget_budget),
                    logit_min, logit_max)
lambda_t = lambda_max * sigmoid(replay_logit)
```

- [ ] Initialize `replay_logit=logit(lambda_init/lambda_max)` with numerical clipping.
- [ ] Reference loss is computed once, after the sample's first model update, using evaluation mode and no gradient; do not silently refresh it.
- [ ] Retention measurement uses the sampled replay batch before the current update.
- [ ] Save `F_t`, mean current/reference loss, and the number of valid replay items.

### 4.4 Backbone update

- [ ] Recompute one combined differentiable loss if earlier gradient probes consumed the graph:

```text
loss_total = loss_new + lambda_t * loss_rep
```

- [ ] Snapshot parameters, set optimizer LR to `eta_t`, run `zero_grad -> backward -> optional fixed gradient clipping -> step`.
- [ ] Compute and store the actual `p_prev` from parameter changes. Include AdamW weight decay because it is part of the applied update.
- [ ] Insert the resolved sample through reservoir sampling and store its post-update `ref_loss` if selected.
- [ ] Never insert unresolved samples or future targets.

## 5. Public interface and saved diagnostics

- [ ] Implement `predict(X, issue_time)`, `observe_resolved(X,Y,sample_id)`, `state_dict()`, and `load_state_dict()`.
- [ ] Checkpoint backbone, optimizer, pending queue, buffer, controller state, RNG states, scaler metadata, and counters.
- [ ] Append one record per resolved sample to `online_metrics.jsonl` with:

```text
sample_id, issue_time, resolve_time, prequential_loss,
eta, lambda, alignment, alignment_ema, forget_score,
loss_new, loss_replay, buffer_size, update_norm,
prediction_ms, update_ms, finite_status
```

- [ ] Save predictions/targets/timestamps to `predictions.parquet`.
- [ ] On NaN/Inf, stop safely, save the last good checkpoint, failing batch IDs, controller state, optimizer state summary, and stack trace.

## 6. Tests

- [ ] `H=1,24,48` feedback-release tests; no update can use an incomplete target.
- [ ] Bounds: `eta_min <= eta_t <= eta_max`, `0 <= lambda_t <= lambda_max` for long random runs.
- [ ] First-step behavior with empty buffer and absent `p_prev`.
- [ ] `beta_eta=beta_lambda=0` reproduces fixed-parameter ER within tolerance.
- [ ] If replay losses worsen above budget, `lambda_t` increases; if consistently below budget, it decreases.
- [ ] Repeated positive alignment raises `eta_t`; repeated negative alignment lowers it.
- [ ] Save/load checkpoint produces identical next prediction/update with a fixed seed.
- [ ] Reservoir capacity, deterministic sampling, scaler isolation, and metric-recomputation tests.

## 7. Experiment and ablation order

- [ ] Smoke: ETTh1, `H=1`, one seed, short recorded online prefix.
- [ ] Compare shared-backbone variants in this order:
  1. OGD;
  2. ER with fixed `eta/lambda`;
  3. `DPST-eta` (`beta_lambda=0`);
  4. `DPST-lambda` (`beta_eta=0`);
  5. `DPST-Full`.
- [ ] Only after smoke/tests pass, run the full benchmark defined in `baseline_benchmark_todo.md`.
- [ ] Do not add relevance weighting until DPST-Core ablations are complete. If added later, create a separate config/ablation and never rewrite Core results.

## 8. Required DPST report

- [ ] Generate `reports/dpst_report.md` automatically.
- [ ] State the exact DPST equations, causal event order, configuration, data hashes, commit, and hardware.
- [ ] Report primary forecasting metrics plus runtime/memory and mean/min/max/final `eta` and `lambda`.
- [ ] Plot or link trajectories for prequential loss, rolling MASE, `eta`, `lambda`, alignment, and forgetting score on the same time axis.
- [ ] Include ablation tables and identify whether gains come from plasticity control, stability control, or both.
- [ ] Include debug notes for saturation at bounds, oscillation, empty/biased replay, exploding update norm, and delayed-feedback mistakes.
- [ ] Discuss failures honestly: DPST may protect obsolete samples, reference loss may be noisy, and adaptive replay may add compute.
- [ ] End with exact reproduction commands and paths to configs/logs/checkpoints/predictions.

## Definition of done

- [ ] DPST-Core is causal, bounded, checkpointable, and passes all tests.
- [ ] Fixed-controller DPST matches ER, proving the wrapper itself does not change the experiment.
- [ ] Smoke report is readable and sufficient to reproduce/debug every result.
- [ ] Full results are not claimed until shared-backbone ablations and fairness audit are complete.
