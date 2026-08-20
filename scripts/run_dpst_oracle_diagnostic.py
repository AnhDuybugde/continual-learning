"""Diagnose whether adaptive replay has counterfactual value on ETTh1.

This is an artifact-producing diagnostic, not a benchmark method.  The
reference trajectory is fixed ER at lambda=0.5.  At each resolved sample,
each candidate performs one cloned update using the same training replay
draw, while a disjoint audit replay batch scores the post-update model.
All audit items were already resolved before the decision point.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from continual_forecasting.data import TrainOnlyStandardScaler, load_dataset, make_window, target_indices
from continual_forecasting.methods import ER
from continual_forecasting.replay import ReplayItem
from continual_forecasting.runner import SmokeConfig, _method, _warm_start, set_seed
from continual_forecasting.queue import PendingForecast, PendingQueue


CANDIDATES = (0.0, 0.5, 1.0)


def _loss(model: ER, item: ReplayItem) -> float:
    return float(model._loss(item.inputs, item.target).detach().cpu())


def _audit_batch(model: ER, training_ids: set[int], rng: np.random.Generator, size: int) -> list[ReplayItem]:
    candidates = [item for item in model.buffer.items if item.sample_id not in training_ids]
    if not candidates:
        return []
    count = min(size, len(candidates))
    indices = rng.choice(len(candidates), size=count, replace=False)
    return [candidates[int(index)] for index in indices]


def run_diagnostic(dataset_path: str, output: Path, device: str, prefix: int, seed: int, block_size: int) -> dict:
    set_seed(seed)
    dataset = load_dataset(dataset_path)
    cfg = SmokeConfig(seed=seed, horizon=1, online_prefix=prefix)
    scaler = TrainOnlyStandardScaler().fit(dataset.features[: dataset.split.train_end])
    values = scaler.transform(dataset.features).astype(np.float32)
    targets = target_indices(dataset)

    validation_scores: dict[float, float] = {}
    for lr in cfg.learning_rates:
        candidate = _method("ER", values.shape[1], len(targets), cfg, lr, device)
        _warm_start(candidate, values, targets, dataset.split.train_end, cfg)
        losses = []
        for end in range(dataset.split.train_end, dataset.split.validation_end - cfg.horizon):
            inputs, target = make_window(values, end, cfg.lookback, cfg.horizon, targets)
            losses.append(float(candidate._loss(inputs, target).detach().cpu()))
        validation_scores[lr] = float(np.mean(losses))
    selected_lr = min(validation_scores, key=validation_scores.get)
    reference = _method("ER", values.shape[1], len(targets), cfg, selected_lr, device)
    _warm_start(reference, values, targets, dataset.split.train_end, cfg)

    queue = PendingQueue(cfg.horizon)
    audit_rng = np.random.default_rng(seed + 100_003)
    block_rows: list[dict] = []
    current_block: list[dict] = []
    predictions: dict[int, np.ndarray] = {}
    start = dataset.split.validation_end
    stop = min(len(values) - cfg.horizon, start + prefix)
    for current_time in range(start, stop):
        pending = queue.pop_resolved(current_time)
        if pending is not None:
            target = values[current_time - cfg.horizon + 1 : current_time + 1, list(targets)].astype(np.float32)
            training_probe = copy.deepcopy(reference.buffer)
            training_replay = training_probe.sample(cfg.replay_batch_size)
            training_ids = {item.sample_id for item in training_replay}
            audit = _audit_batch(reference, training_ids, audit_rng, cfg.replay_batch_size)
            if audit:
                candidate_losses: dict[str, float] = {}
                for replay_weight in CANDIDATES:
                    cloned = _method("ER", values.shape[1], len(targets), cfg, selected_lr, device)
                    cloned.load_state_dict(copy.deepcopy(reference.state_dict()))
                    cloned.replay_weight = replay_weight
                    cloned.observe_resolved(pending.inputs, target, pending.issue_time)
                    candidate_losses[str(replay_weight)] = float(np.mean([_loss(cloned, item) for item in audit]))
                current_block.append({"sample_id": int(pending.issue_time), "candidate_audit_loss": candidate_losses, "audit_ids": sorted(item.sample_id for item in audit), "training_replay_ids": sorted(training_ids)})

            reference.replay_weight = 0.5
            reference.observe_resolved(pending.inputs, target, pending.issue_time)

        inputs, _ = make_window(values, current_time, cfg.lookback, cfg.horizon, targets)
        forecast = reference.predict(inputs, current_time)
        queue.add(PendingForecast(current_time, inputs, forecast, reference.model_step))
        if len(current_block) >= block_size:
            block_rows.append(_summarize_block(current_block))
            current_block = []
        if sum(int(row["n_samples"]) for row in block_rows) + len(current_block) >= prefix:
            break
    if current_block:
        block_rows.append(_summarize_block(current_block))

    all_losses = {
        str(candidate): [row["mean_audit_loss"][str(candidate)] for row in block_rows]
        for candidate in CANDIDATES
    }
    oracle_loss = sum(row["oracle_loss"] * row["n_samples"] for row in block_rows) / max(sum(row["n_samples"] for row in block_rows), 1)
    fixed_losses = {candidate: float(np.mean(values_)) for candidate, values_ in all_losses.items()}
    best_fixed = min(fixed_losses, key=fixed_losses.get)
    fixed_best_loss = fixed_losses[best_fixed]
    result = {
        "diagnostic": "blockwise_minimax_regret",
        "dataset": dataset.name,
        "seed": seed,
        "device": device,
        "horizon": cfg.horizon,
        "online_prefix": prefix,
        "resolved_samples": int(sum(row["n_samples"] for row in block_rows)),
        "candidate_lambdas": list(CANDIDATES),
        "block_size": block_size,
        "selected_learning_rate": selected_lr,
        "validation_scores": validation_scores,
        "reference": "ER_fixed_lambda_0.5",
        "fixed_mean_audit_loss": fixed_losses,
        "best_fixed_lambda": float(best_fixed),
        "best_fixed_mean_audit_loss": fixed_best_loss,
        "oracle_mean_audit_loss": oracle_loss,
        "oracle_gain_vs_best_fixed": fixed_best_loss - oracle_loss,
        "oracle_potential": bool(oracle_loss < fixed_best_loss),
        "blocks": block_rows,
        "causal_note": "Audit items are sampled only from previously resolved buffer items and excluded from the training replay draw.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _summarize_block(rows: list[dict]) -> dict:
    losses = {str(candidate): [row["candidate_audit_loss"][str(candidate)] for row in rows] for candidate in CANDIDATES}
    means = {candidate: float(np.mean(values)) for candidate, values in losses.items()}
    winner = min(means, key=means.get)
    return {"start_sample_id": rows[0]["sample_id"], "end_sample_id": rows[-1]["sample_id"], "n_samples": len(rows), "mean_audit_loss": means, "oracle_lambda": float(winner), "oracle_loss": means[winner], "candidate_losses": losses, "audit_size": len(rows[0]["audit_ids"])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/all_six_datasets/ETT-small/ETTh1.csv")
    parser.add_argument("--output", default="artifacts/dpst_milestone/oracle_diagnostic.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prefix", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--block-size", type=int, default=100)
    args = parser.parse_args()
    print(json.dumps(run_diagnostic(args.data, Path(args.output), args.device, args.prefix, args.seed, args.block_size), indent=2))


if __name__ == "__main__":
    main()
