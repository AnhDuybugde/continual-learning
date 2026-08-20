"""Run a short causal smoke through the pinned official FSNet/OneNet nets."""

from __future__ import annotations

import json
import time
import argparse
from pathlib import Path

import numpy as np
import torch

from continual_forecasting.data import TrainOnlyStandardScaler, load_dataset, make_window, target_indices
from continual_forecasting.external_adapters import build_official_adapter
from continual_forecasting.queue import PendingForecast, PendingQueue
from continual_forecasting.runner import SmokeConfig


def warm_start(adapter, values: np.ndarray, targets: tuple[int, ...], train_end: int, lookback: int, horizon: int, epochs: int) -> None:
    """Warm-start an external method only on the chronological train split."""
    limit = max(lookback, train_end - horizon)
    for _ in range(epochs):
        for end in range(lookback - 1, limit):
            inputs, target = make_window(values, end, lookback, horizon, targets)
            adapter.observe_resolved(inputs, target, end)


def select_and_warm_start(name: str, values: np.ndarray, timestamps, train_end: int, validation_end: int, targets: tuple[int, ...], cfg: SmokeConfig, device: str):
    """Select LR on validation after identical train-only warm-start."""
    scores = {}
    for lr in cfg.learning_rates:
        candidate = build_official_adapter(name, values.shape[1], len(targets), cfg.horizon, cfg.lookback, lr, cfg.seed, device)
        candidate.set_timestamps(timestamps)
        warm_start(candidate, values, targets, train_end, cfg.lookback, cfg.horizon, cfg.warm_epochs)
        losses = []
        for end in range(train_end, validation_end - cfg.horizon):
            inputs, target = make_window(values, end, cfg.lookback, cfg.horizon, targets)
            prediction = candidate.predict(inputs, end)
            losses.append(float(np.mean((prediction - target) ** 2)))
        scores[lr] = float(np.mean(losses))
    selected_lr = min(scores, key=scores.get)
    adapter = build_official_adapter(name, values.shape[1], len(targets), cfg.horizon, cfg.lookback, selected_lr, cfg.seed, device)
    adapter.set_timestamps(timestamps)
    warm_start(adapter, values, targets, train_end, cfg.lookback, cfg.horizon, cfg.warm_epochs)
    return adapter, selected_lr, scores


def run_one(name: str, values: np.ndarray, timestamps, train_end: int, validation_end: int, online_start: int, targets: tuple[int, ...], cfg: SmokeConfig, output: Path, device: str) -> dict:
    adapter, selected_lr, validation_scores = select_and_warm_start(name, values, timestamps, train_end, validation_end, targets, cfg, device)
    adapter.set_timestamps(timestamps)
    queue = PendingQueue(cfg.horizon)
    rows = []
    start = online_start
    stop = min(len(values) - cfg.horizon, start + cfg.online_prefix + cfg.horizon)
    wall = time.perf_counter()
    for current_time in range(start, stop):
        pending = queue.pop_resolved(current_time)
        if pending is not None:
            target = values[current_time - cfg.horizon + 1 : current_time + 1, list(targets)].astype(np.float32)
            prequential_error = pending.prediction - target
            before = time.perf_counter()
            diagnostic = adapter.observe_resolved(pending.inputs, target, pending.issue_time)
            rows.append({"sample_id": pending.issue_time, "resolve_time": current_time, "prequential_mae": float(np.mean(np.abs(prequential_error))), "prequential_mse": float(np.mean(prequential_error ** 2)), "update_seconds": time.perf_counter() - before, **diagnostic})
        inputs, _ = make_window(values, current_time, cfg.lookback, cfg.horizon, targets)
        forecast = adapter.predict(inputs, current_time)
        queue.add(PendingForecast(current_time, inputs, forecast, adapter.model_step))
        if len(rows) >= cfg.online_prefix:
            break
    result = {"name": name, "source": {"repository": adapter.source.repository, "commit": adapter.source.commit}, "selected_learning_rate": selected_lr, "validation_scores": validation_scores, "prefix": cfg.online_prefix, "resolved": len(rows), "finite": all(row["finite_status"] for row in rows), "prequential_mae": float(np.mean([row["prequential_mae"] for row in rows])), "prequential_mse": float(np.mean([row["prequential_mse"] for row in rows])), "runtime_seconds": time.perf_counter() - wall, "rows": rows}
    (output / name).mkdir(parents=True, exist_ok=True)
    (output / name / "smoke.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=int, default=8)
    parser.add_argument("--methods", nargs="+", default=("FSNet", "OneNet"))
    args = parser.parse_args()
    dataset = load_dataset("data/all_six_datasets/ETT-small/ETTh1.csv")
    scaler = TrainOnlyStandardScaler().fit(dataset.features[: dataset.split.train_end])
    values = scaler.transform(dataset.features).astype(np.float32)
    cfg = SmokeConfig(online_prefix=args.prefix)
    output = Path("artifacts/external_smoke")
    output.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = [run_one(name, values, dataset.timestamps, dataset.split.train_end, dataset.split.validation_end, dataset.split.validation_end, target_indices(dataset), cfg, output, device) for name in args.methods]
    (output / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
