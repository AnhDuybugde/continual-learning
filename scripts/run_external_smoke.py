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


def run_one(name: str, values: np.ndarray, timestamps, online_start: int, targets: tuple[int, ...], lookback: int, horizon: int, prefix: int, output: Path, device: str) -> dict:
    adapter = build_official_adapter(name, values.shape[1], len(targets), horizon, lookback, 1e-3, 7, device)
    adapter.set_timestamps(timestamps)
    queue = PendingQueue(horizon)
    rows = []
    start = online_start
    stop = min(len(values) - horizon, start + prefix + horizon)
    wall = time.perf_counter()
    for current_time in range(start, stop):
        pending = queue.pop_resolved(current_time)
        if pending is not None:
            target = values[current_time - horizon + 1 : current_time + 1, list(targets)].astype(np.float32)
            before = time.perf_counter()
            diagnostic = adapter.observe_resolved(pending.inputs, target, pending.issue_time)
            rows.append({"sample_id": pending.issue_time, "resolve_time": current_time, "update_seconds": time.perf_counter() - before, **diagnostic})
        inputs, _ = make_window(values, current_time, lookback, horizon, targets)
        forecast = adapter.predict(inputs, current_time)
        queue.add(PendingForecast(current_time, inputs, forecast, adapter.model_step))
        if len(rows) >= prefix:
            break
    result = {"name": name, "source": {"repository": adapter.source.repository, "commit": adapter.source.commit}, "prefix": prefix, "resolved": len(rows), "finite": all(row["finite_status"] for row in rows), "runtime_seconds": time.perf_counter() - wall, "rows": rows}
    (output / name).mkdir(parents=True, exist_ok=True)
    (output / name / "smoke.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=int, default=8)
    args = parser.parse_args()
    dataset = load_dataset("data/all_six_datasets/ETT-small/ETTh1.csv")
    scaler = TrainOnlyStandardScaler().fit(dataset.features[: dataset.split.train_end])
    values = scaler.transform(dataset.features).astype(np.float32)
    output = Path("artifacts/external_smoke")
    output.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = [run_one(name, values, dataset.timestamps, dataset.split.validation_end, target_indices(dataset), 60, 1, args.prefix, output, device) for name in ("FSNet", "OneNet")]
    (output / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
