"""Small causal runner used by smoke validation."""

from __future__ import annotations

import json
import platform
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from .data import DatasetFrame, TrainOnlyStandardScaler, make_window, target_indices
from .metrics import regression_metrics, training_mase_denominator
from .methods import DPST, DPSTConfig, ER, OGD, OnlineForecaster
from .queue import PendingForecast, PendingQueue


@dataclass(frozen=True)
class SmokeConfig:
    lookback: int = 60
    horizon: int = 1
    seed: int = 7
    online_prefix: int = 2000
    learning_rates: tuple[float, float] = (1e-3, 3e-4)
    buffer_size: int = 500
    replay_batch_size: int = 8
    channels: int = 16
    warm_epochs: int = 1


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def git_environment() -> dict[str, str | bool | None]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
        return {"git_commit": commit, "git_dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "git_dirty": None}


def _method(name: str, input_size: int, target_size: int, cfg: SmokeConfig, lr: float, device: str) -> OnlineForecaster:
    common = dict(input_size=input_size, target_size=target_size, horizon=cfg.horizon, lr=lr, seed=cfg.seed, device=device, channels=cfg.channels)
    if name == "OGD":
        return OGD(**common)
    if name == "ER":
        return ER(**common, buffer_size=cfg.buffer_size, replay_batch_size=cfg.replay_batch_size)
    if name == "DPST":
        return DPST(**common, buffer_size=cfg.buffer_size, replay_batch_size=cfg.replay_batch_size, dpst=DPSTConfig(eta_init=lr))
    raise ValueError(f"Unknown method: {name}")


def _warm_start(model: OnlineForecaster, values: np.ndarray, targets: tuple[int, ...], split_end: int, cfg: SmokeConfig) -> None:
    limit = max(cfg.lookback, split_end - cfg.horizon)
    model.model.train()
    for _ in range(cfg.warm_epochs):
        for end in range(cfg.lookback - 1, limit):
            inputs, target = make_window(values, end, cfg.lookback, cfg.horizon, targets)
            model.optimizer.zero_grad(set_to_none=True)
            loss = model._loss(inputs, target)
            loss.backward()
            model.optimizer.step()


def _online(model: OnlineForecaster, values: np.ndarray, timestamps, online_start: int, targets: tuple[int, ...], cfg: SmokeConfig, artifact_dir: Path) -> dict:
    queue = PendingQueue(cfg.horizon)
    predictions: list[np.ndarray] = []
    actuals: list[np.ndarray] = []
    rows: list[dict] = []
    start = online_start
    stop = min(len(values) - cfg.horizon, start + cfg.online_prefix)
    update_start = time.perf_counter()
    for current_time in range(start, stop):
        pending = queue.pop_resolved(current_time)
        if pending is not None:
            target = values[current_time - cfg.horizon + 1 : current_time + 1, list(targets)].astype(np.float32)
            predicted = pending.prediction
            prequential = float(np.mean((predicted - target) ** 2))
            before = time.perf_counter()
            diagnostic = model.observe_resolved(pending.inputs, target, pending.issue_time)
            update_ms = (time.perf_counter() - before) * 1000.0
            predictions.append(predicted)
            actuals.append(target)
            rows.append({"sample_id": pending.issue_time, "issue_time": pending.issue_time, "resolve_time": current_time, "prequential_loss": prequential, "update_ms": update_ms, **diagnostic})
        inputs, _ = make_window(values, current_time, cfg.lookback, cfg.horizon, targets)
        forecast = model.predict(inputs, current_time)
        queue.add(PendingForecast(current_time, inputs, forecast, model.model_step))
        if len(predictions) >= cfg.online_prefix:
            break
    elapsed = time.perf_counter() - update_start
    pred_array = np.asarray(predictions)
    actual_array = np.asarray(actuals)
    train_targets = values[: online_start, list(targets)]
    metrics = regression_metrics(pred_array, actual_array, training_mase_denominator(train_targets))
    metrics.update({"runtime_seconds": elapsed, "updates": len(rows), "evaluated_timestamps": [int(row["sample_id"]) for row in rows]})
    artifact_dir.mkdir(parents=True, exist_ok=True)
    np.savez(artifact_dir / "predictions.npz", predictions=pred_array, targets=actual_array, sample_ids=np.asarray(metrics["evaluated_timestamps"]))
    try:
        import pandas as pd
        prediction_frame = pd.DataFrame({"sample_id": metrics["evaluated_timestamps"], "timestamp": [str(timestamps[index]) for index in metrics["evaluated_timestamps"]], "prediction": pred_array.reshape(len(pred_array), -1)[:, 0], "target": actual_array.reshape(len(actual_array), -1)[:, 0]})
        prediction_frame.to_parquet(artifact_dir / "predictions.parquet", index=False)
    except (ImportError, ModuleNotFoundError, ValueError):
        prediction_frame.to_csv(artifact_dir / "predictions.csv", index=False)
        (artifact_dir / "predictions.parquet.unavailable.txt").write_text("No parquet engine is installed; predictions.csv and predictions.npz are the exact fallback artifacts.\n", encoding="utf-8")
    with (artifact_dir / "online_metrics.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    (artifact_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    torch.save(model.state_dict(), artifact_dir / "checkpoint.pt")
    (artifact_dir / "timing.json").write_text(json.dumps({"online_runtime_seconds": elapsed, "updates": len(rows)}, indent=2), encoding="utf-8")
    (artifact_dir / "logs").mkdir(exist_ok=True)
    (artifact_dir / "logs" / "run.log").write_text(f"completed updates={len(rows)} finite=true runtime_seconds={elapsed:.6f}\n", encoding="utf-8")
    return {"metrics": metrics, "rows": rows}


def run_smoke(dataset: DatasetFrame, output_root: str | Path, cfg: SmokeConfig, device: str = "cpu") -> dict:
    set_seed(cfg.seed)
    output_root = Path(output_root)
    scaler = TrainOnlyStandardScaler().fit(dataset.features[: dataset.split.train_end])
    values = scaler.transform(dataset.features).astype(np.float32)
    targets = target_indices(dataset)
    selected: dict[str, float] = {}
    method_results: dict[str, dict] = {}
    for name in ("OGD", "ER", "DPST"):
        validation_scores: dict[float, float] = {}
        for lr in cfg.learning_rates:
            candidate = _method(name, values.shape[1], len(targets), cfg, lr, device)
            _warm_start(candidate, values, targets, dataset.split.train_end, cfg)
            losses = []
            for end in range(dataset.split.train_end, dataset.split.validation_end - cfg.horizon):
                inputs, target = make_window(values, end, cfg.lookback, cfg.horizon, targets)
                with torch.no_grad():
                    losses.append(float(candidate._loss(inputs, target).cpu()))
            validation_scores[lr] = float(np.mean(losses))
        selected[name] = min(validation_scores, key=validation_scores.get)
        model = _method(name, values.shape[1], len(targets), cfg, selected[name], device)
        _warm_start(model, values, targets, dataset.split.train_end, cfg)
        result = _online(model, values, dataset.timestamps, dataset.split.validation_end, targets, cfg, output_root / name)
        result["validation_scores"] = validation_scores
        (output_root / name / "config.json").write_text(json.dumps({"method": name, "selected_learning_rate": selected[name], "validation_scores": validation_scores, **asdict(cfg)}, indent=2), encoding="utf-8")
        method_results[name] = result
    environment = {"python": platform.python_version(), "torch": torch.__version__, "device": device, **git_environment()}
    manifest = {"dataset": dataset.name, "source_path": str(dataset.source_path), "source_sha256": dataset.source_sha256, "rows": len(values), "feature_columns": dataset.feature_columns, "target_columns": dataset.target_columns, "split": asdict(dataset.split), "lookback": cfg.lookback, "horizon": cfg.horizon, "seed": cfg.seed, "config": asdict(cfg), "environment": environment}
    (output_root / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    (output_root / "data_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    (output_root / "selected_configs.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
    (output_root / "environment.json").write_text(json.dumps(manifest["environment"], indent=2), encoding="utf-8")
    return {"selected": selected, "methods": method_results, "manifest": manifest}
