"""Prequential metrics and recomputation helpers."""

from __future__ import annotations

import numpy as np


def regression_metrics(predictions: np.ndarray, targets: np.ndarray, mase_denominator: float) -> dict[str, float]:
    errors = np.asarray(predictions, dtype=np.float64) - np.asarray(targets, dtype=np.float64)
    absolute = np.abs(errors)
    squared = errors**2
    denominator = max(float(mase_denominator), np.finfo(float).eps)
    return {"mae": float(absolute.mean()), "mse": float(squared.mean()), "mase": float(absolute.mean() / denominator), "count": int(errors.shape[0])}


def training_mase_denominator(train_targets: np.ndarray) -> float:
    values = np.asarray(train_targets, dtype=np.float64).reshape(-1)
    if len(values) < 2:
        return 1.0
    return float(np.mean(np.abs(np.diff(values)))) or 1.0
