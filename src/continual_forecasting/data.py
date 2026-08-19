"""Data loading, chronological splits, and train-only scaling."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ChronologicalSplit:
    train_end: int
    validation_end: int
    online_end: int

    @classmethod
    def from_length(cls, length: int) -> "ChronologicalSplit":
        train_end = int(length * 0.20)
        validation_end = train_end + int(length * 0.05)
        return cls(train_end, validation_end, length)


@dataclass(frozen=True)
class DatasetFrame:
    name: str
    timestamps: pd.DatetimeIndex
    features: np.ndarray
    feature_columns: tuple[str, ...]
    target_columns: tuple[str, ...]
    split: ChronologicalSplit
    source_path: Path
    source_sha256: str


class TrainOnlyStandardScaler:
    """A small immutable-after-fit standard scaler for causal experiments."""

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, values: np.ndarray) -> "TrainOnlyStandardScaler":
        values = np.asarray(values, dtype=np.float64)
        self.mean_ = values.mean(axis=0)
        scale = values.std(axis=0)
        self.scale_ = np.where(scale > 0.0, scale, 1.0)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Scaler must be fitted on the training segment first")
        return (np.asarray(values, dtype=np.float64) - self.mean_) / self.scale_

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Scaler must be fitted first")
        return np.asarray(values, dtype=np.float64) * self.scale_ + self.mean_


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_dataset(path: str | Path, target_columns: Sequence[str] = ("OT",)) -> DatasetFrame:
    source_path = Path(path)
    frame = pd.read_csv(source_path)
    if "date" not in frame.columns:
        raise ValueError(f"Expected a date column in {source_path}")
    timestamps = pd.to_datetime(frame.pop("date"), errors="raise")
    if not timestamps.is_monotonic_increasing or timestamps.duplicated().any():
        raise ValueError("Timestamps must be sorted and unique")
    if len(timestamps) > 2 and not (timestamps.diff().dropna() == timestamps.diff().dropna().iloc[0]).all():
        raise ValueError("Timestamps must have a regular sampling interval")
    numeric = frame.apply(pd.to_numeric, errors="raise")
    targets = tuple(target_columns)
    missing = sorted(set(targets) - set(numeric.columns))
    if missing:
        raise ValueError(f"Target columns not found: {missing}")
    if numeric.isna().any().any():
        raise ValueError("Raw dataset contains missing numeric values")
    return DatasetFrame(
        name=source_path.stem,
        timestamps=pd.DatetimeIndex(timestamps),
        features=numeric.to_numpy(dtype=np.float64),
        feature_columns=tuple(numeric.columns),
        target_columns=targets,
        split=ChronologicalSplit.from_length(len(numeric)),
        source_path=source_path,
        source_sha256=file_sha256(source_path),
    )


def target_indices(dataset: DatasetFrame) -> tuple[int, ...]:
    return tuple(dataset.feature_columns.index(column) for column in dataset.target_columns)


def make_window(values: np.ndarray, end: int, lookback: int, horizon: int, targets: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    if end - lookback + 1 < 0 or end + horizon >= len(values):
        raise IndexError("Window is outside the available chronological data")
    inputs = values[end - lookback + 1 : end + 1]
    target = values[end + 1 : end + horizon + 1, list(targets)]
    return inputs.astype(np.float32), target.astype(np.float32)
