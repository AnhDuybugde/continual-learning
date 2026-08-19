"""Causal pending forecast queue."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass
class PendingForecast:
    issue_time: int
    inputs: np.ndarray
    prediction: np.ndarray
    model_step: int

    @property
    def resolve_time(self) -> int:
        return self.issue_time + self.prediction.shape[0]


class PendingQueue:
    def __init__(self, horizon: int) -> None:
        if horizon < 1:
            raise ValueError("horizon must be positive")
        self.horizon = horizon
        self._items: dict[int, PendingForecast] = {}

    def add(self, item: PendingForecast) -> None:
        if item.prediction.shape[0] != self.horizon:
            raise ValueError("prediction horizon does not match queue horizon")
        if item.issue_time in self._items:
            raise ValueError("duplicate issue time")
        self._items[item.issue_time] = item

    def pop_resolved(self, current_time: int) -> PendingForecast | None:
        return self._items.pop(current_time - self.horizon, None)

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[PendingForecast]:
        return iter(self._items.values())

    def state_dict(self) -> list[dict]:
        return [{"issue_time": item.issue_time, "inputs": item.inputs, "prediction": item.prediction, "model_step": item.model_step} for item in self._items.values()]
