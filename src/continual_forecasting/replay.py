"""Deterministic reservoir replay buffer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ReplayItem:
    inputs: np.ndarray
    target: np.ndarray
    reference_loss: float
    insert_step: int
    sample_id: int


class ReservoirBuffer:
    def __init__(self, capacity: int, seed: int) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._rng = np.random.default_rng(seed)
        self.items: list[ReplayItem] = []
        self.seen = 0

    def add(self, item: ReplayItem) -> bool:
        self.seen += 1
        if len(self.items) < self.capacity:
            self.items.append(item)
            return True
        slot = int(self._rng.integers(0, self.seen))
        if slot >= self.capacity:
            return False
        self.items[slot] = item
        return True

    def sample(self, batch_size: int) -> list[ReplayItem]:
        if not self.items or batch_size <= 0:
            return []
        count = min(batch_size, len(self.items))
        indices = self._rng.choice(len(self.items), size=count, replace=False)
        return [self.items[int(index)] for index in indices]

    def __len__(self) -> int:
        return len(self.items)
