"""Shared lightweight causal TCN backbone."""

from __future__ import annotations

import torch
from torch import nn


class CausalTCN(nn.Module):
    def __init__(self, input_size: int, target_size: int, horizon: int, channels: int = 32) -> None:
        super().__init__()
        self.horizon = horizon
        self.target_size = target_size
        self.network = nn.Sequential(
            nn.Conv1d(input_size, channels, kernel_size=3, padding=2),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=2),
            nn.ReLU(),
        )
        self.head = nn.Linear(channels, horizon * target_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        encoded = self.network(inputs.transpose(1, 2))[:, :, -1]
        return self.head(encoded).view(-1, self.horizon, self.target_size)
