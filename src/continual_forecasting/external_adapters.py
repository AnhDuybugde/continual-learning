"""Causal boundaries for official external forecasting implementations.

The adapters deliberately do not reimplement FSNet or OneNet.  They wrap an
official model and optimizer supplied by an isolated factory, while keeping
the benchmark's prediction-before-feedback-update ordering in this package.
"""

from __future__ import annotations

import copy
import importlib
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import numpy as np
import torch
from torch import nn


class ExternalModel(Protocol):
    def __call__(self, inputs: torch.Tensor) -> torch.Tensor: ...


@dataclass(frozen=True)
class OfficialSource:
    name: str
    repository: str
    commit: str
    path: Path


FSNET_SOURCE = OfficialSource(
    "FSNet",
    "https://github.com/salesforce/fsnet",
    "c776afc623fa6384a6a559121aacadd2bbea5968",
    Path(".external/fsnet"),
)
ONENET_SOURCE = OfficialSource(
    "OneNet",
    "https://github.com/yfzhang114/OneNet",
    "65eed9d6c878133a4d81d9c381c69e742ad47fd0",
    Path(".external/onenet"),
)


class CausalExternalAdapter:
    """Small benchmark-facing wrapper around an official model.

    ``predict`` has no target argument.  ``observe_resolved`` is the only
    method allowed to consume a target, so the caller can enforce delayed
    feedback with :class:`PendingQueue` exactly as for native methods.
    """

    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, device: str = "cpu", input_transform: Callable[[np.ndarray, int], np.ndarray] | None = None, post_update: Callable[[], None] | None = None) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = torch.device(device)
        self.model_step = 0
        self.input_transform = input_transform
        self.post_update = post_update

    def _inputs(self, inputs: np.ndarray, issue_time: int) -> torch.Tensor:
        if self.input_transform is not None:
            inputs = self.input_transform(inputs, issue_time)
        return torch.as_tensor(inputs, dtype=torch.float32, device=self.device).unsqueeze(0)

    def predict(self, inputs: np.ndarray, issue_time: int = 0) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            output = self.model(self._inputs(inputs, issue_time))
        result = output.detach().cpu().numpy()
        if result.ndim == 3 and result.shape[0] == 1:
            result = result[0]
        if not np.isfinite(result).all():
            raise FloatingPointError("non-finite external prediction")
        return result

    def observe_resolved(self, inputs: np.ndarray, target: np.ndarray, sample_id: int) -> dict:
        self.model.train()
        x = self._inputs(inputs, sample_id)
        y = torch.as_tensor(target, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.optimizer.zero_grad(set_to_none=True)
        prediction = self.model(x)
        loss = torch.mean((prediction - y) ** 2)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite external loss")
        loss.backward()
        self.optimizer.step()
        if self.post_update is not None:
            self.post_update()
        self.model_step += 1
        value = float(loss.detach().cpu())
        return {
            "loss_new": value,
            "loss_replay": 0.0,
            "total_loss": value,
            "eta": float(self.optimizer.param_groups[0]["lr"]),
            "lambda": 0.0,
            "alignment": 0.0,
            "alignment_ema": 0.0,
            "forget_score": 0.0,
            "buffer_size": 0,
            "update_norm": 0.0,
            "finite_status": True,
        }

    def state_dict(self) -> dict:
        return {"model": copy.deepcopy(self.model.state_dict()), "optimizer": copy.deepcopy(self.optimizer.state_dict()), "model_step": self.model_step}

    def load_state_dict(self, state: dict) -> None:
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.model_step = int(state["model_step"])


class OneNetTCNCausalAdapter:
    """Causal reproduction of the official ``onenet_tcn`` online update.

    The official implementation has two OCP controls: a long-term sigmoid
    weight optimized on detached branch predictions and a short-term bias
    produced by the decision MLP.  Both are updated only after a resolved
    target arrives.  For the benchmark's OT target, the time branch's final
    channel is compared with the scalar cross-variable branch output.
    """

    def __init__(self, official: nn.Module, decision: nn.Module, input_size: int, horizon: int, target_channel: int, lr: float, device: str) -> None:
        self.model = official.to(device)
        self.decision = decision.to(device)
        self.input_size = input_size
        self.horizon = horizon
        self.target_channel = target_channel
        self.device = torch.device(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.weight = nn.Parameter(torch.zeros(1, device=self.device))
        self.bias = torch.zeros(1, device=self.device)
        self.weight_optimizer = torch.optim.Adam([self.weight], lr=lr)
        self.bias_optimizer = torch.optim.Adam(self.decision.parameters(), lr=lr)
        self.model_step = 0
        self.timestamps: list = []
        self.source = ONENET_SOURCE

    def set_timestamps(self, timestamps) -> None:
        self.timestamps = list(timestamps)

    def _inputs(self, inputs: np.ndarray, issue_time: int) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.timestamps:
            raise RuntimeError("official adapter timestamps were not configured")
        start = issue_time - len(inputs) + 1
        marks = _timestamp_features(self.timestamps[start : issue_time + 1])
        x = torch.as_tensor(inputs, dtype=torch.float32, device=self.device).unsqueeze(0)
        x_mark = torch.as_tensor(marks, dtype=torch.float32, device=self.device).unsqueeze(0)
        return x, x_mark

    def _branches(self, inputs: np.ndarray, issue_time: int) -> tuple[torch.Tensor, torch.Tensor]:
        x, x_mark = self._inputs(inputs, issue_time)
        _, y1, y2 = self.model.forward_weight(x, x_mark, 1.0, 0.0)
        # Official time branch emits one prediction per input channel.  The
        # benchmark evaluates OT, whose channel is the final ETTh1 column.
        y1 = y1.reshape(1, -1, self.input_size)[:, :, self.target_channel]
        y2 = y2.reshape(1, self.horizon, -1)[:, :, 0]
        return y1, y2

    def predict(self, inputs: np.ndarray, issue_time: int = 0) -> np.ndarray:
        self.model.eval()
        self.decision.eval()
        with torch.no_grad():
            y1, y2 = self._branches(inputs, issue_time)
            gate = torch.sigmoid(self.weight + self.bias).view(1, 1, 1)
            output = gate * y1 + (1.0 - gate) * y2
        result = output.detach().cpu().numpy()
        if not np.isfinite(result).all():
            raise FloatingPointError("non-finite OneNet prediction")
        return result[0, :, None]

    def observe_resolved(self, inputs: np.ndarray, target: np.ndarray, sample_id: int) -> dict:
        self.model.train()
        self.decision.train()
        y = torch.as_tensor(target, dtype=torch.float32, device=self.device).reshape(1, self.horizon, 1)
        y1, y2 = self._branches(inputs, sample_id)
        gate = torch.sigmoid(self.weight + self.bias).view(1, 1, 1)
        prediction = gate * y1 + (1.0 - gate) * y2
        pre_update_loss = torch.mean((prediction - y) ** 2)

        # Official onenet_tcn order: update both forecasters first.
        self.optimizer.zero_grad(set_to_none=True)
        branch_loss = torch.mean((y1 - y) ** 2) + torch.mean((y2 - y) ** 2)
        branch_loss.backward()
        self.optimizer.step()

        # Short-term RL/decision update from current resolved outcome.
        y1_detached, y2_detached = y1.detach(), y2.detach()
        gate_long = torch.sigmoid(self.weight).view(1, 1, 1)
        decision_input = torch.cat([gate_long * y1_detached, (1.0 - gate_long) * y2_detached, y], dim=2).reshape(1, -1)
        bias_prediction = self.decision(decision_input)
        gate_short = torch.sigmoid(self.weight.detach() + bias_prediction).view(1, 1, 1)
        self.bias_optimizer.zero_grad(set_to_none=True)
        bias_loss = torch.mean((gate_short * y1_detached + (1.0 - gate_short) * y2_detached - y) ** 2)
        bias_loss.backward()
        self.bias_optimizer.step()
        self.bias = bias_prediction.detach().reshape(1)

        # Long-term OCP/EGD weight update on detached branch forecasts.
        self.weight_optimizer.zero_grad(set_to_none=True)
        gate_long = torch.sigmoid(self.weight).view(1, 1, 1)
        weight_loss = torch.mean((gate_long * y1_detached + (1.0 - gate_long) * y2_detached - y) ** 2)
        weight_loss.backward()
        self.weight_optimizer.step()
        self.model_step += 1
        values = (float(pre_update_loss.detach().cpu()), float(branch_loss.detach().cpu()), float(bias_loss.detach().cpu()), float(weight_loss.detach().cpu()))
        if not np.isfinite(values).all():
            raise FloatingPointError("non-finite OneNet loss")
        return {
            "loss_new": values[0], "loss_replay": 0.0, "total_loss": values[0],
            "branch_loss": values[1], "decision_loss": values[2], "weight_loss": values[3],
            "eta": float(self.optimizer.param_groups[0]["lr"]),
            "lambda": float(torch.sigmoid(self.weight).detach().cpu()),
            "alignment": float(torch.sigmoid(self.weight).detach().cpu()),
            "alignment_ema": float(torch.sigmoid(self.bias).detach().cpu()),
            "forget_score": 0.0, "buffer_size": 0, "update_norm": 0.0,
            "finite_status": True,
        }

    def state_dict(self) -> dict:
        return {"model": copy.deepcopy(self.model.state_dict()), "decision": copy.deepcopy(self.decision.state_dict()), "optimizer": copy.deepcopy(self.optimizer.state_dict()), "weight": self.weight.detach().clone(), "bias": self.bias.detach().clone(), "weight_optimizer": copy.deepcopy(self.weight_optimizer.state_dict()), "bias_optimizer": copy.deepcopy(self.bias_optimizer.state_dict()), "model_step": self.model_step}

    def load_state_dict(self, state: dict) -> None:
        self.model.load_state_dict(state["model"])
        self.decision.load_state_dict(state["decision"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.weight.data.copy_(state["weight"])
        self.bias = state["bias"].to(self.device)
        self.weight_optimizer.load_state_dict(state["weight_optimizer"])
        self.bias_optimizer.load_state_dict(state["bias_optimizer"])
        self.model_step = int(state["model_step"])


def official_source_status(source: OfficialSource, root: str | Path = ".") -> dict:
    path = Path(root) / source.path
    return {"name": source.name, "repository": source.repository, "commit": source.commit, "path": str(path), "available": path.is_dir()}


def probe_official_import(source: OfficialSource, root: str | Path = ".") -> dict:
    """Check that the pinned source can be imported without mutating sys.path.

    The official projects use a top-level ``models`` package, so importing
    both in one interpreter is intentionally avoided.  This probe reports the
    exact failure and leaves the shared benchmark environment untouched.
    """
    path = Path(root) / source.path
    if not path.is_dir():
        return {**official_source_status(source, root), "importable": False, "error": "official source directory missing"}
    entry = path / ("exp/exp_fsnet.py" if source.name == "FSNet" else "exp/exp_onenet_tcn.py")
    if not entry.exists():
        return {**official_source_status(source, root), "importable": False, "error": f"entry point missing: {entry}"}
    spec = importlib.util.spec_from_file_location(f"_official_{source.name.lower()}_entry", entry)
    if spec is None or spec.loader is None:
        return {**official_source_status(source, root), "importable": False, "error": "could not create import spec"}
    module = importlib.util.module_from_spec(spec)
    old_path = list(sys.path)
    namespace_roots = ("models", "exp", "utils", "data", "layers")
    old_modules = {name: value for name, value in sys.modules.items() if name.split(".", 1)[0] in namespace_roots}
    try:
        for name in list(sys.modules):
            if name.split(".", 1)[0] in namespace_roots:
                del sys.modules[name]
        sys.path.insert(0, str(path))
        spec.loader.exec_module(module)
    except Exception as exc:  # report exact provenance/runtime blocker
        return {**official_source_status(source, root), "importable": False, "error": f"{type(exc).__name__}: {exc}"}
    finally:
        sys.path[:] = old_path
        for name in list(sys.modules):
            if name.split(".", 1)[0] in namespace_roots:
                del sys.modules[name]
        sys.modules.update(old_modules)
    return {**official_source_status(source, root), "importable": True, "entry_point": str(entry)}


def make_contract_adapter(factory: Callable[[], tuple[nn.Module, torch.optim.Optimizer]], device: str = "cpu") -> CausalExternalAdapter:
    """Build an adapter from an official factory without changing its model."""
    model, optimizer = factory()
    return CausalExternalAdapter(model, optimizer, device)


def _load_official_module(source: OfficialSource):
    """Load one official ``net`` class while isolating legacy top-level packages."""
    path = Path(source.path)
    entry = path / ("exp/exp_fsnet.py" if source.name == "FSNet" else "exp/exp_onenet_tcn.py")
    spec = importlib.util.spec_from_file_location(f"_official_{source.name.lower()}_net", entry)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load official {source.name} entry point: {entry}")
    old_path = list(sys.path)
    roots = ("models", "exp", "utils", "data", "layers")
    old_modules = {name: value for name, value in sys.modules.items() if name.split(".", 1)[0] in roots}
    try:
        for name in list(sys.modules):
            if name.split(".", 1)[0] in roots:
                del sys.modules[name]
        sys.path.insert(0, str(path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = old_path
        for name in list(sys.modules):
            if name.split(".", 1)[0] in roots:
                del sys.modules[name]
        sys.modules.update(old_modules)


def _load_official_net(source: OfficialSource):
    return _load_official_module(source).net


def _timestamp_features(timestamps) -> np.ndarray:
    """Match the official timeenc=2 seven-column feature order."""
    values = []
    for stamp in timestamps:
        values.append([stamp.minute, stamp.hour, stamp.dayofweek, stamp.day, stamp.dayofyear, stamp.month, stamp.isocalendar().week])
    return np.asarray(values, dtype=np.float32)


def build_official_adapter(name: str, input_size: int, target_size: int, horizon: int, lookback: int, lr: float, seed: int, device: str = "cpu") -> CausalExternalAdapter:
    """Instantiate a pinned official model behind the causal contract.

    The caller must call ``set_timestamps`` before prediction.  This keeps
    timestamp-derived covariates aligned with the shared dataset stream.
    """
    source = FSNET_SOURCE if name == "FSNet" else ONENET_SOURCE
    torch.manual_seed(seed)
    official_module = _load_official_module(source)
    net_class = official_module.net
    args = type("OfficialArgs", (), {})()
    args.enc_in = input_size
    args.c_out = target_size
    args.pred_len = horizon
    args.seq_len = lookback
    args.individual = False
    official = net_class(args, torch.device(device))
    official = official.to(device)

    if name == "OneNet":
        decision = official_module.MLP(n_inputs=3 * horizon * target_size, n_outputs=1, mlp_width=32, mlp_depth=3, mlp_dropout=0.1, act=torch.nn.Tanh())
        return OneNetTCNCausalAdapter(official, decision, input_size, horizon, input_size - 1, lr, device)

    class Wrapped(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            if name == "FSNet":
                output = self.model(x)
            else:
                raw, marks = x[..., :input_size], x[..., input_size:]
                _, y1, y2 = self.model.forward_weight(raw, marks, 0.5, 0.5)
                y1 = y1.reshape(x.shape[0], horizon, input_size)[..., -target_size:]
                y2 = y2.reshape(x.shape[0], horizon, target_size)
                output = 0.5 * y1 + 0.5 * y2
            return output.reshape(x.shape[0], horizon, target_size)

    wrapped = Wrapped(official)
    optimizer = torch.optim.AdamW(wrapped.parameters(), lr=lr)
    timestamps_holder: list = []

    def transform(inputs: np.ndarray, issue_time: int) -> np.ndarray:
        if not timestamps_holder:
            raise RuntimeError("official adapter timestamps were not configured")
        start = issue_time - len(inputs) + 1
        marks = _timestamp_features(timestamps_holder[start : issue_time + 1])
        return np.concatenate([inputs, marks], axis=1)

    post_update = getattr(official, "store_grad", None) if name == "FSNet" else None
    adapter = CausalExternalAdapter(wrapped, optimizer, device, transform, post_update)
    adapter.set_timestamps = lambda timestamps: timestamps_holder.extend(list(timestamps))
    adapter.source = source
    return adapter
