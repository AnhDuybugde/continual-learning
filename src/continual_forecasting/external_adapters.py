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


def _load_official_net(source: OfficialSource):
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
        return module.net
    finally:
        sys.path[:] = old_path
        for name in list(sys.modules):
            if name.split(".", 1)[0] in roots:
                del sys.modules[name]
        sys.modules.update(old_modules)


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
    net_class = _load_official_net(source)
    args = type("OfficialArgs", (), {})()
    args.enc_in = input_size
    args.c_out = target_size
    args.pred_len = horizon
    args.seq_len = lookback
    args.individual = False
    official = net_class(args, torch.device(device))
    official = official.to(device)

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
