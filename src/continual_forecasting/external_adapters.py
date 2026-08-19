"""Causal boundaries for official external forecasting implementations.

The adapters deliberately do not reimplement FSNet or OneNet.  They wrap an
official model and optimizer supplied by an isolated factory, while keeping
the benchmark's prediction-before-feedback-update ordering in this package.
"""

from __future__ import annotations

import copy
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

    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, device: str = "cpu") -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = torch.device(device)
        self.model_step = 0

    def predict(self, inputs: np.ndarray, issue_time: int = 0) -> np.ndarray:
        del issue_time
        self.model.eval()
        with torch.no_grad():
            output = self.model(torch.as_tensor(inputs, dtype=torch.float32, device=self.device).unsqueeze(0))
        result = output.detach().cpu().numpy()
        if not np.isfinite(result).all():
            raise FloatingPointError("non-finite external prediction")
        return result

    def observe_resolved(self, inputs: np.ndarray, target: np.ndarray, sample_id: int) -> dict:
        del sample_id
        self.model.train()
        x = torch.as_tensor(inputs, dtype=torch.float32, device=self.device).unsqueeze(0)
        y = torch.as_tensor(target, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.optimizer.zero_grad(set_to_none=True)
        prediction = self.model(x)
        loss = torch.mean((prediction - y) ** 2)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite external loss")
        loss.backward()
        self.optimizer.step()
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
