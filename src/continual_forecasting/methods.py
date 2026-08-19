"""Shared-backbone OGD, ER, and DPST-Core implementations."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .backbone import CausalTCN
from .replay import ReplayItem, ReservoirBuffer


@dataclass(frozen=True)
class DPSTConfig:
    eta_init: float = 1e-3
    eta_min: float = 1e-5
    eta_max: float = 1e-2
    lambda_init: float = 0.5
    lambda_max: float = 1.0
    beta_eta: float = 0.01
    beta_lambda: float = 0.01
    alignment_ema_decay: float = 0.9
    forget_budget: float = 0.10
    forget_score_clip: float = 5.0
    controller_eps: float = 1e-8


def _finite(value: float) -> None:
    if not math.isfinite(value):
        raise FloatingPointError(f"non-finite loss: {value}")


class OnlineForecaster:
    def __init__(self, input_size: int, target_size: int, horizon: int, lr: float, seed: int, device: str = "cpu", channels: int = 32) -> None:
        torch.manual_seed(seed)
        self.device = torch.device(device)
        self.model = CausalTCN(input_size, target_size, horizon, channels).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.horizon = horizon
        self.target_size = target_size
        self.model_step = 0
        self.rng = np.random.default_rng(seed)

    def predict(self, inputs: np.ndarray, issue_time: int = 0) -> np.ndarray:
        del issue_time
        self.model.eval()
        with torch.no_grad():
            tensor = torch.as_tensor(inputs, dtype=torch.float32, device=self.device).unsqueeze(0)
            output = self.model(tensor)[0]
        result = output.detach().cpu().numpy()
        if not np.isfinite(result).all():
            raise FloatingPointError("non-finite prediction")
        return result

    def _loss(self, inputs: np.ndarray, target: np.ndarray) -> torch.Tensor:
        x = torch.as_tensor(inputs, dtype=torch.float32, device=self.device).unsqueeze(0)
        y = torch.as_tensor(target, dtype=torch.float32, device=self.device).unsqueeze(0)
        return torch.mean((self.model(x) - y) ** 2)

    def observe_resolved(self, inputs: np.ndarray, target: np.ndarray, sample_id: int) -> dict:
        del sample_id
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        loss = self._loss(inputs, target)
        loss.backward()
        self.optimizer.step()
        value = float(loss.detach().cpu())
        _finite(value)
        self.model_step += 1
        return self._diagnostic(value, value, 0.0, 0.0, 0.0, 0.0)

    def _diagnostic(self, loss_new: float, loss_replay: float, eta: float, lam: float, alignment: float, forget: float) -> dict:
        return {
            "loss_new": loss_new,
            "loss_replay": loss_replay,
            "eta": eta,
            "lambda": lam,
            "alignment": alignment,
            "alignment_ema": alignment,
            "forget_score": forget,
            "buffer_size": 0,
            "update_norm": 0.0,
            "finite_status": True,
        }

    def state_dict(self) -> dict:
        return {"model": copy.deepcopy(self.model.state_dict()), "optimizer": copy.deepcopy(self.optimizer.state_dict()), "model_step": self.model_step}

    def load_state_dict(self, state: dict) -> None:
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.model_step = state["model_step"]


class OGD(OnlineForecaster):
    """One gradient update on the current resolved sample only."""

    pass


class ER(OnlineForecaster):
    def __init__(self, *args, buffer_size: int = 500, replay_batch_size: int = 8, replay_weight: float = 0.5, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.buffer = ReservoirBuffer(buffer_size, kwargs.get("seed", 0))
        self.replay_batch_size = replay_batch_size
        self.replay_weight = replay_weight

    def observe_resolved(self, inputs: np.ndarray, target: np.ndarray, sample_id: int) -> dict:
        self.model.train()
        replay = self.buffer.sample(self.replay_batch_size)
        self.optimizer.zero_grad(set_to_none=True)
        new_loss = self._loss(inputs, target)
        replay_loss = torch.tensor(0.0, device=self.device)
        if replay:
            replay_loss = torch.stack([self._loss(item.inputs, item.target) for item in replay]).mean()
        total = new_loss + self.replay_weight * replay_loss
        total.backward()
        before = [p.detach().clone() for p in self.model.parameters()]
        self.optimizer.step()
        update_norm = float(torch.sqrt(sum(torch.sum((after.detach() - prior) ** 2) for after, prior in zip(self.model.parameters(), before))).cpu())
        new_value = float(new_loss.detach().cpu())
        replay_value = float(replay_loss.detach().cpu())
        _finite(new_value)
        _finite(replay_value)
        self.model_step += 1
        reference = float(self._loss(inputs, target).detach().cpu())
        self.buffer.add(ReplayItem(inputs.copy(), target.copy(), reference, self.model_step, sample_id))
        return {"loss_new": new_value, "loss_replay": replay_value, "eta": self.optimizer.param_groups[0]["lr"], "lambda": self.replay_weight, "alignment": 0.0, "alignment_ema": 0.0, "forget_score": 0.0, "buffer_size": len(self.buffer), "update_norm": update_norm, "finite_status": True}

    def state_dict(self) -> dict:
        state = super().state_dict()
        state.update({"buffer": self.buffer, "replay_batch_size": self.replay_batch_size, "replay_weight": self.replay_weight})
        return state


class DPST(ER):
    """DPST-Core: bounded adaptive plasticity and adaptive replay weight."""

    def __init__(self, *args, dpst: DPSTConfig | None = None, **kwargs) -> None:
        config = dpst or DPSTConfig(eta_init=kwargs.get("lr", 1e-3))
        kwargs["lr"] = config.eta_init
        super().__init__(*args, replay_weight=config.lambda_init, **kwargs)
        self.config = config
        self.log_eta = math.log(config.eta_init)
        ratio = min(max(config.lambda_init / config.lambda_max, config.controller_eps), 1.0 - config.controller_eps)
        self.replay_logit = math.log(ratio / (1.0 - ratio))
        self.alignment_ema = 0.0
        self.p_prev: list[torch.Tensor] | None = None
        self.controller_updates = 0

    def observe_resolved(self, inputs: np.ndarray, target: np.ndarray, sample_id: int) -> dict:
        self.model.train()
        replay = self.buffer.sample(self.replay_batch_size)
        eps = self.config.controller_eps
        alignment = 0.0
        if self.p_prev is not None:
            self.optimizer.zero_grad(set_to_none=True)
            probe = self._loss(inputs, target)
            gradients = torch.autograd.grad(probe, tuple(self.model.parameters()), allow_unused=True)
            g_norm = torch.sqrt(sum(torch.sum((g if g is not None else torch.zeros_like(p)) ** 2) for g, p in zip(gradients, self.model.parameters())))
            p_norm = torch.sqrt(sum(torch.sum(p ** 2) for p in self.p_prev))
            dot = sum(torch.sum((g if g is not None else torch.zeros_like(p)) * old) for g, old, p in zip(gradients, self.p_prev, self.model.parameters()))
            alignment = float(torch.clamp(dot / (g_norm * p_norm + eps), -1.0, 1.0).detach().cpu())
        self.alignment_ema = self.config.alignment_ema_decay * self.alignment_ema + (1 - self.config.alignment_ema_decay) * alignment
        self.log_eta = float(np.clip(self.log_eta + self.config.beta_eta * self.alignment_ema, math.log(self.config.eta_min), math.log(self.config.eta_max)))
        eta = math.exp(self.log_eta) if self.p_prev is not None else self.config.eta_init
        replay_loss_values: list[float] = []
        forget_values: list[float] = []
        for item in replay:
            current = float(self._loss(item.inputs, item.target).detach().cpu())
            replay_loss_values.append(current)
            forget_values.append(max(0.0, (current - item.reference_loss) / (item.reference_loss + eps)))
        forget = float(np.clip(np.mean(forget_values) if forget_values else 0.0, 0.0, self.config.forget_score_clip))
        if replay:
            self.replay_logit = float(np.clip(self.replay_logit + self.config.beta_lambda * (forget - self.config.forget_budget), -20.0, 20.0))
            lam = self.config.lambda_max / (1.0 + math.exp(-self.replay_logit))
        else:
            lam = 0.0
        for group in self.optimizer.param_groups:
            group["lr"] = eta
        self.optimizer.zero_grad(set_to_none=True)
        new_loss = self._loss(inputs, target)
        replay_loss = torch.stack([self._loss(item.inputs, item.target) for item in replay]).mean() if replay else torch.tensor(0.0, device=self.device)
        (new_loss + lam * replay_loss).backward()
        before = [p.detach().clone() for p in self.model.parameters()]
        self.optimizer.step()
        self.p_prev = [(prior - after.detach()) / max(eta, eps) for prior, after in zip(before, self.model.parameters())]
        update_norm = float(torch.sqrt(sum(torch.sum((after.detach() - prior) ** 2) for after, prior in zip(self.model.parameters(), before))).cpu())
        new_value = float(new_loss.detach().cpu())
        replay_value = float(replay_loss.detach().cpu())
        _finite(new_value)
        _finite(replay_value)
        self.model_step += 1
        reference = float(self._loss(inputs, target).detach().cpu())
        self.buffer.add(ReplayItem(inputs.copy(), target.copy(), reference, self.model_step, sample_id))
        self.controller_updates += 1
        return {"loss_new": new_value, "loss_replay": replay_value, "eta": eta, "lambda": lam, "alignment": alignment, "alignment_ema": self.alignment_ema, "forget_score": forget, "buffer_size": len(self.buffer), "update_norm": update_norm, "finite_status": True}

    def state_dict(self) -> dict:
        state = super().state_dict()
        state.update({"config": self.config, "log_eta": self.log_eta, "replay_logit": self.replay_logit, "alignment_ema": self.alignment_ema, "p_prev": self.p_prev, "controller_updates": self.controller_updates})
        return state

    def load_state_dict(self, state: dict) -> None:
        super().load_state_dict(state)
        self.log_eta = state["log_eta"]
        self.replay_logit = state["replay_logit"]
        self.alignment_ema = state["alignment_ema"]
        self.p_prev = state["p_prev"]
        self.controller_updates = state["controller_updates"]
