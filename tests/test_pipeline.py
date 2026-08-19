from __future__ import annotations

import sys
import unittest
import copy
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from continual_forecasting.data import ChronologicalSplit, TrainOnlyStandardScaler
from continual_forecasting.metrics import regression_metrics
from continual_forecasting.queue import PendingForecast, PendingQueue
from continual_forecasting.replay import ReplayItem, ReservoirBuffer
from continual_forecasting.methods import DERPP, DPST, DPSTConfig, ER


class PipelineTests(unittest.TestCase):
    def test_split_is_chronological_and_exact_ratio_policy(self) -> None:
        split = ChronologicalSplit.from_length(1000)
        self.assertEqual((split.train_end, split.validation_end, split.online_end), (200, 250, 1000))

    def test_scaler_uses_train_only(self) -> None:
        train = np.array([[0.0], [2.0]])
        all_values = np.vstack([train, [[100.0]]])
        scaler = TrainOnlyStandardScaler().fit(train)
        self.assertAlmostEqual(float(scaler.mean_[0]), 1.0)
        self.assertAlmostEqual(float(scaler.transform(all_values)[-1, 0]), 99.0)

    def test_pending_queue_releases_only_after_horizon(self) -> None:
        queue = PendingQueue(3)
        item = PendingForecast(10, np.zeros((2, 1)), np.zeros((3, 1)), 0)
        queue.add(item)
        self.assertIsNone(queue.pop_resolved(12))
        self.assertIs(queue.pop_resolved(13), item)

    def test_feedback_release_for_required_horizons(self) -> None:
        for horizon in (1, 24, 48):
            queue = PendingQueue(horizon)
            item = PendingForecast(100, np.zeros((2, 1)), np.zeros((horizon, 1)), 0)
            queue.add(item)
            self.assertIsNone(queue.pop_resolved(100 + horizon - 1))
            self.assertIs(queue.pop_resolved(100 + horizon), item)

    def test_reservoir_capacity_and_determinism(self) -> None:
        def sample() -> list[int]:
            buffer = ReservoirBuffer(3, 11)
            for index in range(20):
                buffer.add(ReplayItem(np.zeros((1, 1)), np.zeros((1, 1)), 1.0, index, index))
            return [item.sample_id for item in buffer.items]
        self.assertLessEqual(len(sample()), 3)
        self.assertEqual(sample(), sample())

    def test_metric_recomputation(self) -> None:
        metrics = regression_metrics(np.array([[1.0], [3.0]]), np.array([[0.0], [1.0]]), 1.0)
        self.assertEqual(metrics["count"], 2)
        self.assertAlmostEqual(metrics["mae"], 1.5)
        self.assertAlmostEqual(metrics["mase"], 1.5)

    def test_checkpoint_restore_prediction(self) -> None:
        kwargs = dict(input_size=2, target_size=1, horizon=1, lr=1e-3, seed=5, device="cpu", channels=4, buffer_size=4, replay_batch_size=2)
        model = ER(**kwargs)
        inputs = np.zeros((4, 2), dtype=np.float32)
        target = np.ones((1, 1), dtype=np.float32)
        model.observe_resolved(inputs, target, 0)
        restored = ER(**kwargs)
        restored.load_state_dict(model.state_dict())
        np.testing.assert_allclose(model.predict(inputs), restored.predict(inputs), atol=1e-6)

    def test_dpst_zero_controller_matches_er_trajectory(self) -> None:
        kwargs = dict(input_size=2, target_size=1, horizon=1, lr=1e-3, seed=9, device="cpu", channels=4, buffer_size=4, replay_batch_size=2)
        source = ER(**kwargs)
        warm_inputs = np.full((4, 2), 0.25, dtype=np.float32)
        warm_target = np.ones((1, 1), dtype=np.float32)
        source.observe_resolved(warm_inputs, warm_target, 0)
        er = ER(**kwargs)
        dpst = DPST(**kwargs, dpst=DPSTConfig(eta_init=1e-3, lambda_init=0.5, beta_eta=0.0, beta_lambda=0.0))
        checkpoint = source.state_dict()
        er.load_state_dict(copy.deepcopy(checkpoint))
        dpst.load_state_dict(copy.deepcopy(checkpoint) | {"config": dpst.config, "log_eta": np.log(1e-3), "replay_logit": 0.0, "alignment_ema": 0.0, "p_prev": None, "controller_updates": 0})
        for sample_id in range(1, 5):
            inputs = np.full((4, 2), sample_id / 10.0, dtype=np.float32)
            target = np.full((1, 1), 1.0 + sample_id / 10.0, dtype=np.float32)
            er_result = er.observe_resolved(inputs, target, sample_id)
            dpst_result = dpst.observe_resolved(inputs, target, sample_id)
            for key in ("loss_new", "loss_replay", "total_loss", "eta", "lambda"):
                self.assertAlmostEqual(dpst_result[key], er_result[key], delta=1e-8)
            for left, right in zip(er.model.parameters(), dpst.model.parameters()):
                torch.testing.assert_close(left, right, rtol=1e-6, atol=1e-8)
            self.assertEqual(er.buffer.seen, dpst.buffer.seen)
            self.assertEqual([item.sample_id for item in er.buffer.items], [item.sample_id for item in dpst.buffer.items])
            torch.testing.assert_close(torch.tensor(er.optimizer.state_dict()["state"][0]["exp_avg"]), torch.tensor(dpst.optimizer.state_dict()["state"][0]["exp_avg"]), rtol=1e-6, atol=1e-8)

    def test_derpp_stores_replay_prediction(self) -> None:
        method = DERPP(input_size=2, target_size=1, horizon=1, lr=1e-3, seed=3, device="cpu", channels=4, buffer_size=4, replay_batch_size=1)
        method.observe_resolved(np.zeros((4, 2), dtype=np.float32), np.ones((1, 1), dtype=np.float32), 0)
        self.assertIsNotNone(method.buffer.items[0].reference_prediction)


if __name__ == "__main__":
    unittest.main()
