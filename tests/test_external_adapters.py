import unittest

import numpy as np
import torch
from torch import nn

from continual_forecasting.external_adapters import (
    CausalExternalAdapter,
    FSNET_SOURCE,
    ONENET_SOURCE,
    official_source_status,
    probe_official_import,
)


class _OfficialShapedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x[:, -1:, :])


class ExternalAdapterTests(unittest.TestCase):
    def test_prediction_is_before_update_and_target_is_only_consumed_on_resolve(self) -> None:
        torch.manual_seed(3)
        model = _OfficialShapedModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        adapter = CausalExternalAdapter(model, optimizer)
        inputs = np.ones((4, 2), dtype=np.float32)
        target = np.zeros((1, 1), dtype=np.float32)
        before = adapter.predict(inputs)
        self.assertEqual(adapter.model_step, 0)
        diagnostic = adapter.observe_resolved(inputs, target, sample_id=9)
        after = adapter.predict(inputs)
        self.assertEqual(adapter.model_step, 1)
        self.assertNotEqual(float(before[0, 0]), float(after[0, 0]))
        self.assertTrue(diagnostic["finite_status"])

    def test_checkpoint_restore_is_exact_for_adapter_contract(self) -> None:
        torch.manual_seed(4)
        model = _OfficialShapedModel()
        adapter = CausalExternalAdapter(model, torch.optim.AdamW(model.parameters(), lr=1e-3))
        inputs = np.ones((4, 2), dtype=np.float32)
        adapter.observe_resolved(inputs, np.zeros((1, 1), dtype=np.float32), 0)
        restored_model = _OfficialShapedModel()
        restored = CausalExternalAdapter(restored_model, torch.optim.AdamW(restored_model.parameters(), lr=1e-3))
        restored.load_state_dict(adapter.state_dict())
        np.testing.assert_allclose(adapter.predict(inputs), restored.predict(inputs), rtol=0, atol=0)

    def test_pinned_sources_and_probe_are_explicit(self) -> None:
        for source in (FSNET_SOURCE, ONENET_SOURCE):
            status = official_source_status(source)
            self.assertTrue(status["available"])
            result = probe_official_import(source)
            self.assertEqual(result["commit"], source.commit)
            self.assertIn("importable", result)


if __name__ == "__main__":
    unittest.main()
