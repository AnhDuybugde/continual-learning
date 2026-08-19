from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from continual_forecasting.data import load_dataset
from continual_forecasting.runner import SmokeConfig, run_smoke


class SmokeExecutionTests(unittest.TestCase):
    def test_short_etth1_smoke_path(self) -> None:
        dataset = load_dataset("data/all_six_datasets/ETT-small/ETTh1.csv")
        result = run_smoke(dataset, "artifacts/smoke_etth1", SmokeConfig(online_prefix=2000, channels=4), "cpu")
        self.assertEqual(set(result["methods"]), {"OGD", "ER", "DPST", "DERPP"})
        self.assertTrue(all(value["metrics"]["count"] >= 1999 for value in result["methods"].values()))


if __name__ == "__main__":
    unittest.main()
