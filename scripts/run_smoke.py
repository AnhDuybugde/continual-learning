"""Run the bounded ETTh1 smoke benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from continual_forecasting.data import load_dataset
from continual_forecasting.runner import SmokeConfig, run_smoke


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/all_six_datasets/ETT-small/ETTh1.csv")
    parser.add_argument("--output", default="artifacts/smoke_etth1")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--prefix", type=int, default=2000)
    args = parser.parse_args()
    dataset = load_dataset(args.data)
    result = run_smoke(dataset, args.output, SmokeConfig(online_prefix=args.prefix), args.device)
    print(json.dumps({"selected": result["selected"], "metrics": {name: value["metrics"] for name, value in result["methods"].items()}}, indent=2))


if __name__ == "__main__":
    main()
