"""Run the predeclared DER++ three-seed H=1 comparison."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from continual_forecasting.data import load_dataset
from continual_forecasting.runner import SmokeConfig, run_smoke


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/all_six_datasets/ETT-small/ETTh1.csv")
    parser.add_argument("--output", default="artifacts/dpst_milestone/H1/DERPP")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prefix", type=int, default=2000)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = parser.parse_args()

    dataset = load_dataset(args.data)
    results = {}
    for seed in args.seeds:
        cfg = SmokeConfig(seed=seed, horizon=1, online_prefix=args.prefix)
        artifact = Path(args.output) / f"seed_{seed}"
        result = run_smoke(dataset, artifact, cfg, args.device, ("DERPP",))
        results[f"seed_{seed}"] = {
            "metrics": result["methods"]["DERPP"]["metrics"],
            "validation_scores": result["methods"]["DERPP"]["validation_scores"],
            "artifact": str(artifact / "DERPP"),
        }
    output = Path(args.output) / "results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
