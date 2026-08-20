"""Run the predeclared fixed-replay-weight comparison on ETTh1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from continual_forecasting.data import load_dataset
from continual_forecasting.runner import SmokeConfig, run_smoke


FIXED_LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def run_fixed_lambda(
    dataset_path: str,
    output_root: Path,
    device: str,
    prefix: int,
    seeds: list[int],
) -> dict:
    dataset = load_dataset(dataset_path)
    results: dict[str, dict] = {}
    for seed in seeds:
        for replay_weight in FIXED_LAMBDAS:
            cfg = SmokeConfig(
                seed=seed,
                horizon=1,
                online_prefix=prefix,
                dpst_beta_eta=0.0,
                dpst_beta_lambda=0.0,
                dpst_lambda_init=replay_weight,
            )
            artifact = output_root / "H1" / f"seed_{seed}" / f"DPST_fixed_lambda_{replay_weight:g}"
            result = run_smoke(dataset, artifact, cfg, device, ("DPST",))
            method_result = result["methods"]["DPST"]
            results[f"H1/DPST_fixed_lambda_{replay_weight:g}/seed_{seed}"] = {
                "method": "DPST",
                "config": {
                    "seed": seed,
                    "lambda": replay_weight,
                    "beta_eta": 0.0,
                    "beta_lambda": 0.0,
                },
                "metrics": method_result["metrics"],
                "validation_scores": method_result["validation_scores"],
                "artifact": str(artifact),
            }
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/all_six_datasets/ETT-small/ETTh1.csv")
    parser.add_argument("--output", default="artifacts/dpst_milestone/fixed_lambda")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prefix", type=int, default=2000)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    results = run_fixed_lambda(args.data, output, args.device, args.prefix, args.seeds)
    (output / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
