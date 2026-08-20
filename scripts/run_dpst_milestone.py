"""Run the bounded DPST ablation and delayed-feedback smoke milestone."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from continual_forecasting.data import load_dataset
from continual_forecasting.runner import SmokeConfig, run_smoke


ABLATIONS = {
    "ER": (0.0, 0.0),
    "DPST-eta": (0.01, 0.0),
    "DPST-lambda": (0.0, 0.01),
    "DPST-full": (0.01, 0.01),
}


def run_ablation(dataset_path: str, output_root: Path, device: str, prefix: int, seeds: list[int]) -> dict:
    dataset = load_dataset(dataset_path)
    results = {}
    for seed in seeds:
        for name, (beta_eta, beta_lambda) in ABLATIONS.items():
            cfg = SmokeConfig(
                seed=seed,
                horizon=1,
                online_prefix=prefix,
                dpst_beta_eta=beta_eta,
                dpst_beta_lambda=beta_lambda,
            )
            method = "ER" if name == "ER" else "DPST"
            artifact = output_root / "H1" / f"seed_{seed}" / name
            result = run_smoke(dataset, artifact, cfg, device, (method,))
            results[f"H1/{name}/seed_{seed}"] = {
                "method": method,
                "config": {"seed": seed, "beta_eta": beta_eta, "beta_lambda": beta_lambda},
                "metrics": result["methods"][method]["metrics"],
                "validation_scores": result["methods"][method]["validation_scores"],
                "artifact": str(artifact),
            }
    return results


def run_h24_smoke(dataset_path: str, output_root: Path, device: str, prefix: int, seed: int) -> dict:
    dataset = load_dataset(dataset_path)
    cfg = SmokeConfig(seed=seed, horizon=24, online_prefix=prefix)
    artifact = output_root / "H24"
    result = run_smoke(dataset, artifact, cfg, device, ("ER", "DPST", "DERPP"))
    return {
        name: {
            "metrics": value["metrics"],
            "validation_scores": value["validation_scores"],
            "artifact": str(artifact / name),
        }
        for name, value in result["methods"].items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/all_six_datasets/ETT-small/ETTh1.csv")
    parser.add_argument("--output", default="artifacts/dpst_milestone")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--prefix", type=int, default=2000)
    parser.add_argument("--h24-prefix", type=int, default=200)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = parser.parse_args()

    root = Path(args.output)
    result = {
        "ablation_h1": run_ablation(args.data, root, args.device, args.prefix, args.seeds),
        "delayed_feedback_h24": run_h24_smoke(args.data, root, args.device, args.h24_prefix, args.seeds[0]),
    }
    (root / "milestone_results.json").parent.mkdir(parents=True, exist_ok=True)
    (root / "milestone_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
