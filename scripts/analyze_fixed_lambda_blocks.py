"""Analyze fixed-lambda prediction artifacts block by block.

This is artifact-only: it never reruns a model and is not confirmatory
statistics.  MAE and MSE winners are reported separately because a block can
have different winners under the two losses.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


DEFAULT_LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def _load_predictions(root: Path, seed: int, replay_weight: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    label = f"{replay_weight:g}"
    path = root / "H1" / f"seed_{seed}" / f"DPST_fixed_lambda_{label}" / "DPST" / "predictions.npz"
    with np.load(path) as artifact:
        return artifact["predictions"].copy(), artifact["targets"].copy(), artifact["sample_ids"].copy()


def _best(values: dict[str, float]) -> str:
    return min(values, key=values.get)


def _changes(sequence: list[str]) -> int:
    return sum(previous != current for previous, current in zip(sequence, sequence[1:]))


def analyze(root: Path, seeds: list[int], block_size: int, lambdas: tuple[float, ...]) -> dict:
    lambda_one_key = str(1.0)
    seed_results: dict[str, dict] = {}
    all_blocks: dict[str, list[dict]] = {}
    for seed in seeds:
        loaded = {weight: _load_predictions(root, seed, weight) for weight in lambdas}
        reference_ids = loaded[lambdas[0]][2]
        for weight, (_, _, sample_ids) in loaded.items():
            if not np.array_equal(reference_ids, sample_ids):
                raise ValueError(f"evaluated sample IDs differ for seed={seed}, lambda={weight}")
        count = len(reference_ids)
        blocks: list[dict] = []
        for start in range(0, count, block_size):
            stop = min(start + block_size, count)
            mae = {str(weight): float(np.mean(np.abs(loaded[weight][0][start:stop] - loaded[weight][1][start:stop])) ) for weight in lambdas}
            mse = {str(weight): float(np.mean((loaded[weight][0][start:stop] - loaded[weight][1][start:stop]) ** 2)) for weight in lambdas}
            best_mae = _best(mae)
            best_mse = _best(mse)
            blocks.append({
                "start_index": start,
                "end_index": stop - 1,
                "start_sample_id": int(reference_ids[start]),
                "end_sample_id": int(reference_ids[stop - 1]),
                "n_samples": stop - start,
                "mae": mae,
                "mse": mse,
                "best_lambda_mae": float(best_mae),
                "best_lambda_mse": float(best_mse),
                "mae_gain_oracle_vs_lambda_1": mae[lambda_one_key] - mae[best_mae],
                "mse_gain_oracle_vs_lambda_1": mse[lambda_one_key] - mse[best_mse],
            })
        mae_winners = [str(block["best_lambda_mae"]) for block in blocks]
        mse_winners = [str(block["best_lambda_mse"]) for block in blocks]
        seed_results[str(seed)] = {
            "resolved_samples": count,
            "blocks": len(blocks),
            "mae": {
                "best_lambda_by_block": [float(value) for value in mae_winners],
                "best_lambda_changes": _changes(mae_winners),
                "lambda_1_win_blocks": sum(value == lambda_one_key for value in mae_winners),
                "lambda_1_win_rate": float(np.mean([value == lambda_one_key for value in mae_winners])),
                "oracle_gain_total_vs_lambda_1": float(sum(block["mae_gain_oracle_vs_lambda_1"] * block["n_samples"] for block in blocks) / count),
            },
            "mse": {
                "best_lambda_by_block": [float(value) for value in mse_winners],
                "best_lambda_changes": _changes(mse_winners),
                "lambda_1_win_blocks": sum(value == lambda_one_key for value in mse_winners),
                "lambda_1_win_rate": float(np.mean([value == lambda_one_key for value in mse_winners])),
                "oracle_gain_total_vs_lambda_1": float(sum(block["mse_gain_oracle_vs_lambda_1"] * block["n_samples"] for block in blocks) / count),
            },
        }
        all_blocks[str(seed)] = blocks

    aggregate: dict[str, dict] = {}
    total_samples = sum(block["n_samples"] for blocks in all_blocks.values() for block in blocks)
    for metric in ("mae", "mse"):
        baseline = sum(block[metric]["1.0"] * block["n_samples"] for blocks in all_blocks.values() for block in blocks) / total_samples
        oracle = sum(min(block[metric].values()) * block["n_samples"] for blocks in all_blocks.values() for block in blocks) / total_samples
        winner_values = [block[f"best_lambda_{metric}"] for blocks in all_blocks.values() for block in blocks]
        aggregate[metric] = {
            "total_blocks": len(winner_values),
            "total_samples": total_samples,
            "lambda_1_win_blocks": sum(value == 1.0 for value in winner_values),
            "lambda_1_win_rate": float(np.mean([value == 1.0 for value in winner_values])),
            "oracle_mean_loss": float(oracle),
            "fixed_lambda_1_mean_loss": float(baseline),
            "oracle_gain_vs_lambda_1": float(baseline - oracle),
            "oracle_relative_gain_vs_lambda_1": float((baseline - oracle) / baseline),
            "best_lambda_changes_by_seed": {seed: seed_results[seed][metric]["best_lambda_changes"] for seed in seed_results},
        }
    return {
        "analysis": "fixed_lambda_blockwise_oracle",
        "artifact_only": True,
        "block_size": block_size,
        "lambdas": list(lambdas),
        "seeds": seeds,
        "seed_results": seed_results,
        "blocks": all_blocks,
        "aggregate": aggregate,
        "interpretation": "Oracle block winners are counterfactual because each fixed lambda has its own trajectory; use this as headroom diagnostic, not confirmatory evidence.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="artifacts/dpst_milestone/fixed_lambda")
    parser.add_argument("--output", default="artifacts/dpst_milestone/fixed_lambda/blockwise_oracle.json")
    parser.add_argument("--block-size", type=int, default=100)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = parser.parse_args()
    result = analyze(Path(args.root), args.seeds, args.block_size, DEFAULT_LAMBDAS)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["seed_results"], indent=2))


if __name__ == "__main__":
    main()
