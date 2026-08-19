"""Generate plots from saved JSONL artifacts without rerunning a model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def generate(root: Path, output: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    for method_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        source = method_dir / "online_metrics.jsonl"
        if not source.exists():
            continue
        rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not rows:
            continue
        x = [row["sample_id"] for row in rows]
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        axes[0].plot(x, [row["prequential_loss"] for row in rows], label="prequential MSE")
        axes[0].set_ylabel("loss")
        axes[0].legend()
        for key in ("eta", "lambda", "alignment", "forget_score"):
            if key in rows[0]:
                axes[1].plot(x, [row[key] for row in rows], label=key)
        axes[1].set_xlabel("resolved sample")
        axes[1].legend()
        fig.suptitle(method_dir.name)
        figure_path = output / f"{method_dir.name.lower()}_trajectories.png"
        fig.tight_layout()
        fig.savefig(figure_path, dpi=140)
        plt.close(fig)
        generated.append(figure_path)
    return generated


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for path in generate(args.artifacts, args.output):
        print(path)
