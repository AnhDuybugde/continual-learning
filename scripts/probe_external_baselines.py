"""Probe pinned official external baselines without running a benchmark."""

from __future__ import annotations

import json
from pathlib import Path

from continual_forecasting.external_adapters import FSNET_SOURCE, ONENET_SOURCE, probe_official_import


def main() -> None:
    results = [probe_official_import(source) for source in (FSNET_SOURCE, ONENET_SOURCE)]
    output = Path("artifacts/external_baseline_probe.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
