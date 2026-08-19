# Benchmark smoke report

Status: **PASS — engineering smoke only**.

Scope is deliberately limited to ETTh1, `L=60`, `H=1`, one fixed seed, two learning-rate candidates, and a recorded online prefix. Smoke artifacts are engineering/debugging artifacts and are not confirmatory benchmark results.

## Command

```text
python scripts/run_smoke.py --data data/all_six_datasets/ETT-small/ETTh1.csv --output artifacts/smoke_etth1 --prefix 2000
```

## Methods

OGD, ER, and DPST-Core share the same causal TCN, optimizer family, split, scaler, timestamps, and evaluated samples. DER++, FSNet, OneNet, and NatSR remain pending official-source integration.

## Results

ETTh1 contains 17,420 hourly rows. The fixed split is train `[0, 3484)`, validation `[3484, 4355)`, online `[4355, 17420)`. The scaler was fit only on the first 3,484 rows. The smoke emitted 1,999 resolved samples per method (the first issued forecast is pending under `H=1`).

| Method | Selected LR | MAE | MSE | MASE | Updates | Online runtime (s) |
|---|---:|---:|---:|---:|---:|---:|
| OGD | 0.001 | 0.109258 | 0.019369 | 1.230648 | 1,999 | 5.99 |
| ER | 0.001 | 0.091804 | 0.014680 | 1.034047 | 1,999 | 27.15 |
| DPST-Core | 0.001 | 0.088667 | 0.014002 | 0.998713 | 1,999 | 40.81 |

All three methods used the same evaluated sample IDs (`4355..6353`), finite predictions/losses, the same split/scaler, and fixed seed `7`. These results are not final scientific claims.

## Tests and artifacts

`python -m unittest discover -s tests -p test_pipeline.py -q` passes 7/7. The smoke execution test passes 1/1. Artifacts are under `artifacts/smoke_etth1/`: per-method `config.json`, `metrics.json`, `online_metrics.jsonl`, `predictions.npz`, checkpoint, timing, and logs; parquet availability is recorded by `predictions.parquet.unavailable.txt` when no parquet engine is installed. No full benchmark was run.
