# DPST-Core report

Status: **DPST-Core smoke PASS; full benchmark blocked by protocol stage**.

DPST-Core currently contains only the specified adaptive learning-rate (`eta`) and replay-weight (`lambda`) controls around the shared TCN + AdamW + reservoir replay path. It does not include a neural controller, drift detector, relevance-aware replay, or NatSR/Fisher machinery.

Smoke configuration: ETTh1, target `OT`, seed `7`, `L=60`, `H=1`, train/validation/online `20%/5%/75%`, LR candidates `{1e-3, 3e-4}`, selected `1e-3`, reservoir capacity `500`, replay batch `8`. Data SHA-256 is in `artifacts/smoke_etth1/data_manifest.json`.

| Method | MAE | MSE | MASE | Updates |
|---|---:|---:|---:|---:|
| OGD | 0.109258 | 0.019369 | 1.230648 | 1,999 |
| ER | 0.091804 | 0.014680 | 1.034047 | 1,999 |
| DPST-Core | 0.088667 | 0.014002 | 0.998713 | 1,999 |

DPST trajectories (`eta`, `lambda`, `alignment`, `forget_score`) are logged per resolved sample in `artifacts/smoke_etth1/DPST/online_metrics.jsonl`; the checkpoint is `artifacts/smoke_etth1/DPST/checkpoint.pt`. The fixed-controller regression test passes the smoke invariant checks; a stricter bitwise equality test over identical initial state and replay RNG remains a follow-up before full experiments.

No full benchmark or external baseline claim is made. Runtime/memory logging is currently wall-clock timing; peak memory instrumentation and automated trajectory plots remain to be added before final reporting.
