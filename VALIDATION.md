# Validation status

Updated 2026-10-13 on one DGX Spark (GB10) and one RTX 4090.

| Item | Status |
|---|---|
| Full 262,144-token context, Q4 and FP16 KV | Measured, 4/4 needles ([README](README.md#full-context-results-one-dgx-spark)) |
| dSparK exact output vs target-only | Identical at 1K prompts and at every sweep depth to 130K |
| dSparK short-prompt speedup | 1.36x on Spark (69.48 to 94.70 tok/s) |
| Fidelity on serving runtime | 93.43% top-1, 0.0315 KL (20,480 positions) |
| ExLlamaV3 GPU tests incl. drafted == target-only greedy | 37/37 on Spark |
| Clean `setup.sh` on Spark | Exit 0 in 2 min 46 s on fork `475461e9` (parallel ninja build, 20 jobs) |
| SixCat on current runtime | Not rerun |
| Vision API with dSparK | Not supported (drafter is text-only) |
| Batch > 1 with dSparK | Not measured; blocked by config |

CPU helper checks: [recipe tests](docs/recipe-tests.md).
