# Validation status

Updated 2026-10-13. No new GPU model load or benchmark
was performed while changing the recipe.

| Item | Status |
|---|---|
| MiniCPM text architecture registered in fork | Confirmed on current master |
| MiniCPM vision and image processing in fork | Confirmed |
| Dedicated Qwen3DSparkModel in fork | Confirmed |
| Rolling runtime selection | Latest default branches, no version pins |
| Quant model file manifest | Exact published release hashes |
| Publisher's fidelity, SixCat, Spark speed | Source-attributed, not rerun here |
| Clean install, API model load, vision API | Not rerun for these changes |
| 24 GB GPU and Spark co-resident serving | Not independently validated |
| 262K live context with this recipe | Not validated |

CPU helper checks and workflow status are in
[recipe tests](docs/recipe-tests.md).
