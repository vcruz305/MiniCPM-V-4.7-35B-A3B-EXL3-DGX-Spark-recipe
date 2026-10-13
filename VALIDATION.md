# Validation status

Snapshot date: 2026-10-13. No new GPU workload was executed for this recipe.

| Evidence | Status |
|---|---|
| HF source, model config and weight manifest read | Verified during recipe preparation |
| Public checkpoint pinned and per-file integrity metadata captured | Verified |
| Published two-pass 125,215-tensor equality checks | Publisher-reported, not independently rerun |
| Held-out/independent fidelity and SixCat | Publisher-reported, exact pack |
| 69.70 tok/s Spark target-only decode | Publisher-reported |
| Native image smoke, red/blue | Publisher-reported, 2/2 |
| Public MiniCPM architecture source/commit | Missing from inspected public branch |
| Original server dependency freeze / KV dtype / chunk size | Not supplied by model card |
| This recipe clean install / server load | Not performed |
| OpenAI image-input path | New smoke helper supplied, not run on GPU |
| 24 GB discrete GPU / long context / co-resident second model | Not validated |
| dSparK | Publisher reports exact greedy parity but slower; disabled by default |

Generated test results are recorded in docs/recipe-tests.md after local tests.
They cover helpers and safety/validation logic, not CUDA kernel or model parity.
