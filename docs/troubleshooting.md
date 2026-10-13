# Troubleshooting

**Unknown MiniCPMV4_7ForConditionalGeneration**: this is the known runtime
publication gap. Obtain the actual MiniCPM adapter or use its qualified local
installation. Never rename config.json to Qwen or bypass the registry check.

**No compatible CUDA PyTorch / build failure**: check Python, Torch CUDA, driver,
nvcc, compiler and detected architecture together. A cp312 x86_64 wheel is not a
Spark aarch64 wheel. This recipe does not change your driver or system Python.

**Runtime identity changed**: stop the recipe-owned server, inspect what changed,
then rerun setup --adopt deliberately. Do not edit a live generated config.

**Vision disabled**: require a native MiniCPM vision component plus the 8-bit
vision files and a working TabbyAPI image path. The text backbone alone is not
the complete model.

**Slower with the bundled drafter**: expected in the published exact dSparK run.
Target-only is the recommended default. Do not substitute ordinary MTP/DFlash
flags for dedicated Qwen3DSparkModel support.

**Hash mismatch**: confirm the pinned revision and download path. Preserve any
local changes elsewhere; do not silently mutate a checkpoint. Use a fresh local
model directory to obtain the pinned snapshot.

**OOM**: reduce batch, context, image resolution and prefill chunks. Inspect all
processes and total unified memory without killing another owner's workload.
The 24 GB and co-serving profiles are starting points, not universal fit proofs.

**Speed differs from 69.70**: preserve model/runtime IDs, prompt/output token
counts, thinking, cache, warmups and measurement type. The original runtime pin
and full benchmark harness are not yet published; this recipe's generic API
benchmark only measures end-to-end throughput and does not reproduce that exact
decode figure by construction.
