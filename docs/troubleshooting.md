# Troubleshooting

**Old or missing MiniCPM support:** verify which Python/runtime is
actually imported. `setup.sh` resolves the latest fork HEAD. The
native model and vision support is now merged; do not rename
the checkpoint to generic Qwen to work around a stale engine.

**CUDA installation fails:** use a compatible GPU-specific PyTorch,
Triton, `nvcc` and compiler. On DGX Spark use aarch64 wheels
where appropriate. This recipe does not touch system drivers.

**Remote GitHub unavailable:** rolling updates fail visibly,
not silently fall back to older code. Existing workloads keep
their active source version until restarted by their owner.

**Auth rejected:** use `$STATE_DIR/api_tokens.yml`'s inference key.
These private credentials persist across staged runtime updates.

**Vision requests fail:** use `PROFILE=vision` and verify the
native vision component was loaded. API image testing and full
multimodal evaluation are separate.

**dSparK slower:** expected on long prompts. Its draft acceptance
collapses as the prompt grows, so it only helps short chat prompts;
on one DGX Spark it is 1.36x faster at 1K prompts and slower from 8K. The recipe blocks `DRAFT_MODE=dspark` above 8,192
tokens of context. Output is identical either way.

**dSparK crashes with Q4/Q6/Q8 KV:** your ExLlamaV3 build predates
fork commit `475461e9`. Rerun `setup.sh` to pick up the latest fork.

**Setup takes 30+ minutes:** the CUDA extension is being compiled
serially. Current setup refuses to build without a runnable `ninja`
and prints `Parallel extension build: ninja at ..., MAX_JOBS=N`;
expect a few minutes on a Spark.

**Hash mismatch:** the quant weight release must be the exact
named set in model.lock.json. Download a clean copy rather than
changing config or repacking weights.

**OOM:** lower context, batch and chunk size. Check other memory
users without killing them. Discrete 24 GB and companion-model
profiles are starting points, not fit guarantees.

**Performance differs from these results:** compare actual
prompts/output counts, caching, sampler, hardware, runtime
commits, KV cache mode and benchmark method. Raw measured values
are in `bench/measured-results.json`.
