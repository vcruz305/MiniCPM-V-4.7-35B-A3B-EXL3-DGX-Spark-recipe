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

**dSparK slower:** expected in the published matched run.
Target-only stays the default. Do not treat a downloaded drafter
as active MTP acceleration.

**Hash mismatch:** the quant weight release must be the exact
named set in model.lock.json. Download a clean copy rather than
changing config or repacking weights.

**OOM:** lower context, batch and chunk size. Check other memory
users without killing them. Discrete 24 GB and companion-model
profiles are starting points, not fit guarantees.

**Performance differs from the model card:** compare actual
prompts/output counts, caching, sampler, hardware, runtime
commits, cache dtype and benchmark method. Published metrics
were not remeasured during this recipe update.
