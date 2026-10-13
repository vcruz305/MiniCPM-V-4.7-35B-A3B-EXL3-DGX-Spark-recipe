# Runtime source and updates

The current ExLlamaV3 fork includes MiniCPM-V 4.7 with native
text and vision support:
- `exllamav3/architecture/minicpmv4_7.py` for MiniCPM text and
  canvas M-RoPE;
- `minicpmv4_7_vision.py` and `mm_processing/minicpmv4_7.py`
  for the vision and image pipelines;
- `qwen3_dspark.py` for the separate dedicated dSparK model;
- entries in the runtime's architecture registry.

These changes were merged into the current default branch, so
the earlier missing-support warning no longer applies.

## Always use the latest forks

The runtime policy resolves both Git remotes' symbolic HEAD at
each setup/start, regardless of whether the default branch is
`main` or `master`. A changed HEAD, compiler/CUDA environment
or installer source causes a staged rebuild. Unchanged installations
are checked and reused.

A staged installation is activated only after pip and runtime
checks pass. If updating fails, the previous installation remains
untouched but the requested new launch exits. An already-running
server is never replaced or stopped.

No fixed software versions, tags or commits are selected.
For benchmark attribution only, private `active.json` and
`deployment.json` retain resolved SHAs and environment identities.
These values are not inputs to future version selection.

The installer retains the user's existing hardware-compatible
PyTorch/Triton stack rather than replacing it with generic
architecture-incompatible wheels.

## Qualification scope

Runtime source availability does not itself establish a clean
on-device installation or downstream model quality. The
publisher's speed and fidelity results are retained separately.
24 GB discrete GPU fit, two-model co-residency and full long
context still require controlled on-hardware validation.
