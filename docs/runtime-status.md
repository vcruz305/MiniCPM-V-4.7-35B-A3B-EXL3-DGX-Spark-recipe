# Runtime publication status

The pinned HF card says a qualified ExLlamaV3 fork served this model, including
native MiniCPM vision and a correctness-first Qwen3DSparkModel path. It does not
identify its source commit, patch files, TabbyAPI commit, or dependency freeze.

The publicly inspected vcruz305/exllamav3 master was
`4d98d39ac22641231356e9a431eb1d0c12625fc2`. Its source contains neither the shipped
`MiniCPMV4_7ForConditionalGeneration` registration nor `Qwen3DSparkModel`.
The public TabbyAPI fork inspected for config schema was
`17ad217f2bc35e8bc5efee4da4d0c3176faba6d4`.
These are inspection identities, NOT claimed working MiniCPM pins.

The target config is model_type minicpmv4_7, with a qwen3_5_moe_text backbone,
a 27-layer vision tower, 16x downsampling and insertion at layer 6. Renaming it
to a generic Qwen architecture would discard part of its contract. We do not do it.

## What works in this recipe today

1. Pin and verify the public model files without downloading BF16 weights.
2. Adopt an existing MiniCPM-capable Python + TabbyAPI environment without updating it.
3. Check the runtime architecture registry before model loading.
4. Generate conservative target-only configs, capture local runtime identities,
   and run text/API/vision verification helpers.
5. Build from user-supplied immutable MiniCPM source commits in a separate location.

The third item is a compatibility check, not proof of numerical correctness.
A changed local source digest forces re-adoption rather than silently claiming
that an earlier runtime receipt still applies.

## Publish the missing runtime reproducibly

From the actual qualified installation, run the recipe's setup --adopt command.
Its private adoption.json records Python/package versions, CUDA/device identity,
source hashes and Git metadata. Review that report locally; do not upload it
unredacted. Publish the reviewed runtime changes in their own repository and
supply immutable Git refs, including any vision and TabbyAPI integration patches.
Then fill runtime.lock.json's qualified fields and rerun a clean installation.

No source code or inference behavior is patched by this recipe.
