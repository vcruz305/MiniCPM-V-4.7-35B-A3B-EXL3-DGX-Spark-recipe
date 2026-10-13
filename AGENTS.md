# Agent instructions

Read README.md, VALIDATION.md, runtime.json, model.lock.json
and docs/runtime-status.md before editing.

- Run both forks from their **latest default branches**; never add
  fixed engine versions, tags, or Git SHA pins. Use SHAs only for
  diagnostic receipts and explaining benchmark conditions.
- Native MiniCPM text/vision and Qwen3DSparkModel are now merged
  into the fork. Do not reintroduce the obsolete missing-code warning.
- This README stands alone. Do not compare its layout with unrelated
  model recipes or call it a derivative of another recipe.
- Keep the MiniCPM-native architecture, vision insertion, canvas
  M-RoPE, tokenizer, original template, and EOS metadata.
- The weight manifest identifies this EXL3 *model release*, not
  an inference-runtime software version.
- Keep target-only decoding as the throughput default; matched dSparK
  was slower despite parity.
- Never invent GPU benchmarks. Distinguish published model-card data
  from helper tests and downstream OpenBMB baseline results.
- Do not kill existing services, hot-upgrade workloads, modify existing
  code checkouts, expose credentials or auto-rent hardware.
- 24 GB discrete GPU fit, secondary-model co-residency and full 262K
  context remain unverified on this new deployment procedure.
- CI is CPU-only: it checks helper behavior, not native CUDA inference.

## Tests

python3 -m unittest discover -s tests -v
python3 -m compileall -q tools bench tests
for f in exllamav3-tabby/*.sh; do bash -n "$f"; done
