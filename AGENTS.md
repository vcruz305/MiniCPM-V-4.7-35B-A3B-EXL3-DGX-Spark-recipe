# Agent instructions

Read README.md, VALIDATION.md, docs/runtime-status.md, model.lock.json and
runtime.lock.json before changing deployment behavior.

- This is the MiniCPM recipe, NOT Qwen Flash Next. Never rewrite the architecture
  to Qwen3.5 just to pass a loader. Preserve the MiniCPM vision insertion/resampler.
- Public model measurements are evidence, not results from these new scripts.
- Never claim the public ExLlamaV3 pin supports MiniCPM if the registry rejects it.
- Do not invent a working runtime commit, KV mode, prefill chunk, or dSparK flag.
- Target-only remains default. The matched dSparK path was slower.
- No automatic GPU rental, model load, process kill, service enable, system pip,
  firewall change, or mutation of existing source checkouts while editing this repo.
- Use separate virtual environments and source directories. Respect optional
  cooperative GPU locks. A different port alone does not isolate VRAM.
- Do not publish local adoption.json, access tokens, API keys, local source diffs,
  benchmark prompts from private workloads, or raw pip direct URLs.
- Keep exactly one authoritative model revision in model.lock.json. Do not modify
  config.json, tokenizer, chat template or weight files inside a downloaded pack.
- 262K is source configuration, not proven deployment here. 24 GB cards and
  co-resident use are targets, not completed hardware tests.
- Native image smoke and OpenAI image-input validation are different gates.
- Published quality is BF16-reference agreement, not a downstream multimodal score.
- Do not put OpenBMB's upstream benchmark scores into the quant-results table.

## Release completion gate

Obtain the exact MiniCPM-enabled ExLlamaV3 source (including all patches) and the
measured TabbyAPI identity from the running qualification environment. Review
those changes, publish their source in the correct runtime repository, and pin
both immutable commits in runtime.lock.json. Then perform a clean install and
capture successful text, image, EOS/tool, memory and speed receipts. Only then
remove the runtime publication warning. A source-marker or import test alone is
not sufficient.

## Tests

python3 -m unittest discover -s tests -v
python3 -m compileall -q tools bench tests
for f in exllamav3-tabby/*.sh; do bash -n "$f"; done

CI is CPU-only; passing it never means the model has been GPU-qualified.
