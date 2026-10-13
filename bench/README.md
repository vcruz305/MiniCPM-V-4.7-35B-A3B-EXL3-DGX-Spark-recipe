# Benchmarks

`published-results.json` is a transcription of the pinned HF card, not output
from these new helpers. Preserve that separation when publishing results.

## Quick API checks

```bash
export TABBY_API_KEY  # already set to your inference key
python3 bench/smoke.py
# Requires PROFILE=vision server; these are API color-smoke tests, not OCR eval.
python3 bench/smoke.py --vision
```

## New end-to-end measurement helper

```bash
python3 bench/api_benchmark.py --warmups 2 --samples 7 \
  --concurrency 1 --max-tokens 256 --output results/text-c1.json
python3 bench/api_benchmark.py --warmups 2 --samples 16 \
  --concurrency 4 --max-tokens 256 --output results/text-c4.json
```

The program resolves the actual model ID, adds a fresh nonce to each prompt,
uses temperature 0, sends non-streamed chat requests and counts tokens from
server `usage`, never SSE chunks. It reports total successful output tokens /
full measured wall including queueing and failures. Each request records E2E,
actual prompt/output counts, finish reason, success and response hash.

It intentionally reports no TTFT, isolated prefill or raw decode estimate from a
non-streamed response. A token cap is not proof of exactly that output length.
Unique nonces discourage whole-prompt cache reuse but do not prove no partial
prefix was reused. This helper does not claim 1,024 uncached prompt tokens.

## Reproducing the publisher's exact speed table

The original benchmark harness and its full runtime settings are not published in the model card, although MiniCPM support is now in the public fork. The recorded contract is 1,024 uncached prompt tokens, 256 completion
tokens, greedy, concurrency 1, two warmups and seven samples. Also capture actual
KV dtype, prefill chunk, vision state, server/engine commits, GPU clocks and
memory. Reject mismatched token lengths rather than silently average them.

## Quality

Do not recreate the KLD/top-1 scores from public prose or tiny synthetic samples.
They need the original immutable BF16-reference logits and matching token IDs.
The published pooled set is 20,480 positions. Keep source precision and model
identity fixed; never compare BF16-reference fidelity with a separate FP8 teacher.

The published SixCat v0.7.0 setup has temperature 0.7, seed 1, thinking on,
concurrency 4 and all 120 first-scored items. Use that exact version and the
original selection/scorer identity to reproduce it. Review its guarded code
execution requirements first. A newer SixCat scorer is a separate result.

No benchmark is run automatically during installation or CI.
