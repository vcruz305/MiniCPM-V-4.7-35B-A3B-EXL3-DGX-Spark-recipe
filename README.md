# MiniCPM-V 4.7 35B-A3B SAGE EXL3 on one DGX Spark

Serve the **3.74 bpw SAGE EXL3** release of MiniCPM-V 4.7 through
[ExLlamaV3](https://github.com/vcruz305/exllamav3) and
[TabbyAPI](https://github.com/vcruz305/tabbyAPI), at up to the full
**262,144-token** context, with optional **exact dSparK** speculative decoding.

| One DGX Spark, measured 2026-10-13 | Target-only | dSparK width 7 |
|---|---:|---:|
| Decode, 1,024-token prompts | 69.48 tok/s | **94.70 tok/s** (1.36x) |
| Decode at 262,144 tokens, Q4 KV | **45.03 tok/s** | 24.45 tok/s |
| Prefill, cold 261,886 tokens, Q4 KV | 1,056 tok/s | — |
| Needle retrieval at 262K (4 needles) | 4/4 | 4/4 |

dSparK output is byte-identical to target-only greedy decoding in every run above.
For a single 24 GB GPU, see the [RTX 4090 recipe](https://github.com/vcruz305/MiniCPM-V-4.7-35B-A3B-EXL3-RTX-4090-recipe)
(88 tok/s decode at the full 262K context).
It is faster on short prompts and slower on long ones, so it is opt-in
(see [dSparK](#dspark-exact-speculative-decoding)).

## Quick start

On Linux with a CUDA-enabled PyTorch/Triton, CUDA toolkit and a C++ compiler:

```bash
git clone https://github.com/vcruz305/MiniCPM-V-4.7-35B-A3B-EXL3-DGX-Spark-recipe.git
cd MiniCPM-V-4.7-35B-A3B-EXL3-DGX-Spark-recipe

bash exllamav3-tabby/setup.sh                       # build the latest forks
bash exllamav3-tabby/download.sh --include-drafter  # model + optional dSparK

# Full 262,144-token context (Q4 KV, target-only)
PROFILE=long bash exllamav3-tabby/serve.sh

# Or: 8K chat with exact dSparK speculative decoding
PROFILE=single DRAFT_MODE=dspark bash exllamav3-tabby/serve.sh
```

If `python3` is not your CUDA environment:
`BASE_PYTHON=/path/to/cuda-venv/bin/python bash exllamav3-tabby/setup.sh`.
Preview the generated server config without loading anything:
`DRY_RUN=1 PROFILE=long bash exllamav3-tabby/serve.sh`.

## Profiles

| Profile | Context | KV cache | Vision | Use |
|---|---:|---|---|---|
| `long` | 262,144 | Q4 | off | Full context, long documents |
| `single` | 8,192 | FP16 | off | Chat; supports `DRAFT_MODE=dspark` |
| `vision` | 4,096 | FP16 | on | Images |
| `24gb` | 4,096 | FP16 | off | Conservative small-memory start |
| `companion` | 4,096 | FP16 | off | Co-resident with another model |

Overrides: `MODEL_DIR`, `PROFILE`, `PORT`, `HOST`, `MAX_SEQ_LEN`, `CACHE_SIZE`,
`CACHE_MODE` (`FP16`/`Q8`/`Q6`/`Q4`), `CHUNK_SIZE`, `MAX_BATCH_SIZE`, `VISION`,
`DRAFT_MODE` (`disabled`/`dspark`), `DRAFT_TOKENS` (1–7), `RECIPE_HOME`,
`STATE_DIR`, `GPU_LOCK_FILE`.

**Why Q4 KV at full context:** at 262,144 tokens Q4 KV decodes at 45.03 tok/s
vs 27.26 tok/s for FP16 KV, uses less memory (18.3 vs 21.5 GiB process peak),
and still found 4/4 needles.

## Full-context results (one DGX Spark)

One cold 261,886-token request with four passcodes at 10/35/60/90% depth, then three
decode samples that fill the context exactly (261,888 prompt + 256 generated = 262,144).

| KV | Mode | Prefill | TTFT | Needles | Decode median |
|---|---|---:|---:|---:|---:|
| Q4 | target-only | 1,056 tok/s | 248.8 s | 4/4 | **45.03 tok/s** |
| Q4 | dSparK w7 | 1,058 tok/s | 249.8 s | 4/4 | 24.45 tok/s |
| FP16 | target-only | 1,073 tok/s | 245.0 s | 4/4 | 27.26 tok/s |
| FP16 | dSparK w7 | 1,061 tok/s | 247.7 s | 4/4 | 17.23 tok/s |

Decode samples were byte-identical between target-only and dSparK for each KV mode.

## dSparK: exact speculative decoding

The release ships OpenBMB's official dSparK drafter (`Qwen3DSparkModel`, 5 layers,
778 MB) under `dspark/`. With `DRAFT_MODE=dspark` the recipe verifies up to 7 drafted
tokens per target pass. The ExLlamaV3 fork's kernels are row-invariant, so a multi-token
verify pass gives exactly the same logits as one-token decoding: **output is identical
to target-only greedy decoding**, only faster when drafts are accepted.

Short prompts, TabbyAPI, 1,024-token prompts, 256 tokens out, greedy, 2 warmups + 7 runs,
FP16 KV, 8K context:

| | Target-only | dSparK width 7 |
|---|---:|---:|
| Decode p50 | 69.48 tok/s | **94.70 tok/s** |
| Prefill p50 | 742 tok/s | 737 tok/s |
| TTFT p50 | 0.948 s | 1.012 s |
| Accepted / proposed | — | 1,125 / 1,966 (57%) |
| Responses identical | | yes |

Acceptance falls as the prompt grows (Q4 KV, 131K context, 256-token essay answers):

| Prompt tokens | Target-only | dSparK w7 |
|---:|---:|---:|
| 2,040 | 60.3 | **64.5** |
| 8,183 | **60.2** | 55.3 |
| 16,376 | **59.4** | 45.8 |
| 32,758 | **57.5** | 40.6 |
| 129,992 | **54.9** | 34.8 |

So the recipe allows dSparK only with `MAX_SEQ_LEN <= 8192`, text-only, batch 1.
Use it for chat and short prompts; use `PROFILE=long` without a drafter for documents.

## Quality

Re-scored on the serving runtime used above against the model card's BF16 final-logit
references (20,480 pooled positions): **93.43% top-1 agreement, 0.0315 mean KL**
(model card: 93.39%, 0.0316). SixCat v0.7.0 on the card (79.17) was measured on an
earlier runtime and was not rerun.

## API

Authenticated localhost API at `http://127.0.0.1:8898/v1`. Keys persist under
`$STATE_DIR/api_tokens.yml`; use the **inference** key.

```bash
read -rsp 'TabbyAPI inference key: ' TABBY_API_KEY; echo; export TABBY_API_KEY
curl --fail-with-body http://127.0.0.1:8898/v1/chat/completions \
  -H "Authorization: Bearer $TABBY_API_KEY" -H 'Content-Type: application/json' \
  -d '{"model": "MiniCPM-V-4.7-35B-A3B-EXL3-SAGE-3.74bpw",
       "messages": [{"role": "user", "content": "Explain DNS caching in three sentences."}],
       "temperature": 0, "max_tokens": 256,
       "chat_template_kwargs": {"enable_thinking": false}}'
```

## Model facts

| Property | Value |
|---|---|
| Quant | SAGE EXL3, 3.7357 bpw target (3.74) |
| Weights | 19 shards, 17,999,304,996 bytes |
| Output head / vision | 8-bit / 8-bit |
| dSparK drafter | 778,452,858 bytes, optional |

## How these numbers were measured

TabbyAPI over HTTP, streamed, server-reported `usage` timings, one request at a time,
thinking disabled. Raw results: [`bench/measured-results.json`](bench/measured-results.json).
Engine: ExLlamaV3 fork `master` at `dc778646` (short context, full-context target-only)
and `475461e9` (dSparK with quantized KV, depth sweep). `475461e9` fixes a crash when
dSparK is combined with Q4/Q6/Q8 KV. Setup always installs the latest fork, so use a
build at or after `475461e9`.

Validation details: [VALIDATION.md](VALIDATION.md). Older card transcription:
[`bench/published-results.json`](bench/published-results.json).

## Credits and licenses

OpenBMB: MiniCPM-V 4.7 and the official dSparK drafter. turboderp: ExLlamaV3.
The Royal Lab: TabbyAPI. [r0b0tlab](https://github.com/r0b0tlab): MiniCPM-V 4.7
dSparK conversion improvements merged into the fork. Victor Cruz: SAGE EXL3 release,
exact one-pass dSparK verification, recipe and measurements.

Recipe code and docs: MIT. Model and runtimes keep their own licenses; see
[third-party notices](THIRD_PARTY_NOTICES.md).

AI agents: read [AGENTS.md](AGENTS.md) before editing.
