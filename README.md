# MiniCPM-V 4.7 35B-A3B SAGE EXL3 on NVIDIA DGX Spark

Run the standalone **3.74 bpw SAGE EXL3** release with a MiniCPM-enabled
ExLlamaV3 runtime and an OpenAI-compatible TabbyAPI server. The layout follows
[the Qwen Flash Next recipe](https://github.com/vcruz305/Qwen3.8-Flash-Next-EXL3-DGX-Spark-recipe):
`exllamav3-tabby/` for deployment, `bench/` for checks, and `docs/` for details.

**24 GB GPUs are a deployment target, not a completed discrete-GPU benchmark.**
The published serving measurements are on one DGX Spark. A compact second model
on a Spark is another intended use, but co-residency with a larger model has not
been validated.

> **Runtime publication gap, checked 2026-10-13:** the model card reports a
> qualified MiniCPM runtime but does not publish its commit or patch set. The
> inspected public `vcruz305/exllamav3` branch does not register
> `MiniCPMV4_7ForConditionalGeneration`. A generic Qwen-only runtime is NOT a
> substitute. This recipe supports adopting an existing MiniCPM-enabled setup,
> and building from explicit source commits once available. It deliberately
> refuses to promise a clean-machine install from an unsupported public pin.
> See [runtime status](docs/runtime-status.md).

**Start here:** [quick start](#quick-start-with-an-existing-qualified-runtime),
[installation](docs/installation.md), [vision](docs/vision.md),
[benchmarks](bench/README.md), and [validation status](VALIDATION.md).
Agents should read [AGENTS.md](AGENTS.md) first.

## Release at a glance

| Item | This release |
|---|---|
| Model | MiniCPM-V-4.7-35B-A3B-EXL3-SAGE-3.74bpw |
| Source | OpenBMB MiniCPM-V 4.7 35B-A3B |
| Model revision | `855cb7601710064aeb7452b6b7e45e60a40be76b` |
| Target weights | 19 shards, **17,999,304,996 bytes** (18 GB decimal ceiling) |
| Bitrate | Publisher target average 3.7357, rounded to 3.74 bpw |
| Head / vision precision | 8-bit head / 8-bit vision path |
| Default deployment | One request, 8K text context, target-only decoding |
| API | `http://127.0.0.1:8898/v1`, authentication enabled |
| Source architecture limit | 262,144 tokens; not a verified serving window here |
| Speculation | Off by default; qualified dSparK was slower in the matched test |

The 18 GB ceiling is for the 19 **target weight shards**, not a promise about
full download size, GPU allocation, or a package including `dspark/`.
The optional bundled drafter adds 778,452,858 weight bytes. Configs, tokenizer,
index, and other ancillary files also occupy disk. The download manifest records
all of them separately.

## Quick start with an existing qualified runtime

Clone this recipe on the machine that will serve the model:

```bash
git clone https://github.com/vcruz305/MiniCPM-V-4.7-35B-A3B-EXL3-DGX-Spark-recipe.git
cd MiniCPM-V-4.7-35B-A3B-EXL3-DGX-Spark-recipe

# These are YOUR existing MiniCPM-enabled Python and TabbyAPI directories.
# This validates and records them. It does not upgrade or modify either one.
bash exllamav3-tabby/setup.sh --adopt \
  --python /absolute/path/to/qualified-venv/bin/python \
  --tabby-src /absolute/path/to/qualified-tabbyAPI

# Uses the adopted environment's huggingface_hub; no BF16 source download.
bash exllamav3-tabby/download.sh

# Preview only: no model load, no GPU query, no live config change.
PROFILE=single DRY_RUN=1 bash exllamav3-tabby/serve.sh

# Foreground server. Refuses busy port/state; never kills another workload.
PROFILE=single bash exllamav3-tabby/serve.sh
```

A runtime passing the architecture check is a **candidate**, not an automatic
reproduction of the publisher's benchmark. Run the checks below on your exact
installation. Adoption and fresh builds are documented separately so an old
ExLlamaV3 wheel cannot masquerade as MiniCPM support.

Use an already-downloaded unmodified pack:

```bash
MODEL_DIR="$HOME/models/MiniCPM-V-4.7-35B-A3B-EXL3-SAGE-3.74bpw" \
  bash exllamav3-tabby/download.sh --verify-only
```

The downloader pins the Hub revision and verifies file sizes and hashes. It
excludes `dspark/` by default. Add `--include-drafter` only to obtain that
optional artifact; downloading it does not enable or accelerate drafting.

## API and first request

TabbyAPI keeps its inference/admin keys in its own private `api_tokens.yml`.
Use the **inference** key, not the admin key. In a second terminal:

```bash
read -rsp 'TabbyAPI inference key: ' TABBY_API_KEY; echo
export TABBY_API_KEY
curl --fail-with-body http://127.0.0.1:8898/v1/models \
  -H "Authorization: Bearer $TABBY_API_KEY"

curl --fail-with-body http://127.0.0.1:8898/v1/chat/completions \
  -H "Authorization: Bearer $TABBY_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "MiniCPM-V-4.7-35B-A3B-EXL3-SAGE-3.74bpw",
    "messages": [{"role": "user", "content": "Explain what a DNS cache does in three sentences."}],
    "temperature": 0,
    "max_tokens": 256,
    "chat_template_kwargs": {"enable_thinking": false},
    "stream": false
  }'
```

Use the actual model ID returned by `/v1/models` if you renamed the local folder.
The preserved model template controls prompt formatting; no Qwen Flash Next
or Cyber Frost template override is applied.

## Profiles, context and memory

| Profile | Context / shared KV pool | Active requests | Vision | Status |
|---|---:|---:|---|---|
| `single` (default) | 8,192 / 8,192 | 1 | Off | Conservative recipe settings near the reported text test |
| `vision` | 4,096 / 4,096 | 1 | On | API verification candidate; native color smoke is published |
| `24gb` | 4,096 / 4,096 | 1 | Off | Discrete 24 GB GPU target, not on-card validated |
| `companion` | 4,096 / 4,096 | 1 | Off | Co-resident starting point, not validated with a second model |

The defaults use `FP16` KV. The model card does **not** specify all settings of
its measured runtime, including KV dtype and chunk size, so these profiles are
not claimed to be exact reproductions of that server. `single` uses a 1,024-token
prefill chunk; the smaller profiles use 512. Quantized KV and longer windows
need a fresh accuracy/memory evaluation.

Overrides: `MODEL_DIR`, `PROFILE`, `PORT`, `HOST`, `CACHE_SIZE`, `MAX_SEQ_LEN`,
`MAX_BATCH_SIZE`, `CACHE_MODE`, `CHUNK_SIZE`, `VISION`, `CUDA_VISIBLE_DEVICES`,
`STATE_DIR`, `GPU_LOCK_FILE`. Profile changes do not create additional memory.
Cache size must be a multiple of 256 and at least the per-request sequence limit.
The launcher refuses a sequence limit above the source's 262,144-token maximum.

A recipe configuration exceeding the small published context tests is labeled
**unvalidated** in the deployment receipt. Weight fit alone does not validate
long-context text, vision, or simultaneous models. See [memory planning](docs/memory.md).

## Measured results from the model publisher

These values are transcribed from the pinned [model card](https://huggingface.co/vcruz305/MiniCPM-V-4.7-35B-A3B-EXL3-SAGE-3.74bpw/blob/855cb7601710064aeb7452b6b7e45e60a40be76b/README.md).
They were **not rerun while creating this recipe**. A source snapshot and structured
results are in [`evidence/`](evidence/README.md).

### Exact shipping-pack fidelity

| Set | Top-1 agreement | Mean KL, nats |
|---|---:|---:|
| Held-out | 93.642578125% | 0.031476575675059786 |
| Independent | 93.134765625% | 0.03168882649150257 |
| Equal-token pooled | 93.388671875% | 0.03158270108328118 |

20,480 pooled token positions, against immutable BF16-reference final-logit
stores. The publisher reports two runs on the exact 19-shard pack. Top-1 measures
matching the reference's most-likely next token, not task accuracy. KL compares
full next-token distributions; lower is closer. The raw logit stores and
individual quality receipts are not included in this repository.

### Target-only text serving on one DGX Spark

| Metric | Published result |
|---|---:|
| Decode p50 | **69.70 tok/s** |
| Prefill p50 | **1,024.00 tok/s** |
| TTFT p50 | **1.103 s** |
| Aggregate output, concurrency 1 | **53.57 tok/s end-to-end** |

1,024 uncached prompt tokens, 256 generated tokens, greedy sampling, two warmups,
seven measured requests. Peak CUDA allocation was **20.33 GiB** in the 8K text
quality configuration and **18.33 GiB** in a separate 4K text-plus-vision smoke.
Those allocation rows are not identical to the throughput workload.

### SixCat v0.7.0

**79.17 overall, 95/120 correct**, 20 first-scored responses per category.
Knowledge / Math / Truth / Instruct / Code / Tools: **50 / 100 / 80 / 70 / 85 / 90**.
Custom temperature 0.7, seed 1, thinking on, concurrency 4, guarded code execution.
All 120 items scored, no truncations or low-confidence parses; two instruction
items were loop failures. This is not a multimodal benchmark suite.

## Why target-only is the default

The official bundled dSparK is a **separate `Qwen3DSparkModel` checkpoint**.
It is not automatically compatible with an ordinary MTP or DFlash toggle.

| Matched publisher run | Target-only | dSparK width 4 |
|---|---:|---:|
| Decode p50 | 69.33 tok/s | 45.31 tok/s |
| Prefill p50 | 764.18 tok/s | 752.94 tok/s |
| TTFT p50 | 0.963 s | 1.035 s |
| Accepted / proposed | N/A | 988 / 1,438 (68.71%) |

Greedy parity passed 27/27 width/prompt checks, and full response hashes matched
for the paired speed run. Nevertheless, exact verification was **34.65% slower**.
Do not substitute the separate 69.70 tok/s row into this matched comparison.
The recipe keeps drafting disabled and does not fabricate a public dSparK launcher.

## Validation and operation

```bash
bash exllamav3-tabby/setup.sh --check
python3 bench/smoke.py --base-url http://127.0.0.1:8898/v1
python3 bench/api_benchmark.py --base-url http://127.0.0.1:8898/v1 \
  --warmups 2 --samples 7 --concurrency 1 --max-tokens 256
```

The new benchmark records real usage and end-to-end wall time, with no fabricated
prefill/decode decomposition. It is a reproducibility aid, not the unpublished
harness that produced the model card's exact-token performance table.
See [bench/README.md](bench/README.md), [vision smoke](docs/vision.md),
[tool calling](docs/tool-calling.md), and [service operation](docs/service.md).

## Credits and licenses

OpenBMB: MiniCPM-V model and official dSparK. Qwen: language-backbone lineage.
Turboderp/ExLlamaV3: EXL3 format and engine. The Royal Lab/TabbyAPI: API server.
Victor Cruz: this SAGE-branded checkpoint, recipe, and published measurements.

The model card declares Apache-2.0 for the model. The recipe's original scripts
and docs are MIT; external model/runtime code keeps its own license. No private
SAGE allocation internals, credentials, model weights, or inference patches are
redistributed. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
