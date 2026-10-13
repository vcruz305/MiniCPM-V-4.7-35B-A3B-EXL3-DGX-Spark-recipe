# MiniCPM-V 4.7 35B-A3B SAGE EXL3 on NVIDIA GPUs

Run the **3.74 bpw SAGE EXL3** release through the current
[ExLlamaV3](https://github.com/vcruz305/exllamav3) and
[TabbyAPI](https://github.com/vcruz305/tabbyAPI) forks. Native MiniCPM
text, vision and Qwen3DSparkModel support are now included upstream
in Victor's ExLlamaV3 fork.

**One DGX Spark:** 69.70 tok/s target-only decode p50 and
1,024 tok/s prefill p50 on the publisher's test.
The 19 target weight shards total 17,999,304,996 bytes.

**Latest-fork policy:** setup and every new server start check the latest
default branch of both forks. No fixed runtime commit or version pins.
Resolved commits are saved in private deployment receipts solely for
troubleshooting and honest benchmark attribution. Running servers are
never hot-updated.

## Quick start

On Linux with CUDA-enabled PyTorch/Triton, CUDA toolkit and a C++ compiler:

```bash
git clone https://github.com/vcruz305/MiniCPM-V-4.7-35B-A3B-EXL3-DGX-Spark-recipe.git
cd MiniCPM-V-4.7-35B-A3B-EXL3-DGX-Spark-recipe

# Build latest engine and API fork without touching system drivers.
bash exllamav3-tabby/setup.sh

# Download and verify the EXL3 model release.
bash exllamav3-tabby/download.sh

# Check fork updates and serve with target-only decoding.
PROFILE=single bash exllamav3-tabby/serve.sh
```

If `python3` does not belong to your existing CUDA environment:

```bash
BASE_PYTHON=/absolute/path/to/cuda-venv/bin/python bash exllamav3-tabby/setup.sh
```

Existing Torch/Triton and source checkouts are preserved. New runtime builds
go into separate managed directories. `setup.sh --check` checks remote
updates and your existing installation without modifying it.

Offline configuration preview (no model load or update):

```bash
DRY_RUN=1 PROFILE=single bash exllamav3-tabby/serve.sh
```

See [installation](docs/installation.md), [runtime updates](docs/runtime-status.md),
[vision](docs/vision.md), and [validation](VALIDATION.md).

## API

Default: authenticated localhost API at `http://127.0.0.1:8898/v1`.
Private API keys persist under `$STATE_DIR/api_tokens.yml` across
runtime rebuilds. Use the **inference** key.

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
    "messages": [{"role": "user", "content": "Explain DNS caching in three sentences."}],
    "temperature": 0,
    "max_tokens": 256,
    "chat_template_kwargs": {"enable_thinking": false},
    "stream": false
  }'
```

Use the actual model name reported by `/v1/models` if you renamed the
folder. Preserve the release's tokenizer, stop IDs and chat template.

## Profiles, memory and limits

| Profile | Request context | Shared KV | Active jobs | Vision |
|---|---:|---:|---:|---|
| `single` | 8,192 | 8,192 | 1 | off |
| `vision` | 4,096 | 4,096 | 1 | on |
| `24gb` | 4,096 | 4,096 | 1 | off |
| `companion` | 4,096 | 4,096 | 1 | off |

```bash
PROFILE=vision bash exllamav3-tabby/serve.sh
# After setting TABBY_API_KEY on another terminal:
python3 bench/smoke.py --vision
```

The 24 GB discrete-GPU and DGX Spark co-resident profiles are
deployment starting points, not completed fit tests. The source's
262,144-token configuration is not proof of long-context service.
Peak CUDA allocation was 20.33 GiB in the publisher's 8K text
quality configuration, and 18.33 GiB in a separate 4K vision smoke.

All profiles start with FP16 KV and target-only decoding. The optional
official dSparK is not faster on the published matched test, so it is
not enabled by default.

Overrides: `MODEL_DIR`, `PROFILE`, `PORT`, `HOST`, `CACHE_SIZE`,
`MAX_SEQ_LEN`, `MAX_BATCH_SIZE`, `CACHE_MODE`, `CHUNK_SIZE`,
`VISION`, `RECIPE_HOME`, `STATE_DIR`, `GPU_LOCK_FILE`.

Separate live installations should use distinct ports, state
directories and recipe homes. See [memory planning](docs/memory.md)
and [service operations](docs/service.md).

## Model facts

| Property | Exact release |
|---|---|
| Quant | 3.7357 target bpw, rounded 3.74 |
| Weights | 19 EXL3 shards; 17,999,304,996 bytes |
| Output head / vision | 8-bit / 8-bit |
| Packaging | Standalone EXL3; no BF16 checkpoint needed at serve |
| Optional dSparK | Separate 778,452,858-byte checkpoint |

The weight byte total excludes tokenizer/index/other files and the
optional drafter. Model-file checksums are for **release identity**,
not runtime version pinning.

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
The current ExLlamaV3 fork includes dedicated Qwen3DSparkModel support. Target-only remains the default because the matched dSparK test was slower. The optional drafter is not automatically enabled.

## Validation and operation

```bash
bash exllamav3-tabby/setup.sh --check
python3 bench/smoke.py --base-url http://127.0.0.1:8898/v1
python3 bench/api_benchmark.py --base-url http://127.0.0.1:8898/v1 \
  --warmups 2 --samples 7 --concurrency 1 --max-tokens 256
```

The new benchmark records real usage and end-to-end wall time, with no fabricated
prefill/decode decomposition. It is a reproducibility aid, not the original
harness that produced the model card's exact-token performance table.
See [bench/README.md](bench/README.md), [vision smoke](docs/vision.md),
[tool calling](docs/tool-calling.md), and [service operation](docs/service.md).


## Credits and licenses

OpenBMB: MiniCPM-V and official dSparK. Qwen: language-backbone
lineage. Turboderp/ExLlamaV3: EXL3 codec and engine. The Royal
Lab/TabbyAPI: OpenAI-compatible serving. Victor Cruz: SAGE EXL3
release, recipe and measurements.

The model card declares Apache-2.0. Original recipe code and docs
are MIT; external model and runtime code retains its own license.
See [third-party notices](THIRD_PARTY_NOTICES.md).

AI agents: read [AGENTS.md](AGENTS.md) before editing.
