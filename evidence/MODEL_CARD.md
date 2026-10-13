---
license: apache-2.0
base_model: openbmb/MiniCPM-V-4.7-35B-A3B
base_model_relation: quantized
library_name: exllamav3
pipeline_tag: image-text-to-text
language:
  - en
  - zh
tags:
  - minicpm-v
  - multimodal
  - moe
  - exl3
  - exllamav3
  - quantized
  - sage
---

# MiniCPM-V 4.7 35B-A3B — SAGE EXL3 3.74 bpw

An EXL3 quantization of [OpenBMB MiniCPM-V 4.7 35B-A3B](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B), released under the SAGE name. This is a standalone quantized checkpoint; the original weights are not needed at load time. The upstream model card declares Apache-2.0; this card keeps that license attribution and links the exact upstream base.

The shipping weights are a tensor-identical compact re-shard of the selected quantization. All 125,215 tensors were checked twice against the source pack by key, dtype, shape and raw-byte SHA-256 before release.

## At a glance

| Item | Value |
|---|---|
| Base | [openbmb/MiniCPM-V-4.7-35B-A3B](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B) |
| Intended Hub ID | `vcruz305/MiniCPM-V-4.7-35B-A3B-EXL3-SAGE-3.74bpw` |
| Format | EXL3 for ExLlamaV3 |
| Target average | 3.74 bits per weight (rounded from 3.7357) |
| Weight layout | 19 shards; 17,999,304,996 serialized bytes including safetensors headers |
| Byte ceiling | 18,000,000,000 bytes; verified margin 695,004 bytes |
| Vision weights | 8-bit vision path; not full precision |
| License | Apache-2.0, as declared by the upstream model card |
| Integrity | 125,215 tensors; two full raw-byte verification passes; zero mismatches |

## Vision precision

The vision path is stored at 8-bit precision. This is not a full-precision vision checkpoint. On the exact shipping files, a native ExLlamaV3 smoke test produced distinct, finite 64×2048 FP16 embeddings for synthetic red and blue images and returned the correct one-word color for both under greedy decoding. This is a functional validation, not a replacement for the upstream multimodal benchmark suite.

## Load and serve

Install a compatible CUDA-enabled PyTorch and ExLlamaV3 build first. From an ExLlamaV3 checkout, the upstream interactive example takes a local model directory and a prompt format:

```bash
python examples/chat.py -m /path/to/model-directory -mode <prompt-mode>
```

For an OpenAI-compatible API, TabbyAPI is the ExLlamaV3 backend. Put the downloaded model folder under its configured `model_dir`, set `model.model_name` in `config.yml` to that folder name, then launch:

```bash
python main.py
```

The API defaults to `http://127.0.0.1:5000`; query `GET /v1/models` to confirm a model is loaded before sending requests. Use the upstream [TabbyAPI setup guide](https://github.com/theroyallab/tabbyAPI/wiki/01.-Getting-Started) for installation and configuration.

**Compatibility note:** ExLlamaV3 and TabbyAPI support depends on the installed release supporting the MiniCPM-V 4.7 architecture and multimodal path. Confirm this on the intended version before deployment; the interactive example above does not itself establish image/video support.

## Benchmarks and measurements

### Official upstream benchmarks

The values in this section are **OpenBMB's published results for the original checkpoint**, transcribed from the charts in the [official MiniCPM-V 4.7 card](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B). They were not rerun on this EXL3 release and must not be interpreted as quantized-model scores.

OpenBMB also reports **13+ tokens/s on an iPhone 17 Pro Max** for its on-device stack. That number uses OpenBMB's own runtime, scheduling, quantization and speculative-sampling setup; it is not an ExLlamaV3 performance result.

#### Vision-language — upstream 35B-A3B Instruct

MiniCPM-V 4.7 supports selectable 4× and 16× visual-token compression. Higher is better unless marked with ↓.

| Task | Benchmark | 4× | 16× |
|---|---|---:|---:|
| General | MMBench | 91.72 | 90.87 |
| General | MMStar | 80.67 | 80.13 |
| General | MMT-Bench | 75.47 | 75.12 |
| General | SimpleVQA | 54.14 | 51.41 |
| STEM | MMMU | 80.22 | 80.00 |
| STEM | MathVista | 88.00 | 87.30 |
| STEM | MMMU-Pro | 69.71 | 67.66 |
| STEM | MathVerse_MINI | 86.45 | 84.80 |
| STEM | MathVision | 74.24 | 72.93 |
| Document, OCR & chart | OCRBench | 91.20 | 90.40 |
| Document, OCR & chart | OCRBench_v2 | 68.47 | 65.67 |
| Document, OCR & chart | OmniDocBench_v1.6 | 94.24 | 93.25 |
| Document, OCR & chart | CharXiv_Reasoning | 75.60 | 69.20 |
| Document, OCR & chart | CCOCR | 81.14 | 78.29 |
| Document, OCR & chart | TextVQA_VAL | 84.99 | 81.72 |
| Multi-image | BLINK | 70.33 | 68.12 |
| Multi-image | MMSI-Bench | 16.00 | 15.20 |
| Hallucination | Hallusion-Bench | 63.83 | 64.42 |
| Hallucination | MMHal-Score | 4.72 | 4.67 |
| Hallucination | MMHal-Hallrate ↓ | 21.88 | 22.92 |
| Grounding & GUI | RefCOCO | 90.93 | 89.61 |
| Grounding & GUI | ScreenSpot | 86.99 | 85.01 |
| Instruction following | MM-IFEval | 65.41 | 64.62 |
| Video understanding | Video-MME (w/o subs) | 76.40 | 74.70 |
| Video understanding | LongVideoBench | 68.90 | 67.60 |
| Video understanding | MotionBench | 68.12 | 68.02 |

#### Language and tool use — upstream 35B-A3B Instruct

| Task | Benchmark | Score |
|---|---|---:|
| Knowledge | CMMLU | 90.21 |
| Knowledge | MMLU-Pro | 83.45 |
| Knowledge | SuperGPQA | 61.23 |
| Instruction following | IFBench | 47.67 |
| Math | AIME26@avg16 | 76.46 |
| Coding | LiveCodeBench-v6 | 56.57 |
| Tool call | BFCL-V4 | 60.06 |
| Tool call | τ²-Bench | 84.65 |

### Measurements for this quantization

The quality rows below were run twice on the exact 19-shard shipping pack against immutable BF16 final-logit stores. The pooled row contains 20,480 equally weighted tokens.

| Measurement | Exact-release result |
|---|---:|
| Held-out BF16-reference top-1 agreement | 0.93642578125 |
| Held-out mean KL divergence (nats) | 0.031476575675059786 |
| Independent BF16-reference top-1 agreement | 0.93134765625 |
| Independent mean KL divergence (nats) | 0.03168882649150257 |
| Equal-token pooled top-1 agreement | 0.93388671875 |
| Equal-token pooled mean KL divergence (nats) | 0.03158270108328118 |
| Native 8-bit vision functional smoke | 2/2 correct; distinct finite embeddings |
| Sixcat v0.7.0, exact shipping pack | 79.17 overall; 95/120 correct |
| Sixcat categories: Knowledge / Math / Truth / Instruct / Code / Tools | 50 / 100 / 80 / 70 / 85 / 90 |
| DGX Spark target-only prefill p50 | 1,024.00 tok/s |
| DGX Spark target-only decode p50 | 69.70 tok/s |
| DGX Spark target-only TTFT p50 | 1.103 s |
| DGX Spark target-only aggregate output rate | 53.57 tok/s end-to-end at concurrency 1 |

The throughput workload used a production TabbyAPI/ExLlamaV3 server, 1,024 uncached prompt tokens, 256 generated tokens, greedy decoding, concurrency 1, two warmups and seven measured requests. Every measured prompt and completion had the intended token count. Peak CUDA allocation was 20.33 GiB in the 8,192-token text-quality configuration; the separate 4,096-token text-plus-vision smoke peaked at 18.33 GiB.

Sixcat used v0.7.0 with 20 first-scored responses per category, custom temperature 0.7, seed 1, thinking enabled, host-guarded code execution and concurrency 4. All 120 items were scored with no truncations or low-confidence parses; two instruction rows were flagged as loop failures.

### dSparK status

The pinned official OpenBMB dSparK checkpoint is included under `dspark/`. Dedicated `Qwen3DSparkModel` support in the qualified ExLlamaV3 fork now preserves target-only greedy output by committing verified target tokens through the ordinary one-token target path.

On the exact shipping pack, widths 1, 4 and 14 each matched target-only output on all nine deterministic prompts (**27/27** width/prompt checks). The matched width-4 workload also produced identical complete response hashes for both warmups and all seven measured requests.

| Exact-release DGX Spark measurement | Target only | dSparK width 4 |
|---|---:|---:|
| Decode p50 | 69.33 tok/s | 45.31 tok/s |
| Prefill p50 | 764.18 tok/s | 752.94 tok/s |
| TTFT p50 | 0.963 s | 1.035 s |
| Accepted / proposed draft tokens | — | 988 / 1,438 (68.71%) |
| Relative decode throughput | 1.000× | 0.654× |

This correctness-first integration is **not an acceleration result**: exact dSparK verification was 34.65% slower than target-only decoding in the matched run. Target-only decoding remains recommended for throughput. The bundled checkpoint alone does not enable speculative decoding; use a runtime with dedicated `Qwen3DSparkModel` support and revalidate exact output under the intended decoding settings.

## Source and provenance

The base model is the public OpenBMB MiniCPM-V 4.7 35B-A3B checkpoint. The bundled `dspark/` directory comes from [`openbmb/MiniCPM-V-4.7-35B-A3B-DSpark`](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B-DSpark), pinned at revision `5facb75c251ffd524524a544c1f5fd97ba0d6c5c`. Its `model.safetensors` SHA-256 is `211eda7b962795ba727b0d7db072d8c474dc3b94075f85f123fe363caf5e9679`.

## Limitations

This is a quantized model. Quantization may change output quality, including vision quality. Actual memory use, throughput, supported context length and multimodal behavior depend on the compatible runtime, hardware and final pack. Test the exact files and runtime configuration for your use case.

This model is provided without warranty. Review the upstream license and model documentation, and apply appropriate safeguards to generated outputs.
