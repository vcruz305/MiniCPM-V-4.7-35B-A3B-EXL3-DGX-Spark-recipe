# Installation

## A. Existing MiniCPM-enabled environment (recommended until runtime is published)

Do not replace a working installation with a generic wheel. Point the recipe at
its Python executable and its TabbyAPI source tree:

```bash
bash exllamav3-tabby/setup.sh --adopt \
  --python /path/to/working/venv/bin/python --tabby-src /path/to/tabbyAPI
bash exllamav3-tabby/setup.sh --check
bash exllamav3-tabby/download.sh
```

Adoption imports the runtime, checks its registered architecture and CUDA
availability, hashes its source, and records selected dependency versions. It
never runs pip or modifies the existing checkouts. The private receipt is stored
under `.state/` (or `STATE_DIR`). The model is not loaded during adoption.

## B. Fresh source build (requires the missing MiniCPM commits)

There is deliberately no guessed default ExLlamaV3 pin. Supply exact 40-character
source commits once the compatible runtime code is public. BASE_PYTHON must
already have a suitable CUDA-enabled PyTorch; use the hardware vendor's supported
Torch environment, especially on DGX Spark/aarch64.

```bash
BASE_PYTHON=/path/to/cuda-python \
EXLLAMAV3_REF=<40-character-MiniCPM-enabled-commit> \
TABBY_REF=<40-character-compatible-TabbyAPI-commit> \
  bash exllamav3-tabby/setup.sh --build
```

This creates a separate venv inheriting the existing CUDA Torch, clones source
into a recipe-owned location, checks for the model architecture before building,
and installs TabbyAPI without CUDA extras. Torch and existing Triton versions are
constrained during installation. `pip check` must pass. It does not use
TabbyAPI's auto-updating start.sh or enable developer.unsafe_launch.

`MAX_JOBS=4` limits build parallelism. `TORCH_CUDA_ARCH_LIST` is derived from the
selected CUDA GPU unless explicitly set (GB10 is 12.1). `nvcc` and a working C++
compiler are required. The recipe does not install system drivers or repair
binary wheels. Stop if Torch, CUDA toolkit or compiler compatibility is wrong.

Build configuration may be taken from runtime.lock.json only after qualified
commits are populated. Source checks are not performance qualification.

## C. Download/verify without a runtime

The downloader itself needs Python and huggingface_hub, not CUDA:

```bash
python3 -m venv .download-venv
.download-venv/bin/python -m pip install huggingface_hub
DOWNLOAD_PYTHON="$PWD/.download-venv/bin/python" \
  bash exllamav3-tabby/download.sh
```

Only a pinned snapshot is requested. The manifest verifies Git-blob hashes for
small files and SHA-256 for LFS-backed files. `--verify-only` uses the Python
standard library and works without huggingface_hub. `--include-drafter` also
verifies the optional pinned dspark checkpoint. Files in the root `.cache/` are
Hub bookkeeping and are not model shards.

Downloading the optional drafter is not a recommendation to enable it.
