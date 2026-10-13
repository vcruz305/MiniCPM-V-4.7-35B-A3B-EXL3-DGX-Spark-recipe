# Installation

## Install the current forks

Requires Linux, `git`, a CUDA-enabled Python with PyTorch/Triton,
`nvcc` and C++ compiler. DGX Spark requires an aarch64-compatible
CUDA/Torch environment, not an x86 wheel.

```bash
bash exllamav3-tabby/setup.sh

# If needed, point at an existing working GPU Python:
BASE_PYTHON=/absolute/path/to/cuda-venv/bin/python \
  bash exllamav3-tabby/setup.sh
```

Setup resolves both forks' latest default branch, creates a
separate managed venv, builds ExLlamaV3 and installs TabbyAPI.
It preserves the selected base PyTorch/Triton build; it does not
install GPU drivers, modify unrelated checkouts, or start serving.

`MAX_JOBS=4` controls build parallelism. GPU architecture is
detected automatically, with `TORCH_CUDA_ARCH_LIST` as an override.

## Updates

```bash
# Install latest code if it changed:
bash exllamav3-tabby/setup.sh

# Check for changes without an update:
bash exllamav3-tabby/setup.sh --check
```

Every new `serve.sh` start also checks for fork changes. An unchanged
installation is reused. When code changes, a new build is staged;
failed updates do not overwrite an active installation or silently
fall back to an old one. Existing server processes stay running
until their owner stops them.

Do not set `EXLLAMAV3_REF` or `TABBY_REF`; these old pin variables
are explicitly rejected. Runtime commits in local receipts are
diagnostic facts, not version constraints.

`RECIPE_HOME` defaults to `~/minicpm47-exl3`. Private runtime
receipts and API keys are under `STATE_DIR` (default `.state`).
Separate live services need separate homes and state directories.

## Existing qualified environment

Optional read-only recording of an existing MiniCPM environment:

```bash
bash exllamav3-tabby/setup.sh --adopt \
  --python /path/to/venv/bin/python \
  --tabby-src /path/to/tabbyAPI
```

Normal setup and serve still install latest forks into a managed
environment; this does not force old versions to stay installed.

## Model download

```bash
bash exllamav3-tabby/download.sh
bash exllamav3-tabby/download.sh --verify-only
bash exllamav3-tabby/download.sh --include-drafter
```

The separate quant file manifest is integrity-checked against
the published release. It does not pin inference software.
Downloading the optional dSparK does not enable acceleration.

Download-only without a GPU environment:

```bash
python3 -m venv .download-venv
.download-venv/bin/python -m pip install huggingface_hub
DOWNLOAD_PYTHON="$PWD/.download-venv/bin/python" \
  bash exllamav3-tabby/download.sh
```
