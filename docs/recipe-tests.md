# Recipe helper validation

Validated 2026-10-13 with Python 3.10 on Linux, without loading a model or
running a GPU benchmark.

- `python3 -m unittest discover -s tests -v`: **49 tests passed**.
- `python3 -m compileall -q tools bench tests`: passed.
- `bash -n` on every `exllamav3-tabby/*.sh`: passed.

Tests cover configuration limits, target-only defaults, authentication, profile
selection, dry-run behavior, missing runtime pins, pinned manifest totals and
checksums, source snapshot hashes, complete/mock model verification, stale or
corrupted shards, optional-drafter exclusion, directory escapes, private receipt
permissions, aggregate metric accounting and local documentation links.

The GitHub Actions workflow repeats CPU checks on Python 3.12. No CUDA,
MiniCPM weights, inference calls or rented compute are used in CI. Passing these
checks does not fill the runtime publication gap or validate the new launch
profiles on a Spark or discrete GPU.
