# Recipe tests

CPU-only checks, not CUDA/model qualification:

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall -q tools bench tests
for f in exllamav3-tabby/*.sh; do bash -n "$f"; done
```

Tests cover config/profile validation, safe offline preview,
remote default-branch parsing, refusal of fixed runtime refs,
model checksum verification, corrupted/stale shard rejection,
diagnostic receipt protections and benchmark metric integrity.

GitHub Actions repeats these helper checks. A passing suite is
not a claim that the new installation was GPU-benchmarked.
