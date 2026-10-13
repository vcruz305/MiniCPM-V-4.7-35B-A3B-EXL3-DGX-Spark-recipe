# Context, 24 GB GPUs and a second model on Spark

The published 19 target shards total 17,999,304,996 bytes, approximately 16.76 GiB.
This is serialized storage, not the full CUDA allocation. The optional dSparK
weights, tokenizer/index, CUDA workspace, KV and recurrent state are additional.

The publisher reports 20.33 GiB peak CUDA allocation at 8K text context and
18.33 GiB in a separate 4K text-plus-vision smoke on a Spark. Do not subtract
those numbers from a nominal 24 GB device and promise the remainder is usable:
framework allocations, display use, image resolution, batch and kernel support
can differ. A 24 GB Turing Quadro RTX 6000 is not the 96 GB RTX PRO 6000 Blackwell.
This recipe makes no Turing support claim.

Start with PROFILE=24gb, one job, 4K and no vision. Run text validation, then
try the separate vision profile. Increase context, image size and concurrency
one at a time, recording peak allocations and successful outputs. The source
config allows 262,144 positions but no such long-context run is supplied here.

For a Spark companion, use a distinct API port and state directory. Both models'
weights, KV/recurrent state, vision buffers, page cache and OS must fit in the same
unified memory. Free memory cannot be calculated from weight-file sizes alone.
Do not automatically evict, unload, kill or restart the primary model.

The optional GPU_LOCK_FILE is cooperative exclusion for workflows requiring
exclusive use. Use the same lock path in other supervisors for it to be effective.
It is not a memory allocator or a way to partition the GPU between servers.
