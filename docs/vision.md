# Vision

The pack retains an **8-bit** vision path, not BF16 vision. The publisher's native
ExLlamaV3 check used two synthetic colors, finite distinct 64x2048 FP16 embeddings,
and correct one-word red/blue answers. It does not establish a complete OCR,
document, video or multimodal quality score, nor the OpenAI image-input route.

With a MiniCPM-enabled runtime adopted, select the smaller image profile:

```bash
PROFILE=vision bash exllamav3-tabby/serve.sh
python3 bench/smoke.py --vision
```

The smoke helper creates red and blue PNGs in memory, sends standard image_url
data URLs, and records actual answers. No external image URL is fetched and no
model-specific Python from the Hub is executed by the client. It does not execute
any returned code. Check that the server actually reports vision enabled; some
TabbyAPI versions warn and continue text-only if the architecture has no vision
component. This recipe's model-load precheck rejects that case when vision is on.

Passing two color questions is only a smoke test. Next validate varied documents,
small text, OCR, multi-image ordering, prompt/image alignment and any intended
video path. Do not quote the original OpenBMB multimodal benchmark as a score of
this 8-bit-vision EXL3 release.
