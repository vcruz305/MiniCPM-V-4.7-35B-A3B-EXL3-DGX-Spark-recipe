# Third-party notices

- Model: OpenBMB MiniCPM-V 4.7 35B-A3B, quantization published by Victor Cruz.
  The upstream/model cards declare Apache-2.0. Model weights are downloaded
  separately and are not redistributed by this recipe. Review upstream terms.
- evidence/MODEL_CARD.md and evidence/model-config.json are attributed snapshots
  of the public vcruz305 model release; retain their source links and notices.
- dSparK: official OpenBMB checkpoint, pinned by the model publisher. Optional,
  separately downloaded; no speedup is promised.
- ExLlamaV3/EXL3: Turboderp and contributors. Engine/codec license remains its own.
- TabbyAPI: The Royal Lab and contributors, AGPL-3.0. Installed separately; this
  repository does not copy or modify its source.
- Structural reference: Victor's Qwen3.8-Flash-Next-EXL3-DGX-Spark-recipe, MIT.
  This recipe's scripts are newly written rather than copied inference code.

Original recipe scripts, tests and explanatory docs use the root MIT LICENSE.
External packages and downloaded artifacts are not relicensed by that file.
