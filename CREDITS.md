# Credits

This release is by [Aevonix Research](https://aevonix.com). It builds on the following work:

- **[TensorFold](https://github.com/ashhart/TensorFold)** by Ash Hart
  ([X @ashxhart](https://x.com/ashxhart)) and its contributors: the inference engine, EXL3 kernels,
  speculative verification and API. TensorFold is Apache-2.0; its earlier code retains MIT notices.
- **[Mia's AI Lab](https://huggingface.co/Mia-AiLab)**
  ([X @MiaAI_lab](https://x.com/MiaAI_lab)): the
  [EXL3 checkpoint](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold),
  [recipe and original patches](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-4x-DGX-Sparks-TensorFold),
  [two-Spark presentation and startup flow](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold),
  and [sparkDash](https://github.com/MiaAI-Lab/sparkDash). This release keeps 54 of Mia's patches, ported
  to TensorFold v0.6.6; patches 0023 and 0037 are superseded upstream. The v0.6.6 rebase regenerates four
  of her patches by changing hunk line positions only. Her patches remain hers. Mia's AI Lab approved publication.
- **[Z.ai](https://huggingface.co/zai-org/GLM-5.3-Flash)**: GLM-5.3-Flash, its architecture and base weights.
- **[Inco AI](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2)**: DFlash2 drafter weights,
  **CC BY-NC-ND 4.0**, non-commercial, not redistributed here. Set `DRAFTER=mtp` before the first start
  to use the checkpoint's own MTP head instead, with `PARALLEL=1`.
- **[ExLlamaV3](https://github.com/turboderp-org/exllamav3)** by turboderp: the EXL3 format (MIT).
- **[b12x](https://github.com/local-inference-lab/b12x)**: the transport design behind Mia's RoCE patch
  and the CUDA IPC buffer design that informed the local PCIe transport (Apache-2.0).
- **[glm53-tensorfold-spark](https://github.com/jayleaton/glm53-tensorfold-spark)** by Jay Leaton:
  tool-call handling, L2 prefetch and 16-byte EXL3 load ideas adapted in Mia's patches (Apache-2.0).
- **NVIDIA, PyTorch, Triton, Hugging Face, PyAV and xgrammar**: the GPU runtime, tensor and kernel stack,
  model hosting, media decoding and structured output support, under their respective licenses.
- **Geist and Geist Mono** by the Geist Project Authors: fonts used in the Aevonix Research artwork,
  under the SIL Open Font License 1.1.
- **DejaVu Sans Mono**: terminal artwork rendering, including the block and box-drawing glyphs.

Aevonix Research adds the single-host PCIe recipe and 32 patches (0000 and 0057-0087).
The [patch table](docs/PATCHES.md) describes each contribution. [NOTICE](NOTICE) preserves attribution and
modification notices; [LICENSE](LICENSE) covers this repository's code and documentation.
