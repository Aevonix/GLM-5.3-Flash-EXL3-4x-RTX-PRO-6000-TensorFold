# Changelog

## 1.2.0 - 2026-10-08

- Rebase the 86-patch series onto TensorFold v0.6.6.
- Merge the new `--name-priority ID=background` server option with our priority lanes. Explicit priorities and title requests keep their existing behavior.
- Regenerate six patches for the rebase. Mia's AI Lab's four regenerated patches change only hunk positions and remain hers. Eighty patches are byte-identical.
- Verify strict replay against the expected tree and pass 31 CPU priority cases. On four RTX PRO 6000 Blackwell Server Edition GPUs, matched v0.6.5/v0.6.6 gates, long-context gates, first tokens (40/40), image input and tool bursts agree. Speed is within about 1%, with no systematic regression. The name-priority smoke passed.
- Retain published performance measurements from the v0.6.5 build and older functional scores as labeled. See [validation notes](docs/NOTES.md#tensorfold-v066).

## 1.1.0 - 2026-10-04

- Port the patch series to TensorFold v0.6.5 while preserving served defaults and the runtime header installation fix.
- Keep the unified communicator, API authentication and metrics behavior compatible with the recipe.
- Refresh benchmark facts and artwork from the release validation window.

## 1.0.1 (2026-10-03)

- Image build: copy TensorFold's C/CUDA headers into the installed package. The pip install left out `cuda/ipc.h`,
  so the CUDA IPC extension could not compile and the index split's gathers fell back to NCCL. The reported
  measurements used images that had the header. `./start.sh` rebuilds the image once after this update.

## 1.0.0 (2026-10-03)

- First public release by Aevonix Research for one Linux host with four RTX PRO 6000 Blackwell GPUs.
- One-command setup and launch: checks, pinned image build and model downloads, kernel compilation,
  launch-table autotuning, API readiness and a smoke test. Restart, dry run and ten saved log archives.
- TensorFold v0.6.2 with 87 patches: 55 from Mia's AI Lab, ported to v0.6.2, and 32 from Aevonix Research.
- Measured defaults: FP8 dense and KV, stock DFlash2, `fnc7:0.3`, 128-row verify window,
  40 concurrent requests, 1,048,576-token context and port 8020.
- Single-host PCIe collectives, prompt lanes and kernels, batched drafting, priority scheduling,
  incremental warm turns and TensorFold v0.6.2 compatibility fixes. See [every patch](docs/PATCHES.md).
- Greedy sparkDash comparison, separate cold and warm latency measurements, dense-format fidelity results,
  and retained tools for independent speed and exactness checks.
