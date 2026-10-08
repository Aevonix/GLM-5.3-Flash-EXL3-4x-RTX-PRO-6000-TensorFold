# Measurement and validation notes

This page records the TensorFold v0.6.6 validation and retains historical v0.6.2 measurements.
Current engine details are in [the v0.6.6 rebase notes](PORT-v0.6.6.md).

The published performance measurements describe one PCIe host with four RTX PRO 6000 Blackwell Max-Q GPUs
at **250 W per GPU**. They do not predict another topology. [README](../README.md) reports the production
FP8 configuration; [PATCHES](PATCHES.md) records historical component measurements, many from the q4
optimization sweeps.

## TensorFold v0.6.6

The recipe now runs TensorFold v0.6.6 at `cb2ebf0540f42604e2759b2ddef497861e928248` with 86 patches.
The published performance measurements come from the v0.6.5 build. Older functional and dense-fidelity
scores remain attributed to v0.6.2.

The 2026-10-08 A/B check compared A, the current v0.6.5 build, with B, v0.6.6, using the same checkpoint
and settings on four RTX PRO 6000 Blackwell Server Edition GPUs at 600 W each. This was a native A/B
regression check on a different host from the published Max-Q measurements.

- Gate and gatelong results passed identically through concurrency levels 4, 8, 16, 32 and 40. Long prompts
  covered approximately 8K, 32K and 100K tokens. The image probe also passed identically.
- First tokens matched in 40/40 probes. Complete returned token sequences also matched in 40/40 probes.
- Tool bursts matched A: default choice scored 80/90 and required choice scored 90/90 on both arms.
  The ten shared default-choice misses remain functional misses.
- Decode and cold-prefill speed stayed within about 1%, with no systematic regression. This does not
  replace the published Max-Q throughput measurements.
- The `--name-priority ID=background` smoke passed. Omitted request priority used the model-name
  background default; explicit normal and realtime priorities still won. The merge with our priority
  lanes preserves explicit priorities and title-request behavior.

Strict CPU replay of all 86 patches produced the expected tree
`9494ac6d2099ffe7fd9e33bfb412905ea5dc0f08`, with no fuzz, offsets, `.orig` or `.rej` files.
All 31 standalone CPU priority cases passed. The upstream name-priority pytest module was skipped
because `tokenizers` was unavailable. The [rebase notes](PORT-v0.6.6.md) record the six regenerated
patches and preserved ownership.

## Reproduce the current patch tree without GPUs

Use the retained check with an independent v0.6.6 release patch export and a local, unmodified upstream clone:

```bash
python3 tools/check_apply.py --release-patches /path/to/reference-patches \
  --upstream /path/to/unmodified-TensorFold
```

The check reads the reference and upstream clone without changing them, applies both series to temporary
trees and compares their files and modes. No network or GPU is required. Without `--upstream`, it verifies
patch identity only. Current replay manifests use one sorted line per file: file SHA-256, two spaces,
relative POSIX path and newline, excluding Git metadata. File modes are checked separately.

- Entire upstream checkout: 999 files, manifest `6c92449163454ad349dd7404ed26f15554b04219e14d7a929ba3090b2d7a47d3`.
- `src/tensorfold/`: 534 files, manifest `5873855ae27f410e7bb0ad2d3a94c41dba73c03703398e473aff6b167d7ccdca`.

## Compare the same workload

To compare engines, use the same client, prompts, token budget, sampling settings and weights. Keep aggregate
decode separate from per-request speed, and decode-only speed separate from time to first token. Cold prefill
and cached warm turns are different workloads.

Sampled requests that omit `top_k` are not equivalent across engines: TensorFold serves an omitted `top_k` as 20,
while other servers may apply no limit. Send `top_k` explicitly in every sampled comparison, and state the
effective top-k setting with any sampled TensorFold result.

Exactness references belong to a checkpoint, precision and prompt arithmetic. Check drafted against
`"draft": false`, concurrent against solo, and long prompts as well as short prompts. FP8/q4 dense, FP8 KV,
and chunked KDA need their own references. Matching those references does not establish BF16 fidelity.
The quality figures are small functional checks, not a broad model-quality evaluation. The recorded
qualification used a modified drafter for some tool/image and stress checks; stock DFlash2 was restored
for the final production performance run and its 40-stream exactness gate. That qualification's JSON
subtest scored 15/20; separate production-default JSON probes scored 20/20 both plain and with
`response_format`. These were distinct tests.

## Lessons behind the defaults

- **Measure each transport operation.** Fast peer copies do not imply fast NCCL send/receive. The measured
  combination uses P2P NCCL collectives, a separate shared-memory send/receive communicator, and copy engines
  for large prompt exchanges. General decode all-gathers stay on NCCL. Changing IOMMU settings did not cure
  slow SM peer stores; the scripts do not change system settings.
- **Measure settled clocks.** At a sustained power cap, interleave kernel candidates after clocks settle.
  K12-p80 measured 1.10-1.12x the baseline prompt-expert speed at 250 W. Isolated KDA recurrence improved
  4.3x, but whole-engine gains were smaller. EXL3's decoded 16-bit prompt math also differs from NVFP4 W4A4.
- **Measure useful tokens per round.** A faster drafter or higher offline acceptance does not establish
  faster serving. Wider windows and cheaper exchange helped; 192/256-row windows did not improve on 128
  in the recorded sweep and used more memory. The production window is 128.
- **Preserve correctness guards.** Keep first-use bit checks and fallback kernels. FP8/BF16 dense requires
  `TF_GLM_LANE_INPUTS=0`. Patch 0083 keeps rank-local stop strings or disconnections from changing collective
  participation. Recheck stop/disconnect stress after changing draft-ahead.
- **Profile carefully.** Synchronous timers distort overlap. Live per-node CUDA graph tracing stalled a rank
  in the study; use engine event profiles or an isolated kernel profiler. Retain the fixed streamed-expert
  copy form; its earlier cache-hinted form produced a GPU fault.

300 W was faster through 16 streams, but the Max-Q cards overheated in the tested chassis. No complete
300 W sweep, cold/warm latency stage or sparkDash stage finished. The published baseline remains 250 W.

## Historical TensorFold v0.6.2 patch replay

The v0.6.2 release used **87 patches**, 55 from Mia's AI Lab and 32 from Aevonix Research, applied in global
filename order to TensorFold v0.6.2 at `56e2e3ec55bc0ae1d7d5158c4fa2c79a3567ab21`.
The full application check was rerun locally for that release on 2026-10-03, using the pinned local
upstream and an independent release patch export, without network or GPU access. It produced no fuzz,
offsets, `.orig` or `.rej` files. The two applied trees matched every file and file mode. The SHA-256 manifest convention is
one sorted line per file: file SHA-256, two spaces, relative POSIX path, newline; Git metadata excluded.

- Entire upstream checkout: 904 files, manifest `f6366817109497461944a3540c2e5cf457c2c51f95ea1ac8a72cdb92206efda1`.
- `src/tensorfold/`: 502 files, manifest `48c3dcd407ec834a84c8936c7ed9b2ce99c708b115c4e11d350f5ea5f5231fcd`.

At the time of that packaging check, the launcher and build integration still needed a real four-GPU
first-run test. The later v0.6.5 flow is recorded in [its port notes](PORT-v0.6.5.md#release-validation).
