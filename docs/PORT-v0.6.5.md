# TensorFold v0.6.5 patch port

This page records the historical v0.6.5 port. The recipe now runs [TensorFold v0.6.6](PORT-v0.6.6.md).

Pinned upstream: `609ca419abecebdc5a059498a613680bd3aa847f`. The 87-patch source series becomes 86 patches: 63 unchanged, 23 adapted, one supplied upstream. Patch numbers remain stable.

The image still copies runtime C/CUDA headers. Served defaults retain thinking off, the checkpoint's earlier-turn clearing (`TF_GLM_CLEAR_THINKING=1`), and unauthenticated routes unless `TENSORFOLD_API_KEY` is explicitly configured. With a key, `/health` stays open and `/metrics` requires authentication.

| Patch | Status | Reason |
| --- | --- | --- |
| 0000-glm-exl3-route.patch | unchanged | Applies byte-for-byte without offsets. |
| 0001-glm-exl3-prompt-experts.patch | unchanged | Applies byte-for-byte without offsets. |
| 0002-glm-dense-fp8.patch | unchanged | Applies byte-for-byte without offsets. |
| 0003-glm-vision.patch | adapted | Merge GLM image/video frontend with upstream tool-result images (4825c01), video input (6091427), visual-token budgets (362610f), and clear_thinking (ad15127). Keep GLM frame sampling and recipe media byte limits. |
| 0004-glm-prompt-kernels.patch | adapted | Rebased hunk context/line positions; three-way merge preserves both changes. |
| 0005-glm-dense-q4.patch | unchanged | Applies byte-for-byte without offsets. |
| 0006-cuda-roce-allgather.patch | adapted | Rebased hunk context/line positions; three-way merge preserves both changes. |
| 0007-glm-copy-drafts.patch | unchanged | Applies byte-for-byte without offsets. |
| 0008-glm-prompt-grid.patch | unchanged | Applies byte-for-byte without offsets. |
| 0009-glm-prefill-kernels.patch | unchanged | Applies byte-for-byte without offsets. |
| 0010-glm-hc-split.patch | adapted | Keep prompt row exchanges, upstream communicator protocol/dtypes, timeout_s and rendezvous key (c3d92bb). Final asymmetric NCCL compatibility is retained by 0087. |
| 0011-glm-draft-sim.patch | unchanged | Applies byte-for-byte without offsets. |
| 0012-glm-kda-chunked.patch | unchanged | Applies byte-for-byte without offsets. |
| 0013-glm-decode-rounds.patch | adapted | Rebased hunk context/line positions; three-way merge preserves both changes. |
| 0014-glm-kda-chunked-gb10.patch | unchanged | Applies byte-for-byte without offsets. |
| 0015-glm-shared-prefix.patch | unchanged | Applies byte-for-byte without offsets. |
| 0016-glm-decode-kernels.patch | unchanged | Applies byte-for-byte without offsets. |
| 0017-glm-overlap-priority.patch | unchanged | Applies byte-for-byte without offsets. |
| 0018-glm-noise-policies.patch | unchanged | Applies byte-for-byte without offsets. |
| 0019-glm-decode-kernels2.patch | unchanged | Applies byte-for-byte without offsets. |
| 0020-glm-prompt-experts-order.patch | unchanged | Applies byte-for-byte without offsets. |
| 0021-glm-dflash-policy-env.patch | unchanged | Applies byte-for-byte without offsets. |
| 0022-glm-overlap-normal-priority.patch | unchanged | Applies byte-for-byte without offsets. |
| 0024-glm-prompt-select-rows.patch | unchanged | Applies byte-for-byte without offsets. |
| 0025-glm-dflash2-ring.patch | unchanged | Applies byte-for-byte without offsets. |
| 0026-glm-multi-kda.patch | unchanged | Applies byte-for-byte without offsets. |
| 0027-glm-multi-dflash2.patch | unchanged | Applies byte-for-byte without offsets. |
| 0028-glm-lean-prompt-scratch.patch | adapted | Rebased hunk context/line positions; three-way merge preserves both changes. |
| 0029-glm-multi-dsa.patch | unchanged | Applies byte-for-byte without offsets. |
| 0030-glm-multi-stream-engine.patch | adapted | Rebased hunk context/line positions; three-way merge preserves both changes. |
| 0031-glm-decode-index-regs.patch | unchanged | Applies byte-for-byte without offsets. |
| 0032-glm-code-copy-drafts.patch | unchanged | Applies byte-for-byte without offsets. |
| 0033-glm-prefill-overlap2.patch | unchanged | Applies byte-for-byte without offsets. |
| 0034-cuda-nucleus-union.patch | unchanged | Applies byte-for-byte without offsets. |
| 0035-glm-multi-rounds.patch | unchanged | Applies byte-for-byte without offsets. |
| 0036-glm-tool-calls.patch | adapted | Combine recipe tool-call streaming/history recovery with upstream clear_thinking (ad15127), stop-sequence reporting (e9fade0), and max_tokens-inside-thinking warning (bcb8f01). |
| 0037-cuda-tokenize.patch | dropped-upstream | 8f5e680 (integrated by 45325f7): CUDA token-id prompts, tokenize/detokenize aliases, special-token and generation flags, vocabulary validation; upstream also supports return_token_strs. Only redundant unused helper/constants remained. |
| 0038-glm-kv-fp8.patch | adapted | Rebased hunk context/line positions; three-way merge preserves both changes. |
| 0039-glm-kda-chunked-kernel.patch | unchanged | Applies byte-for-byte without offsets. |
| 0040-glm-parallel-deadlocks.patch | unchanged | Applies byte-for-byte without offsets. |
| 0041-glm-parallel-ring-base.patch | unchanged | Applies byte-for-byte without offsets. |
| 0042-glm-prompt-replay.patch | unchanged | Applies byte-for-byte without offsets. |
| 0043-glm-visible-pools.patch | unchanged | Applies byte-for-byte without offsets. |
| 0044-cuda-context-errors.patch | adapted | Upstream 8f5e680 already supplies text context errors; retain the GLM vision context_length_exceeded wording added by this patch. |
| 0045-cuda-metrics.patch | adapted | Keep live GLM pool/health metrics and snapshot guards alongside upstream per-request decode counters (c9259c3), process footprint (0a57d1f), and HTTP/auth counters (bcb8f01). |
| 0046-glm-l2-prefetch.patch | unchanged | Applies byte-for-byte without offsets. |
| 0047-glm-exl3-decode-loads.patch | unchanged | Applies byte-for-byte without offsets. |
| 0048-glm-timing-tokens.patch | unchanged | Applies byte-for-byte without offsets. |
| 0049-glm-multi-prefill.patch | unchanged | Applies byte-for-byte without offsets. |
| 0050-glm-many-media.patch | adapted | Keep custom GLM image/video loaders and 96 MiB request limit; use upstream chunked-body reader (dc8e9eb) with an explicit limit and preserve configurable visual-token budgets. |
| 0051-glm-tool-history-recovery.patch | adapted | Rebased hunk context/line positions; three-way merge preserves both changes. |
| 0052-cuda-roce-startup.patch | unchanged | Applies byte-for-byte without offsets. |
| 0053-glm-whole-tool-calls.patch | adapted | Rebased hunk context/line positions; three-way merge preserves both changes. |
| 0054-glm-tp-n.patch | adapted | Pass recipe world size to upstream open_comm (7933ed6); retain arbitrary local rank counts and rank-ordered reductions. |
| 0055-glm-tp3-split-pad.patch | unchanged | Applies byte-for-byte without offsets. |
| 0056-glm-tpn-split-buffer-rows.patch | unchanged | Applies byte-for-byte without offsets. |
| 0057-glm-parallel-cap.patch | adapted | Rebased hunk context/line positions; three-way merge preserves both changes. |
| 0058-cuda-ipc-allgather.patch | adapted | Rebased hunk context/line positions; three-way merge preserves both changes. |
| 0059-cuda-nccl-exchange-comm.patch | adapted | Second NCCL exchange communicator remains isolated; preserve upstream timeout_s and key parameters (c3d92bb), derive exchange key from the supplied rendezvous key. |
| 0060-glm-index-split.patch | unchanged | Applies byte-for-byte without offsets. |
| 0061-glm-prefill-digest.patch | unchanged | Applies byte-for-byte without offsets. |
| 0062-glm-hc-exchange-ce.patch | unchanged | Applies byte-for-byte without offsets. |
| 0063-glm-prefill-lanes.patch | unchanged | Applies byte-for-byte without offsets. |
| 0064-glm-launch-tables.patch | unchanged | Applies byte-for-byte without offsets. |
| 0065-gpu-sampling.patch | unchanged | Applies byte-for-byte without offsets. |
| 0066-glm-priority-lanes.patch | adapted | Keep priority scheduling while using upstream token_routes and flag helpers; avoid restoring helpers made redundant by dropped 0037. |
| 0067-glm-segment-grids-by-rows.patch | unchanged | Applies byte-for-byte without offsets. |
| 0068-glm-kept-prompt-arrays.patch | unchanged | Applies byte-for-byte without offsets. |
| 0069-glm-decode-fusions.patch | unchanged | Applies byte-for-byte without offsets. |
| 0070-cuda-ipc-allgather-protocols.patch | unchanged | Applies byte-for-byte without offsets. |
| 0071-glm-prefill-2.patch | adapted | Retain copy-engine exchange and draft-prefill optimization; keep caller-supplied communicator rendezvous key. |
| 0072-glm-kda-split.patch | unchanged | Applies byte-for-byte without offsets. |
| 0073-glm-expert-prompt-kernels.patch | unchanged | Applies byte-for-byte without offsets. |
| 0074-glm-moe-glue.patch | unchanged | Applies byte-for-byte without offsets. |
| 0075-glm-draft-fast.patch | unchanged | Applies byte-for-byte without offsets. |
| 0076-glm-prefill-3.patch | unchanged | Applies byte-for-byte without offsets. |
| 0077-glm-wide-windows.patch | adapted | Rebased hunk context/line positions; three-way merge preserves both changes. |
| 0078-glm-round-cap-twoshot.patch | unchanged | Applies byte-for-byte without offsets. |
| 0079-glm-warm-turn-incremental.patch | adapted | Keep warm-turn encoding reuse with upstream token-route validation helpers, including return_token_strs. |
| 0080-glm-topk-sparse-ws.patch | unchanged | Applies byte-for-byte without offsets. |
| 0081-glm-lane-inputs.patch | unchanged | Applies byte-for-byte without offsets. |
| 0082-glm-lane-partials.patch | unchanged | Applies byte-for-byte without offsets. |
| 0083-glm-draft-ahead-shared-ends.patch | unchanged | Applies byte-for-byte without offsets. |
| 0084-glm-decisions-tp-n.patch | unchanged | Applies byte-for-byte without offsets. |
| 0085-glm-port-settings-check.patch | unchanged | Applies byte-for-byte without offsets. |
| 0086-glm-port-lane-exl3-view.patch | unchanged | Applies byte-for-byte without offsets. |
| 0087-glm-port-stage2-routes.patch | adapted | Preserve GLM route guards and seed salt; retain upstream c3d92bb asymmetric NCCL exchanges, while optional copy-engine exchanges require symmetric buffers. |

## Release validation

The fresh-clone preparation/start/smoke/check/stop flow passed on four RTX PRO 6000 GPUs at 250 W each. Drafted/serial and solo/concurrent token hashes matched the v0.6.2 references, including sampled and greedy concurrency through 40 requests, tools, image input, and 8K/32K/100K long prompts. A conversation replay after restart matched its cached continuation.

The capacity check accepted 1,048,568 prompt tokens plus an eight-token reply budget and returned two tokens. The live defaults probe confirmed thinking off and earlier-turn reasoning clearing. Startup logs confirmed CUDA IPC for the copy-engine exchanges and index gathers.

The README retains the first user-flow benchmark sweep, including outliers. Paired v0.6.2/v0.6.5 repeats found similar steady prefill/decode performance. The large first-use code TTFT spike also occurred on v0.6.2; larger single decode-cell differences did not persist across both repeats. Small warm-turn latency differences remain dependent on sampled replies and evolving histories. Historical quality and dense-fidelity scores in the README are explicitly labeled and were not rerun for this port.

## Rebase to v0.6.6

The current rebase, regenerated patches and priority integration are recorded in
[the v0.6.6 rebase notes](PORT-v0.6.6.md). The port table and release validation above describe v0.6.5.
