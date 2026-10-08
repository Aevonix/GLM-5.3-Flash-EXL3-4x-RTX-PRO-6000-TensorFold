# TensorFold v0.6.6 patch rebase

The current base is TensorFold v0.6.6 at `cb2ebf0540f42604e2759b2ddef497861e928248`.
This is a rebase only. All 86 filenames, header lines and patch order are preserved.
Mia's AI Lab retains credit for her 54 patches. The shared patches were regenerated once and reused
in both recipes. Eighty patch files remain byte-for-byte unchanged.

| Regenerated patch | Rebase change |
| --- | --- |
| `0003-glm-vision.patch` | Hunk line positions only. |
| `0036-glm-tool-calls.patch` | Hunk line positions only. |
| `0053-glm-whole-tool-calls.patch` | Hunk line positions only. |
| `0054-glm-tp-n.patch` | Hunk line positions only. |
| `0066-glm-priority-lanes.patch` | Hunk positions and context, plus the priority merge described below. |
| `0079-glm-warm-turn-incremental.patch` | Hunk positions and context around upstream's new `self.background_ids`. |

In `src/tensorfold/cuda/server.py`, the `App.run` conflict keeps both upstream's model-name background
default and the recipe's priority lanes. It passes `default=prio.BACKGROUND if by_name else None` to
`prio.request_class`. Explicit request priorities and title-request handling retain their precedence.
The later upstream comment change is retained with the same combined behavior.

Neither source revision contains optional patch files. Current digests are in
[the SHA256 table](PATCH-SHA256.md). The [v0.6.5 release validation](PORT-v0.6.5.md#release-validation) remains historical.

Each recipe's strict replay produced the expected Git tree
`9494ac6d2099ffe7fd9e33bfb412905ea5dc0f08`: its old patched tree plus upstream's three commits.
The recipe replay check, source compilation, all seven shell syntax checks and `DRY_RUN=1 ./start.sh`
passed for each recipe. A separate `App.run` check passed all 31 CPU priority cases for each recipe.
The upstream name-priority pytest module was skipped because `tokenizers`
was unavailable (pytest exit 5). No packages were installed. These checks used CPU only.

## Server name priorities

Upstream adds `--name-priority ID=background`. It assigns background priority to requests for the
matching served model ID only when the request omits `priority`. Patch 0066 passes that default into
our priority lanes. Explicit request priorities still win, title requests stay background, and the
environment default remains in effect when no model-name default is selected. Request bodies are unchanged.

## GPU A/B validation

On 2026-10-08, A (v0.6.5) and B (v0.6.6) used the same checkpoint and settings on four RTX PRO 6000
Blackwell Server Edition GPUs at 600 W each. Gates and long-context gates passed identically.
First tokens and complete returned sequences matched in 40/40 probes. Tool bursts were unchanged:
default choice 80/90, required choice 90/90. The ten shared default-choice misses remain misses.
The image probe passed identically. Speed stayed within about 1%, with no systematic regression.
The name-priority smoke passed, including explicit normal and realtime overrides.

These native checks used a different host from the Max-Q performance measurements. Published performance
measurements remain attributed to the v0.6.5 build. See [validation notes](NOTES.md#tensorfold-v066).
