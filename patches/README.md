# TensorFold v0.6.6 patches

The 86 patches are stored once, by origin: 54 in [miaai-lab/](miaai-lab/) and 32 in
[aevonix/](aevonix/). They apply in global filename order, 0000 through 0087 with 0023
and 0037 omitted because TensorFold already includes their behavior.

Six patches were regenerated for v0.6.6. Mia's four change only hunk line positions and remain hers.
Patch 0066 merges upstream's model-name background defaults with the recipe's priority lanes.
Explicit priorities and title requests keep their behavior. See [the rebase notes](../docs/PORT-v0.6.6.md).

`./start.sh` builds them into the image. To apply just the patches to a fresh local
TensorFold v0.6.6 checkout:

```bash
scripts/apply-patches.sh /path/to/TensorFold
```

With no argument the helper clones v0.6.6 and verifies commit
`cb2ebf0540f42604e2759b2ddef497861e928248`. `--list` prints the ordered paths.
Application rejects fuzz, offsets, `.rej` and `.orig` files.

See [the patch table](../docs/PATCHES.md), [validation notes](../docs/NOTES.md),
[NOTICE](../NOTICE) and [LICENSE](../LICENSE).
