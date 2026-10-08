#!/usr/bin/env python3
"""Launch-table autotune harness for TensorFold's GLM-5.3-Flash CUDA kernels: pick, per kernel, shape and row count,
the fastest launch configuration among those whose outputs equal the default configuration's BIT FOR BIT, on the GPU
it runs on, and write them to the launch table the engine reads at start (TF_GLM_TUNE,
``tensorfold/families/glm5_next/cuda/tune.py``), as that GPU model's table (other GPUs keep the default launches).

Exactness first: every candidate runs through the engine's own code path (the table lookup is overridden for the call),
on several random inputs and at the smallest and largest row count of each row bucket, and its outputs are compared
byte for byte (raw bits: -0 is not +0, NaN payloads count) with the default configuration's. Only candidates that match
on every input are timed; a candidate that differs once is rejected for that whole shape. Before that, the default
configuration itself is checked against an independent implementation where the engine has one (EXL3: the grouped
kernel path TF_GLM_EXL3_DEC=0; KDA: the fused one-block chain), so the reference is the engine's bits; for the Q4
tiles the reference is the tile the engine uses today (the stock tile, or the GB10 table's for its shapes). Output buffers start from a sentinel pattern before every run, so a candidate that leaves an output unwritten is
caught.

Timing: L2 flushed before every run (a forward reads each weight once, from DRAM), the run's activations re-read into
L2, CUDA events around the run only, median of --reps runs. A winner must beat the default by --min-gain (2%) and by
more than the runs' spread; otherwise the default stays.

Runs where the engine runs (it imports tensorfold and JIT-builds its kernels into the extension cache, which the
engine then reuses), on ONE idle GPU, with the engine stopped (``./start.sh`` does it in a container).

Kernels (``--kernels``):
  exl3_dec     decode routed experts (gate/up and down kernels): 16-byte loads 0/1/2/4 steps ahead, 1/2/4 warps along
               N, fused or separate epilogues, one rotated input a row or a pair; per row bucket up to 63 rows
  q4_dec       4-bit dense matmuls of decode windows: the 16 tile configs of qmm.cu (and the stock tile), per shape and
               row bucket up to 64 rows
  q4_prefill   4-bit dense matmuls of prompt chunks: the 12 prefill tiles and patches 0195 / 0197's 24 variants
               (tiles 32-55), per shape (and per row bucket with several --prefill-rows, e.g. 256,512,2048,4096)
  f8_dec       the FP8 head (and other FP8 matmuls) of decode windows: Triton column tile, warps, stages
  b16_dec      BF16 matmuls of decode windows (only shapes from --shapes; none in the q4 recipe)
  kda_step     the KDA chain's step kernel: warps a block (blocks a head) and staged rows, per row bucket
  exl3_prompt  prompt EXL3 experts: 64- or 128-member passes, 2-4 cp.async pipeline stages
  sparse_scores  the multi-stream indexer's score programs a segment (a grid stride: any count, the same bits)
  l2pf         not measured: --l2pf-wave writes one_wave = 4 x the GPU's SMs (GB10's 192 = 4 x 48)

Shapes: --shapes <file> (TF_GLM_TUNE_RECORD of one engine start plus a few requests: exactly what runs), else derived
from --config (the checkpoint's config.json) for --tp / --rank (the GLM-5.3-Flash layouts of tp.py, DENSE=q4).

Licensed under the Apache License, Version 2.0. Builds on TensorFold (Ash Hart and the TensorFold contributors) and on
the GLM-5.3-Flash recipe and patches 0001-0056 by Mia's AI Lab."""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import platform
import statistics
import sys
import time
from typing import Any, Callable

HARNESS = "glm_autotune 1"
DEC_BUCKETS = (1, 2, 4, 8, 16, 32, 63)      # decode window rows (upper bounds of the row buckets)
SENTINEL = 0x7F                             # bytes outputs start from (fp16 NaN, fp32 3.4e38): an unwritten one shows


# ---------------------------------------------------------------------------------------------------- utilities ---
def log(msg: str) -> None:
    print(f"[autotune {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def bits(t):
    """A tensor's raw bytes (for bitwise comparison: torch.equal on floats says -0 == +0 and NaN != NaN)."""
    return t.detach().contiguous().view(-1).view(__import__("torch").uint8)


def same_bits(a, b) -> bool:
    import torch

    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(bits(a), bits(b))


def first_diff(a, b) -> str:
    import torch

    if a.shape != b.shape or a.dtype != b.dtype:
        return f"shape/dtype {tuple(a.shape)} {a.dtype} vs {tuple(b.shape)} {b.dtype}"
    x, y = bits(a), bits(b)
    idx = torch.nonzero(x != y)
    n = idx.numel()
    return f"{n} of {x.numel()} bytes differ (first at byte {int(idx[0]) if n else -1})"


def sentinel(*ts) -> None:
    """Fill whole (contiguous) output buffers with the sentinel bytes."""
    for t in ts:
        if not t.is_contiguous():
            raise ValueError("sentinel: contiguous buffers only")
        bits(t).fill_(SENTINEL)


def buckets_of(rows: list[int], bounds: tuple[int, ...]) -> list[tuple[int, int]]:
    """(lowest, highest) rows of each bucket whose upper bound is in ``rows``: a bucket starts after the previous
    listed bound, as the engine's lookup (the smallest listed bound >= R) serves it."""
    out, lo = [], 1
    for b in bounds:
        if b in rows:
            out.append((lo, b))
            lo = b + 1
    return out


class Overrides:
    """Replaces ``tune.pick`` while a candidate runs: the engine's own code paths run the candidate's launch."""

    def __init__(self) -> None:
        from tensorfold.families.glm5_next.cuda import tune

        self.tune = tune
        self.orig = tune.pick
        self.values: dict[str, Any] = {}
        tune.reset(path="", record_path="")             # no table, no recording while the harness runs
        tune.pick = self._pick

    def _pick(self, kernel: str, key: str, rows: int | None = None) -> Any:
        return self.values.get(kernel)

    def set(self, **values: Any) -> "Overrides":
        self.values = {k: v for k, v in values.items() if v is not None}
        return self

    def clear(self) -> None:
        self.values = {}


class Timer:
    """Times a callable: L2 flushed, the activations re-read, CUDA events around the call only, median of reps."""

    def __init__(self, reps: int) -> None:
        import torch

        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        self.l2 = int(getattr(props, "L2_cache_size", 0) or (128 << 20))
        self.flush = torch.empty((max(2 * self.l2, 256 << 20) // 4,), dtype=torch.int32, device="cuda")
        self.reps = reps

    HEAD_START = 200_000                         # GPU clock cycles of busy wait before each timed run, at least
    HEAD_MARGIN = 3.0                            # times the host's measured time to queue a run
    HEAD_MAX_MS = 50.0
    _cycles_ms: float | None = None

    def cycles_per_ms(self) -> float:
        """torch.cuda._sleep's cycles a millisecond on this GPU (measured once)."""

        import torch

        if self._cycles_ms is None:
            n = 4_000_000
            torch.cuda._sleep(n)
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            torch.cuda._sleep(n)
            e.record()
            e.synchronize()
            self._cycles_ms = n / max(s.elapsed_time(e), 1e-3)
        return self._cycles_ms

    def head_start(self, fn: Callable[[], None]) -> int:
        """Cycles of busy wait that cover the host's time to queue one run of ``fn`` (HEAD_MARGIN times the measured
        time, at least HEAD_START, at most HEAD_MAX_MS), so no host gap falls inside the timed span."""

        import torch

        ms = 0.0
        for _ in range(3):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            ms = max(ms, (time.perf_counter() - t0) * 1e3)
        torch.cuda.synchronize()
        cpm = self.cycles_per_ms()
        return int(min(max(self.HEAD_START, self.HEAD_MARGIN * ms * cpm), self.HEAD_MAX_MS * cpm))

    def time(self, fn: Callable[[], None], warm: list | None = None, reps: int | None = None) -> dict:
        return self.time_many({"_": fn}, warm, reps)["_"]

    def time_many(self, fns: dict, warm: list | None = None, reps: int | None = None) -> dict:
        """Each callable's stats, timed in rounds that run every candidate once (in a rotating order), so clock and
        temperature drift fall on all of them alike; before each run the L2 is flushed, the activations re-read and
        the GPU kept busy long enough for the host to queue the whole run (``head_start``: events then time the GPU
        alone)."""

        import torch

        reps = reps or self.reps
        keys = list(fns)
        for k in keys:
            fns[k]()
        torch.cuda.synchronize()
        heads = {k: self.head_start(fns[k]) for k in keys}
        events: dict = {k: [] for k in keys}
        for rep in range(reps):
            shift = rep % len(keys)
            for k in keys[shift:] + keys[:shift]:
                self.flush.zero_()
                for t in warm or ():
                    bits(t).max()                       # the activations back into L2, as in a forward
                torch.cuda._sleep(heads[k])
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record()
                fns[k]()
                e.record()
                events[k].append((s, e))
        torch.cuda.synchronize()
        out = {}
        for k in keys:
            ms = sorted(s.elapsed_time(e) for s, e in events[k])
            q = lambda f: ms[min(len(ms) - 1, int(f * len(ms)))]                 # noqa: E731
            out[k] = {"ms": statistics.median(ms), "p10": q(0.1), "p90": q(0.9), "n": len(ms)}
        return out


def better(base: dict, cand: dict, min_gain: float) -> bool:
    """cand beats base by min_gain and by more than either's spread (p90 - p10 of the runs)."""
    spread = max(base["p90"] - base["p10"], cand["p90"] - cand["p10"]) / 2
    return cand["ms"] <= base["ms"] * (1 - min_gain) and base["ms"] - cand["ms"] > spread


def cfg_key(cfg: Any) -> str:
    return json.dumps(cfg, sort_keys=True) if isinstance(cfg, dict) else str(cfg)


class Result:
    """Per kernel: winners (the table) and what was measured (an audit trail in the table's 'measured')."""

    def __init__(self) -> None:
        self.kernels: dict[str, dict] = {}
        self.measured: dict[str, dict] = {}

    def put(self, kernel: str, key: str, bucket: int | None, value: Any, info: dict) -> None:
        if bucket is None:
            self.kernels.setdefault(kernel, {})[key] = value
            self.measured.setdefault(kernel, {})[key] = info
        else:
            self.kernels.setdefault(kernel, {}).setdefault(key, {})[str(bucket)] = value
            self.measured.setdefault(kernel, {}).setdefault(key, {})[str(bucket)] = info


def sweep(name: str, key: str, buckets: list[tuple[int, int]], candidates: list, baseline_of: Callable[[int], Any],
          run: Callable[[Any, int, int], list], timed: Callable[[Any, int], Callable[[], None]],
          warm_of: Callable[[int], list], seeds: int, timer: Timer, min_gain: float, result: Result,
          table_kernel: str, value_of: Callable[[Any], Any] = lambda c: c, keep_default: bool = True) -> None:
    """The common loop. Exactness first, for the whole shape: the reference outputs (baseline_of(rows)) at the lowest
    and highest rows of EVERY bucket for every seed, and every candidate's outputs compared with them bit for bit; a
    candidate that differs anywhere (or fails) is out for every bucket. Then, per bucket, the exact candidates timed at
    the bucket's highest rows and the winner (or the baseline) stored. ``run(cfg, rows, seed)`` -> output tensors
    (fresh clones); ``timed(cfg, rows)`` -> a no-argument callable for timing."""

    points = sorted({r for lo, hi in buckets for r in (lo, hi)})
    refs = {}
    try:
        for lo, hi in buckets:                           # each bucket's own baseline (the same bits by design)
            base = baseline_of(hi)
            for rows in sorted({lo, hi}):
                for seed in range(seeds):
                    refs[(rows, seed)] = run(base, rows, seed)
    except Exception as exc:  # noqa: BLE001 - the default launch itself fails here: nothing to compare with
        log(f"{name} {key}: the default launch failed ({type(exc).__name__}: {str(exc).splitlines()[0][:160]}); "
            "shape skipped")
        result.measured.setdefault("skipped", {})[f"{table_kernel} {key}"] = f"default failed: {exc}"[:300]
        return
    # every bucket's baseline must give the reference bits at every point too (they are candidates like the rest)
    pool = list(candidates)
    for lo, hi in buckets:
        if all(cfg_key(c) != cfg_key(baseline_of(hi)) for c in pool):
            pool.append(baseline_of(hi))
    exact, rejected = [], {}
    for cand in pool:
        ok, why = True, ""
        try:
            for rows in points:
                for seed in range(seeds):
                    got = run(cand, rows, seed)
                    for i, (a, b) in enumerate(zip(refs[(rows, seed)], got)):
                        if not same_bits(a, b):
                            ok, why = False, f"output {i} at {rows} rows, seed {seed}: {first_diff(a, b)}"
                            break
                    if not ok:
                        break
                if not ok:
                    break
        except Exception as exc:  # noqa: BLE001 - a launch the GPU refuses is not a candidate
            ok, why = False, f"error: {type(exc).__name__}: {str(exc).splitlines()[0][:200] if str(exc) else ''}"
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except Exception:  # noqa: BLE001
                pass
        if ok:
            exact.append(cand)
        else:
            rejected[cfg_key(cand)] = why
    for lo, hi in buckets:
        base = baseline_of(hi)
        bk = cfg_key(base)
        if bk in rejected:
            log(f"{name} {key}: the default {bk} does not reproduce the reference ({rejected[bk]}): shape skipped")
            result.measured.setdefault("skipped", {})[f"{table_kernel} {key}"] = f"default: {rejected[bk]}"[:300]
            return
    for lo, hi in buckets:
        base = baseline_of(hi)
        bk = cfg_key(base)
        fns = {cfg_key(cand): timed(cand, hi) for cand in exact}
        if bk not in fns:
            fns[bk] = timed(base, hi)
        times = timer.time_many(fns, warm=warm_of(hi))
        best, best_t = base, times[bk]
        for cand in exact:
            t = times[cfg_key(cand)]
            if t["ms"] < best_t["ms"] and better(times[bk], t, min_gain):
                best, best_t = cand, t
        gain = 1 - best_t["ms"] / times[bk]["ms"] if times[bk]["ms"] > 0 else 0.0
        info = {"rows": [lo, hi], "baseline": {"cfg": base, **times[bk]}, "best": {"cfg": best, **best_t},
                "gain": round(gain, 4), "exact": len(exact), "rejected": rejected,
                "times": {k: round(v["ms"], 5) for k, v in times.items()}}
        if cfg_key(best) != bk or keep_default:
            result.put(table_kernel, key, hi, value_of(best), info)
        log(f"{name} {key} rows {lo}-{hi}: {len(exact)}/{len(pool)} exact, baseline {bk} "
            f"{times[bk]['ms'] * 1000:.1f} us -> {cfg_key(best)} {best_t['ms'] * 1000:.1f} us ({gain * 100:+.1f}%)"
            + (f"; rejected {len(rejected)}: {sorted(rejected)[:4]}" if rejected else ""))


# ------------------------------------------------------------------------------------------------------- shapes ---
def derived_shapes(config: str, tp: int, rank: int) -> dict:
    """The dense and expert shapes of a GLM-5.3-Flash rank from config.json (weights.py's stacks, tp.py's split)."""

    from tensorfold.families.glm5_next.cuda import tp as tpm

    with open(config, encoding="utf-8") as f:
        doc = json.load(f)
    t = doc.get("text_config", doc)
    lay = tpm.Layout.from_text(t, rank, tp)
    D = int(t["hidden_size"])
    lin = t.get("linear_attn_config") or {}
    hd = int(lin.get("head_dim", 128))
    HL, H = lay.lin_heads, lay.heads
    qk, vd = int(t["qk_head_dim"]), int(t["v_head_dim"])
    ql, kvl = int(t["q_lora_rank"]), int(t["kv_lora_rank"]) + int(t.get("qk_rope_head_dim", 0) or 0)
    inh, ihd = int(t.get("index_n_heads", 32)), int(t.get("index_head_dim", 128))
    q4 = {
        (3 * HL * hd + 2 * hd + HL, D),          # KDA proj [q | k | v | f_a | g_a | b]
        (HL * hd, hd),                            # KDA f_b, g_b
        (D, HL * hd),                             # KDA o_proj
        (ql + kvl, D),                            # DSA proj [q_a | kv_a]
        (H * qk, ql),                             # DSA q_b
        (D, H * vd),                              # DSA o_proj
        (ihd + inh, D),                           # indexer [wk | weights_proj]
        (inh * ihd, ql),                          # indexer wq_b
        (2 * lay.dense, D), (D, lay.dense),       # dense MLP (first layers)
        (2 * lay.shared, D), (D, lay.shared),     # shared expert
    }
    return {"q4": sorted(q4), "f8": [(lay.vocab, D)], "b16": [],
            "exl3": {"D": D, "NI": lay.moe, "E": int(t["n_routed_experts"]), "top_k": int(t["num_experts_per_tok"]),
                     "limit": float(t.get("swiglu_limit", 10.0))},
            "kda_heads": [HL], "index": {"heads": inh, "dim": ihd}}


def recorded_shapes(path: str, base: dict) -> dict:
    """Shapes from a TF_GLM_TUNE_RECORD file (what an engine start ran), over ``base`` (derived) for the rest."""

    with open(path, encoding="utf-8") as f:
        calls = json.load(f).get("calls", {})
    out = dict(base)

    def nk(keys):
        return sorted({tuple(int(v) for v in k.split("x")) for k in keys})

    if "q4_dec" in calls or "q4_prefill" in calls:
        out["q4"] = nk(set(calls.get("q4_dec", {})) | set(calls.get("q4_prefill", {})))
    if "f8_dec" in calls:
        out["f8"] = nk(calls["f8_dec"])
    if "b16_dec" in calls:
        out["b16"] = nk(calls["b16_dec"])
    if "kda_step" in calls:
        out["kda_heads"] = sorted({int(k) for k in calls["kda_step"]})
    rows = sorted({int(r) for v in calls.values() for rs in v.values() for r in rs if int(r) > 0})
    out["recorded_rows"] = rows
    return out


# ---------------------------------------------------------------------------------------------------- tuners -------
def tune_exl3_dec(sh: dict, args, ov: Overrides, timer: Timer, result: Result) -> None:
    import torch

    from tensorfold.cuda import experts as grouped
    from tensorfold.families.glm5_next.cuda import exl3_mm

    p = sh["exl3"]
    D, NI, E, K8, limit = p["D"], p["NI"], p["E"], p["top_k"], p["limit"]
    if D != 4096 or NI not in exl3_mm.DEC_WIDTHS:
        log(f"exl3_dec: D {D} / width {NI} do not take the decode kernel; skipped")
        return
    slots, dev = K8 + 1, "cuda"
    g = torch.Generator(device=dev).manual_seed(1234)

    def words(*shape):
        return torch.randint(-2 ** 31, 2 ** 31 - 1, shape, dtype=torch.int32, device=dev, generator=g)

    def half(n, sc, rows=E):
        return (torch.randn((rows, n), device=dev, generator=g) * sc).to(torch.float16)

    suh = half(D, 0.02, 1).expand(E, D).contiguous()     # one shared gate/up suh, as the checkpoint's every layer
    ex = exl3_mm.Exl3Experts(words(E, D // 16, NI // 16, 32), words(E, D // 16, NI // 16, 32),
                             words(E, NI // 16, D // 16, 32), suh, suh.clone(), half(NI, 0.5), half(NI, 0.5),
                             half(NI, 0.05), half(D, 0.2), E, NI, D)
    assert ex.shared_suh is not None
    RMAX = max(DEC_BUCKETS)
    plan = grouped.Plan(RMAX, slots, E + 1, dev)
    s = exl3_mm.Scratch(RMAX, slots, D, NI, dev)
    y = torch.empty((RMAX * slots, D), dtype=torch.float32, device=dev)
    pick = torch.full((RMAX, slots), E, dtype=torch.int32, device=dev)
    xs = {}

    def inputs(rows: int, seed: int):
        gen = torch.Generator(device=dev).manual_seed(1000 * seed + rows)
        if (rows, seed) not in xs:
            x = torch.randn((rows, D), device=dev, generator=gen).to(torch.bfloat16)
            pool = E if seed != 1 else min(E, K8 + 4)  # seed 1: few experts, so items fill to 16 pairs and split
            pk = torch.stack([torch.randperm(pool, device=dev, generator=gen)[:K8] for _ in range(rows)]).to(torch.int32)
            xs[(rows, seed)] = (x, pk)
        x, pk = xs[(rows, seed)]
        pick.fill_(E)
        pick[:rows, :K8] = pk
        grouped.route(pick[:rows].contiguous(), plan, plan.tile)
        return x

    env = {"gu": exl3_mm.dec_config("gu", D, NI, 1), "dn": exl3_mm.dec_config("dn", NI, D, 1)}
    log(f"exl3_dec: D {D}, width {NI}, {E} experts, top {K8}; baseline (environment) gu {env['gu']}, dn {env['dn']}")

    def call(cfg_gu, cfg_dn, rows, x):
        ov.set(exl3_dec_gu=cfg_gu, exl3_dec_dn=cfg_dn)
        try:
            exl3_mm.routed(x, pick, plan, ex, s, y, rows, limit)
        finally:
            ov.clear()

    def outputs(rows):
        P = rows * slots
        return [s.xd[:P].clone(), y[:P].clone()]

    # 1. the reference against the grouped kernel path (TF_GLM_EXL3_DEC=0): the engine's bits
    for seed in range(args.seeds):
        for rows in (1, 7, RMAX):
            x = inputs(rows, seed)
            sentinel(y, s.xd)
            call(env["gu"], env["dn"], rows, x)
            ref = y[:rows * slots].clone()
            old = os.environ.get("TF_GLM_EXL3_DEC")
            os.environ["TF_GLM_EXL3_DEC"] = "0"
            try:
                sentinel(y)
                exl3_mm.routed(x, pick, plan, ex, s, y, rows, limit)
            finally:
                if old is None:
                    os.environ.pop("TF_GLM_EXL3_DEC", None)
                else:
                    os.environ["TF_GLM_EXL3_DEC"] = old
            alt = y[:rows * slots].clone()
            routed_rows = torch.arange(rows * slots, device=dev) % slots != slots - 1
            if not same_bits(ref[routed_rows], alt[routed_rows]):
                raise SystemExit(f"exl3_dec: the decode kernel differs from the grouped kernel at {rows} rows "
                                 f"(seed {seed}): {first_diff(ref[routed_rows], alt[routed_rows])}; not tuning")
    log("exl3_dec: the reference equals the grouped kernel path bit for bit")

    lds, gus, dns = (0, 1, 2, 3), [], []
    for ld in lds:
        for wn in ((1, 2, 4) if ld == 0 else (1, 2)):
            for fuse in (0, 2):
                for xrow in (0, 1):
                    gus.append({"ld": ld, "wn": wn, "fuse": fuse, "xrow": xrow})
            for fuse in (0, 1):
                dns.append({"ld": ld, "wn": wn, "fuse": fuse})
    if args.quick:
        gus = [c for c in gus if c["fuse"] == env["gu"]["fuse"] and c["xrow"] == env["gu"]["xrow"]]
        dns = [c for c in dns if c["fuse"] == env["dn"]["fuse"]]
    buckets = buckets_of(args.rows_list, DEC_BUCKETS)

    def run_gu(cfg, rows, seed):
        x = inputs(rows, seed)
        sentinel(y, s.xd)
        call(cfg, env["dn"], rows, x)
        return outputs(rows)

    def run_dn(cfg, rows, seed):
        x = inputs(rows, seed)
        sentinel(y, s.xd)
        call(env["gu"], cfg, rows, x)
        return outputs(rows)

    def timed_gu(cfg, rows):
        x = inputs(rows, 0)
        return lambda: call(cfg, env["dn"], rows, x)

    def timed_dn(cfg, rows):
        x = inputs(rows, 0)
        return lambda: call(env["gu"], cfg, rows, x)

    def warm(rows):
        return [inputs(rows, 0), plan.items, plan.members, pick]

    key_gu, key_dn = f"{D}x{NI}", f"{NI}x{D}"
    sweep("exl3_dec gate/up", key_gu, buckets, gus, lambda r: env["gu"], run_gu, timed_gu, warm, args.seeds, timer,
          args.min_gain, result, "exl3_dec_gu")
    sweep("exl3_dec down", key_dn, buckets, dns, lambda r: env["dn"], run_dn, timed_dn, warm, args.seeds, timer,
          args.min_gain, result, "exl3_dec_dn")

    # 2. the winners together, through the table lookup as the engine does it, against the reference
    gu_t = result.kernels.get("exl3_dec_gu", {}).get(key_gu, {})
    dn_t = result.kernels.get("exl3_dec_dn", {}).get(key_dn, {})
    for lo, hi in buckets:
        for rows in sorted({lo, hi}):
            for seed in range(args.seeds):
                x = inputs(rows, seed)
                sentinel(y, s.xd)
                call(env["gu"], env["dn"], rows, x)
                ref = outputs(rows)
                sentinel(y, s.xd)
                call(gu_t.get(str(hi), env["gu"]), dn_t.get(str(hi), env["dn"]), rows, x)
                got = outputs(rows)
                if not all(same_bits(a, b) for a, b in zip(ref, got)):
                    raise SystemExit(f"exl3_dec: the winners together differ at {rows} rows; not writing a table")
        base = timer.time(timed_gu(env["gu"], hi), warm(hi))
        x_hi = inputs(hi, 0)
        both = timer.time(lambda: call(gu_t.get(str(hi), env["gu"]), dn_t.get(str(hi), env["dn"]), hi, x_hi),
                          warm(hi))
        log(f"exl3_dec rows {lo}-{hi}: whole routed experts {base['ms'] * 1000:.1f} -> {both['ms'] * 1000:.1f} us "
            f"({(1 - both['ms'] / base['ms']) * 100:+.1f}%)")
        result.measured.setdefault("exl3_dec_total", {})[str(hi)] = {"baseline_ms": base["ms"], "tuned_ms": both["ms"]}


def _q4_weight(n: int, k: int, gen):
    import torch

    from tensorfold.families.glm5_next.cuda import qmm

    words = torch.randint(-2 ** 31, 2 ** 31 - 1, (n, k // 8), dtype=torch.int32, device="cuda", generator=gen)
    scales = (torch.rand((n, k // 64), device="cuda", generator=gen) * 0.02 + 0.001).to(torch.bfloat16)
    biases = (torch.randn((n, k // 64), device="cuda", generator=gen) * 0.05).to(torch.bfloat16)
    return qmm.make_q4(words, scales, biases)


def tune_q4_dec(sh: dict, args, ov: Overrides, timer: Timer, result: Result) -> None:
    import torch

    from tensorfold.cuda.kernels import qmm as shared
    from tensorfold.families.glm5_next.cuda import qmm

    buckets = buckets_of([min(r, 64) for r in args.rows_list] + ([64] if 63 in args.rows_list else []),
                         (1, 2, 4, 8, 16, 32, 64))
    probe = _q4_weight(128, 128, torch.Generator(device="cuda").manual_seed(1))
    try:
        xp = torch.zeros((17, 128), dtype=torch.bfloat16, device="cuda")
        shared.matmul(xp, probe, None, sk=1, variant=0)
        torch.cuda.synchronize()
    except RuntimeError as exc:
        buckets = [(lo, hi) for lo, hi in buckets if hi <= 16]
        log(f"q4_dec: this image's tile configs take at most 16 rows ({str(exc).splitlines()[0][:80]}): windows of "
            "more rows keep the stock tiles (patch 0133 lifts the limit)")
    for n, k in sh["q4"]:
        if k % 64:
            continue
        gen = torch.Generator(device="cuda").manual_seed(n * 7 + k)
        q = _q4_weight(n, k, gen)
        RMAX = 64
        out32 = torch.empty((RMAX, n), dtype=torch.float32, device="cuda")
        out16 = torch.empty((RMAX, n), dtype=torch.bfloat16, device="cuda")
        xs = {}

        def x_of(rows, seed):
            if (rows, seed) not in xs:
                gg = torch.Generator(device="cuda").manual_seed(seed * 131 + rows)
                xs[(rows, seed)] = torch.randn((rows, k), device="cuda", generator=gg).to(torch.bfloat16)
            return xs[(rows, seed)]

        def call(cfg, rows, seed, f32):
            x = x_of(rows, seed)
            out = (out32 if f32 else out16)[:rows]
            ov.set(q4_dec=cfg)
            try:
                qmm.matmul(x, q, None, out=out, f32=f32)
            finally:
                ov.clear()
            return out

        ov.clear()
        stock = {r: qmm.dec_tile(n, k, r) for r in (1, 16, 17, 64)}

        def baseline(rows):
            v = stock[1] if rows <= 16 else stock[64]
            return -1 if v is None else v

        def run(cfg, rows, seed):
            sentinel(out16, out32)
            a = call(cfg, rows, seed, False).clone()
            b = call(cfg, rows, seed, True).clone()
            return [a, b]

        def timed(cfg, rows):
            return lambda: call(cfg, rows, 0, True)

        cands = list(range(-1, 16)) if not args.quick else [-1, 0, 2, 12]
        sweep("q4_dec", f"{n}x{k}", buckets, cands, baseline, run, timed, lambda r: [x_of(r, 0)], args.seeds, timer,
              args.min_gain, result, "q4_dec")
        del q
        torch.cuda.empty_cache()


def _q4p2_candidates(quick: bool) -> list[int]:
    """Patch 0195's prompt-matmul variants as table tiles (32 + v), when the image has qmm_prefill2; none otherwise."""
    try:
        from tensorfold.cuda.kernels import qmm_prefill2
    except ImportError:
        return []
    ids = [qmm_prefill2.BASE + v for v in range(qmm_prefill2.COUNT)]
    return ids if not quick else ids[:4]


def tune_q4_prefill(sh: dict, args, ov: Overrides, timer: Timer, result: Result) -> None:
    import torch

    from tensorfold.cuda.kernels import qmm as shared
    from tensorfold.families.glm5_next.cuda import qmm

    for n, k in sh["q4"]:
        if k % 64:
            continue
        gen = torch.Generator(device="cuda").manual_seed(n * 11 + k)
        q = _q4_weight(n, k, gen)
        M = max(args.prefill_list)
        out32 = torch.empty((M, n), dtype=torch.float32, device="cuda")
        out16 = torch.empty((M, n), dtype=torch.bfloat16, device="cuda")
        xs = {}

        def x_of(rows, seed):
            if (rows, seed) not in xs:
                gg = torch.Generator(device="cuda").manual_seed(seed * 17 + rows)
                xs[(rows, seed)] = torch.randn((rows, k), device="cuda", generator=gg).to(torch.bfloat16)
            return xs[(rows, seed)]

        def call(cfg, rows, seed, f32):
            x = x_of(rows, seed)
            out = (out32 if f32 else out16)[:rows]
            ov.set(q4_prefill=cfg)
            try:
                shared.prefill_matmul(x, q, f32=f32, out=out, tile=qmm.prefill_tile(n, k, rows))
            finally:
                ov.clear()
            return out

        def run(cfg, rows, seed):
            sentinel(out16, out32)
            return [call(cfg, rows, seed, False).clone(), call(cfg, rows, seed, True).clone()]

        def timed(cfg, rows):
            return lambda: call(cfg, rows, 0, False)

        cands = list(range(0, 12)) if not args.quick else [0, 1, 3, 7]
        cands += _q4p2_candidates(args.quick)          # patch 0195's variants (tile 32 + v), when the image has them
        if len(args.prefill_list) == 1:
            # rows: the chunk (exact at a partial last chunk too: 1,000 rows)
            bks = [(min(1000, M), M)]
        else:
            # one bucket a listed row count (e.g. 512: the index split's blocks, 2,048: a two-lane chunk's lanes, 4,096:
            # whole chunks), each checked at its lowest and highest rows and timed at its highest
            bounds = args.prefill_list
            bks = [(1 if i == 0 else bounds[i - 1] + 1, b) for i, b in enumerate(bounds)]
        sweep("q4_prefill", f"{n}x{k}", bks, cands, lambda r: 0, run, timed, lambda r: [x_of(r, 0)],
              max(1, args.seeds - 1), timer, args.min_gain, result, "q4_prefill", keep_default=True)
        # one bucket: one table value for every chunk size (flatten {"<M>": v} to v); several: {"<rows>": v}
        tab = result.kernels.get("q4_prefill", {})
        if len(bks) == 1 and f"{n}x{k}" in tab and isinstance(tab[f"{n}x{k}"], dict):
            tab[f"{n}x{k}"] = tab[f"{n}x{k}"][str(M)]
        del q
        torch.cuda.empty_cache()


def _triton_dense(kind: str, sh: dict, args, ov: Overrides, timer: Timer, result: Result) -> None:
    import torch

    from tensorfold.families.glm5_next.cuda import qmm

    shapes = sh["f8"] if kind == "f8" else sh["b16"]
    buckets = buckets_of([min(r, 64) for r in args.rows_list] + ([64] if 63 in args.rows_list else []),
                         (1, 2, 4, 8, 16, 32, 64))
    for n, k in shapes:
        gen = torch.Generator(device="cuda").manual_seed(n + 3 * k)
        w = (torch.randn((n, k), device="cuda", generator=gen) * 0.02).to(torch.bfloat16)
        q = qmm.make_f8(w) if kind == "f8" else qmm.make_b16(w)
        if kind == "f8" and not isinstance(q, qmm.F8):
            continue
        del w
        RMAX = 64
        out32 = torch.empty((RMAX, n), dtype=torch.float32, device="cuda")
        out16 = torch.empty((RMAX, n), dtype=torch.bfloat16, device="cuda")
        part = torch.empty((8 * RMAX * n,), dtype=torch.float32, device="cuda")
        xs = {}

        def x_of(rows, seed):
            if (rows, seed) not in xs:
                gg = torch.Generator(device="cuda").manual_seed(seed * 19 + rows)
                xs[(rows, seed)] = torch.randn((rows, k), device="cuda", generator=gg).to(torch.bfloat16)
            return xs[(rows, seed)]

        def call(cfg, rows, seed, f32):
            out = (out32 if f32 else out16)[:rows]
            ov.set(**{f"{kind}_dec": cfg})
            try:
                qmm.matmul(x_of(rows, seed), q, None, out=out, f32=f32, part=part)
            finally:
                ov.clear()
            return out

        def baseline(rows):
            bm = qmm.bucket(rows)
            if kind == "f8":
                w_, s_ = qmm.F8_CONFIG[bm]
                return {"warps": w_, "stages": s_, "bn": qmm.F8_BN_DECODE if bm == 16 else qmm.F8_BN}
            w_, s_ = qmm.B16_CONFIG[bm]
            return {"warps": w_, "stages": s_}

        def run(cfg, rows, seed):
            sentinel(out16, out32)
            return [call(cfg, rows, seed, False).clone(), call(cfg, rows, seed, True).clone()]

        def timed(cfg, rows):
            return lambda: call(cfg, rows, 0, False)

        cands = []                          # each (bn, warps, stages) compiles once a row tile: kept to 24 / 8
        for warps in (4, 8):
            for stages in (2, 3, 4, 6):
                if kind == "f8":
                    for bn in (16, 32, 64):
                        cands.append({"warps": warps, "stages": stages, "bn": bn})
                else:
                    cands.append({"warps": warps, "stages": stages})
        if args.quick:
            cands = cands[::7]
        sweep(f"{kind}_dec", f"{n}x{k}", buckets, cands, baseline, run, timed, lambda r: [x_of(r, 0)], args.seeds,
              timer, args.min_gain, result, f"{kind}_dec")
        del q
        torch.cuda.empty_cache()


def tune_f8_dec(sh, args, ov, timer, result) -> None:
    _triton_dense("f8", sh, args, ov, timer, result)


def tune_b16_dec(sh, args, ov, timer, result) -> None:
    if not sh["b16"]:
        log("b16_dec: no BF16 decode matmul in these shapes (DENSE=q4); skipped")
        return
    _triton_dense("b16", sh, args, ov, timer, result)


def tune_kda_step(sh: dict, args, ov: Overrides, timer: Timer, result: Result) -> None:
    import torch

    from tensorfold.families.glm5_next.cuda import kda

    buckets = buckets_of(args.rows_list, DEC_BUCKETS)
    RMAX = max(DEC_BUCKETS)
    for H in sh["kda_heads"]:
        C = 3 * H * kda.DK
        b_off = C + 256
        g = torch.Generator(device="cuda").manual_seed(H)
        p = (torch.randn((RMAX, b_off + H + 32), device="cuda", generator=g) * 0.5).to(torch.bfloat16)
        a = torch.randn((RMAX, H * kda.DK), device="cuda", generator=g).to(torch.bfloat16)
        gate = torch.randn((RMAX, H * kda.DV), device="cuda", generator=g).to(torch.bfloat16)
        cs = (torch.randn((3, C), device="cuda", generator=g) * 0.5).to(torch.bfloat16)
        cw = (torch.randn((C, 4), device="cuda", generator=g) * 0.5).to(torch.bfloat16)
        states = [(torch.randn((H, kda.DV, kda.DK), device="cuda", generator=g) * 0.1) for _ in range(args.seeds)]
        a_log = (torch.rand(H, device="cuda", generator=g) * 2 - 1)
        dt_bias = (torch.randn(H * kda.DK, device="cuda", generator=g) * 0.1)
        norm_w = (torch.rand(kda.DV, device="cuda", generator=g) + 0.5).to(torch.bfloat16)
        kda.reserve(RMAX, H, "cuda")
        sc = kda.KDAScratch(RMAX, H, "cuda")
        st_out = torch.empty_like(states[0])

        def call(cfg, rows, seed, wide=True):
            ov.set(kda_step=cfg)
            try:
                return kda.chain(p[:rows], b_off, a[:rows], gate[:rows], cs, cw, states[seed], a_log, dt_bias, norm_w,
                                 1e-5, -5.0, rows, sc, st_out, wide=wide)
            finally:
                ov.clear()

        def run(cfg, rows, seed):
            sentinel(sc.out, st_out, sc.k, sc.v, sc.g, sc.b)
            out = call(cfg, rows, seed)
            return [out.clone(), st_out.clone(), sc.k[:rows].clone(), sc.v[:rows].clone(), sc.g[:rows].clone(),
                    sc.b[:rows].clone()]

        # the reference against the fused one-block chain (chain_kernel): the engine's bits
        for seed in range(args.seeds):
            for rows in (1, 5, RMAX):
                ref = run(None, rows, seed)
                sentinel(sc.out, st_out, sc.k, sc.v, sc.g, sc.b)
                ov.clear()
                out = kda.chain(p[:rows], b_off, a[:rows], gate[:rows], cs, cw, states[seed], a_log, dt_bias, norm_w,
                                1e-5, -5.0, rows, sc, st_out, wide=False)
                alt = [out.clone(), st_out.clone(), sc.k[:rows].clone(), sc.v[:rows].clone(), sc.g[:rows].clone(),
                       sc.b[:rows].clone()]
                if not all(same_bits(x, y) for x, y in zip(ref, alt)):
                    raise SystemExit(f"kda_step: the wide chain differs from the fused chain at {rows} rows")
                if not bool(torch.isfinite(ref[0].float()).all()):
                    log("kda_step: warning: non-finite outputs in the test data")
        log(f"kda_step: {H} heads; the reference equals the fused chain bit for bit")

        def timed(cfg, rows):
            return lambda: call(cfg, rows, 0)

        cands = [{"warps": w, "tr": t} for w in (1, 2, 4, 8) for t in (8, 16)]
        sweep("kda_step", str(H), buckets, cands, lambda r: {"warps": kda.STEP_DEFAULT[0], "tr": kda.STEP_DEFAULT[1]},
              run, timed, lambda r: [p[:r], a[:r], gate[:r], states[0]], args.seeds, timer, args.min_gain, result,
              "kda_step")


def tune_exl3_prompt(sh: dict, args, ov: Overrides, timer: Timer, result: Result) -> None:
    import torch

    from tensorfold.cuda import experts as grouped
    from tensorfold.families.glm5_next.cuda import exl3_mm

    p = sh["exl3"]
    D, NI, E, K8, limit = p["D"], p["NI"], p["E"], p["top_k"], p["limit"]
    slots, dev = K8 + 1, "cuda"
    M = max(args.prefill_list)
    g = torch.Generator(device=dev).manual_seed(99)

    def words(*shape):
        return torch.randint(-2 ** 31, 2 ** 31 - 1, shape, dtype=torch.int32, device=dev, generator=g)

    def half(n, sc, rows=E):
        return (torch.randn((rows, n), device=dev, generator=g) * sc).to(torch.float16)

    suh = half(D, 0.02, 1).expand(E, D).contiguous()
    ex = exl3_mm.Exl3Experts(words(E, D // 16, NI // 16, 32), words(E, D // 16, NI // 16, 32),
                             words(E, NI // 16, D // 16, 32), suh, suh.clone(), half(NI, 0.5), half(NI, 0.5),
                             half(NI, 0.05), half(D, 0.2), E, NI, D)
    y = torch.empty((M * slots, D), dtype=torch.float32, device=dev)
    pick = torch.full((M, slots), E, dtype=torch.int32, device=dev)
    state = {}
    for passm in (64, 128):
        state[passm] = (grouped.Plan(M, slots, E + 1, dev, prefill=True, tile=passm),
                        exl3_mm.Scratch(M, slots, D, NI, dev, prompt=True))
    xs = {}

    def inputs(rows, seed, passm):
        if (rows, seed) not in xs:
            gen = torch.Generator(device=dev).manual_seed(77 * seed + rows)
            x = torch.randn((rows, D), device=dev, generator=gen).to(torch.bfloat16)
            pk = torch.argsort(torch.rand((rows, E), device=dev, generator=gen), dim=1)[:, :K8].to(torch.int32)
            xs[(rows, seed)] = (x, pk)
        x, pk = xs[(rows, seed)]
        pick.fill_(E)
        pick[:rows, :K8] = pk
        plan, s = state[passm]
        grouped.route(pick[:rows].contiguous(), plan, passm)
        return x, plan, s

    def routed(cfg, x, plan, s, rows):
        ov.set(exl3_prompt=cfg)                      # the prompt kernels' stages come from the table lookup
        try:
            exl3_mm.routed(x, pick, plan, ex, s, y, rows, limit)
        finally:
            ov.clear()

    def call(cfg, rows, seed):
        x, plan, s = inputs(rows, seed, cfg["passm"])
        routed(cfg, x, plan, s, rows)

    def run(cfg, rows, seed):
        sentinel(y)
        call(cfg, rows, seed)
        P = rows * slots
        keep = torch.arange(P, device=dev) % slots != slots - 1
        return [y[:P][keep].clone()]

    def timed(cfg, rows):
        x, plan, s = inputs(rows, 0, cfg["passm"])
        return lambda: routed(cfg, x, plan, s, rows)

    ov.clear()
    from tensorfold.families.glm5_next.cuda import tune as _t

    try:
        _t.CHOICES["exl3_prompt"]({"passm": 64, "stages": 3})
        staged = hasattr(exl3_mm, "prompt_stages")
    except ValueError:
        staged = False
    if staged:
        base = {"passm": exl3_mm.prompt_pass(), "stages": exl3_mm.prompt_stages()}
        cands = [{"passm": pm, "stages": st} for pm in (64, 128) for st in (2, 3, 4)]
    else:
        log("exl3_prompt: this image has no prompt stages setting (patch 0133): passes only")
        base = {"passm": exl3_mm.prompt_pass()}
        cands = [{"passm": 64}, {"passm": 128}]
    sweep("exl3_prompt", "all", [(min(1000, M), M)], cands, lambda r: base, run, timed, lambda r: [],
          max(1, args.seeds - 1), timer, args.min_gain, result, "exl3_prompt")
    tab = result.kernels.get("exl3_prompt", {})
    if isinstance(tab.get("all"), dict) and str(M) in tab["all"]:
        tab["all"] = tab["all"][str(M)]


def tune_sparse_scores(sh: dict, args, ov: Overrides, timer: Timer, result: Result) -> None:
    """The multi-stream indexer's score grid (programs a segment) through the engine's seg_select_tokens at the
    engine's context (--sparse-context): every grid must give the default's scores, tokens and counts on bf16 AND on
    FP8 pooled keys (kv8's own quantizer), at several context lengths up to it, one and four segments; the winner
    minimizes the summed time over those contexts on the key format the engine runs (TF_GLM_KV, default bf16). The
    entry holds for indexer scratches of up to the tuned context's pools (an engine with a longer context keeps the
    default grid)."""

    import torch

    from tensorfold.families.glm5_next.cuda import kv8, segments as segm, sparse

    from tensorfold.families.glm5_next.cuda import tune as _t

    if "sparse_scores" not in _t.CHOICES:
        log("sparse_scores: this image's engine has no such table entry (patch 0133): skipped")
        return
    H, D = sh["index"]["heads"], sh["index"]["dim"]
    cap = int(args.sparse_context)
    cap_rows = -(-cap // segm.EXTENT) * segm.EXTENT
    contexts = [c for c in (32768, 131072, 524288, 1048576) if c < cap] + [cap]
    kind = kv8.kv_kind()                                   # the engine's key format (TF_GLM_KV)
    dev = "cuda"
    cands = [256, 512, 768, 1024, 1536, 2048, 3072, 4096]
    totals: dict = {}
    pools = None
    for S in (1, 4):
        per = 2
        R = S * per
        g = torch.Generator(device=dev).manual_seed(S)
        keys = (torch.randn((S * cap_rows // sparse.POOL, D), device=dev, generator=g) * 0.5).to(torch.bfloat16)
        pks = {"bf16": keys, "fp8": kv8.quantize_rows(keys)}
        qi = torch.randn((R, H * D), device=dev, generator=g).to(torch.bfloat16)
        ikr = torch.randn((R, D + H), device=dev, generator=g).to(torch.bfloat16)
        wts = ikr[:, D:]
        sel = segm.SelectScratch(R, cap, dev)
        pools = sel.scores.shape[1]
        rows = segm.SegRows(R, dev, max_segs=S)

        def run(G, ctx, fmt):
            rows.set([(s * cap_rows, ctx - per, per) for s in range(S)])
            sentinel(sel.scores)
            ov.set(sparse_scores={"grid": G})
            try:
                tokens, counts = sparse.seg_select_tokens(qi, wts, pks[fmt], rows, sel)
            finally:
                ov.clear()
            return [sel.scores.clone(), tokens.clone(), counts.clone()]

        refs = {(ctx, fmt): run(sparse.SEG_SCORE_GRID, ctx, fmt) for ctx in contexts for fmt in pks}
        for G in cands:
            if G in totals and totals[G] is None:
                continue
            ok = all(all(same_bits(a, b) for a, b in zip(refs[(ctx, fmt)], run(G, ctx, fmt)))
                     for ctx in contexts for fmt in pks)
            if not ok:
                totals[G] = None
                log(f"sparse_scores: grid {G} differs from the default's outputs: rejected")
                continue
            t = 0.0
            for ctx in contexts:
                rows.set([(s * cap_rows, ctx - per, per) for s in range(S)])

                def go(G=G):
                    ov.set(sparse_scores={"grid": G})
                    try:
                        sparse.seg_select_tokens(qi, wts, pks[kind], rows, sel)
                    finally:
                        ov.clear()
                t += timer.time(go, warm=[qi, ikr])["ms"]
            totals[G] = (totals.get(G) or 0.0) + t
        del pks, keys
        torch.cuda.empty_cache()
    base = totals.get(sparse.SEG_SCORE_GRID)
    best = min((G for G, t in totals.items() if t is not None), key=lambda G: totals[G])
    if base is None:
        raise SystemExit("sparse_scores: the default grid was not timed")
    if totals[best] > base * (1 - args.min_gain):
        best = sparse.SEG_SCORE_GRID
    result.put("sparse_scores", f"{H}x{D}", pools, {"grid": best},
               {"context": cap, "pools": pools, "contexts": contexts, "segments": [1, 4], "keys": kind,
                "checked_keys": ["bf16", "fp8"], "ms_summed": {str(G): t for G, t in totals.items()}})
    log(f"sparse_scores {H}x{D}: grid {sparse.SEG_SCORE_GRID} {base:.3f} ms -> {best} {totals[best]:.3f} ms "
        f"(summed over contexts {contexts} and 1 / 4 segments, {kind} keys; for scratches up to {pools} pools)")


TUNERS = {"exl3_dec": tune_exl3_dec, "kda_step": tune_kda_step, "q4_dec": tune_q4_dec, "f8_dec": tune_f8_dec,
          "b16_dec": tune_b16_dec, "q4_prefill": tune_q4_prefill, "exl3_prompt": tune_exl3_prompt,
          "sparse_scores": tune_sparse_scores}


# --------------------------------------------------------------------------------------------------------- main ---
def device_info() -> dict:
    import torch

    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    info = {"name": props.name, "sms": props.multi_processor_count, "capability": [props.major, props.minor],
            "memory_bytes": props.total_memory, "l2_bytes": int(getattr(props, "L2_cache_size", 0) or 0),
            "torch": torch.__version__, "cuda": torch.version.cuda, "host": platform.node()}
    try:
        import triton

        info["triton"] = triton.__version__
    except Exception:  # noqa: BLE001
        pass
    return info


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--out", required=True, help="the table file to write; this GPU's table replaces an earlier one "
                    "for the same GPU model in it, other GPUs' tables are kept (TF_GLM_TUNE points the engine at it)")
    ap.add_argument("--kernels", default="exl3_dec,kda_step,q4_dec,f8_dec,b16_dec,q4_prefill,exl3_prompt,sparse_scores")
    ap.add_argument("--sparse-context", type=int, default=1048576,
                    help="the engine's context (its --context; start.sh: CONTEXT): sparse_scores is tuned at "
                         "contexts up to it, and its entry holds for indexer scratches of up to that many pools")
    ap.add_argument("--shapes", help="a TF_GLM_TUNE_RECORD file: exactly the shapes an engine start ran")
    ap.add_argument("--config", help="the checkpoint's config.json (shapes derived for --tp / --rank)")
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--rank", type=int, default=0, help="the rank whose shares to tune (rank 0 holds the largest)")
    ap.add_argument("--rows", default=",".join(map(str, DEC_BUCKETS)), help="decode row buckets to tune (upper bounds)")
    ap.add_argument("--prefill-rows", default="2048", help="prompt chunk rows (TF_GLM_PREFILL_ROWS)")
    ap.add_argument("--reps", type=int, default=40, help="timed runs a candidate (median)")
    ap.add_argument("--seeds", type=int, default=3, help="random inputs a candidate must match the reference on")
    ap.add_argument("--min-gain", type=float, default=0.02, help="a winner beats the default by this fraction")
    ap.add_argument("--quick", action="store_true", help="a few candidates a kernel: checks the harness end to end")
    ap.add_argument("--l2pf-wave", action="store_true", help="also write l2pf one_wave = 4 x SMs (not measured)")
    ap.add_argument("--merge", help="a table file: this GPU's entries for kernels not tuned now are kept")
    args = ap.parse_args(argv)
    args.rows_list = sorted({int(r) for r in args.rows.split(",") if r.strip()})
    args.prefill_list = sorted({int(r) for r in args.prefill_rows.split(",") if r.strip()})
    bad = [r for r in args.rows_list if r not in DEC_BUCKETS]
    if bad:
        ap.error(f"--rows: bucket bounds among {DEC_BUCKETS}, not {bad}")

    import torch

    if not torch.cuda.is_available():
        log("no CUDA GPU visible")
        return 2
    dev = device_info()
    log(f"{HARNESS} on {dev['name']} ({dev['sms']} SMs, sm_{dev['capability'][0]}{dev['capability'][1]}, "
        f"L2 {dev['l2_bytes'] >> 20} MiB), torch {dev['torch']}")
    free, total = torch.cuda.mem_get_info()
    if free < 0.5 * total:
        log(f"only {free >> 30} of {total >> 30} GiB free on this GPU: is the engine still running? stopping")
        return 3
    cfg_path = args.config
    if cfg_path is None and os.path.exists("/model/config.json"):     # the checkpoint mounted where the engine reads it
        cfg_path = "/model/config.json"
    if cfg_path is None:
        ap.error("--config: the checkpoint's config.json (or --shapes with it)")
    shapes = derived_shapes(cfg_path, args.tp, args.rank)
    if args.shapes:
        shapes = recorded_shapes(args.shapes, shapes)
    log(f"shapes ({'recorded' if args.shapes else 'derived'}): q4 {shapes['q4']}, f8 {shapes['f8']}, b16 "
        f"{shapes['b16']}, exl3 {shapes['exl3']}, kda heads {shapes['kda_heads']}")
    env = {k: v for k, v in os.environ.items() if k.startswith("TF_GLM_")}
    log(f"engine settings seen (the baselines follow them): {env or 'none (engine defaults)'}")

    ov = Overrides()
    timer = Timer(args.reps)
    result = Result()
    if args.merge:                                       # this GPU's entries for kernels not tuned now
        from tensorfold.families.glm5_next.cuda import tune as _t

        with open(args.merge, encoding="utf-8") as f:
            for d, k in _t.parse(json.load(f), args.merge):
                if _t.same_device(d, {"name": dev["name"], "sms": dev["sms"], "capability": dev["capability"]}):
                    result.kernels.update(json.loads(json.dumps(k)))
    t0 = time.time()
    names = [k.strip() for k in args.kernels.split(",") if k.strip()]
    for name in names:
        if name not in TUNERS:
            ap.error(f"--kernels: {name} is not one of {', '.join(TUNERS)}")
    from tensorfold.families.glm5_next.cuda import tune

    key = {"name": dev["name"], "sms": dev["sms"], "capability": dev["capability"]}

    def save() -> None:
        """This GPU's table merged into --out (other GPUs' kept); the engine's own check of what it will load."""
        for k in [k for k in result.kernels if k not in tune.CHOICES]:
            log(f"{k}: this image's engine has no such table entry; left out")
            result.kernels.pop(k)
        tune.validate_kernels(result.kernels, "harness output")
        entry = {"device": key, "kernels": result.kernels, "harness": HARNESS, "about": dev,
                 "created": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                 "args": {k: v for k, v in vars(args).items() if k not in ("rows_list", "prefill_list")},
                 "settings": env, "measured": result.measured, "seconds": round(time.time() - t0, 1)}
        doc = {"format": tune.FORMAT, "tables": []}
        if os.path.exists(args.out):
            with open(args.out, encoding="utf-8") as f:
                doc = json.load(f)
            tune.parse(doc, args.out)                     # a broken file is not overwritten silently
            doc["tables"] = [t for t in doc["tables"] if not tune.same_device(t["device"], key)]
        doc["tables"].append(entry)
        tune.parse(doc, "the new table file")
        tmp = args.out + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=1, default=str)
        os.replace(tmp, args.out)

    owned = {"exl3_dec": ("exl3_dec_gu", "exl3_dec_dn")}
    for name in names:
        t1 = time.time()
        log(f"--- {name}")
        for prefix in owned.get(name, (name,)):
            result.kernels.pop(prefix, None)
        try:
            TUNERS[name](shapes, args, ov, timer, result)
            torch.cuda.synchronize()
            log(f"{name}: {time.time() - t1:.0f} s")
        except (Exception, SystemExit) as exc:  # noqa: BLE001 - one kernel's failure must not lose the others
            log(f"{name}: FAILED ({type(exc).__name__}: {str(exc).splitlines()[0][:200] if str(exc) else ''}); its "
                "entries are left out, the other kernels go on")
            result.measured.setdefault("errors", {})[name] = f"{type(exc).__name__}: {exc}"[:500]
            for prefix in owned.get(name, (name,)):
                result.kernels.pop(prefix, None)
            try:
                torch.cuda.synchronize()
            except Exception:  # noqa: BLE001 - a sticky CUDA error: later tuners will say so too
                pass
        ov.clear()
        save()
    if args.l2pf_wave:
        result.kernels["l2pf"] = {"all": {"one_wave": 4 * dev["sms"]}}
        save()
    n = sum(len(v) for v in result.kernels.values())
    log(f"wrote {args.out}: {n} entries over {len(result.kernels)} kernels in {time.time() - t0:.0f} s; "
        f"start the engine with TF_GLM_TUNE=<that path inside the container>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
