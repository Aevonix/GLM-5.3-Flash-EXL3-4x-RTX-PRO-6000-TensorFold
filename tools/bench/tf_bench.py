#!/usr/bin/env python3
"""Same-client benchmark phases for TensorFold and compatible servers.

Uses the bundled bench_client.py helper, overridable with --bench-client or
TF_BENCH_CLIENT. See README.md for optional dependencies and phase commands.
Results retain the original record structure; the T=1.0 profile is named sampled.
"""
import json
import os
import importlib.util
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
def _bootstrap():
    import argparse
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--bench-client", default=os.environ.get("TF_BENCH_CLIENT", str(HERE / "bench_client.py")),
                    help="path to the same-client helper (default: bundled bench_client.py; override: TF_BENCH_CLIENT)")
    ap.add_argument("--results-dir", default=os.environ.get("TF_BENCH_RESULTS", str(HERE / "results")))
    ap.add_argument("--reference-dir", default=os.environ.get("TF_BENCH_REFERENCES", str(HERE / "references")))
    known, rest = ap.parse_known_args()
    if "--help" in rest or "-h" in rest:
        print("usage: tf_bench.py [--bench-client FILE] --id RUN [--base URL] [--model MODEL] PHASE ...")
        print("phases: gate gatelong speed3 prefill3 warm exact warmthink speed2 peak1m")
        print("The helper also supplies smoke, speed, prefill, needle, cachecheck, refusal, qualify and soak.")
        print("--base defaults to TF_BENCH_BASE or http://127.0.0.1:8020/v1")
        print("--model defaults to TF_BENCH_MODEL or glm-5.3-flash")
        print("--host defaults to TF_BENCH_HOST; SSH GPU telemetry is off unless a host is supplied")
        print("BMC sensor reads are off unless TF_BENCH_BMC_CMD is set (see README.md)")
        print("--qualifier PATH defaults to TF_BENCH_QUALIFIER; qualify is unavailable without it")
        print("--gate-levels 4 --gate-tokens 256 --gate-write-ref --ref-tag TAG")
        print("--agg-levels 2,4,8 --speed-variants kk,sampled,nucleus")
        print("--prefill-sizes 32000,100000,140000 --peak-tokens 1000000")
        ap.print_help()
        raise SystemExit(0)
    helper = Path(known.bench_client).expanduser().resolve()
    if not helper.is_file():
        ap.error(f"bench client helper does not exist: {helper}")
    sys.argv[1:] = rest
    spec = importlib.util.spec_from_file_location("bench_client", helper)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.path.insert(0, str(helper.parent))
    spec.loader.exec_module(module)
    # The helper writes HERE/results/<id>. Redirect only that output root.
    result_dir = Path(known.results_dir).expanduser().resolve()
    if result_dir.name != "results":
        ap.error("--results-dir must end in /results (the helper appends that component)")
    module.HERE = result_dir.parent
    return module, Path(known.reference_dir).expanduser().resolve()


bc, REFERENCE_DIR = _bootstrap()

STRUCT = ("Return a JSON array of 25 objects describing fictional warehouse inventory items. Each object has the keys "
          "id (integer), sku (string), name (string), quantity (integer), unit_price (number), tags (array of strings) "
          "and location (object with aisle, shelf and bin). Output only the JSON.")


def ph_speed2(c, a):
    R = {}
    bc.one_decode(c, "Say hi.", n=16)
    for name, p in (("code", bc.CODE), ("prose", bc.PROSE), ("structured", STRUCT)):
        runs = [bc.one_decode(c, p) for _ in range(3)]
        R[f"single_{name}"] = dict(runs=runs, tps_median=statistics.median(r["tps"] for r in runs),
                                   ttft_median=statistics.median(r["ttft"] for r in runs))
        print(name, R[f"single_{name}"], flush=True)
    R["single_tps"] = round(statistics.mean([R["single_code"]["tps_median"], R["single_prose"]["tps_median"]]), 1)
    for conc in (2, 4, 8):
        t = time.time()
        with ThreadPoolExecutor(conc) as ex:
            res = list(ex.map(lambda i: bc.one_decode(c, (bc.CODE if i % 2 else bc.PROSE) + f" (variant {i})"), range(conc)))
        wall = time.time() - t
        R[f"agg{conc}"] = dict(aggregate_tps=round(sum(r["ct"] for r in res) / wall, 1),
                               per_stream_min=min(r["tps"] for r in res), per_stream_median=statistics.median(r["tps"] for r in res),
                               ttft_median=statistics.median(r["ttft"] for r in res), ttft_max=max(r["ttft"] for r in res))
        print(conc, R[f"agg{conc}"], flush=True)
    # Structured output at four concurrent requests.
    t = time.time()
    with ThreadPoolExecutor(4) as ex:
        res = list(ex.map(lambda i: bc.one_decode(c, STRUCT + f" (variant {i})"), range(4)))
    R["agg4_structured"] = dict(aggregate_tps=round(sum(r["ct"] for r in res) / (time.time() - t), 1),
                                ttft_max=max(r["ttft"] for r in res))
    print("agg4 structured", R["agg4_structured"], flush=True)
    return R


def ph_warmthink(c, a, turns=5):
    sysmsg = bc.nonce() + "The following is a long-running session history.\n" + c.filler_tokens(128000, seed=4711)
    msgs, out = [{"role": "system", "content": sysmsg}], []
    for t in range(turns):
        if a.tel.abort:
            break
        msgs.append({"role": "user", "content": c.filler_tokens(2000, seed=2000 + t) +
                     f"\nTurn {t + 1}: take the last four-digit record id above, add its digits, multiply the sum by {t + 3}, and answer with the final number only."})
        s = c.stream(msgs, max_tokens=4096, temperature=0.6, chat_template_kwargs={"enable_thinking": True, "reasoning_effort": "high"})
        am = {"role": "assistant", "content": s["content"]}
        if s["reasoning"]:
            am["reasoning_content"] = s["reasoning"]
        msgs.append(am)
        out.append(dict(turn=t + 1, prompt_tokens=s["prompt_tokens"], cached_tokens=s["cached_tokens"], ttft=round(s["ttft"], 3),
                        ttfc=round(s["ttfc"], 3) if s["ttfc"] else None, reasoning_chars=len(s["reasoning"]), content=s["content"][:40]))
        print("warmthink", out[-1], flush=True)
    warm = out[1:]
    return dict(turns=out, warm_ttft_p50=statistics.median(w["ttft"] for w in warm) if warm else None,
                warm_ttft_max=max(w["ttft"] for w in warm) if warm else None,
                warm_cached_share_min=min((w["cached_tokens"] / w["prompt_tokens"] for w in warm if w["cached_tokens"] is not None), default=None))


def ph_exact(c, a):
    """Drafted replies vs TensorFold's serial reference ("draft": false) on the same request."""
    cases = [("code", bc.CODE, dict(temperature=0)), ("prose", bc.PROSE, dict(temperature=0)),
             ("structured", STRUCT, dict(temperature=0)),
             ("prose_t1_seed", bc.PROSE, dict(temperature=1.0, top_p=0.95, seed=1234)),
             ("code_t1_seed", bc.CODE, dict(temperature=1.0, top_p=0.95, seed=99))]
    rows = []
    for name, p, kw in cases:
        msgs = [{"role": "user", "content": "[exactness probe] " + p}]
        d, td = c.chat(msgs, max_tokens=384, **bc.THINK_OFF, **kw)
        s, ts = c.chat(msgs, max_tokens=384, draft=False, **bc.THINK_OFF, **kw)
        dt, st = d["choices"][0]["message"].get("content") or "", s["choices"][0]["message"].get("content") or ""
        n = next((i for i, (x, y) in enumerate(zip(dt, st)) if x != y), min(len(dt), len(st)))
        row = dict(case=name, same=dt == st, first_diff_char=None if dt == st else n, drafted_chars=len(dt), serial_chars=len(st),
                   drafted_s=round(td, 2), serial_s=round(ts, 2), drafted_tf=d.get("tensorfold", {}).get("drafts"),
                   serial_tf=s.get("tensorfold", {}).get("drafts"),
                   drafted_tokens=d["usage"]["completion_tokens"], serial_tokens=s["usage"]["completion_tokens"],
                   accepted=d.get("tensorfold", {}).get("accepted"), rounds=d.get("tensorfold", {}).get("rounds"))
        rows.append(row)
        print(json.dumps(row), flush=True)
    return dict(rows=rows, identical=f"{sum(r['same'] for r in rows)}/{len(rows)}")


bc.PHASES.update(speed2=ph_speed2, warmthink=ph_warmthink, exact=ph_exact)

# ---------------------------------------------------------------- stage 1 (2026-10-02): gate, speed3, prefill3, conc
import base64
import random
import hashlib
import io
import threading as _th

REF = REFERENCE_DIR / "gate-reference.json"
SAMPLED = dict(temperature=1.0, top_p=0.95)                      # T=1.0, top_p=0.95 (no top_k: TF default 20)
KKP = dict(temperature=0.7, top_p=0.95)                          # the KK speed protocol (bench_client.one_decode)
GATE_CASES = [
    ("code_t0", bc.CODE, dict(temperature=0)),
    ("prose_t0", bc.PROSE, dict(temperature=0)),
    ("struct_t0", STRUCT, dict(temperature=0)),
    ("prose_sampled_s1234", bc.PROSE, dict(SAMPLED, seed=1234)),
    ("code_sampled_s99", bc.CODE, dict(SAMPLED, seed=99)),
    ("prose_topk20_s7", bc.PROSE, dict(SAMPLED, top_k=20, seed=7)),
    ("prose_nucleus_s5", bc.PROSE, dict(SAMPLED, top_k=-1, seed=5)),
    ("think_low_s11", "A train leaves at 9:40 and arrives at 13:05. How long is the trip? Explain briefly.",
     dict(temperature=0.6, seed=11, chat_template_kwargs={"enable_thinking": True, "reasoning_effort": "low"})),
]
TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["c", "f"]}},
                         "required": ["city"]}}},
         {"type": "function", "function": {"name": "lookup", "description": "Look up a fact",
          "parameters": {"type": "object", "properties": {"topic": {"type": "string"}}, "required": ["topic"]}}},
         {"type": "function", "function": {"name": "add_event", "description": "Add a calendar event",
          "parameters": {"type": "object", "properties": {"title": {"type": "string"}, "date": {"type": "string"},
                         "time": {"type": "string"}}, "required": ["title", "date"]}}}]
TOOL_PROMPTS = ["What's the weather in Lisbon right now, in celsius?",
                "Use the lookup tool to find the population of Reykjavik.",
                "Put 'dentist' on my calendar for 2026-11-03 at 15:30.",
                "Is it raining in Osaka? Check with the tool.",
                "Look up who designed the Eddystone lighthouse, using the tool."]


def _png_b64():
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (256, 256), (255, 255, 255))
    d = ImageDraw.Draw(im)
    d.rectangle([20, 20, 120, 120], fill=(220, 20, 20))
    d.ellipse([140, 140, 236, 236], fill=(20, 60, 220))
    b = io.BytesIO()
    im.save(b, format="PNG")
    return base64.b64encode(b.getvalue()).decode()


def _sha(j):
    return (j.get("tensorfold") or {}).get("token_sha")


def _same_sha(left, right):
    return bool(left) and bool(right) and left == right


def ph_gate(c, a):
    """Compare drafted/serial, concurrent/solo and saved baseline token hashes, plus tool and image checks.
    --gate-write-ref establishes a baseline; it does not prove agreement with an absent reference."""
    out, fails = dict(cases={}, conc={}, tools={}, image={}), []
    REF = REFERENCE_DIR / f"gate-reference{a.ref_tag}.json"
    ref = json.loads(REF.read_text()) if REF.exists() else None
    out["reference_status"] = "baseline" if ref is None and a.gate_write_ref else "checked" if ref is not None else "missing"
    if ref is None and not a.gate_write_ref:
        fails.append("reference missing: first run the known baseline with --gate-write-ref")
    for name, p, kw in GATE_CASES:
        msgs = [{"role": "user", "content": "[gate] " + p}]
        extra = {} if "chat_template_kwargs" in kw else dict(bc.THINK_OFF)
        d, td = c.chat(msgs, max_tokens=384, **extra, **kw)
        s, ts = c.chat(msgs, max_tokens=384, draft=False, **extra, **kw)
        row = dict(drafted=_sha(d), serial=_sha(s), same=_same_sha(_sha(d), _sha(s)),
                   tokens=d["usage"]["completion_tokens"], drafted_s=round(td, 2), serial_s=round(ts, 2),
                   rounds=(d.get("tensorfold") or {}).get("rounds"))
        if not row["same"]:
            fails.append(f"{name}: drafted {row['drafted']} != serial {row['serial']}")
        if ref is not None and not _same_sha(ref.get("cases", {}).get(name), row["drafted"]):
            fails.append(f"{name}: missing or different reference hash")
            row["ref_differs"] = True
        out["cases"][name] = row
        print("gate", name, json.dumps(row), flush=True)
    # concurrent == solo: mixed prompts, sampled sampling with seeds, and greedy; every concurrent reply's hash equals
    # the same request alone (drafted), and alone equals serial on the first two
    for level in a.gate_levels:
        for label, kw in (("sampled", dict(SAMPLED)), ("greedy", dict(temperature=0))):
            specs = [((bc.CODE if i % 2 else bc.PROSE), (2000 + i // 2) if kw["temperature"] else None) for i in range(level)]
            def one(spec, draft=True, k=kw):
                body = dict(max_tokens=a.gate_tokens, ignore_eos=True, **bc.THINK_OFF, **k)
                if spec[1] is not None:
                    body["seed"] = spec[1]
                if not draft:
                    body["draft"] = False
                j, _ = c.chat([{"role": "user", "content": "[conc] " + spec[0]}], **body)
                return _sha(j)
            alone = {}
            for sp in specs:
                if sp not in alone:
                    alone[sp] = one(sp)
            serial_ok = [_same_sha(one(sp, draft=False), alone[sp]) for sp in list(alone)[:2]]
            with ThreadPoolExecutor(level) as ex:
                got = list(ex.map(one, specs))
            eq = [_same_sha(g, alone[sp]) for g, sp in zip(got, specs)]
            cell = dict(level=level, equal_alone=f"{sum(eq)}/{len(eq)}", serial_equal=f"{sum(serial_ok)}/{len(serial_ok)}",
                        alone={f"{'code' if sp[0] == bc.CODE else 'prose'}:{sp[1]}": h for sp, h in alone.items()})
            if not all(eq) or not all(serial_ok):
                fails.append(f"conc {label} x{level}: equal {cell['equal_alone']}, serial {cell['serial_equal']}")
            if ref is not None:
                for k2, h in cell["alone"].items():
                    if not _same_sha(ref.get("alone", {}).get(f"{label}:{k2}"), h):
                        fails.append(f"conc {label} {k2}: missing or different reference hash")
            out["conc"][f"{label}x{level}"] = cell
            print("gate conc", label, level, cell["equal_alone"], cell["serial_equal"], flush=True)
    for i, p in enumerate(TOOL_PROMPTS):
        j, _ = c.chat([{"role": "user", "content": p}], tools=TOOLS, max_tokens=512, temperature=0, **bc.THINK_OFF)
        msg = j["choices"][0]["message"]
        calls = [(t["function"]["name"], t["function"]["arguments"]) for t in (msg.get("tool_calls") or [])]
        row = dict(sha=_sha(j), calls=calls, ok=bool(calls))
        if not row["ok"]:
            fails.append(f"tool{i}: no call")
        if not row["sha"]:
            fails.append(f"tool{i}: token hash missing")
        if ref is not None and not _same_sha(ref.get("tools", {}).get(str(i)), row["sha"]):
            fails.append(f"tool{i}: missing or different reference hash")
        out["tools"][str(i)] = row
        print("gate tool", i, json.dumps(row), flush=True)
    j, _ = c.chat([{"role": "user", "content": [
        {"type": "text", "text": "Name the two shapes in this image and their colours, in one line."},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + _png_b64()}}]}],
        max_tokens=96, temperature=0, **bc.THINK_OFF)
    txt = (j["choices"][0]["message"].get("content") or "")
    out["image"] = dict(sha=_sha(j), text=txt[:200], ok=("red" in txt.lower() and "blue" in txt.lower()))
    if not out["image"]["ok"]:
        fails.append("image: colours not named")
    if not out["image"]["sha"]:
        fails.append("image: token hash missing")
    if ref is not None and not _same_sha(ref.get("image", {}).get("sha"), out["image"]["sha"]):
        fails.append("image: missing or different reference hash")
    print("gate image", json.dumps(out["image"]), flush=True)
    out["fails"], out["pass_"] = fails, not fails
    if a.gate_write_ref and not fails:
        REF.parent.mkdir(parents=True, exist_ok=True)
        REF.write_text(json.dumps(dict(cases={k: v["drafted"] for k, v in out["cases"].items()},
                                       alone={f"{lab.split('x')[0]}:{k}": h for lab, cell in out["conc"].items() for k, h in cell["alone"].items()},
                                       tools={k: v["sha"] for k, v in out["tools"].items()}, image=dict(sha=out["image"]["sha"]),
                                       written=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), id=a.id), indent=1))
        print("reference written", REF, flush=True)
    status = "FAIL" if fails else "BASELINE" if out["reference_status"] == "baseline" else "PASS"
    print("GATE", status, fails, flush=True)
    return out


SEEDS = (101, 202, 303)                                         # fixed request seeds across runs


def _fixed(c, prompt, n, kw, seed=None):
    """Measure a fixed-length streamed reply with an unchanged prompt when seeded.
    Matching prompts and seeds do not guarantee matching tokens across engines or sampling defaults."""
    msg = prompt if seed is not None else bc.nonce() + prompt
    extra = dict(seed=seed) if seed is not None else {}
    s = c.stream([{"role": "user", "content": msg}], max_tokens=n, ignore_eos=True, **bc.THINK_OFF, **kw, **extra)
    ct, dec = s["completion_tokens"] or 0, (s["t_last"] or 0) - (s["ttft"] or 0)
    return dict(ttft=round(s["ttft"] or -1, 3), ct=ct, tps=round((ct - 1) / dec, 1) if dec > 0 else None)


def ph_speed3(c, a):
    """Single stream (code, prose, structured; 3 runs, median; 512 tokens fixed) under the KK protocol's sampling
    (T 0.7, top_p 0.95), sampled's (T 1.0, top_p 0.95, no top_k) and full nucleus (top_k -1); aggregates (sampled) at
    the given levels, mixed code/prose, 3 reps, median. sampled with no top_k == top_k 20 on TensorFold (its default)."""
    R = {}
    _fixed(c, "Say hi.", 16, SAMPLED)
    for sname, kw in (("kk", KKP), ("sampled", SAMPLED), ("nucleus", dict(SAMPLED, top_k=-1))):
        if sname not in a.speed_variants:
            continue
        for name, p in (("code", bc.CODE), ("prose", bc.PROSE), ("structured", STRUCT)):
            runs = [_fixed(c, p, 512, kw, seed=sd) for sd in SEEDS]
            R[f"{sname}_{name}"] = dict(runs=runs, tps_median=statistics.median(r["tps"] for r in runs),
                                        ttft_median=statistics.median(r["ttft"] for r in runs))
            print(sname, name, R[f"{sname}_{name}"]["tps_median"], [r["tps"] for r in runs], flush=True)
        R[f"{sname}_single"] = round(statistics.mean([R[f"{sname}_code"]["tps_median"], R[f"{sname}_prose"]["tps_median"]]), 1)
        print(sname, "single (mean of code and prose medians)", R[f"{sname}_single"], flush=True)
    for conc in a.agg_levels:
        reps = []
        for rep in range(3):
            t = time.time()
            with ThreadPoolExecutor(conc) as ex:
                res = list(ex.map(lambda i: _fixed(c, (bc.CODE if i % 2 else bc.PROSE) + f" (variant {i})", 512, SAMPLED,
                                                   seed=SEEDS[rep] + i), range(conc)))
            wall = time.time() - t
            reps.append(dict(aggregate_tps=round(sum(r["ct"] for r in res) / wall, 1),
                             per_stream_median=statistics.median(r["tps"] for r in res if r["tps"]),
                             per_stream_min=min(r["tps"] for r in res if r["tps"]),
                             ttft_median=statistics.median(r["ttft"] for r in res), ttft_max=max(r["ttft"] for r in res)))
        R[f"agg{conc}"] = dict(reps=reps, aggregate_tps=statistics.median(r["aggregate_tps"] for r in reps),
                               per_stream_median=statistics.median(r["per_stream_median"] for r in reps),
                               ttft_max=statistics.median(r["ttft_max"] for r in reps))
        print("agg", conc, R[f"agg{conc}"]["aggregate_tps"], [r["aggregate_tps"] for r in reps], flush=True)
    return R


def ph_prefill3(c, a):
    """Cold TTFT (fresh nonce each), 3 runs each at 32K, 100K and 140K, median."""
    R = {}
    for n in a.prefill_sizes:
        runs = []
        for k in range(3):
            s = c.stream([{"role": "user", "content": bc.nonce() + c.filler_tokens(n, seed=n) + "\n\nReply with the single word: done."}],
                         max_tokens=8, **bc.THINK_OFF)
            runs.append(dict(prompt_tokens=s["prompt_tokens"], ttft=round(s["ttft"], 2), cached=s["cached_tokens"]))
        med = statistics.median(r["ttft"] for r in runs)
        R[f"cold_{n // 1000}k"] = dict(runs=runs, ttft_median=med, prefill_tps=round(runs[0]["prompt_tokens"] / med))
        print("cold", n, med, [r["ttft"] for r in runs], R[f"cold_{n // 1000}k"]["prefill_tps"], flush=True)
    return R


bc.PHASES.update(gate=ph_gate, speed3=ph_speed3, prefill3=ph_prefill3)

REFLONG = REFERENCE_DIR / "gatelong-reference.json"


def ph_gatelong(c, a):
    """Long-prompt exactness (the prompt-chunk features: row split exchanges, lanes, index split, chunk size): fixed
    prompts of about 8K, 32K and 100K tokens (fixed filler, no nonce), a greedy reply cold, then a seeded T 1.0 reply
    on the same prompt (resumed from the kept prompt state). Token hashes must equal references/gatelong-reference.json."""
    REFLONG = REFERENCE_DIR / f"gatelong-reference{a.ref_tag}.json"
    ref = json.loads(REFLONG.read_text()) if REFLONG.exists() else None
    out, fails = {}, []
    reference_status = "baseline" if ref is None and a.gate_write_ref else "checked" if ref is not None else "missing"
    if ref is None and not a.gate_write_ref:
        fails.append("reference missing: first run the known baseline with --gate-write-ref")
    for n in (8000, 32000, 100000):
        prompt = c.filler_tokens(n, seed=424242 + n) + "\n\nQuote the last record id above, then name three words from it."
        for label, kw in (("greedy", dict(temperature=0)), ("seeded", dict(SAMPLED, seed=77))):
            j, dt = c.chat([{"role": "user", "content": prompt}], max_tokens=48, **bc.THINK_OFF, **kw)
            row = dict(sha=_sha(j), prompt_tokens=j["usage"]["prompt_tokens"],
                       cached=(j["usage"].get("prompt_tokens_details") or {}).get("cached_tokens"), s=round(dt, 2),
                       text=(j["choices"][0]["message"].get("content") or "")[:80])
            key = f"{n // 1000}k_{label}"
            if not row["sha"]:
                fails.append(f"{key}: token hash missing")
            if ref is not None and not _same_sha(ref.get(key), row["sha"]):
                fails.append(f"{key}: missing or different reference hash")
                row["ref_differs"] = True
            out[key] = row
            print("gatelong", key, json.dumps(row), flush=True)
    if a.gate_write_ref and not fails:
        REFLONG.parent.mkdir(parents=True, exist_ok=True)
        REFLONG.write_text(json.dumps({**{k: v["sha"] for k, v in out.items()}, "id": a.id}, indent=1))
        print("long reference written", REFLONG, flush=True)
    status = "FAIL" if fails else "BASELINE" if reference_status == "baseline" else "PASS"
    print("GATELONG", status, fails, flush=True)
    return dict(rows=out, fails=fails, pass_=not fails, reference_status=reference_status)


bc.PHASES.update(gatelong=ph_gatelong)

def ph_peak1m(c, a):
    """S7 memory peak: a needle prompt of about TF size (default 1,000,000 tokens: three codes at 1/6, 1/2, 5/6) while
    4 decode streams run and one image request arrives; the per-GPU memory maximum comes from the phase telemetry
    (nvidia-smi every second). Recall is reported too (the needle at full context)."""
    n = a.peak_tokens
    rnd = random.Random(n)
    codes = [f"{rnd.choice(['KESTREL', 'BASALT', 'HERON', 'ZEPHYR'])}-{rnd.randint(1000, 9999)}" for _ in range(3)]
    doc = bc.nonce()
    for k in range(3):
        part = c.filler_tokens(n // 3 - 300, seed=n + k)
        cut = len(part) // 2
        doc += part[:cut] + f" SECRET VAULT CODE NUMBER {k + 1} IS {codes[k]}. " + part[cut:] + " "
    q = doc + "\n\nList the three secret vault codes (numbers 1, 2 and 3) exactly as written, one per line as: code1=..., code2=..., code3=..."
    stop = _th.Event()
    side = []

    def streams(i):
        while not stop.is_set():
            r = _fixed(c, (bc.CODE if i % 2 else bc.PROSE) + f" (peak {i})", 256, SAMPLED, seed=500 + i)
            side.append(r["ct"])

    def image():
        time.sleep(30)
        j, _ = c.chat([{"role": "user", "content": [
            {"type": "text", "text": "Name the two shapes in this image and their colours, in one line."},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + _png_b64()}}]}],
            max_tokens=64, temperature=0, **bc.THINK_OFF)
        side.append(("image", (j["choices"][0]["message"].get("content") or "")[:80]))

    ths = [_th.Thread(target=streams, args=(i,), daemon=True) for i in range(4)] + [_th.Thread(target=image, daemon=True)]
    for t in ths:
        t.start()
    t0 = time.time()
    s_ = c.stream([{"role": "user", "content": q}], max_tokens=300, temperature=0.0, **bc.THINK_OFF)
    stop.set()
    for t in ths:
        t.join(timeout=120)
    hits = sum(x in s_["content"] for x in codes)
    out = dict(prompt_tokens=s_["prompt_tokens"], hits=f"{hits}/3", ttft=round(s_["ttft"], 1), wall=round(time.time() - t0, 1),
               content=s_["content"][:160], side_requests=len([x for x in side if not isinstance(x, tuple)]),
               image=[x for x in side if isinstance(x, tuple)])
    print("peak1m", json.dumps(out), flush=True)
    return out


bc.PHASES.update(peak1m=ph_peak1m)



bc.ORDER[:] = ["smoke", "gate", "gatelong", "peak1m", "speed3", "prefill3", "speed2", "speed", "prefill", "warm", "warmthink", "needle", "cachecheck", "refusal", "qualify", "exact", "soak"]

def _stage1_args():
    """Stage-1 options, taken out of argv before bench_client parses it."""
    import argparse
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--gate-levels", default="4")
    ap.add_argument("--gate-tokens", type=int, default=256)
    ap.add_argument("--gate-write-ref", action="store_true")
    ap.add_argument("--agg-levels", default="2,4,8")
    ap.add_argument("--speed-variants", default="kk,sampled,nucleus")
    ap.add_argument("--prefill-sizes", default="32000,100000,140000")
    ap.add_argument("--peak-tokens", type=int, default=1000000)
    ap.add_argument("--ref-tag", default="", help="gate references references/gate-reference<TAG>.json (per checkpoint)")
    known, rest = ap.parse_known_args(sys.argv[1:])
    sys.argv[1:] = rest
    return known


if __name__ == "__main__":
    S1 = _stage1_args()
    if not any(arg == "--model" or arg.startswith("--model=") for arg in sys.argv):
        sys.argv += ["--model", os.environ.get("TF_BENCH_MODEL", "glm-5.3-flash")]
    if not any(arg == "--base" or arg.startswith("--base=") for arg in sys.argv):
        sys.argv += ["--base", os.environ.get("TF_BENCH_BASE", "http://127.0.0.1:8020/v1")]
    # Helpers append /v1 to the origin; accept either form from callers.
    for i, arg in enumerate(sys.argv):
        if arg == "--base" and i + 1 < len(sys.argv):
            sys.argv[i + 1] = sys.argv[i + 1].rstrip("/").removesuffix("/v1")
        elif arg.startswith("--base="):
            sys.argv[i] = "--base=" + arg.split("=", 1)[1].rstrip("/").removesuffix("/v1")
    _orig_main_parse = None
    import argparse as _ap
    _real_parse = _ap.ArgumentParser.parse_args

    def _parse(self, *x, **k):
        ns = _real_parse(self, *x, **k)
        ns.gate_levels = [int(v) for v in S1.gate_levels.split(",") if v]
        ns.gate_tokens, ns.gate_write_ref = S1.gate_tokens, S1.gate_write_ref
        ns.agg_levels = [int(v) for v in S1.agg_levels.split(",") if v]
        ns.speed_variants = S1.speed_variants.split(",")
        ns.prefill_sizes = [int(v) for v in S1.prefill_sizes.split(",") if v]
        ns.peak_tokens = S1.peak_tokens
        ns.ref_tag = S1.ref_tag
        return ns
    _ap.ArgumentParser.parse_args = _parse
    bc.main()
