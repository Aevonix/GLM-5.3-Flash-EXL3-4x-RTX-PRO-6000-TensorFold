#!/usr/bin/env python3
"""Run the original 20 JSON contract cases, with and without response_format.

The original contract definitions are supplied with --contracts-file. Request
bodies, scoring and the JSON result schema match the measured client.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path

import requests


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("output", nargs="?", help="output JSON path (legacy positional form)")
    ap.add_argument("--out", help="output JSON path")
    ap.add_argument("--base", default=os.environ.get("JV_BASE", os.environ.get("TF_BENCH_BASE", "http://127.0.0.1:8020/v1")))
    ap.add_argument("--model", default=os.environ.get("JV_MODEL", os.environ.get("TF_BENCH_MODEL", "glm-5.3-flash")))
    ap.add_argument("--contracts-file", type=Path, default=os.environ.get("TF_BENCH_CONTRACTS"),
                    help="original qualify_contracts.py containing JSON_CASES and check_json")
    a = ap.parse_args()
    output = a.out or a.output
    if not output:
        ap.error("supply --out FILE or a positional output path")
    if not a.contracts_file or not a.contracts_file.is_file():
        ap.error("the original contract definitions are not bundled; supply --contracts-file FILE or TF_BENCH_CONTRACTS")
    spec = importlib.util.spec_from_file_location("q", a.contracts_file)
    q = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(q)
    base = a.base.rstrip("/").removesuffix("/v1") + "/v1"
    out = {}
    for variant, extra in (("thinking_off", {}), ("thinking_off_response_format", {"response_format": {"type": "json_object"}})):
        res = []
        for i in range(20):
            prompt, shape = q.JSON_CASES[i % len(q.JSON_CASES)]
            body = {"model": a.model, "messages": [{"role": "system", "content": "Reply with one JSON object and nothing else."},
                    {"role": "user", "content": prompt}], "max_tokens": 2000, "chat_template_kwargs": {"enable_thinking": False}, **extra}
            r = requests.post(base + "/chat/completions", json=body, timeout=300)
            res.append(q.check_json(r.json()["choices"][0]["message"], shape) if r.status_code == 200 else f"http_{r.status_code}")
        out[variant] = {"ok": res.count("ok"), "fenced": res.count("fenced"), "other": [x for x in res if x not in ("ok", "fenced")]}
        print(variant, out[variant], flush=True)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
