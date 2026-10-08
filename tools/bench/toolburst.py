#!/usr/bin/env python3
"""Tool-burst probe: the soak's burst request on its own (no other load), with and without "tool_choice": "required".

The request is exactly the Qwen-shape soak's (bench_client.soak_qwen_shape tools()): one user line "Use the lookup
tool to find the population of city #k in list n.", the single `lookup(topic)` tool, max_tokens 1024, thinking off,
no sampling fields (server defaults). Bursts n = 1..30 of three concurrent requests (k = 0, 1, 2) = the soak's 90
requests. A call counts when the first tool call carries a non-empty "topic" argument.

    python3 toolburst.py --base http://127.0.0.1:8020/v1 --model glm-5.3-flash --out results/<id>/toolburst.json
"""
import os
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

TOOLS = [{"type": "function", "function": {"name": "lookup", "description": "Look up a fact", "parameters": {
    "type": "object", "properties": {"topic": {"type": "string"}}, "required": ["topic"]}}}]


def one(base, model, n, k, required):
    body = dict(model=model, messages=[{"role": "user", "content": f"Use the lookup tool to find the population of city #{k} in list {n}."}],
                tools=TOOLS, max_tokens=1024, chat_template_kwargs={"enable_thinking": False})
    if required:
        body["tool_choice"] = "required"
    t = time.time()
    r = requests.post(base.rstrip("/").removesuffix("/v1") + "/v1/chat/completions", json=body, timeout=600)
    dt = time.time() - t
    if r.status_code != 200:
        return dict(n=n, k=k, ok=False, http=r.status_code, err=r.text[:200], s=round(dt, 2))
    ch = r.json()["choices"][0]
    tc = ch["message"].get("tool_calls") or []
    topic = None
    if tc:
        try:
            topic = json.loads(tc[0]["function"]["arguments"]).get("topic")
        except (ValueError, TypeError, AttributeError):
            topic = None
    row = dict(n=n, k=k, ok=bool(topic), finish=ch.get("finish_reason"), s=round(dt, 2))
    if not row["ok"]:
        row["content"] = (ch["message"].get("content") or "")[:160]
        row["tool_calls"] = json.dumps(tc)[:200]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.environ.get("TF_BENCH_BASE", "http://127.0.0.1:8020/v1"))
    ap.add_argument("--model", default=os.environ.get("TF_BENCH_MODEL", "glm-5.3-flash"))
    ap.add_argument("--bursts", type=int, default=30)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = {}
    for label, required in (("default", False), ("required", True)):
        rows = []
        for n in range(1, a.bursts + 1):
            with ThreadPoolExecutor(3) as ex:
                rows += list(ex.map(lambda k: one(a.base, a.model, n, k, required), range(3)))
        ok = sum(r["ok"] for r in rows)
        out[label] = dict(ok=f"{ok}/{len(rows)}", misses=[r for r in rows if not r["ok"]][:20],
                          s_median=sorted(r["s"] for r in rows)[len(rows) // 2])
        print(label, out[label]["ok"], "median s", out[label]["s_median"], flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
