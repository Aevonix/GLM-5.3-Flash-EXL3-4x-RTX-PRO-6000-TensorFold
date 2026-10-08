#!/usr/bin/env python3
"""First-token agreement and common-prefix length, not KL divergence. N fixed prompts of 2K-32K tokens built from TensorFold's open-source text (docs and Python sources of the
public v0.6.1 tree), each with one of several tasks; greedy, thinking off, 32 tokens, reply ids returned. One run
writes the ids; --compare reports first-token agreement, the mean common prefix (tokens) and full-match share.
    python3 first_tokens.py --base http://127.0.0.1:8020/v1 --out a.json [--compare b.json] [--n 100]"""
import os
import argparse, json, random, time, urllib.request
from pathlib import Path

SRC = Path(os.environ.get("TF_BENCH_SOURCE", "TensorFold"))
TASKS = ["Summarize the text above in three sentences.", "List the five most important names defined above.",
         "What is the main purpose of the text above? Answer in one paragraph.",
         "Write one question a reviewer would ask about the text above, then answer it.",
         "Continue the text above with the next two sentences.", "Translate the first sentence above into French."]


def corpus() -> list[str]:
    files = sorted(list((SRC / "docs").rglob("*.md")) + list((SRC / "src").rglob("*.py")) + [SRC / "README.md", SRC / "CHANGELOG.md"])
    return [f.read_text(errors="ignore") for f in files if f.is_file() and f.stat().st_size > 2000]


def prompts(n: int) -> list[dict]:
    rng = random.Random(20261002)
    texts = corpus()
    if not texts:
        raise ValueError("no source texts found; pass --source-tree with a TensorFold checkout")
    big = "\n\n".join(texts)
    out = []
    for i in range(n):
        tokens = rng.randint(2000, 32000)
        chars = tokens * 3                       # code and prose run about 3-4 characters a token
        start = rng.randint(0, max(0, len(big) - chars - 1))
        out.append({"id": i, "approx_tokens": tokens, "text": big[start:start + chars], "task": TASKS[i % len(TASKS)]})
    return out


def ask(base: str, model: str, p: dict) -> dict:
    body = {"model": model, "messages": [{"role": "user", "content": p["text"] + "\n\n" + p["task"]}],
            "max_tokens": 32, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False},
            "return_token_ids": True}
    req = urllib.request.Request(base.rstrip("/").removesuffix("/v1") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        j = json.loads(r.read())
    ids = (j.get("tensorfold") or {}).get("token_ids") or []
    return {"ids": ids, "prompt_tokens": j.get("usage", {}).get("prompt_tokens"), "s": round(time.time() - t, 2)}


def main() -> None:
    global SRC
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.environ.get("TF_BENCH_BASE", "http://127.0.0.1:8020/v1"))
    ap.add_argument("--model", default=os.environ.get("TF_BENCH_MODEL", "glm-5.3-flash"))
    ap.add_argument("--source-tree", type=Path, default=SRC, help="fixed TensorFold checkout used as the prompt corpus")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", required=True)
    ap.add_argument("--compare")
    a = ap.parse_args()
    SRC = a.source_tree
    res = {}
    for p in prompts(a.n):
        res[str(p["id"])] = ask(a.base, a.model, p)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res))
    print(f"{len(res)} prompts, prompt tokens {min(r['prompt_tokens'] or 0 for r in res.values())}-"
          f"{max(r['prompt_tokens'] or 0 for r in res.values())}")
    if a.compare:
        ref = json.loads(Path(a.compare).read_text())
        keys = [k for k in res if k in ref and res[k]["ids"] and ref[k]["ids"]]
        first = sum(res[k]["ids"][0] == ref[k]["ids"][0] for k in keys)
        def prefix(x, y):
            n = 0
            for u, v in zip(x, y):
                if u != v:
                    break
                n += 1
            return n
        pre = [prefix(res[k]["ids"], ref[k]["ids"]) for k in keys]
        full = sum(res[k]["ids"] == ref[k]["ids"] for k in keys)
        print(json.dumps({"prompts": len(keys), "first_token_agree": f"{first}/{len(keys)}",
                          "mean_common_prefix_tokens": round(sum(pre) / max(1, len(pre)), 2),
                          "all_32_equal": f"{full}/{len(keys)}"}))


if __name__ == "__main__":
    main()
