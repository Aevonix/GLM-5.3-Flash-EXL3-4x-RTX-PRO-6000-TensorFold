#!/usr/bin/env python3
"""Fixed-prompt answer-agreement set: the quality check for a change that alters bits (a new reference).

40 fixed prompts with known short answers (arithmetic and word problems, facts, code output, logic, format), greedy
(temperature 0), thinking off, 160 tokens, "final answer only". Records every reply; scores each against its known
answer (normalized match or containment); with --compare REF.json also reports how many replies keep the same final
answer and how many are byte-identical to the reference run (a bit-identical build gives 40/40 identical).

    python3 agree.py --base http://127.0.0.1:8020/v1 --model glm-5.3-flash --out results/<arm>/agree.json [--compare REF]
"""
import os
import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

SET = [
    ("What is 17 * 23?", "391"),
    ("What is 1234 + 5678?", "6912"),
    ("What is 144 divided by 12?", "12"),
    ("What is 2 to the power of 10?", "1024"),
    ("What is 15% of 260?", "39"),
    ("A train travels 180 km in 2 hours. What is its average speed in km per hour?", "90"),
    ("I have 3 boxes with 12 apples each and I eat 5 apples. How many apples are left?", "31"),
    ("What is the next number in the sequence 2, 6, 12, 20, 30?", "42"),
    ("How many minutes are there in 3.5 hours?", "210"),
    ("What is the sum of the integers from 1 to 100?", "5050"),
    ("A rectangle is 7 cm by 9 cm. What is its area in square centimetres?", "63"),
    ("What is 81 minus 19, multiplied by 2?", "124"),
    ("If x + 7 = 19, what is x?", "12"),
    ("How many days are in a leap year?", "366"),
    ("What is the greatest common divisor of 48 and 180?", "12"),
    ("What is the capital of Australia?", "canberra"),
    ("What is the chemical symbol for gold?", "au"),
    ("Which planet is known as the Red Planet?", "mars"),
    ("In which year did the Apollo 11 mission land on the Moon?", "1969"),
    ("What is the largest ocean on Earth?", "pacific"),
    ("Who wrote the play Romeo and Juliet?", "shakespeare"),
    ("What gas do plants absorb from the air for photosynthesis?", "carbon dioxide"),
    ("What is the boiling point of water at sea level in degrees Celsius?", "100"),
    ("How many sides does a hexagon have?", "6"),
    ("What is the smallest prime number?", "2"),
    ("What does this Python print? print(len('lighthouse'))", "10"),
    ("What does this Python print? print(sorted([3, 1, 2])[-1])", "3"),
    ("What does this Python print? print('abc'[::-1])", "cba"),
    ("What does this Python print? print(sum(range(5)))", "10"),
    ("What does this Python print? print(7 // 2, 7 % 2)", "3 1"),
    ("All cats are animals. Tom is a cat. Is Tom an animal? Answer yes or no.", "yes"),
    ("If it is Tuesday today, what day will it be in 3 days?", "friday"),
    ("Which is heavier: one kilogram of feathers or one kilogram of iron? Answer feathers, iron, or neither.", "neither"),
    ("Alice is taller than Bob. Bob is taller than Carol. Who is the shortest?", "carol"),
    ("A shop sells pens at 3 for 2 dollars. How many dollars do 12 pens cost?", "8"),
    ("Spell the word 'beacon' backwards, in lowercase.", "nocaeb"),
    ("Write the word 'harbor' in uppercase.", "HARBOR"),
    ("Give the three primary colors of light, comma separated, lowercase, in the order red, green, blue.", "red, green, blue"),
    ("What is the first letter of the English alphabet that is a vowel after 'a'?", "e"),
    ("Convert 5 kilometres to metres.", "5000"),
]


def norm(s):
    return re.sub(r"[^a-z0-9 ,.-]", "", s.lower()).strip(" .")


def ask(base, model, q):
    body = dict(model=model, messages=[{"role": "system", "content": "Answer with the final answer only, no explanation."},
                                       {"role": "user", "content": q}],
                max_tokens=160, temperature=0, chat_template_kwargs={"enable_thinking": False})
    r = requests.post(base.rstrip("/").removesuffix("/v1") + "/v1/chat/completions", json=body, timeout=600)
    r.raise_for_status()
    return (r.json()["choices"][0]["message"].get("content") or "").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.environ.get("TF_BENCH_BASE", "http://127.0.0.1:8020/v1"))
    ap.add_argument("--model", default=os.environ.get("TF_BENCH_MODEL", "glm-5.3-flash"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--compare", default="")
    a = ap.parse_args()
    with ThreadPoolExecutor(4) as ex:
        replies = list(ex.map(lambda item: ask(a.base, a.model, item[0]), SET))
    rows = []
    for (q, want), rep in zip(SET, replies):
        n, w = norm(rep), norm(want)
        rows.append(dict(q=q, want=want, reply=rep[:200], correct=(n == w or w in n.split() or (len(w) > 2 and w in n))))
    out = dict(correct=sum(r["correct"] for r in rows), of=len(rows), rows=rows)
    if a.compare:
        ref = json.loads(Path(a.compare).read_text())["rows"]
        same_text = sum(r["reply"] == x["reply"] for r, x in zip(rows, ref))
        same_answer = sum(norm(r["reply"]) == norm(x["reply"]) or r["correct"] == x["correct"] == True  # noqa: E712
                          for r, x in zip(rows, ref))
        out.update(identical=f"{same_text}/{len(rows)}", same_answer=f"{same_answer}/{len(rows)}",
                   ref_correct=sum(x["correct"] for x in ref), changed=[r["q"] for r, x in zip(rows, ref) if r["reply"] != x["reply"]])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}))
    sys.exit(0)


if __name__ == "__main__":
    main()
