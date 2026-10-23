"""Served against reference, position by position, on the x203 reply (prompt + greedy reply 0-691).

The reference side is analysis/x203_reference.json (DeepSeek's CPU reference, teacher-forced). The
served side is runs/x203-served-prompt-logprobs.json: production r6x's /v1/completions response for
the same ids with max_tokens 1, temperature 0 and prompt_logprobs 5 (a prefill, teacher-forced).

For every predicted position p (ids[p] from ids[:p]): the gap |served - reference| in the given
token's log-probability, and whether both put the same token first. Bins of 64 positions; the
before/after split at 128, V4.1's sliding window; and the predictions made from the rows whose
reference sink column is at least 20,000. Gaps up to about 3 nats are FP8 quantization noise
(the reference noise-floor work).

Usage: python3 analysis/x203_gap.py
"""

import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
reference = json.loads((HERE / "x203_reference.json").read_text())
served = json.loads((HERE.parent / "runs" / "x203-served-prompt-logprobs.json").read_text())
served_rows = served["choices"][0]["prompt_logprobs"]
ids, prompt_len, sink = reference["ids"], reference["prompt_len"], reference["sink"]
WINDOW, NOISE = 128, 3.0

rows = []
for entry in reference["positions"]:
    p = entry["p"]
    s = served_rows[p]
    given = s[str(ids[p])]
    served_top = min(s.items(), key=lambda kv: kv[1]["rank"])[0]
    rows.append({"p": p, "gap": abs(given["logprob"] - entry["logprob"]),
                 "served": given["logprob"], "reference": entry["logprob"],
                 "agree": int(served_top) == entry["top"][0][0]})


def describe(group: list[dict]) -> str:
    gaps = sorted(r["gap"] for r in group)
    return (f"n {len(group):3d}  median {statistics.median(gaps):.3f}  p90 {gaps[int(0.9 * (len(gaps) - 1))]:.3f}"
            f"  max {gaps[-1]:6.3f}  top-1 agree {100 * sum(r['agree'] for r in group) / len(group):5.1f}%"
            f"  over {NOISE:g} nats {sum(g > NOISE for g in gaps)}")


print(f"x203: {len(ids)} ids (prompt {prompt_len}, reply {len(ids) - prompt_len}); {len(rows)} predicted positions\n")
print("by 64-position bin (absolute position of the predicted token):")
for start in range(0, len(ids), 64):
    group = [r for r in rows if start <= r["p"] < start + 64]
    if group:
        print(f"  {start:4d}-{start + 63:4d}  {describe(group)}")
print(f"\nbefore the {WINDOW}-token window ends (p < {WINDOW}):  {describe([r for r in rows if r['p'] < WINDOW])}")
print(f"past it (p >= {WINDOW}):                       {describe([r for r in rows if r['p'] >= WINDOW])}")
print("\npredictions from rows whose reference sink column is >= 20,000 (row p predicts p+1):")
for p, value in enumerate(sink):
    if p > 0 and value >= 20000 and p + 1 < len(ids):
        r = rows[p]  # rows[k] predicts position k + 1
        print(f"  row {p:4d} (reply {p - prompt_len:4d}) sink {value:>7,}: next-token gap {r['gap']:.3f}"
              f" (served {r['served']:.3f}, reference {r['reference']:.3f}), top-1 {'agree' if r['agree'] else 'DIFFER'}")
worst = sorted(rows, key=lambda r: -r["gap"])[:8]
print("\nlargest gaps:", ", ".join(f"p {r['p']} ({r['gap']:.2f})" for r in worst))
