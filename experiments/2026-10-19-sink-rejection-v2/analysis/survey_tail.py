"""The survey's high sink-column rows (runs/survey/): counts at or above each value, the largest
ordinary rows, the collapse rows, and the false-positive rate a threshold would have if the tail
kept falling as it does between 50,000 and 60,000.

Usage: python3 analysis/survey_tail.py [runs/survey]
"""

import json
import math
import sys
from pathlib import Path

directory = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parents[1] / "runs" / "survey")
summaries, events = {}, {}
for path in sorted(directory.glob("survey-*-survey.jsonl")):
    name = path.name.split("-")[1]
    events[name] = []
    for line in path.read_text().splitlines():
        if not line.startswith("{"):
            continue
        record = json.loads(line)
        if "summary" in record:
            summaries[name] = record["summary"]
        else:
            events[name] += [e["value"] for e in record.get("events", [])]

ordinary = sorted(events.get("default", []) + events.get("extra", []), reverse=True)
tokens = sum(summaries[n]["generated_tokens"] for n in ("default", "extra") if n in summaries)
print(f"ordinary text: {tokens:,} generated tokens ({', '.join(f'{n} {summaries[n]['requests']} requests' for n in ('default', 'extra') if n in summaries)})")
bins = (20000, 30000, 40000, 50000, 60000, 70000, 80000)
print("rows at or above:", ", ".join(f"{b:,}: {sum(v >= b for v in ordinary)}" for b in bins))
print("largest ordinary rows:", ", ".join(f"{v:,}" for v in ordinary[:5]))
print("collapse probes (sunk rows):", ", ".join(f"{v:,}" for v in sorted(events.get("probes", []), reverse=True)))
at50, at60 = sum(v >= 50000 for v in ordinary), sum(v >= 60000 for v in ordinary)
fall = math.log(at50 / at60) / 10000  # per unit of value
for threshold in (70000, 80000, 85000, 90000):
    expected = at60 * math.exp(-fall * (threshold - 60000))
    print(f"threshold {threshold:,}: about one false positive per {tokens / expected / 1e6:,.0f} M tokens "
          f"(tail falling {at50 / at60:.0f}x per 10,000)")
