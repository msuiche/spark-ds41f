"""Gate item 7: the correction arm's greedy tokens with speculation against without.

runs/gate/identity-correction-ranks.jsonl (speculation on, rank check on) and
runs/gate/identity-correction-nospec.jsonl (speculation off, port 8001), from identity.py: the 128
screen prompts and the 16 collapse probes, temperature 0. Identical token ids are the pass.

Usage: python3 analysis/gate_identity.py
"""

import json
from pathlib import Path

RUNS = Path(__file__).resolve().parents[1] / "runs" / "gate"


def load(label: str) -> dict:
    out = {}
    for line in (RUNS / f"identity-{label}.jsonl").read_text().splitlines():
        if line.startswith("{") and '"summary"' not in line:
            record = json.loads(line)
            out[record["id"]] = record
    return out


spec, nospec = load("correction-ranks"), load("correction-nospec")
same = [i for i in spec if spec[i]["tokens"] == nospec.get(i, {}).get("tokens")]
print(f"{len(same)}/{len(spec)} replies identical with and without speculation")
for i in spec:
    if i in same:
        continue
    a, b = spec[i]["tokens"], nospec[i]["tokens"]
    first = next((k for k, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    print(f"  {i}: first difference at reply token {first} of {len(a)} / {len(b)}; "
          f"flags {(spec[i]['sink_rejection'] or {}).get('flagged_rows')} / {(nospec[i]['sink_rejection'] or {}).get('flagged_rows')}")
for i in ("p00", "p02"):
    print(f"{i}: speculation {spec[i]['sink_rejection']}, no speculation {nospec[i]['sink_rejection']}")
