"""Served against reference, row by row, on the attribution run's flagged rows.

runs/attribution/survey-attribution-attribution.jsonl comes from survey.py --set attribution on the
detector-only arm (threshold 20,000) with vLLM 5ac8364ec, whose Sink: line names the output position
of the row holding each value. analysis/reference_sink_rows.json holds DeepSeek's CPU reference, teacher-forced
through the same greedy replies (d027 and d045 to reply token 232 and 238, x203 to 691). For every
served event whose row the reference covers, and whose served tokens equal the reference's up to that
row, both values of that one row and their ratio.

Usage: python3 analysis/attribution_compare.py
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
reference = json.loads((HERE / "reference_sink_rows.json").read_text())["cases"]
results = [json.loads(line) for line in
           (HERE.parent / "runs" / "attribution" / "survey-attribution-attribution.jsonl").read_text().splitlines()
           if line.startswith("{") and '"summary"' not in line]

ratios = []
for result in results:
    ref = reference[result["id"]]
    n = ref["prompt_len"]
    same_prompt = result["prompt_ids"] == ref["ids"][:n]
    reply = ref["ids"][n:]
    print(f"{result['id']}: {result['prompt'][:60]!r} served {result['generated_tokens']} tokens; "
          f"prompt ids equal: {same_prompt}; reference covers reply tokens 0-{len(reply) - 1}")
    for event in sorted(result.get("events", []), key=lambda e: e["row_position"]):
        q = event["row_position"]
        if q < 0 or q >= len(reply):
            print(f"  reply {q:4d}  served {event['value']:>7,}  (past the reference) {event['window']!r}")
            continue
        same = result["token_ids"][:q + 1] == reply[:q + 1]
        ref_value = ref["sink"][n + q]
        ratio = event["value"] / ref_value if ref_value else float("inf")
        if same:
            ratios.append(ratio)
        print(f"  reply {q:4d}  served {event['value']:>7,}  reference {ref_value:>7,}  ratio {ratio:5.2f}"
              f"{'' if same else '  (served text differs before this row: not compared)'}  {event['window']!r}")
if ratios:
    ratios.sort()
    print(f"\n{len(ratios)} rows compared: served/reference median {ratios[len(ratios) // 2]:.2f}, "
          f"range {ratios[0]:.2f}-{ratios[-1]:.2f}")
