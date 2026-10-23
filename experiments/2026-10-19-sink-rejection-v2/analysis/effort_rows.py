"""The sink channel at every sampled row of the effort-line variants' reference runs.

experiments/2026-10-19-effort-line-variants ran DeepSeek's CPU reference greedily (DeepSeek-V4.1-Flash-1017,
at most 120 tokens) on seven renderings of the collapsing prompts and on the 16-prompt set, recording for
every position |residual dimension 2455| after the last layer, the maximum over the four hyper-connection
streams: the value the vLLM detector reads (the pre-norm residual of the row, before the final collapse
and RMSNorm). A sampled row is the last prompt row (it samples the first token) or a generated token's row.

Usage: python3 effort_rows.py <out-dir> [threshold]   (dgx4: ~/spark-lab/think-collapse-effort/out)
"""

import json
import pathlib
import sys

out = pathlib.Path(sys.argv[1])
threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 50000.0


def variant(case: dict) -> str:
    effort = case.get("reasoning_effort")
    return "omit" if effort == "omit" else str(effort)


rows = []
for path in sorted(out.glob("*.json")):
    data = json.loads(path.read_text())
    summary = data.get("summary")
    if not summary or not summary.get("generated_positions"):
        continue
    case = data["case"]
    last_layer = data["scan"][0]["layers"][
        str(max(map(int, data["scan"][0]["layers"])))
    ]
    prompt_watch = last_layer["watch"]
    generated = summary["generated_positions"]
    sunk = [g for g in generated if g["watch_last_layer"] > threshold]
    clean = [g for g in generated if g["watch_last_layer"] <= threshold]
    clean_max = max(clean, key=lambda g: g["watch_last_layer"]) if clean else None
    prompt_max = summary["prompt_max_watch_last_layer_excluding_bos"]
    rows.append(
        dict(
            case=path.stem,
            variant=variant(case),
            prompt=case["messages"][-1]["content"],
            finished=summary.get("finished"),
            generated=len(generated),
            last_prompt_row=prompt_watch[-1],
            prompt_max=prompt_max,
            bos=summary.get("prompt_bos_watch_last_layer"),
            sunk=[(g["watch_last_layer"], g["text"]) for g in sunk],
            clean_max=(clean_max["watch_last_layer"], clean_max["text"])
            if clean_max
            else None,
        )
    )

print(f"threshold {threshold:g}; {len(rows)} runs with a summary in {out}")
print(
    f"{'case':28s} {'gen':>4s} {'finished':>8s} | {'last prompt row':>15s} | "
    f"{'prompt max (not BOS, not sampled)':>36s} | {'clean sampled max':>26s} | sunk sampled rows"
)
for r in rows:
    pm = r["prompt_max"]
    cm = r["clean_max"]
    print(
        f"{r['case']:28s} {r['generated']:4d} {str(r['finished']):>8s} | {r['last_prompt_row']:15,.0f} | "
        f"{pm['watch_last_layer']:>14,.0f} {pm['text']!r:>12s} at {pm['position']:3d} | "
        f"{(cm[0] if cm else 0):12,.0f} {(repr(cm[1]) if cm else ''):>13s} | "
        + ", ".join(f"{v:,.0f} {t!r}" for v, t in r["sunk"])
    )

sunk_values = [v for r in rows for v, _ in r["sunk"]]
clean_values = [r["clean_max"][0] for r in rows if r["clean_max"]]
print()
print(
    f"sampled rows above the threshold: {len(sunk_values)}, "
    f"{min(sunk_values):,.0f} to {max(sunk_values):,.0f}"
    if sunk_values
    else "no sampled row above it"
)
print(f"largest sampled row at or below it: {max(clean_values):,.0f}")
print(
    f"largest last prompt row (the prefill's sampled row): {max(r['last_prompt_row'] for r in rows):,.0f}"
)
print(
    "largest prompt row other than BOS (never sampled unless it is the last): "
    f"{max(r['prompt_max']['watch_last_layer'] for r in rows):,.0f}; BOS {max(r['bos'] for r in rows):,.0f}"
)
