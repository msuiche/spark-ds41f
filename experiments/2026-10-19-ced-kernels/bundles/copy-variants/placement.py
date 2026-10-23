"""The large bf16 copies across allocation placements: each instance first allocates a
random-size spacer so its inputs land elsewhere; candidate and Triton-against-itself arms."""
import json
import random
import statistics

import torch

import bench

random.seed(1019)
SHAPES = [
    dict(rows=4048, retained=128, padding=64, dtype="torch.bfloat16", tail=(4, 5120), scatter=False),
    dict(rows=4048, retained=128, padding=0, dtype="torch.bfloat16", tail=(4, 5120), scatter=False),
    dict(rows=4048, retained=128, padding=64, dtype="torch.bfloat16", tail=(5120,), scatter=False),
    dict(rows=4048, retained=128, padding=64, dtype="torch.bfloat16", tail=(4, 5120), scatter=True),
]
builds = {}
for kernel, shape, build in bench.copy_cases():
    key = {k: shape[k] for k in ("rows", "retained", "padding", "dtype", "tail", "scatter")}
    for wanted in SHAPES:
        if key == wanted:
            builds[json.dumps(wanted)] = build

for wanted in SHAPES:
    build = builds[json.dumps(wanted)]
    for arm in ("candidate", "aa"):
        changes, references = [], []
        for i in range(24):
            spacer = torch.empty(random.randrange(1, 64 << 20), dtype=torch.uint8, device="cuda")
            samples, change = bench.instance(build, i % 2, arm == "aa")
            del spacer
            changes.append(100 * change)
            references.append(statistics.mean(samples[0]))
        print(json.dumps(dict(shape=wanted, arm=arm, mean_pct=statistics.mean(changes),
                              ci95_pct=2.069 * statistics.stdev(changes) / 24 ** 0.5,
                              changes=[round(c, 2) for c in changes],
                              reference_us=[round(r, 2) for r in references])), flush=True)
