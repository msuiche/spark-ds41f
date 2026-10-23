# TP3 long-prefill budget with the CED skip (prefill follow-ups, item 5)

Each shape chooses its own `max_num_batched_tokens`; this sweep is tp3-triangle's, with
the skip (experiments/2026-10-19-ced-gates, both arms bit-identical to r6x; the skip +11%
on release-tp3 prefill at 4,096). A TP3 rank holds a third of the model, so its memory
headroom differs from TP4's; tp4-ring's sweep waits for the ring cabling.

Arms, the gates' skip overlays on production r6x TP3 at 1M, differing only in the budget
and its threshold (budget - 8 x 6): `skip-4096-tp3.json` (production's 4,096 / 4,048),
`skip-6144-tp3.json` (6,144 / 6,096), `skip-8192-tp3.json` (8,192 / 8,144). Without the
skip, 8,192 already booted past dgx1's startup guard and kept it 1.5 GiB above its running
guard through a 1M prefill (experiments/2026-10-19-r6x-tp3-1m), so no in-between arm is
needed for memory; 6,144 maps the trade between the two.

`ports-4096-tp3.json` is the gates' ports arm (no skip) at 4,096, the same-session
reference for the skip at 1M and for its effect on the stall.

`sweep.json` measures the skip at 4,096, the ports at 4,096, the skip at 6,144 and 8,192,
and the skip at 4,096 again (the repeat), with `suite.json`:
- release-tp3's prefill sizes, 32K to 500K, three repeats;
- `servebench stall`: what seven decoding streams wait while a cold 131K prompt
  prefills (TP3 serves eight sequences), three repeats: the worst wait (the longest step
  carrying a prefill chunk), the median and p99 step during the prefill, the baseline;
- three cold 1M prefills (TP3 serves 1M since r6x), during which the run's memory watch
  records dgx1's lowest MemAvailable against its 3 GiB guard.

Gate for a larger budget: dgx1 above its guard at boot and through the 1M prefill, and a
stall the owner accepts for the prefill it buys. tests/test_recipe.py pins each shape's
budget, so a change to one cannot move another. No promotion or production change.

## Results (2026-10-19 11:31-13:56 UTC, TP3 triangle at 1M)

Every arm booted past dgx1's 5 GiB startup guard. Verdicts against production's
scheduling without the skip (`ports4096`, outputs identical to r6x), checked by the
run's own repeat (skip at 4,096 against itself: -0.2 to -0.4%, the median stall step
+0.2%).

| Arm | 1M prefill (tok/s) | 32K-500K prefill | 131K under 7 streams | Median step during that prefill | dgx1 lowest MemAvailable |
| --- | ---: | ---: | ---: | ---: | ---: |
| ports4096 (no skip) | 3,216 | 4,359 / 4,355 / 4,163 / 3,806 | 4,102 | 0.949 s | 5.22 GiB |
| skip 4,096 | 3,569 (+10.95% ±0.25) | +12.2 / +12.7 / +12.4 / +12.1% | +10.1% ±0.9 | 0.857 s (-9.7% ±2.2) | 5.26 GiB |
| skip 6,144 | 3,797 (+18.05% ±0.28) | 5,309 / 5,338 / 5,068 / 4,588 | +23.1% ±1.1 | 1.143 s (+20.4% ±2.3) | 5.03 GiB |
| skip 8,192 | 3,907 (+21.46% ±0.29) | 5,473 / 5,543 / 5,253 / 4,749 | +28.7% ±1.5 | 1.444 s (+52.2% ±2.2) | 4.57 GiB |

The decode step without a prefill is 0.100 s in every arm. dgx1's lowest MemAvailable,
through three 1M prefills, stays 1.6-2.3 GiB above its 3 GiB running guard at every budget.

**Tails are not resolved.** The worst wait (1.64, 1.47, 1.62, 2.47 s) and the p99 step
during the prefill have 95% intervals of 72-182% over three repeats: one long step decides
each. Only the median step is measured to ±2.2%. Deciding on the tail needs many more
repeats of the stall step.

**Reading.** The skip alone is better on every resolved measure. A larger budget buys
prefill (+7% at 1M for 6,144 over the skip at 4,096, +3% more for 8,192) and costs the
other streams a longer step while a long prompt prefills (+33% and +68% over the skip at
4,096). Interleaving (experiments/2026-10-19-prefill-interleave-tp3) can shorten their
median wait at any budget. The budget is the owner's choice; recipe unchanged.
