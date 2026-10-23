# CED nonfinal row skip: CPU preparation

Candidate source 54c4fb11a. Startup requires TileLang, aligned chunked prefill, DSpark and a threshold covering the replay reach (199 tokens at k=5). Every multimodal request is ineligible. A skipped chunk records its promised next start and row count. A changed or shortened successor raises before metadata or GPU work; no replay-floor adjustment or environment switch remains.

This proposal is based on the pre-port CED call sites and is not wired into serving. It must be reconciled with the qualified TileLang ports before GPU/full-model gates. The simulation loads the actual candidate functions through AST and fingerprints target dependencies against no skip. Valid schedules have no target differences or unwritten target reads; 4,981 shortened promised successors fail before staging. Arbitrarily shortened unpromised chunks are a separate fault case and are not claimed deterministic.

GPU tests, teacher-forced identity, multi-request numerical identity, sampled prefix-hit acceptance and release prefill remain required. No promotion.

## Rebased on the TileLang CED ports (vLLM beb1b2972, on 1a15f6b6b)

`candidate.patch` is now the skip on the switched ports: `nonfinal_limits` feeds the
TileLang requests kernel's per-request limits through `CEDState.stage`, so the Triton
kernel changes of 54c4fb11a are gone. Unchanged: the startup requirements (TileLang,
aligned chunked prefill, DSpark, a chunk threshold covering the replay reach), multimodal
ineligibility, and the promised-successor fail-stop. TP3's threshold is 4,048 (4,096 - 8 x 6)
against a 199-row reach, so the production triangle qualifies.

**Simulation** (`ced_skip_sim.py`, the rewritten Python loaded unmodified; only the limit
constant's name follows the ports, `KEEP_ALL`):

| Sweep | Cases | Skipped | Target differences | Unwritten target reads |
| --- | ---: | ---: | ---: | ---: |
| exhaustive, scaled window (k = 1-6) | 337,699 | 196,904 | 0 | 0 |
| real constants (W 128, k 5 and 6, T to 24,464) | 10,930 | 5,841 | 0 | 0 |
| prefix hits on an earlier request's pages | 10,044 | 3,674 | 0 | 0 |

Fault injection: all 4,981 shortened promised successors stop before staging. (Its other
differences come from no-skip runs whose own chunking was cut, against an uncut baseline.)
The prefix-hit sweep's draft-side differences are item 4's question; serving's drafter
pages are private (experiments/2026-10-19-prefix-replay).

**Tests** (`bundles/skip-tests`: the skip's tests and the four release-gated CED files, a
test container on DGX4): 267 passed, again on the ports' follow-ups (1a15f6b6b). Two existing model-state tests now set the skip's
state, which they construct by hand.

Full-model gates: experiments/2026-10-19-ced-gates. No promotion.
