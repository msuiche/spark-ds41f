# CED ports and skip: full-model gates on the TP3 triangle

Two arms on the production r6x image and recipe (TP3, 1M), each a runtime overlay of
vLLM source with no test code:

- `ports-tp3.json`: the TileLang CED requests, row map and row copy, and the CED state
  that declares them (vLLM 1a15f6b6b, experiments/2026-10-19-ced-kernels patches
  0186-0211 without the sampling repair 0185).
- `skip-tp3.json`: the ports plus the nonfinal row skip (vLLM beb1b2972,
  experiments/2026-10-19-ced-skip).

**`gate.json`, determinism.** `token_determinism_text.py` with and without logprobs at two
long prompts: 700 lines (about 10.5K tokens, chunks ending at 4,048 and 8,096 both
skipped) and 2,200 lines (about 33K tokens, seven skipped chunks). Its mixed scenarios put
other streams' decode rows in the long prompt's skipped steps, which answers whether
shrinking one request's decoder rows changes another's bits. Gates:
- ports: every digest equals r6x's (experiments/2026-10-19-r6x-tp3-1m gate, 700 lines:
  62313cb5 / 9742ac2c with logprobs, 4d143977 / 12fd8257 text),
  and the ports boot through kernel preparation, so the declared copy layouts cover
  serving (an undeclared one stops its first CED step);
- skip: every digest equals the ports' at both lengths, every scenario identical.

Teacher-forced scoring cannot exercise the skip: a request with prompt logprobs keeps
every decoder row.

**`prefill.json`, the release-tp3 prefill step** (32K to 500K, three repeats; skip, ports,
skip again): the skip's prefill gain against the ports, with the repeat for drift. The
modeled ceiling at 128K is about 8-9% (experiments/2026-10-19-prefill-ceiling).

No promotion and no production change: the window restores the production profile.

## Results (2026-10-19, TP3 triangle at 1M, window 09:53-11:04 UTC)

Both arms booted through kernel preparation (ports +98.2 s, skip +94.0 s ready); no
undeclared CED kernel and no copy precondition failure in any run.

**Determinism** (`gate.json`): every digest and every scenario identical.

| Run | ports | skip | r6x |
| --- | --- | --- | --- |
| 700 lines, logprobs | 62313cb5 / 9742ac2c | same | 62313cb5 / 9742ac2c |
| 700 lines, greedy | 4d143977 / 12fd8257 | same | 4d143977 / 12fd8257 |
| 2,200 lines, logprobs | 62313cb5 / f6171075 | same | (not run) |
| 2,200 lines, greedy | 4d143977 / 60d90183 | same | (not run) |

All six scenarios of all eight runs have zero token and log-probability differences,
including mixed and mixed15, whose decode rows share steps with the long prompt's
skipped chunks: shrinking one request's decoder rows to one does not change another's
bits.

**Prefill** (`prefill.json`, release-tp3 sizes, three repeats; verdicts against the
run's own repeat, skip to skip again):

| Size | ports (tok/s) | skip (tok/s) | Change | ±95% | Repeat | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 32,768 | 4,392 | 4,858 | +10.61% | 6.94% | +0.41% | better |
| 131,072 | 4,391 | 4,876 | +11.06% | 0.55% | +0.38% | better |
| 262,144 | 4,193 | 4,649 | +10.86% | 0.19% | +0.45% | better |
| 500,000 | 3,833 | 4,239 | +10.57% | 0.28% | +0.42% | better |

The skip is about 11% faster at every size, above the modeled 8-9% at 128K. The ports
alone are level with r6x's release run of the same sizes (another session: 4,196 at
262,144, 3,835 at 500,000).

Not measured here: the ports' decode (engine grid, closed-loop streams), model behaviour,
memory and cold boot against r6x, and the full release tests. No promotion.
