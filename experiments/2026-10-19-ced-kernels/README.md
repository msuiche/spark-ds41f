# CED row-selection kernels

Base: r6x f3b58ed9d. First port: per-request integer kept-row limits and compaction starts, positions, replay floors and counts. Limits 128 and INT32_MAX reproduce the old 128/full-row interface. The module is not wired into serving until exactness and per-call timing pass.

Source module 304bd84a1. Tests cover empty and padded batches, zero/short/long requests, prefix floors and mixed keep-full flags at 1, 8, 16 and 32 requests.

First GPU attempt: the request kernel did not compile because TileLang inferred incompatible fragment layouts for a prefix entry and the total. No serving path was changed. Candidate 9772676cb moves the small prefix arrays to shared memory; the replay, index-map and row-copy gates follow. Keep the original failure output under results/private/lab/postlaunch/ced-requests on DGX4.

Second attempt: 30 request/index exactness cases pass. The generic row copier fails during specialization because a symbolic loop variable indexed a Python layout tuple; candidate 85e2e4a37 constructs the strided address at specialization time. Request metadata with the shared-memory scan is 10–24% slower per call across the measured shapes (15 alternating blocks, 95% intervals exclude zero); candidate 05f74aca2 uses a warp fragment scan and consistent lane indexing. Neither implementation is connected to serving.

Third attempt: all 40 metadata/row-copy cases pass. TileLang's fragment scan still lowers through shared memory and retains the timing regression. Candidate 614873756 uses explicit lane shuffles, with no shared-memory scan, and extends correctness coverage across two warp-sized request batches. The expanded timing gate measures actual live versus maximum request counts, index maps and gather/scatter rows (30 alternating blocks, no statistically supported per-call regression allowed).

Fourth timing attempt: all 67 benchmark shapes compile and match exactly. The index port is 10–31% faster. Identity request plans are level or 1–10% faster, but compacting plans are 3–14% slower; row copies are up to 128% slower. Neither rejected port is switched into serving. The next candidates specialize the CPU-known single-request case and coalesce four contiguous copy elements per thread.

## Result: all three ports pass and are switched (vLLM e4d57b3fb, patches 0204-0209)

**Why the request and copy ports were slower.** Their generated code branched on each
row's validity and loaded inside the branch (`sass/` disassembles both implementations).
An index load, the source pointer's constant load and the row load then ran in series,
element after element, where Triton issues every predicated load before any use. The
request kernels loaded sequence ends and prefix bounds after the scan or between stores.

**What changed (module commits 0205, 0206, 0208).**
- Requests: every input loads first, by TileLang's predicated loads (zero when off, as a
  dead lane is defined; volatile, so the compiler cannot sink them into the branch that
  uses them). Only compacting kernels specialize on one request.
- Row copy: loads issue before any branch, indices, then rows, then stores. Indices,
  whole-row pieces and gathered elements use predicated loads; the kernel defines the
  1-, 2- and 8-byte forms TileLang lacks or returns as a register pair. Element layouts
  keep Triton's coalescing; rows copy in 16-byte pieces when their tail is packed. The
  source is not restrict, because the read-only path is slower for these dependent loads.
  The row stride is an argument, a constant when rows are packed.
- A correctness fix: the old vector path loaded float32 rows before checking the index,
  so the plan's -1 padding read 16 bytes before the source on every step. Every load is
  now predicated or clamped in bounds. The copy raises when there is no row to copy or a
  whole-row source is not 16-byte aligned.

**The benchmark (0204).** Triton against itself showed the old intervals too narrow,
about one case in five read faster or slower by chance: two graphs of one kernel differ
by up to a percent, a graph's first replay is slow and the reference always went first,
and a large copy's time depends on where its inputs land (one Triton copy took 9.7 to
24 us across placements, `copy-variants/placement.py`). Each case now checks exactness
on separate outputs, then times both kernels on shared outputs over eight instances:
fresh inputs after a seeded random spacer, both graphs replayed untimed first, the first
graph alternating. The interval is over the eight instance differences; a gated case
that reads slower is re-measured on fresh placements and fails only if slower twice.
Triton against itself on the gated copies is reported with every run.

**Timing gate, `requests-bench`, two runs (DGX4, r6x image).** Change per call against
Triton; the gate is serving shapes (metadata on every step, copies under a CED plan, which
keeps at least 128 rows). Both runs pass; Triton against itself: 0 of 28 slower, 0 faster.

| Kernel | Cases | Change | Largest upper 95% bound |
| --- | ---: | ---: | ---: |
| requests (1-32 requests, 1 to 4,096 rows) | 56 | -24.0 to -2.0% | +0.0% |
| indices | 18 | -31.1 to -10.3% | -10.1% |
| copy gather, elements (int32, int64, bool) | 12 | -0.8 to +0.8% | +2.5% |
| copy gather, whole rows (float32, bf16) | 16 | -7.0 to -1.0% | -0.3% |
| copy scatter, elements | 12 | -19.7 to -12.7% | -10.4% |
| copy scatter, whole rows | 16 | -16.2 to -4.3% | -4.0% |
| copies below any plan (1 and 48 rows, not gated) | 56 | -20.1 to +5.7% | +5.9% |

The only slower cases are one-row bf16 gathers (about +5%), which no CED plan launches.

**Exactness.** `ced-tests` runs the four release-gated CED files (TileLang metadata, CED
state and packing, CED model, router) against the switched source: 257 passed. They add
the serving layouts with -1 padding and out-of-range indices across a block boundary,
views of wider rows, every 1- to 8-byte element type, the preconditions, and a startup
test that clears the CED kernels' in-memory caches before the freeze.

**Switches (0207, 0209).** CEDState stages per-request int32 limits (2^31-1 keeps every
row, 128 otherwise) and declares the three request specializations, the row map and the
packed copy layouts the deployment launches for preparation. The Triton request and copy
kernels and the copy warmup are removed; the benchmark keeps frozen copies as references.

**Follow-ups (0210, 0211).** The single-request and unchanged-row specializations compile
as `ced_requests` like the compacting one (one kernel name per kernel in traces, as the
indexer's specializations share theirs), and the CED kernel tests move to
`tests/kernels/test_deepseek_v41_tilelang_ced.py`, named after the module and benchmark.
Only a non-last pipeline stage scatters the float32 pre-mix, so that copy is declared only
under pipeline parallelism: a TP deployment prepares nine copy layouts, every one it can
launch. The series 0185-0211 on r6x f3b58ed9d reproduces vLLM tree 9f6715b7 (1a15f6b6b).

**Not ported:** `_gather_lookback` (model_state.py) is Engram's token-history window, not
CED row selection.

GPU gates on the full model (startup preparation of the declared layouts, determinism
against r6x) run in experiments/2026-10-19-ced-gates. No promotion.
