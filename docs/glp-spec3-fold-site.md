# GLP spec_version 3 and the fold-site vector (handoff, 2026-10-29/30)

What changed upstream in the weightless GLP format and in the published
dsv41 vectors, and what this lane needs to do about it.

## Format: spec_version 3

`weightless/docs/spec/GLP.md` (commit "GLP spec_version 3: hc fold site")
adds one hook value and one conformance rule.

- **`glp.hook_point = hc_fold_pre_engram`** — the post-fold, pre-Engram
  point on hyper-connection archs. On DeepSeek V4.1 the stream is
  `[T, 4, 5120]` and each layer's FFN write folds into it at the start of
  the *next* layer's forward; `residual_stream_post_layer` names no tensor
  that exists between layers on this arch.
- **Op**: `h = mean over the 4 copies`, then `copy_c -= alpha * (h . d) d`
  for every copy — shifts the mean by exactly `alpha*(h.d)d`, no
  stochasticity assumption.
- **Layer mapping**: `direction.N` steers the fold consuming layer N's FFN
  write (runs at the start of layer N+1; post-loop fold for the last
  layer). The `direction.N` ↔ layer N contract is unchanged;
  `direction.0` stays invalid.
- **Companion writer is mandatory**: the attention `o_proj` writer runs
  armed with the same stack and the same alpha. A reader that implements
  only the fold under-doses by construction and must refuse the file.
- spec_version 3 is required on any file whose hook names an hc fold site;
  version-2 keys (rank-k, `dir_scales`, `layer_scales`) stay valid under
  v3. Pre-3 readers refuse v3 files by design.

Tooling status: captain-vector 0.5.0 writes and validates v3
(`tools/captain-vector/captain_vector.py`); the vllm-plugin container
reader accepts v3 but implements no fold site yet, so a v3 file fails
closed on the hook match there — the intended behavior until a dsv41
plugin adapter lands. Container tests: 27/27.

## New vector: GLP-27 (recommended for this lane)

`msuiche/DeepSeek-V4.1-Flash-1017-abliterated-cyber-GLP-27-L11-37-a3.5`
(public, gated with the Responsible Use Agreement):

- `glp.deepseek-v41-flash-1017-GLP-27-L11-37-a3.5.gguf` — 544 KB,
  spec_version 3, `hc_fold_pre_engram`, `alpha_default=3.5`, base pinned
  to revision `1bae0e1088993b9b82de08f59de5be5a34e42170`,
  `content_sha256` `f0aaba2fd1d5d52deb605495e0b99b8e868d44a4e612c994688cf243a95dfcb4`.

The raw `.pt` stack the hotfix consumes (`{layer: tensor(5120)}`,
zero-padded outside L11–37) is not on HF; the working copy lives on the
Modal volume `dsv41-flash:out-1017/dsv41-perlayer-dirs-ffn.pt` (fetch with
`MODAL_PROFILE=tolmo modal volume get dsv41-flash
out-1017/dsv41-perlayer-dirs-ffn.pt <dst>`), or re-export one from the
GGUF — tensors are the same unit directions.

27 unit directions, L11–37, weight-SVD from the FFN writers (no activation
contrast at derivation). Measured on the Modal TP4 lane (dsv41-feat wheel,
greedy, thinking on): weaponize32 20/32 → **4/32 refused at alpha 3.5**;
refusal32 32 → 6/32; benign32-holdout 9/32 (stock 8/32, keyword noise).
Alpha 4.0 buys one more item but corrupts exact tokens (CVE-ID digit
repetition); 5.0 regresses on refusal count and coherence. Operating
point: **3.5**.

The patch implementing this hook:
`refusal-research/experiments/20261024-dsv41-1017-weaponize-glp/staging/patch_dsv41.py`
— fail-closed, V2-runner + PIECEWISE CUDA-graph safe, live alpha control
via `DSPARK_STEER_CTRL` JSON + ack file. Full measurement log:
`refusal-research/experiments/20261024-dsv41-1017-weaponize-glp/RESULT.md`.

## GLP-39 repo state (what was removed)

`msuiche/DeepSeek-V4.1-Flash-1017-abliterated-cyber-GLP-39-L1-39-a0.75`
now carries only the base file:

- kept: `glp.deepseek-v41-flash-1017-GLP-39-L1-39-a0.75.gguf` (spec v1,
  `residual_stream_post_layer`, alpha 0.75, contrast-derived)
- deleted 2026-10-26: `...-a0.75-stretch.gguf`
- deleted 2026-10-30: `...-a1.0-dealign-attn.gguf`

If any config here references the stretch or dealign-attn files, repoint.

## Decision for this lane

1. **Switch steering to GLP-27**: fetch `dsv41-perlayer-dirs-ffn.pt` from
   the GLP-27 repo, apply the fold-site patch to the serving wheel, set
   `WEIGHTLESS_STEER_PATH` and `WEIGHTLESS_STEER_ALPHA=3.5`.
2. **Or stay on GLP-39 a0.75** with the existing residual-stream patch —
   still published and valid at its own hook, but the weaker recipe on the
   weaponization suites.
3. Never carry alpha between the two recipes: different sites, different
   alpha semantics (the spec is explicit that hook changes recalibrate
   alpha).
