## GLP spec v3 fold-site steering on the 1022 lane

### What

Ports the GLP `spec_version 3` / `hc_fold_pre_engram` hook to the 1022
candidate tree and deploys it as the lane's steering mechanism, replacing
the r6v residual-stream hook (`residual_stream_post_layer`, GLP v2),
which steered the loop-level `hidden_states` — on this tree that tensor is
the layer's pending FFN write *pre-fold*, the wrong site for a vector
calibrated at the fold (the spec's warning: adjacent sites degrade
silently rather than fail).

### The hooking function (what actually runs per token)

Inside `DeepseekV4DecoderLayer.forward`, at the top of every layer with an
incoming residual — **after** the mHC fold materializes, **before** Engram
injection:

```python
residual = self.mhc.post(x, residual, post_mix, res_mix)   # the fold
_ds41_h = residual.mean(dim=1)                            # h = single stream
residual = residual - (
    (alpha * (_ds41_h @ direction)).unsqueeze(-1) * direction
).unsqueeze(1)                                            # every hc copy
```

The mean over the 4 hc copies shifts by exactly `alpha*(h.d)d` — no
stochasticity assumption. A second identical application runs at the
post-loop fold (the last layer's FFN write → `direction[end_layer-1]`).
Layer mapping per the spec: layer *i*'s start-fold consumes layer
*i-1*'s FFN write, so it applies `direction[i-1]`; row 0 stays zero
(`direction.0` is invalid). The **mandatory companion** attention writer
(`attn._o_proj` output, same stack and alpha) is armed in
`_ds41_load_steer` — spec v3 refuses a fold-only reader by construction.

On this tree the fold is normally **fused into `mhc.pre`** and only
materializes on Engram layers, so the armed path reuses the tree's own
Engram materialization ("do not fuse across it") at every layer.

### Disabled by default — off is byte-identical stock

Until `WEIGHTLESS_STEER_PATH` (or a ctrl `steer_path`) arms a vector,
`_ds41_steer_on` is `False` and the fold sites run the tree's **original
fused path unchanged** — zero extra kernels, no CUDA-graph delta, the
lane's zero-overhead-when-off discipline (the r6v lesson). `ctrl
{"off": true}` restores the fused path at runtime. Fail-closed: patch
anchors are exact; drift aborts the build, vector width/layer checks
refuse rather than run unsteered.

### Deployment

- `scripts/hotfix-dsv41-fold-steering.py` — the fail-closed patcher
  (model.py, attention.py, both model runners for the ctrl poll).
- `docker/Dockerfile.glp27` — image layer on the pinned 1022-4 base.
- `config/cluster.json` — GLP-27 at `WEIGHTLESS_STEER_ALPHA=3.5`,
  `DSPARK_STEER_CTRL/ACK` live-alpha channel on `/cache`.
- Vector: `msuiche/…-GLP-27-L11-37-a3.5` (spec-v3 validated,
  `content_sha256` recomputes clean; base pinned to `1bae0e1…`).

Verified live on the four-node ring (2026-10-31): TP4 serving with
DSpark speculative decoding, `dspark fold-site steering active:
alpha=3.500`, CUDA graphs captured, steer-ctrl round-trip acked.

### Not in this PR

- `upstreams.lock.json` / source-manifest alignment (1022 series + tree
  labels) rides along as `site:` commits on the branch; review whether
  the lock belongs in the release flow instead.
- Branch is fork-side (`msuiche/spark-ds41f`); `cluster sync` still only
  deploys origin/main — fold-branch rollouts detach checkouts manually.
