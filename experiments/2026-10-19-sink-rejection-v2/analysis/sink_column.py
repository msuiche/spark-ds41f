"""Post-final-norm value of the sink channel for the sampled rows of the reference dumps.

Each dump holds, per start_pos, the residual after every layer at the last position: [40, 4 streams, 5120].
The reference combines the four streams with hc_pre at the last layer, then applies RMSNorm(dim, eps)
with the checkpoint's norm.weight. hc_pre's per-token weights are not saved, so this reports the
normalized value for each stream and for the stream sum (the mix is a convex-ish combination of them).
"""
import json, os, sys, torch
from safetensors import safe_open

SNAP = sys.argv[1]
DIM, COL, EPS = 5120, 2455, 1e-6
idx = json.load(open(os.path.join(SNAP, "model.safetensors.index.json")))["weight_map"]
with safe_open(os.path.join(SNAP, idx["norm.weight"]), "pt") as f:
    gamma = f.get_tensor("norm.weight").float()
print(f"gamma[{COL}] = {gamma[COL]:.4f}  |gamma| median {gamma.abs().median():.4f} max {gamma.abs().max():.4f}  sqrt(dim)={DIM**0.5:.2f}")
print(f"saturation sqrt(dim)*|gamma[col]| = {DIM**0.5*abs(gamma[COL]):.2f}")

def norm(x):
    x = x.float(); var = x.square().mean(-1, keepdim=True); return gamma * x * torch.rsqrt(var + EPS)

cases = [("k0 (Say hi., sunk ' can')", "/lab/think-collapse/out/ref-k0-hidden.pt", 0),
         ("k0 next row (\\n after sink)", "/lab/think-collapse/out/ref-k0-hidden.pt", 40),
         ("thanks (sunk)", "/lab/think-collapse4/out/experts-thanks-hidden.pt", 0),
         ("k5 clean ' can'", "/lab/think-collapse/out/ref-k5-hidden.pt", 0),
         ("k5 ' I'", "/lab/think-collapse/out/ref-k5-hidden.pt", 45),
         ("k5 ' help'", "/lab/think-collapse/out/ref-k5-hidden.pt", 46),
         ("k5 ' you'", "/lab/think-collapse/out/ref-k5-hidden.pt", 47),
         ("guard-k5 clean", "/lab/think-collapse6/out/guard-k5-hidden.pt", 0),
         ("guard-k5 45", "/lab/think-collapse6/out/guard-k5-hidden.pt", 45),
         ("guard-k5 46", "/lab/think-collapse6/out/guard-k5-hidden.pt", 46),
         ("guard-k5 47", "/lab/think-collapse6/out/guard-k5-hidden.pt", 47),
         ("experts-k5 clean", "/lab/think-collapse5/out/experts-k5-hidden.pt", 0)]
print(f"{'case':30s} {'layer':>5s} {'pre |x| max':>12s} {'pre x[col]':>11s} | post-norm x[col]: per stream | sum  | cos per stream | sum")
for name, path, sp in cases:
    d = torch.load(path, map_location="cpu")
    if sp not in d: continue
    h = d[sp]  # [40, 4, 5120]
    for layer in (38, 39):
        x = h[layer].float()  # [4, 5120]
        s = x.sum(0)
        post = norm(x)[:, COL].tolist(); posts = norm(s)[COL].item()
        cos = (x[:, COL] / x.norm(dim=-1)).tolist(); coss = (s[COL] / s.norm()).item()
        print(f"{name:30s} {layer:5d} {x.abs().max().item():12.1f} {x[:, COL].abs().max().item():11.1f} | "
              + " ".join(f"{v:7.2f}" for v in post) + f" | {posts:7.2f} | " + " ".join(f"{v:5.2f}" for v in cos) + f" | {coss:5.2f}")
