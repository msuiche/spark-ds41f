# The DGX ring — infra runbook

Everything one keeps rediscovering about the four-node rig. Written during the
GLP-27 deploy (2026-10-31); verify facts before trusting numbers older than
that. Repo: `~/projects/spark-ds41f` on every node, branch `glp-steering-r6x`.

## Nodes, addresses, and which network to use

| alias | ssh host | mgmt IP (wlP9s9, slow) | RoCE fabric IPs (fast) |
|---|---|---|---|
| dgx1 | spark-4687 | 192.168.1.232 | 10.100.227.10, 10.100.224.10, 10.101.227.20, 10.101.224.20 |
| dgx2 | spark-5bc3 | 192.168.1.171 | 10.100.225.11, 10.100.224.11, 10.101.225.21, 10.101.224.21 |
| dgx3 | spark-7425 | 192.168.1.180 | 10.100.225.12, 10.100.226.12, 10.101.225.22, 10.101.226.22 |
| dgx4 | spark-7275 | 192.168.1.206 | 10.100.227.13, 10.100.226.13, 10.101.227.23, 10.101.226.23 |

- The mgmt/wifi link does **~6.5 MB/s**. Never move images over it.
- The RoCE fabric is the switchless cable ring, one /24 per link:
  4687—5bc3 (10.100.224.x), 5bc3—7425 (.225), 7425—7275 (.226),
  7275—4687 (.227); 10.101.x is the second CX-7 card's parallel ring.
- **Image distribution** (measured ~24 GB in 2 min per hop):
  direct from the head to its neighbours, then relay.

```bash
T=<image-tag>
docker save $T | ssh 10.100.224.11 'docker load'   # 5bc3
docker save $T | ssh 10.100.227.13 'docker load'   # 7275
ssh 10.100.224.11 "docker save $T | ssh 10.100.225.12 'docker load'"  # 7425 via 5bc3
```

## Lanes and services per node

- **dsv41 lane** (this repo, `config/cluster.json`): TP4 vLLM serving on
  `:8000`, image `vllm-ds41f-kkref:*`, container `dsv41-karmic-kraken`,
  ready URL `http://127.0.0.1:8000/v1/models`. Managed only via
  `bin/spark cluster start|stop` — never hand-launch ranks.
- **tensorfold TP2 lane** (glm53-flash-tf): `:8888`, watchdog
  `~/tensorfold-watchdog.sh` (systemd `tensorfold-watchdog.service`, armed
  while `~/tensorfold-watchdog.on` exists). Independent of the dsv41 lane.
- **ring-heal** (`ring-heal.service`, head): watches :8888 and orchestrates
  ring restarts on API cascades. Its log: `~/ring-heal.log`.
- **watchdogs**: `memory-watchdog.service`, `network-watchdog.service`
  (all nodes); `spark-management-health.timer` runs
  `/usr/local/libexec/spark-management-health` — the tailscale branch is
  guarded by `systemctl cat tailscaled.service` because **tailscale exists
  only on spark-4687**.
- **agents**: `hermes-agent` (head), `flare-agent` (`:29900`, every node) —
  flare holds `/dev/nvidia*`; killing it reboots the node (supervisor).
- **DeskPi display dashboard**: `spark-dashboard.service` on tty1, files in
  `~/tinydashboard/` (`spark-dashboard.sh` + `spark-status.py`). See below.

## The DeskPi 6.91" bar panel (1424×280, non-standard)

Each node's console screen is a DeskPi 1U bar panel. Working state (all four
verified 2026-10-31):

1. **EDID override** — the panel is not a standard mode; the kernel needs a
   forced EDID. Per node:
   - `/usr/lib/firmware/edid/deskpi.bin` (256 B custom EDID, 1424×280)
   - kernel cmdline `drm.edid_firmware=HDMI-A-1:edid/deskpi.bin`
     (`/etc/default/grub` `GRUB_CMDLINE_LINUX_DEFAULT`)
   - `/etc/initramfs-tools/hooks/deskpi-edid` copies the blob into the
     initrd; then `update-grub && update-initramfs -u`; takes effect at
     the **next reboot**.
2. **Font** — the dashboard script's per-user `setfont` silently fails on
   some nodes. The unit must set it as root (mirrors the working node):

```
ExecStartPre=+/bin/chown msuiche:tty /dev/tty1
ExecStartPre=+/usr/bin/setfont /usr/share/consolefonts/Lat15-TerminusBold32x16.psf.gz -C /dev/tty1
```

   Right state: `sudo stty size < /dev/tty1` → `8 89` (8 rows × 89 cols).
   `17 178` means the EDID is fine but the font is the 8×16 default.
3. Unit runs the tmux session `sparkdash` fullscreen on tty1.

## GLP steering deploy (image layering recipe)

The lane's steering is a **fail-closed runtime patch applied as an image
layer** on the stock release image, armed at serve time by env:

- `scripts/hotfix-dsv41-fold-steering.py` — ports the GLP-3 fold-site
  patch to the 1022 candidate tree (`/opt/spark3/candidate/vllm`). It
  materializes the mhc fold at every layer top (the tree's own Engram
  pattern), steers post-fold pre-Engram (`hc_fold_pre_engram`, GLP
  spec_version 3), and wraps `attn._o_proj` with the mandatory companion
  writer (kwargs-forwarding — this tree passes keywords). Anchors are
  exact; anchor drift = exit 1, never guess.
- `docker/Dockerfile.glp27` — `FROM vllm-ds41f-kkref:…-1022-4`,
  context is **`scripts/`** (the root `.dockerignore` is `*`):

```bash
docker build -f docker/Dockerfile.glp27 \
  -t vllm-ds41f-kkref:19f2c20ed4d6-1022-glp27 scripts
```

- Serve-time env lives in `config/cluster.json` `environment`:
  `WEIGHTLESS_STEER_PATH=/vectors/dsv41-perlayer-dirs-ffn.pt` (host
  `~/vectors/`, ro-mounted), `WEIGHTLESS_STEER_ALPHA=3.5`,
  `DSPARK_STEER_CTRL=/cache/steer-ctrl.json` (mtime-polled live alpha /
  vector swap; ack in `/cache/steer-ack.json`, rank 0 writes).
- Vector: `msuiche/DeepSeek-V4.1-Flash-1017-abliterated-cyber-GLP-27-L11-37-a3.5`
  (gated). Validate with
  `weightless/tools/captain-vector/captain_vector.py --validate FILE.gguf`
  (stdlib-only). The `.pt` `dsv41-perlayer-dirs-ffn.pt` is the raw
  `{layer: tensor(5120)}` stack the lane consumes.

### The label discipline (why starts fail with tree mismatches)

- `upstreams.lock.json` → `source_manifest` names the manifest that pins
  every source tree; `bin/spark` refuses when
  `cluster.json container.expected_labels` ≠ that manifest's
  `expected_tree` entries, and separately when the image's actual labels
  mismatch. The 1022 release tree is **e68aa19c…**; the r6x-era tree
  (`16385ad8…`) is the OLD GLP-39 build — do not mix.
- A patch that changes the vLLM **forward graph** must clear the AOT
  caches on all nodes or a stale graph silently runs unsteered:
  `rm -rf ~/projects/spark-ds41f/cache/kkref/jit/vllm-r5o …/torchinductor`.
  First boot then recompiles (long — `ready_timeout_seconds` is 3600).
- Stale containers: `cluster start --apply --replace`.
- `cluster sync --apply` only deploys commits on origin/main; for fork
  branches, detach each node manually:
  `git fetch fork && git checkout --detach <sha>`.

### Rollout order (validated 2026-10-31)

1. Commit + push the branch (fork: `msuiche/spark-ds41f`).
2. Every node: `git fetch fork && git checkout --detach <sha>`.
3. Build the image on the head, distribute over the **fabric IPs**.
4. Place the vector in `~/vectors/` on all four nodes.
5. Clear JIT caches when the forward changed.
6. `bin/spark --cluster-config config/cluster.json cluster start --apply --replace`.
7. Smoke: `/v1/models`, a completion, and the boot log line
   `dspark fold-site steering active: alpha=3.500 layers=27`.
8. `bin/spark bench --suite manifests/suites/2026-10-22-spark-ds41f-1022-tp4 --label …`.

## Known failure modes

- **GPU "No devices were found"** with dmesg
  `RmInitAdapter: Cannot initialize GSP firmware RM` +
  `kgspPrepareForBootstrap_HAL` timeouts: the GSP microcontroller is
  wedged; **warm reboots do not fix it** — power off, unplug AC ~60 s,
  reboot. (spark-7275, 2026-10-31: reproduced across 3 warm boots, fixed
  by cold cycle.) While wedged, a `flare-agent` holds `/dev/nvidia*`;
  killing it reboots the node (supervisor), it does not fix the GPU.
- **Management-plane checks**: `spark-management-health` failing with
  status 5 means `tailscaled` is absent — intentional off the head;
  the guarded script version skips it.
- **SSH via aliases is slow**: the aliases resolve to the wifi mgmt IPs.
  For bulk data use the fabric IPs from the table.
