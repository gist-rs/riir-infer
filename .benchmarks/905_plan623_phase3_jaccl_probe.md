# Bench 905 — Plan 623 Phase-3 gate opens: JACCL (RDMA over TB) measured

**Date:** 2026-10-27 (amo + bmo, post-reboot, `rdma_ctl enable` DONE on both)
**Session:** zed-amo-t04-mlx
**Unblocks:** [Plan 623](../.plans/623_multi_host_pipeline_inference_lane.md)
Phase 3 (RDMA transport) + the Phase 4 TP arithmetic gate's RTT input.
**Owner acts consumed:** `rdma_ctl enable` in macOS Recovery on BOTH boxes
(issue 1109 T4 / D2 — closed); amo Remote Login (T-analog); firewall allowlist
for the uv python (Bench 904 §5 trap 1 remediated).

## Setup that worked (no IP surgery)

`ibv_devices` shows 6× `rdma_en*` per box. The single TB cable sits on
receptacle 1 → `en2` → **`rdma_en2`** on BOTH (verified via
`system_profiler SPThunderboltDataType -json` + `networksetup
-listallhardwareports`). `mlx.distributed_config --backend jaccl` was NOT
needed: its auto-setup wants `sudo ifconfig bridge0 down` + /30 IPs per cable
(killing the 192.168.10.x TCP transport) and prompts interactively. The raw
posture works without any of it:

```jsonc
// .raw/mlx-ibv.json (both boxes)
[["rdma_en2", null] && [null, "rdma_en2"], ["rdma_en2", null]]  // → [[null,"rdma_en2"],["rdma_en2",null]]
```

```sh
# rank0 (amo) first, then rank1 (bmo); coordinator on rank0 over TCP
MLX_RANK=0 MLX_JACCL_COORDINATOR=192.168.10.1:50100 \
MLX_IBV_DEVICES=$PWD/.raw/mlx-ibv.json mlx-env/bin/python scripts/plan623_jaccl_probe.py
```

`jaccl init ok size=2` on the first attempt — the RDMA CM resolves over the
cable without extra IPs (mlx 0.32.0).

## Rows (median, µs; two runs agree)

| op | size | ring-TCP (Bench 904) | **JACCL** | speedup |
|---|---|---|---|---|
| all_sum | 4 B | 325 | **16.1–22.5** | 14–20× |
| all_sum | 16 KiB | 366 | 18.1–24.4 | 15× |
| all_sum | hidden 10 KiB (the PP hop) | 322 | **19.0–22.8** | 14× |
| all_sum | 1 MiB | 1052 (2.0 GB/s) | **156–166 (12.6–13.4 GB/s)** | 6.5× |
| all_sum | 64 MiB | 36 242 (3.7 GB/s) | **7 264 (18.5 GB/s)** | 5× |
| all_gather | 1 MiB | 832 | 160 | 5.2× |
| p2p ping-pong | 5 KiB bf16 | DEADLOCKED | **DEADLOCKED** | — |

Full JSON: `905_plan623_phase3_jaccl_probe.json`.

## The p2p verdict hardens — upstream issue material

The send/recv ping-pong hangs on **jaccl too**: both ranks post, then spin in
`mlx::core::array::wait()` under `jaccl::MeshImpl::recv` →
`tbt_poll_qp_recv` (libthunderboltrdma) forever (sampled stack). Combined
with Bench 904's ring result:

> **mlx 0.32.0, 2-node macOS: p2p `send`/`recv` is broken on BOTH backends
> (ring-TCP and jaccl-RDMA) while ALL collectives work on both.**

The shared shape (both ranks wedged in recv, no bytes delivered, and in
jaccl's case a clean QP-poll spin with no error) points at the
distributed-layer p2p path above the transports, not at a transport. The
ring.cpp recv-order comment ("check the sockets with the opposite order of
send so that they work even with 2 nodes where left and right is the same
neighbor") documents the intended pairing; the jaccl failure on a simple
peer pair suggests the bug is one level up. Upstream filing GO'd by the
owner (Bench 904 §7, item E) — repro: `.raw/min_p2p.py` (ring) +
`scripts/plan623_jaccl_probe.py` pp_rtt leg (jaccl); zero riir IP.

## What this re-opens in Plan 623

- **Phase 4 TP arithmetic gate** (`go only if 2·L·RTT < ½ the 1-box decode
  step`): with per-op RTT now ~16–20 µs, 2·L·RTT for a 64-layer TP exchange
  ≈ 2–2.5 ms vs a ~16–17 ms 1-box decode step → **the gate PASSES on
  paper**. TP over jaccl is now a measured-feasible posture — a first real
  TP cell (two half-model ranks, all_sum after each block) is the next
  ungated measurement.
- **PP hop cost**: the socket hop (0.1–0.3 ms) was already small vs the
  +6 ms pipeline serialization cost (Bench 904 §4); jaccl shaves the hop
  further but does NOT move the serialization wall — the 2-box PP decode
  verdict (−26%) is expected to hold within noise. A jaccl-PP re-read is
  cheap and confirms it.
- **Phase 3 transport**: the RDMA transport work proceeds with real numbers
  (18.5 GB/s bulk, 16 µs latency class).

## Artifacts

- `905_plan623_phase3_jaccl_probe.json` — jaccl rows + hand-updated p2p verdict
- `scripts/plan623_jaccl_probe.py` (rsync twin on bmo) · `.raw/mlx-ibv.json`
- Environment: both boxes rebooted with `rdma_ctl enable`; TB bridge IPs
  intact post-reboot; venv + pack survived; firewall allowlist persisted.
