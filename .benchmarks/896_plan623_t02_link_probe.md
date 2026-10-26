# Bench 896 — Plan 623 T0.2: amo⇄bmo link probe over the TB5 bridge

**Date:** 2026-10-26 ~18:00+07 · **Plan:** [623](../.plans/623_multi_host_pipeline_inference_lane.md) T0.2 · **Proposal:** [042](../.proposals/042_multi_mac_distributed_inference.md)
**Instrument:** `examples/link_probe.rs` (std-only TCP, `TCP_NODELAY`, u32-BE length-prefixed frames; RTT = echo round trip in-process; bandwidth = 64 MiB one-way push + 8-byte ack, best of 5). Commit at measurement: this file's sibling commit.
**Session:** zed-amo, 1793208000

## Box state (both ends)

| | amo | bmo |
|---|---|---|
| model / RAM / cores | Mac Studio M5 Ultra, 96 GiB, hw.ncpu 30 (amo) | Mac17,15, Apple M5 Ultra, 96 GiB (103079215104 B), hw.ncpu 30 |
| OS | macOS 27.0.1 **26A434** | macOS 27.0.1 **26A434** (IDENTICAL — the RDMA identical-build requirement is met) |
| link | ONE TB5 cable, 80 Gb/s negotiated (Bus 0); bridge `192.168.10.1/24` ⇄ `192.168.10.2/24`, ICMP 0.53 ms | same |
| power / load | AC; `uptime` load 5.0–9.3 across the runs (llama.cpp `-j16` build had FINISHED before the clean cells) | AC; idle-ish |
| firewall | ENABLED (State=1) — probe binary allowed | enabled — probe binary allowed |
| tailnet | healthy (UDP true) | **offline** (relay row stale) — the tailnet arm of T0.2 is DEFERRED until bmo rejoins |

## Results — RTT (user-space TCP echo, µs)

| frame | direction | min | median | p90 (support) | max | n |
|---|---|---|---|---|---|---|
| 4 B | amo→bmo | 55.9 | 96.2 | 117.3 (200) | 265.8 | 2000 |
| 4 B | bmo→amo | 59.6 | 93.9 | 133.9 (200) | 270.9 | 2000 |
| 16 KiB | amo→bmo | 48.7 | 83.4 | 101.6 (100) | 146.2 | 1000 |
| 16 KiB | bmo→amo | 49.0 | 72.5 | 99.5 (100) | 164.7 | 1000 |
| 1 MiB | amo→bmo | 297.6 | 334.1 | 372.0 (20) | 970.9 | 200 |
| 1 MiB | bmo→amo | 299.5 | 327.3 | 363.9 (20) | 1137.4 | 200 |
| 64 MiB | amo→bmo | 15 929 | 16 163 | 23 486 (1) | 23 486 | 10 |
| 64 MiB | bmo→amo | 15 738 | 16 037 | 22 266 (1) | 22 266 | 10 |

## Results — bandwidth (64 MiB one-way push, GB/s, best of 5)

| direction | rounds | best |
|---|---|---|
| amo→bmo | 8.509–8.617 (all 5 tight) | **8.617** |
| bmo→amo | 7.889–8.541 | **8.541** |

≈ **8.5 GB/s sustained user-space TCP** over the TB5 bridge both directions — 68% of the 80 Gb/s line rate through a std-only socket path with no tuning. The transport lane does not need kernel bypass to saturate any plausible activation stream: a 4 KiB decode-stage activation frame is wire-trivial (<1 µs at line rate); the REAL per-hop cost is the ~90–100 µs small-frame RTT class (syscalls + scheduler), which bounds PP stage-hop overhead, not bandwidth.

## Derivation for Plan 623

- **PP stage hop (TCP posture):** ~0.09–0.10 ms per one-way hop (4–16 KiB class). At ~80 layer boundaries per token the 2-stage split has exactly ONE boundary crossing per token → decode overhead budget ≈ 0.09 ms against a ~5–7 ms decode step (league posture) ≈ **1.3–1.8% — comfortably inside the ≤5% PP-overhead G2 bar.**
- **1 MiB prefill chunk:** ~0.33 ms RTT / ~0.12 ms one-way at 8.5 GB/s — chunked pipelined prefill can keep both stages busy at chunk sizes ≥ 256 KiB.
- **TP go/no-go (Plan 623 Phase 4 gate):** 2·L·RTT with L≈80 allreduces → ~14.4 ms per token over TCP vs the ~3 ms half-step bar → **TP over TCP is REFUTED by arithmetic before any code is written** (as the proposal anticipated); only the RDMA posture (D2) can reopen it. Recorded so Phase 4 is not attempted on the TCP transport.
- **Bandwidth headroom:** 8.5 GB/s ≈ 8.5 ms per GiB — a full 27B fp16 activation stream for micro-batching (T2.4) is nowhere near saturation.

## Traps paid for (deployment law for this lane)

1. **In-place scp overwrite of an executed signed binary → `Killed: 9` at exec** (AMFI invalidates the CD hash of an inode rewritten after it ran; the kernel SIGKILLs the next exec). Symptom chain: server "listening" then dead, bandwidth rounds collapsing mid-run. **Law: `rm` the remote binary BEFORE every scp — never overwrite in place.**
2. **macOS Application Firewall silently defers inbound connections for an unsigned binary:** client `connect()` COMPLETES (the kernel backlog finishes the handshake) while the app's `accept()` never returns — reads as a hung server with no error anywhere. Fix: allow the binary in socketfilterfw (or firewall off).
3. `.benchmarks/.highwater` contained the literal `-n 895` — a prior session's non-portable `echo -n` under `/bin/sh` wrote the flag into the file. This bench allocates **896** and rewrites the counter clean (plain number).
4. Earlier anomalous cells (bmo→amo "8.3 GB/s then 5 MB/s rounds"; amo→bmo bw write failure) were artifacts of traps 1+2 on a then-stale binary — superseded by the clean runs above; the amo-side `-j16` llama.cpp build overlap was also eliminated as a variable before the clean cells.

## Verdict

**T0.2 PASS.** TCP-over-TB5 is GO for Plan 623 Phase 2 (the `riir-infer-dist` std-TCP transport): RTT class ~0.09 ms, bandwidth 8.5 GB/s, both directions, identical OS builds. Tailnet arm deferred (bmo offline); RDMA arm deferred behind D2. Incumbent link numbers (llama.cpp RPC / exo / MLX jaccl) belong to T0.4's scoreboard rows, measured with the same frames.
