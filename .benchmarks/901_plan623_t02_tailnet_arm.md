# Bench 901 — Plan 623 T0.2 tailnet arm: amo⇄bmo over Tailscale (fallback transport)

**Status:** LANDED (completes T0.2 — the bridge arm is [Bench 896](896_plan623_t02_link_probe.md))
**Date:** 2026-10-26 ~23:00+07
**Plan:** [623](../.plans/623_multi_host_pipeline_inference_lane.md) T0.2 · **Issue:** [1109](../.issues/1109_bmo_cluster_bringup_and_routing.md) T2
**Instrument:** `link_probe` @ the Bench-896 build ( amo `target/release` 18:07 == bmo `~/link_probe` 18:08, 518176 B; amo's copy was mid-rebuild by a sibling session, so the bmo byte-identical copy ran from `/tmp/link_probe_896` on amo — same instrument, both ends). Port 46230, TCP_NODELAY, 64 MiB push best-of-5.

## Box state

- bmo tailscale **direct/live** both ways (Issue 1109 T2 DONE — daemon self-recovered; rides the WiFi LAN 192.168.1.x, UDP true, easy NAT).
- amo load 4–7 (sibling agent builds); bmo idle-ish. Same both boxes as Bench 900.

## Results

| direction | RTT @4 B (median) | RTT @16 KiB | RTT @1 MiB | bandwidth (64 MiB push, best of 5) |
|---|---:|---:|---:|---:|
| amo → bmo (client amo) | 9444.9 µs | 11646.6 µs | 94251.2 µs | **0.025 GB/s** |
| bmo → amo (client bmo) | 9455.7 µs | 11823.7 µs | 94287.0 µs | **0.028 GB/s** |

## Comparison vs the TB5 bridge (Bench 896)

| | TB5 bridge (192.168.10.x) | tailnet (100.x) | ratio |
|---|---:|---:|---:|
| bandwidth | 8.5–8.6 GB/s | 0.025–0.028 GB/s | **~340× slower** |
| RTT @4 B | ~94 µs | ~9.4 ms | **~100× worse** |

## Verdict

1. **The tailnet is a CONTROL-PLANE / relay-join transport only.** A PP hop at 9.4 ms RTT (>3 RTTs per decode step ≈ 30 ms) is ~1.8× the ENTIRE 17 ms single-stream decode step — activation frames over tailnet are dead on arrival, consistent with the T0.2 derivation and now measured, not just derived.
2. WireGuard + WiFi LAN explains the shape: ~25 MB/s tunnel throughput (WiFi-class, not tailnet-relay — `via 192.168.1.x` direct), latency floor ~9.4 ms ≈ WireGuard crypto + WiFi RTT + userspace round trip.
3. T2's routing law stands: bridge (192.168.10.x) primary, tailnet fallback for join/control; never for the data plane.

## Trap noted

amo's `target/release/link_probe` vanished mid-session (sibling session rebuilding `examples/link_probe.rs` — uncommitted WIP in the shared worktree). The bmo byte-identical copy substituted cleanly. **Law: shared-worktree instruments used in a bench must be pinned somewhere immutable (or quoted by blake3) before the run, not referenced live from `target/`.**
