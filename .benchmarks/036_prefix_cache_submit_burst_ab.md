# Bench 036 — issue 041 T3: the prefix-cache submit-burst wall-time A/B (Ada / RTX 4090)

**Status:** RECORD — EXECUTED 2026-10-13 (idle-loop window unit; the T3 GOAT surface the
issue staged). Verdict: **the pointer-table copy path wins the warm insert/restore cycle on
every axis** — submit probe confirms the designed 96 → 2 launches per direction (192 dtod
memcpies vs 4 launches for a warm refresh+restore cycle, zero table htods warm); the host
enqueue burst drops **18.5–22×** (paired medians 0.054× insert / 0.045× restore) and the
sync-inclusive wall halves (**0.49× insert / 0.48× restore**) at the real 27B GDN state
shape (~151 MiB ≈ 159 MB decimal per checkpoint). Stability: second full run reproduced the
verdict within noise (0.053×/0.047× enq, 0.499×/0.504× sync).

## Box state (provenance — Issue-021 law)

| Axis | Value |
|---|---|
| GPU | RTX 4090 24 GB (sm_89, Ada), KB5121794 post-era |
| GPU util / mem pre-run | 4–5% / 376 MiB (desktop composition only; `--query-compute-apps` lists NO CUDA compute process) |
| CPU load / free RAM | 7% / 18.5 GiB |
| Transport | Tailscale SSH (LAN 192.168.1.33 port 22 timed out); box synced via the `gitsync` scheduled task |
| Power | AC (desktop) |
| Build | release, 1m41s (warm deps from bench 035's target), `CARGO_TARGET_DIR=E:\tmp\infer038`, throwaway worktree `E:\git\wt041t3` @ `971b842` + the 3 T3 files scp'd (removed after validation — the pre-existing `wt038infer` left untouched at its own `33d2ef6`) |
| Instrument | `crates/riir-infer-gpu/examples/prefix_cache_ab.rs` — 41 order-rotated paired reps, 5 warmup, restores into the live state object (the production shape: the table arm's state-table freshness probe stays warm) |
| Gates | module tests 4/4 GREEN (`bit_identity_against_dtod_loop`, `refresh_second_entry_and_eviction`, `refuses_nonuniform_layers`, + the new `submit_probe_counts` count-regression test); clippy `-D warnings` clean at the feature-ON posture (lib + example) and the default posture |

## Submit-count probe (the code-level claim, now measured)

| path | warm refresh+restore (both arms) | fresh insert |
|---|---|---|
| dtod loop | `[dtod=192, allocs=0, launches=0, htods=0]` | `[96, 96, 0, 0]`, wall 1.64–1.70 ms |
| table copy | `[0, 0, 4, 0]` — 2 launches per direction | `[0, 96, 2, 4]`, wall 9.5–9.7 ms incl. the lazy engine build (nvrtc compile rides this number, disclosed) |

The counters are module-level `submit_probe_counts()` diagnostics (Relaxed adds on a
≤ ~200-submit path); the `test_submit_probe_counts` gate pins the counts at the test shape
forever — a future edit that adds a submit reds CI.

## Wall-time A/B (medians over 41 order-rotated paired reps)

| run | arm | enq_ins µs | sync_ins µs | enq_res µs | sync_res µs |
|---|---|---|---|---|---|
| 1 | dtod | 760.7 | 770.4 | 775.2 | 782.2 |
| 1 | table | 40.8 | 381.0 | 34.7 | 378.0 |
| 1 | paired t/d | **0.054×** | **0.491×** | **0.045×** | **0.483×** |
| 2 | dtod | 728.1 | 755.2 | 736.0 | 751.2 |
| 2 | table | 38.8 | 379.8 | 34.3 | 377.3 |
| 2 | paired t/d | **0.053×** | **0.499×** | **0.047×** | **0.504×** |

## Reading

1. **The dtod path is HOST-bound on driver submits**: enqueue ≈ sync (~760 vs ~770 µs) —
   the 96 `memcpy_dtod` calls cost ~8 µs of driver slice each (~770 µs of host CPU per
   direction), and the device copies complete almost as fast as the host can enqueue them.
   This is exactly the burst the issue targeted (Research 397's "192+ submits riding the
   WDDM driver slice").
2. **The table path moves the cost to the device**: enqueue drops to 35–41 µs (2 launches);
   the sync-inclusive wall (~378 µs) is then the gather/scatter kernel's execution — ~151 MiB
   through pointer tables at ~400 GB/s effective, i.e. the kernel runs at good dtod-class
   bandwidth and the win is the 96-submits → 2-launches collapse, not bandwidth.
3. **Per serving cycle** (insert + restore): dtod ~1.55 ms (mostly host CPU) vs table
   ~0.76 ms (mostly device) — ~2× wall, ~20× host-CPU. The issue's honest-magnitude
   estimate (~1–2 ms/cycle) is confirmed at the top of its band.
4. **Short-suffix TTFT context** (the issue's priority regime): a dtod restore is 782 µs =
   ~7.8% of a 10 ms short-suffix budget; the table path halves that to ~3.8%. The ~20×
   enqueue win additionally releases ~0.74 ms of HOST stall per restore — the serving
   loop's CPU is free ~20× sooner.
5. **Fresh insert is unchanged territory** (T4's surface): the 96-alloc burst is identical
   on both paths (allocs=96 each); the table arm adds 4 one-time table htods. Cold-insert
   walls are report-only (the table arm's includes the nvrtc engine build).
6. **Not a scored league lane** — serving-lane (TTFT) only; the bout benches never touch
   the prefix cache. Opt-in stays (`QWEN38_PREFIX_TABLE_COPY=1`); promotion is not gated on
   this bench (the issue's T5 graph co-existence re-proof and T4 arena pool remain staged).

## Refs

- issue 041 (T1+T2 landed `971b842`; this bench closes T3)
- riir-ai `.research/397` (the 96-submit measurement this A/B confirms)
- llama.cpp PR #29807 (the same move upstream, merged 2026-10-09)
- `crates/riir-infer-gpu/examples/prefix_cache_ab.rs` (the instrument)

## T4 addendum — the arena pool (validated same window, 2026-10-13)

The T4 arena pool landed after this bench's T3 read: each entry's snapshots live in ONE
contiguous buffer per class, recycled through a pool (cap `max_entries`) on eviction and
lineage prune. Measured on the same box/posture (same instrument, post-T4 build):

| fresh insert | pre-T4 | post-T4 |
|---|---|---|
| dtod arm wall | 1,642–1,704 µs | **928 µs** (96 per-layer allocs → 2 arena allocs) |
| table arm allocs | 96 | **2** (0 on a geometry-matched pool hit) |
| warm cycle | probe `[192,0,4,0]`, enq 0.054×/0.045×, sync 0.49×/0.48× | **unchanged** (probe `[192,0,4,0]`, enq 0.057×/0.049×, sync 0.511×/0.513×) |

The warm path is untouched by design (the arenas only change WHERE snapshots live); the
fresh-insert saving is ~0.78 ms of driver slice per new prefix (≈8 µs/alloc × 96 — the
same µs-class rate the enqueue analysis measured). Gates: 5/5 GREEN (the four T3 gates +
the new `test_arena_pool_reuse_zero_allocs` — pool hit = 0 allocs AND a full byte-compare
restore through a recycled arena); clippy `-D warnings` clean at both postures. The table
arm's fresh-insert wall (9.1 ms) remains engine-build-dominated (one nvrtc compile per
process, disclosed).
