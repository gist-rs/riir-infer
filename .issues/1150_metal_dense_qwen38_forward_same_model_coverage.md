# Issue 1150 — Metal dense-qwen3.8 forward (`qwen38_dense_wgpu`): the only road to same-model comparisons vs uzu and MLX

**Status:** IN PROGRESS — decode + parity G1 LANDED (2026-10-28); prefill unlock LANDED 2026-10-29 (dense-f16 matmul2d, A/B 50.22× kernel-level); bmo four-way RUN EXECUTED (two outcomes: GEMM-wall CONFIRMED+REMOVED, e2e 4 → 18 tok/s class; G2 vs MLX FAIL ~75×); **stage-timer pass LANDED same evening** (`306540e`): shares measured (ragged 40–42% / GDN 31–32% / tail_logits 15–16% / attn 11%), the ragged-tail bench defect found+fixed (`5b41121` — pp cells now pure prefill), decode rows recorded (f16 decode 1.86/1.23 tok/s — prefill-experiment posture, never serving), and the NEW top lead is the ~3× cross-process bimodality (18 vs 6 tok/s same posture, neither mode explained); full record: `.benchmarks/i1150_qwen38_wgpu_session.md`

**Follow-ups (stage-timer pass LANDED 2026-10-29 evening — shares measured, order re-derived; see `.benchmarks/i1150_qwen38_wgpu_session.md` §"Stage timers"):**

- [x] Stage timers on one clean pp 2484 pass — LANDED `306540e`, breakdown measured twice (shares reproducible): ragged tail 40–42% / GDN 31–32% / tail_logits 15–16% / attn 10.7% / wte ~0.4%
- [x] The ragged-tail bench defect it exposed — FIXED `5b41121`: pp targets round up to the chunk multiple (2484→2560), padded count disclosed; the four-way pp cells were NOT pure prefill and the morning's OURS column must be re-read
- [x] Decode rows recorded — f16 posture decode 1.86 tok/s raw / 1.23 tok/s ctx2048 (~28× regression vs the packed incumbent): the f16 posture is a prefill experiment, never a serving decode posture
- [x] **Stability/bimodality audit (was: residency audit — now the TOP lead)**: pure prefill reads 6 tok/s stable in a fresh quiet-box process while the four-way process read 18 tok/s on comparable work — a ~3× cross-process bimodality with neither mode explained; absorb the two-weight-copies residency question (54 GB f16 + ~10 GB host wte + lm staging + mmap file-cache) into it; explain the split or neither number pins. **— RESOLVED 2026-11-04 (zed-986a7538, bmo exclusive job `01a1a479-5f4f`, exit 0):** the split is **POSTURE, not process variance** — the discriminator was `RIIR_QWEN38_DENSE_MM2D`, never box state. Incumbent Q4_K: **7 tok/s flat** across 7 fresh processes (pass-wall spread <0.2% @pp2484, <0.05% @pp4954, both lengths + pp512). Dense-f16 MM2D: **22 tok/s** across 4 fresh processes (spread <0.4%) — 3.16× the incumbent; verified-idle 22.4 exceeds the four-way's co-tenant 18 (the co-tenancy cost, as suspected). Poison adjudicated: `mediaanalysisd` flagged but DORMANT (0.0% CPU at every read); all 10 passes admitted CLEAN by their 5 s traces (every spike = our own build/load ramps; peaks ≤1.14 external). Full tables + provenance: the audit addendum in [`.benchmarks/i1150_qwen38_wgpu_session.md`](../.benchmarks/i1150_qwen38_wgpu_session.md); artifacts `~/riir-bench/i1150_stability/` (durable, 10 logs + 10 watchdog traces + warm autotune target).
**Record-admission gates (Claude verdict round 3):** (a) the job came up **POISONED ×1** — poison mail READ 2026-11-04 ~03:2xZ: the source is **`mediaanalysisd` (pid 1125), a DORMANT system daemon** (macOS media-vision analysis; 0.0% CPU at every read since the flag; cumulative CPU-time 21:55 over a 46 h box life; NOT in the run's top consumers). **DECISION: KEEP** (mailed to jobs@bmo `01a1a48e`) — the per-pass 5 s watchdog traces corroborate: every traced window reads load 0.2–0.7 after warm-up; the only >2.0 spikes are w1's first-ever-compile ramp (02:53–02:54) and a ~3-minute mild bump at a1's start (03:03:53–03:04:28, peak 2.27, decayed), both attributed and neither overlapping a mediaanalysisd activity signature. At record time each pass is classified CLEAN or POISON-OVERLAP strictly by its own trace window; only CLEAN passes feed the between-process stability verdict; a pass whose trace shows real contention (esp. in the dense-f16 arm) is redone individually. A stability audit exists to measure between-process variation, so a disturbed run is exactly what it cannot absorb unexplained — the trace-per-pass attribution is that explanation. (b) PROVENANCE: the audit's code is pinned at `c52ca79` (the script recorded `git rev-parse HEAD` at start) — the bench worktree moved to `71774e5` MID-RUN (the i1117 gate fix), which touches nothing the audit builds (different bin, and the build had already finished into `$OUT/target`); the record states both facts; (c) process lesson recorded: never move a worktree a running exclusive job is using — next time each job gets its own worktree. The bimodal split's explanation + the record write are the session tasks when the job mails back. Ordering note corrected per verdict round 2: the job is bmo's exclusive FIFO head-of-line after its queue state at queue time — the claimed "after the G5 census job" was WRONG as written (bmo's queue showed NO census job; bench 937 records the census as landed 11-02, so the job most likely completed and was reaped — the 1167 holder was ASKED (`01a1a488`), not guessed).
Early audit reads (in-flight, pre-poison-decision): w1/w2 warm passes read 7 tok/s at pp512 with <0.7% spread across fresh processes; a1 (pp2484 incumbent) warm-up + pass 1 read 7 tok/s (360.76 s for 2560 padded) — the incumbent posture looks stable so far; the bimodality question hinges on the dense-f16 arm + the cross-process comparison, still ahead.
**Lease closed:** zed-986a7538@amo, epoch 1794169000 (claimed 2026-11-04 ~09:40Z for the audit arc, was expires 2026-11-06T04:00:00Z) — scope done same day: the poison mail read + KEEP adjudication (`01a1a48e`), all 10 passes admitted CLEAN by trace, the bimodality record written (the audit addendum in `.benchmarks/i1150_qwen38_wgpu_session.md`).
- [-] GDN chunked scan routing (48/64 layers, 31–32% measured share) — **MEASURED NEGATIVE (deferred as opt-in) 2026-11-04** (zed-986a7538@amo, bench 938): the route LANDED (`RIIR_QWEN38_GDN_CHUNKED=1`, promote-only) — Issue-734-T4 chunkwise family wired into `qwen38_dense_cubecl` over 64-token windows; **perf PARITY at every cell** (pp512 21 vs 21 tok/s, pp2484→2560 21 vs 22, linear at chunk 1024) because the 31–32% share was a BUCKET MISREAD — mostly GDN-layer GEMMs on the Q4_K posture; the recurrence scan itself is ~6% of a synchronized pass on the dense posture (1.44 s OFF vs 1.67 s ON of 25 s @pp512) and the multi-token kernel is latency-adequate at n_v=8. The route G1 also caught + FIXED a real bug in the shared family (`cum_decay`'s cumulative −60 floor corrupted both-floored γ ratios — max_out_rel 7.2e+1 on a floor-crossing fixture, e2e walk flipped at token 3; per-step α clamp now: 6.7e-6, walk **byte-identical** OFF vs ON). Full record: [`.benchmarks/938_issue1150_gdn_chunked_route_parity.md`](../.benchmarks/938_issue1150_gdn_chunked_route_parity.md)
**Lease closed:** zed-986a7538@amo, epoch 1794465837 (claimed 2026-11-04 ~07:24Z for the routing arc, folded at the bench-938 landing same day; was expires 2026-11-06T07:24:00Z)
- [ ] lm_head tail GEMV routing (15–16% measured share at pp2484, f32 gemv class) — **PROMOTED TO THE TOP LEAD by bench 938**: 90% of the synchronized pass at pp512 (22 s of 25 — 8 f32 GEMV sweeps over the 248k-row lm_head for ONE row per 512-token chunk) and 15–16% at pp2484; the real bottleneck on this lane, unclaimed
- [x] Re-run protocol: reproduce-or-retire the 18 tok/s pass posture first; then single pass per cell, per-pass memory watchdog, warm-shape sweep for fresh-box runs — EXECUTED as the audit itself (fresh process per cell, 5 s watchdog traces, warm-shape sweep w1/w2); the posture question is answered (see the resolved audit row) and every future A/B pins the posture env per cell

**Lease closed:** zed-m3-i1150-fourway, epoch 1793237775 (claimed 2026-10-29T01:36Z for 48 h, folded at the four-way run's landing same day; claim record: zed-m3-i1150-fourway, epoch 1793237775, was expires 2026-10-31T01:36:15+00:00) — recorded `e8af37b`
**Lease closed:** zed-m3-i1150-stagetimer, epoch 1793411000 (claimed 2026-10-29 ~20:30 +07 for 48 h, folded at the stage-timer landing same evening; claim record: zed-m3-i1150-stagetimer, epoch 1793411000, was expires 2026-10-31T06:23:40+00:00) — landed `306540e`+`5b41121`, recorded in the bench file
**Claim scope:** follow-up task 1 (stage timers) + the one timed pp 2484 pass on bmo
**Claim scope:** the dense-f16 matmul2d prefill arm (session-record step 1, the speed unlock) + the f16 gemv completing the F16 load posture for both paths; bmo four-way re-run stays the named follow-up, co-tenant-disclosed
**Lease closed:** zed-m3-i1150-densef16, epoch 1793272600 (claimed for 48 h, folded at landing same-day) — landed `7e20540`

## Problem

The 2026-10-28 bmo three-way (`.benchmarks/uzu27b/README.md` in riir-ai; issue 851 arm (e)
addendum) answered the ranking question same-model (qwen3.8-27B, 4-bit class, same box):
**MLX-LM is the prefill bar at every cell; uzu beats the fork at mid-P only (1.29× @ ~5k),
collapses long-P (0.49×)** — but OUR engine could not enter its own league's comparison,
because:

1. The only qwen3.8 GGUF forward in the engine is `qwen38_dense_cudarc.rs` —
   `cfg(all(feature = "ternary_gemv_cuda_raw", not(target_os = "macos")))`, CUDA-only.
2. **uzu is Apple-Silicon-only** (Rust host + MSL kernels, Research 127) — it can never run
   on the 4090. MLX is Apple-only too.
3. uzu can never run Bonsai (our ternary format is not in the mirai catalog — closed door).

Consequence: a same-model ours-vs-uzu comparison is structurally impossible in BOTH
directions on current code. Cross-box (4090-CUDA ours vs bmo opponents) is invalid per the
league's Issue-890 protocol. The ours row in the bmo table is Bonsai, model-crossed,
context-only (`.benchmarks/uzu27b/OURS_ROW.md`), and the two same-model tables (Bonsai:
ours 266 / fork 1066 / MLX-2bit 1111 @2048; qwen3.8: fork 898 / uzu 1157 / MLX 1411 @~5k)
can never be joined without a Metal qwen3.8 path.

## Proposal

`qwen38_dense_wgpu` — a wgpu/CubeCL dense-qwen3.8 forward (opt-in feature), ported from the
cudarc reference, reusing existing substrate (verify each before wiring; substrate-first
grep at implementation time):

- `crates/riir-infer-gpu/src/gemv_q4k_batched_rmsnorm_cubecl.rs` — the Q4K GEMV kernel
  exists (which lane owns it; whether it covers the prefill GEMM shapes needs a read).
- `crates/riir-infer-gpu/src/qwen_attention_cubecl.rs` — Qwen3.5-family full-attention
  kernels (partial RoPE, per-head RMSNorm, GQA flash decode; Issue 599). Prefill-tiled
  flash exists in the Bonsai lane (`qwen_attention_prefill_tiled_*`) — same family.
- The GDN wgpu kernels from `ternary_deltanet_gpu_forward.rs` (conv/recurrence/z-gate) —
  qwen3.8 IS the GDN hybrid (the cudarc forward already uses deltanet kernels; the arch is
  the same family as Bonsai — Bonsai is ternary-quantized qwen3.8).
- GGUF loading for the `qwen35` arch + Q4_K_M (the cudarc path's loader is the reference;
  share its parsing, replace the backend).
- `qwen38_dense_cudarc.rs` (~7k lines) stays the bit-parity oracle: same GGUF on the 4090
  vs the Metal forward, FNV/argmax pins (the G1).

## Acceptance gates (league posture — never a scored row before these pass)

- **G1 bit-parity:** Metal forward vs the cudarc reference on the SAME qwen3.8-27B Q4_K_M
  GGUF — FNV/argmax pins over a fixed token walk (cross-backend tolerance band set BEFORE
  the run; the f32 kernel-order classes get the certified-precedent treatment
  katgpt-rs `closure/bridge_certified`, never post-hoc bands).
- **G2 positioning cells:** pp 2484/4954/9856/16392 + decode ctx2048 on the M5 Ultra class,
  against the bmo three-way numbers (fork 898 @5k / MLX 1411 / uzu 1157) — the row that
  makes the four-way table same-model.
- **G3 no-regression:** Bonsai league cells byte-identical (the ternary lane untouched;
  feature-off builds byte-identical).
- **G4 alloc:** hot path alloc-free per house norm.

## Non-goals

- No change to the Bonsai lane or the scored league rows.
- No uzu/Bonsai pursuit (closed door, by format).
- Not a prefill-defense substitute: the Issue-1113 levers (ktile64, GEMV occupancy) target
  the scored Bonsai row and stay the priority; this lane buys COMPARISON COVERAGE, and a
  second dense model on Metal also de-risks the single-model lane (arch-generic kernels).

## Provenance

- bmo three-way + ours row: riir-ai `.benchmarks/uzu27b/` (2026-10-28), issue 851 arm (e)
  addendum; the owner's "queue bench on bmo so we can compare" and the follow-up
  "why not use same model?" — this issue is the answer's work item.
