# Issue 1150 — Metal dense-qwen3.8 forward (`qwen38_dense_wgpu`): the only road to same-model comparisons vs uzu and MLX

**Status:** IN PROGRESS — decode + parity G1 LANDED (2026-10-28); prefill unlock LANDED 2026-10-29 (dense-f16 matmul2d, A/B 50.22× kernel-level); bmo four-way RUN EXECUTED (two outcomes: GEMM-wall CONFIRMED+REMOVED, e2e 4 → 18 tok/s class; G2 vs MLX FAIL ~75×); **stage-timer pass LANDED same evening** (`306540e`): shares measured (ragged 40–42% / GDN 31–32% / tail_logits 15–16% / attn 11%), the ragged-tail bench defect found+fixed (`5b41121` — pp cells now pure prefill), decode rows recorded (f16 decode 1.86/1.23 tok/s — prefill-experiment posture, never serving), and the NEW top lead is the ~3× cross-process bimodality (18 vs 6 tok/s same posture, neither mode explained); full record: `.benchmarks/i1150_qwen38_wgpu_session.md`

**Follow-ups (stage-timer pass LANDED 2026-10-29 evening — shares measured, order re-derived; see `.benchmarks/i1150_qwen38_wgpu_session.md` §"Stage timers"):**

- [x] Stage timers on one clean pp 2484 pass — LANDED `306540e`, breakdown measured twice (shares reproducible): ragged tail 40–42% / GDN 31–32% / tail_logits 15–16% / attn 10.7% / wte ~0.4%
- [x] The ragged-tail bench defect it exposed — FIXED `5b41121`: pp targets round up to the chunk multiple (2484→2560), padded count disclosed; the four-way pp cells were NOT pure prefill and the morning's OURS column must be re-read
- [x] Decode rows recorded — f16 posture decode 1.86 tok/s raw / 1.23 tok/s ctx2048 (~28× regression vs the packed incumbent): the f16 posture is a prefill experiment, never a serving decode posture
- [ ] **Stability/bimodality audit (was: residency audit — now the TOP lead)**: pure prefill reads 6 tok/s stable in a fresh quiet-box process while the four-way process read 18 tok/s on comparable work — a ~3× cross-process bimodality with neither mode explained; absorb the two-weight-copies residency question (54 GB f16 + ~10 GB host wte + lm staging + mmap file-cache) into it; explain the split or neither number pins
- [ ] GDN chunked scan routing (48/64 layers, 31–32% measured share) — gated on the stability audit landing
- [ ] lm_head tail GEMV routing (15–16% measured share, f32 gemv class) — real but second-order; after stability
- [ ] Re-run protocol: reproduce-or-retire the 18 tok/s pass posture first; then single pass per cell, per-pass memory watchdog, warm-shape sweep for fresh-box runs (the autotune cache persisted across processes tonight — the 20–32 min cost is first-ever-compile only)

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
