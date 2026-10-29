# Issue 1150 — Metal dense-qwen3.8 forward (`qwen38_dense_wgpu`): the only road to same-model comparisons vs uzu and MLX

**Status:** IN PROGRESS — decode + parity G1 LANDED (465c1d8..63c8847, 2026-10-28); prefill engine = the named follow-up (see `.benchmarks/i1150_qwen38_wgpu_session.md`: all in-crate CubeCL GEMMs measure ~110 GFLOP/s on Metal — the matmul2d dense-f16 route is the speed unlock)
**Claimed:** zed-m3-i1150-densef16, epoch=1793272600, expires=2026-10-30T02:40:00+00:00 — the dense-f16 matmul2d prefill arm (session record step 1 + the f16 gemv so the posture is complete both paths; bmo four-way re-run stays the named follow-up, co-tenant-disclosed)

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
