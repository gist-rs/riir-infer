# Issue 1113 — mma module's `kv_f16_to_f32` misses the +112 exponent re-bias (reachable under f16-KV + the mma knob)

**Status:** Fix LANDED 2026-10-24 (zed-m3-docsync-956-close) — the one-line re-bias (`(exp + 112u) << 23`), the KNOWN-BROKEN header note (the arm's verdict is UNCHANGED — its root cause is the tf32 fragment-assembly cross-talk, independent of the KV decode; the arm stays default-off), and the issue's pin (2) landed as TWO device-free tests in the module's tests mod: (a) the TWIN PIN — the mma CUDA `kv_f16_to_f32` body extracted from the source string and held token-identical to the canonical `ATTENTION_CUDA_SRC` copy (mechanical, no device, no transcription — the transcription-time catch); (b) the HOST PARITY PIN — a Rust mirror of the mma decode vs `qwen38_dense_cudarc::kv_f16_bits::f16_to_f32` over the FULL 65_536-pattern lattice. ⚠ VALIDATION PENDING on the 4090 (the module is `#[cfg(all(feature = "ternary_gemv_cuda_raw", not(target_os = "macos")))]` — compiled to nothing on the M3; the green-zero trap, respected): the two pins + the existing kb flag-off bit-identity determinism test + `cargo clippy -p riir-infer-gpu` run there; the pins' M3 execution is structurally impossible, not skipped.

## The defect (pre-fix record)

`crates/riir-infer-gpu/src/cudarc_kernels/attention_score_mma.rs` — `ATTENTION_SCORE_MMA_CUDA_SRC`, the
`#ifdef KV_F16` block's decoder:

```c
return __uint_as_float(sign | (exp << 23) | (mant << 13));   // ← missing +112
```

The f16 biased exponent must re-bias to f32 as `exp + 112` (127 − 15). The module's subnormal
branch is correct (`(113 - s) << 23`), and the main module's `kv_f16_to_f32`
(`cudarc_kernels/attention.rs`) is correct — this copy's NORMAL branch alone decodes every
normal f16 as ~2^112× too small (1.0 → 2^-112).

## Reachability

`AttentionScoreMmaKernels::new(.., kv_f16 = true)` compiles the source with `#define KV_F16 1`;
`Qwen38DenseForward::with_parts` resolves `kv_f16` from `QWEN38_KV_DTYPE=f16|half` ONCE per
process and constructs the mma score arm behind its own knob (Plan 551 / Issue 773 T2). So the
posture **f16 KV cache + the mma score-phase arm** feeds the broken decoder real f16 K/V —
scores are garbage in that posture.

## Why not fixed in the same commit

The mma score arm is the DECLARED KNOWN-BROKEN lane (Plan 622 T1.2: transcription-only, the
byte-identity protocol needs determinism not correctness) — a silent repair would invalidate
whatever posture records exist for it without its owner's read. The Plan 622 f16-wire work
deliberately did NOT reuse this decoder: the kb bias decode is a separate module-local
`kb_f16_to_f32` (the one-instruction `cvt.f32.f16` widen — exact in both KV hatches).

## Fix shape (one line + gates)

1. `(exp + 112u) << 23` in the mma module's normal branch.
2. A host/device parity pin: the mma module's decode vs `qwen38_dense_cudarc::kv_f16_bits` on a
   lattice of f16 bit patterns (the round-trip test's corpus), run in the mma lane's own test —
   the pin that would have caught this at transcription time.
3. Re-run the mma lane's determinism tests + whatever correctness posture the Plan-551/773
   owner declares for the arm.

**Found-by:** bench 886's module-local decode work (the kb path needed an exact decode in both
KV hatches; comparing the four modules' copies surfaced the divergence).
