# Bench 886 — Plan 622 T1.4: the f16 bias-table lever (the armed-tax cut + the downward ratchet re-pin)

**Status:** RECORD — G2 measured; the plan's named demotion lever LANDED and beat its prediction.
**Date:** 2026-10-23
**Box:** 4090 (RTX 4090, Windows, `--release`, GPU idle — desktop composition only; no compute
apps in `nvidia-smi --query-compute-apps`, SM clock at the 330 MHz idle baseline between runs).
**Lane:** `tests/plan622_keybias_fa_cost_gate.rs` — rotated-order interleaved medians, 48 iters
after 4 warmup, league FA shapes (DKQ=DV=256, 24:4 GQA, p ∈ {2048, 4096}, base_pos 64).
**Commits:** `dfcb76e` (the f16 wire) → `b227d3a` (the canonical macro brace) → `6790a40`
(the cvt decode) → `1590b08` (the shared-program name) → `7357b3a` (the ratchet re-pin).

## The lever

T1.4's f32-wire record named the demotion lever: an f16 bias table — half the bytes per element,
~6% expected — with the ratchet re-pinning downward on its landing. Landed same day.

**Wire:** `kb_bias` is `const unsigned short*` at every kb kernel (the 16 decode-family
launchers, the mma twin, the FA prefill pair, the canonical 128-form macro). The host API is
unchanged: `KeyBiasGpuSession::attach`/`write_table` still take `&[f32]` and convert RN-even via
`kv_f16_bits::f32_to_f16` (the round-trip-tested twin) at the wire. Layout contract unchanged
(`KeyBiasLayout { head_stride }`, head-major, slot-indexed).

## Measured (4090, release, desktop-composition-only provenance printed per run)

| wire | p | incumbent | zero-armed | varying-armed | ratios |
|---|---|---|---|---|---|
| f32 (Bench-in-T1.4) | 2048 | — | — | — | 1.1012..1.1234 |
| f32 | 4096 | — | — | — | 1.1258..1.1338 |
| f16, bit-manip decode | 2048 | 0.959 ms | 1.187 ms | 1.233 ms | **1.2375 / 1.2857** |
| f16, `cvt.f32.f16` run 1 | 2048 | 0.950 | 0.975 | 0.970 | 1.0260 / 1.0204 |
| | 4096 | 2.987 | 3.034 | 3.084 | 1.0158 / 1.0326 |
| run 2 | 2048 | 0.956 | 0.977 | 0.978 | 1.0218 / 1.0233 |
| | 4096 | 2.980 | 3.021 | 3.023 | 1.0136 / 1.0143 |
| run 3 | 2048 | 0.945 | 0.963 | 0.963 | 1.0186 / 1.0187 |
| | 4096 | 2.965 | 3.003 | 3.020 | 1.0129 / 1.0186 |
| ratchet-commit run | 2048 | 0.957 | 0.974 | 0.976 | 1.0180 / 1.0202 |
| | 4096 | 2.976 | 3.019 | 3.029 | 1.0144 / 1.0179 |

**Verdict: 1.013–1.033 by cell, verdict-stable ×4 — a ~9–11-point tax cut at both league
lengths, beating the ~6% prediction.** Measured max 1.0326; the inter-run spread (±1–2%) is the
noise floor. **The ratchet re-pins ≤1.05**: ~1.7% headroom over the observed max, and every
demoted implementation still reds (f32 direct 1.0875+, smem stage 1.1657, bit-manip decode
1.2375).

## The two measured lessons

1. **The decode arithmetic is the tax, not the bytes.** The first cut decoded the u16 table
   with the branchy textbook bit-manip form (~10 ALU ops/element) and measured WORSE than the
   f32 table (1.2375/1.2857): the T1.4 hoist moved only the LOADS under the KQ mma — the decode
   arithmetic runs after it, on the critical path, 8 elements/thread. The fix is the
   one-instruction hardware widen `cvt.f32.f16` (IEEE-exact identical to the bit form —
   subnormals/Inf/NaN included; inline PTX already proven in the FA module by `f32_to_f16_rn`'s
   `cvt.rn.f16.f32`). Every kb module carries a module-local PTX decode; the shared
   `kv_f16_to_f32` is UNTOUCHED so the Issue-753 f16-KV kernels' compiled code stays
   byte-identical.
2. **The canonical + FA sources share one nvrtc program** (`CudaAttnKernels::new_standalone()`
   concatenates them): a global-scope `kb_f16_to_f32` in the canonical module is ambiguous with
   `fa_mma::kb_f16_to_f32` through `using namespace fa_mma` at the FA call site — the
   canonical helper is `kb_f16_canon_to_f32`.

## Byte-identity regression (the f16 wire must not move a bit at the inert/zero/f16-exact postures)

- 45 in-file attention tests incl. the 6 kb byte-identity tests (decode, devpos, splitgqa rows,
  verify2p, canonical, mma flag-off) — GREEN.
- `plan622_keybias_fa_gate` (bit-identity arms + stride-pad independence + devpos twin + causal
  CPU twins at the bench_946 bars) — GREEN.
- `plan620_prefill_fa_si_gate` — GREEN.
- League walk (real Bonsai-2, FA arm, graphs ON): flag-off/zero/disarm/detach bit-identical,
  moved deterministic ×2 at **Δlogit 1.0105 — the f32 record's exact value** (the ±12 fixture
  is f16-exact).
- Synth walk: all arms green, moved ×2 at **Δlogit 0.0831, same fnv** (`ccbff3976582f6f2`).

## En-route records

- The f16 edit's fuzzy match doubled a closing brace in the canonical ATTENTION macro — the M3
  windows clippy cannot see inside the CUDA strings; nvrtc caught it at the box (`b227d3a`).
  All four kb modules now carry a brace-balance check in review.
- The mma module's own `kv_f16_to_f32` transcription misses the +112 exponent re-bias in its
  normal branch — REACHABLE under `QWEN38_KV_DTYPE=f16` + the mma score knob. Left untouched
  (the known-broken arm's declared posture); flagged for the Plan-551/773 lane's owner.

## Cost-gate posture (unchanged law)

The DEFAULT posture stays free (flag-dead branch — the kb gate's flag-off byte-identity); the
armed tax is the Phase-2 pruned-prefill admission price, paid only armed. Any promotion cites
this record and is OWNER-GATED.
