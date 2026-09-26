# Bench 871 — Issue 879 Phase 1: GDN quant-certification lockstep (arXiv:2609.04098 T1.1/T1.2)

> **Moved 2026-09-26:** from riir-ai `.benchmarks/` (same number; Issue 879 was resolved in the riir-ai ledger — its record lives in riir-ai HISTORY.md). The harness moved here with the carve: `tests/issue879_gdn_quant_certification.rs` in THIS repo (was `crates/riir-infer-core/tests/…` pre-carve).

**Verdict: CERTIFIED (synthetic + production-geometry arms)** — the paper's two
survival mechanisms REPRODUCE on our stack at our (more aggressive) quantization
point: the Q2_0 state-error plateau is FLAT (no compounding over the window) and
the delta rule erases state impulses faster than decay implies.

**Status:** RECORD (2026-09-06, CPU-only; harness
`crates/riir-infer-core/tests/issue879_gdn_quant_certification.rs`, target
registered with `required-features = ["deltanet_ternary_inference"]`; 4090 box,
no GPU use — CPU matvecs only).

## The instrument

Lockstep per the paper's §5: the same GDN layer runs twice on identical input
streams — **clean** = `Proj::Dense` f32 weights; **quant** = the exact PrismML
`quantize_row_q2_0_ref` rounding (`d = f16(amax)` per 128-weight group,
`code = round(w/d)+1` ∈ {0,1,2}) packed into real `BlockQ2_0`s and served
through the PRODUCTION container + matvec (`repack_q2_0_to_ternary_group` →
`Proj::Ternary` → `simd_ternary_group_matvec`). Shared f32 (never quantized,
per [Bench 870](870_ternary_recipe_audit.md)): conv1d, `a_log`, `dt_bias`,
norms. Everything upstream of the state diff is `forward_deltanet_layer`
itself — the audited CPU reference — zero duplicated substrate.

Fixtures: xorshift64* seeded streams (platform-stable); gate regimes set via
`a_log` scaling (a_raw ≈ 0.05·U ⇒ softplus ≈ 0.71, so a_log_scale 1.0 →
ᾱ ≈ 0.49, a_log_scale 0.014 → ᾱ ≈ 0.99).

## T1.1 — plateau flat: PASS

Predicate: `max(relS)/median(relS[last quarter]) ≤ 1.2`.

| Arm | Geometry | Window | plateau | max | flatness |
|---|---|---|---|---|---|
| fast-decay | n_embd 512, 2k/6v heads | 4096 | 0.6516 | 0.7193 | **1.104** |
| long-memory | n_embd 512, 2k/6v heads | 4096 | 0.6741 | 0.6853 | **1.017** |
| **production geometry** | **n_embd 5120, 16k/48v, hd 128** | **8192** (`I879_WINDOW`) | **0.6799** | **0.6857** | **1.009** |

The absolute plateau (~0.68) is high — Q2_0 is 2.125 bpw, far past the paper's
W4A4 — but the claim that matters is SHAPE: the error is established within the
first steps and holds flat for the whole window. No compounding, no drift, no
divergence. The recurrent half of the hybrid does not accumulate quantization
noise; this is the paper's core mechanism, confirmed at a harsher operating
point than the paper's own.

## T1.2 — impulse erasure: PASS

One-off impulse = a random direction with L2 norm exactly 1% of ‖S‖, injected
into one of two clean trajectories (the per-step map is affine in S, so the
trajectory gap IS the impulse response). Predicate: the contraction bound —
steps-to-1/e ≤ `1/(1−λ_max)` (the per-head max-decay horizon); the paper's
separation claim (erasure ≪ decay) is then visible in the numbers.

| Arm | steps-to-1/e | horizon (mean ᾱ) | horizon (λ_max) |
|---|---|---|---|
| fast-decay | **1** | 2.1 | 2.2 |
| long-memory (synthetic) | **83** | 109.8 | 120.1 |
| **production geometry** | **82** | **107.2** | **181.4** |

The production-geometry row is the paper's separation, live: erasure in 82
steps beats even the mean-decay horizon (107) by 1.3× and the λ_max bound
(181) by 2.2× — the rank-1 overwrites erase state error faster than decay
alone would.

## Harness-bug lessons (recorded for the next person)

1. **Impulse must be norm-scaled, not element-scaled.** The first run injected
   a constant into every state element: the actual injected norm was
   `delta·√state_len` ≈ 313× the recorded 1%, and steps-to-1/e exceeded the
   contraction bound by ~ln(313·e)/1 extra time constants — 10 observed vs 2.2
   bound. The fix (normalized direction × 0.01·‖S‖) turned the same dynamics
   into 1-step erasure (fast regime). A "1% impulse" is a NORM claim.
2. **The contraction bound is λ_max, not mean-ᾱ.** Per-head maps are
   contractions bounded by their own λ; the gap decays no faster than the
   slowest head allows. Asserting against the mean-decay horizon fails
   spuriously in multi-head fixtures with head-spread decay.
3. `sort_by(katgpt_core::float_order::asc)` does not compile over `Vec<f32>` —
   the deref lesson (riir-ai Issue 841) again: closures take `&f32`, the
   helpers take `f32`. Use `|a, b| asc(*a, *b)`.

## Scope + what remains

- **T1.3 (production artifact) — DONE, reframed honestly.** The paper's §5
  clean-vs-fake-quant lockstep is STRUCTURALLY INAPPLICABLE to a PTQ'd
  checkpoint: dequantize(stored blocks) → re-quantize reproduces the identical
  blocks (d = amax{−d,0,+d} = d — the roundtrip is idempotent), so relS ≡ 0
  identically (first run: max=0.00000 plateau=0.00000). The absolute plateau
  on real weights needs the PRE-PTQ checkpoint (a riir-train training-pipeline
  artifact). What T1.3 measured instead (unique, valid): the REAL decay
  distribution + real-weight dynamics at 2048-step window:

  | Layer | ᾱ (mean) | per-step decay max | impulse 1/e | run λ_max horizon |
  |---|---|---|---|---|
  | blk.0 | 0.813 | **0.992** | **27** | 61.1 |
  | blk.30 | 0.795 | 0.954 | 5 | 11.3 |
  | blk.62 | 0.751 | 0.963 | 4 | 10.2 |

  Layer 0 carries the long-memory heads (λ up to 0.992/step ⇒ horizons to
  ~128); erasure beats the λ_max horizon 2.3× there, matching the paper's
  separation. State norms bounded (0.274/0.390/0.816). The real head
  heterogeneity (max ≫ mean) is exactly why the mean-only horizon was the
  wrong predicate — recorded above.
- Runbook (production artifact):
  `CARGO_TARGET_DIR=/tmp/i879_cert I879_WINDOW=2048 cargo test -p riir-infer-core
  --features deltanet_ternary_inference --test issue879_gdn_quant_certification
  --release -- --ignored t1_production_artifact --nocapture` (29.3 min; the
  ternary GEMV on stored Q2_0 blocks is the cost).
- Phase 2 (T2) is complete — [Bench 870](870_ternary_recipe_audit.md).
- **T3** (KV × weight-quant interaction, NLL-by-position) remains — needs a
  full-model forward harness, a separate lane.
- **T4** (league doc note) remains — doc-only.
