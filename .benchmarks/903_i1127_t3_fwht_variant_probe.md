# Bench 903 — Issue 1127 T3 kernel probe: the bit-identical fix space of the `sign_fwht_quantize` family is ~0.3% e2e — the rung does not open

**Status:** COMPLETE — T3 CLOSED measured-negative at the bit-identical boundary. The Bench-902 census verdict (T3 census-POSITIVE, 7.2% of token at 8–95× over roofline) stands as MEASURED but its reclaimable share under the FNV-pin law is ~0.3% — an order under the +2.71% re-cross bar. No rung; the decode-hold re-cross rides the already-staged arms (Bench 900 T1 +0.77%, Bench 899 2′ +0.6–1.0%, GLU pp rows) in the owner-gated T4 window.
**Date:** 2026-10-26/27 (amo → shikuwa-4090, ssh + schtasks; zed-amo-i1127-t1)
**Vehicle:** `crates/riir-infer-gpu/tests/bench903_fwht_rotq_variant_probe.rs` — standalone nvrtc source: the shipped K1/K4/K5 verbatim (P0) + bit-identical-by-construction variants; per-shape identity gate (i8 eq + ascale to_bits eq) → interleaved burst A/B (B903_BURST=300, B903_PAIRS=24, arms alternate per pair, median of ratios) + A/A control. Box: 4090, PROVENANCE line quoted in-log (3% util desktop noise at start); NOT a scored league cell.

## The variants (all shipped-math-preserving by construction)

| arm | content | identity verdict |
|---|---|---|
| P0 | shipped kernels verbatim | reference |
| V1 | float4 GLOBAL loads in the fill (scalar smem stores; full shipped butterfly) | **OK ×3 shapes** |
| V2 | V1 + register-tiled butterfly stages 2+4 + shfl_down reduction tail | DRIFT ×3 |
| V3 | V1 + shfl_down reduction tail only (full smem butterfly) | **OK (rmsnorm — the only kind with a full-row reduction)** |
| V4 | V1 + register-tile only (shipped reduction) | DRIFT (rmsnorm) — the isolating arm |

## The bisect: the reg-tile is the drifter, and the signature is the recorded nvcc contraction-pairing class

V4's drift: **0 i8 mismatches + ascale-only mismatches** (106/320 @5120 rmsnorm, 130 @5120 gate_sigmoid, 407/1088 @17408 swiglu). i8 = `round(v/absmax·127)` is invariant to a common-mode 1-ULP scale of a group; the stored absmax float is not — exactly the ±1-ULP common-mode signature. The same pairwise butterfly ops on register-resident operands pair differently under nvrtc than the smem-load form (the Bench-900 record's nvcc contraction-pairing finding, now reproduced in isolation). Dead by the FNV-pin law — a ±1-ULP walk drift breaks `98412dc406c4ea19`/`63496f6688c8f742` and every committed hash downstream of the family.

V3's shfl-tail (levels 16..1 as in-warp `__shfl_down_sync`, distance pairing, upper lanes discarded — NOT an xor tree) is **bit-identical**: same adds, same order, provably and measured.

## Timing (per-launch, back-to-back bursts — the in-graph view)

| shape | P0 µs | V1 | V3 |
|---|---|---|---|
| 5120 rmsnorm | 7.65 | 7.46 (**+3.1%**) | 7.59 → 1.067× (**+6.7%**) |
| 5120 gate_sigmoid | 7.21 | 7.56 (**−6.3%, a LOSS**) | — (n/a: no reduction) |
| 17408 swiglu | 7.30 | 7.02 (**+3.7%**) | — |

(A/A control ratio 0.9895 — the harness noise reference. The absolute per-launch µs carry ~1-2 µs burst-gap overhead vs the census's in-graph numbers; the RATIOS are the measurement. gate_sigmoid's V1 loss: its 16/token launches at dim 5120 are sigmoid-SFU-latency-dominated; the vectorized fill buys nothing and the wider loads cost issuing bandwidth.)

## The verdict arithmetic

Census family weights (Bench 902): rmsnorm 408 + gate_sigmoid 77 + swiglu 303 = **788 µs/token (7.2% of the 10.9 ms token)**. Shipping-ratio projection (V3 where it exists, V1 elsewhere): **756 µs/token → 32 µs saved = 0.29% e2e**.

The census's 8–95× over-roofline headroom is REAL but sits behind the bit-identity wall: reclaiming it needs value-changing restructuring (the reg-tile class drifts 1-ULP; a from-scratch multi-block reduction reorders the pinned reduction tree). Any of those breaks the league G1 walk/logits pins — which is a RE-PIN decision (lossy-law conversation, owner), not a modelless rung. **T3 closes: the modelless, bit-identical fix space of this family is ~0.3% e2e.**

## What remains for the decode-hold re-cross (+2.71% needed)

The staged arms, all measured-positive, all riding the owner-gated T4 window: T1 attention-chain fusion +0.77% (Bench 900, opt-in `RIIR_ATTN_CHAIN_FUSED=1`) + 2′ beta/expand +0.6–1.0% (Bench 899, opt-in) + GLU pp rows (Bench 897 class) + this probe's V3 (+0.29%-class, opt-in env if ever wired — NOT wired; the probe is the record). Sum ≈ +1.7–2.6% — at or just under the bar; the window bundles them with the pending opponent tag per the league's batching law.

## Refs

- riir-infer issue 1127 (T1–T3 + the owner-gated T4 re-cell bundle) · Bench 902 (the census) · Bench 900 (T1 rung + the nvcc contraction-pairing finding this probe isolated) · Bench 899/898 (the rung precedents) · riir-ai Bench 1004 (the RED) · issue 734 T5 (`rmsnorm_quantize_mb` — the multi-block pattern that WOULD reclaim the headroom behind a re-pin)
