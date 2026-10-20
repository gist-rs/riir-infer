# Bench 885 — Issue 049: the GDN delta-roll state journal (rank-1 unwind POC)

**Date:** 2026-10-20 · **Box:** shikuwa / 4090 RTX (sm_89), Windows 11, AC power, i7-13700K
(16 cores; CPU-side f32 math only — this primitive is CPU substrate, no GPU lane) ·
**GPU state at bench:** idle (9% desktop residue, 448 MiB — no compute consumers) ·
**Build:** riir-infer-core `--features gdn_delta_roll`, dev (gate tests) + release
(timing lane), pinned 1.98.1 toolchain, isolated `CARGO_TARGET_DIR=/tmp/riir-infer-049` ·
**Claim:** issue-049 live claim `4af9b75`, shikuwa/4090 riir-refine idle lane,
expires 2026-10-22T17:04Z.

## What landed

- `src/deltanet/delta_roll.rs` — `GdnStateJournal` (feature `gdn_delta_roll = ["deltanet_inference"]`,
  opt-in): post-commit GDN state recovery by rank-1 unwind from the final state + per-token
  `(u, k, β, g)` contexts. `u = v − kv_mem` is recorded at write time (kv_mem is the step's
  own post-decay prediction, already written by `gated_deltanet_step_inplace`), which makes
  the inverse `S_prev = (S − β·u·kᵀ)·(1/g)` safe for ANY β — the naive re-derivation divides
  by `(1 − β‖k‖²)`, singular exactly at the normalized-key / β = 1 operating point GDN lives at.
  The issue's per-key-dimension `diag(α)` collapses here: this repo's lane carries per-head
  SCALAR decay, so the conditioning product is one per-head scalar per token.
- `tests/gdn_delta_roll_gates.rs` — the G1 drift sweep + gate laws (`#![cfg]` +
  `[[test]] required-features`, green-zero law).
- `tests/gdn_delta_roll_g4_alloc.rs` — G4, own binary (the counting-allocator isolation law).

## The measured drift table (G1 — dev and release identical, deterministic seeds)

Unwind vs a FRESH sequential replay from the same base; rel = max-element error / state max-abs.
Bar: 1e-3 relative (asserted); floor `MIN_PRODUCT_EPS_DEFAULT = 0.25` on the per-head f64
decay product over the unwind window.

```
         g=0.999 W=512  t*= 128 product=6.810e-1 max_abs=1.180e-5 rel=5.741e-6
         g=0.999 W=512  t*= 256 product=7.740e-1 max_abs=8.702e-6 rel=3.857e-6
         g=0.999 W=512  t*= 384 product=8.798e-1 max_abs=4.530e-6 rel=1.683e-6
          g=0.99 W=128  t*=  32 product=3.810e-1 max_abs=8.017e-6 rel=6.339e-6
          g=0.99 W=128  t*=  64 product=5.256e-1 max_abs=5.007e-6 rel=3.243e-6
          g=0.99 W=128  t*=  96 product=7.250e-1 max_abs=2.325e-6 rel=1.567e-6
          g=0.95 W=64   t*=  48 product=4.401e-1 max_abs=6.855e-7 rel=8.149e-7
mixed[.90,1] W=128  t*=  96 product=2.658e-1 max_abs=1.311e-6 rel=1.822e-6
          g=0.95 W=64   t*= 16/32   REFUSED product=2.378e-1 (< 0.25)   fail-closed verified
mixed[.90,1] W=128  t*= 32/64   REFUSED product=2.251e-1 (< 0.25)   fail-closed verified
```

Worst admitted cell: **6.3e-6 relative** — 160× under the bar. Drift grows with window depth
(per-step f32 rounding under the `Π 1/g` amplification) exactly as the conditioning gate's law
predicts; the gate's floor keeps every admitted cell orders of magnitude inside the pin.
f64 escape hatch (same f32 contexts): 1.25e-6 vs f32 5.5e-6 at the drift-decomposition cell —
the unwind arithmetic contributes ~4.4× over the record rounding, both negligible.

## Timing (release, H=8, K=V=128, W=4096)

| lane | wall | per token |
|---|---|---|
| record + forward (incl. the test's own per-token k/v generation) | 325.7 ms | 79.5 µs |
| **unwind 2048 tokens (pure journal)** | **18.3 ms** | **8.9 µs** |
| fresh replay 2048 tokens (the recovery it replaces) | 109.0 ms | 53.2 µs |

Recovery is **~6× cheaper than re-running the forward**, with no weights and no raw `(k, v)`
rows needed — and any intermediate state, not just checkpoints the caller happened to keep.
Storage: **8 256 B/token vs 65 536 B/token materialized = 63.5×** (~64× at K = V = 128, the
issue's own arithmetic). Disclosures: the record-lane wall includes the test's per-token RNG
generation (allocation-bearing, not the journal's cost — `record_step` itself is two
`extend_from_slice` + one subtract pass); the unwind is scalar f32 with auto-vectorization,
no SIMD kernel written (a future `simd`-module arm is an obvious lever, deliberately not
gold-plated into the POC).

## Gates

- **G1 (correctness + drift pin):** PASS — table above; the envelope lesson below is the
  load-bearing finding.
- **G2 (perf):** recorded (the table above) — a POC capability claim, not a
  beat-the-incumbent gate: there IS no incumbent recovery path (that is the issue's premise).
- **G3 (no-regression):** PASS — `gdn_delta_roll` is not in default; default `cargo clippy
  --all-targets` clean (the module compiles to nothing); feature-on clippy `--all-targets`
  clean, 0 warnings.
- **G4 (alloc):** PASS — `tests/gdn_delta_roll_g4_alloc.rs`: W=512 record (against the real
  step) + gate + unwind + clear + re-record within a pre-reserved journal = **0 allocations**;
  counter liveness canary green. (En-route: the first run measured 4 phantom allocs — the
  warmup had consumed one capacity slot so the 512th measured record reallocated all four
  planes. `clear()` before the measured section; the sections instrument names record/gate/
  unwind individually now.)

## The envelope lesson (the sweep's real finding)

The first sweep run exploded to **rel = 1.1e8 (max_abs 4e13)** — not a journal defect: the
test generator produced UNNORMALIZED keys (`k ~ N(0, 0.3)`, `β‖k‖² ≈ 5.75 >> 1`), taking the
rank-1 write outside GDN's stability envelope (the repo's forward contract is
"L2-normalized by caller"; `β‖k‖² < 1` is what makes the write a contraction). The synthetic
forward itself was an unstable recurrence, its 512-step state was f32 garbage, and no
recovery machinery — journal or snapshot — is meaningful over garbage. The generators now
normalize per head and the drift table is the honest envelope posture. Pinned in the gates
file's `normalize_heads` doc: **the journal's laws hold inside the forward's caller contract;
outside it, nothing does.**

## Status

Feature stays OPT-IN (`gdn_delta_roll`) — promotion rides a consumer: the named seams are
the spec-decode retry paths, `kv_evict` eviction recovery, and the disaggregated handoff
lane (issue028); a handoff consumer wanting integrity hashes the state at journal birth
(BLAKE3 seam, deliberately not wired in v1). Schedule-specificity and local-only (never
synced raw) are pinned in the module doc.
