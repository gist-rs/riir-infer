# Issue 046: GDN spectral-class spec test (closed-form) + report-only state-norm ratio probe

**Status:** Open — modelless, spec-test primary; monitor demoted to report-only probe (premise corrected at verdict round 1).

## Correction record (read first)

The ORIGINAL premise of this issue — "heads with g→1 accumulate state norm ~1/(1-g), silent f32 erosion" — was **WRONG for this lane**, caught at Research 620's verdict round 1. The `‖u‖/(1-ρ)` blow-up bound (arXiv:2610.13341 §C.3) applies to ADDITIVE writes (linear attention, or Mamba with phase removed). Our recurrence (`src/deltanet/forward.rs:14`) is the **delta rule** with L2-normalized keys and β = sigmoid:

```
S = g·S + k̃ ⊗ β(v − (g·S)ᵀk̃)
```

Along the key direction (k̃ unit-norm): `Sk ← g(1−β)·Sk + β·v`, steady state `β‖v‖/(1−g+gβ) ≤ ‖v‖` for **every** g ≤ 1, β ∈ (0,1] — the error-correcting write IS a keyed erase, and the lane is bounded by construction. There is no growth hazard to monitor here.

## What remains actionable

1. **Spectral-class spec test** (primary deliverable, the house FV convention): assert the Bonsai GDN transition's spectral class **with the correct algebra** — one step is `T = g·(I − βk̃k̃ᵀ)`, eigenvalues `{g(1−β)` along k̃, `g` with multiplicity d−1}` (this is arXiv:2610.13341 App. A.4's Gated-DeltaNet pair {−(γ+b), −γ} in continuous time, b = −ln(1−β)/Δ, γ = −ln g/Δ). ⚠ Do NOT pin `{g, 0}` — that is the DeltaNet row (App. A.3, no decay gate), a different recurrence. Numeric half: recovered eigenvalue magnitudes match `expm` of the frozen continuous-time generator to f32 tolerance. Lean one-sitter + paired Rust spec-match, or a plain numeric spec test if the FV instance is judged overweight.
2. **Report-only state-norm ratio probe** (demoted, conditional): `‖S‖/max‖v‖` per (layer, head) on a fixed replay workload, negative-control designed (the boundedness argument predicts ratio O(1); if the probe never exceeds a modest factor across the fleet replay, the probe earns nothing and the issue closes as spec-test-only). `None` tap = bit-identical (the `kv_evict` seam precedent).

## Non-goals

- No behavior change to the recurrence (the oscillation lane is riir-train Plan 459's, and it is probe-shaped).
- No attention-KV application: `kv_evict` rows carry no per-entry (ρ,θ).
- No growth-alarm threshold: the boundedness argument above forbids the growth class this issue originally feared.

## References

- Research: `../katgpt-rs/.research/620_Mamba3_Oscillatory_Dynamics_Damping_Regimes.md` (§2 Path 0; §3.1 duality — the delta term is the COARSE keyed erase)
- Training-side twin: `../riir-train/.plans/459_oscillatory_gdn_damping_probe.md`
- Paper: arXiv:2610.13341 App. A.3 (DeltaNet spectrum — the {1−β, 0} row), App. A.4 (Gated DeltaNet — OUR row), §C.3 (the additive-write bound that does NOT transfer here)
