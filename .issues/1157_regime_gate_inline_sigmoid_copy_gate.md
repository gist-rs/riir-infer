# riir-infer Issue 1157 — regime_gate.rs inline `fn sigmoid` — copy-gate class, delegation available

**Status:** OPEN — detection-only (substrate-first drift audit 2026-10-30; fix deferred to a quiet build window)

## The site

Found by the substrate-first Mode-2 delta audit (window: after `35fa03065`
2026-10-30T00:16:28Z; landed in `459984b`, i1156 T4/T5).

**Site:** `crates/riir-infer-gpu/src/regime_gate.rs:265` —
`fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }` (two-branch stable form).

**Why this is the copy-gate class, not a lead:**

- `katgpt-core` is a DIRECT workspace dep of this crate (`Cargo.toml` line 35,
  "katgpt-rs being the one sanctioned public upstream"). Delegation is
  trivially available.
- The doc comment ("Stable sigmoid — never softmax (the latent-space rule).
  Local decision arithmetic, not a synced value") justifies the CHOICE of
  sigmoid over softmax — it says nothing about why the substrate's
  implementation is not consumed. That is the missing half of the copy-gate
  convention (in-source rationale for the COPY itself).
- No delegation-divergence pin. The existing test (`sigmoid_is_monotone_and_
  bounded`, line 878) pins the MATH against the closed form; it is not the
  delegation pin the convention asks for.

**Vocabulary translation (substrate-first skill):** searched
`fn sigmoid|1.0/(1.0+` across the 261-file post-anchor delta. Substrate:
`katgpt_core::sigmoid` (f32, Cephes ~1 ULP) / `katgpt_core::exact_sigmoid_f64`
(bit-exact reference). Sanctioned precedent: riir-refine `drafter.rs:286`
(the ndb Issue-611 delegation wrapper) and reflex Issue 102's two discharged
sites (`router.rs:111`, `robust_ab.rs:278` — `exact_sigmoid_f64` +
`sigmoid_delegation_matches_frozen_legacy_body` pin).

**Proposed fix (deferred — build window):**

Delegate to `katgpt_core::sigmoid`, keeping the local name as a thin wrapper
(the sanctioned delegation shape), OR keep the local body and add the two
convention lines: the in-source copy rationale (numerics mirror of the CUDA
arm, if that is the reason) + a divergence-failing test against
`katgpt_core::sigmoid` over a fixed grid.

Note: if the body intentionally mirrors a CUDA-side sigmoid bit-for-bit
(i1156's kernel-parity posture), delegation is the WRONG fix — say so
in-source instead and cite the kernel; the convention requires the
rationale either way.

**Classification:** DRY violation (copy-gate class), detection-only. Sibling
site noted WITHOUT filing: seal-remake `seal-view/src/locomotion_gait.rs:72`
(pre-window 10-08 `a00bae11`) — katgpt-core is NOT a direct seal-view dep and
the crate is wasm32-first, so the feature-isolated local-math class applies;
it is missing only its one-line rationale marker (recorded in the audit run
log, katgpt-rs substrate-first SKILL.md 2026-10-30 row).
