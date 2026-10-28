//! Issue 1151 — the dist `gpu` posture floor.
//!
//! WHAT THIS GUARDS: with `gpu` forwarding `riir-infer-gpu/production`,
//! this test passes BY CONSTRUCTION. Its job is to red the day someone
//! replaces that forward with a hand-typed rung list again and drops a
//! promoted rung (the 1141/1146/1146-2a class — three fires in nine
//! days, worst cell 1.74x, one cell WRONG LOGITS), or the day the
//! substrate's `PRODUCTION_RUNGS` manifest drains to empty (which would
//! make the missing-check vacuous). Delete this test and nothing else
//! pins the posture. It runs only under `gpu` — the `[[test]]`
//! required-features row keeps every other posture from compiling it to
//! a green zero.
//!
//! WHAT IT DOES NOT CLAIM: that the rungs are CORRECT at runtime (that is
//! the per-rung G1 gates) or that the binaries were BUILT with this
//! posture (only this test compilation proves selection).

#![cfg(feature = "gpu")]

#[test]
fn gpu_posture_carries_every_promoted_production_rung() {
    // The drained-manifest defense FIRST: an emptied PRODUCTION_RUNGS
    // would make the missing-check below vacuously true.
    assert!(
        !riir_infer_gpu::PRODUCTION_RUNGS.is_empty(),
        "riir-infer-gpu PRODUCTION_RUNGS drained — the posture floor is \
         vacuous (Issue 1151)"
    );
    let missing = riir_infer_gpu::missing_production_rungs();
    assert!(
        missing.is_empty(),
        "dist gpu posture is missing promoted serving rungs {missing:?} — \
         forward `riir-infer-gpu/production` (Issue 1151); hand-typing the \
         rung set here is the 1141/1146/1146-2a class"
    );
    assert!(riir_infer_gpu::production_posture_complete());
}
