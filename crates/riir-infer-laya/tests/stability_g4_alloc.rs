//! G4 — the Issue 1106 stability tracker never allocates on its hot
//! path (`observe` + the depth scan).
//!
//! Its OWN test target, deliberately: the counting allocator is
//! process-global and `cargo test` runs a target's tests in parallel
//! threads in ONE process, so sibling tests' allocations leak into any
//! count taken beside them (the root crate's `twt_g4_alloc` documented
//! 1849 phantom allocs from exactly that). One test per binary is the
//! isolation.
//!
//! Runs: `cargo test -p riir-infer-laya --features twt_profile --test
//! stability_g4_alloc`
#![cfg(feature = "twt_profile")]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use riir_infer_laya::laya::riir::encoder::CaptureStage;
use riir_infer_laya::laya::riir::stability::{StabilityBand, StabilityTracker};

static ALLOCS: AtomicUsize = AtomicUsize::new(0);

struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

#[test]
fn g4_stability_track_is_alloc_free() {
    let (stages, d) = (8usize, 64usize);
    let mut tr = StabilityTracker::with_stage_capacity(stages, d);
    // Pre-build the stage states (construction allocations happen
    // BEFORE the counted region — the gate measures the track).
    let states: Vec<Vec<f32>> = (0..stages)
        .map(|s| (0..d).map(|i| ((s * 31 + i) % 7) as f32 * 0.125 - 0.5).collect())
        .collect();
    let stage_at = |s: usize| {
        if s == 0 {
            CaptureStage::Embedding
        } else {
            CaptureStage::AfterLayer(s - 1)
        }
    };
    let band = StabilityBand { cos: 0.999, l2: 0.01 };

    let before = ALLOCS.load(Ordering::Relaxed);
    for (s, st) in states.iter().enumerate() {
        tr.observe(stage_at(s), st);
    }
    let depth = tr.stabilization_depth(&band);
    let max = tr.max_leaving_delta();
    let after = ALLOCS.load(Ordering::Relaxed);
    assert_eq!(before, after, "observe + depth scan must not allocate");
    // Both reads ran (the gate must not pass on skipped work).
    assert_eq!(tr.stages_observed(), stages);
    assert!(depth.is_some() || max.is_some());
}
