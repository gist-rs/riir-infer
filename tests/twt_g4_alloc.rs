//! G4 — the TWT accumulate path is allocation-free (Issue 022 T1.5).
//!
//! Its OWN test target, deliberately: the counting allocator is
//! process-global and `cargo test` runs a target's tests in parallel
//! threads in ONE process, so sibling tests' allocations leak into any
//! count taken beside them (the first cut measured 1849 phantom allocs
//! from exactly that). One test per binary is the isolation.
//!
//! Runs: `cargo test -p riir-infer-core --features twt_profile --test twt_g4_alloc`
#![cfg(feature = "twt_profile")]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use riir_infer_core::twt::{planted_corpus, PairCosineAccum, SMatrixBuilder};

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
fn g4_accumulate_path_is_alloc_free() {
    let corpus = planted_corpus(9, 6, 12, 32, &[&[1, 2]]);
    let mut b = SMatrixBuilder::new(6, 12);
    b.begin_forward(32).unwrap();
    let before = ALLOCS.load(Ordering::Relaxed);
    for layer in 0..6 {
        for (r, states) in corpus.iter().enumerate() {
            b.push(layer, r, &states[layer]).unwrap();
        }
    }
    b.end_forward().unwrap();
    let after = ALLOCS.load(Ordering::Relaxed);
    assert_eq!(before, after, "push/fold must not allocate");

    // PairCosineAccum directly (T1.1's own surface).
    let mut acc = PairCosineAccum::new();
    let before = ALLOCS.load(Ordering::Relaxed);
    for states in &corpus {
        acc.add(&states[1], &states[2]);
    }
    assert_eq!(before, ALLOCS.load(Ordering::Relaxed), "add must not allocate");
    assert_eq!(acc.distance(), 0.0, "clone pairs finalize at exact zero");
}
