//! G4 — the Phase-4 arms' accumulate paths are allocation-free (Issue 022
//! T4.1, the twt_g4_alloc precedent). Its OWN test target, deliberately:
//! the counting allocator is process-global and parallel sibling tests
//! would leak allocations into any count taken beside them — one test
//! per binary is the isolation.
//!
//! Runs: `cargo test -p riir-infer-core --features twt_collapse --test twt_ternarize_g4`
#![cfg(feature = "twt_collapse")]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use riir_infer_core::twt::ternarize::{materialization_rel_err, ARM_B_TAU_CODE};

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
fn g4_the_rel_err_loop_is_alloc_free_with_caller_scratch() {
    let rows = 8usize;
    let cols = 256usize;
    let n_x = 32usize;
    // Deterministic operators + inputs.
    let mut seed = 0x12345678u64;
    let mut next = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let v = ((seed >> 33) as u32) as f32 / u32::MAX as f32 - 0.5;
        v * 0.1
    };
    let a: Vec<f32> = (0..rows * cols).map(|_| next()).collect();
    let b: Vec<f32> = (0..rows * cols).map(|_| next()).collect();
    let xs: Vec<f32> = (0..n_x * cols).map(|_| next()).collect();
    let mut ya = vec![0f32; rows];
    let mut yb = vec![0f32; rows];

    let before = ALLOCS.load(Ordering::Relaxed);
    let err =
        materialization_rel_err(&a, &b, &xs, rows, cols, n_x, &mut ya, &mut yb).unwrap();
    let after = ALLOCS.load(Ordering::Relaxed);
    assert_eq!(before, after, "the rel-err loop must not allocate");
    assert!(err.is_finite());

    // The pre-registered τ sits where the gates read it.
    assert_eq!(ARM_B_TAU_CODE, 1);
}
