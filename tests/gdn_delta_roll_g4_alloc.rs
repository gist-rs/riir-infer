//! Issue 049 G4 — the GDN delta-roll journal's hot path is allocation-free.
//!
//! Its OWN test target, deliberately: the counting allocator is
//! process-global and `cargo test` runs a target's tests in parallel
//! threads in ONE process, so sibling tests' allocations leak into any
//! count taken beside them (the twt_g4_alloc lesson). One test per binary.
//!
//! Runs: `cargo test -p riir-infer-core --features gdn_delta_roll --test
//! gdn_delta_roll_g4_alloc`
#![cfg(feature = "gdn_delta_roll")]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use riir_infer_core::deltanet::delta_roll::{GdnStateJournal, MIN_PRODUCT_EPS_DEFAULT};
use riir_infer_core::deltanet::forward::gated_deltanet_step_inplace;

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
fn g4_record_gate_unwind_path_is_alloc_free() {
    const H: usize = 8;
    const K: usize = 128;
    const V: usize = 128;
    const W: usize = 512;
    let mut journal = GdnStateJournal::with_capacity(H, K, V, W);
    let mut state = vec![0.1f32; H * K * V];
    let mut output = vec![0.0f32; H * V];
    let mut kv_mem = vec![0.0f32; H * V];
    let mut delta = vec![0.0f32; H * V];
    let k: Vec<f32> = (0..H * K).map(|i| (i % 17) as f32 * 0.01).collect();
    let q: Vec<f32> = (0..H * K).map(|i| (i % 13) as f32 * 0.01).collect();
    let v: Vec<f32> = (0..H * V).map(|i| (i % 7) as f32 * 0.01).collect();
    let beta: Vec<f32> = (0..H).map(|h| 0.3 + 0.01 * h as f32).collect();
    let decay: Vec<f32> = (0..H).map(|h| 0.9999 - 0.0001 * h as f32).collect();

    // Warmup: lazy init of anything the path touches (rayon globals, etc.).
    gated_deltanet_step_inplace(
        &q,
        &k,
        &v,
        &mut state,
        &beta,
        &decay,
        H,
        K,
        V,
        &mut output,
        &mut kv_mem,
        &mut delta,
    );
    journal.record_step(&k, &v, &kv_mem, &beta, &decay);
    let _ = journal.window_depth_limit(MIN_PRODUCT_EPS_DEFAULT);
    let _ = journal.unwind_to(&mut state, journal.len() - 1, MIN_PRODUCT_EPS_DEFAULT);

    // Measured section: W steps of record (against the REAL step, whose
    // kv_mem the journal consumes) + gate + full unwind + clear + re-record
    // — all within the reserved capacity.
    // Measured section: W steps of record (against the REAL step, whose
    // kv_mem the journal consumes) + gate + full unwind + clear + re-record
    // — all within the reserved capacity. The warmup's one record is
    // cleared first so the measured window fits the reservation exactly.
    journal.clear();
    ALLOCS.store(0, Ordering::Relaxed);
    for _ in 0..W {
        gated_deltanet_step_inplace(
            &q,
            &k,
            &v,
            &mut state,
            &beta,
            &decay,
            H,
            K,
            V,
            &mut output,
            &mut kv_mem,
            &mut delta,
        );
        journal.record_step(&k, &v, &kv_mem, &beta, &decay);
    }
    let after_record = ALLOCS.load(Ordering::Relaxed);
    let limit = journal.window_depth_limit(MIN_PRODUCT_EPS_DEFAULT);
    let after_gate = ALLOCS.load(Ordering::Relaxed);
    let unwound = journal.unwind_to(&mut state, journal.len() - limit, MIN_PRODUCT_EPS_DEFAULT);
    let after_unwind = ALLOCS.load(Ordering::Relaxed);
    assert!(
        unwound.is_ok(),
        "the 0.9999-class window must admit its own depth"
    );
    journal.clear();
    journal.record_step(&k, &v, &kv_mem, &beta, &decay);
    let measured = ALLOCS.load(Ordering::Relaxed);
    eprintln!(
        "g4 sections: record={after_record} gate={after_gate} unwind={after_unwind} total={measured}"
    );
    assert_eq!(
        measured, 0,
        "record + gate + unwind + clear allocated {measured} times"
    );
}

/// The allocator liveness canary: the counter itself must be live — a
/// counting allocator that counts nothing proves nothing (the green-zero
/// law at the instrument layer).
#[test]
fn alloc_counter_is_live() {
    let before = ALLOCS.load(Ordering::Relaxed);
    let probe: Vec<u8> = vec![0u8; 1024];
    assert!(probe.len() == 1024);
    let after = ALLOCS.load(Ordering::Relaxed);
    assert!(after > before, "the counting allocator counted nothing");
}
