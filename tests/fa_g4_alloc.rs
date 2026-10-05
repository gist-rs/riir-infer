//! Issue 035 P0 — G4: the `fa_posterior::sample_joint` hot path allocates
//! nothing in steady state. Its OWN test target, deliberately (the twt
//! G4 precedent): the counting allocator is process-global and parallel
//! sibling tests would leak allocations into any count taken beside them.
//!
//! The scratch is warmed once (setup allocations allowed); the measured
//! phase runs many draws and must report ZERO allocations.
//!
//! Runs: `cargo test -p riir-infer-core --test fa_g4_alloc`

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

static ALLOCS: AtomicUsize = AtomicUsize::new(0);
static DEALLOCS: AtomicUsize = AtomicUsize::new(0);

struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        DEALLOCS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

use riir_infer_core::fa_posterior::{AutomatonBuilder, FaScratch, FREE, SplitMix64};

#[test]
fn sample_joint_is_allocation_free_in_steady_state() {
    // A mid-size automaton: 8 nodes, vocab 64, mixed fanouts + token sets
    // (an edge may allow MORE tokens than a node's fanout — the scratch
    // sizing case the weights buffer must cover).
    let mut b = AutomatonBuilder::new(8, 64, 0);
    b.accept(7);
    for s in 0..7 {
        if s % 2 == 0 {
            b.edge(s, s + 1, &(0..64).collect::<Vec<u32>>());
        } else {
            b.edge(s, s + 1, &[0, 1, 2, 3, 4, 5]);
            b.edge(s, (s + 2).min(7), &[6, 7, 8]);
        }
    }
    // The accept node holds with a full-vocab self-loop (blocks longer than
    // the walk stay satisfiable; the posterior keeps drawing inside it).
    b.edge(7, 7, &(0..64).collect::<Vec<u32>>());
    let fa = b.build().unwrap();

    let len = 32;
    let logits: Vec<f32> = (0..len * 64).map(|i| ((i * 7919) % 101) as f32 / 8.0 - 6.0).collect();
    let forced = [FREE; 32];

    // Warm the scratch across several draws of the LARGEST block (ensure()
    // grows lazily; the warmup also touches the full token-set path).
    let mut scratch = FaScratch::new();
    let mut rng = SplitMix64::new(7);
    let mut out = vec![0u32; len];
    for _ in 0..8u64 {
        fa.sample_joint(&mut scratch, &logits, len, &forced, 1.0, false, &mut rng, &mut out)
            .unwrap();
    }

    // ── the measured phase ──────────────────────────────────────────
    let before = ALLOCS.load(Ordering::Relaxed);
    for i in 0..64u64 {
        let seed = 1000 + i;
        let mut r = SplitMix64::new(seed);
        fa.sample_joint(&mut scratch, &logits, len, &forced, 1.0, false, &mut r, &mut out)
            .unwrap();
    }
    let after = ALLOCS.load(Ordering::Relaxed);
    assert_eq!(
        after - before,
        0,
        "sample_joint allocated in steady state: {before} → {after}"
    );
}
