//! Issue 1155 G4 — the exact-slot RAM posture's hot loop is
//! allocation-free: step + journal record + unwind within a pre-reserved
//! capacity, driven through the REAL kernel one head at a time.
//!
//! Its OWN test target, deliberately: the counting allocator is
//! process-global and `cargo test` runs a target's tests in parallel
//! threads in ONE process, so sibling tests' allocations leak into any
//! count taken beside them (the twt_g4_alloc lesson). One test per binary.
//!
//! Runs: `cargo test --test issue1155_exact_slot_g4_alloc --features
//! gdn_delta_roll`
#![cfg(feature = "gdn_delta_roll")]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use riir_infer_core::deltanet::delta_roll::GdnStateJournal;
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

/// Deterministic LCG — same generator as the exactness battery.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    fn grid(&mut self) -> f32 {
        let r = self.next();
        let m = (r % 1024) as f32 + 1.0;
        if (r >> 63) & 1 == 1 { -m } else { m }
    }
    fn slot(&mut self, k: usize) -> usize {
        (self.next() % k as u64) as usize
    }
}

/// The pre-built rig: every buffer the RAM loop touches is allocated at
/// construction — the measured surface is the kernel + record + unwind
/// path itself (the G4 contract), not harness plumbing.
struct RamRig {
    h: usize,
    kd: usize,
    vd: usize,
    state: Vec<f32>,
    journal: GdnStateJournal,
    k_full: Vec<f32>,
    v_full: Vec<f32>,
    kv_full: Vec<f32>,
    kv_head: Vec<f32>,
    q_head: Vec<f32>,
    output: Vec<f32>,
    delta: Vec<f32>,
    kh: Vec<f32>,
    vh: Vec<f32>,
    betas: Vec<f32>,
    decays: Vec<f32>,
}

impl RamRig {
    fn new(steps: usize) -> Self {
        let (h, kd, vd) = (2usize, 16usize, 8usize);
        Self {
            h,
            kd,
            vd,
            state: vec![1.0f32; h * kd * vd],
            journal: GdnStateJournal::with_capacity(h, kd, vd, steps),
            k_full: vec![0.0f32; h * kd],
            v_full: vec![0.0f32; h * vd],
            kv_full: vec![0.0f32; h * vd],
            kv_head: vec![0.0f32; vd],
            q_head: vec![0.0f32; kd],
            output: vec![0.0f32; vd],
            delta: vec![0.0f32; vd],
            kh: vec![0.0f32; kd],
            vh: vec![0.0f32; vd],
            betas: vec![1.0f32; h],
            decays: vec![1.0f32; h],
        }
    }

    /// One full RAM pass: `steps` writes through one-hot keys + journal
    /// record per step + one unwind, all within pre-reserved capacity.
    fn run(&mut self, steps: usize, seed: u64) {
        let (h, kd, vd) = (self.h, self.kd, self.vd);
        let mut rng = Lcg(seed);
        let rig = self;
        rng.grid_fill(&mut rig.state); // nonzero in-domain init, no alloc

        for t in 0..steps {
            // keep / halve alternation — the product over any window of 2
            // is ≥ 0.25, inside the default conditioning floor.
            let g = if t % 2 == 0 { 1.0 } else { 0.5 };
            for head in 0..h {
                rig.kh.iter_mut().for_each(|x| *x = 0.0);
                rig.kh[rng.slot(kd)] = 1.0;
                for x in rig.vh.iter_mut() {
                    *x = rng.grid();
                }
                let head_state = &mut rig.state[head * kd * vd..(head + 1) * kd * vd];
                gated_deltanet_step_inplace(
                    &rig.q_head,
                    &rig.kh,
                    &rig.vh,
                    head_state,
                    &rig.betas,
                    &[g],
                    1,
                    kd,
                    vd,
                    &mut rig.output,
                    &mut rig.kv_head,
                    &mut rig.delta,
                );
                rig.k_full[head * kd..(head + 1) * kd].copy_from_slice(&rig.kh);
                rig.v_full[head * vd..(head + 1) * vd].copy_from_slice(&rig.vh);
                rig.kv_full[head * vd..(head + 1) * vd].copy_from_slice(&rig.kv_head);
            }
            rig.decays.iter_mut().for_each(|d| *d = g);
            let (k, v, kv, b, d) = (
                &rig.k_full,
                &rig.v_full,
                &rig.kv_full,
                &rig.betas,
                &rig.decays,
            );
            rig.journal.record_step(k, v, kv, b, d);
        }
        // One unwind within the recorded window (alloc-free by law 6).
        let _ = rig.journal.unwind_to(&mut rig.state, steps / 2, 0.25);
    }
}

impl Lcg {
    /// Fill a slice with nonzero grid values (±[1, 1024]) in place.
    fn grid_fill(&mut self, out: &mut [f32]) {
        for x in out.iter_mut() {
            *x = self.grid();
        }
    }
}

#[test]
fn g4_exact_slot_ram_hot_loop_is_alloc_free() {
    // The rig (and its warmup pass) absorb every once-only and harness
    // allocation BEFORE the counter resets.
    let mut rig = RamRig::new(256);
    rig.run(8, 0xDEAD_BEEF);
    rig.journal.clear();
    ALLOCS.store(0, Ordering::Relaxed);

    // Measured pass: the sustained RAM posture (256 writes + records + an
    // unwind) must not allocate at all.
    rig.run(256, 0x1155_CAFE);

    assert_eq!(
        ALLOCS.load(Ordering::Relaxed),
        0,
        "the exact-slot RAM hot loop allocated — G4 violated"
    );
}
