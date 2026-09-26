//! Probe (reflex issue 020 leftovers): what do the two deferred costs
//! actually weigh?
//!
//! 1. `Metal::new()` — the runtime MSL compile + pipeline builds. Run the
//!    example as several fresh PROCESSES: the first after a shader-cache
//!    miss pays the full compile; later ones hit Apple's on-disk cache.
//! 2. `ops::rope_tables` — rebuilt every forward today, at hd = 64, for
//!    the geometries Bench 050 prices (m 106–895) and both thetas.
//!
//! `cargo run --release -p riir-infer-laya --features laya-riir-metal --example startup_rope_probe`

use std::hint::black_box;
use std::time::Instant;

use riir_infer_laya::laya::riir::ops;

fn best_of_us(reps: usize, mut f: impl FnMut()) -> f64 {
    let mut best = f64::MAX;
    for _ in 0..reps {
        let t = Instant::now();
        f();
        best = best.min(t.elapsed().as_secs_f64() * 1e6);
    }
    best
}

fn main() {
    let t = Instant::now();
    let m = riir_infer_laya::laya::riir::metal::Metal::new().expect("metal device");
    let new_ms = t.elapsed().as_secs_f64() * 1e3;
    black_box(&m);
    println!("metal_new_ms={new_ms:.2}");

    for &seq in &[106usize, 188, 512, 895] {
        // one forward builds BOTH thetas (full + sliding) today
        let us = best_of_us(50, || {
            let a = ops::rope_tables(black_box(seq), 64, black_box(160_000.0));
            let b = ops::rope_tables(black_box(seq), 64, black_box(10_000.0));
            black_box((a, b));
        });
        println!("rope_pair seq={seq} hd=64 best_us={us:.1}");
    }
}
