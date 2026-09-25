//! Issue 020 T11 — the split-K decision per GEMM SHAPE (measurement only):
//! for every encoder projection geometry (n, k) and a row count m, time
//! the unsliced dispatch vs forced split-K, PAIRED (alternating order per
//! round, median of per-round ratios), `REPS` back-to-back GEMMs per timed
//! sample so the per-sync cost is amortized.
//!
//! ```sh
//! cargo test --release --features laya-riir-metal --test metal_splitk_shape_sweep \
//!     -- --ignored --nocapture
//! ```
#![cfg(all(target_os = "macos", feature = "laya-riir-metal"))]

use riir_infer_laya::laya::riir::backend::Backend;
use riir_infer_laya::laya::riir::metal::{Metal, SplitRule};

const ROUNDS: usize = 15;
const REPS: usize = 20;

fn vec_of(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((s >> 8) as f32) / 8_388_608.0 - 1.0
        })
        .collect()
}

fn time_reps(
    b: &Metal,
    a: &[f32],
    m: usize,
    k: usize,
    w: &[f32],
    n: usize,
    outs: &mut [Vec<f32>],
) -> f64 {
    b.begin_pass();
    let t = std::time::Instant::now();
    for o in outs.iter_mut() {
        b.matmul_w(a, m, k, w, n, o);
    }
    let mut sink = vec![0f32; m * n];
    b.download_into(outs.last().expect("reps"), &mut sink);
    std::hint::black_box(&sink);
    t.elapsed().as_secs_f64() * 1e6 / REPS as f64
}

fn med(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

#[test]
#[ignore = "measurement-only shape sweep (issue 020 T11) — run with --ignored --nocapture"]
fn t11_splitk_shape_sweep() {
    // Two arms: unsplit vs FORCED split (every call over two slices).
    let off = Metal::with_split_rule(SplitRule {
        on: false,
        ..SplitRule::DEFAULT
    })
    .expect("metal off");
    let force = SplitRule {
        on: true,
        max_row_tiles: u32::MAX,
        ..SplitRule::DEFAULT
    };
    let nar = Metal::with_split_rule(force).expect("metal forced split");
    let arms = [&off, &nar];
    let shapes: &[(&str, usize, usize)] = &[
        ("attn_out", 1024, 1024),
        ("mlp_down", 1024, 2624),
        ("qkv", 3072, 1024),
        ("mlp_up", 5248, 1024),
    ];
    let ms = [24usize, 46, 54, 64, 80, 106, 128, 160, 188, 256, 317, 512];
    println!("per-GEMM µs, median of {ROUNDS} rotated rounds · ratios vs unsplit (wins/rounds)");
    for &(name, n, k) in shapes {
        let w = vec_of(n * k, 7);
        for &m in &ms {
            let a = vec_of(m * k, 11 + m as u32);
            let mut outs: Vec<Vec<Vec<f32>>> = (0..2)
                .map(|_| (0..REPS).map(|_| vec![0f32; m * n]).collect())
                .collect();
            for (i, arm) in arms.iter().enumerate() {
                time_reps(arm, &a, m, k, &w, n, &mut outs[i]);
                time_reps(arm, &a, m, k, &w, n, &mut outs[i]);
            }
            let mut t: Vec<Vec<f64>> = vec![Vec::new(); 2];
            for round in 0..ROUNDS {
                let mut sample = [0f64; 2];
                for j in 0..2 {
                    let i = (round + j) % 2;
                    sample[i] = time_reps(arms[i], &a, m, k, &w, n, &mut outs[i]);
                }
                for (ti, si) in t.iter_mut().zip(sample) {
                    ti.push(si);
                }
            }
            let ratio = |i: usize| -> (f64, usize) {
                let r: Vec<f64> = t[i].iter().zip(&t[0]).map(|(x, y)| x / y).collect();
                let wins = r.iter().filter(|v| **v < 1.0).count();
                (med(r), wins)
            };
            let (rn, wn) = ratio(1);
            let rule = SplitRule::DEFAULT.splits(m as u32, n as u32, k as u32);
            println!(
                "{name:<9} n {n:>4} k {k:>4} m {m:>3}: unsplit {:7.1} · split {rn:.3} ({wn:>2}/{ROUNDS}) · rule {}",
                med(t[0].clone()),
                if rule { "SPLIT" } else { "-" }
            );
        }
    }
}
