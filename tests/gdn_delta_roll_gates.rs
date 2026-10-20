//! Issue 049 — GDN delta-roll journal gates (the G1 drift pin + gate laws).
//!
//! Runs: `cargo test -p riir-infer-core --features gdn_delta_roll --test
//! gdn_delta_roll_gates -- --nocapture` (the sweep prints its drift table —
//! the Bench-885 record cites it; run the table cells in `--release`).
//!
//! The core claim under test: the journal's rank-1 unwind reconstructs an
//! earlier GDN state to within a PINNED drift of the fresh sequential
//! replay, whenever the per-head decay product over the window stays at or
//! above the pinned floor — and refuses (fail-closed, state untouched)
//! below it. Window sizes are per-regime so each regime carries admitted
//! cells near the gate floor, not just trivial ones.
#![cfg(feature = "gdn_delta_roll")]

use katgpt_core::Rng;
use riir_infer_core::deltanet::delta_roll::{
    ConditioningRefusal, GdnStateJournal, MIN_PRODUCT_EPS_DEFAULT,
};

/// The drift bar: relative max-element error vs the fresh replay, pinned
/// with headroom from the Bench-885 sweep (the table prints the measured
/// values; the bar is the assertion, the table is the evidence).
const MAX_REL_DRIFT: f32 = 1e-3;

/// L2-normalizes `k` per head — the forward's own caller contract
/// (`gated_deltanet_step_inplace`: keys are L2-normalized). Outside this
/// envelope (`‖k‖ > 1` with β > 1/‖k‖²) the rank-1 write is an EXPANSION
/// along k, the recurrence itself is unstable, and no recovery machinery —
/// journal or snapshot — is meaningful over the garbage; measured first-hand
/// (the pre-fix sweep exploded to 4e13 at depth 384 and the record keeps
/// the lesson).
fn normalize_heads(k: &mut [f32], heads: usize, key_dim: usize) {
    for h in 0..heads {
        let kh = &mut k[h * key_dim..(h + 1) * key_dim];
        let n = kh.iter().map(|&x| x * x).sum::<f32>().sqrt().max(1e-12);
        for x in kh.iter_mut() {
            *x /= n;
        }
    }
}

/// One committed step of the forward's own math (`gated_deltanet_step`
/// inlined, scalar per-head decay) — the ground-truth generator AND the
/// replay generator. Returns the updated state and the step's `kv_mem`
/// (the post-decay prediction the journal consumes).
#[allow(clippy::too_many_arguments)]
fn step_forward(
    state: &mut [f32],
    k: &[f32],
    v: &[f32],
    kv_mem: &mut [f32],
    beta: &[f32],
    decay: &[f32],
    heads: usize,
    key_dim: usize,
    val_dim: usize,
) {
    for h in 0..heads {
        let g = decay[h];
        let b = beta[h];
        let k_h = &k[h * key_dim..(h + 1) * key_dim];
        let s_h = &mut state[h * key_dim * val_dim..(h + 1) * key_dim * val_dim];
        // Decay first (the forward scales the whole head block).
        for s in s_h.iter_mut() {
            *s *= g;
        }
        // Predict + delta + rank-1 write.
        for row in 0..val_dim {
            let s_row = &mut s_h[row * key_dim..(row + 1) * key_dim];
            let kv: f32 = s_row.iter().zip(k_h.iter()).map(|(&s, &kc)| s * kc).sum();
            kv_mem[h * val_dim + row] = kv;
            let d = b * (v[h * val_dim + row] - kv);
            for (s, &kc) in s_row.iter_mut().zip(k_h.iter()) {
                *s += d * kc;
            }
        }
    }
}

struct Regime {
    name: &'static str,
    /// Window size — chosen so the regime carries admitted drift cells
    /// near the gate floor (products in the 0.25..1 band), not just
    /// refusals and no-ops.
    window: usize,
    /// Per-token per-head decay sampler.
    decay: fn(&mut Rng) -> f32,
}

const REGIMES: [Regime; 4] = [
    Regime {
        name: "g=0.999",
        window: 512,
        decay: |_| 0.999,
    },
    Regime {
        name: "g=0.99",
        window: 128,
        decay: |_| 0.99,
    },
    Regime {
        name: "g=0.95",
        window: 64,
        decay: |_| 0.95,
    },
    Regime {
        name: "mixed[0.90,1.00]",
        window: 128,
        decay: |rng| 0.90 + 0.10 * rng.uniform(),
    },
];

/// The G1 core: W committed steps from a seeded state → unwind to t* →
/// compare against a FRESH sequential replay of the first t* steps from the
/// same base. Cells whose window decay product dips under the gate floor
/// must REFUSE; admitted cells must sit under the pinned drift bar.
#[test]
fn g1_unwind_matches_fresh_replay_per_regime() {
    const H: usize = 2;
    const K: usize = 128;
    const V: usize = 128;

    for regime in REGIMES {
        let w = regime.window;
        let mut rng = Rng::new(0x0490_4949);
        let base: Vec<f32> = (0..H * K * V).map(|_| rng.normal() * 0.1).collect();

        let mut journal = GdnStateJournal::with_capacity(H, K, V, w);
        let mut state = base.clone();
        let mut kv_mem = vec![0.0f32; H * V];
        let schedules: Vec<(Vec<f32>, Vec<f32>, Vec<f32>)> = (0..w)
            .map(|_| {
                let mut k = (0..H * K).map(|_| rng.normal() * 0.3).collect::<Vec<f32>>();
                normalize_heads(&mut k, H, K);
                (
                    k,
                    (0..H * V).map(|_| rng.normal()).collect::<Vec<f32>>(),
                    (0..H).map(|_| rng.uniform()).collect::<Vec<f32>>(),
                )
            })
            .collect();
        let decays: Vec<Vec<f32>> = (0..w)
            .map(|_| (0..H).map(|_| (regime.decay)(&mut rng)).collect())
            .collect();
        for (t, (k, v, b)) in schedules.iter().enumerate() {
            step_forward(&mut state, k, v, &mut kv_mem, b, &decays[t], H, K, V);
            journal.record_step(k, v, &kv_mem, b, &decays[t]);
        }
        assert_eq!(journal.len(), w);

        // Quarterly targets + the base: every window depth class.
        let targets = [w / 4, w / 2, 3 * w / 4, w];
        for &t_star in targets.iter() {
            let end = state.clone();
            let min_product = (0..H)
                .map(|h| (t_star..w).map(|t| decays[t][h] as f64).product::<f64>())
                .fold(f64::INFINITY, f64::min);
            let mut unwound = end.clone();
            let res = journal.unwind_to(&mut unwound, t_star, MIN_PRODUCT_EPS_DEFAULT);
            if min_product < MIN_PRODUCT_EPS_DEFAULT as f64 {
                let err = res.err().unwrap_or_else(|| {
                    panic!(
                        "{} t*={t_star}: gate must refuse at product {min_product:e}",
                        regime.name
                    )
                });
                assert_eq!(
                    unwound, end,
                    "{} t*={t_star}: refusal must leave the state untouched",
                    regime.name
                );
                println!(
                    "{:>16} W={w:<4} t*={t_star:>4} REFUSED head={} product={:e} (< {})",
                    regime.name, err.head, err.product, err.eps
                );
                continue;
            }
            let steps = res.unwrap_or_else(|e| {
                panic!(
                    "{} t*={t_star}: gate refused a product={min_product:e} window: {e:?}",
                    regime.name
                )
            });
            assert_eq!(steps, w - t_star);

            // Ground truth: fresh replay of the first t* steps.
            let mut truth = base.clone();
            let mut truth_kv = vec![0.0f32; H * V];
            for (t, (k, v, b)) in schedules.iter().take(t_star).enumerate() {
                step_forward(&mut truth, k, v, &mut truth_kv, b, &decays[t], H, K, V);
            }
            let state_scale = truth.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
            let max_err = unwound
                .iter()
                .zip(truth.iter())
                .map(|(&a, &b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            let rel = max_err / state_scale;
            println!(
                "{:>16} W={w:<4} t*={t_star:>4} product={min_product:.3e} max_abs={max_err:.3e} rel={rel:.3e}",
                regime.name
            );
            assert!(
                rel < MAX_REL_DRIFT,
                "{} t*={t_star}: relative replay drift {rel:.3e} >= bar {MAX_REL_DRIFT:e}",
                regime.name
            );
        }
    }
}

/// The gate's depth law: `window_depth_limit` accepts exactly the depths
/// `unwind_to` accepts — the boundary cell unwinds, one deeper refuses.
#[test]
fn conditioning_gate_boundary_matches_depth_limit() {
    const H: usize = 1;
    const K: usize = 16;
    const V: usize = 16;
    const W: usize = 256;
    // 0.97^d crosses 0.25 at d = 46 (0.97^45 ≈ 0.254, 0.97^46 ≈ 0.246).
    let mut rng = Rng::new(0x0d00_0d49);
    let mut journal = GdnStateJournal::with_capacity(H, K, V, W);
    let mut state: Vec<f32> = (0..H * K * V).map(|_| rng.normal() * 0.1).collect();
    let mut kv_mem = vec![0.0f32; H * V];
    for _ in 0..W {
        let mut k: Vec<f32> = (0..K).map(|_| rng.normal() * 0.3).collect();
        normalize_heads(&mut k, H, K);
        let v: Vec<f32> = (0..V).map(|_| rng.normal()).collect();
        let b = [rng.uniform()];
        let g = [0.97f32];
        step_forward(&mut state, &k, &v, &mut kv_mem, &b, &g, H, K, V);
        journal.record_step(&k, &v, &kv_mem, &b, &g);
    }
    let end = state;
    let limit = journal.window_depth_limit(MIN_PRODUCT_EPS_DEFAULT);
    assert!(
        (40..=50).contains(&limit),
        "0.97-decay depth limit sane, got {limit}"
    );

    let mut probe = end.clone();
    let ok = journal.unwind_to(&mut probe, W - limit, MIN_PRODUCT_EPS_DEFAULT);
    assert!(ok.is_ok(), "boundary depth {limit} must unwind: {ok:?}");
    let mut probe = end.clone();
    if limit < W {
        let refused: ConditioningRefusal = journal
            .unwind_to(&mut probe, W - limit - 1, MIN_PRODUCT_EPS_DEFAULT)
            .expect_err("one deeper than the limit must refuse");
        assert_eq!(probe, end, "refusal is fail-closed");
        assert_eq!(refused.token, W - limit - 1);
    }
}

/// The f64 escape hatch isolates unwind-arithmetic drift: at the same f32
/// contexts the f64 unwinder lands strictly closer to the replay than the
/// f32 one (the contexts' own record rounding is common to both).
#[test]
fn f64_unwind_dominates_f32_drift() {
    const H: usize = 2;
    const K: usize = 64;
    const V: usize = 64;
    const W: usize = 128;
    const T_STAR: usize = 32;
    let mut rng = Rng::new(0xF64D_0049);
    let base: Vec<f32> = (0..H * K * V).map(|_| rng.normal() * 0.1).collect();
    let mut journal = GdnStateJournal::with_capacity(H, K, V, W);
    let mut state = base.clone();
    let mut kv_mem = vec![0.0f32; H * V];
    let mut all_k = Vec::new();
    let mut all_v = Vec::new();
    let mut all_b = Vec::new();
    let mut all_g = Vec::new();
    for _ in 0..W {
        let mut k: Vec<f32> = (0..H * K).map(|_| rng.normal() * 0.3).collect();
        normalize_heads(&mut k, H, K);
        let v: Vec<f32> = (0..H * V).map(|_| rng.normal()).collect();
        let b: Vec<f32> = (0..H).map(|_| rng.uniform()).collect();
        let g: Vec<f32> = (0..H).map(|_| 0.99).collect();
        step_forward(&mut state, &k, &v, &mut kv_mem, &b, &g, H, K, V);
        journal.record_step(&k, &v, &kv_mem, &b, &g);
        all_k.push(k);
        all_v.push(v);
        all_b.push(b);
        all_g.push(g);
    }
    // f32 unwind (window product 0.99^96 ≈ 0.38 ≥ floor).
    let mut f32_state = state.clone();
    journal
        .unwind_to(&mut f32_state, T_STAR, MIN_PRODUCT_EPS_DEFAULT)
        .expect("gate admits the 0.99^96 window");
    // f64 unwind of the same end state.
    let mut f64_state: Vec<f64> = state.iter().map(|&x| x as f64).collect();
    journal
        .unwind_to_f64(&mut f64_state, T_STAR)
        .expect("gate admits");
    // Fresh replay truth.
    let mut truth = base.clone();
    let mut truth_kv = vec![0.0f32; H * V];
    for t in 0..T_STAR {
        step_forward(
            &mut truth,
            &all_k[t],
            &all_v[t],
            &mut truth_kv,
            &all_b[t],
            &all_g[t],
            H,
            K,
            V,
        );
    }
    let scale = truth.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
    let err32 = f32_state
        .iter()
        .zip(truth.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let err64 = f64_state
        .iter()
        .zip(truth.iter())
        .map(|(&a, &b)| (a - b as f64).abs() as f32)
        .fold(0.0f32, f32::max);
    println!(
        "f32 unwind rel={:e}  f64 unwind rel={:e}",
        err32 / scale,
        err64 / scale
    );
    assert!(err64 <= err32, "f64 unwinder must not drift more than f32");
}

/// The storage claim: ~64× at K = V = 128 (258 f32/token vs 16 384).
#[test]
fn compression_ratio_is_the_64x_class() {
    let j = GdnStateJournal::with_capacity(8, 128, 128, 1);
    let ratio = j.compression_ratio();
    println!(
        "journal {} B/token vs snapshot {} B/token = {:.1}x",
        j.bytes_per_token(),
        j.snapshot_bytes_per_token(),
        ratio
    );
    assert!(
        (63.0..=64.5).contains(&ratio),
        "ratio {ratio} off the 64x class"
    );
}

/// Law 3 in motion: clear + re-record under a NEW schedule unwinds under
/// the new contexts (the schedule is carried BY the contexts, not the
/// journal shell).
#[test]
fn clear_re_record_rewinds_under_new_schedule() {
    const H: usize = 1;
    const K: usize = 16;
    const V: usize = 16;
    const W: usize = 32;
    let mut rng = Rng::new(0xC13A_0049);
    let mut journal = GdnStateJournal::with_capacity(H, K, V, W);

    let mut run_window = |journal: &mut GdnStateJournal, g: f32| -> (Vec<f32>, Vec<f32>) {
        let base: Vec<f32> = (0..H * K * V).map(|_| rng.normal() * 0.1).collect();
        let mut state = base.clone();
        let mut kv_mem = vec![0.0f32; H * V];
        for _ in 0..W {
            let mut k: Vec<f32> = (0..K).map(|_| rng.normal() * 0.3).collect();
            normalize_heads(&mut k, H, K);
            let v: Vec<f32> = (0..V).map(|_| rng.normal()).collect();
            let b = [rng.uniform()];
            let gg = [g];
            step_forward(&mut state, &k, &v, &mut kv_mem, &b, &gg, H, K, V);
            journal.record_step(&k, &v, &kv_mem, &b, &gg);
        }
        (base, state)
    };

    // Window 1: strong decay — the deep unwind refuses.
    let (_base1, end1) = run_window(&mut journal, 0.90);
    let mut probe1 = end1.clone();
    assert!(
        journal
            .unwind_to(&mut probe1, 0, MIN_PRODUCT_EPS_DEFAULT)
            .is_err(),
        "0.90^32 ≈ 0.034 must refuse"
    );
    assert_eq!(probe1, end1, "refusal is fail-closed");
    journal.clear();
    assert!(journal.is_empty());

    // Window 2 (same shell, new schedule): the deep unwind reconstructs
    // the base to the epsilon class.
    let (base2, end2) = run_window(&mut journal, 0.9999);
    let mut unwound = end2.clone();
    journal
        .unwind_to(&mut unwound, 0, MIN_PRODUCT_EPS_DEFAULT)
        .expect("0.9999^32 ≈ 0.997 admits the full depth");
    let max_err = unwound
        .iter()
        .zip(base2.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let scale = base2.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
    assert!(
        max_err / scale < 1e-4,
        "base reconstruction drift {max_err:.3e}"
    );
}

/// Timing lane (the Bench-885 G2 numbers; NOT a gate): per-token cost of
/// record vs unwind vs the fresh replay it replaces, real geometry,
/// `--release --ignored --nocapture`. Decay 0.9999 keeps the 2048-step
/// window product ≈ 0.815 above the floor so the unwind actually runs.
#[test]
#[ignore = "measurement lane — run in release for the Bench-885 record"]
fn measure_record_unwind_replay_wall() {
    use std::hint::black_box;
    use std::time::Instant;
    const H: usize = 8;
    const K: usize = 128;
    const V: usize = 128;
    const W: usize = 4096;
    let mut rng = Rng::new(0x71C3_0490);
    let base: Vec<f32> = (0..H * K * V).map(|_| rng.normal() * 0.1).collect();
    let mut journal = GdnStateJournal::with_capacity(H, K, V, W);
    let mut state = base.clone();
    let mut kv_mem = vec![0.0f32; H * V];
    let mut all_k = Vec::with_capacity(W);
    let mut all_v = Vec::with_capacity(W);
    let mut all_b = Vec::with_capacity(W);
    let mut all_g = Vec::with_capacity(W);

    let t0 = Instant::now();
    for _ in 0..W {
        let mut k: Vec<f32> = (0..H * K).map(|_| rng.normal() * 0.3).collect();
        normalize_heads(&mut k, H, K);
        let v: Vec<f32> = (0..H * V).map(|_| rng.normal()).collect();
        let b: Vec<f32> = (0..H).map(|_| rng.uniform()).collect();
        let g: Vec<f32> = (0..H).map(|_| 0.9999).collect();
        step_forward(&mut state, &k, &v, &mut kv_mem, &b, &g, H, K, V);
        journal.record_step(&k, &v, &kv_mem, &b, &g);
        all_k.push(k);
        all_v.push(v);
        all_b.push(b);
        all_g.push(g);
    }
    let record_wall = t0.elapsed();

    let t1 = Instant::now();
    let mut probe = state.clone();
    let steps = journal
        .unwind_to(&mut probe, W / 2, MIN_PRODUCT_EPS_DEFAULT)
        .expect("0.9999^2048 ≈ 0.815 admits the half-window");
    let unwind_wall = t1.elapsed();

    let t2 = Instant::now();
    let mut truth = base.clone();
    let mut truth_kv = vec![0.0f32; H * V];
    for t in 0..W / 2 {
        step_forward(
            &mut truth,
            &all_k[t],
            &all_v[t],
            &mut truth_kv,
            &all_b[t],
            &all_g[t],
            H,
            K,
            V,
        );
    }
    let replay_wall = t2.elapsed();

    black_box((probe, truth));
    println!(
        "H={H} K={K} V={V} record+forward {W} tok: {:?} ({:.0} ns/tok) | unwind {steps} tok: {:?} ({:.0} ns/tok) | fresh replay {} tok: {:?} ({:.0} ns/tok) | journal {} B/tok ({:.1}x)",
        record_wall,
        record_wall.as_nanos() as f64 / W as f64,
        unwind_wall,
        unwind_wall.as_nanos() as f64 / steps.max(1) as f64,
        W / 2,
        replay_wall,
        replay_wall.as_nanos() as f64 / (W / 2) as f64,
        journal.bytes_per_token(),
        journal.compression_ratio(),
    );
}
