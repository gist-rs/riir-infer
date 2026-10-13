//! Issue 046 — the spectral class of the Bonsai GDN transition, as a spec test
//! against the REAL decode kernel (`deltanet::forward::gated_deltanet_step_inplace`).
//!
//! The house recurrence (`src/deltanet/forward.rs`, the Qwen3.5 reference form):
//!
//! ```text
//! S ← exp(g)·S + k̃ ⊗ β(v − (exp(g)·S)ᵀk̃)
//! ```
//!
//! has the homogeneous transition `T = g·(I − β·k̃k̃ᵀ)` — the kernel decays the
//! state IN PLACE first, then reads `kv` from the decayed row, then writes the
//! rank-1 correction, so the (decay → read → write) order is exactly the
//! reference's `(exp(g)·S)ᵀk̃` post-decay read. The spectral class is the
//! Gated-DeltaNet row of arXiv:2610.13341 App. A.4:
//!
//! ```text
//! eig(T) = { g(1−β)  along k̃,
//!            g        with multiplicity d−1 }
//! ```
//!
//! NOT DeltaNet's `{1−β, 0}` (App. A.3, no decay gate), and NOT the
//! additive-write bound of §C.3 (`‖u‖/(1−g)` — linear attention / phase-free
//! Mamba). The delta-rule write is a keyed erase along k̃, which is WHY the
//! growth class this issue originally feared does not exist on this lane.
//!
//! The transition is RECOVERED from the kernel by basis probing (v = 0, one
//! key-basis column at a time), so the assertions never re-derive T from the
//! closed form: a transcription change in the kernel's operation order (kv
//! computed pre-decay, or the rank-1 write transposed — Issue 594's class)
//! moves the recovered spectrum and reds this file.
//!
//! Continuous-time link: with γ = −ln g and b = −ln(1−β),
//! `T ≡ expm(−γ·I − b·k̃k̃ᵀ)` entrywise (I commutes with the rank-1
//! projection), so the frozen generator's App-A.4 pair {−(γ+b), −γ} integrates
//! to exactly the discrete class above. Asserted entrywise to f32 tolerance.
//!
//! Boundedness half (the corrected premise): along the key direction the
//! scalar recurrence `c ← g(1−β)·c + β·v` has the fixed point
//! `c* = β·v/(1−g+gβ) ≤ v` for EVERY g ≤ 1, β ∈ (0,1] — pinned through the
//! real kernel at the aligned (worst-case) excitation, including the near-1
//! decay heads that motivated the original additive-write fear.
//!
//! ```bash
//! cargo test --features deltanet_inference --test issue046_gdn_spectral_class_spec -- --nocapture
//! ```
//!
//! Report-only probe note (the issue's deliverable 2, closed spec-test-only):
//! the aligned steady state IS the negative control — it bounds the per-head
//! state gain at β/(1−g+gβ) ≤ 1 by construction, so a fleet-replay
//! ‖S‖/max‖v‖ tap has no alarm semantics left to earn (the issue's Non-goals
//! forbid a growth-alarm threshold; the growth class is refuted below).

#![cfg(feature = "deltanet_inference")]

use riir_infer_core::deltanet::forward::gated_deltanet_step_inplace;

/// Deterministic splitmix64 uniform in `[-amp, amp]` — the house `fill` shape
/// (`tests/gdn_step_parallel_goat.rs`), so fixtures stay reproducible.
fn fill(seed: u64, n: usize, amp: f32) -> Vec<f32> {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = x;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            (((z >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0) * amp
        })
        .collect()
}

/// L2-normalize (the kernel's caller contract for `k`).
fn unit_k(seed: u64, d: usize) -> Vec<f32> {
    let k = fill(seed, d, 1.0);
    let n = k
        .iter()
        .map(|x| (*x as f64) * (*x as f64))
        .sum::<f64>()
        .sqrt();
    k.iter().map(|x| x / (n as f32)).collect()
}

/// One decode step through the REAL kernel (single head, square d×d state).
fn kernel_step(k: &[f32], v: &[f32], state: &mut [f32], beta: f32, g: f32, d: usize) {
    let q = vec![0.0f32; d];
    let (mut out, mut kv, mut dl) = (vec![0.0f32; d], vec![0.0f32; d], vec![0.0f32; d]);
    gated_deltanet_step_inplace(
        &q,
        k,
        v,
        state,
        &[beta],
        &[g],
        1,
        d,
        d,
        &mut out,
        &mut kv,
        &mut dl,
    );
}

/// Recover the homogeneous transition `T` from the kernel by basis probing:
/// feed `S = e_0 ⊗ e_c` (row 0 = key-basis c) with v = 0; the step maps
/// `S[0,:] ← e_c·T`, i.e. state[0..d] after the call is row c of T.
fn extract_transition(k: &[f32], beta: f32, g: f32, d: usize) -> Vec<f64> {
    let v = vec![0.0f32; d];
    let mut t = vec![0.0f64; d * d];
    for (c, tc) in t.chunks_exact_mut(d).enumerate() {
        let mut state = vec![0.0f32; d * d];
        state[c] = 1.0;
        kernel_step(k, &v, &mut state, beta, g, d);
        for (tcj, &x) in tc.iter_mut().zip(state.iter()) {
            *tcj = x as f64;
        }
    }
    t
}

/// Classic cyclic Jacobi eigendecomposition of a symmetric row-major matrix
/// (f64; the f32 kernel's rounding is the quantity under test, so the linear
/// algebra runs in f64). Returns (unsorted diagonal, eigenvector matrix V with
/// A ≈ V·diag·Vᵀ).
fn jacobi_eig(mut a: Vec<f64>, n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut v = vec![0.0f64; n * n];
    for (i, vi) in v.chunks_exact_mut(n).enumerate() {
        vi[i] = 1.0;
    }
    for _ in 0..100 {
        let mut off = 0.0f64;
        for p in 0..n {
            for q in p + 1..n {
                off += a[p * n + q] * a[p * n + q];
            }
        }
        if off < 1e-28 {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                let apq = a[p * n + q];
                if apq.abs() <= 1e-20 {
                    continue;
                }
                let theta = 0.5 * (2.0 * apq).atan2(a[q * n + q] - a[p * n + p]);
                let (c, s) = (theta.cos(), theta.sin());
                for j in 0..n {
                    let (apj, aqj) = (a[p * n + j], a[q * n + j]);
                    a[p * n + j] = c * apj - s * aqj;
                    a[q * n + j] = s * apj + c * aqj;
                }
                for i in 0..n {
                    let (aip, aiq) = (a[i * n + p], a[i * n + q]);
                    a[i * n + p] = c * aip - s * aiq;
                    a[i * n + q] = s * aip + c * aiq;
                }
                for i in 0..n {
                    let (vip, viq) = (v[i * n + p], v[i * n + q]);
                    v[i * n + p] = c * vip - s * viq;
                    v[i * n + q] = s * vip + c * viq;
                }
            }
        }
    }
    let diag = (0..n).map(|i| a[i * n + i]).collect();
    (diag, v)
}

/// `expm` of a symmetric matrix via its eigendecomposition (permutation- and
/// sign-convention-invariant, so Jacobi's unsorted output is fine).
fn expm_symmetric(a: Vec<f64>, n: usize) -> Vec<f64> {
    let (eig, v) = jacobi_eig(a, n);
    let mut e = vec![0.0f64; n * n];
    for (i, ei) in e.chunks_exact_mut(n).enumerate() {
        for (j, eij) in ei.iter_mut().enumerate() {
            *eij = v[i * n + j] * eig[j].exp();
        }
    }
    let mut out = vec![0.0f64; n * n];
    for (i, oi) in out.chunks_exact_mut(n).enumerate() {
        for (j, oij) in oi.iter_mut().enumerate() {
            *oij = (0..n).map(|m| e[i * n + m] * v[j * n + m]).sum();
        }
    }
    out
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

#[test]
fn transition_recovered_from_the_kernel_is_symmetric() {
    for (g, beta) in [(0.9f32, 0.5f32), (0.9999, 0.9), (0.99, 0.1)] {
        let k = unit_k(7, 16);
        let t = extract_transition(&k, beta, g, 16);
        let mut asym = 0.0f64;
        for c in 0..16 {
            for j in 0..16 {
                asym = asym.max((t[c * 16 + j] - t[j * 16 + c]).abs());
            }
        }
        assert!(
            asym <= 1e-6,
            "T not symmetric at g={g} beta={beta}: max|T−Tᵀ|={asym:e}"
        );
    }
}

#[test]
fn spectral_class_is_the_gated_deltanet_row() {
    // β = 1 included: the along-k̃ eigenvalue degenerates to exactly 0.
    let grid: &[(f32, f32)] = &[
        (0.9, 1.0),
        (0.9, 0.5),
        (0.9, 0.1),
        (0.99, 0.9),
        (0.99, 0.5),
        (0.99, 0.1),
        (0.9999, 0.9),
        (0.9999, 0.5),
        (0.9999, 0.1),
    ];
    for &(g, beta) in grid {
        let k = unit_k(11, 16);
        assert_class(&k, g, beta, 16);
    }
    // Canonical Bonsai geometry (128×128 state) at the motivating corner.
    let k = unit_k(13, 128);
    assert_class(&k, 0.9999, 0.5, 128);
}

fn assert_class(k: &[f32], g: f32, beta: f32, d: usize) {
    let t = extract_transition(k, beta, g, d);
    let mut eig = jacobi_eig(t, d).0;
    eig.sort_by(|x, y| x.partial_cmp(y).expect("finite eigenvalues"));
    let mut expected = vec![g as f64 * (1.0 - beta as f64)];
    expected.extend(std::iter::repeat_n(g as f64, d - 1));
    expected.sort_by(|x, y| x.partial_cmp(y).expect("finite expectation"));
    let tol = if d > 16 { 3e-5 } else { 1e-5 };
    for (i, (lam, exp)) in eig.iter().zip(&expected).enumerate() {
        assert!(
            (lam - exp).abs() <= tol,
            "g={g} beta={beta} d={d}: eig[{i}]={lam:e} expected {exp:e} (sorted class {{g(1−β), g×(d−1)}})"
        );
    }
    // The class must be the Gated-DeltaNet row, not DeltaNet's {1−β, 0}:
    // sanity-pin the multiplicities explicitly (guards a future edit that
    // swaps in the wrong closed form on BOTH sides of the comparison).
    let distinct_g = eig.iter().filter(|x| (**x - g as f64).abs() <= tol).count();
    assert_eq!(
        distinct_g,
        d - 1,
        "g={g} beta={beta} d={d}: multiplicity of eigenvalue g"
    );
    assert!(
        (eig[0] - g as f64 * (1.0 - beta as f64)).abs() <= tol,
        "g={g} beta={beta} d={d}: along-k̃ eigenvalue must be g(1−β)"
    );
}

#[test]
fn transition_is_the_matrix_exponential_of_the_frozen_generator() {
    // β = 1 excluded: b = −ln(1−β) diverges (the projection limit).
    let grid: &[(f32, f32)] = &[
        (0.9, 0.5),
        (0.9, 0.1),
        (0.99, 0.9),
        (0.99, 0.5),
        (0.9999, 0.5),
        (0.9999, 0.1),
    ];
    for &(g, beta) in grid {
        let k = unit_k(17, 16);
        assert_expm(&k, g, beta, 16);
    }
    let k = unit_k(19, 128);
    assert_expm(&k, 0.9999, 0.5, 128);
}

fn assert_expm(k: &[f32], g: f32, beta: f32, d: usize) {
    let t = extract_transition(k, beta, g, d);
    // A = −γ·I − b·k̃k̃ᵀ built from the SAME f32 k̃ the kernel consumed
    // (upcast), so the comparison isolates kernel arithmetic, not the f32
    // normalization rounding of k̃ itself.
    let (gamma, b) = (-(g as f64).ln(), -(1.0 - beta as f64).ln());
    let mut a = vec![0.0f64; d * d];
    for (j, aj) in a.chunks_exact_mut(d).enumerate() {
        aj[j] = -gamma;
        for (c, ajc) in aj.iter_mut().enumerate() {
            *ajc -= b * (k[j] as f64) * (k[c] as f64);
        }
    }
    let e = expm_symmetric(a, d);
    let diff = max_abs_diff(&t, &e);
    let tol = if d > 16 { 3e-5 } else { 1e-5 };
    assert!(
        diff <= tol,
        "g={g} beta={beta} d={d}: max|T_kernel − expm(−γI − b·k̃k̃ᵀ)| = {diff:e} > {tol:e}"
    );
}

#[test]
fn aligned_steady_state_gain_is_bounded_by_one_for_every_gate_pair() {
    // The corrected premise, through the real kernel: constant v, fixed k̃ —
    // the worst-case (aligned) excitation. Row r converges to
    // c*_r·k̃ with c*_r = β·v[r]/(1−g+gβ), so ‖S*‖_F = (β/(1−g+gβ))·‖v‖ ≤ ‖v‖.
    let grid: &[(f32, f32)] = &[
        (0.9, 1.0),
        (0.9, 0.5),
        (0.9, 0.01),
        (0.99, 0.5),
        (0.99, 0.1),
        (0.9999, 0.5),
        (0.9999, 0.1),
        (0.9999, 0.01),
    ];
    for &(g, beta) in grid {
        let k = unit_k(23, 16);
        assert_steady_state(&k, g, beta, 16);
    }
    // Canonical geometry at the near-1-decay corner that motivated the issue.
    let k = unit_k(29, 128);
    assert_steady_state(&k, 0.9999, 0.1, 128);
}

fn assert_steady_state(k: &[f32], g: f32, beta: f32, d: usize) {
    // Alternating ±1 values: unit scale, no sign cancellation with k̃.
    let v: Vec<f32> = (0..d)
        .map(|r| if r % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    let gain = beta as f64 / (1.0 - g as f64 + g as f64 * beta as f64);
    // Error contraction rate along k̃ is g(1−β); 12 e-folds reach ~6e-6 rel.
    let rate = g as f64 * (1.0 - beta as f64);
    let steps = ((12.0 / -rate.ln()).ceil() as usize).clamp(1, 200_000);
    let mut state = vec![0.0f32; d * d];
    for _ in 0..steps {
        kernel_step(k, &v, &mut state, beta, g, d);
    }
    // Per-row coefficient along k̃ and orthogonal residual.
    let mut fro = 0.0f64;
    for (r, row) in state.chunks_exact(d).enumerate() {
        let c_r: f64 = (0..d).map(|j| (row[j] as f64) * (k[j] as f64)).sum();
        let target = gain * (v[r] as f64);
        assert!(
            (c_r - target).abs() <= 1e-4 * target.abs().max(1e-6),
            "g={g} beta={beta} d={d}: row {r} coefficient {c_r:e} != closed form {target:e}"
        );
        for j in 0..d {
            let resid = (row[j] as f64) - c_r * (k[j] as f64);
            assert!(
                resid.abs() <= 1e-4,
                "g={g} beta={beta} d={d}: row {r} leaves the k̃ direction ({resid:e})"
            );
            fro += (row[j] as f64) * (row[j] as f64);
        }
    }
    let v_norm = (d as f64).sqrt();
    let ratio = fro.sqrt() / v_norm;
    assert!(
        (ratio - gain).abs() <= 1e-4,
        "g={g} beta={beta} d={d}: measured ‖S*‖_F/‖v‖ = {ratio:.6} != closed-form gain {gain:.6}"
    );
    assert!(
        ratio <= 1.0 + 1e-4,
        "g={g} beta={beta} d={d}: aligned steady-state ratio {ratio:.6} exceeds 1 — the growth class this issue refuted"
    );
    println!(
        "issue046 steady-state: g={g} beta={beta} d={d} steps={steps} gain={gain:.6} measured={ratio:.6}"
    );
}
