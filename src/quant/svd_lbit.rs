//! svd_lbit — Issue 036 T1: the LittleBit-derived init-only sub-1-bit PTQ
//! transform (Research 009, arXiv:2506.13771).
//!
//! The paper's closed-form skeleton, evaluated training-free:
//! 1. Seeded deterministic randomized (Halko) truncated SVD — the T0
//!    substrate decision (katgpt-core's `thin_svd_into` is 10⁴× too slow at
//!    gemma2 dims; a fixed-seed `fastrand` sketch keeps everything
//!    deterministic and dependency-free).
//! 2. Dual-SVID init (paper §1.3): `U_sign`/`V_sign` from the truncated
//!    factors (`U′ = U_r·diag(σ_r)`, `V′ = V_r`), magnitude scales from the
//!    rank-1 Perron pairs of `|U′|` and `|V′|`, latent scale `ℓ = ℓᵤ⊙ℓᵥ`.
//! 3. Optional k-stage residual restack (paper §1.4, Prop 2): stage s+1
//!    factorizes `W − Σ_{i≤s} Ŵ_i` at the paths-priced rank.
//! 4. BPW planner pinned against the paper's Appendix D worked examples.
//!
//! # Construction note (the flow's dimension annotations)
//!
//! The issue text annotates the Halko flow `U_r = Q2·Ũ_r (dout×r)` — a slip:
//! `Q2` spans the RIGHT subspace (`Q2 ∈ R^{din×k}`), so `Q2·Ũ_r` is
//! `[din×r]`. The mathematically sound construction implemented here (with
//! `B = Q2ᵀ·Wᵀ`, i.e. `Bᵀ = W·Q2`):
//!
//! - `Y2 = W·Q2` (this is `Bᵀ`; the "recompute Y-side" step),
//! - symmetric Jacobi eigendecomposition of `BBᵀ = Y2ᵀ·Y2` `[k,k]` →
//!   eigenvalues `λ` desc + eigenvector matrix `C`,
//! - `σ_i = √λ_i`; **u**_i (R^dout) `= Y2·C[:,i]/σ_i` (the issue's
//!   `ṽ_i = Bᵀũ_i/σ_i` formula); **v**_i (R^din) `= Q2·C[:,i]`,
//! - which satisfies `W ≈ U·Σ·Vᵀ` exactly as the API requires.
//!
//! # Determinism law
//!
//! Every stage is a pure function of `(input bytes, dims, seed)` with a
//! fixed consumption order: the Gaussian sketch is filled row-major via
//! consecutive Box-Muller pairs (`z0` = cos arm first), MGS walks columns in
//! order, Jacobi sweeps the fixed cyclic `p<q` order, and the rank-1 power
//! iteration starts from the all-ones vector. Same box + same seed →
//! bit-identical output (pinned by the tests). Cross-box byte-identity is
//! NOT claimed here: different SIMD widths reorder float sums — that is the
//! issue's separate T2 gate (quantize-once-commit-bytes discipline).
//!
//! # Approximation disclosure (the T0 basis-sharing decision)
//!
//! ONE `k_max` factorization per tensor (with `k = r_max + 16` oversampling)
//! serves all BPW targets and the plain-SVD baseline by truncation — exact
//! per-r independent SVD is a documented approximation class, disclosed in
//! the bench record. The library entry points (`rand_trunc_svd`,
//! `dual_svid_init`, `dual_svid_init_staged`) always factorize at their own
//! `k = rank + 16`; only the eval bin composes the shared-basis path via
//! `rand_trunc_svd_oversampled` + `dual_svid_from_svd`.
//!
//! Everything here is MEASUREMENT-ONLY (Issue 036 P0 law / lossy-surface
//! law): opt-in behind the `svd_lbit` feature, never a serving path.

use fastrand::Rng;

/// GEMM tile edge (cache blocking for the big products).
const BM: usize = 64;
/// GEMM tile edge (cache blocking for the big products).
const BN: usize = 64;
/// GEMM tile edge (cache blocking for the big products).
const BK: usize = 64;
/// Blocked-transpose tile edge.
const TB: usize = 32;
/// Default Halko oversampling beyond the target rank.
const OVERSAMPLE: usize = 16;
/// Jacobi sweep cap (k ≤ ~1128 here; convergence at 1e-7 lands ≤ ~15 sweeps).
const JACOBI_MAX_SWEEPS: usize = 60;
/// Jacobi convergence: off-diagonal Frobenius vs the initial total.
const JACOBI_TOL: f64 = 1e-7;
/// Rank-1 power-iteration cap / convergence delta (paper-agnostic; fixed).
const POWER_ITERS: usize = 100;
/// Rank-1 power-iteration convergence: ‖x' − x‖ < this.
const POWER_TOL: f32 = 1e-12;
/// Degenerate-norm floor (all-zero inputs).
const DEGENERATE: f32 = 1e-30;
/// MGS relative drop: a row whose post-projection norm² falls below its own
/// pre-projection norm² × this is fp-noise residue (rank deficiency) — zero
/// it rather than normalize (normalizing noise to unit length destroys
/// orthogonality; measured: gram err ~9 on an exact-rank-8 parent without
/// this guard).
const MGS_REL_DROP: f32 = 1e-12;
/// σ guard: singular values below `σ_max · this` are zeroed.
const SIGMA_REL_EPS: f64 = 1e-12;
/// Stage-salt for `dual_svid_init_staged` seed derivation.
const STAGED_SALT: u64 = 0xD0A1_5EED;

// ── BPW planner ─────────────────────────────────────────────────────────────

/// Solve `b = [paths·r·(dout+din) + 32·(dout+din) + 32·r] / (dout·din)` for
/// `r` (paper Appendix D; the `32·(dout+din)` term prices the fp16 `h`/`g`
/// scales, `32·r` the `ℓ` vector, `paths·r·(dout+din)` the ±1 factor bodies).
///
/// Floor, clamped to `1..=min(dout,din)`. Pins: `r=546` at `0.55` bpw on
/// 4096×4096 with `paths=2`, `r=133` at `0.1` on 4096×11008.
pub fn rank_for_bpw(bpw: f64, dout: usize, din: usize, paths: usize) -> usize {
    let dsum = (dout + din) as f64;
    let area = (dout * din) as f64;
    let p = paths.max(1) as f64;
    let r = ((bpw * area - 32.0 * dsum) / (p * dsum + 32.0)).floor();
    let cap = dout.min(din) as f64;
    r.clamp(1.0, cap) as usize
}

/// The BPW formula forward: bits-per-weight of a rank-`r` factorization.
pub fn bpw_for_rank(r: usize, dout: usize, din: usize, paths: usize) -> f64 {
    let dsum = (dout + din) as f64;
    (paths.max(1) as f64 * r as f64 * dsum + 32.0 * dsum + 32.0 * r as f64)
        / (dout * din) as f64
}

/// Rank for the plain truncated-SVD baseline (f16 `U`,`V` factors, no
/// scales): `b = 16·r·(dout+din)/(dout·din)` solved for `r`.
pub fn plain_rank_for_bpw(bpw: f64, dout: usize, din: usize) -> usize {
    let r = (bpw * (dout * din) as f64 / (16.0 * (dout + din) as f64)).floor();
    let cap = dout.min(din) as f64;
    r.clamp(1.0, cap) as usize
}

/// Deterministic 64-bit seed mixer (splitmix64 finalizer over folded
/// fields) — the ONE derivation shared by the library and the eval bin, so
/// per-tensor/per-stage seeds are stable across both.
pub fn mix_seed(seed: u64, a: u64, b: u64, c: u64) -> u64 {
    let mut z = seed
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(a.wrapping_mul(0xBF58_476D_1CE4_E5B9))
        .wrapping_add(b.wrapping_mul(0x94D0_49BB_1331_11EB))
        .wrapping_add(c);
    z ^= z >> 30;
    z = z.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z ^= z >> 27;
    z = z.wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

// ── Deterministic kernels ───────────────────────────────────────────────────

/// `sign(v) ∈ {−1,+1}` with `0 → +1` (the paper's sign convention).
fn sign_f32(v: f32) -> f32 {
    if v < 0.0 {
        -1.0
    } else {
        1.0
    }
}

/// Fill `out` with standard-normal samples: consecutive uniform pairs via
/// Box-Muller, `z0 = R·cos(θ)` written first, then `z1 = R·sin(θ)`; the odd
/// tail drops `z1`. Fixed consumption order = fixed bytes for a fixed seed.
fn gaussian_fill(rng: &mut Rng, out: &mut [f32]) {
    let mut idx = 0;
    while idx < out.len() {
        let u1 = 1.0 - rng.f64(); // (0, 1] — ln-safe
        let u2 = rng.f64();
        let rad = (-2.0 * u1.ln()).sqrt();
        let (sin_t, cos_t) = (core::f64::consts::TAU * u2).sin_cos();
        out[idx] = (rad * cos_t) as f32;
        idx += 1;
        if idx < out.len() {
            out[idx] = (rad * sin_t) as f32;
            idx += 1;
        }
    }
}

/// `c[m,n] += a[m,ka]·b[ka,n]` — all row-major. Caller zeroes `c`.
fn gemm_nn(a: &[f32], b: &[f32], c: &mut [f32], m: usize, ka: usize, n: usize) {
    for i0 in (0..m).step_by(BM) {
        let im = (i0 + BM).min(m);
        for k0 in (0..ka).step_by(BK) {
            let km = (k0 + BK).min(ka);
            for j0 in (0..n).step_by(BN) {
                let jm = (j0 + BN).min(n);
                for i in i0..im {
                    for k in k0..km {
                        let av = a[i * ka + k];
                        let brow = &b[k * n + j0..k * n + jm];
                        let crow = &mut c[i * n + j0..i * n + jm];
                        for (cj, &bv) in crow.iter_mut().zip(brow) {
                            *cj += av * bv;
                        }
                    }
                }
            }
        }
    }
}

/// `c[m,n] += Σ_x a[x,m]·b[x,n]` — both operands row-major with leading `x`.
fn gemm_tn(a: &[f32], b: &[f32], c: &mut [f32], x_len: usize, m: usize, n: usize) {
    for x0 in (0..x_len).step_by(BK) {
        let xm = (x0 + BK).min(x_len);
        for j0 in (0..n).step_by(BN) {
            let jm = (j0 + BN).min(n);
            for i0 in (0..m).step_by(BM) {
                let im = (i0 + BM).min(m);
                for x in x0..xm {
                    let brow = &b[x * n + j0..x * n + jm];
                    for i in i0..im {
                        let av = a[x * m + i];
                        let crow = &mut c[i * n + j0..i * n + jm];
                        for (cj, &bv) in crow.iter_mut().zip(brow) {
                            *cj += av * bv;
                        }
                    }
                }
            }
        }
    }
}

/// `src [rows, cols] → dst [cols, rows]`, blocked.
fn transpose(src: &[f32], dst: &mut [f32], rows: usize, cols: usize) {
    for i0 in (0..rows).step_by(TB) {
        let im = (i0 + TB).min(rows);
        for j0 in (0..cols).step_by(TB) {
            let jm = (j0 + TB).min(cols);
            for i in i0..im {
                for j in j0..jm {
                    dst[j * rows + i] = src[i * cols + j];
                }
            }
        }
    }
}

/// `src [rows, cols] → [cols, rows]` (allocating).
fn transposed(src: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut dst = vec![0.0f32; rows * cols];
    transpose(src, &mut dst, rows, cols);
    dst
}

/// Modified Gram-Schmidt over the ROWS of `q [k, cols]` (contiguous — the
/// transpose of the column layout the algorithm is usually written against).
/// Degenerate rows (‖·‖² < [`DEGENERATE`]) are zeroed: deterministic guard,
/// measure-zero on real inputs.
fn mgs_rows(q: &mut [f32], k: usize, cols: usize) {
    for j in 0..k {
        // Pre-projection norm — the row's own degeneracy reference.
        let mut orig = 0.0f32;
        for x in 0..cols {
            let e = q[j * cols + x];
            orig += e * e;
        }
        for i in 0..j {
            let mut dot = 0.0f32;
            for x in 0..cols {
                dot += q[i * cols + x] * q[j * cols + x];
            }
            for x in 0..cols {
                q[j * cols + x] -= dot * q[i * cols + x];
            }
        }
        let mut nrm = 0.0f32;
        for x in 0..cols {
            let e = q[j * cols + x];
            nrm += e * e;
        }
        if nrm < DEGENERATE || nrm < orig * MGS_REL_DROP {
            for x in 0..cols {
                q[j * cols + x] = 0.0;
            }
        } else {
            let inv = 1.0 / nrm.sqrt();
            for x in 0..cols {
                q[j * cols + x] *= inv;
            }
        }
    }
}

/// Symmetric cyclic Jacobi eigendecomposition of `a [k,k]` (row-major,
/// destroyed). Returns eigenvalues DESC (negatives clamped to 0) and the
/// eigenvector matrix `C [k,k]` whose COLUMN `i` is the eigenvector of the
/// i-th returned eigenvalue. Stable index tiebreak on equal eigenvalues.
fn jacobi_eig(a: &mut [f32], k: usize) -> (Vec<f64>, Vec<f32>) {
    let mut fro = 0.0f64;
    for &v in a.iter() {
        fro += f64::from(v) * f64::from(v);
    }
    let scale = fro.sqrt().max(f64::MIN_POSITIVE);
    let mut cvecs = vec![0.0f32; k * k];
    for i in 0..k {
        cvecs[i * k + i] = 1.0;
    }
    for _sweep in 0..JACOBI_MAX_SWEEPS {
        let mut off = 0.0f64;
        for i in 0..k {
            for j in (i + 1)..k {
                let e = f64::from(a[i * k + j]);
                off += 2.0 * e * e;
            }
        }
        if off.sqrt() <= JACOBI_TOL * scale {
            break;
        }
        for p in 0..k {
            for q in (p + 1)..k {
                let apq = f64::from(a[p * k + q]);
                if apq.abs() <= 1e-30 {
                    continue;
                }
                let app = f64::from(a[p * k + p]);
                let aqq = f64::from(a[q * k + q]);
                let theta = (aqq - app) / (2.0 * apq);
                let t = if theta >= 0.0 {
                    1.0 / (theta + (1.0 + theta * theta).sqrt())
                } else {
                    1.0 / (theta - (1.0 + theta * theta).sqrt())
                };
                let csr = 1.0 / (1.0 + t * t).sqrt();
                let snr = t * csr;
                let (cf, sf) = (csr as f32, snr as f32);
                for j in 0..k {
                    let (apj, aqj) = (a[p * k + j], a[q * k + j]);
                    a[p * k + j] = cf * apj - sf * aqj;
                    a[q * k + j] = sf * apj + cf * aqj;
                }
                for j in 0..k {
                    let (ajp, ajq) = (a[j * k + p], a[j * k + q]);
                    a[j * k + p] = cf * ajp - sf * ajq;
                    a[j * k + q] = sf * ajp + cf * ajq;
                }
                for j in 0..k {
                    let (vjp, vjq) = (cvecs[j * k + p], cvecs[j * k + q]);
                    cvecs[j * k + p] = cf * vjp - sf * vjq;
                    cvecs[j * k + q] = sf * vjp + cf * vjq;
                }
            }
        }
    }
    let mut idx: Vec<usize> = (0..k).collect();
    idx.sort_by(|&x, &y| {
        let lx = f64::from(a[x * k + x]);
        let ly = f64::from(a[y * k + y]);
        ly.partial_cmp(&lx)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(x.cmp(&y))
    });
    let lambda: Vec<f64> = idx
        .iter()
        .map(|&i| f64::from(a[i * k + i]).max(0.0))
        .collect();
    let mut cmat = vec![0.0f32; k * k];
    for (j, &src) in idx.iter().enumerate() {
        for r in 0..k {
            cmat[r * k + j] = cvecs[r * k + src];
        }
    }
    (lambda, cmat)
}

// ── Randomized truncated SVD ────────────────────────────────────────────────

/// A seeded deterministic randomized (Halko) truncated SVD basis: `u`
/// `[dout,k]` (σ NOT folded), `sigma` `[k]` descending (zeroed below the
/// relative ε), `v` `[din,k]`, with `W ≈ u·diag(σ)·vᵀ`. Callers truncate to
/// any `r ≤ k` by taking the leading columns.
pub struct TruncSvd {
    /// Left singular vectors `[dout, k]`, columns in σ-desc order.
    pub u: Vec<f32>,
    /// Singular values `[k]`, descending; near-zero entries are 0.
    pub sigma: Vec<f32>,
    /// Right singular vectors `[din, k]`, columns in σ-desc order.
    pub v: Vec<f32>,
    /// Output rows of the source matrix.
    pub dout: usize,
    /// Input columns of the source matrix.
    pub din: usize,
    /// Basis width actually computed (`rank + oversampling`, capped).
    pub k: usize,
}

/// Seeded Halko truncated SVD at `k = rank + 16` oversampling (capped at
/// `min(dout,din)`): Gaussian sketch → thin MGS QR → q=1 power iteration →
/// small symmetric Jacobi SVD (see the module doc for the construction).
pub fn rand_trunc_svd(w: &[f32], dout: usize, din: usize, rank: usize, seed: u64) -> TruncSvd {
    let k = (rank.max(1) + OVERSAMPLE).min(dout.min(din)).max(1);
    rand_trunc_svd_oversampled(w, dout, din, k, seed)
}

/// The explicit-k form the eval bin drives (the T0 shared-basis path).
pub fn rand_trunc_svd_oversampled(
    w: &[f32],
    dout: usize,
    din: usize,
    k: usize,
    seed: u64,
) -> TruncSvd {
    assert_eq!(w.len(), dout * din, "weight buffer must be [dout, din]");
    let k = k.clamp(1, dout.min(din));
    // 1. Ω sketch [din, k]
    let mut omega = vec![0.0f32; din * k];
    gaussian_fill(&mut Rng::with_seed(seed), &mut omega);
    // 2. Y0 = W·Ω [dout, k]
    let mut y0 = vec![0.0f32; dout * k];
    gemm_nn(w, &omega, &mut y0, dout, din, k);
    // 3. Q1 (MGS over the rows of the transpose = columns of Y0)
    let mut q1t = transposed(&y0, dout, k);
    mgs_rows(&mut q1t, k, dout);
    let q1 = transposed(&q1t, k, dout);
    // 4. Z = Wᵀ·Q1 [din, k]
    let mut z = vec![0.0f32; din * k];
    gemm_tn(w, &q1, &mut z, dout, din, k);
    // 5. Q2 (right-subspace basis after one power iteration)
    let mut q2t = transposed(&z, din, k);
    mgs_rows(&mut q2t, k, din);
    let q2 = transposed(&q2t, k, din);
    // 6. Y2 = W·Q2 [dout, k] — this is Bᵀ for B = Q2ᵀ·Wᵀ
    let mut y2 = vec![0.0f32; dout * k];
    gemm_nn(w, &q2, &mut y2, dout, din, k);
    // 7. M = Y2ᵀ·Y2 = BBᵀ [k, k]
    let mut m = vec![0.0f32; k * k];
    gemm_tn(&y2, &y2, &mut m, dout, k, k);
    // 8. symmetric Jacobi eigendecomposition
    let (lambda, cmat) = jacobi_eig(&mut m, k);
    // 9. σ with the relative guard
    let smax = lambda.first().copied().unwrap_or(0.0).sqrt();
    let eps = (smax * SIGMA_REL_EPS).max(f64::MIN_POSITIVE);
    let mut sigma = vec![0.0f32; k];
    for (si, &lv) in sigma.iter_mut().zip(lambda.iter()) {
        let s = lv.sqrt();
        *si = if s > eps {
            s as f32
        } else {
            0.0
        };
    }
    // 10. u = Y2·C (column-scaled by 1/σ), v = Q2·C
    let mut u = vec![0.0f32; dout * k];
    gemm_nn(&y2, &cmat, &mut u, dout, k, k);
    let mut v = vec![0.0f32; din * k];
    gemm_nn(&q2, &cmat, &mut v, din, k, k);
    for i in 0..k {
        if sigma[i] > 0.0 {
            let inv = 1.0 / sigma[i];
            for r in 0..dout {
                u[r * k + i] *= inv;
            }
        } else {
            for r in 0..dout {
                u[r * k + i] = 0.0;
            }
            for r in 0..din {
                v[r * k + i] = 0.0;
            }
        }
    }
    TruncSvd {
        u,
        sigma,
        v,
        dout,
        din,
        k,
    }
}

/// Dense reconstruction `Ŵ = U_r·Σ_r·V_rᵀ` from a basis (the plain-SVD arm).
pub fn reconstruct_lowrank(svd: &TruncSvd, rank: usize) -> Vec<f32> {
    let r = rank.min(svd.k);
    let mut a = vec![0.0f32; svd.dout * r];
    for i in 0..svd.dout {
        for t in 0..r {
            a[i * r + t] = svd.u[i * svd.k + t] * svd.sigma[t];
        }
    }
    let mut vt = vec![0.0f32; r * svd.din];
    for j in 0..svd.din {
        for t in 0..r {
            vt[t * svd.din + j] = svd.v[j * svd.k + t];
        }
    }
    let mut out = vec![0.0f32; svd.dout * svd.din];
    gemm_nn(&a, &vt, &mut out, svd.dout, r, svd.din);
    out
}

// ── Dual-SVID init ──────────────────────────────────────────────────────────

/// Dominant nonnegative rank-1 pair of `a [rows, cols]` (both nonneg) by
/// power iteration from the all-ones seed (Perron): returns `(h, x)` with
/// `x` unit-norm and `h = a·x` (σ₁ absorbed), so `a ≈ h·xᵀ`.
fn rank1_nonneg(a: &[f32], rows: usize, cols: usize) -> (Vec<f32>, Vec<f32>) {
    let mut x = vec![1.0f32 / (cols as f32).sqrt(); cols];
    for _ in 0..POWER_ITERS {
        // y = A·x
        let mut y = vec![0.0f32; rows];
        for (i, yi) in y.iter_mut().enumerate() {
            let row = &a[i * cols..(i + 1) * cols];
            let mut s = 0.0f32;
            for (&av, &xv) in row.iter().zip(x.iter()) {
                s += av * xv;
            }
            *yi = s;
        }
        let mut ny = 0.0f32;
        for &v in y.iter() {
            ny += v * v;
        }
        if ny < DEGENERATE {
            break;
        }
        let inv = 1.0 / ny.sqrt();
        for v in y.iter_mut() {
            *v *= inv;
        }
        // x' = Aᵀ·y
        let mut xn = vec![0.0f32; cols];
        for i in 0..rows {
            let yi = y[i];
            let row = &a[i * cols..(i + 1) * cols];
            for (xj, &av) in xn.iter_mut().zip(row.iter()) {
                *xj += yi * av;
            }
        }
        let mut nx = 0.0f32;
        for &v in xn.iter() {
            nx += v * v;
        }
        if nx < DEGENERATE {
            break;
        }
        let inv = 1.0 / nx.sqrt();
        let mut delta = 0.0f32;
        for (xnv, &xv) in xn.iter_mut().zip(x.iter()) {
            *xnv *= inv;
            let d = *xnv - xv;
            delta += d * d;
        }
        let converged = delta.sqrt() < POWER_TOL;
        x.copy_from_slice(&xn);
        if converged {
            break;
        }
    }
    // absorb σ₁: h = A·x, unnormalized
    let mut h = vec![0.0f32; rows];
    for (i, hi) in h.iter_mut().enumerate() {
        let row = &a[i * cols..(i + 1) * cols];
        let mut s = 0.0f32;
        for (&av, &xv) in row.iter().zip(x.iter()) {
            s += av * xv;
        }
        *hi = s;
    }
    (h, x)
}

/// The Dual-SVID factorization (paper §1.3): the ±1 sandwich
/// `Ŵ = diag(h)·U_sign·diag(ℓ)·V_signᵀ·diag(g)` initialized from a truncated
/// SVD basis. `paths` is the pricing arity recorded on the factors (1 =
/// single path, 2 = a primary+residual pair) — it only affects the reported
/// achieved BPW, never the math.
pub fn dual_svid_from_svd(svd: &TruncSvd, rank: usize, paths: usize) -> SvdLbitFactors {
    let r = rank.min(svd.k).max(1);
    let (dout, din) = (svd.dout, svd.din);
    // |U′| = |U_r|·diag(σ_r) (σ folded into the U side per §1.3)
    let mut up_abs = vec![0.0f32; dout * r];
    for i in 0..dout {
        for t in 0..r {
            up_abs[i * r + t] = svd.u[i * svd.k + t].abs() * svd.sigma[t];
        }
    }
    let mut vp_abs = vec![0.0f32; din * r];
    for j in 0..din {
        for t in 0..r {
            vp_abs[j * r + t] = svd.v[j * svd.k + t].abs();
        }
    }
    // signs (0 → +1; zeroed σ columns contribute +1 columns deterministically)
    let mut u_sign = vec![1.0f32; dout * r];
    for i in 0..dout {
        for t in 0..r {
            u_sign[i * r + t] = sign_f32(svd.u[i * svd.k + t] * svd.sigma[t]);
        }
    }
    let mut v_sign = vec![1.0f32; din * r];
    for j in 0..din {
        for t in 0..r {
            v_sign[j * r + t] = sign_f32(svd.v[j * svd.k + t]);
        }
    }
    let (h, lu) = rank1_nonneg(&up_abs, dout, r);
    let (g, lv) = rank1_nonneg(&vp_abs, din, r);
    let ell: Vec<f32> = lu.iter().zip(lv.iter()).map(|(&a, &b)| a * b).collect();
    SvdLbitFactors {
        h,
        g,
        ell,
        u_sign,
        v_sign,
        dout,
        din,
        rank: r,
        paths: paths.max(1),
    }
}

/// One-shot Dual-SVID init (own basis at `k = rank + 16`, `paths = 1`).
pub fn dual_svid_init(w: &[f32], dout: usize, din: usize, rank: usize, seed: u64) -> SvdLbitFactors {
    let svd = rand_trunc_svd(w, dout, din, rank, seed);
    dual_svid_from_svd(&svd, rank, 1)
}

/// Staged (residual-restack) init, paper §1.4: `stages` parallel paths, each
/// at `rank_for_bpw(bpw, …, paths = stages)`, stage `s+1` factorizing the
/// running residual. Never mutates the input. `Ŵ = Σ_s reconstruct(f_s)`.
pub fn dual_svid_init_staged(
    w: &[f32],
    dout: usize,
    din: usize,
    bpw: f64,
    stages: usize,
    seed: u64,
) -> Vec<SvdLbitFactors> {
    let paths = stages.max(1);
    let r = rank_for_bpw(bpw, dout, din, paths);
    let mut out = Vec::with_capacity(paths);
    let mut residual: Vec<f32> = w.to_vec();
    for s in 0..paths {
        let sseed = mix_seed(seed, s as u64 + 1, STAGED_SALT, 0);
        let svd = rand_trunc_svd(&residual, dout, din, r, sseed);
        let f = dual_svid_from_svd(&svd, r, paths);
        if s + 1 < paths {
            let ws = reconstruct(&f);
            for (rr, &wv) in residual.iter_mut().zip(ws.iter()) {
                *rr -= wv;
            }
        }
        out.push(f);
    }
    out
}

/// The Dual-SVID factor set: `Ŵ = diag(h)·U_sign·diag(ℓ)·V_signᵀ·diag(g)`.
#[derive(Clone, Debug)]
pub struct SvdLbitFactors {
    /// Row scales `[dout]` (σ₁ of `|U′|` absorbed — magnitudes, ≥ 0).
    pub h: Vec<f32>,
    /// Column scales `[din]` (σ₁ of `|V′|` absorbed — magnitudes, ≥ 0).
    pub g: Vec<f32>,
    /// Latent scales `[rank]` = `ℓᵤ ⊙ ℓᵥ`.
    pub ell: Vec<f32>,
    /// `[dout × rank]` ±1 entries (stored f32 for now; packing is T4's lane).
    pub u_sign: Vec<f32>,
    /// `[din × rank]` ±1 entries.
    pub v_sign: Vec<f32>,
    /// Output dimension.
    pub dout: usize,
    /// Input dimension.
    pub din: usize,
    /// Latent rank actually stored (`≤` the requested rank).
    pub rank: usize,
    /// Pricing arity (1 = single path; 2 = primary+residual pair).
    pub paths: usize,
}

impl SvdLbitFactors {
    /// Achieved BPW under the Appendix-D accounting.
    pub fn achieved_bpw(&self) -> f64 {
        bpw_for_rank(self.rank, self.dout, self.din, self.paths)
    }
}

/// Dense reconstruction of one factor set (the eval path — a serving kernel
/// would run the sandwich skinny-GEMV form and never materialize this).
pub fn reconstruct(f: &SvdLbitFactors) -> Vec<f32> {
    let (dout, din, r) = (f.dout, f.din, f.rank);
    // A = U_sign·diag(ℓ) [dout, r]
    let mut a = vec![0.0f32; dout * r];
    for i in 0..dout {
        for t in 0..r {
            a[i * r + t] = f.u_sign[i * r + t] * f.ell[t];
        }
    }
    // V_signᵀ [r, din]
    let mut vt = vec![0.0f32; r * din];
    for j in 0..din {
        for t in 0..r {
            vt[t * din + j] = f.v_sign[j * r + t];
        }
    }
    let mut out = vec![0.0f32; dout * din];
    gemm_nn(&a, &vt, &mut out, dout, r, din);
    // sandwich scales on the way out
    for i in 0..dout {
        let hi = f.h[i];
        let orow = &mut out[i * din..(i + 1) * din];
        for (o, &gv) in orow.iter_mut().zip(f.g.iter()) {
            *o *= hi * gv;
        }
    }
    out
}

/// Dense reconstruction of a staged set: `Σ_s reconstruct(f_s)`.
pub fn reconstruct_staged(fs: &[SvdLbitFactors]) -> Vec<f32> {
    match fs.split_first() {
        Some((first, rest)) => {
            let mut acc = reconstruct(first);
            for f in rest {
                debug_assert_eq!(
                    (f.dout, f.din),
                    (first.dout, first.din),
                    "stage dims must match"
                );
                for (a, &w) in acc.iter_mut().zip(reconstruct(f).iter()) {
                    *a += w;
                }
            }
            acc
        }
        None => Vec::new(),
    }
}

/// Naive RTN-binary comparator: per-row `α_i = mean|W_i,:|`, `Ŵ = α·sign(W)`
/// (the training-free floor the ladder must beat). Also the `rtn` arm.
pub fn rtn_binary_rows(w: &[f32], dout: usize, din: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; dout * din];
    for i in 0..dout {
        let row = &w[i * din..(i + 1) * din];
        let alpha = row.iter().map(|v| v.abs()).sum::<f32>() / din as f32;
        let orow = &mut out[i * din..(i + 1) * din];
        for (o, &v) in orow.iter_mut().zip(row) {
            *o = alpha * sign_f32(v);
        }
    }
    out
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn gauss_vec(rng: &mut Rng, n: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; n];
        gaussian_fill(rng, &mut v);
        v
    }

    /// Low-rank core with per-direction geometric σ decay + Gaussian noise —
    /// the decaying-spectrum (realistic weight) shape. `nonneg = true` builds
    /// the core from |gaussian| factors (a Perron-dominant parent — the regime
    /// where the paper's rank-1 magnitude init is near-exact, and the shape
    /// the refine test pins). The decay/nonneg knobs matter for the Dual-SVID
    /// refinement claim: the sandwich's rank-1 magnitude factorization is
    /// near-exact when the top spectral pair is nonneg-dominant (real learned
    /// weights); on FLAT SIGNED spectra the V-side magnitude fluctuation is
    /// irreducible and single-path init does NOT refine naive row-binarization
    /// (staging does — see the restack test). `decay = 1.0, nonneg = false`
    /// reproduces the plain flat signed low-rank-plus-noise parent.
    fn mat_from_rng(
        seed: u64,
        dout: usize,
        din: usize,
        rank: usize,
        noise: f32,
        decay: f32,
        nonneg: bool,
    ) -> Vec<f32> {
        let mut rng = Rng::with_seed(seed);
        let mut a = gauss_vec(&mut rng, dout * rank); // [dout, rank]
        let mut b = gauss_vec(&mut rng, rank * din); // [rank, din]
        let g = gauss_vec(&mut rng, dout * din);
        if nonneg {
            a = a.iter().map(|v| v.abs()).collect();
            b = b.iter().map(|v| v.abs()).collect();
        }
        let mut w = vec![0.0f32; dout * din];
        for t in 0..rank {
            let s = decay.powi(t as i32);
            for i in 0..dout {
                for j in 0..din {
                    w[i * din + j] += s * a[i * rank + t] * b[t * din + j];
                }
            }
        }
        for idx in 0..dout * din {
            w[idx] += noise * g[idx];
        }
        w
    }

    fn rel_frob(a: &[f32], b: &[f32]) -> f64 {
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        for (&x, &y) in a.iter().zip(b) {
            let d = f64::from(x - y);
            num += d * d;
            den += f64::from(x) * f64::from(x);
        }
        (num / den.max(f64::MIN_POSITIVE)).sqrt()
    }

    #[test]
    fn planner_pins_appendix_d() {
        assert_eq!(rank_for_bpw(0.55, 4096, 4096, 2), 546);
        assert_eq!(rank_for_bpw(0.1, 4096, 11008, 2), 133);
        let b = bpw_for_rank(546, 4096, 4096, 2);
        assert!((b - 0.5495).abs() < 1e-3, "bpw_for_rank(546,…) = {b}");
        // paths=1 roughly doubles the factor budget at the same bpw (the
        // +32 denominator term keeps it just under 2× 546).
        assert_eq!(rank_for_bpw(0.55, 4096, 4096, 1), 1090);
    }

    #[test]
    fn planner_roundtrip_within_1pct() {
        for &b in &[0.1, 0.3, 0.55, 1.0] {
            for (dout, din) in [(4096usize, 4096usize), (4096, 11008), (2304, 9216)] {
                for paths in [1usize, 2usize] {
                    let r = rank_for_bpw(b, dout, din, paths);
                    let back = bpw_for_rank(r, dout, din, paths);
                    assert!(back <= b, "bpw {b} dims ({dout},{din}) paths {paths} → r {r} → {back}");
                    assert!(
                        (b - back) / b <= 0.01,
                        "bpw {b} dims ({dout},{din}) paths {paths} → r {r} → {back}"
                    );
                }
            }
        }
    }

    #[test]
    fn plain_rank_planner_matches_formula() {
        for &b in &[0.55f64, 1.0] {
            for (dout, din) in [(2304usize, 2304usize), (9216, 2304), (2048, 2304)] {
                let r = plain_rank_for_bpw(b, dout, din);
                let back = 16.0 * r as f64 * (dout + din) as f64 / (dout * din) as f64;
                assert!(back <= b + 1e-9, "b {b} dims ({dout},{din}) r {r} → {back}");
                assert!(
                    (b - back) / b <= 0.02,
                    "b {b} dims ({dout},{din}) r {r} → {back}"
                );
            }
        }
    }

    #[test]
    fn rand_trunc_svd_exact_rank_8() {
        let (dout, din, rank) = (128usize, 96usize, 8usize);
        let w = mat_from_rng(0x036_0001, dout, din, rank, 0.0, 1.0, false);
        let svd = rand_trunc_svd(&w, dout, din, rank, 0xA11CE);
        assert_eq!(svd.k, (rank + OVERSAMPLE).min(dout.min(din)));
        for s in 1..svd.k {
            assert!(svd.sigma[s - 1] >= svd.sigma[s], "sigma must be descending");
        }
        let what = reconstruct_lowrank(&svd, rank);
        let err = rel_frob(&w, &what);
        assert!(err < 1e-2, "exact-rank-8 reconstruction err = {err}");
    }

    #[test]
    fn svd_and_dual_are_bit_deterministic_and_seed_sensitive() {
        let (dout, din) = (96usize, 80usize);
        let w = mat_from_rng(0x036_0002, dout, din, 6, 0.2, 1.0, false);
        // SVD: same seed → bit-identical; different seed → different
        let a = rand_trunc_svd(&w, dout, din, 12, 0xC0FFEE);
        let b = rand_trunc_svd(&w, dout, din, 12, 0xC0FFEE);
        for (name, (xs, ys)) in [("u", (&a.u, &b.u)), ("sigma", (&a.sigma, &b.sigma)), ("v", (&a.v, &b.v))] {
            assert_eq!(xs.len(), ys.len());
            for (x, y) in xs.iter().zip(ys.iter()) {
                assert_eq!(x.to_bits(), y.to_bits(), "{name} not bit-identical");
            }
        }
        let c = rand_trunc_svd(&w, dout, din, 12, 0xD1FE);
        assert!(a.u.iter().zip(&c.u).any(|(x, y)| x.to_bits() != y.to_bits()));
        // Dual path: reconstruct bytes bit-identical / seed-sensitive
        let r1 = reconstruct(&dual_svid_init(&w, dout, din, 12, 0xC0FFEE));
        let r2 = reconstruct(&dual_svid_init(&w, dout, din, 12, 0xC0FFEE));
        for (x, y) in r1.iter().zip(&r2) {
            assert_eq!(x.to_bits(), y.to_bits());
        }
        let r3 = reconstruct(&dual_svid_init(&w, dout, din, 12, 0xD1FE));
        assert!(r1.iter().zip(&r3).any(|(x, y)| x.to_bits() != y.to_bits()));
    }

    #[test]
    fn rank1_nonneg_recovers_dominant_pair() {
        let mut rng = Rng::with_seed(0x036_0003);
        let av: Vec<f32> = gauss_vec(&mut rng, 40).iter().map(|v| v.abs()).collect();
        let bv: Vec<f32> = gauss_vec(&mut rng, 24).iter().map(|v| v.abs()).collect();
        let mut m = vec![0.0f32; 40 * 24];
        for i in 0..40 {
            for j in 0..24 {
                m[i * 24 + j] = av[i] * bv[j];
            }
        }
        let (h, x) = rank1_nonneg(&m, 40, 24);
        let mut rec = vec![0.0f32; 40 * 24];
        for i in 0..40 {
            for j in 0..24 {
                rec[i * 24 + j] = h[i] * x[j];
            }
        }
        assert!(rel_frob(&m, &rec) < 1e-3);
    }

    #[test]
    fn dual_svid_refines_naive_row_binarization() {
        // Perron-dominant parent (nonneg core + σ decay + noise): the regime
        // where the paper's rank-1 magnitude init is near-exact and the
        // sandwich STRICTLY refines naive row-binarization. On flat SIGNED
        // parents the V-side magnitude fluctuation is irreducible and the
        // single-path init does not refine naive (measured: 0.62 vs 0.61 at
        // decay 0.1) — staging recovers it (the restack test); that is the
        // init-only reality this issue measures, recorded here so the
        // fixture choice is not mistaken for cherry-picking.
        let (dout, din, r) = (160usize, 128usize, 32usize);
        let w = mat_from_rng(0x036_0004, dout, din, 8, 0.25, 0.55, true);
        let f = dual_svid_init(&w, dout, din, r, 0x036_0004);
        let dual_err = rel_frob(&w, &reconstruct(&f));
        let naive_err = rel_frob(&w, &rtn_binary_rows(&w, dout, din));
        println!("dual {dual_err:.4} vs naive {naive_err:.4}");
        assert!(dual_err < naive_err, "dual {dual_err} vs naive {naive_err}");
    }

    #[test]
    fn restack_two_stage_within_5pct_at_matched_bpw() {
        let (dout, din) = (256usize, 192usize);
        let w = mat_from_rng(0x036_0005, dout, din, 16, 0.25, 1.0, false);
        let bpw = 1.0f64;
        let one = dual_svid_init_staged(&w, dout, din, bpw, 1, 0x036_0005);
        let two = dual_svid_init_staged(&w, dout, din, bpw, 2, 0x036_0005);
        let e1 = rel_frob(&w, &reconstruct_staged(&one));
        let e2 = rel_frob(&w, &reconstruct_staged(&two));
        println!(
            "restack @bpw {bpw}: 1-stage err {e1:.4} (r={}), 2-stage err {e2:.4} (r={}×2)",
            one[0].rank, two[0].rank
        );
        assert!(e2 <= e1 * 1.05, "2-stage {e2} vs bound {}", e1 * 1.05);
    }

    #[test]
    fn staged_shapes_and_budget_consistent() {
        let (dout, din) = (96usize, 64usize);
        let w = mat_from_rng(0x036_0006, dout, din, 6, 0.3, 1.0, false);
        for stages in [1usize, 2usize, 3usize] {
            let fs = dual_svid_init_staged(&w, dout, din, 1.2, stages, 0x036_0006);
            assert_eq!(fs.len(), stages);
            for f in &fs {
                assert_eq!(f.dout, dout);
                assert_eq!(f.din, din);
                assert_eq!(f.u_sign.len(), dout * f.rank);
                assert_eq!(f.v_sign.len(), din * f.rank);
                assert_eq!(f.h.len(), dout);
                assert_eq!(f.g.len(), din);
                assert_eq!(f.ell.len(), f.rank);
                assert!(f.achieved_bpw() <= 1.2, "achieved {} > budget 1.2", f.achieved_bpw());
            }
        }
    }

    #[test]
    fn rtn_binary_rows_shape_and_scale() {
        let (dout, din) = (12usize, 16usize);
        let w = mat_from_rng(0x036_0007, dout, din, 3, 0.4, 1.0, false);
        let out = rtn_binary_rows(&w, dout, din);
        assert_eq!(out.len(), dout * din);
        for i in 0..dout {
            let row = &w[i * din..(i + 1) * din];
            let alpha = row.iter().map(|v| v.abs()).sum::<f32>() / din as f32;
            for j in 0..din {
                let expect = alpha * sign_f32(row[j]);
                assert!((out[i * din + j] - expect).abs() < 1e-6);
            }
        }
    }
}
