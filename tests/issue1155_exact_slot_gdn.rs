//! Issue 1155 — the exact slot-addressable GDN state PoC (the Qwenputer
//! identity on OUR kernels): [`gated_deltanet_step_inplace`] used as exact,
//! addressable RAM.
//!
//! Premise (riir-infer Research 018, Qwenputer distill): a one-hot key
//! addresses ONE key COLUMN of the state (our layout is `S[val_row,
//! key_col]` — Qwenputer's "row" wording is transposed for us). With
//! gate `= 1`, write-β `= 1`, and payloads on the EXACT GRID (bounded
//! integers with every intermediate `|a|, |v|, |v−a| ≤ 2^24` exactly
//! representable) plus finite-only state, the delta-form write
//! `S[r,j] + b·(v − S[r,j])` reproduces `v` BIT-EXACTLY — exactness is a
//! value-domain contract, NOT delta-rule algebra (verdict round 1:
//! `a = 1e8, v = 1 → 0.0` is the cancellation counterexample).
//!
//! Batteries:
//! - **G1a** — in-domain: addressed cells round-trip bit-exactly, every
//!   non-addressed cell is bit-unchanged, over a shape sweep including the
//!   real Bonsai geometry (8 heads × 128 × 128) and named grid-edge
//!   payloads (±2^23, ±2^24, 0).
//! - **G1b** — negative control, MUST FAIL: arbitrary-f32 payloads break
//!   it (the 1e8→0.0 cancellation pair as a named case; the overflow row
//!   poisoning `inf·0 = NaN` as the finite-only-contract counterexample).
//!   G1b passing with zero mismatches is a harness bug, not a success.
//! - **Decay selectors** — keep (`g=1`) is bit-stable; halve (`g=0.5`) is
//!   an exact exponent decrement down to `2^-149`, where `2^-149·0.5`
//!   ties-to-even into `0.0` and odd-mantissa subnormals round to even
//!   (`3·2^-149 → 2^-148`) — "an exponent bound, not a mantissa one";
//!   clear (`g=0`) destroys exactly (sign of zero preserved: `x·0 = ±0`
//!   by sign of x).
//! - **Sign-of-zero footnote** — a cell holding `-0.0` flips to `+0.0`
//!   under a positive-delta step's `δ·k[c] = +0` addend. Numerically
//!   `−0 == +0`; Qwenputer's integer validators have ONE zero — recorded
//!   as the identity's one bit-level IEEE exception, asserted to HAPPEN.
//! - **Journal hypothesis** (opt-in `gdn_delta_roll`) — an in-domain
//!   pinned-slot region journaled + unwound via `GdnStateJournal` under
//!   keep/halve restores bit-exactly (law 2's "algebraically exact, NOT
//!   bit-exact" refuted IN-DOMAIN; out-of-domain stays law-2 drift).
//!   Clear is out of scope — journal law 1 refuses `g=0` windows, and
//!   the fail-closed refusal is asserted.
//!
//! G3: this file adds tests only — zero production source change, the
//! default posture stays byte-identical. G4 (zero-alloc hot loop) lives
//! in its OWN binary `issue1155_exact_slot_g4_alloc` (the counting
//! allocator is process-global). Determinism: assertions are per-cell
//! bit compares over a seeded LCG — no reduction folds cross runs
//! (katgpt-rs Issue 940 compliance).
//!
//! Runs: `cargo test --test issue1155_exact_slot_gdn` (G1a/G1b/decay),
//! add `--features gdn_delta_roll` for the journal battery.
#![cfg(feature = "deltanet_inference")]

use riir_infer_core::deltanet::forward::gated_deltanet_step_inplace;

/// Deterministic LCG (the plain Knuth constants) — the battery is fully
/// reproducible run to run.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    /// Nonzero grid integer in ±[1, 4096] — every `|v−a|` against a cell
    /// from the same range stays ≤ 2^13, far inside the 2^24 grid.
    fn grid(&mut self) -> f32 {
        let r = self.next();
        let m = (r % 4096) as f32 + 1.0;
        if (r >> 63) & 1 == 1 { -m } else { m }
    }
    fn slot(&mut self, k: usize) -> usize {
        (self.next() % k as u64) as usize
    }
}

/// One-hot key over `key_dim`: `k[j] = 1.0`, the rest `+0.0` (the IEEE
/// sign of the zeros is load-bearing — see the footnote test).
fn one_hot(key_dim: usize, j: usize) -> Vec<f32> {
    let mut k = vec![0.0f32; key_dim];
    k[j] = 1.0;
    k
}

/// The REAL kernel, one call for all heads (heads are independent; the
/// `kv_mem`/`delta` scratches are per-head and their final content is the
/// last head's — irrelevant to these batteries).
#[allow(clippy::too_many_arguments)]
fn ram_step(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    state: &mut [f32],
    decay: f32,
    heads: usize,
    key_dim: usize,
    val_dim: usize,
    output: &mut [f32],
    kv_mem: &mut [f32],
    delta: &mut [f32],
) {
    let betas = vec![1.0f32; heads];
    let decays = vec![decay; heads];
    gated_deltanet_step_inplace(
        q, k, v, state, &betas, &decays, heads, key_dim, val_dim, output, kv_mem, delta,
    );
}

fn bits(s: &[f32]) -> Vec<u32> {
    s.iter().map(|x| x.to_bits()).collect()
}

// ---------------------------------------------------------------------------
// G1a — in-domain slots are bit-exact memory cells
// ---------------------------------------------------------------------------

#[test]
fn g1a_in_domain_slots_are_bit_exact_memory_cells() {
    // (heads, key_dim, val_dim) — the last entry is the real Bonsai GDN
    // geometry (Issue 594: key_dim == val_dim == 128 on every real
    // checkpoint, which is WHY the transpose defect hid for so long).
    let shapes: &[(usize, usize, usize)] = &[(1, 16, 8), (2, 64, 16), (8, 128, 128)];

    for &(h, kd, vd) in shapes {
        let mut rng = Lcg(0x1155_0001);
        // Initial "unwritten memory": nonzero grid integers (the strictest
        // form of the claim — no zero cells, no subnormals, all in-domain).
        let mut state: Vec<f32> = (0..h * kd * vd).map(|_| rng.grid()).collect();
        let initial = bits(&state);
        let mut written = vec![vec![false; kd]; h];

        // RAM semantics: ~2 writes per slot, a fresh one-hot key + payload
        // vector per step, keep selector throughout.
        let mut output = vec![0.0f32; h * vd];
        let mut kv_mem = vec![0.0f32; vd];
        let mut delta = vec![0.0f32; vd];

        for _ in 0..2 * kd {
            // Each head addresses its OWN slot — the keys are full-width
            // [h·kd], one one-hot per head (the kernel slices k per head).
            let slots: Vec<usize> = (0..h).map(|_| rng.slot(kd)).collect();
            let mut k = vec![0.0f32; h * kd];
            let mut q = vec![0.0f32; h * kd];
            for (head, &j) in slots.iter().enumerate() {
                k[head * kd + j] = 1.0;
                q[head * kd + j] = 1.0; // readout through the same slot
            }
            // Head h writes its own payload scalar into val row r of its
            // column — the RAM cell is the whole column, one scalar per row.
            let v: Vec<f32> = (0..h * vd).map(|_| rng.grid()).collect();
            let before = bits(&state);
            ram_step(
                &q,
                &k,
                &v,
                &mut state,
                1.0,
                h,
                kd,
                vd,
                &mut output,
                &mut kv_mem,
                &mut delta,
            );
            let after = bits(&state);

            for head in 0..h {
                let j = slots[head];
                for r in 0..vd {
                    // Addressed cell: bit-exactly the payload.
                    assert_eq!(
                        after[head * kd * vd + r * kd + j],
                        v[head * vd + r].to_bits(),
                        "G1a [{h}x{kd}x{vd}] addressed cell (head {head}, row {r}, col {j}) \
                         must round-trip bit-exactly"
                    );
                    // Non-addressed cells: bit-unchanged.
                    for c in 0..kd {
                        if c == j {
                            continue;
                        }
                        assert_eq!(
                            after[head * kd * vd + r * kd + c],
                            before[head * kd * vd + r * kd + c],
                            "G1a [{h}x{kd}x{vd}] non-addressed cell (head {head}, row {r}, \
                             col {c}) moved"
                        );
                    }
                }
                written[head][slots[head]] = true;
            }
        }
        // Unwritten slots still hold their initial bits.
        let final_bits = bits(&state);
        for head in 0..h {
            for r in 0..vd {
                for c in 0..kd {
                    if !written[head][c] {
                        assert_eq!(
                            final_bits[head * kd * vd + r * kd + c],
                            initial[head * kd * vd + r * kd + c],
                            "unwritten slot (head {head}, row {r}, col {c}) moved"
                        );
                    }
                }
            }
        }
    }
}

/// Grid-EDGE payloads: the corners of the 2^24 contract, plus the readout
/// leg for power-of-four key dims (scale = 1/√K a power of two, so
/// `out[r] = v·scale` is exact too).
#[test]
fn g1a_grid_edge_payloads_round_trip() {
    let (h, kd, vd) = (1usize, 8usize, 4usize);
    let j = 3;
    // Edge (prior-cell a, payload v) pairs ON the grid: |a|,|v|,|v−a| ≤ 2^24
    // (the ±2^24 corners pair only with 0; the ±2^23 corners pair with each
    // other).
    let edges: &[(f32, f32)] = &[
        (2f32.powi(23), -2f32.powi(23)),
        (-2f32.powi(23), 2f32.powi(23)),
        (2f32.powi(24), 0.0),
        (-2f32.powi(24), 0.0),
        (1.0, 0.0),
        (-1.0, 0.0),
        (4096.0, -4096.0),
    ];
    for &(a, v) in edges {
        let mut state = vec![1.0f32; h * kd * vd];
        for r in 0..vd {
            state[r * kd + j] = a;
        }
        let k = one_hot(kd, j);
        let q = one_hot(kd, j);
        let vv = vec![v; vd];
        let mut output = vec![0.0f32; vd];
        let mut kv_mem = vec![0.0f32; vd];
        let mut delta = vec![0.0f32; vd];
        ram_step(
            &q,
            &k,
            &vv,
            &mut state,
            1.0,
            h,
            kd,
            vd,
            &mut output,
            &mut kv_mem,
            &mut delta,
        );
        for r in 0..vd {
            assert_eq!(
                state[r * kd + j].to_bits(),
                v.to_bits(),
                "G1a edge (a={a}, v={v}): addressed cell must be bit-exact"
            );
        }
    }
    // Readout leg: K ∈ {16, 64} → scale ∈ {0.25, 0.125}, both powers of
    // two, so the readout is v·scale with ONE exact exponent shift.
    for (kdim, scale) in [(16usize, 0.25f32), (64usize, 0.125f32)] {
        let vd = 4;
        let mut state = vec![1.0f32; kdim * vd];
        let j = kdim - 1;
        let v = 2048.0f32;
        for r in 0..vd {
            state[r * kdim + j] = v;
        }
        let k = one_hot(kdim, j);
        let q = one_hot(kdim, j);
        let vv = vec![v; vd];
        let mut output = vec![0.0f32; vd];
        let mut kv_mem = vec![0.0f32; vd];
        let mut delta = vec![0.0f32; vd];
        ram_step(
            &q,
            &k,
            &vv,
            &mut state,
            1.0,
            1,
            kdim,
            vd,
            &mut output,
            &mut kv_mem,
            &mut delta,
        );
        for &out in output.iter().take(vd) {
            assert_eq!(
                out.to_bits(),
                (v * scale).to_bits(),
                "G1a readout at K={kdim} must be bit-exact (v·scale, scale a power of two)"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// G1b — negative control: arbitrary f32 payloads MUST break exactness
// ---------------------------------------------------------------------------

#[test]
fn g1b_arbitrary_f32_payloads_break_exactness() {
    let (h, kd, vd) = (1usize, 8usize, 4usize);
    let j = 5;
    let k = one_hot(kd, j);
    let q = one_hot(kd, j);
    let mut output = vec![0.0f32; vd];
    let mut kv_mem = vec![0.0f32; vd];
    let mut delta = vec![0.0f32; vd];

    // Named case 1 — the cancellation pair from the verdict: a = 1e8,
    // v = 1.0. fl(1 − 1e8) rounds to −1e8, and 1e8 + (−1e8) = 0: the cell
    // reads 0.0, NOT 1.0.
    {
        let mut state = vec![1.0f32; kd * vd];
        for r in 0..vd {
            state[r * kd + j] = 1e8;
        }
        let vv = vec![1.0f32; vd];
        ram_step(
            &q,
            &k,
            &vv,
            &mut state,
            1.0,
            h,
            kd,
            vd,
            &mut output,
            &mut kv_mem,
            &mut delta,
        );
        for r in 0..vd {
            let cell = state[r * kd + j];
            assert_ne!(
                cell.to_bits(),
                1.0f32.to_bits(),
                "G1b cancellation pair must NOT round-trip (harness bug if it does)"
            );
            assert_eq!(cell, 0.0, "the cancellation pair collapses to exactly 0.0");
        }
    }

    // Named case 2 — the finite-only contract: overflow payloads make
    // delta = inf; the outer product writes inf into the addressed cell and
    // inf·0 = NaN into EVERY other column of the row.
    {
        let mut state = vec![1.0f32; kd * vd];
        for r in 0..vd {
            state[r * kd + j] = -3.4e38;
        }
        let vv = vec![3.4e38f32; vd];
        ram_step(
            &q,
            &k,
            &vv,
            &mut state,
            1.0,
            h,
            kd,
            vd,
            &mut output,
            &mut kv_mem,
            &mut delta,
        );
        for r in 0..vd {
            assert_eq!(
                state[r * kd + j],
                f32::INFINITY,
                "addressed cell overflows to inf"
            );
            let nan_cols = (0..kd)
                .filter(|&c| c != j && state[r * kd + c].is_nan())
                .count();
            assert_eq!(
                nan_cols,
                kd - 1,
                "every non-addressed column of an overflowed row must go NaN (inf·0)"
            );
        }
    }

    // Random arbitrary-f32 battery: payloads across magnitudes — mismatches
    // MUST occur. G1b passing with zero mismatches is a harness bug.
    let mut rng = Lcg(0x1155_000B);
    let mut mismatches = 0u32;
    for _ in 0..64 {
        let mut state: Vec<f32> = (0..kd * vd)
            .map(|_| {
                let r = rng.next();
                // Exponents across ±30, random mantissa — well off-grid.
                let exp = ((r % 61) as i32) - 30;
                let mant = 1.0 + ((r >> 8) % 1024) as f32 / 1024.0;
                mant * 10f32.powi(exp)
            })
            .collect();
        let jj = rng.slot(kd);
        let kk = one_hot(kd, jj);
        let qq = one_hot(kd, jj);
        let vv: Vec<f32> = (0..vd).map(|_| rng.grid()).collect();
        let finite_v = vv.iter().all(|x| x.is_finite());
        ram_step(
            &qq,
            &kk,
            &vv,
            &mut state,
            1.0,
            h,
            kd,
            vd,
            &mut output,
            &mut kv_mem,
            &mut delta,
        );
        if finite_v {
            for r in 0..vd {
                if state[r * kd + jj].to_bits() != vv[r].to_bits() {
                    mismatches += 1;
                }
            }
        }
    }
    assert!(
        mismatches > 0,
        "G1b found ZERO mismatches over arbitrary f32 payloads — the harness is wrong, \
         not the identity"
    );
}

// ---------------------------------------------------------------------------
// Decay selectors — keep / halve / clear
// ---------------------------------------------------------------------------

#[test]
fn decay_keep_halve_clear_semantics() {
    let (h, kd, vd) = (1usize, 8usize, 4usize);
    let j = 2;
    let k = one_hot(kd, j);
    let q = one_hot(kd, j);
    let mut output = vec![0.0f32; vd];
    let mut kv_mem = vec![0.0f32; vd];
    let mut delta = vec![0.0f32; vd];

    // KEEP (g=1): the whole state is bit-stable. v is PER ROW: delta[r] =
    // v[r] − S[r,j], so the no-op posture sets v[r] := that row's own
    // addressed cell (the decay is identity, kv[r] reads back exactly that).
    {
        let mut state: Vec<f32> = (0..kd * vd).map(|i| (i % 7) as f32 + 1.0).collect();
        let before = bits(&state);
        let vv: Vec<f32> = (0..vd).map(|r| state[r * kd + j]).collect();
        ram_step(
            &q,
            &k,
            &vv,
            &mut state,
            1.0,
            h,
            kd,
            vd,
            &mut output,
            &mut kv_mem,
            &mut delta,
        );
        assert_eq!(bits(&state), before, "keep must be bit-stable");
    }

    // HALVE (g=0.5): an exact exponent decrement per step — powers of two
    // halve bit-exactly from 2^-1 down to 2^-149; at 2^-149 the next halve
    // is an exact tie between 0 and 2^-149 and ties-to-even picks 0.0.
    {
        let mut state = vec![1.0f32; kd * vd];
        for t in 1..=150i32 {
            let expected = if t <= 149 { 2f32.powi(-t) } else { 0.0 };
            // v := the value the cells hold AFTER this step's decay, so the
            // write cancels (delta = 0) and only the decay moves state.
            let vv = vec![expected; vd];
            ram_step(
                &q,
                &k,
                &vv,
                &mut state,
                0.5,
                h,
                kd,
                vd,
                &mut output,
                &mut kv_mem,
                &mut delta,
            );
            for r in 0..vd {
                assert_eq!(
                    state[r * kd + j].to_bits(),
                    expected.to_bits(),
                    "halve ladder t={t}: cell must be {expected} (2^-{t} exact, then 0.0 \
                     at the 2^-150 tie)"
                );
            }
        }
    }

    // HALVE, odd-mantissa subnormal: 3·2^-149 decays toward 1.5·2^-149, an
    // exact tie between 2^-149 (odd mantissa) and 2^-148 (even) —
    // ties-to-even rounds UP to 2^-148. The issue's "exponent bound, not a
    // mantissa one", demonstrated. (v := 2^-148 keeps the write cancelled —
    // kv reads back the TIE-ROUNDED value, so delta = 0.)
    {
        let mut state = vec![2f32.powi(-149); kd * vd]; // start from k=1
        state[j] = 3.0 * 2f32.powi(-149); // the odd-mantissa cell (addressed)
        let vv = vec![2f32.powi(-148); vd];
        ram_step(
            &q,
            &k,
            &vv,
            &mut state,
            0.5,
            h,
            kd,
            vd,
            &mut output,
            &mut kv_mem,
            &mut delta,
        );
        assert_eq!(
            state[j].to_bits(),
            2f32.powi(-148).to_bits(),
            "odd-mantissa subnormal must tie-to-even UP to 2^-148 (not 1.5·2^-149)"
        );
    }

    // CLEAR (g=0): destruction — the state lands on +0 UNIFORMLY. x·0 does
    // produce ±0 by the sign of x, but delta is always +0 here (v=+0 minus
    // a zero kv is +0) and (+0)·k[c] = +0 for EVERY column, so the write's
    // addend rewrites every −0 to +0 in the same step: sign(x) is not
    // observable post-step. "Zeroes the state exactly" holds; the sign is
    // a +0.
    {
        let mut state: Vec<f32> = (0..kd * vd)
            .map(|i| {
                if i % 2 == 0 {
                    -(i as f32) - 1.0
                } else {
                    i as f32 + 1.0
                }
            })
            .collect();
        let vv = vec![0.0f32; vd]; // v irrelevant: kv = 0, delta = +0
        ram_step(
            &q,
            &k,
            &vv,
            &mut state,
            0.0,
            h,
            kd,
            vd,
            &mut output,
            &mut kv_mem,
            &mut delta,
        );
        for (i, cell) in state.iter().enumerate() {
            assert_eq!(*cell, 0.0, "clear: cell {i} must be zero-valued");
            assert_eq!(
                cell.to_bits(),
                0u32,
                "clear: every cell lands on +0 (the delta=+0 addend rewrites x·0's −0)"
            );
        }
    }
}

/// The identity's one bit-level IEEE exception, asserted to HAPPEN: a cell
/// holding `-0.0` flips to `+0.0` under a positive-delta step (`δ·k[c] =
/// +0` addend; `(−0) + (+0) = +0` under round-to-nearest). Numerically a
/// non-event (`−0 == +0`), and Qwenputer's integer validators have ONE
/// zero — recorded so nobody re-derives this as a contradiction.
#[test]
fn sign_of_zero_is_the_one_bit_level_exception() {
    let (h, kd, vd) = (1usize, 8usize, 4usize);
    let j = 1;
    let mut state = vec![0.0f32; kd * vd];
    // The footnote cell: a NEGATIVE zero in the addressed row, off-column.
    let footnote = (j + 3) % kd;
    state[footnote] = -0.0;
    // A positive-delta write at column j (delta = v − S[j] > 0).
    state[j] = -4.0;
    let k = one_hot(kd, j);
    let q = one_hot(kd, j);
    let vv = vec![8.0f32; vd]; // delta = 12 > 0 → the addend is +0
    let mut output = vec![0.0f32; vd];
    let mut kv_mem = vec![0.0f32; vd];
    let mut delta = vec![0.0f32; vd];
    ram_step(
        &q,
        &k,
        &vv,
        &mut state,
        1.0,
        h,
        kd,
        vd,
        &mut output,
        &mut kv_mem,
        &mut delta,
    );
    assert_eq!(
        state[footnote].to_bits(),
        0u32,
        "−0 must flip to +0 under a positive delta"
    );
    assert_eq!(state[footnote], 0.0, "…and numerically it is still zero");
}

// ---------------------------------------------------------------------------
// Journal interplay — the pre-registered hypothesis (opt-in feature)
// ---------------------------------------------------------------------------

#[cfg(feature = "gdn_delta_roll")]
mod journal_hypothesis {
    use super::Lcg;
    use riir_infer_core::deltanet::delta_roll::{GdnStateJournal, MIN_PRODUCT_EPS_DEFAULT};
    use riir_infer_core::deltanet::forward::gated_deltanet_step_inplace;

    /// The REAL kernel driven one head at a time (`n_v_heads = 1` runs the
    /// same head-loop body the full call runs) — this is what yields the
    /// per-head post-decay predictions `record_step` needs (the forward's
    /// own `kv_mem` scratch is `[val_dim]`, reused per head).
    #[allow(clippy::too_many_arguments)]
    fn step_one_head(
        k: &[f32],
        v: &[f32],
        state: &mut [f32],
        beta: f32,
        decay: f32,
        kd: usize,
        vd: usize,
        kv_out: &mut [f32],
    ) {
        let q = vec![0.0f32; kd]; // readout unused by the hypothesis
        let mut output = vec![0.0f32; vd];
        let mut delta = vec![0.0f32; vd];
        gated_deltanet_step_inplace(
            &q,
            k,
            v,
            state,
            &[beta],
            &[decay],
            1,
            kd,
            vd,
            &mut output,
            kv_out,
            &mut delta,
        );
    }

    fn bits(s: &[f32]) -> Vec<u32> {
        s.iter().map(|x| x.to_bits()).collect()
    }

    /// HYPOTHESIS: an in-domain pinned-slot region journaled + unwound via
    /// `GdnStateJournal` under keep/halve restores BIT-EXACTLY — journal
    /// law 2 ("algebraically exact, NOT bit-exact") is refuted IN-DOMAIN:
    /// the scale-then-add vs subtract-then-scale reassociation rounds
    /// nothing when every intermediate is on the exact grid and every
    /// scaling is a power of two. A failure here is a RESULT — record the
    /// drift, never silently rewrite the premise.
    #[test]
    fn journal_keep_halve_pinned_slots_restore_bit_exactly() {
        let (h, kd, vd) = (2usize, 16usize, 8usize);
        // Per-step decays, applied to BOTH heads (per-head products equal
        // the per-step product). A: mixed keep/halve, window product 0.5 —
        // above the default floor. B: eight halve steps, product 2^-8 —
        // the gate passes AT the explicit eps (refusal is `p < eps`, never
        // `p ≤ eps`).
        let schedule_a: &[f32] = &[1.0, 0.5, 1.0];
        let schedule_b: &[f32] = &[0.5; 8];
        for sched in [schedule_a, schedule_b] {
            let steps = sched.len();
            let mut rng = Lcg(0x1155_000C);
            // In-domain init: nonzero grid integers ±[1, 1024]; eight halve
            // steps from ≥1 stays ≥ 2^-8 — every intermediate a normal.
            let mut state: Vec<f32> = (0..h * kd * vd)
                .map(|_| {
                    let r = rng.next();
                    let m = (r % 1024) as f32 + 1.0;
                    if (r >> 63) & 1 == 1 { -m } else { m }
                })
                .collect();
            let mut journal = GdnStateJournal::with_capacity(h, kd, vd, steps);
            let mut snapshots: Vec<Vec<u32>> = vec![bits(&state)];

            let mut k_full = vec![0.0f32; h * kd];
            let mut v_full = vec![0.0f32; h * vd];
            let mut kv_full = vec![0.0f32; h * vd];
            let mut kv_head = vec![0.0f32; vd];

            for &g in sched.iter() {
                for head in 0..h {
                    let j = rng.slot(kd);
                    let kh = {
                        let mut x = vec![0.0f32; kd];
                        x[j] = 1.0;
                        x
                    };
                    let vh: Vec<f32> = (0..vd).map(|_| rng.grid()).collect();
                    let head_state = &mut state[head * kd * vd..(head + 1) * kd * vd];
                    step_one_head(&kh, &vh, head_state, 1.0, g, kd, vd, &mut kv_head);
                    k_full[head * kd..(head + 1) * kd].copy_from_slice(&kh);
                    v_full[head * vd..(head + 1) * vd].copy_from_slice(&vh);
                    kv_full[head * vd..(head + 1) * vd].copy_from_slice(&kv_head);
                }
                let betas = vec![1.0f32; h];
                let decays = vec![g; h];
                journal.record_step(&k_full, &v_full, &kv_full, &betas, &decays);
                snapshots.push(bits(&state));
            }

            // Unwind to EVERY target and demand the bit-exact snapshot.
            let window_eps = if steps > 3 {
                2f32.powi(-(steps as i32))
            } else {
                MIN_PRODUCT_EPS_DEFAULT
            };
            for (target, snap) in snapshots.iter().enumerate() {
                let mut restored = state.clone();
                let n = journal
                    .unwind_to(&mut restored, target, window_eps)
                    .unwrap_or_else(|e| panic!("unwind_to({target}) refused: {e:?}"));
                assert_eq!(n, steps - target);
                assert_eq!(
                    bits(&restored),
                    *snap,
                    "journal hypothesis FAILED: keep/halve in-domain unwind to \
                     target {target} is not bit-exact — a RESULT, record the drift"
                );
            }
        }
    }

    /// Journal law 1 stays load-bearing: a `g=0` window is refused
    /// FAIL-CLOSED — the state buffer is byte-identical after the Err.
    #[test]
    fn journal_clear_window_refused_fail_closed() {
        let (h, kd, vd) = (1usize, 8usize, 4usize);
        let mut state = vec![1.0f32; h * kd * vd];
        let mut journal = GdnStateJournal::with_capacity(h, kd, vd, 1);
        let kh = vec![1.0f32; kd];
        let vh = vec![2.0f32; vd];
        let mut kv = vec![0.0f32; vd];
        step_one_head(&kh, &vh, &mut state, 1.0, 0.0, kd, vd, &mut kv);
        journal.record_step(&kh, &vh, &kv, &[1.0], &[0.0]);
        let before = bits(&state);
        let err = journal
            .unwind_to(&mut state, 0, MIN_PRODUCT_EPS_DEFAULT)
            .expect_err("a g=0 window must be refused (irreversible by construction)");
        assert_eq!(err.head, 0);
        assert_eq!(err.product, 0.0);
        assert_eq!(
            bits(&state),
            before,
            "fail-closed: the refusal must not touch state"
        );
    }
}
