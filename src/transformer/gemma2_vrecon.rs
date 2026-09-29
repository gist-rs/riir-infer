//! gemma2_vrecon — Issue 013 T3: the P3 V-cache reconstruction lane over
//! the gemma-2 f16 decode path. The deployment claim: drop the persistent V
//! cache and serve, at read time, `V = G(−θp)·K̂ + λ·E_l[s]` — one
//! one-directional inverse rotation of the cached post-RoPE key plus the
//! P2-fitted table row (katgpt-rs Issue 883 P3; the trait is
//! [`katgpt_core::position_group_action::PositionGroupAction`], the selector
//! `VReadPath`/`reconstruct_v_from_rope_k`).
//!
//! ## The half-split convention (Issue 013 trap 1)
//!
//! riir-engine's RoPE (and therefore the cache) uses the **rotate-half**
//! convention — pairs `(i, i + half)` — matching HuggingFace `rotate_half`.
//! katgpt-core's [`katgpt_core::position_group_action::RopeAction`] rotates
//! **adjacent** pairs `(2i, 2i+1)` — a different rotation subgroup, and the
//! wrong one here (the same mismatch the RoVE note in `crate::rope`
//! records). [`HalfSplitRopeInverse`] is the model-bound action: it reads
//! THIS forward's own frequency table and inverts exactly the rotation the
//! cache write applied — the same `angle = pos·freq` expression, the same
//! `f32::sin_cos`, and the same pos-0 early return
//! ([`crate::rope::apply_rope_with_freq`] skips the rotation entirely at
//! pos 0, so the inverse copies bitwise there — the −0.0 class included).
//!
//! ## What is claimed / not claimed
//!
//! The reconstruction reproduces T2's store-time serve (`v_from_k_plus`
//! over the pre-RoPE K tap) **up to rotation rounding** (the 1.83ε class —
//! the forward rotation and this inverse are transposes over the same
//! f32 angles; the G1 pairing measures the exact delta). MEASUREMENT-ONLY:
//! promotion is katgpt-rs-side and waits on T3's gates; the W_V FLOP claim
//! and the kernel levers are T4's lane.

use katgpt_core::fitted_value_table::{reconstruct_v_from_rope_k, FittedTokenTable};
use katgpt_core::position_group_action::PositionGroupAction;

use crate::transformer::gemma2::ValueStoreHook;

/// The rotate-half `PositionGroupAction` over the forward's own frequency
/// table (`head_dim/2` entries, e.g. 128 for gemma-2-2b's hd 256).
///
/// `apply_at` is the cache write's forward rotation (round-trip tests only —
/// never the hot path); `apply_inverse_at` is the ONE one-directional
/// inverse the read path uses (Issue 013 trap 2: never round-trip). Both are
/// zero-alloc and in-place-safe (each index is written once, after its last
/// read).
pub struct HalfSplitRopeInverse<'a> {
    freq: &'a [f32],
}

impl<'a> HalfSplitRopeInverse<'a> {
    /// `freq_table` must be the forward's own table (`head_dim/2` entries —
    /// `RopeFreqTable::new(theta, head_dim)`), so the angles are the exact
    /// values the cache write used.
    #[must_use]
    pub fn new(freq_table: &'a [f32]) -> Self {
        Self { freq: freq_table }
    }
}

impl PositionGroupAction for HalfSplitRopeInverse<'_> {
    /// One head's width (`head_dim`).
    fn dim(&self) -> usize {
        self.freq.len() * 2
    }

    /// Forward rotate-half — `apply_rope_heads_precomputed`'s exact
    /// convention: `out[i] = x[i]·cos − x[i+half]·sin`,
    /// `out[i+half] = x[i]·sin + x[i+half]·cos`.
    fn apply_at(&self, n: f32, x: &[f32], out: &mut [f32]) {
        // pos 0 is the forward's early return (identity, bitwise).
        if n == 0.0 {
            out.copy_from_slice(x);
            return;
        }
        let half = self.freq.len();
        debug_assert_eq!(x.len(), half * 2, "action dim mismatch");
        for (i, &f) in self.freq.iter().enumerate() {
            let (sin_a, cos_a) = (n * f).sin_cos();
            let (x0, x1) = (x[i], x[i + half]);
            out[i] = x0 * cos_a - x1 * sin_a;
            out[i + half] = x0 * sin_a + x1 * cos_a;
        }
    }

    /// The cache write's exact inverse — cos identical, sin negated
    /// (the rotation matrix transposed). pos 0 copies bitwise, matching the
    /// forward's early return (a computed identity would flip −0.0 → +0.0
    /// in `x·cos + y·sin` terms; the cache at pos 0 holds the raw key).
    fn apply_inverse_at(&self, n: f32, x: &[f32], out: &mut [f32]) {
        if n == 0.0 {
            out.copy_from_slice(x);
            return;
        }
        let half = self.freq.len();
        debug_assert_eq!(x.len(), half * 2, "action dim mismatch");
        for (i, &f) in self.freq.iter().enumerate() {
            let (sin_a, cos_a) = (n * f).sin_cos();
            let (x0, x1) = (x[i], x[i + half]);
            out[i] = x0 * cos_a + x1 * sin_a;
            out[i + half] = -x0 * sin_a + x1 * cos_a;
        }
    }
}

/// The Issue-013 T3 [`ValueStoreHook`] state — the P3 read-path lane.
///
/// Attention reads THIS state's scratch: rows `0..t_n` of
/// `V = G(−θp)·K̂ + λ_l·E_l[s]`, reconstructed from the cached post-RoPE
/// keys each time attention asks. The persistent V cache is never read (the
/// raw V store still executes in this instrument — one memcpy per step,
/// recorded as an instrument caveat; a production P3 cache drops the
/// allocation and the bytes/token claim is the recorded arithmetic).
///
/// The table borrow is `'static` BY CALLER CONTRACT (the bin `Box::leak`s
/// the loaded table — process-lifetime by design, the `KToVState` shape).
pub struct VReconState {
    table: &'static FittedTokenTable,
    /// Per-layer λ (the schedule; a uniform arm is `vec![λ; L]`).
    lam: Vec<f32>,
    /// Token id per position (`u32::MAX` = unset ⇒ the miss path: exact
    /// `G(−θp)·K̂`, never a zero-row guess).
    tokens: Vec<u32>,
    /// The forward's own rope frequency table (cloned — the hook cannot
    /// borrow the `ForwardContext`; `head_dim/2` floats).
    freq: Vec<f32>,
    kvd: usize,
    /// The served-V scratch (`max_seq_len × kvd`), allocated once.
    scratch: Vec<f32>,
    /// Arm-lifetime telemetry: rows reconstructed.
    pub rows_served: u64,
}

impl VReconState {
    /// `lam` must have one entry per layer; `freq_table` must be the
    /// forward's own `RopeFreqTable` contents; the table width must equal
    /// `kvd`; `kvd` must be a head multiple of the action dim.
    #[must_use]
    pub fn new(
        table: &'static FittedTokenTable,
        lam: Vec<f32>,
        n_layers: usize,
        kvd: usize,
        freq_table: &[f32],
        max_seq_len: usize,
    ) -> Self {
        assert_eq!(
            lam.len(),
            n_layers,
            "VReconState: lam schedule must cover every layer"
        );
        assert_eq!(
            table.width(),
            kvd,
            "VReconState: table width must equal kv_dim"
        );
        assert!(
            kvd.is_multiple_of(freq_table.len() * 2),
            "VReconState: kv_dim not a head multiple of the action dim"
        );
        Self {
            table,
            lam,
            tokens: vec![u32::MAX; max_seq_len],
            freq: freq_table.to_vec(),
            kvd,
            scratch: vec![0.0; max_seq_len * kvd],
            rows_served: 0,
        }
    }

    /// Uniform-λ arm constructor.
    #[must_use]
    pub fn new_uniform(
        table: &'static FittedTokenTable,
        lam: f32,
        n_layers: usize,
        kvd: usize,
        freq_table: &[f32],
        max_seq_len: usize,
    ) -> Self {
        Self::new(
            table,
            vec![lam; n_layers],
            n_layers,
            kvd,
            freq_table,
            max_seq_len,
        )
    }

    /// Record the token at `pos` (call once per decode step, before the
    /// forward — including every prefill position: the read path looks up
    /// rows for ALL cached positions, not just the current one).
    #[inline]
    pub fn set_token(&mut self, pos: usize, token: u32) {
        if let Some(t) = self.tokens.get_mut(pos) {
            *t = token;
        }
    }

    /// Reset for a new sequence (the token map; telemetry survives).
    pub fn reset(&mut self) {
        self.tokens.fill(u32::MAX);
    }

    /// Overwrite the λ schedule in place.
    pub fn set_lam(&mut self, layer: usize, value: f32) {
        self.lam[layer] = value;
    }

    /// The live schedule.
    #[must_use]
    pub fn lam(&self) -> &[f32] {
        &self.lam
    }
}

impl ValueStoreHook for VReconState {
    // keys_pre_rope / value_stored stay default no-ops: the P3 lane reads
    // the cached post-RoPE K at attention time and never touches the store
    // path — the raw V rows in the cache are dead weight this instrument
    // still writes (the recorded caveat above).

    fn values_for_attention(
        &mut self,
        layer_idx: usize,
        _pos: usize,
        t_n: usize,
        k_cache: &[f32],
    ) -> Option<&[f32]> {
        let Self {
            table,
            lam,
            tokens,
            freq,
            kvd,
            scratch,
            rows_served,
        } = self;
        let kvd = *kvd;
        let lam_l = lam[layer_idx];
        let action = HalfSplitRopeInverse::new(freq);
        for p in 0..t_n {
            let k = &k_cache[p * kvd..(p + 1) * kvd];
            let out = &mut scratch[p * kvd..(p + 1) * kvd];
            let tok = tokens.get(p).copied().unwrap_or(u32::MAX);
            let row = table.row(layer_idx, tok);
            reconstruct_v_from_rope_k(&action, p as f32, k, row, lam_l, out);
        }
        *rows_served += t_n as u64;
        Some(&scratch[..t_n * kvd])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use katgpt_core::fitted_value_table::{v_from_k_plus, VReadPath, read_v};

    const HD: usize = 16;
    const N_KV: usize = 4;
    const KVD: usize = N_KV * HD;

    fn freq_table() -> Vec<f32> {
        crate::rope::RopeFreqTable::new(10_000.0, HD).as_slice().to_vec()
    }

    fn k_pre_row(seed: usize) -> Vec<f32> {
        (0..KVD)
            .map(|i| ((seed * 31 + i * 7) % 23) as f32 / 4.0 - 2.5)
            .collect()
    }

    /// The repo's own forward rotation over a whole multi-head buffer — the
    /// exact code path the cache write uses.
    fn forward_rotate(k: &mut [f32], pos: usize, freq: &[f32]) {
        let mut q = vec![0.0f32; KVD];
        crate::rope::apply_rope_with_freq(&mut q, k, pos, HD, freq);
    }

    /// G-convention: inverse(forward(x)) recovers x to f32 rounding, and
    /// pos 0 recovers BITWISE (the forward's early return) — −0.0 included.
    #[test]
    fn inverse_matches_the_forward_convention() {
        let freq = freq_table();
        let action = HalfSplitRopeInverse::new(&freq);
        for pos in [0usize, 1, 3, 17, 255] {
            let mut x: Vec<f32> = (0..HD).map(|i| ((pos * 13 + i * 5) % 9) as f32 - 4.0).collect();
            if pos == 0 {
                x[2] = -0.0;
                x[5] = 0.0;
            }
            let orig = x.clone();
            let mut rot = vec![0.0f32; HD];
            action.apply_at(pos as f32, &x, &mut rot);
            let mut back = vec![0.0f32; HD];
            action.apply_inverse_at(pos as f32, &rot, &mut back);
            if pos == 0 {
                assert_eq!(
                    back.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    orig.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    "pos 0 inverse must be bitwise (−0.0 class)"
                );
            } else {
                for (a, b) in back.iter().zip(&orig) {
                    assert!(
                        (a - b).abs() < 1e-4,
                        "pos {pos}: round-trip {a} vs {b}"
                    );
                }
            }
        }
    }

    /// The round trip through the REPO's own rope (the real cache write):
    /// forward_rotate → reconstruct(λ=0, no row) ≈ k_pre; pos 0 bitwise.
    #[test]
    fn reconstruction_matches_repo_rope_round_trip() {
        let freq = freq_table();
        for pos in [0usize, 1, 5, 130] {
            let mut k = k_pre_row(pos);
            let orig = k.clone();
            forward_rotate(&mut k, pos, &freq);
            let action = HalfSplitRopeInverse::new(&freq);
            let mut out = vec![0.0f32; KVD];
            reconstruct_v_from_rope_k(&action, pos as f32, &k, None, 0.0, &mut out);
            if pos == 0 {
                assert_eq!(out, orig, "pos 0: bitwise recovery");
            } else {
                for (a, b) in out.iter().zip(&orig) {
                    assert!((a - b).abs() < 1e-3, "pos {pos}: {a} vs {b}");
                }
            }
        }
    }

    /// The P3 == P2 law: reconstructing from the ROTATED key with row+λ
    /// matches `v_from_k_plus(k_pre, row, λ)` (the T2 serve) to rotation
    /// rounding; pos 0 is bitwise.
    #[test]
    fn reconstruct_matches_v_from_k_plus_on_rotated_k() {
        let freq = freq_table();
        let table = FittedTokenTable::from_rows(
            1,
            KVD,
            vec![0],
            (0..KVD).map(|i| (i as f32) * 0.125 - 1.0).collect(),
        );
        let row = table.row(0, 0).map(<[f32]>::to_vec).unwrap();
        for (pos, lam) in [(0usize, 0.5f32), (7, 0.5), (33, 1.0), (7, 0.0)] {
            let mut k = k_pre_row(pos * 3 + 1);
            let k_pre = k.clone();
            forward_rotate(&mut k, pos, &freq);
            let action = HalfSplitRopeInverse::new(&freq);
            let mut out = vec![0.0f32; KVD];
            reconstruct_v_from_rope_k(&action, pos as f32, &k, Some(&row), lam, &mut out);
            let mut want = vec![0.0f32; KVD];
            v_from_k_plus(&k_pre, Some(&row), lam, &mut want);
            if pos == 0 && lam == 0.5 {
                // pos 0: the inverse copies bitwise, so λ=0.5's add matches
                // the store path element-for-element too (same add order).
                assert_eq!(out, want, "pos 0: P3 == P2 bitwise");
            }
            for (a, b) in out.iter().zip(&want) {
                assert!(
                    (a - b).abs() < 1e-3,
                    "pos {pos} λ{lam}: {a} vs {b}"
                );
            }
        }
    }

    /// katgpt-core's `read_v` FullCache arm is a bitwise copy; Reconstruct
    /// with λ=0 and no row recovers the pre-RoPE key (through the selector,
    /// not just the raw fn).
    #[test]
    fn read_v_fullcache_is_a_bitwise_copy() {
        let freq = freq_table();
        let v: Vec<f32> = (0..KVD).map(|i| (i as f32) * 0.5 - 3.0).collect();
        let mut out = vec![9.0f32; KVD];
        read_v(
            VReadPath::FullCache,
            &HalfSplitRopeInverse::new(&freq),
            3.0,
            &[0.0; KVD],
            Some(&v),
            None,
            &mut out,
        );
        assert_eq!(
            out.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            v.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
    }

    /// The state end-to-end: pre-rotate rows, then serve through the hook
    /// seam; tracked tokens serve k̂+λ·row, misses serve exact k̂ (recovered
    /// to rounding), and the served row count accumulates.
    #[test]
    fn vrecon_state_serves_reconstructed_rows() {
        let freq = freq_table();
        let table = Box::leak(Box::new(FittedTokenTable::from_rows(
            2,
            KVD,
            vec![0, u32::MAX], // token 0 → row 0, token 1 → miss
            // rows = 1 (max tracked index + 1); every (layer, row 0)
            // identical: (i % KVD) · 0.25 — the expected row below is
            // layer-independent.
            (0..2 * KVD).map(|i| (i % KVD) as f32 * 0.25).collect(),
        )));
        let row: Vec<f32> = (0..KVD).map(|i| (i as f32) * 0.25).collect();
        let mut st = VReconState::new_uniform(table, 0.5, 2, KVD, &freq, 8);
        st.set_token(0, 0);
        st.set_token(1, 1);
        // Build a fake key cache: rows pre-rotated by the repo's rope.
        let mut k_cache = vec![0.0f32; 2 * KVD];
        for p in 0..2 {
            let mut k = k_pre_row(p + 4);
            forward_rotate(&mut k, p, &freq);
            k_cache[p * KVD..(p + 1) * KVD].copy_from_slice(&k);
        }
        let served: Vec<f32> =
            ValueStoreHook::values_for_attention(&mut st, 1, 0, 2, &k_cache)
                .expect("VReconState always overrides")
                .to_vec();
        assert_eq!(st.rows_served, 2);
        // Row 0: tracked → k̂ + 0.5·row (k̂ == k_pre bitwise at pos 0).
        for i in 0..KVD {
            let want = k_pre_row(4)[i] + 0.5 * row[i];
            assert!((served[i] - want).abs() < 1e-3, "row0[{i}] {} vs {want}", served[i]);
        }
        // Row 1: miss → exact recovered k̂ (to rounding).
        for i in 0..KVD {
            let want = k_pre_row(5)[i];
            assert!(
                (served[KVD + i] - want).abs() < 1e-3,
                "row1[{i}] {} vs {want}",
                served[KVD + i]
            );
        }
    }
}
