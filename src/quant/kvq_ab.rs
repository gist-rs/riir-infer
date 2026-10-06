//! KV-cache quantization A/B backends — Issue 919 T3 (the measured-diagonal
//! per-channel exemption vs per-block absmax lane).
//!
//! All three backends implement [`QuantizedKVCache`] (katgpt-types) and run
//! unchanged under [`forward_gemma2_f16_qkv`](crate::transformer::gemma2_quantized)
//! — the T3 measurement adds ZERO forward code: pass 1 collects the cache
//! diagonal with [`DiagKvCache`], the arms are the remaining backends.
//!
//! Arms (the Issue 919 T3 design, re-aimed by T2's verdict):
//!
//! | backend | policy | bpw (S=2) |
//! |---|---|---|
//! | [`RawF32KvCache`] | raw f32 passthrough — the ceiling + the paired per-token baseline | 32.0 |
//! | [`Q8AbsmaxKvCache`] | per-32-block absmax Q8_0 (the Research-487 gap subject) | 8.5 |
//! | [`ExemptQ8KvCache`] | measured top-S channels held at f16, zeroed out of their block's absmax; the rest Q8_0 | 8.5 + S/2 |
//!
//! The equal-budget discriminator is **[`ExemptQ8KvCache`] with the measured
//! channel set vs the same backend with a seeded-random channel set** —
//! identical bpw, so any quality delta isolates the measured diagonal, not
//! the extra bits. (The synthetic mechanism is Bench 691: one massive
//! channel sets its block's scale and its 31 neighbors quantize at ~1 step;
//! the exemption removes the massive channel from the scale computation.)
//!
//! Layout note (the exemption semantics): the exempt channels' values are
//! copied to a per-(layer, position) f16 sidecar BEFORE quantization and
//! zeroed in the row, so the block scale is computed from the remaining
//! channels; `dequantize_*_into` restores them from the sidecar at f16
//! precision. The quantized blocks keep the Q8_0 wire layout — a production
//! consumer would carry the sidecar beside them.
//!
//! Measurement-lane law (the vk_p1_g1 P0 posture): these backends make no
//! serving claim; the A/B bin's pre-registered gates decide quality.

use bytemuck::Zeroable;
use half::f16;
use katgpt_types::QuantizedKVCache;

use crate::quant::q8kv::{BlockQ8_0, dequantize_row_q8_0, quantize_row_q8_0};

// ── Raw passthrough ─────────────────────────────────────────────

/// Raw f32 passthrough — the full-precision ceiling and the paired baseline.
///
/// Stores the rows exactly as handed over (post-RoPE K, raw V) and
/// dequantizes by copy, so the mirror the forward attends over is
/// bit-identical to the stored rows.
pub struct RawF32KvCache {
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    kvd: usize,
    pos: usize,
}

impl RawF32KvCache {
    #[must_use]
    pub fn new(n_layer: usize, max_seq_len: usize, kvd: usize) -> Self {
        Self {
            k: vec![vec![0.0; max_seq_len * kvd]; n_layer],
            v: vec![vec![0.0; max_seq_len * kvd]; n_layer],
            kvd,
            pos: 0,
        }
    }
}

impl QuantizedKVCache for RawF32KvCache {
    #[inline]
    fn store_key(&mut self, layer: usize, pos: usize, key: &[f32]) {
        let off = pos * self.kvd;
        self.k[layer][off..off + self.kvd].copy_from_slice(key);
    }

    #[inline]
    fn store_value(&mut self, layer: usize, pos: usize, value: &[f32]) {
        let off = pos * self.kvd;
        self.v[layer][off..off + self.kvd].copy_from_slice(value);
    }

    #[inline]
    fn dequantize_key_into(&mut self, layer: usize, pos: usize, out: &mut [f32]) {
        let off = pos * self.kvd;
        out.copy_from_slice(&self.k[layer][off..off + self.kvd]);
    }

    #[inline]
    fn dequantize_value_into(&mut self, layer: usize, pos: usize, out: &mut [f32]) {
        let off = pos * self.kvd;
        out.copy_from_slice(&self.v[layer][off..off + self.kvd]);
    }

    #[inline]
    fn reset(&mut self) {
        self.pos = 0;
    }

    #[inline]
    fn pos(&self) -> usize {
        self.pos
    }

    #[inline]
    fn set_pos(&mut self, pos: usize) {
        self.pos = pos;
    }
}

// ── Diagonal collector ─────────────────────────────────────────

/// Exact per-(layer, kind, channel) max |x| and Σx² of the cached rows —
/// the pass-1 diagonal collector. Composition over [`RawF32KvCache`]: the
/// raw rows are needed anyway (the forward's mirror dequantizes from the
/// backend), and the stats accumulate at store time, so the diagonal is
/// EXACT over whatever text the pass feeds (not a capped sample).
pub struct DiagKvCache {
    raw: RawF32KvCache,
    /// `[layer][channel]` max |x|, keys.
    pub max_abs_k: Vec<Vec<f32>>,
    /// `[layer][channel]` max |x|, values.
    pub max_abs_v: Vec<Vec<f32>>,
    /// `[layer][channel]` Σx², keys.
    pub sum_sq_k: Vec<Vec<f64>>,
    /// `[layer][channel]` Σx², values.
    pub sum_sq_v: Vec<Vec<f64>>,
    /// Rows observed (per store call, k and v each count one).
    pub rows_observed: u64,
    pos: usize,
}

impl DiagKvCache {
    #[must_use]
    pub fn new(n_layer: usize, max_seq_len: usize, kvd: usize) -> Self {
        Self {
            raw: RawF32KvCache::new(n_layer, max_seq_len, kvd),
            max_abs_k: vec![vec![0.0; kvd]; n_layer],
            max_abs_v: vec![vec![0.0; kvd]; n_layer],
            sum_sq_k: vec![vec![0.0; kvd]; n_layer],
            sum_sq_v: vec![vec![0.0; kvd]; n_layer],
            rows_observed: 0,
            pos: 0,
        }
    }

    /// Top-`s` channels per (layer, kind) by the selected statistic.
    /// `rms=false` ranks by max |x| — the MA metric (the channel that sets
    /// its block's scale); `rms=true` ranks by RMS. Returns
    /// `(channel, statistic)` sorted descending by the statistic.
    #[must_use]
    pub fn top_channels(&self, layer: usize, is_k: bool, s: usize, rms: bool) -> Vec<(usize, f64)> {
        match (is_k, rms) {
            (true, false) => rank_f32(&self.max_abs_k[layer], s),
            (false, false) => rank_f32(&self.max_abs_v[layer], s),
            (true, true) => rank_f64_rms(&self.sum_sq_k[layer], self.rows_observed, s),
            (false, true) => rank_f64_rms(&self.sum_sq_v[layer], self.rows_observed, s),
        }
    }
}

impl QuantizedKVCache for DiagKvCache {
    fn store_key(&mut self, layer: usize, pos: usize, key: &[f32]) {
        self.raw.store_key(layer, pos, key);
        let max_row = &mut self.max_abs_k[layer];
        let sq_row = &mut self.sum_sq_k[layer];
        for (c, &x) in key.iter().enumerate() {
            let a = x.abs();
            if a > max_row[c] {
                max_row[c] = a;
            }
            sq_row[c] += f64::from(x) * f64::from(x);
        }
        self.rows_observed += 1;
    }

    fn store_value(&mut self, layer: usize, pos: usize, value: &[f32]) {
        self.raw.store_value(layer, pos, value);
        let max_row = &mut self.max_abs_v[layer];
        let sq_row = &mut self.sum_sq_v[layer];
        for (c, &x) in value.iter().enumerate() {
            let a = x.abs();
            if a > max_row[c] {
                max_row[c] = a;
            }
            sq_row[c] += f64::from(x) * f64::from(x);
        }
        self.rows_observed += 1;
    }

    #[inline]
    fn dequantize_key_into(&mut self, layer: usize, pos: usize, out: &mut [f32]) {
        self.raw.dequantize_key_into(layer, pos, out);
    }

    #[inline]
    fn dequantize_value_into(&mut self, layer: usize, pos: usize, out: &mut [f32]) {
        self.raw.dequantize_value_into(layer, pos, out);
    }

    #[inline]
    fn reset(&mut self) {
        self.raw.reset();
        self.pos = 0;
    }

    #[inline]
    fn pos(&self) -> usize {
        self.pos
    }

    #[inline]
    fn set_pos(&mut self, pos: usize) {
        self.pos = pos;
        self.raw.set_pos(pos);
    }
}

/// Rank channel indices by value, descending; `(channel, value)`.
fn rank_f32(v: &[f32], s: usize) -> Vec<(usize, f64)> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| v[b].total_cmp(&v[a]));
    idx.truncate(s);
    idx.into_iter().map(|c| (c, f64::from(v[c]))).collect()
}

/// Rank by RMS = sqrt(Σx²/n), descending.
fn rank_f64_rms(v: &[f64], n: u64, s: usize) -> Vec<(usize, f64)> {
    let n = f64::from(u32::try_from(n.max(1)).unwrap_or(u32::MAX));
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| v[b].total_cmp(&v[a]));
    idx.truncate(s);
    idx.into_iter()
        .map(|c| (c, (v[c] / n).sqrt()))
        .collect()
}

// ── Q8_0 absmax ────────────────────────────────────────────────

/// Per-32-block absmax Q8_0 — the Research-487 gap subject (the shipped
/// quantize path, 8.5 bpw). One massive channel sets its block's scale.
pub struct Q8AbsmaxKvCache {
    k: Vec<Vec<BlockQ8_0>>,
    v: Vec<Vec<BlockQ8_0>>,
    blocks_per_row: usize,
    pos: usize,
}

impl Q8AbsmaxKvCache {
    #[must_use]
    pub fn new(n_layer: usize, max_seq_len: usize, kvd: usize) -> Self {
        assert!(kvd.is_multiple_of(32), "kvd {kvd} must be a multiple of 32");
        let blocks_per_row = kvd / 32;
        let per_layer = max_seq_len * blocks_per_row;
        Self {
            k: vec![vec![BlockQ8_0::zeroed(); per_layer]; n_layer],
            v: vec![vec![BlockQ8_0::zeroed(); per_layer]; n_layer],
            blocks_per_row,
            pos: 0,
        }
    }
}

impl QuantizedKVCache for Q8AbsmaxKvCache {
    fn store_key(&mut self, layer: usize, pos: usize, key: &[f32]) {
        let bpr = self.blocks_per_row;
        quantize_row_q8_0(key, &mut self.k[layer][pos * bpr..(pos + 1) * bpr]);
    }

    fn store_value(&mut self, layer: usize, pos: usize, value: &[f32]) {
        let bpr = self.blocks_per_row;
        quantize_row_q8_0(value, &mut self.v[layer][pos * bpr..(pos + 1) * bpr]);
    }

    fn dequantize_key_into(&mut self, layer: usize, pos: usize, out: &mut [f32]) {
        let bpr = self.blocks_per_row;
        dequantize_row_q8_0(&self.k[layer][pos * bpr..(pos + 1) * bpr], out);
    }

    fn dequantize_value_into(&mut self, layer: usize, pos: usize, out: &mut [f32]) {
        let bpr = self.blocks_per_row;
        dequantize_row_q8_0(&self.v[layer][pos * bpr..(pos + 1) * bpr], out);
    }

    #[inline]
    fn reset(&mut self) {
        self.pos = 0;
    }

    #[inline]
    fn pos(&self) -> usize {
        self.pos
    }

    #[inline]
    fn set_pos(&mut self, pos: usize) {
        self.pos = pos;
    }
}

// ── Q8_0 + f16 channel exemption ───────────────────────────────

/// Q8_0 with per-channel f16 exemption — the T3 treatment arm.
///
/// `exempt_k[layer]` / `exempt_v[layer]` name the channels (per row) held
/// at f16 in a per-(layer, position) sidecar. At store time the exempt
/// channels are copied into the sidecar and ZEROED in the row copy before
/// block quantization, so the block scale comes from the remaining
/// channels; at dequant time the sidecar values overwrite the dequantized
/// rows at those channels.
///
/// bpw = 8.5 + S/2 (the sidecar adds S f16 values per 32 channels).
pub struct ExemptQ8KvCache {
    k: Vec<Vec<BlockQ8_0>>,
    v: Vec<Vec<BlockQ8_0>>,
    /// f16 bits, `[layer][pos * S .. pos * S + S]`, keys.
    sidecar_k: Vec<Vec<u16>>,
    /// f16 bits, values.
    sidecar_v: Vec<Vec<u16>>,
    exempt_k: Vec<Vec<usize>>,
    exempt_v: Vec<Vec<usize>>,
    /// Row copy scratch (exempt channels zeroed before quantize).
    scratch: Vec<f32>,
    blocks_per_row: usize,
    s: usize,
    kvd: usize,
    pos: usize,
}

impl ExemptQ8KvCache {
    /// `exempt_k`/`exempt_v` must carry exactly `s` STRICTLY ASCENDING
    /// channel indices per layer, all `< kvd` (ascending so the zeroing
    /// pass is branch-free and the sets are canonical for comparison).
    #[must_use]
    pub fn new(
        n_layer: usize,
        max_seq_len: usize,
        kvd: usize,
        exempt_k: Vec<Vec<usize>>,
        exempt_v: Vec<Vec<usize>>,
        s: usize,
    ) -> Self {
        assert!(kvd.is_multiple_of(32), "kvd {kvd} must be a multiple of 32");
        assert_eq!(exempt_k.len(), n_layer, "exempt_k per layer");
        assert_eq!(exempt_v.len(), n_layer, "exempt_v per layer");
        for (sets, what) in [(&exempt_k, "exempt_k"), (&exempt_v, "exempt_v")] {
            for (l, set) in sets.iter().enumerate() {
                assert_eq!(set.len(), s, "{what}[{l}] must carry exactly {s} channels");
                assert!(
                    set.windows(2).all(|w| w[0] < w[1]) && set.last().is_none_or(|&c| c < kvd),
                    "{what}[{l}] must be strictly ascending channels < {kvd}"
                );
            }
        }
        let blocks_per_row = kvd / 32;
        let per_layer = max_seq_len * blocks_per_row;
        Self {
            k: vec![vec![BlockQ8_0::zeroed(); per_layer]; n_layer],
            v: vec![vec![BlockQ8_0::zeroed(); per_layer]; n_layer],
            sidecar_k: vec![vec![0; max_seq_len * s]; n_layer],
            sidecar_v: vec![vec![0; max_seq_len * s]; n_layer],
            exempt_k,
            exempt_v,
            scratch: vec![0.0; kvd],
            blocks_per_row,
            s,
            kvd,
            pos: 0,
        }
    }

    /// The exemption channel sets, for reporting/overlap checks.
    #[must_use]
    pub fn exempt_sets(&self, is_k: bool) -> &[Vec<usize>] {
        if is_k { &self.exempt_k } else { &self.exempt_v }
    }

    fn store_row(
        &mut self,
        layer: usize,
        pos: usize,
        is_k: bool,
        row: &[f32],
    ) {
        let bpr = self.blocks_per_row;
        let s = self.s;
        let kvd = self.kvd;
        let (blocks, sidecar, exempt) = match is_k {
            true => (&mut self.k[layer], &mut self.sidecar_k[layer], &self.exempt_k[layer]),
            false => (&mut self.v[layer], &mut self.sidecar_v[layer], &self.exempt_v[layer]),
        };
        let sc = &mut sidecar[pos * s..(pos + 1) * s];
        let scratch = &mut self.scratch;
        scratch[..kvd].copy_from_slice(row);
        for (i, &c) in exempt.iter().enumerate() {
            sc[i] = f16::from_f32(row[c]).to_bits();
            scratch[c] = 0.0;
        }
        quantize_row_q8_0(&scratch[..kvd], &mut blocks[pos * bpr..(pos + 1) * bpr]);
    }

    fn dequant_row_into(
        &self,
        layer: usize,
        pos: usize,
        is_k: bool,
        out: &mut [f32],
    ) {
        let bpr = self.blocks_per_row;
        let s = self.s;
        let blocks = match is_k {
            true => &self.k[layer],
            false => &self.v[layer],
        };
        dequantize_row_q8_0(&blocks[pos * bpr..(pos + 1) * bpr], out);
        let sidecar = match is_k {
            true => &self.sidecar_k[layer],
            false => &self.sidecar_v[layer],
        };
        let exempt = match is_k {
            true => &self.exempt_k[layer],
            false => &self.exempt_v[layer],
        };
        for (i, &c) in exempt.iter().enumerate() {
            out[c] = f16::from_bits(sidecar[pos * s + i]).to_f32();
        }
    }
}

impl QuantizedKVCache for ExemptQ8KvCache {
    fn store_key(&mut self, layer: usize, pos: usize, key: &[f32]) {
        self.store_row(layer, pos, true, key);
    }

    fn store_value(&mut self, layer: usize, pos: usize, value: &[f32]) {
        self.store_row(layer, pos, false, value);
    }

    fn dequantize_key_into(&mut self, layer: usize, pos: usize, out: &mut [f32]) {
        self.dequant_row_into(layer, pos, true, out);
    }

    fn dequantize_value_into(&mut self, layer: usize, pos: usize, out: &mut [f32]) {
        self.dequant_row_into(layer, pos, false, out);
    }

    #[inline]
    fn reset(&mut self) {
        self.pos = 0;
    }

    #[inline]
    fn pos(&self) -> usize {
        self.pos
    }

    #[inline]
    fn set_pos(&mut self, pos: usize) {
        self.pos = pos;
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const L: usize = 2;
    const SEQ: usize = 8;
    const KVD: usize = 64;

    #[test]
    fn raw_roundtrip_is_exact() {
        let mut c = RawF32KvCache::new(L, SEQ, KVD);
        let row: Vec<f32> = (0..KVD).map(|i| (i as f32) * 0.25 - 3.0).collect();
        c.store_key(1, 3, &row);
        let mut out = vec![0.0; KVD];
        c.dequantize_key_into(1, 3, &mut out);
        assert_eq!(out, row);
    }

    #[test]
    fn diag_collects_exact_max_and_rms() {
        let mut c = DiagKvCache::new(1, SEQ, KVD);
        let r0: Vec<f32> = (0..KVD).map(|i| if i == 5 { -40.0 } else { 0.5 }).collect();
        let r1: Vec<f32> = (0..KVD).map(|i| if i == 5 { 30.0 } else { 0.5 }).collect();
        c.store_value(0, 0, &r0);
        c.store_value(0, 1, &r1);
        assert_eq!(c.rows_observed, 2);
        assert_eq!(c.max_abs_v[0][5], 40.0);
        // Channel 5: (-40)² + 30² over 2 rows; the quiet channels: 0.5² × 2.
        let expect5 = ((1600.0f64 + 900.0) / 2.0).sqrt();
        let expect_quiet = (0.25f64).sqrt();
        let top = c.top_channels(0, false, 2, true);
        assert_eq!(top[0].0, 5);
        assert!((top[0].1 - expect5).abs() < 1e-9, "{} vs {expect5}", top[0].1);
        // All quiet channels tie; rank 1 is some non-5 channel at the quiet RMS.
        assert_ne!(top[1].0, 5);
        assert!((top[1].1 - expect_quiet).abs() < 1e-9);
        let top_max = c.top_channels(0, false, 1, false);
        assert_eq!(top_max[0].0, 5);
        assert_eq!(top_max[0].1, 40.0);
    }

    #[test]
    fn exempt_removes_channel_from_block_scale() {
        // One massive channel + quiet neighbors: plain Q8 gives the quiet
        // channels ~1 quant step (Bench 691's row-collapse); the exemption
        // restores their resolution.
        // Spread quiet values (0.001–0.013): under the massive channel's
        // block scale (d = 300/127 ≈ 2.36) every one quantizes to q=0 and
        // dequantizes to 0.0 — the Bench-691 row collapse.
        let mut row = vec![0.0f32; KVD];
        for (i, v) in row.iter_mut().enumerate() {
            *v = if i == 7 { 300.0 } else { 0.001 + ((i % 7) as f32) * 0.002 };
        }
        let mut plain = Q8AbsmaxKvCache::new(1, SEQ, KVD);
        plain.store_value(0, 0, &row);
        let mut out_plain = vec![0.0; KVD];
        plain.dequantize_value_into(0, 0, &mut out_plain);
        let err_plain = (out_plain[0] - row[0]).abs();
        // Channel 0's value (0.001) is lost entirely under the shared scale:
        assert_eq!(out_plain[0], 0.0);
        assert!(err_plain > 0.0005, "plain quiet error {err_plain}");

        let mut ex = ExemptQ8KvCache::new(
            1,
            SEQ,
            KVD,
            vec![vec![0]],
            vec![vec![7]],
            1,
        );
        ex.store_value(0, 0, &row);
        let mut out_ex = vec![0.0; KVD];
        ex.dequantize_value_into(0, 0, &mut out_ex);
        // The exempt channel is restored at f16 (exact for 300.0):
        assert_eq!(out_ex[7], 300.0, "exempt channel restored at f16");
        // The neighbor's error collapses now that the block scale is set by
        // the quiet channels alone (d = max(quiet)/127):
        let err_ex = (out_ex[0] - row[0]).abs();
        assert!(err_ex < err_plain * 0.05, "exempt neighbor error {err_ex} vs plain {err_plain}");
    }

    #[test]
    fn exempt_sidecar_roundtrip_multi_position() {
        let mut ex = ExemptQ8KvCache::new(
            1,
            SEQ,
            KVD,
            vec![vec![1, 60]],
            vec![vec![2, 9]],
            2,
        );
        let k0: Vec<f32> = (0..KVD).map(|i| if i == 1 { -123.5 } else { 0.7 }).collect();
        let v1: Vec<f32> = (0..KVD).map(|i| if i == 9 { 88.25 } else { -0.2 }).collect();
        ex.store_key(0, 0, &k0);
        ex.store_value(0, 1, &v1);
        let mut out = vec![0.0; KVD];
        ex.dequantize_key_into(0, 0, &mut out);
        assert_eq!(out[1], -123.5, "k exempt ch restored at pos 0");
        assert_eq!(out[60], f16::from_f32(0.7).to_f32());
        let mut out = vec![0.0; KVD];
        ex.dequantize_value_into(0, 1, &mut out);
        assert_eq!(out[9], 88.25, "v exempt ch restored at pos 1");
        assert_eq!(out[2], f16::from_f32(-0.2).to_f32());
    }

    #[test]
    #[should_panic(expected = "strictly ascending")]
    fn exempt_rejects_unsorted_sets() {
        let _ = ExemptQ8KvCache::new(1, SEQ, KVD, vec![vec![5, 3]], vec![vec![], ], 2);
    }
}
