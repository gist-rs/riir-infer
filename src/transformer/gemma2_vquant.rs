//! gemma2_vquant — Issue 013 T1: the lossy-V seam state over the gemma-2
//! f16 decode path. A [`KVarNKVCache`] stores the layer's V rows at 2/3/4
//! bits, optionally decorated with katgpt-core's [`MeanRemovedValueCache`]
//! (the P1 token-mean product, katgpt-rs `0b768e95d` / Bench 895); the rows
//! a real reader would see are rewritten into the plain cache so the
//! existing attention kernels consume the lossy surface unmodified.
//!
//! ## The simulation law — rows are rewritten at TILE close, not at store
//!
//! `KVarNKVCache::store_value` buffers raw rows and quantizes a V tile only
//! when it FILLS (`quantize_val_tile`, 128 rows); until then
//! `dequantize_value_into` serves the RAW row (the Issue-896 exact-read
//! semantics). A real streaming consumer therefore reads within-open-tile
//! rows losslessly and closed-tile rows dequantized. This state reproduces
//! that visibility exactly: each call feeds the backend the raw row the
//! forward just stored (the plain cache holds it at `pos*kvd`), then — at
//! a tile's closing store, detected via `value_row_view(..).is_some()` —
//! overwrites every row of the tile in the plain cache with its
//! dequantized (mean-restored) form, BEFORE attention reads. A row is read
//! lossy iff a KVarN-backed cache would serve it lossy; within an open
//! tile nothing diverges from the plain path.
//!
//! ## Gate-1 telemetry
//!
//! At each flush the state accumulates, per layer: Σ(orig − served)²,
//! Σ orig², rows, max |orig| (the plain arm's absmax range), max
//! |orig − E^V_l[s]| (the mean-removed arm's encode-side range — the absmax
//! caveat's arbiter: if the encode range GREW, the RTN step grew and the
//! `1 − ρ` prediction degrades in that direction), and the tracked-token
//! row count (the realized table coverage on the eval slice).
//!
//! MEASUREMENT-ONLY (the issue's gate lane): the lossy surface is the
//! rewritten cache rows; storage is virtual (the KVarN buffers are the
//! record's bytes/token accounting, the plain cache is scratch). Promotion
//! is katgpt-rs-side and waits on the gates.

use katgpt_core::fitted_value_table::{FittedTokenTable, MeanRemovedValueCache};
use katgpt_core::types::QuantizedKVCache;
use katgpt_kv::kvarn::kv_cache::{KVarNConfig, KVarNKVCache};

use crate::transformer::gemma2::ValueStoreHook;

/// Per-layer quantizer telemetry (Gate 1 + the absmax caveat arbiter).
#[derive(Clone, Copy, Default, Debug)]
pub struct LayerMse {
    /// Σ (orig − served)² over the layer's flushed rows.
    pub sq_err: f64,
    /// Σ orig² over the same rows (signal energy).
    pub sq_ref: f64,
    /// Rows flushed (= rows whose later reads are lossy).
    pub rows: u64,
    /// max |orig| seen at a flush (the absmax range of the raw V rows).
    pub max_abs: f32,
    /// max |orig − E^V_l[s]| (the encode-side range; untracked tokens
    /// contribute |orig| — they encode unchanged).
    pub max_abs_enc: f32,
    /// Flushed rows whose token had a table row (realized coverage).
    pub rows_tracked: u64,
}

impl LayerMse {
    /// Per-element mean squared error of the served rows.
    #[must_use]
    pub fn mse(&self, kvd: usize) -> f64 {
        if self.rows == 0 || kvd == 0 {
            return 0.0;
        }
        self.sq_err / (self.rows as f64 * kvd as f64)
    }

    /// Per-element signal energy of the same rows.
    #[must_use]
    pub fn energy(&self, kvd: usize) -> f64 {
        if self.rows == 0 || kvd == 0 {
            return 0.0;
        }
        self.sq_ref / (self.rows as f64 * kvd as f64)
    }
}

/// Which V storage backs the arm: plain KVarN, or the P1 token-mean
/// decorator over it (the wrapper OWNS the backend by value).
enum Backing {
    Plain(KVarNKVCache),
    MeanRemoved(MeanRemovedValueCache<'static, KVarNKVCache>),
}

/// The Issue-013 T1 [`ValueStoreHook`] state — one per eval arm.
///
/// The table borrow is `'static` BY CALLER CONTRACT: the frozen table
/// outlives every arm (the bin `Box::leak`s the phase-B tables — they live
/// for the whole process by design), so no unsafe lives here.
pub struct VQuantState {
    backing: Backing,
    table: &'static FittedTokenTable,
    /// token id per position (`u32::MAX` = unset) — telemetry's own copy
    /// (the wrapper keeps another; `set_token` feeds both).
    tokens: Vec<u32>,
    tile_size: usize,
    kvd: usize,
    /// Per-layer quantizer telemetry; accumulates across chunks (an
    /// arm-lifetime quantity — `reset` clears the cache, never this).
    pub tel: Vec<LayerMse>,
    scratch_orig: Vec<f32>,
    scratch_row: Vec<f32>,
}

// The wrapper's table borrow is erased to 'static for storage; the bin
// leaks nothing — it owns the table for the whole process (the frozen
// artifact lives longer than every arm). Documented safety: the table
// reference outlives the state by construction of the bin's phases.
impl VQuantState {
    fn new(
        bits: u8,
        n_layers: usize,
        kvd: usize,
        max_seq_len: usize,
        tile_size: usize,
        table: &'static FittedTokenTable,
        mean_removed: bool,
    ) -> Self {
        let cfg = KVarNConfig {
            n_layers,
            kv_dim: kvd,
            max_seq_len,
            bits,
            tile_size,
            ..KVarNConfig::default()
        };
        let q = KVarNKVCache::with_config(&cfg);
        let backing = if mean_removed {
            Backing::MeanRemoved(MeanRemovedValueCache::new(q, table, max_seq_len))
        } else {
            Backing::Plain(q)
        };
        Self {
            backing,
            table,
            tokens: vec![u32::MAX; max_seq_len],
            tile_size,
            kvd,
            tel: vec![LayerMse::default(); n_layers],
            scratch_orig: vec![0.0; kvd],
            scratch_row: vec![0.0; kvd],
        }
    }

    /// The plain-KVarN arm (P1's control at the same bits).
    #[must_use]
    pub fn new_plain(
        bits: u8,
        n_layers: usize,
        kvd: usize,
        max_seq_len: usize,
        tile_size: usize,
        table: &'static FittedTokenTable,
    ) -> Self {
        Self::new(bits, n_layers, kvd, max_seq_len, tile_size, table, false)
    }

    /// The P1 mean-removed arm: stores `quant(V − E^V_l[s])`, reads
    /// `dequant + E^V_l[s]` (the [`MeanRemovedValueCache`] product verbatim).
    #[must_use]
    pub fn new_mean_removed(
        bits: u8,
        n_layers: usize,
        kvd: usize,
        max_seq_len: usize,
        tile_size: usize,
        table: &'static FittedTokenTable,
    ) -> Self {
        Self::new(bits, n_layers, kvd, max_seq_len, tile_size, table, true)
    }

    /// Record the token at `pos` (call once per decode step, before the
    /// forward). Untracked tokens take the plain path inside the wrapper.
    #[inline]
    pub fn set_token(&mut self, pos: usize, token: u32) {
        if let Some(t) = self.tokens.get_mut(pos) {
            *t = token;
        }
        if let Backing::MeanRemoved(m) = &mut self.backing {
            m.set_token(pos, token);
        }
    }

    /// Reset for a new sequence (cache + token maps; telemetry survives).
    pub fn reset(&mut self) {
        match &mut self.backing {
            Backing::Plain(q) => q.reset(),
            Backing::MeanRemoved(m) => m.reset(),
        }
        self.tokens.fill(u32::MAX);
    }

    /// The packed V bytes/token the backend occupies (the record's storage
    /// accounting; the KVarN V layout: `kv_dim·bits/8` packed + per-row RTN
    /// scale/zero + the tile's var-norm scales amortized over `tile_size`).
    #[must_use]
    pub fn v_bytes_per_token(&self, bits: u8) -> usize {
        let packed = (self.kvd * bits as usize).div_ceil(8);
        let per_row_rtn = 2 * 4; // f32 scale + f32 zero point per row
        let varn_amortized = 2 * 4 * self.kvd / self.tile_size.max(1); // s_col + s_row share
        packed + per_row_rtn + varn_amortized
    }

    /// Per-layer mean-ratio vector Gate 1 consumes (called by the bin at
    /// arm end; `kvd` normalizes to per-element MSE).
    #[must_use]
    pub fn layer_mse(&self) -> &[LayerMse] {
        &self.tel
    }

    /// Flush one row: orig (raw, from the plain cache) → dequantized
    /// (mean-restored) served form, telemetry in between.
    fn flush_row(&mut self, layer: usize, t: usize, layer_values: &mut [f32]) {
        let kvd = self.kvd;
        let off = t * kvd;
        self.scratch_orig.copy_from_slice(&layer_values[off..off + kvd]);
        match &mut self.backing {
            Backing::Plain(q) => q.dequantize_value_into(layer, t, &mut self.scratch_row),
            Backing::MeanRemoved(m) => m.dequantize_value_into(layer, t, &mut self.scratch_row),
        }
        // Telemetry — orig vs served, on the raw row snapshot.
        let tel = &mut self.tel[layer];
        let tok = self.tokens.get(t).copied().unwrap_or(u32::MAX);
        let row = self.table.row(layer, tok);
        let mut sq_err = 0.0f64;
        let mut sq_ref = 0.0f64;
        let mut max_abs = 0.0f32;
        let mut max_abs_enc = 0.0f32;
        for (i, &o) in self.scratch_orig.iter().enumerate() {
            let s = self.scratch_row[i];
            let d = o - s;
            sq_err += (d as f64) * (d as f64);
            sq_ref += (o as f64) * (o as f64);
            let a = o.abs();
            if a > max_abs {
                max_abs = a;
            }
            let e = match row {
                Some(m) => (o - m[i]).abs(),
                None => a,
            };
            if e > max_abs_enc {
                max_abs_enc = e;
            }
        }
        tel.sq_err += sq_err;
        tel.sq_ref += sq_ref;
        tel.rows += 1;
        if max_abs > tel.max_abs {
            tel.max_abs = max_abs;
        }
        if max_abs_enc > tel.max_abs_enc {
            tel.max_abs_enc = max_abs_enc;
        }
        if row.is_some() {
            tel.rows_tracked += 1;
        }
        layer_values[off..off + kvd].copy_from_slice(&self.scratch_row);
    }
}

impl ValueStoreHook for VQuantState {
    fn value_stored(&mut self, layer_idx: usize, pos: usize, layer_values: &mut [f32]) {
        let kvd = self.kvd;
        let off = pos * kvd;
        // Feed the backend the raw row the forward just stored in the plain
        // cache (the mean-removed wrapper removes its own mean on the way
        // in; untracked tokens are stored verbatim).
        match &mut self.backing {
            Backing::Plain(q) => q.store_value(layer_idx, pos, &layer_values[off..off + kvd]),
            Backing::MeanRemoved(m) => {
                m.store_value(layer_idx, pos, &layer_values[off..off + kvd]);
            }
        }
        // Tile-close detection: the row's tile quantized on THIS store
        // (before it the tile is raw and `value_row_view` is None — the
        // flush is a no-op, matching the open-tile visibility).
        let closed = match &mut self.backing {
            Backing::Plain(q) => q.value_row_view(layer_idx, pos).is_some(),
            Backing::MeanRemoved(m) => m.inner().value_row_view(layer_idx, pos).is_some(),
        };
        if !closed {
            return;
        }
        let t0 = pos - pos % self.tile_size;
        for t in t0..=pos {
            self.flush_row(layer_idx, t, layer_values);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transformer::gemma2::NoVQuant;

    /// A same-shape zero table — the P1 G3 fixture (mean removal is a
    /// no-op, so the mr arm must match the plain arm bitwise).
    fn zero_table(kvd: usize) -> FittedTokenTable {
        FittedTokenTable::from_rows(1, kvd, vec![0], vec![0.0; kvd])
    }

    fn leak(t: FittedTokenTable) -> &'static FittedTokenTable {
        Box::leak(Box::new(t))
    }

    /// The visibility law: within an OPEN tile the cache rows are untouched
    /// (raw); at the tile's closing store every row of the tile becomes the
    /// dequantized form (differs from raw at 2-bit on range-bearing rows).
    #[test]
    fn open_tile_raw_closed_tile_lossy() {
        let kvd = 32usize;
        let tile = 4usize;
        let max_seq = 8usize;
        let table = zero_table(kvd);
        let tstatic = leak(table);
        let mut st = VQuantState::new_plain(2, 1, kvd, max_seq, tile, tstatic);
        let mut vals = vec![0.0f32; max_seq * kvd];
        // Deterministic rows with range (RTN step > 0 at 2 bits).
        let raw: Vec<f32> = (0..max_seq)
            .flat_map(|p| (0..kvd).map(move |i| ((p * 31 + i * 7) % 13) as f32 - 6.0))
            .collect();
        for p in 0..max_seq {
            // The forward stores the raw row into the plain cache, then the
            // hook feeds the backend + flushes a closed tile (the seam's
            // exact contract).
            vals[p * kvd..(p + 1) * kvd].copy_from_slice(&raw[p * kvd..(p + 1) * kvd]);
            ValueStoreHook::value_stored(&mut st, 0, p, &mut vals);
            let t0 = p - p % tile;
            if p % tile != tile - 1 {
                // Open tile: the CURRENT tile's rows are identical to raw
                // (rows of already-CLOSED tiles are lossy by design).
                assert_eq!(
                    &vals[t0 * kvd..(p + 1) * kvd],
                    &raw[t0 * kvd..(p + 1) * kvd],
                    "row {p}: open tile must stay raw"
                );
            } else {
                // Closed tile: some row of the tile diverged from raw
                // (2-bit RTN on range ±6 ⇒ step = 4 ⇒ error > 0
                // somewhere), and telemetry counted the tile's rows.
                let mut any_diverged = false;
                for t in t0..=p {
                    if vals[t * kvd..(t + 1) * kvd] != raw[t * kvd..(t + 1) * kvd] {
                        any_diverged = true;
                    }
                }
                assert!(any_diverged, "tile at {p}: closed tile must be lossy");
                assert_eq!(st.tel[0].rows, (p + 1) as u64);
            }
        }
    }

    /// The P1 G3 class at the model-bound seam: a ZERO table makes the
    /// mean-removed arm bit-identical to the plain arm (v − 0 = v; the
    /// read add restores exactly, and the wrapper's miss path is not
    /// taken since token 0 is tracked with a zero row).
    #[test]
    fn zero_table_mean_removed_matches_plain_bitwise() {
        let kvd = 32usize;
        let tile = 4usize;
        let max_seq = 8usize;
        let table = zero_table(kvd);
        let tstatic = leak(table);
        let mut plain = VQuantState::new_plain(4, 1, kvd, max_seq, tile, tstatic);
        let mut mr = VQuantState::new_mean_removed(4, 1, kvd, max_seq, tile, tstatic);
        let mut a = vec![0.0f32; max_seq * kvd];
        let mut b = vec![0.0f32; max_seq * kvd];
        for p in 0..max_seq {
            let tok = 0u32; // tracked, zero row
            plain.set_token(p, tok);
            mr.set_token(p, tok);
            for i in 0..kvd {
                let x = ((p * 17 + i * 5) % 9) as f32 - 4.0;
                a[p * kvd + i] = x;
                b[p * kvd + i] = x;
            }
            ValueStoreHook::value_stored(&mut plain, 0, p, &mut a);
            ValueStoreHook::value_stored(&mut mr, 0, p, &mut b);
        }
        assert_eq!(a, b, "zero table: mr must be bit-identical to plain");
    }

    /// The no-op hook is a no-op (the NoVQuant overhead law, pinned).
    #[test]
    fn noop_hook_leaves_rows_untouched() {
        let mut vals = vec![1.5f32; 64];
        let snapshot = vals.clone();
        NoVQuant.value_stored(0, 3, &mut vals);
        assert_eq!(vals, snapshot);
    }
}
