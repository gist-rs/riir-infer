//! gemma2_ktov — Issue 013 T2: the K=V+ serve state over the gemma-2 f16
//! decode path. Serves `V = K + λ·E_l[s]` (katgpt-core P2,
//! [`katgpt_core::fitted_value_table::v_from_k_plus`]) by rewriting the
//! just-stored V cache row, so the existing attention kernels consume the
//! K=V+ surface unmodified — the quality ladder the deployment's
//! `VReadPath::Reconstruct` (T3) is measured against.
//!
//! ## The seam pair
//!
//! [`ValueStoreHook::keys_pre_rope`] fires between the QKV projections and
//! the in-place RoPE — exactly the Bench-004 tap point — and the state
//! copies the row (RoPE mutates `ctx.k` immediately after). The existing
//! [`ValueStoreHook::value_stored`] then rewrites row `pos` to
//! `v_from_k_plus(k_pre, row, λ_l)`. `v_from_k_plus` copies `k` and adds
//! `λ·row` with an early return at `λ = 0` and on a missing row (an
//! untracked token), so **λ = 0 and misses are the bitwise `V := K` path**
//! — the multiply is never executed (katgpt-core's G3 law, pinned by the
//! tests below at the model-bound seam).
//!
//! ## What this does NOT claim
//!
//! W_V keeps running in the forward — the FLOP/byte claim belongs to T3's
//! reconstruction lane (this is the quality ladder; the served V surface is
//! the only delta, which is exactly the deployment's quality simulation).
//! Attention scores are untouched (the K path is unchanged).
//!
//! MEASUREMENT-ONLY (the issue's gate lane): the store-time substitution is
//! bitwise the value T3's read-path reconstruction serves up to rotation
//! rounding (1.83ε); promotion is katgpt-rs-side and waits on the gates.

use katgpt_core::fitted_value_table::{v_from_k_plus, FittedTokenTable};

use crate::transformer::gemma2::ValueStoreHook;

/// Per-layer K=V+ telemetry — the energy walk the report reads.
///
/// Accumulated over every served row of the arm (arm-lifetime; `reset`
/// clears the cache and the token map, never this). All sums are f64
/// dot-accumulations of f32 rows.
#[derive(Clone, Copy, Default, Debug)]
pub struct KvLayerTel {
    /// Rows served (= decode steps through the layer).
    pub rows: u64,
    /// Rows whose token had a table row (realized coverage).
    pub rows_tracked: u64,
    /// Σ‖k_pre‖² — the served K component's energy.
    pub sum_k2: f64,
    /// Σ‖V_raw‖² — the forward's own W_V output (pre-overwrite) energy.
    pub sum_v2: f64,
    /// Σ⟨k_pre, V_raw⟩ — the K↔V alignment (cos = sum_kv/√(sum_k2·sum_v2)).
    pub sum_kv: f64,
    /// Σ‖E_l[s]‖² over tracked rows (the table row energy).
    pub sum_row2: f64,
    /// Σλ²‖E_l[s]‖² over tracked rows (the refund actually applied).
    pub sum_lrow2: f64,
}

impl KvLayerTel {
    /// Aggregate cos(K, V) over the layer's served rows (0 when empty).
    #[must_use]
    pub fn cos_kv(&self) -> f64 {
        if self.sum_k2 <= 0.0 || self.sum_v2 <= 0.0 {
            return 0.0;
        }
        self.sum_kv / (self.sum_k2 * self.sum_v2).sqrt()
    }

    /// The applied-refund share λ²‖E‖²/‖V‖² over tracked rows.
    #[must_use]
    pub fn refund_share(&self) -> f64 {
        if self.sum_v2 <= 0.0 {
            return 0.0;
        }
        self.sum_lrow2 / self.sum_v2
    }
}

/// The Issue-013 T2 [`ValueStoreHook`] state — one per eval arm.
///
/// The table borrow is `'static` BY CALLER CONTRACT (the bin `Box::leak`s
/// the frozen tables — process-lifetime by design), so no unsafe lives here.
pub struct KToVState {
    table: &'static FittedTokenTable,
    /// Per-layer λ (the schedule; mutated by the grid search between passes).
    lam: Vec<f32>,
    /// token id per position (`u32::MAX` = unset).
    tokens: Vec<u32>,
    /// The current position's pre-RoPE K row (copied at `keys_pre_rope`;
    /// strictly layer-local — the layer loop interleaves the two hook calls).
    k_pre: Vec<f32>,
    kvd: usize,
    /// Arm-lifetime per-layer telemetry.
    pub tel: Vec<KvLayerTel>,
}

impl KToVState {
    /// `lam` must have one entry per layer (the λ schedule; `vec![λ; L]`
    /// for a uniform arm).
    #[must_use]
    pub fn new(
        table: &'static FittedTokenTable,
        lam: Vec<f32>,
        n_layers: usize,
        kvd: usize,
        max_seq_len: usize,
    ) -> Self {
        assert_eq!(
            lam.len(),
            n_layers,
            "KToVState: lam schedule must cover every layer"
        );
        assert_eq!(
            table.width(),
            kvd,
            "KToVState: table width must equal kv_dim"
        );
        Self {
            table,
            lam,
            tokens: vec![u32::MAX; max_seq_len],
            k_pre: vec![0.0; kvd],
            kvd,
            tel: vec![KvLayerTel::default(); n_layers],
        }
    }

    /// Uniform-λ arm constructor.
    #[must_use]
    pub fn new_uniform(
        table: &'static FittedTokenTable,
        lam: f32,
        n_layers: usize,
        kvd: usize,
        max_seq_len: usize,
    ) -> Self {
        Self::new(table, vec![lam; n_layers], n_layers, kvd, max_seq_len)
    }

    /// Record the token at `pos` (call once per decode step, before the
    /// forward). Untracked tokens serve the bitwise `V := K` path.
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

    /// Overwrite the λ schedule in place (the grid search mutates one layer
    /// at a time between passes).
    pub fn set_lam(&mut self, layer: usize, value: f32) {
        self.lam[layer] = value;
    }

    /// The live schedule (read back by the report).
    #[must_use]
    pub fn lam(&self) -> &[f32] {
        &self.lam
    }

    /// Arm-lifetime telemetry.
    #[must_use]
    pub fn layer_tel(&self) -> &[KvLayerTel] {
        &self.tel
    }
}

impl ValueStoreHook for KToVState {
    fn keys_pre_rope(&mut self, _layer_idx: usize, _pos: usize, k_pre: &[f32]) {
        self.k_pre.copy_from_slice(k_pre);
    }

    fn value_stored(&mut self, layer_idx: usize, pos: usize, layer_values: &mut [f32]) {
        let kvd = self.kvd;
        let off = pos * kvd;
        let v = &mut layer_values[off..off + kvd];

        // Telemetry on the RAW row (pre-overwrite) — the W_V output the
        // deployment would have served.
        let tel = &mut self.tel[layer_idx];
        let tok = self.tokens.get(pos).copied().unwrap_or(u32::MAX);
        let row = self.table.row(layer_idx, tok);
        let mut k2 = 0.0f64;
        let mut v2 = 0.0f64;
        let mut kv = 0.0f64;
        for (i, x) in v.iter().enumerate().take(kvd) {
            let k = self.k_pre[i];
            k2 += (k as f64) * (k as f64);
            v2 += (*x as f64) * (*x as f64);
            kv += (k as f64) * (*x as f64);
        }
        tel.rows += 1;
        tel.sum_k2 += k2;
        tel.sum_v2 += v2;
        tel.sum_kv += kv;
        if let Some(e) = row {
            tel.rows_tracked += 1;
            let lam = f64::from(self.lam[layer_idx]);
            let mut row2 = 0.0f64;
            for &m in e.iter() {
                row2 += (m as f64) * (m as f64);
            }
            tel.sum_row2 += row2;
            tel.sum_lrow2 += lam * lam * row2;
        }

        // THE SERVE — katgpt-core P2 verbatim. λ=0 / missing row ⇒ the
        // bitwise `V := K` copy (the primitive's early return).
        let lam = self.lam[layer_idx];
        v_from_k_plus(&self.k_pre, row, lam, v);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use katgpt_core::fitted_value_table::FittedTokenTable;

    const KVD: usize = 8;
    const LAYERS: usize = 2;
    const SEQ: usize = 4;

    /// A tiny table: token 0 → zero row (the G3 fixture), token 1 → a
    /// nonzero row, token 2 → untracked.
    fn fixture_table() -> &'static FittedTokenTable {
        let row_of_token = vec![0u32, 1, u32::MAX];
        let mut data = vec![0.0f32; LAYERS * 2 * KVD];
        for l in 0..LAYERS {
            for (i, d) in data[(l * 2 + 1) * KVD..(l * 2 + 2) * KVD]
                .iter_mut()
                .enumerate()
            {
                *d = ((l * 10 + i) % 7) as f32 - 3.0;
            }
        }
        Box::leak(Box::new(FittedTokenTable::from_rows(
            LAYERS, KVD, row_of_token, data,
        )))
    }

    /// Drive one decode step through the hook pair: copy k_pre, store the
    /// raw V row, let the state serve. Returns the served row.
    fn serve<H: ValueStoreHook>(
        st: &mut H,
        layer: usize,
        pos: usize,
        k: &[f32],
        v_raw: &[f32],
    ) -> Vec<f32> {
        let mut vals = vec![0.0f32; (pos + 1) * KVD];
        vals[pos * KVD..(pos + 1) * KVD].copy_from_slice(v_raw);
        ValueStoreHook::keys_pre_rope(st, layer, pos, k);
        ValueStoreHook::value_stored(st, layer, pos, &mut vals);
        vals[pos * KVD..(pos + 1) * KVD].to_vec()
    }

    fn row_with_neg_zero(p: usize) -> Vec<f32> {
        (0..KVD)
            .map(|i| {
                if (p + i).is_multiple_of(3) {
                    -0.0 // the −0.0 class: 0·x and −0.0 + 0.0 flips its sign
                } else {
                    ((p * 13 + i * 5) % 9) as f32 - 4.0
                }
            })
            .collect()
    }

    /// G3 (the hard gate): λ=0 with a TRACKED token serves the bitwise
    /// `V := K` copy — including −0.0 entries (the multiply is never
    /// executed, katgpt-core's early return; a naive `k + 0.0·row` would
    /// flip −0.0 to +0.0 and this test would catch it).
    #[test]
    fn lambda_zero_serves_bitwise_k_tracked() {
        let table = fixture_table();
        let mut st = KToVState::new_uniform(table, 0.0, LAYERS, KVD, SEQ);
        st.set_token(0, 1); // tracked, nonzero row
        let k = row_with_neg_zero(0);
        let v_raw = vec![9.5f32; KVD];
        let served = serve(&mut st, 0, 0, &k, &v_raw);
        assert_eq!(
            served.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            k.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            "λ=0 tracked: served row must be the bitwise K copy"
        );
    }

    /// G3 (the hard gate): a λ=0 state and a direct-copy twin produce
    /// bit-identical served rows across tracked, zero-row, and untracked
    /// tokens.
    #[test]
    fn lambda_zero_matches_direct_copy_hook() {
        let table = fixture_table();
        let mut lam_state = KToVState::new_uniform(table, 0.0, LAYERS, KVD, SEQ);
        let mut copy_state = DirectCopyState::new(KVD);
        for pos in 0..SEQ {
            let tok: u32 = match pos {
                0 => 1,    // tracked, nonzero row
                1 => 0,    // tracked, ZERO row
                2 => 2,    // untracked
                _ => 999,  // out of table vocab
            };
            lam_state.set_token(pos, tok);
            copy_state.set_token(pos, tok);
            let k = row_with_neg_zero(pos);
            let v_raw: Vec<f32> = (0..KVD).map(|i| ((pos * 7 + i) % 5) as f32 - 2.0).collect();
            let a = serve(&mut lam_state, pos % LAYERS, pos, &k, &v_raw);
            let b = serve(&mut copy_state, pos % LAYERS, pos, &k, &v_raw);
            assert_eq!(
                a.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                b.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                "pos {pos}: λ=0 must be bit-identical to the V:=K copy"
            );
        }
    }

    /// λ=0.5 serves k + 0.5·row on a tracked token; an untracked token
    /// serves exact k regardless of λ.
    #[test]
    fn lambda_semantics_tracked_and_miss() {
        let table = fixture_table();
        let row: Vec<f32> = table.row(1, 1).expect("tracked").to_vec();
        let mut st = KToVState::new_uniform(table, 0.5, LAYERS, KVD, SEQ);
        st.set_token(0, 1);
        let k: Vec<f32> = (0..KVD).map(|i| (i as f32) * 0.25 - 1.0).collect();
        let served = serve(&mut st, 1, 0, &k, &[3.0f32; KVD]);
        for i in 0..KVD {
            assert!(
                (served[i] - (k[i] + 0.5 * row[i])).abs() < 1e-6,
                "λ=0.5 tracked: served[{i}] = {} want {}",
                served[i],
                k[i] + 0.5 * row[i]
            );
        }
        // Miss: untracked token serves exact k even at λ=1.
        let mut st1 = KToVState::new_uniform(table, 1.0, LAYERS, KVD, SEQ);
        st1.set_token(0, 2);
        let served = serve(&mut st1, 1, 0, &k, &[3.0f32; KVD]);
        assert_eq!(served, k, "miss at λ=1 must serve exact k");
    }

    /// Telemetry: energies land where they should (k²/v²/kv/row²/λ²row²).
    #[test]
    fn telemetry_accumulates_energies() {
        let table = fixture_table();
        let mut st = KToVState::new_uniform(table, 0.5, LAYERS, KVD, SEQ);
        st.set_token(0, 1);
        let k: Vec<f32> = vec![2.0; KVD];
        let v_raw: Vec<f32> = vec![1.0; KVD];
        let _ = serve(&mut st, 0, 0, &k, &v_raw);
        let t = st.layer_tel()[0];
        assert_eq!(t.rows, 1);
        assert_eq!(t.rows_tracked, 1);
        assert!((t.sum_k2 - 4.0 * KVD as f64).abs() < 1e-9);
        assert!((t.sum_v2 - 1.0 * KVD as f64).abs() < 1e-9);
        assert!((t.sum_kv - 2.0 * KVD as f64).abs() < 1e-9);
        // cos = 2·1/(2·1) = 1
        assert!((t.cos_kv() - 1.0).abs() < 1e-9);
        // row energy: fixture row 1 of layer 0 = (i%7)-3 → Σ = 4·1+3·4 = 16? hand-checked below
        let want: f64 = table
            .row(0, 1)
            .unwrap()
            .iter()
            .map(|&m| f64::from(m) * f64::from(m))
            .sum();
        assert!((t.sum_row2 - want).abs() < 1e-9);
        assert!((t.sum_lrow2 - 0.25 * want).abs() < 1e-9);
    }

    /// The direct `V := K` copy twin — the G3 reference path (what a
    /// deployment's `VReadPath` would serve at λ=0).
    struct DirectCopyState {
        tokens: Vec<u32>,
        k_pre: Vec<f32>,
        kvd: usize,
    }

    impl DirectCopyState {
        fn new(kvd: usize) -> Self {
            Self {
                tokens: vec![u32::MAX; SEQ],
                k_pre: vec![0.0; kvd],
                kvd,
            }
        }
        fn set_token(&mut self, pos: usize, token: u32) {
            self.tokens[pos] = token;
        }
    }

    impl ValueStoreHook for DirectCopyState {
        fn keys_pre_rope(&mut self, _l: usize, _p: usize, k_pre: &[f32]) {
            self.k_pre.copy_from_slice(k_pre);
        }
        fn value_stored(&mut self, _layer_idx: usize, pos: usize, layer_values: &mut [f32]) {
            let kvd = self.kvd;
            let off = pos * kvd;
            layer_values[off..off + kvd].copy_from_slice(&self.k_pre);
        }
    }
}
