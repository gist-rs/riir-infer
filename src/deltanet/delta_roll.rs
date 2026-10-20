//! GDN delta-roll state journal — rank-1 unwinding for GDN state checkpoints
//! (riir-infer Issue 049; the riir-infer half of katgpt-rs Research 629 E2).
//!
//! The hybrid `QwenDeltaNet` recurrence (`crate::deltanet::forward::
//! gated_deltanet_step_inplace`) carries per-head **scalar** decay, so the
//! update is a decayed rank-1 write:
//!
//! ```text
//! S_pre = g·S            (whole-matrix scale, g = decay[head] ∈ (0, 1])
//! kv    = S_pre · k      (row dots over key_dim)
//! u     = v − kv         (the β-free residual — this journal records it)
//! δ     = β·u
//! S     = S_pre + δ·kᵀ   (rank-1 outer product, row = val_dim index)
//! ```
//!
//! Each step is exactly invertible up to f32 rounding:
//!
//! ```text
//! S_prev[row, c] = (S[row, c] − δ[row]·k[c]) · (1/g)
//! ```
//!
//! Recording `u` at write time is what makes the inverse safe for ANY β: the
//! naive alternative (recovering `u` from `v` during unwind) divides by
//! `(1 − β‖k‖²)`, which is singular exactly at the normalized-key / β = 1
//! operating point GDN lives at. The β-free residual sidesteps the inversion
//! entirely.
//!
//! # Why (the consumer seams)
//!
//! The rollback-free tree-verify path deliberately never writes S₀
//! (`katgpt_core::gdn_tree_verify`); post-commit writes DO land — spec-decode
//! retry paths, `kv_evict` eviction recovery, the disaggregated handoff lane
//! (`issue028`). Today those cases have no cheap earlier-state recovery: the
//! only fallback is re-running the forward from an earlier checkpoint. The
//! journal reconstructs ANY intermediate state from the final state +
//! per-token contexts at `O(T·(K + V + 2))` bytes instead of `O(T·K·V)`
//! snapshot storage — ~64× at K = V = 128 (258 f32/token vs 16 384).
//!
//! # Laws (pinned; the tests enforce the measurable ones)
//!
//! 1. **Conditioning gate is load-bearing, not optional.** Unwinding divides
//!    by the per-head decay product over the window. Refuse to unwind past
//!    `min_head Π g < eps` (f64 product, `MIN_PRODUCT_EPS_DEFAULT` pinned
//!    from the Bench-885 drift sweep) and fall back to a materialized
//!    snapshot for the remaining depth. A journal that unwinds through a
//!    near-zero-decay stretch silently amplifies drift by `Π 1/g`. The gate
//!    is fail-closed: on refusal the state buffer is untouched. A head with
//!    `g = 0` anywhere in the window is irreversible by construction — the
//!    gate refuses (the state is provably destroyed, not drifted).
//! 2. **f32 unwind is algebraically exact, NOT bit-exact.** The forward's
//!    scale-then-add becomes subtract-then-scale; FMA-contraction and op
//!    ordering differ. The G1 gate (`tests/gdn_delta_roll_gates.rs`) measures
//!    and pins replay drift per decay regime; an f64 unwinder
//!    ([`GdnStateJournal::unwind_to_f64`]) ships as the instrument escape
//!    hatch (drift decomposition: record rounding vs unwind arithmetic).
//! 3. **Schedule-specific.** Contexts are valid ONLY under the recorded
//!    (g, β) schedule — `u` was computed against the live decayed state.
//!    Re-derivation under a different schedule needs the raw `(k, v)` rows,
//!    which the journal deliberately does not carry (`v` is consumed into
//!    `u` at record time). Pin for consumers: re-fit = new journal.
//! 4. **Post-commit only.** No interaction with the rollback-free verify
//!    invariant: verify stays write-free; the journal covers committed
//!    state only.
//! 5. **Local-only.** The reconstruction is a measured-drift recovery, not a
//!    deterministic replay — never substitute it for the raw committed state
//!    across a sync/quorum boundary (katgpt-rs Issue-940 class). A handoff
//!    consumer wanting integrity hashes the state at journal birth (BLAKE3
//!    seam; deliberately not wired in v1).
//! 6. **Zero-alloc hot path.** `with_capacity` reserves all four context
//!    planes; record + gate + unwind within capacity perform zero
//!    allocations (`tests/gdn_delta_roll_g4_alloc.rs`, own binary — the
//!    counting allocator is process-global).
//!
//! # Layout
//!
//! State is the forward's own `[n_heads · key_dim · val_dim]` slice,
//! head-major, `[val_dim × key_dim]` row-major per head (row = val index —
//! `S · k` is the val_dim-vector the step predicts with). Context planes are
//! SoA per token: `ks [H·K]`, `us [H·V]`, `betas [H]`, `decays [H]`.

/// Default conditioning-gate floor for the per-head decay product over an
/// unwind window. Pinned from the Bench-885 drift sweep: at products ≥ this
/// floor the worst measured f32 replay drift stays orders of magnitude under
/// the 1e-3 relative gate; below it the drift grows like `Π 1/g` and a
/// materialized snapshot is the honest fallback. Consumers may pass a
/// tighter/looser `eps`; the default is the measured posture.
pub const MIN_PRODUCT_EPS_DEFAULT: f32 = 0.25;

/// Fail-closed refusal from [`GdnStateJournal::unwind_to`] — the decay
/// product over the requested window dipped below `eps` for `head` (the
/// worst head observed). The state buffer is untouched when this is
/// returned.
#[derive(Debug, Clone, PartialEq)]
pub struct ConditioningRefusal {
    /// The head whose window decay product was lowest.
    pub head: usize,
    /// The requested unwind target (token index from the journal base).
    pub token: usize,
    /// That head's product over the window (f32 rendering of the f64 gate).
    pub product: f32,
    /// The floor it missed.
    pub eps: f32,
}

/// Rank-1 unwind journal over a committed GDN state sequence.
///
/// Record one context per committed step ([`record_step`](Self::record_step)
/// with the SAME `k` / `v` / `kv_mem` / `beta` / `decay` the step consumed —
/// `kv_mem` is the step's post-decay prediction, which the in-place step
/// already writes for inspection), then reconstruct any earlier state with
/// [`unwind_to`](Self::unwind_to).
#[derive(Debug, Clone)]
pub struct GdnStateJournal {
    n_heads: usize,
    key_dim: usize,
    val_dim: usize,
    /// Per token, flattened `[T · H · K]` key contexts.
    ks: Vec<f32>,
    /// Per token, flattened `[T · H · V]` β-free residuals `u = v − kv`.
    us: Vec<f32>,
    /// Per token, `[T · H]` update rates.
    betas: Vec<f32>,
    /// Per token, `[T · H]` decays `g ∈ (0, 1]`.
    decays: Vec<f32>,
}

impl GdnStateJournal {
    /// Reserves all four context planes for `n_tokens` steps up front — the
    /// zero-alloc posture for record + unwind within that window (G4).
    pub fn with_capacity(n_heads: usize, key_dim: usize, val_dim: usize, n_tokens: usize) -> Self {
        Self {
            n_heads,
            key_dim,
            val_dim,
            ks: Vec::with_capacity(n_tokens * n_heads * key_dim),
            us: Vec::with_capacity(n_tokens * n_heads * val_dim),
            betas: Vec::with_capacity(n_tokens * n_heads),
            decays: Vec::with_capacity(n_tokens * n_heads),
        }
    }

    /// Records one committed step. `k` / `v` / `kv_mem` / `beta` / `decay`
    /// are the exact arguments the step consumed; `kv_mem` is the step's
    /// post-decay prediction (`kv[row] = dot(g·S_prev[row], k)`), which
    /// [`gated_deltanet_step_inplace`](crate::deltanet::forward::
    /// gated_deltanet_step_inplace) already writes. The β-free residual
    /// `u = v − kv_mem` is derived HERE, at the schedule that produced it.
    ///
    /// Growing past `with_capacity` amortizes (a realloc, documented — never
    /// the hot-path posture).
    pub fn record_step(
        &mut self,
        k: &[f32],
        v: &[f32],
        kv_mem: &[f32],
        beta: &[f32],
        decay: &[f32],
    ) {
        let (h, kd, vd) = (self.n_heads, self.key_dim, self.val_dim);
        assert_eq!(k.len(), h * kd, "k must be [n_heads * key_dim]");
        assert_eq!(v.len(), h * vd, "v must be [n_heads * val_dim]");
        assert_eq!(kv_mem.len(), h * vd, "kv_mem must be [n_heads * val_dim]");
        assert_eq!(beta.len(), h, "beta must be [n_heads]");
        assert_eq!(decay.len(), h, "decay must be [n_heads]");
        self.ks.extend_from_slice(k);
        for hr in 0..h {
            let vr = &v[hr * vd..(hr + 1) * vd];
            let kv = &kv_mem[hr * vd..(hr + 1) * vd];
            self.us
                .extend(vr.iter().zip(kv.iter()).map(|(&vi, &ki)| vi - ki));
        }
        self.betas.extend_from_slice(beta);
        self.decays.extend_from_slice(decay);
    }

    /// Recorded steps.
    pub fn len(&self) -> usize {
        self.decays.len() / self.n_heads
    }

    /// True when no step is recorded.
    pub fn is_empty(&self) -> bool {
        self.decays.is_empty()
    }

    /// Drops all contexts (capacity retained — reuse across windows is the
    /// intended posture; a fresh schedule is a fresh journal by law 3, which
    /// `clear` + re-record satisfies: the new contexts are recorded under
    /// THEIR schedule).
    pub fn clear(&mut self) {
        self.ks.clear();
        self.us.clear();
        self.betas.clear();
        self.decays.clear();
    }

    /// `(n_heads, key_dim, val_dim)`.
    pub fn dims(&self) -> (usize, usize, usize) {
        (self.n_heads, self.key_dim, self.val_dim)
    }

    /// Journal bytes per recorded token (K + V + 2 scalars per head).
    pub fn bytes_per_token(&self) -> usize {
        (self.n_heads * (self.key_dim + self.val_dim + 2)) * std::mem::size_of::<f32>()
    }

    /// A materialized per-token snapshot's bytes (`H · K · V` f32) — the
    /// denominator of the compression claim.
    pub fn snapshot_bytes_per_token(&self) -> usize {
        self.n_heads * self.key_dim * self.val_dim * std::mem::size_of::<f32>()
    }

    /// Storage compression vs per-token materialized snapshots
    /// (`snapshot_bytes / journal_bytes`).
    pub fn compression_ratio(&self) -> f32 {
        self.snapshot_bytes_per_token() as f32 / self.bytes_per_token() as f32
    }

    /// Largest window depth `d` such that EVERY head's decay product over
    /// the last `d` recorded tokens stays ≥ `eps` — the depth
    /// [`unwind_to`](Self::unwind_to) is guaranteed to accept (to target
    /// `len − d`). f64 products, per-head walks, zero alloc.
    pub fn window_depth_limit(&self, eps: f32) -> usize {
        debug_assert!(eps > 0.0, "eps must be positive");
        let len = self.len();
        let h = self.n_heads;
        let mut limit = len;
        for head in 0..h {
            let mut p = 1.0f64;
            let mut d = 0usize;
            while d < limit {
                p *= self.decays[(len - 1 - d) * h + head] as f64;
                if p < eps as f64 {
                    break;
                }
                d += 1;
            }
            limit = limit.min(d);
            if limit == 0 {
                return 0;
            }
        }
        limit
    }

    /// Reconstructs the state as it stood `target` recorded steps ago
    /// (`target == len` → no-op `Ok(0)`; `target == 0` → the base state),
    /// unwinding from the CURRENT contents of `state` — which MUST be the
    /// state after all `len` recorded steps (the journal does not
    /// fingerprint the buffer; law 5 names the integrity seam).
    ///
    /// Fail-closed: the conditioning gate runs BEFORE any write, so an
    /// [`Err`] return leaves `state` byte-identical.
    pub fn unwind_to(
        &self,
        state: &mut [f32],
        target: usize,
        eps: f32,
    ) -> Result<usize, ConditioningRefusal> {
        assert_eq!(
            state.len(),
            self.n_heads * self.key_dim * self.val_dim,
            "state must be [n_heads * key_dim * val_dim]"
        );
        debug_assert!(eps > 0.0, "eps must be positive");
        assert!(target <= self.len(), "target beyond the journal base");
        let steps = self.len() - target;
        if steps == 0 {
            return Ok(0);
        }
        self.check_conditioning(target, eps)?;
        for t in (target..self.len()).rev() {
            self.unwind_token_f32(state, t);
        }
        Ok(steps)
    }

    /// The f64 escape hatch (law 2): same unwinding against an f64 state
    /// buffer — isolates unwind-arithmetic drift from f32 record rounding in
    /// drift studies. The contexts themselves are f32 (law 3's record
    /// rounding stays).
    pub fn unwind_to_f64(
        &self,
        state: &mut [f64],
        target: usize,
    ) -> Result<usize, ConditioningRefusal> {
        assert_eq!(
            state.len(),
            self.n_heads * self.key_dim * self.val_dim,
            "state must be [n_heads * key_dim * val_dim]"
        );
        assert!(target <= self.len(), "target beyond the journal base");
        let steps = self.len() - target;
        if steps == 0 {
            return Ok(0);
        }
        // The gate itself: f64 products against the DEFAULT floor (the f64
        // unwinder is the instrument posture — the caller chose the depth).
        self.check_conditioning(target, MIN_PRODUCT_EPS_DEFAULT)?;
        let (h, kd, vd) = (self.n_heads, self.key_dim, self.val_dim);
        for t in (target..self.len()).rev() {
            let base_s = t * h;
            let base_k = t * h * kd;
            let base_u = t * h * vd;
            for head in 0..h {
                let g = self.decays[base_s + head] as f64;
                let b = self.betas[base_s + head] as f64;
                let inv_g = 1.0 / g;
                let k_h = &self.ks[base_k + head * kd..base_k + (head + 1) * kd];
                let u_h = &self.us[base_u + head * vd..base_u + (head + 1) * vd];
                let s_h = &mut state[head * kd * vd..(head + 1) * kd * vd];
                for row in 0..vd {
                    let d = b * u_h[row] as f64;
                    let s_row = &mut s_h[row * kd..(row + 1) * kd];
                    for (s, &kc) in s_row.iter_mut().zip(k_h.iter()) {
                        *s = (*s - d * kc as f64) * inv_g;
                    }
                }
            }
        }
        Ok(steps)
    }

    /// The gate: per-head f64 product over `[target, len)`; refuses with the
    /// worst head. Zero alloc (per-head walk, early-exit).
    fn check_conditioning(&self, target: usize, eps: f32) -> Result<(), ConditioningRefusal> {
        let h = self.n_heads;
        let eps64 = eps as f64;
        let mut worst_head = usize::MAX;
        let mut worst_prod = f64::INFINITY;
        for head in 0..h {
            let mut p = 1.0f64;
            for t in (target..self.len()).rev() {
                p *= self.decays[t * h + head] as f64;
                if p < eps64 {
                    break;
                }
            }
            if p < eps64 && p < worst_prod {
                worst_prod = p;
                worst_head = head;
            }
        }
        if worst_head != usize::MAX {
            return Err(ConditioningRefusal {
                head: worst_head,
                token: target,
                product: worst_prod as f32,
                eps,
            });
        }
        Ok(())
    }

    /// Peels ONE token (index `t`) off `state` — the exact inverse of the
    /// forward's scale-then-add as subtract-then-scale, per row fused.
    fn unwind_token_f32(&self, state: &mut [f32], t: usize) {
        let (h, kd, vd) = (self.n_heads, self.key_dim, self.val_dim);
        let base_s = t * h;
        let base_k = t * h * kd;
        let base_u = t * h * vd;
        for head in 0..h {
            let g = self.decays[base_s + head];
            let b = self.betas[base_s + head];
            let inv_g = 1.0 / g;
            let k_h = &self.ks[base_k + head * kd..base_k + (head + 1) * kd];
            let u_h = &self.us[base_u + head * vd..base_u + (head + 1) * vd];
            let s_h = &mut state[head * kd * vd..(head + 1) * kd * vd];
            for row in 0..vd {
                let d = b * u_h[row];
                let s_row = &mut s_h[row * kd..(row + 1) * kd];
                for (s, &kc) in s_row.iter_mut().zip(k_h.iter()) {
                    *s = (*s - d * kc) * inv_g;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests_delta_roll {
    use super::*;

    /// Single-head, single-token, unit decay: the unwind is the exact
    /// arithmetic inverse up to one rounding — drift must be at the
    /// float-epsilon scale, not the accumulated scale.
    #[test]
    fn single_token_unwind_is_epsilon_exact() {
        const H: usize = 1;
        const K: usize = 8;
        const V: usize = 8;
        let state = vec![0.5f32; H * K * V];
        let k = vec![0.25f32; H * K];
        let v = vec![1.0f32; H * V];
        let beta = [0.7f32];
        let decay = [1.0f32];

        // Forward one step (the forward's own math, inlined).
        let mut kv_mem = vec![0.0f32; H * V];
        for hr in 0..H {
            for row in 0..V {
                let s_row = &state[row * K..(row + 1) * K];
                let k_h = &k[hr * K..(hr + 1) * K];
                let kv: f32 = s_row.iter().zip(k_h.iter()).map(|(&s, &kc)| s * kc).sum();
                kv_mem[hr * V + row] = kv;
            }
        }
        let mut fwd = state.clone();
        for hr in 0..H {
            let g = decay[hr];
            let b = beta[hr];
            let k_h = &k[hr * K..(hr + 1) * K];
            for row in 0..V {
                let d = b * (v[hr * V + row] - kv_mem[hr * V + row]);
                let s_row = &mut fwd[row * K..(row + 1) * K];
                for (s, &kc) in s_row.iter_mut().zip(k_h.iter()) {
                    *s = *s * g + d * kc;
                }
            }
        }

        let mut journal = GdnStateJournal::with_capacity(H, K, V, 1);
        journal.record_step(&k, &v, &kv_mem, &beta, &decay);
        let steps = journal
            .unwind_to(&mut fwd, 0, MIN_PRODUCT_EPS_DEFAULT)
            .expect("gate");
        assert_eq!(steps, 1);
        let max_err = fwd
            .iter()
            .zip(state.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < 1e-4, "single-token unwind drift {max_err}");
    }

    /// Shape misuse is a loud assert, never a silent mis-journal.
    #[test]
    #[should_panic(expected = "kv_mem")]
    fn record_step_rejects_shape_mismatch() {
        let mut j = GdnStateJournal::with_capacity(2, 4, 4, 1);
        j.record_step(&[0.0; 8], &[0.0; 8], &[0.0; 4], &[0.5, 0.5], &[0.9, 0.9]);
    }

    /// `target == len` is a no-op; `target > len` asserts.
    #[test]
    fn unwind_target_bounds() {
        let mut j = GdnStateJournal::with_capacity(1, 4, 4, 2);
        let kv = [0.0f32; 4];
        j.record_step(&[0.1; 4], &[0.2; 4], &kv, &[0.5], &[0.99]);
        let mut state = vec![0.0f32; 16];
        assert_eq!(
            j.unwind_to(&mut state, 1, MIN_PRODUCT_EPS_DEFAULT).unwrap(),
            0
        );
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _ = j.unwind_to(&mut state, 2, MIN_PRODUCT_EPS_DEFAULT);
            }))
            .is_err()
        );
    }
}
