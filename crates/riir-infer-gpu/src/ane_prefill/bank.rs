//! Issue 726 T3 — the per-layer ANE program bank.
//!
//! One Form C program per (GDN layer, op): requant the fused ternary
//! weights per-row int8 ([`super::requant`]) + compile through the ObjC
//! bridge at the fixed block shape `[1, ic, 1, block]` (tokens along W).
//! Single procedure per program (T0 P6: multi-procedure banks are REJECTED
//! by ANECCompile on this macOS; eager-compile cost is 0.69-1.20 s/program
//! at REAL Bonsai dims (P9) — the 96-program bank compiles in ≈ 90 s at
//! model load (P8's 0.028 s/program was toy dims).
//!
//! Row-concat requant invariant: `s_row` is PER ROW, so requanting each
//! split weight and concatenating the (q, ws) ROWS is identical to
//! requanting the row-concatenated fused matrix — no merged bit-planes are
//! ever materialized.
//!
//! The memory budget (issue §"Numerics + memory"): ANE-side int8 copies are
//! ~3.8× the Q2_0 originals; the bank enforces a byte ceiling at
//! registration (default 13 GB — the f=1.0 worst case is 12.6 GB for
//! Bonsai's 48 GDN layers × 262.7 M weights) + refuses over budget
//! (fail-open to the GPU path).

#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

use std::sync::Arc;

use katgpt_core::TernaryGroupWeights;

use super::AnePrefillOp;
use super::bridge::BridgeKernel;
use super::requant::requant_per_row_int8;

/// The f=1.0 worst case for Bonsai (48 GDN layers × 262.7 M weights × 1 B)
/// plus slack — the T0 budget-math ceiling.
pub const DEFAULT_MAX_ANE_BYTES: u64 = 13_000_000_000;

/// Op slot index inside the per-layer kernel triple.
const OP_INPROJ: usize = 0;
const OP_GATE_UP: usize = 1;
const OP_DOWN: usize = 2;

fn op_slot(op: AnePrefillOp) -> usize {
    match op {
        AnePrefillOp::InProjConcat => OP_INPROJ,
        AnePrefillOp::GateUpProj => OP_GATE_UP,
        AnePrefillOp::DownProj => OP_DOWN,
    }
}

/// The byte cost of a registration: the sum of the CALLER's slice heights
/// × the shared cols — nothing here re-inflates to full width. Issue 887
/// T4's bank-layer contract, extracted as the single accounting site so it
/// is assertable headless (registration proper is compile-bound to the
/// ANE bridge): 887 T1(a)'s seam-level slicing and 886 T3's caller-side
/// slice views both lean on this staying slice-faithful.
fn split_bytes(splits: &[&TernaryGroupWeights]) -> u64 {
    splits.iter().map(|s| (s.rows * s.cols) as u64).sum()
}

/// The compiled program bank. `[layer][op_slot]` — `None` for attention
/// layers (no in_proj), unregistered slots, or a per-op fail-open (the
/// Plan 549 `try_register_splits` path for `DownProj`).
pub struct AneProgramBank {
    kernels: Vec<[Option<Arc<BridgeKernel>>; 3]>,
    /// Fixed block width in tokens (2048).
    pub block_tokens: usize,
    bytes_total: u64,
    max_bytes: u64,
    /// One-shot guard so a budget-exhausted lenient registration (e.g. all
    /// 48 `DownProj` layers) logs ONCE, not per layer.
    budget_warned: std::sync::atomic::AtomicBool,
}

impl AneProgramBank {
    pub fn new(n_layers: usize, block_tokens: usize, max_bytes: u64) -> Self {
        Self {
            kernels: vec![[None, None, None]; n_layers],
            block_tokens,
            bytes_total: 0,
            max_bytes,
            budget_warned: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// Requant (row-concat over `splits`, in order) + compile one
    /// (layer, op) program. Errors fail the whole bank at `finalize` time
    /// (all-or-nothing — the gate stays simple + honest).
    pub fn register_splits(
        &mut self,
        layer_idx: usize,
        op: AnePrefillOp,
        splits: &[&TernaryGroupWeights],
    ) -> Result<(), String> {
        self.register_splits_inner(layer_idx, op, splits, false)
    }

    /// Plan 549: the per-op fail-open register — budget/compile failures
    /// leave the slot empty with ONE log line instead of poisoning the bank
    /// (the op's seam call site fail-opens to the GPU per layer). Only
    /// caller-contract errors (double registration) still poison.
    pub fn try_register_splits(
        &mut self,
        layer_idx: usize,
        op: AnePrefillOp,
        splits: &[&TernaryGroupWeights],
    ) -> Result<(), String> {
        self.register_splits_inner(layer_idx, op, splits, true)
    }

    fn register_splits_inner(
        &mut self,
        layer_idx: usize,
        op: AnePrefillOp,
        splits: &[&TernaryGroupWeights],
        lenient: bool,
    ) -> Result<(), String> {
        let slot = op_slot(op);
        if self.kernels[layer_idx][slot].is_some() {
            return Err(format!("layer {layer_idx} {op:?} registered twice"));
        }
        if splits.is_empty() {
            return Err(format!("layer {layer_idx} {op:?}: no splits"));
        }
        // All splits share cols (the input dim); rows concatenate.
        let cols = splits[0].cols;
        if let Some(bad) = splits.iter().find(|s| s.cols != cols) {
            return Err(format!(
                "layer {layer_idx} {op:?}: split cols mismatch ({} vs {cols})",
                bad.cols
            ));
        }
        let rows_total: usize = splits.iter().map(|s| s.rows).sum();
        let mut q = Vec::with_capacity(rows_total * cols);
        let mut ws = Vec::with_capacity(rows_total);
        for s in splits {
            let (sq, sws) =
                requant_per_row_int8(&s.pos_bits, &s.neg_bits, &s.group_scale, s.rows, s.cols);
            q.extend_from_slice(&sq);
            ws.extend_from_slice(&sws);
        }
        let bytes = split_bytes(splits);
        if self.bytes_total + bytes > self.max_bytes {
            let msg = format!(
                "ANE memory budget exceeded: {} + {bytes} > {} bytes",
                self.bytes_total, self.max_bytes
            );
            if lenient {
                if !self
                    .budget_warned
                    .swap(true, std::sync::atomic::Ordering::Relaxed)
                {
                    eprintln!("[ane] {op:?} skipped for remaining layers (budget): {msg} — raise RIIR_ANE_MAX_BYTES to enable");
                }
                return Ok(());
            }
            return Err(msg);
        }
        match BridgeKernel::compile_form_c(cols, rows_total, self.block_tokens, &q, &ws) {
            Ok(kernel) => {
                self.bytes_total += bytes;
                self.kernels[layer_idx][slot] = Some(Arc::new(kernel));
                Ok(())
            }
            Err(e) => {
                let msg = format!("layer {layer_idx} {op:?}: {e}");
                if lenient {
                    eprintln!("[ane] {op:?} skipped (compile): {msg}");
                    return Ok(());
                }
                Err(msg)
            }
        }
    }

    /// The compiled kernel for one (layer, op), if registered.
    pub fn kernel(&self, layer_idx: usize, op: AnePrefillOp) -> Option<&Arc<BridgeKernel>> {
        self.kernels.get(layer_idx)?.get(op_slot(op))?.as_ref()
    }

    /// Every GDN layer must have both ops — otherwise the bank is partial
    /// and the ctx must NOT report Ready (a mid-model ANE miss would
    /// silently mix paths layer by layer — the Issue 066 class).
    pub fn covers(&self, gdn_layers: &[usize]) -> bool {
        gdn_layers.iter().all(|&l| {
            self.kernels
                .get(l)
                .is_some_and(|pair| pair[OP_INPROJ].is_some() && pair[OP_GATE_UP].is_some())
        })
    }

    /// Total int8 weight bytes compiled into the bank (budget evidence for
    /// T6's runtime enforcement report).
    pub fn bytes_total(&self) -> u64 {
        self.bytes_total
    }

    /// The configured byte ceiling (the other half of the T6 budget
    /// report — enforcement happens at registration).
    pub fn max_bytes(&self) -> u64 {
        self.max_bytes
    }
}

impl Clone for AneProgramBank {
    fn clone(&self) -> Self {
        Self {
            kernels: self.kernels.clone(),
            block_tokens: self.block_tokens,
            bytes_total: self.bytes_total,
            max_bytes: self.max_bytes,
            budget_warned: std::sync::atomic::AtomicBool::new(self.budget_warned.load(
                std::sync::atomic::Ordering::Relaxed,
            )),
        }
    }
}

impl std::fmt::Debug for AneProgramBank {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let registered = self
            .kernels
            .iter()
            .filter(|p| p[0].is_some() && p[1].is_some())
            .count();
        write!(
            f,
            "AneProgramBank {{ layers_ready: {registered}/{}, block_tokens: {}, bytes: {} }}",
            self.kernels.len(),
            self.block_tokens,
            self.bytes_total
        )
    }
}

#[cfg(test)]
mod tests {
    use super::split_bytes;
    use katgpt_core::TernaryGroupWeights;

    /// Issue 887 T4, bank layer (contract pin, 2026-09-10): the byte cost
    /// of a registration is the sum of the CALLER's slice rows × cols — a
    /// half-row slice costs half the bytes, and nothing in the accounting
    /// re-inflates to full width. Registration proper is compile-bound to
    /// the ANE bridge (device work, post-@32K sequencing per the issue),
    /// so the assertable-now contract is this function; 887 T1(a)'s
    /// seam-level slicing and 886 T3's caller-side slice views both lean
    /// on it staying slice-faithful.
    #[test]
    fn bank_byte_accounting_is_slice_faithful() {
        let cols = 128usize;
        let full = TernaryGroupWeights::new(96, cols);
        let half = TernaryGroupWeights::new(48, cols);
        assert_eq!(split_bytes(&[&full]), (96 * cols) as u64);
        assert_eq!(split_bytes(&[&half]), (48 * cols) as u64);
        assert_eq!(split_bytes(&[&full]), 2 * split_bytes(&[&half]));
        // Multi-split registration (the row-concat contract): bytes sum.
        assert_eq!(split_bytes(&[&half, &half]), split_bytes(&[&full]));
        // Empty registration is free (rejected upstream as an error, but
        // the accounting must not invent bytes for it).
        let empty: [&TernaryGroupWeights; 0] = [];
        assert_eq!(split_bytes(&empty), 0);
    }
}
