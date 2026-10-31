//! riir-ai Plan 644 T1 — the speed-balanced ANE‖GPU CHANNEL split of one
//! projection site.
//!
//! At f=1 the ANE took a whole seam and the GPU idled through the eval —
//! the measured 0.94× (Bench 1150; Issue 887 §2026-10-31). Here the output
//! channels of ONE projection split by rate: the ANE evaluates rows
//! `[0, k)` (one Form C program at `block_tokens`), the GPU evaluates rows
//! `[k, n)` from its own suffix [`TernaryHandle`] concurrently, and both
//! land in the same token-major `[P × n]` output — the ANE through a
//! strided [`super::exec_zc::ZcSegment`], the GPU through a scratch +
//! [`crate::elementwise_cubecl::CopyColsCubeCL`] column scatter.
//!
//! A split is a property of the WEIGHT (it rides the handle like the
//! zero-copy cache), so every `prefill_project` site takes it with no
//! per-site code and no basis logic: the site already passes the right
//! input (rotated on folded files — the map `y = W·x` is the same on both
//! engines; Plan 644 T0).

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use katgpt_core::TernaryGroupWeights;

use super::bridge::BridgeKernel;
use super::requant::requant_per_row_int8;
use crate::gemv_ternary_cubecl::TernaryHandle;

/// One registered site: the ANE prefix program + the GPU suffix rows.
pub struct AneChannelSplit {
    /// Form C program over rows `[0, k)` at `block_tokens` tokens.
    pub kernel: Arc<BridgeKernel>,
    /// GPU rows `[k, n)`; carries its own zero-copy cache when the tensor
    /// path is armed (filled by the forward, which owns the GEMM context).
    pub suffix: TernaryHandle,
    /// ANE row count (a multiple of [`super::ANE_PREFILL_CHANNEL_GRID`]).
    pub k: usize,
    /// Total output channels.
    pub n: usize,
    /// Input channels.
    pub ic: usize,
    pub block_tokens: usize,
}

static ARMED: AtomicBool = AtomicBool::new(false);
/// f64 bits of the ANE row fraction; 0 = unset (use [`DEFAULT_FRACTION`]).
static FRACTION_BITS: AtomicU64 = AtomicU64::new(0);

/// The IN-SITU optimum (Plan 644 T3, async lane, Bonsai-2 @2048, M5 Ultra):
/// f sweep 0.20/0.25/0.30/0.34/0.40 → 1.063/1.077/1.099/1.092/1.053×. The
/// isolated-rate prediction (a/(a+g) ≈ 0.40) overshoots because the ANE
/// slows ≈1.30× while the GPU runs (1.82 vs 1.406 ms/1k-ch). Plan 644 T2
/// replaces this with a per-shape in-situ calibration at registration.
pub const DEFAULT_FRACTION: f64 = 0.30;

/// Arm the split at registration AND dispatch (set before construction).
/// Default OFF — the GOAT gate (Plan 644 T4) promotes it.
pub fn set_prefill_ane_channel_split(on: bool) {
    ARMED.store(on, Ordering::Relaxed);
}

pub fn prefill_ane_channel_split() -> bool {
    ARMED.load(Ordering::Relaxed)
}

/// Override the ANE row fraction (clamped to `[0, 1]`).
pub fn set_prefill_ane_channel_split_fraction(f: f64) {
    FRACTION_BITS.store(f.clamp(0.0, 1.0).to_bits(), Ordering::Relaxed);
}

pub fn prefill_ane_channel_split_fraction() -> f64 {
    match FRACTION_BITS.load(Ordering::Relaxed) {
        0 => DEFAULT_FRACTION,
        bits => f64::from_bits(bits),
    }
}

/// Plan 644 T3 diagnostic: `RIIR_P644_GPU_HALF_ONLY=1` runs only the GPU
/// suffix + scatter of each split (ANE columns unwritten — a PRICING probe,
/// its logits are wrong by design and G1 must fail under it).
pub fn probe_gpu_half_only() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("RIIR_P644_GPU_HALF_ONLY").as_deref() == Ok("1"))
}

/// ANE rows for an `n`-channel site at `fraction`: rounded to the channel
/// grid; `None` when the split degenerates (no rows to either engine — that
/// is how a site the ANE cannot win, e.g. ffn_down's ic-17408 cliff, falls
/// out with no special case).
pub fn split_rows(n: usize, fraction: f64) -> Option<usize> {
    let grid = super::ANE_PREFILL_CHANNEL_GRID;
    let k = ((n as f64 * fraction) / grid as f64).round() as usize * grid;
    match k {
        0 => None,
        k if k >= n => None,
        k => Some(k),
    }
}

/// Rows `[start, end)` of a row-major ternary matrix (contiguous prefix
/// copies of all three planes — 887 finding 4).
pub fn row_range(w: &TernaryGroupWeights, start: usize, end: usize) -> TernaryGroupWeights {
    debug_assert!(start < end && end <= w.rows);
    TernaryGroupWeights {
        rows: end - start,
        cols: w.cols,
        blocks64: w.blocks64,
        groups_per_row: w.groups_per_row,
        pos_bits: w.pos_bits[start * w.blocks64..end * w.blocks64].to_vec(),
        neg_bits: w.neg_bits[start * w.blocks64..end * w.blocks64].to_vec(),
        group_scale: w.group_scale[start * w.groups_per_row..end * w.groups_per_row].to_vec(),
    }
}

/// Build one site's split: requant + compile ONLY the prefix rows (the
/// requant is per-row, so the suffix is never touched on the ANE side) and
/// upload the suffix rows. `Ok(None)` = degenerate split (stay GPU-only).
pub fn build(
    client: &cubecl::client::ComputeClient<crate::cubecl_runtime::ActiveRuntime>,
    w: &TernaryGroupWeights,
    fraction: f64,
    block_tokens: usize,
) -> Result<Option<AneChannelSplit>, String> {
    let Some(k) = split_rows(w.rows, fraction) else {
        return Ok(None);
    };
    let (q, ws) = requant_per_row_int8(
        &w.pos_bits[..k * w.blocks64],
        &w.neg_bits[..k * w.blocks64],
        &w.group_scale[..k * w.groups_per_row],
        k,
        w.cols,
    );
    let kernel = BridgeKernel::compile_form_c(w.cols, k, block_tokens, &q, &ws)?;
    Ok(Some(AneChannelSplit {
        kernel: Arc::new(kernel),
        suffix: TernaryHandle::from_weights(client, &row_range(w, k, w.rows)),
        k,
        n: w.rows,
        ic: w.cols,
        block_tokens,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_rows_rounds_to_grid_and_degenerates_loudly() {
        assert_eq!(split_rows(10240, 0.40), Some(4096));
        assert_eq!(split_rows(17408, 0.40), Some(6976));
        assert_eq!(split_rows(5120, 0.0), None);
        assert_eq!(split_rows(5120, 1.0), None);
        assert_eq!(split_rows(64, 0.2), None);
    }

    #[test]
    fn row_range_is_the_contiguous_slice() {
        let mut w = TernaryGroupWeights::new(4, 128);
        for r in 0..4 {
            w.set(r, r, 1);
            w.set(r, 127 - r, -1);
            w.set_scale(r, 0, 0.5 + r as f32);
        }
        let s = row_range(&w, 1, 3);
        assert_eq!((s.rows, s.cols), (2, 128));
        for r in 0..2 {
            for c in 0..128 {
                assert_eq!(s.get(r, c), w.get(r + 1, c));
            }
            assert_eq!(s.scale_at(r, 0), w.scale_at(r + 1, 0));
        }
    }
}
