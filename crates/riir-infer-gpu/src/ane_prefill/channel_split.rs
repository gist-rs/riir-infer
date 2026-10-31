//! riir-ai Plan 644 T1 — the speed-balanced ANE‖GPU CHANNEL split of one
//! projection site.
//!
//! At f=1 the ANE took a whole seam and the GPU idled through the eval —
//! the measured 0.94× (Bench 1150; Issue 887 §2026-10-31). Here the output
//! channels of ONE projection split by rate: the ANE evaluates rows
//! `[0, k)` (one Form C program at `block_tokens`), the GPU evaluates rows
//! `[k, n)` from an offset view of the resident [`TernaryHandle`]
//! concurrently (riir-ai Issue 1147 B7 — no second copy), and both
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
use std::time::{Duration, Instant};

use katgpt_core::TernaryGroupWeights;

use super::bridge::BridgeKernel;
use super::requant::requant_per_row_int8;
use crate::gemv_ternary_cubecl::TernaryHandle;

/// One projection of a split: its ANE prefix rows `[0, k)` ride the shared
/// program; its GPU suffix rows `[k, n)` are this handle (a row view of the
/// full handle — see [`TernaryHandle::row_view`]).
pub struct SplitPart {
    /// GPU rows `[k, n)`; carries its own zero-copy cache when the tensor
    /// path is armed (filled by the forward, which owns the GEMM context).
    pub suffix: TernaryHandle,
    /// ANE row count (a multiple of [`super::ANE_PREFILL_CHANNEL_GRID`]).
    pub k: usize,
    /// Total output channels.
    pub n: usize,
}

/// One registered split: ONE Form C program over the CONCATENATED prefix
/// rows of every part (Plan 644 T1b — projections sharing an input, e.g.
/// gate+up or qkv+z, fuse into one program: the ANE holds only ~120 loaded
/// programs) + one GPU suffix per part. A single-projection split is the
/// 1-part case.
pub struct AneChannelSplit {
    /// Form C program over `Σ k` rows at `block_tokens` tokens; part i's
    /// rows start at channel `Σ_{j<i} k_j`.
    pub kernel: Arc<BridgeKernel>,
    pub parts: Vec<SplitPart>,
    /// Input channels (shared by every part).
    pub ic: usize,
    pub block_tokens: usize,
}

impl AneChannelSplit {
    /// Largest GPU suffix width — sizes the forward's shared scratch.
    pub fn max_suffix_rows(&self) -> usize {
        self.parts.iter().map(|p| p.n - p.k).max().unwrap_or(0)
    }
}

static ARMED: AtomicBool = AtomicBool::new(false);
/// f64 bits of the ANE row fraction; 0 = unset (use [`DEFAULT_FRACTION`]).
static FRACTION_BITS: AtomicU64 = AtomicU64::new(0);

/// The IN-SITU optimum (Plan 644 T1b, FUSED gate+up / qkv+z programs,
/// async lane, Bonsai-2 @2048, M5 Ultra): f sweep 0.25/0.30/0.35/0.40/0.45
/// → 1.111/1.141/1.161/1.119/1.052×. (Unfused 1-part sites peaked at 0.30;
/// the isolated-rate prediction a/(a+g) ≈ 0.40 overshoots because the ANE
/// slows ≈1.30× while the GPU runs.) Plan 644 T2 replaces this with a
/// per-shape in-situ calibration at registration.
pub const DEFAULT_FRACTION: f64 = 0.35;

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

/// Tokens of a `p`-token chunk the split takes: the whole `block`s (the
/// ANE program's fixed width); the `p % block` tail runs full-width on the
/// GPU (riir-ai Issue 1147 B4). 0 = no split.
pub fn split_head_tokens(p: usize, block: usize) -> usize {
    p / block * block
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

/// Where registration time goes, summed over every [`build`] call (riir-ai
/// Issue 1147 B2 — the load-time cost of the split is the promotion blocker,
/// so the registration line prints this breakdown, never one opaque total).
#[derive(Debug, Default, Clone, Copy)]
pub struct BuildTiming {
    /// Ternary → per-row int8 requant of the ANE prefix rows (CPU).
    pub requant: Duration,
    /// MIL + blob packing, the bridge compile (or cache restore) and load.
    pub compile: Duration,
    /// The GPU suffix handles (a view costs ~nothing; a copy is an upload).
    pub upload: Duration,
    /// Device bytes the suffixes SPAN as offset views of the resident full
    /// handles — zero new bytes (riir-ai Issue 1147 B7).
    pub suffix_viewed_bytes: u64,
    /// Device bytes uploaded as suffix COPIES (a view refused alignment) —
    /// the split's real resident price on top of the GPU-only model.
    pub suffix_copied_bytes: u64,
}

/// Build one split over `parts` — each the host weights of a projection and
/// its RESIDENT full GPU handle, all sharing `cols` and the input: requant +
/// compile ONLY the concatenated prefix rows (per-row requant ⇒ the suffix
/// is never touched on the ANE side); each GPU suffix is an offset view of
/// the full handle (copied only when the view refuses alignment). `Ok(None)`
/// = some part degenerates (the group stays GPU-only).
pub fn build(
    client: &cubecl::client::ComputeClient<crate::cubecl_runtime::ActiveRuntime>,
    parts: &[(&TernaryGroupWeights, &TernaryHandle)],
    fraction: f64,
    block_tokens: usize,
    backoff_ms: &[u64],
    timing: &mut BuildTiming,
) -> Result<Option<AneChannelSplit>, String> {
    let Some(&(first, _)) = parts.first() else {
        return Ok(None);
    };
    let ws: Vec<&TernaryGroupWeights> = parts.iter().map(|&(w, _)| w).collect();
    if ws.iter().any(|w| w.cols != first.cols) {
        return Err("channel split group: parts disagree on input width".into());
    }
    let Some(ks) = ws
        .iter()
        .map(|w| split_rows(w.rows, fraction))
        .collect::<Option<Vec<usize>>>()
    else {
        return Ok(None);
    };
    let k_total: usize = ks.iter().sum();
    let (mut pos, mut neg, mut scale) = (Vec::new(), Vec::new(), Vec::new());
    for (w, &k) in ws.iter().zip(&ks) {
        pos.extend_from_slice(&w.pos_bits[..k * w.blocks64]);
        neg.extend_from_slice(&w.neg_bits[..k * w.blocks64]);
        scale.extend_from_slice(&w.group_scale[..k * w.groups_per_row]);
    }
    let t = Instant::now();
    let (q, wscale) = requant_per_row_int8(&pos, &neg, &scale, k_total, first.cols);
    timing.requant += t.elapsed();
    let t = Instant::now();
    let kernel = BridgeKernel::compile_form_c_with_backoff(
        first.cols,
        k_total,
        block_tokens,
        &q,
        &wscale,
        backoff_ms,
    );
    timing.compile += t.elapsed();
    let kernel = kernel?;
    let t = Instant::now();
    let parts = parts
        .iter()
        .zip(&ks)
        .map(|(&(w, full), &k)| {
            debug_assert_eq!((full.m, full.n), (w.rows, w.cols));
            let suffix = match full.row_view(k) {
                Some(view) => {
                    timing.suffix_viewed_bytes += view.resident_bytes();
                    view
                }
                None => {
                    let copy = TernaryHandle::from_weights(client, &row_range(w, k, w.rows));
                    timing.suffix_copied_bytes += copy.resident_bytes();
                    copy
                }
            };
            SplitPart {
                suffix,
                k,
                n: w.rows,
            }
        })
        .collect();
    timing.upload += t.elapsed();
    Ok(Some(AneChannelSplit {
        kernel: Arc::new(kernel),
        parts,
        ic: first.cols,
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
    fn split_head_takes_whole_blocks_only() {
        assert_eq!(split_head_tokens(2048, 2048), 2048);
        assert_eq!(split_head_tokens(3000, 2048), 2048);
        assert_eq!(split_head_tokens(4095, 2048), 2048);
        assert_eq!(split_head_tokens(2047, 2048), 0);
        assert_eq!(split_head_tokens(1, 2048), 0);
    }

    /// Issue 1147 B7: the suffix VIEW must dispatch bit-identically to the
    /// suffix COPY it replaces, report the bytes it spans, and refuse an
    /// offset wgpu cannot bind (the caller then copies).
    #[test]
    fn row_view_matches_the_suffix_copy_bit_identically() {
        use crate::cubecl_runtime::ActiveRuntime;
        use crate::gemv_ternary_cubecl::GemvTernaryCubeCL;
        use cubecl::prelude::*;

        let ctx = crate::context::GpuContext::new().expect("GpuContext should init");
        let client = ctx.cubecl_client();
        let (rows, cols, k) = (256usize, 512usize, 128usize);
        let mut w = TernaryGroupWeights::new(rows, cols);
        for r in 0..rows {
            for c in 0..cols {
                w.set(r, c, [1i8, 0, -1][(r * 7 + c * 3) % 3]);
            }
            for g in 0..w.groups_per_row {
                w.set_scale(r, g, 0.25 + 0.01 * ((r + g) % 17) as f32);
            }
        }
        let x: Vec<f32> = (0..cols)
            .map(|i| ((i * 37) % 101) as f32 * 0.01 - 0.5)
            .collect();
        let gemv = |h: &TernaryHandle| {
            let input = client.create_from_slice(f32::as_bytes(&x));
            let out = client.empty(h.m * std::mem::size_of::<f32>());
            unsafe {
                GemvTernaryCubeCL::launch::<ActiveRuntime>(&client, h, input, out.clone());
            }
            f32::from_bytes(&client.read_one(out).expect("read")).to_vec()
        };

        let full = TernaryHandle::from_weights(&client, &w);
        let copy = TernaryHandle::from_weights(&client, &row_range(&w, k, rows));
        let view = full.row_view(k).expect("256-aligned offsets must view");
        assert_eq!((view.m, view.n), (rows - k, cols));
        assert_eq!(view.resident_bytes(), copy.resident_bytes());
        let (got, want) = (gemv(&view), gemv(&copy));
        assert_eq!(got.len(), rows - k);
        assert!(
            got.iter()
                .zip(&want)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "view GEMV must be bit-identical to the copy"
        );
        assert!(
            full.row_view(1).is_none(),
            "64-byte plane offset must refuse"
        );
        assert!(full.row_view(0).is_none() && full.row_view(rows).is_none());
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
