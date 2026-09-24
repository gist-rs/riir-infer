//! EXL3 (trellis-coded) GPU dequantization — Issue 001 T7b.
//!
//! The 4090 arm of the EXL3 weight pipeline: three `CubeCL` kernels that turn a
//! quantized layer's trellis bitstream + channel scales into dense f32
//! weights. Two-tier parity against the CPU scalar reference
//! (`Exl3Layer::dequantize_f32`), matching how T4b itself adjudicated the
//! native oracle:
//!
//! 1. **Trellis decode — BIT-EXACT.** [`exl3_trellis_decode`] contains zero
//!    floating-point arithmetic: integer prefix/window math + a lookup into
//!    the SAME 65536-entry f32 table the CPU LUT arm uses (uploaded
//!    verbatim). Pinned by `gpu_decode_matches_cpu_bit_exact` (synthetic) and
//!    a real-pack tile probe.
//! 2. **Hadamard stages — FMA-contraction TOLERANCE.** The Hadamard kernels
//!    replicate the CPU's exact accumulation order (kk ascending, mul then
//!    add), but the GPU toolchain contracts `a*b + c` into fused multiply-add
//!    by default and there is no per-kernel opt-out through CubeCL's CUDA C++
//!    writer (plain `*`/`+` operators under NVRTC's default `-fmad=true`; the
//!    wgpu/SPIR-V lane contracts identically). FMA is one-directionally MORE
//!    accurate (one rounding per step instead of two) — the same divergence
//!    class as the reference implementation's own fp16 Hadamard intermediates
//!    (T4b §12.6: rel-Frobenius 2.96e-08, "the expected class"). The parity
//!    gate is a measured tight bound on max-ulp and rel-Frobenius, NOT bit
//!    equality; forcing bit-equality would require a global `-fmad=false`
//!    that would de-optimize every FMA-contracted GEMV kernel in the
//!    workspace — rejected deliberately.
//!
//! # Kernels
//!
//! - [`exl3_trellis_decode`] — one 256-thread workgroup per 16×16 tile; each
//!   thread decodes ONE ring position via the closed-form prefix sum (pinned
//!   against the sequential one by the core test
//!   `closed_form_prefix_matches_sequential`), MSB-first window extraction
//!   with per-bit ring wrap, tensor-core-order scatter.
//! - [`exl3_left_hadamard`] — `diag(suh)·(I⊗H)·W_rot`, one thread per output
//!   element, sequential 128-step loop in the CPU's accumulation order, row
//!   scale fused as a single multiply (the same one rounding the CPU's
//!   separate in-place pass performs).
//! - [`exl3_right_hadamard`] — `(I⊗H)·tmp·diag(svh)`, same shape, column
//!   scales fused the same way.
//!
//! The Hadamard matrix and codebook LUT are uploaded as the exact CPU table
//! bytes (`sylvester_hadamard_128` / `codebook_lut`) — no re-derivation on
//! the GPU, no Sylvester-parity proof obligation. Scales are expanded to f32
//! on the host via the core `suh_f32`/`svh_f32` helpers — the same expansion
//! the CPU reference applies.
//!
//! # Streaming (chunked over out-columns)
//!
//! The two f32 intermediates and the output chunk are bounded by
//! [`CHUNK_BYTE_BUDGET`] — the driver processes the out axis in 128-column
//! chunks (a multiple of both the Hadamard block and the tile), reading each
//! chunk back into the caller's Vec. Chunking changes NO arithmetic (each
//! output element's inputs and accumulation order are unchanged), so parity
//! holds for any chunk size — pinned by the multi-chunk test. VRAM peak ≈ 3 ×
//! budget/3 + trellis + tables, independent of layer size.
//!
//! # Numerics caveats (stated, not gated)
//!
//! - Denormal f32 values could flush on some GPU backends — the CPU honors
//!   them. Codebook values, scale magnitudes, and Hadamard sums in real packs
//!   sit orders of magnitude above the denormal range; the synthetic fixtures
//!   keep every value normal. A pack with denormal-scale weights would surface
//!   as a parity-test failure, not silent divergence.

use riir_infer_core::quant::exl3::{Exl3Layer, codebook_lut, sylvester_hadamard_128};

use cubecl::prelude::*;

/// Threads (units) per workgroup — the repo-wide `CubeCL` dispatch width (one full
/// 16×16 trellis tile per group per stride step in the decode kernel).
const EXL3_THREADS: u32 = 256;

/// wgpu caps a dispatch dimension at 65535 workgroups — the kernels are
/// grid-strided so one launch covers ANY layer size within this cap.
const MAX_WORKGROUPS: u32 = 65535;

/// Workgroup count + the exact thread total the grid-stride loops use.
fn stride_grid(total: u32) -> (u32, u32) {
    let n_wg = total.div_ceil(EXL3_THREADS).clamp(1, MAX_WORKGROUPS);
    (n_wg, n_wg * EXL3_THREADS)
}

/// VRAM budget for the three per-chunk f32 buffers (`w_rot`, tmp, out chunk).
const CHUNK_BYTE_BUDGET: usize = 768 << 20;

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

/// Trellis decode: one thread per (tile, ring position) → one rotated-basis
/// weight, GRID-STRIDED (wgpu caps dispatch group count at 65535 — real
/// layers need up to ~4.9M tile-threads, so each thread walks
/// `total_threads`-strided positions). One 256-thread workgroup covers a
/// full 16×16 tile per stride step, so each group's 256 scattered writes
/// land in one 16-row output strip — L2 merges them into near-optimal DRAM
/// traffic.
///
/// `c0_tiles` is the chunk's first out-TILE in the FULL layer (the trellis
/// tile grid is `[in/16, full_out/16]` row-major over the whole layer); the
/// write index is chunk-local. `tile_words` is u32 words per tile
/// (`ring_bits >> 5`).
#[cube(launch_unchecked)]
fn exl3_trellis_decode(
    trellis: &[u32],
    lut: &[f32],
    w_rot: &mut [f32],
    c0_tiles: u32,
    full_out_tiles: u32,
    chunk_out_tiles: u32,
    chunk_nout: u32,
    tile_words: u32,
    ring_bits: u32,
    ka: u32,
    half: u32,
    total: u32,
    total_threads: u32,
) {
    let stride = total_threads;
    let mut pos = ABSOLUTE_POS as u32;
    while pos < total {
        let local_tile = pos >> 8;
        let p = pos & 255;
        let a = local_tile / chunk_out_tiles;
        let c_local = local_tile % chunk_out_tiles;
        let c_full = c0_tiles + c_local;

        // Closed-form prefix sum S(p) = Σ_{q≤p} bits_for_step(q):
        // integer K → K·(p+1); half-K → ka·(p+1) + ⌊(p+1)/2⌋ (odd steps +1).
        let pp = p + 1;
        let s = if half != 0 { ka * pp + pp / 2 } else { ka * pp };

        // 16-bit window ENDING at ring bit s (exclusive), MSB-first, wrapping.
        let start = (s + ring_bits - 16) % ring_bits;
        let tile_base = (a * full_out_tiles + c_full) * tile_words;
        let mut w = 0u32;
        for m in 0..16u32 {
            let qm = (start + m) % ring_bits;
            let word = trellis[(tile_base + (qm >> 5)) as usize];
            w = (w << 1) | ((word >> (31 - (qm & 31))) & 1u32);
        }
        let v = lut[w as usize];

        // Tensor-core element order (core `ring_pos_to_tile_element`).
        let t = p >> 3;
        let j = p & 7;
        let in_off = ((t & 3) * 2) + (j & 1) + 8 * ((j >> 1) & 1);
        let out_off = (t >> 2) + 8 * ((j >> 2) & 1);
        let row = a * 16 + in_off;
        let col = c_local * 16 + out_off;
        w_rot[(row * chunk_nout + col) as usize] = v;

        pos += stride;
    }
}

/// Left Hadamard + row scale: `tmp[row][col] = suh[row] · Σ_kk H[r][kk]·w_rot[row_b+kk][col]`
/// — the CPU reference's accumulation order (kk ascending, mul then add),
/// grid-strided over the chunk's elements.
#[cube(launch_unchecked)]
fn exl3_left_hadamard(
    w_rot: &[f32],
    hmat: &[f32],
    suh: &[f32],
    tmp: &mut [f32],
    chunk_nout: u32,
    total: u32,
    total_threads: u32,
) {
    let stride = total_threads;
    let mut pos = ABSOLUTE_POS as u32;
    while pos < total {
        let row = pos / chunk_nout;
        let col = pos % chunk_nout;
        let row_block = row >> 7;
        let r = row & 127;
        let base = ((row_block * 128) * chunk_nout + col) as usize;
        let h_row = (r * 128) as usize;
        let stride_f = chunk_nout as usize;
        let mut acc = f32::new(0.0f32);
        let mut kk = 0u32;
        while kk < 128u32 {
            acc += hmat[h_row + kk as usize] * w_rot[base + kk as usize * stride_f];
            kk += 1;
        }
        // Row scale fused — one multiply, the same single rounding the CPU's
        // separate in-place pass performs.
        tmp[pos as usize] = acc * suh[row as usize];
        pos += stride;
    }
}

/// Right Hadamard + column scale:
/// `out[row][col] = svh[global_col] · Σ_kk tmp[row][cb·128+kk]·H[kk][c]`.
/// `c0_cols` is the chunk's first column in the FULL layer (svh is global).
/// Grid-strided over the chunk's elements.
#[cube(launch_unchecked)]
fn exl3_right_hadamard(
    tmp: &[f32],
    hmat: &[f32],
    svh: &[f32],
    out: &mut [f32],
    chunk_nout: u32,
    c0_cols: u32,
    total: u32,
    total_threads: u32,
) {
    let stride = total_threads;
    let mut pos = ABSOLUTE_POS as u32;
    while pos < total {
        let row = pos / chunk_nout;
        let col = pos % chunk_nout;
        let col_block = col >> 7;
        let c = col & 127;
        let src = (row * chunk_nout + col_block * 128) as usize;
        let mut acc = f32::new(0.0f32);
        let mut kk = 0u32;
        while kk < 128u32 {
            acc += tmp[src + kk as usize] * hmat[(kk * 128 + c) as usize];
            kk += 1;
        }
        out[pos as usize] = acc * svh[(c0_cols + col) as usize];
        pos += stride;
    }
}

// ---------------------------------------------------------------------------
// T7c v2 decode — word-aligned window extraction (Issue 001 §17.2)
// ---------------------------------------------------------------------------

/// Word-aligned 16-bit ring window (the v2 extraction, §17.2): MSB-first
/// window ENDING at ring bit `end` (exclusive) — identical semantics to v1's
/// per-bit loop, computed as 1–2 u32 loads + shift/mask with
/// conditional-subtract wrap. `ring_bits` is a multiple of 128 by
/// construction (`stream_bits_per_tile`), so `start` stays word-aligned-domain
/// and the only wrap case is `wi_hi == tile_words`.
#[cube]
fn window16_fast(trellis: &[u32], tile_base: u32, end: u32, ring_bits: u32, tile_words: u32) -> u32 {
    // start = end - 16, wrapping once (end in 1..=ring_bits).
    let start = if end >= 16u32 { end - 16u32 } else { end + ring_bits - 16u32 };
    let wi = start >> 5;
    let shift = start & 31u32;
    let word_lo = trellis[(tile_base + wi) as usize];
    if shift <= 16u32 {
        // Window entirely within word_lo: bits [16-shift, 31-shift].
        (word_lo >> (16u32 - shift)) & 0xFFFFu32
    } else {
        // Straddles: low (32-shift) bits of word_lo + top (shift-16) bits of
        // the NEXT word (wrapping to 0 at the ring end).
        let mut wi_hi = wi + 1u32;
        if wi_hi == tile_words {
            wi_hi = 0u32;
        }
        let word_hi = trellis[(tile_base + wi_hi) as usize];
        ((word_lo << (shift - 16u32)) | (word_hi >> (48u32 - shift))) & 0xFFFFu32
    }
}

/// v2 trellis decode — same output contract as [`exl3_trellis_decode`]
/// (bit-exact; integer-only math + the same LUT), with the word-aligned
/// window extraction (§17.2 arm A2).
#[cube(launch_unchecked)]
fn exl3_trellis_decode_v2(
    trellis: &[u32],
    lut: &[f32],
    w_rot: &mut [f32],
    c0_tiles: u32,
    full_out_tiles: u32,
    chunk_out_tiles: u32,
    chunk_nout: u32,
    tile_words: u32,
    ring_bits: u32,
    ka: u32,
    half: u32,
    total: u32,
    total_threads: u32,
) {
    let stride = total_threads;
    let mut pos = ABSOLUTE_POS as u32;
    while pos < total {
        let local_tile = pos >> 8;
        let p = pos & 255;
        let a = local_tile / chunk_out_tiles;
        let c_local = local_tile % chunk_out_tiles;
        let c_full = c0_tiles + c_local;

        let pp = p + 1;
        let s = if half != 0 { ka * pp + pp / 2 } else { ka * pp };

        let tile_base = (a * full_out_tiles + c_full) * tile_words;
        let w = window16_fast(trellis, tile_base, s, ring_bits, tile_words);
        let v = lut[w as usize];

        let t = p >> 3;
        let j = p & 7;
        let in_off = ((t & 3) * 2) + (j & 1) + 8 * ((j >> 1) & 1);
        let out_off = (t >> 2) + 8 * ((j >> 2) & 1);
        let row = a * 16 + in_off;
        let col = c_local * 16 + out_off;
        w_rot[(row * chunk_nout + col) as usize] = v;

        pos += stride;
    }
}

/// Bench arm A3 (§17.2): v2 extraction with the LUT gather replaced by an
/// arithmetic function of `w` — isolates the surviving `lut[w]` gather
/// (1 load/weight). NOT bit-exact vs v1 — bench-only, never a parity path.
#[cube(launch_unchecked)]
fn exl3_trellis_decode_v2_nolut(
    trellis: &[u32],
    w_rot: &mut [f32],
    c0_tiles: u32,
    full_out_tiles: u32,
    chunk_out_tiles: u32,
    chunk_nout: u32,
    tile_words: u32,
    ring_bits: u32,
    ka: u32,
    half: u32,
    total: u32,
    total_threads: u32,
) {
    let stride = total_threads;
    let mut pos = ABSOLUTE_POS as u32;
    while pos < total {
        let local_tile = pos >> 8;
        let p = pos & 255;
        let a = local_tile / chunk_out_tiles;
        let c_local = local_tile % chunk_out_tiles;
        let c_full = c0_tiles + c_local;

        let pp = p + 1;
        let s = if half != 0 { ka * pp + pp / 2 } else { ka * pp };

        let tile_base = (a * full_out_tiles + c_full) * tile_words;
        let w = window16_fast(trellis, tile_base, s, ring_bits, tile_words);
        // Consume `w` arithmetically so the trellis loads stay live.
        let v = w as f32 * (1.0f32 / 65536.0f32);

        let t = p >> 3;
        let j = p & 7;
        let in_off = ((t & 3) * 2) + (j & 1) + 8 * ((j >> 1) & 1);
        let out_off = (t >> 2) + 8 * ((j >> 2) & 1);
        let row = a * 16 + in_off;
        let col = c_local * 16 + out_off;
        w_rot[(row * chunk_nout + col) as usize] = v;

        pos += stride;
    }
}

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------

/// Error type for the EXL3 GPU dequant driver.
#[derive(Debug)]
pub enum Exl3DequantError {
    /// Layer geometry exceeds the u32 kernel-index space (`in·out ≥ 2³²`).
    LayerTooLarge { in_features: usize, out_features: usize },
    /// GPU readback failed (device lost / sync error).
    Readback(String),
    /// The stable bench's work floor: a pass finished under 5 ms — launch
    /// overhead + clock resolution dominate and no timing from this harness
    /// would be sound. The caller raises `reps` (plan 003 contract 2).
    TooFastToTime { pass_secs: f64, reps: usize },
}

impl core::fmt::Display for Exl3DequantError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::LayerTooLarge { in_features, out_features } => write!(
                f,
                "EXL3 layer {in_features}x{out_features} exceeds the GPU kernel's u32 index space"
            ),
            Self::Readback(msg) => write!(f, "EXL3 GPU readback failed: {msg}"),
            Self::TooFastToTime { pass_secs, reps } => write!(
                f,
                "EXL3 bench pass {pass_secs:.3} ms < 5 ms floor at reps={reps} — \
                 raise reps; below the floor no timing from this harness is sound"
            ),
        }
    }
}

impl std::error::Error for Exl3DequantError {}

/// The T7c-1d stable-bench result (plan 003): robust per-pass stats for one
/// decode arm, the instability verdict, and the cross-method agreement.
#[derive(Debug, Clone, Copy)]
pub struct StableBench {
    /// Decoded weights per pass (`in·out`).
    pub weights: u64,
    /// Kernel enqueues per timed pass.
    pub reps: usize,
    /// Timed passes sampled (after warmup).
    pub samples: usize,
    /// MIN pass time / reps — the peak-attained figure (the headline).
    pub min_secs: f64,
    /// Median pass time / reps.
    pub median_secs: f64,
    /// p90 pass time / reps.
    pub p90_secs: f64,
    /// `(p90 − min)/min` — the within-run spread.
    pub spread: f64,
    /// `spread > 0.10` — the run is unstable and publishes NO verdict.
    pub unstable: bool,
    /// The independent reps-differential estimate (the T7c-1b method).
    pub diff_secs: Option<f64>,
    /// min-of-N and the differential agree within 15% — the soundness
    /// cross-check (only meaningful when `!unstable`).
    pub cross_agrees: bool,
}

impl StableBench {
    /// Peak-attained throughput, Gw/s (weights / min pass time).
    pub fn gw_peak(&self) -> f64 {
        self.weights as f64 / self.min_secs / 1e9
    }
    /// Median throughput, Gw/s.
    pub fn gw_median(&self) -> f64 {
        self.weights as f64 / self.median_secs / 1e9
    }
}

/// Per-run timing breakdown (bench instrumentation for the T7b record).
#[derive(Debug, Clone, Copy)]
pub struct Exl3DequantTiming {
    /// Total dequantized weights (`in·out`).
    pub weights: u64,
    /// Wall clock of the whole call: scale expansion + uploads + every chunk
    /// launch + readback, seconds.
    pub wall_secs: f64,
    /// Estimated GPU-side kernel time per dequant pass — the (reps−1)-run
    /// differential ((wall@reps − wall@1)/(reps−1)), which cancels the
    /// upload + readback constants. `None` when `reps == 1`.
    pub kernel_secs: Option<f64>,
}

/// View the trellis bitstream as LE u32 words — the exact word semantics of
/// the reference `ring_bit` (LE u32 at byte offset `w·4`).
///
/// Little-endian hosts (both supported boxes) get the zero-copy view; a
/// big-endian host takes a converting copy.
fn trellis_as_u32(bytes: &[u8]) -> std::borrow::Cow<'_, [u32]> {
    if cfg!(target_endian = "little") {
        std::borrow::Cow::Borrowed(bytemuck::cast_slice(bytes))
    } else {
        std::borrow::Cow::Owned(
            bytes.as_chunks::<4>().0.iter()
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        )
    }
}

/// Decode-only bench arm selector (§17.2 — the discriminating bench).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodeArm {
    /// A1 — the T7b v1 kernel (16 per-bit loads + modulos per weight).
    V1,
    /// A2 — v2 word-aligned windows (1-2 loads, modulo-free).
    V2,
    /// A3 — v2 with the LUT gather replaced arithmetically (isolates the
    /// gather). NOT bit-exact — bench-only.
    V2NoLut,
}

/// Decode-only bench result (§17.2): throughput of ONE decode kernel
/// over the layer's rotated basis — no Hadamards, no scales.
#[derive(Debug, Clone, Copy)]
pub struct DecodeBench {
    /// Decoded weights (`in·out`).
    pub weights: u64,
    /// Single-pass wall (uploads + launches + readback), seconds.
    pub wall_secs: f64,
    /// (wall@reps − wall@1)/(reps−1) — cancels upload+readback.
    pub kernel_secs: Option<f64>,
}

/// EXL3 GPU dequantization (Issue 001 T7b): trellis decode + block-Hadamard
/// incoherence rotation + channel scales, `CubeCL`, bit-identical to the CPU
/// scalar reference.
pub struct Exl3DequantCubeCL;

impl Exl3DequantCubeCL {
    /// Dequantize one layer to dense f32 (`W[in][out]` in-major) with the
    /// default chunk budget and a single pass.
    ///
    /// # Panics
    ///
    /// Panics on geometry the kernels cannot index (never for packs within
    /// the format's own limits — `in·out < 2³²`).
    pub fn dequant_layer_f32<R: Runtime>(
        client: &ComputeClient<R>,
        layer: &Exl3Layer<'_>,
    ) -> Result<Vec<f32>, Exl3DequantError> {
        Self::dequant_layer_f32_chunked(client, layer, None, 1).map(|(w, _)| w)
    }

    /// The streaming seam: `chunk_cols` (a 128-multiple, `None` = derive
    /// from [`CHUNK_BYTE_BUDGET`]) splits the out axis; `reps` re-enqueues
    /// each chunk's kernel sequence that many times (idempotent — same
    /// inputs, same outputs) so the bench can derive kernel-only time from
    /// the wall differential.
    pub fn dequant_layer_f32_chunked<R: Runtime>(
        client: &ComputeClient<R>,
        layer: &Exl3Layer<'_>,
        chunk_cols: Option<usize>,
        reps: usize,
    ) -> Result<(Vec<f32>, Exl3DequantTiming), Exl3DequantError> {
        let t0 = std::time::Instant::now();
        let (kin, nout) = (layer.in_features, layer.out_features);
        if (kin as u64) * (nout as u64) >= u64::from(u32::MAX) {
            return Err(Exl3DequantError::LayerTooLarge { in_features: kin, out_features: nout });
        }

        // Chunk width: 128-multiple, VRAM-bounded, layer-bounded.
        let chunk = chunk_cols.unwrap_or_else(|| {
            let by_budget = (CHUNK_BYTE_BUDGET / 3 / (kin * 4)).max(128) & !127;
            by_budget.min(nout)
        });
        assert!(chunk.is_multiple_of(128) && chunk >= 128 && chunk <= nout,
            "chunk_cols must be a 128-multiple in [128, out_features={nout}], got {chunk}");

        // Host-side expansion + one-shot uploads.
        let suh = layer.suh_f32();
        let svh = layer.svh_f32();
        let trellis = trellis_as_u32(layer.trellis_bytes());
        let trellis_h = client.create_from_slice(u32::as_bytes(&trellis));
        let lut = codebook_lut(layer.codebook);
        let lut_h = client.create_from_slice(f32::as_bytes(&lut[..]));
        // [[f32; 128]; 128] is contiguous — the flat f32 view is the exact
        // CPU table bytes.
        let hmat: &[f32] = bytemuck::cast_slice(sylvester_hadamard_128());
        let hmat_h = client.create_from_slice(f32::as_bytes(hmat));
        let suh_h = client.create_from_slice(f32::as_bytes(&suh));
        let svh_h = client.create_from_slice(f32::as_bytes(&svh));

        let k = layer.k;
        let ring_bits = k.stream_bits_per_tile() as u32;
        let tile_words = ring_bits >> 5;
        let in_tiles = (kin / 16) as u32;
        let full_out_tiles = (nout / 16) as u32;
        let reps = reps.max(1);

        let mut out = vec![0.0f32; kin * nout];
        for c0 in (0..nout).step_by(chunk) {
            let cols = chunk.min(nout - c0);
            let chunk_nout = cols as u32;
            let chunk_out_tiles = (cols / 16) as u32;
            let elems = kin * cols;
            let w_rot_h = client.empty(elems * 4);
            let tmp_h = client.empty(elems * 4);
            let out_h = client.empty(elems * 4);
            for _ in 0..reps {
                let decode_total = in_tiles * chunk_out_tiles * 256;
                let (decode_wg, decode_threads) = stride_grid(decode_total);
                let h_total = elems as u32;
                let (h_wg, h_threads) = stride_grid(h_total);
                unsafe {
                    exl3_trellis_decode::launch_unchecked::<R>(
                        client,
                        CubeCount::Static(decode_wg, 1, 1),
                        CubeDim::new_1d(EXL3_THREADS),
                        BufferArg::from_raw_parts(trellis_h.clone(), trellis.len()),
                        BufferArg::from_raw_parts(lut_h.clone(), lut.len()),
                        BufferArg::from_raw_parts(w_rot_h.clone(), elems),
                        (c0 / 16) as u32,
                        full_out_tiles,
                        chunk_out_tiles,
                        chunk_nout,
                        tile_words,
                        ring_bits,
                        k.ka as u32,
                        u32::from(k.half),
                        decode_total,
                        decode_threads,
                    );
                    exl3_left_hadamard::launch_unchecked::<R>(
                        client,
                        CubeCount::Static(h_wg, 1, 1),
                        CubeDim::new_1d(EXL3_THREADS),
                        BufferArg::from_raw_parts(w_rot_h.clone(), elems),
                        BufferArg::from_raw_parts(hmat_h.clone(), hmat.len()),
                        BufferArg::from_raw_parts(suh_h.clone(), suh.len()),
                        BufferArg::from_raw_parts(tmp_h.clone(), elems),
                        chunk_nout,
                        h_total,
                        h_threads,
                    );
                    exl3_right_hadamard::launch_unchecked::<R>(
                        client,
                        CubeCount::Static(h_wg, 1, 1),
                        CubeDim::new_1d(EXL3_THREADS),
                        BufferArg::from_raw_parts(tmp_h.clone(), elems),
                        BufferArg::from_raw_parts(hmat_h.clone(), hmat.len()),
                        BufferArg::from_raw_parts(svh_h.clone(), svh.len()),
                        BufferArg::from_raw_parts(out_h.clone(), elems),
                        chunk_nout,
                        c0 as u32,
                        h_total,
                        h_threads,
                    );
                }
            }
            let bytes = client
                .read_one(out_h)
                .map_err(|e| Exl3DequantError::Readback(format!("{e:?}")))?;
            let chunk_out = f32::from_bytes(&bytes);
            assert!(
                chunk_out.len() >= elems,
                "readback {} < expected {elems} elements",
                chunk_out.len()
            );
            // The chunk readback is [in × cols] row-major (row-stride cols);
            // scatter each chunk row into the full layer at column offset c0.
            for (r, row) in chunk_out[..elems].chunks_exact(cols).enumerate() {
                out[r * nout + c0..r * nout + c0 + cols].copy_from_slice(row);
            }
        }

        let wall = t0.elapsed().as_secs_f64().max(1e-9);
        Ok((
            out,
            Exl3DequantTiming {
                weights: (kin * nout) as u64,
                wall_secs: wall,
                kernel_secs: None,
            },
        ))
    }

    /// Timed variant for the bench record: runs the full dequant twice —
    /// once with `reps` and once single-pass — and reports both walls plus
    /// the differential kernel-only estimate.
    pub fn dequant_layer_f32_timed<R: Runtime>(
        client: &ComputeClient<R>,
        layer: &Exl3Layer<'_>,
        reps: usize,
    ) -> Result<(Vec<f32>, Exl3DequantTiming), Exl3DequantError> {
        let (w, single) = Self::dequant_layer_f32_chunked(client, layer, None, 1)?;
        if reps <= 1 {
            return Ok((
                w,
                Exl3DequantTiming { weights: single.weights, wall_secs: single.wall_secs, kernel_secs: None },
            ));
        }
        let (_, multi) = Self::dequant_layer_f32_chunked(client, layer, None, reps)?;
        let kernel = (multi.wall_secs - single.wall_secs) / (reps - 1) as f64;
        Ok((
            w,
            Exl3DequantTiming {
                weights: single.weights,
                wall_secs: single.wall_secs,
                kernel_secs: Some(kernel.max(0.0)),
            },
        ))
    }

    /// Decode-only bench (§17.2): runs ONE decode arm over the layer's
    /// rotated basis (w_rot, [in × out] in-major), chunked over out-columns
    /// under [`CHUNK_BYTE_BUDGET`]; `reps` re-enqueues each chunk's decode
    /// kernel (idempotent). Returns the readback w_rot (full for layers that
    /// fit, else the chunks are read back in order — the Vec is ALWAYS the
    /// full layer for the arms that produce it; VRAM-bounded, never resident
    /// beyond one chunk + the trellis/LUT uploads).
    pub fn decode_only_layer<R: Runtime>(
        client: &ComputeClient<R>,
        layer: &Exl3Layer<'_>,
        arm: DecodeArm,
        reps: usize,
    ) -> Result<(Vec<f32>, DecodeBench), Exl3DequantError> {
        Self::decode_only_layer_chunked(client, layer, arm, None, reps)
    }

    /// The chunked decode-only seam (see [`Self::decode_only_layer`]).
    pub fn decode_only_layer_chunked<R: Runtime>(
        client: &ComputeClient<R>,
        layer: &Exl3Layer<'_>,
        arm: DecodeArm,
        chunk_cols: Option<usize>,
        reps: usize,
    ) -> Result<(Vec<f32>, DecodeBench), Exl3DequantError> {
        let t0 = std::time::Instant::now();
        let (kin, nout) = (layer.in_features, layer.out_features);
        if (kin as u64) * (nout as u64) >= u64::from(u32::MAX) {
            return Err(Exl3DequantError::LayerTooLarge { in_features: kin, out_features: nout });
        }
        let chunk = chunk_cols.unwrap_or_else(|| {
            let by_budget = (CHUNK_BYTE_BUDGET / (kin * 4)).max(128) & !127;
            by_budget.min(nout)
        });
        assert!(chunk.is_multiple_of(128) && chunk >= 128 && chunk <= nout,
            "chunk_cols must be a 128-multiple in [128, out_features={nout}], got {chunk}");

        let trellis = trellis_as_u32(layer.trellis_bytes());
        let trellis_h = client.create_from_slice(u32::as_bytes(&trellis));
        let lut = codebook_lut(layer.codebook);
        let lut_h = client.create_from_slice(f32::as_bytes(&lut[..]));

        let k = layer.k;
        let ring_bits = k.stream_bits_per_tile() as u32;
        let tile_words = ring_bits >> 5;
        let in_tiles = (kin / 16) as u32;
        let full_out_tiles = (nout / 16) as u32;
        let reps = reps.max(1);

        let mut out = vec![0.0f32; kin * nout];
        for c0 in (0..nout).step_by(chunk) {
            let cols = chunk.min(nout - c0);
            let chunk_nout = cols as u32;
            let chunk_out_tiles = (cols / 16) as u32;
            let elems = kin * cols;
            let w_rot_h = client.empty(elems * 4);
            let decode_total = in_tiles * chunk_out_tiles * 256;
            let (decode_wg, decode_threads) = stride_grid(decode_total);
            for _ in 0..reps {
                unsafe {
                    match arm {
                        DecodeArm::V1 => {
                            exl3_trellis_decode::launch_unchecked::<R>(
                                client,
                                CubeCount::Static(decode_wg, 1, 1),
                                CubeDim::new_1d(EXL3_THREADS),
                                BufferArg::from_raw_parts(trellis_h.clone(), trellis.len()),
                                BufferArg::from_raw_parts(lut_h.clone(), lut.len()),
                                BufferArg::from_raw_parts(w_rot_h.clone(), elems),
                                (c0 / 16) as u32,
                                full_out_tiles,
                                chunk_out_tiles,
                                chunk_nout,
                                tile_words,
                                ring_bits,
                                k.ka as u32,
                                u32::from(k.half),
                                decode_total,
                                decode_threads,
                            );
                        }
                        DecodeArm::V2 => {
                            exl3_trellis_decode_v2::launch_unchecked::<R>(
                                client,
                                CubeCount::Static(decode_wg, 1, 1),
                                CubeDim::new_1d(EXL3_THREADS),
                                BufferArg::from_raw_parts(trellis_h.clone(), trellis.len()),
                                BufferArg::from_raw_parts(lut_h.clone(), lut.len()),
                                BufferArg::from_raw_parts(w_rot_h.clone(), elems),
                                (c0 / 16) as u32,
                                full_out_tiles,
                                chunk_out_tiles,
                                chunk_nout,
                                tile_words,
                                ring_bits,
                                k.ka as u32,
                                u32::from(k.half),
                                decode_total,
                                decode_threads,
                            );
                        }
                        DecodeArm::V2NoLut => {
                            exl3_trellis_decode_v2_nolut::launch_unchecked::<R>(
                                client,
                                CubeCount::Static(decode_wg, 1, 1),
                                CubeDim::new_1d(EXL3_THREADS),
                                BufferArg::from_raw_parts(trellis_h.clone(), trellis.len()),
                                BufferArg::from_raw_parts(w_rot_h.clone(), elems),
                                (c0 / 16) as u32,
                                full_out_tiles,
                                chunk_out_tiles,
                                chunk_nout,
                                tile_words,
                                ring_bits,
                                k.ka as u32,
                                u32::from(k.half),
                                decode_total,
                                decode_threads,
                            );
                        }
                    }
                }
            }
            let bytes = client
                .read_one(w_rot_h)
                .map_err(|e| Exl3DequantError::Readback(format!("{e:?}")))?;
            let chunk_out = f32::from_bytes(&bytes);
            assert!(chunk_out.len() >= elems,
                "readback {} < expected {elems} elements", chunk_out.len());
            // [in × cols] row-major chunk — scatter per-row (the §16 lesson).
            for (r, row) in chunk_out[..elems].chunks_exact(cols).enumerate() {
                out[r * nout + c0..r * nout + c0 + cols].copy_from_slice(row);
            }
        }

        let wall = t0.elapsed().as_secs_f64().max(1e-9);
        Ok((
            out,
            DecodeBench { weights: (kin * nout) as u64, wall_secs: wall, kernel_secs: None },
        ))
    }

    /// Timed decode-only bench: warmup (JIT) first, then the
    /// reps-differential kernel-only estimate (cancels uploads + readback),
    /// the §17.2 Gw/s figure.
    pub fn decode_only_layer_timed<R: Runtime>(
        client: &ComputeClient<R>,
        layer: &Exl3Layer<'_>,
        arm: DecodeArm,
        reps: usize,
    ) -> Result<(Vec<f32>, DecodeBench), Exl3DequantError> {
        if reps > 1 {
            // Warmup: pay JIT + first-upload outside the timed pair (the T7b
            // bench's pattern — without it the differential reads negative).
            let _ = Self::decode_only_layer_chunked(client, layer, arm, None, 1)?;
        }
        let (w, single) = Self::decode_only_layer_chunked(client, layer, arm, None, 1)?;
        if reps <= 1 {
            return Ok((w, DecodeBench { weights: single.weights, wall_secs: single.wall_secs, kernel_secs: None }));
        }
        let (_, multi) = Self::decode_only_layer_chunked(client, layer, arm, None, reps)?;
        let kernel = (multi.wall_secs - single.wall_secs) / (reps - 1) as f64;
        Ok((
            w,
            DecodeBench {
                weights: single.weights,
                wall_secs: single.wall_secs,
                kernel_secs: Some(kernel.max(0.0)),
            },
        ))
    }

    /// The T7c-1d STABLE sample (plan 003): one pass of the decode arm =
    /// `reps` kernel enqueues + ONE trailing readback, timed sync-bracketed.
    /// No per-chunk readback inside the timed window — the readback cost is
    /// a constant per pass and the readback itself happens after the clock
    /// stops... no: it happens BEFORE the clock stops (the sync inside
    /// read_one is what guarantees the enqueued work completed), so the
    /// readback cost is INSIDE every pass — the same constant every pass,
    /// amortized by reps, and the stats below treat the per-pass mean as the
    /// sample. The differential arm of the cross-check subtracts it.
    fn decode_pass_secs<R: Runtime>(
        client: &ComputeClient<R>,
        layer: &Exl3Layer<'_>,
        arm: DecodeArm,
        reps: usize,
    ) -> Result<f64, Exl3DequantError> {
        let t0 = std::time::Instant::now();
        let (_, timing) = Self::decode_only_layer_chunked(client, layer, arm, None, reps)?;
        let _ = timing;
        Ok(t0.elapsed().as_secs_f64() / reps.max(1) as f64)
    }

    /// The T7c-1d STABLE harness (plan 003 — replaces the T7c-1b
    /// reps-differential-only readout):
    ///
    /// - warmup pass first (JIT + allocator);
    /// - `samples` timed passes, each = `reps` enqueues + one readback;
    /// - stats: min / median / p90 per pass (×reps-normalized);
    /// - the ≥5 ms/pass work floor: refuses (`TooFastToTime`) when the MIN
    ///   pass wall < 5 ms — below that, launch overhead + clock resolution
    ///   dominate and NO number from this harness is sound (the caller
    ///   raises `reps`);
    /// - the 10% instability gate: `(p90 − min)/min > 0.10` ⇒ `unstable`;
    /// - the cross-method gate: the sampling MIN vs the T7c-1b
    ///   reps-differential estimate must agree within 15% ⇒ `cross_agrees`
    ///   (two independent samplings agreeing is the soundness evidence).
    ///
    /// The per-kernel isolation a CUDA event would give is NOT reachable in
    /// cubecl 0.11 (the server's raw `CUstream` is private; its `Fence`
    /// exposes no elapsed) — sync-bracketed system time is the sound
    /// primitive here, sound ONLY above the work floor, and the floor is
    /// enforced, not assumed.
    pub fn bench_arm_stable<R: Runtime>(
        client: &ComputeClient<R>,
        layer: &Exl3Layer<'_>,
        arm: DecodeArm,
        reps: usize,
        samples: usize,
    ) -> Result<StableBench, Exl3DequantError> {
        let reps = reps.max(1);
        let samples = samples.max(5);

        // Warmup (JIT + pool growth) — never sampled.
        let _ = Self::decode_only_layer_chunked(client, layer, arm, None, reps)?;

        let mut passes = Vec::with_capacity(samples);
        for _ in 0..samples {
            passes.push(Self::decode_pass_secs(client, layer, arm, reps)?);
        }
        passes.sort_by(|a, b| a.total_cmp(b));
        let min = passes[0];
        let median = passes[passes.len() / 2];
        let p90 = passes[passes.len() * 9 / 10];

        // The ≥5 ms/pass work floor (plan 003 contract 2).
        let pass_secs = min * f64::from(reps as u32);
        if pass_secs < 5e-3 {
            return Err(Exl3DequantError::TooFastToTime {
                pass_secs,
                reps,
            });
        }

        let spread = if min > 0.0 { (p90 - min) / min } else { f64::INFINITY };
        let unstable = spread > 0.10;

        // Cross-method: the independent reps-differential estimate.
        let (_, diff) = Self::decode_only_layer_timed::<R>(client, layer, arm, reps + 1)?;
        let diff_per = diff.kernel_secs.unwrap_or(f64::INFINITY);
        let cross_agrees = !unstable
            && diff_per.is_finite()
            && ((min - diff_per).abs() / diff_per.max(1e-12) <= 0.15);

        Ok(StableBench {
            weights: (layer.in_features * layer.out_features) as u64,
            reps,
            samples,
            min_secs: min,
            median_secs: median,
            p90_secs: p90,
            spread,
            unstable,
            diff_secs: diff.kernel_secs,
            cross_agrees,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cubecl_runtime::{ActiveRuntime, CubeCLContext};
    use half::f16;
    use riir_infer_core::quant::exl3::{EXL3_MUL1_MARKER, Exl3Codebook, Exl3K};

    /// Deterministic LCG — finite, well-scaled fixture values only (the
    /// denormal caveat in the module doc).
    struct Rng(u32);
    impl Rng {
        fn next_u32(&mut self) -> u32 {
            self.0 = self.0.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            self.0
        }
        fn byte(&mut self) -> u8 {
            (self.next_u32() >> 24) as u8
        }
        /// A finite f16 in [0.25, 4.0) — normal range, no NaN/Inf/denormal.
        fn f16_scale(&mut self) -> f16 {
            let bits = 0x3800u16 | ((self.next_u32() >> 20) as u16 & 0x03FF); // [1.0, 2.0)
            f16::from_bits(bits)
        }
    }

    /// Owned synthetic layer parts + the validated zero-copy view over them.
    struct SynthLayer {
        trellis: Vec<u8>,
        su: Vec<u8>,
        sv: Vec<u8>,
        legacy_signs: bool,
        mcg: Option<u32>,
        mul1: Option<u32>,
        in_f: usize,
        out_f: usize,
    }

    impl SynthLayer {
        fn new(
            in_f: usize,
            out_f: usize,
            k: Exl3K,
            codebook: Exl3Codebook,
            legacy_signs: bool,
            seed: u32,
        ) -> Self {
            let mut rng = Rng(seed ^ 0xC0DE);
            let tiles = (in_f / 16) * (out_f / 16);
            let mut trellis = vec![0u8; tiles * k.words_per_tile() * 2];
            for b in trellis.iter_mut() {
                *b = rng.byte();
            }
            // `su`/`sv` carry the fp16 scales (modern) or packed signs (legacy).
            let mut su = vec![0u8; if legacy_signs { in_f / 8 } else { in_f * 2 }];
            let mut sv = vec![0u8; if legacy_signs { out_f / 8 } else { out_f * 2 }];
            if legacy_signs {
                for b in su.iter_mut().chain(&mut sv) {
                    *b = rng.byte();
                }
            } else {
                for bufs in [&mut su, &mut sv] {
                    for c in bufs.as_chunks_mut::<2>().0 {
                        c.copy_from_slice(&rng.f16_scale().to_bits().to_le_bytes());
                    }
                }
            }
            let (mcg, mul1) = match codebook {
                Exl3Codebook::Cb0 => (None, None),
                Exl3Codebook::Cb1Mcg => (Some(0xCBAC_1FED), None),
                Exl3Codebook::Cb2Mul1 => (None, Some(EXL3_MUL1_MARKER)),
            };
            Self { trellis, su, sv, legacy_signs, mcg, mul1, in_f, out_f }
        }

        fn layer(&self) -> Exl3Layer<'_> {
            Exl3Layer::from_raw_parts(
                &self.trellis,
                if self.legacy_signs { None } else { Some(&self.su) },
                if self.legacy_signs { None } else { Some(&self.sv) },
                if self.legacy_signs { Some(&self.su) } else { None },
                if self.legacy_signs { Some(&self.sv) } else { None },
                self.mcg,
                self.mul1,
                self.in_f,
                self.out_f,
            )
            .expect("synthetic layer validates")
        }
    }

    /// ULP distance between two finite f32s (bit-pattern units) — DIAGNOSTIC
    /// ONLY: a near-zero sum whose tiny FMA error crosses the sign makes this
    /// explode (the metric is meaningless across zero); the GATES are the
    /// scale-aware pair below.
    fn ulp_diff(a: f32, b: f32) -> u32 {
        let (ia, ib) = (a.to_bits(), b.to_bits());
        let (sa, sb) = (ia ^ 0x8000_0000, ib ^ 0x8000_0000);
        let (ma, mb) = (
            if ia >> 31 != 0 { !sa } else { sa },
            if ib >> 31 != 0 { !sb } else { sb },
        );
        ma.abs_diff(mb)
    }

    /// Gates for the measured Hadamard divergence class (GPU FMA contraction
    /// vs the CPU's separate mul+add). Both are SCALE-AWARE — a near-zero sum
    /// whose ~1e-8 FMA error crosses the sign is inside the class, so ulp is
    /// not gateable. Calibrated at 10× the observed max (synthetic + real
    /// pack): a REAL regression (wrong codebook value, wrong placement,
    /// missing scale) misses these by orders of magnitude.
    const HADAMARD_MAX_REL_FRO: f64 = 1e-5;
    /// max |cpu−gpu| / max|W|.
    const HADAMARD_MAX_SCALED_ABS: f64 = 1e-5;

    /// Full-W tolerance compare (the module doc's tier 2): asserts the
    /// FMA-contraction divergence class and nothing wider.
    fn assert_w_within_contraction_class(cpu: &[f32], gpu: &[f32], what: &str) {
        assert_eq!(cpu.len(), gpu.len(), "{what}: output length");
        let mut max_abs = 0.0f64;
        let mut max_w = 0.0f64;
        let mut bit_mismatches = 0usize;
        let mut max_ulp = 0u32;
        let mut worst_abs_at: Option<(usize, f32, f32)> = None;
        let mut sq_diff = 0.0f64;
        let mut sq_ref = 0.0f64;
        for (i, (a, b)) in cpu.iter().zip(gpu).enumerate() {
            assert!(a.is_finite() && b.is_finite(), "{what}[{i}]: non-finite (cpu={a} gpu={b})");
            let d = (*a - *b) as f64;
            let ad = d.abs();
            if ad > max_abs {
                max_abs = ad;
                worst_abs_at = Some((i, *a, *b));
            }
            max_w = max_w.max(a.abs() as f64);
            if a.to_bits() != b.to_bits() {
                bit_mismatches += 1;
                max_ulp = max_ulp.max(ulp_diff(*a, *b));
            }
            sq_diff += d * d;
            sq_ref += (*a as f64) * (*a as f64);
        }
        let rel_fro = (sq_diff / sq_ref.max(1e-300)).sqrt();
        let scaled_abs = max_abs / max_w.max(1e-30);
        assert!(
            scaled_abs <= HADAMARD_MAX_SCALED_ABS,
            "{what}: max |diff| {max_abs:.3e} / max|W| {max_w:.3e} = {scaled_abs:.3e} > gate \
             {HADAMARD_MAX_SCALED_ABS:.1e} (worst at {worst_abs_at:?}) — outside the FMA-contraction \
             class (a real decode/placement/scale regression?)"
        );
        assert!(
            rel_fro <= HADAMARD_MAX_REL_FRO,
            "{what}: rel-Frobenius {rel_fro:.3e} > gate {HADAMARD_MAX_REL_FRO:.1e}"
        );
        eprintln!(
            "{what}: {bit_mismatches}/{} bits differ (FMA class), max |diff|/max|W| {scaled_abs:.2e}, \
             rel-Fro {rel_fro:.3e}, diag max-ulp {max_ulp}",
            cpu.len()
        );
    }

    /// The full parity harness: CPU scalar oracle vs GPU under the two-tier
    /// oracle — decode bit-exact (separately pinned), W within the FMA class.
    fn assert_gpu_matches_cpu(
        in_f: usize,
        out_f: usize,
        k: Exl3K,
        codebook: Exl3Codebook,
        legacy_signs: bool,
        chunk_override: Option<usize>,
        seed: u32,
    ) {
        let synth = SynthLayer::new(in_f, out_f, k, codebook, legacy_signs, seed);
        let layer = synth.layer();
        let cpu = layer.dequantize_f32();

        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();
        let (gpu, _) = Exl3DequantCubeCL::dequant_layer_f32_chunked::<ActiveRuntime>(
            &client,
            &layer,
            chunk_override,
            1,
        )
        .expect("GPU dequant succeeds");
        let what = format!(
            "synth in={in_f} out={out_f} K={}.{} cb={codebook:?} legacy={legacy_signs} chunk={chunk_override:?}",
            k.ka,
            if k.half { 5 } else { 0 },
        );
        assert_w_within_contraction_class(&cpu, &gpu, &what);
    }

    #[test]
    fn gpu_matches_cpu_integer_k_k3() {
        assert_gpu_matches_cpu(256, 384, Exl3K { ka: 3, half: false }, Exl3Codebook::Cb0, false, None, 1);
    }

    #[test]
    fn gpu_matches_cpu_integer_k_k5_wide() {
        // K5 + a non-square layer (asymmetric block counts).
        assert_gpu_matches_cpu(384, 640, Exl3K { ka: 5, half: false }, Exl3Codebook::Cb1Mcg, false, None, 2);
    }

    #[test]
    fn gpu_matches_cpu_half_k_mul1() {
        // Half-K requires the mul1 codebook (validated by from_raw_parts).
        assert_gpu_matches_cpu(256, 512, Exl3K { ka: 4, half: true }, Exl3Codebook::Cb2Mul1, false, None, 3);
    }

    #[test]
    fn gpu_matches_cpu_legacy_signs() {
        assert_gpu_matches_cpu(256, 256, Exl3K { ka: 2, half: false }, Exl3Codebook::Cb0, true, None, 4);
    }

    #[test]
    fn gpu_chunked_matches_whole() {
        // Force the streaming path: 4 chunks of 128 on a 512-col layer —
        // chunking must not move the parity verdict.
        assert_gpu_matches_cpu(256, 512, Exl3K { ka: 4, half: false }, Exl3Codebook::Cb2Mul1, false, Some(128), 5);
    }

    /// Tier-1 oracle: the trellis decode stage is BIT-EXACT (zero float
    /// arithmetic — integer prefix/window math + the CPU's own LUT bytes).
    /// Pins the closed-form prefix sum, MSB-first window extraction with ring
    /// wrap, and the tensor-core placement, across integer/half K and both
    /// scale spellings.
    #[test]
    fn gpu_decode_matches_cpu_bit_exact() {
        use riir_infer_core::quant::exl3::{decode_tile_rot, ring_pos_to_tile_element};

        for (k, codebook) in [
            (Exl3K { ka: 3, half: false }, Exl3Codebook::Cb0),
            (Exl3K { ka: 5, half: false }, Exl3Codebook::Cb1Mcg),
            (Exl3K { ka: 4, half: true }, Exl3Codebook::Cb2Mul1),
        ] {
            let (in_f, out_f) = (256usize, 384usize);
            let synth = SynthLayer::new(in_f, out_f, k, codebook, false, 7);
            // Construct + validate through the same path the driver takes.
            let _layer = synth.layer();

            // CPU w_rot via the reference tile decoder + placement.
            let mut cpu = vec![0.0f32; in_f * out_f];
            let tile_bytes = k.words_per_tile() * 2;
            let cols_tiles = out_f / 16;
            let mut tile = [0.0f32; 256];
            for a in 0..in_f / 16 {
                for c in 0..cols_tiles {
                    let off = (a * cols_tiles + c) * tile_bytes;
                    decode_tile_rot(&mut tile, &synth.trellis[off..off + tile_bytes], k, codebook);
                    for (p, &v) in tile.iter().enumerate() {
                        let (r, co) = ring_pos_to_tile_element(p);
                        cpu[(a * 16 + r) * out_f + c * 16 + co] = v;
                    }
                }
            }

            let ctx = CubeCLContext::new().expect("CubeCL should initialize");
            let client = ctx.client();
            let trellis = trellis_as_u32(&synth.trellis);
            let trellis_h = client.create_from_slice(u32::as_bytes(&trellis));
            let lut = codebook_lut(codebook);
            let lut_h = client.create_from_slice(f32::as_bytes(&lut[..]));
            let elems = in_f * out_f;
            let w_rot_h = client.empty(elems * 4);
            let total = ((in_f / 16) * (out_f / 16) * 256) as u32;
            let (n_wg, n_threads) = stride_grid(total);
            unsafe {
                exl3_trellis_decode::launch_unchecked::<ActiveRuntime>(
                    &client,
                    CubeCount::Static(n_wg, 1, 1),
                    CubeDim::new_1d(EXL3_THREADS),
                    BufferArg::from_raw_parts(trellis_h, trellis.len()),
                    BufferArg::from_raw_parts(lut_h, lut.len()),
                    BufferArg::from_raw_parts(w_rot_h.clone(), elems),
                    0,
                    (out_f / 16) as u32,
                    (out_f / 16) as u32,
                    out_f as u32,
                    k.stream_bits_per_tile() as u32 >> 5,
                    k.stream_bits_per_tile() as u32,
                    k.ka as u32,
                    u32::from(k.half),
                    total,
                    n_threads,
                );
            }
            let gpu = f32::from_bytes(&client.read_one(w_rot_h).unwrap()).to_vec();
            assert_eq!(cpu.len(), gpu.len());
            let bad = cpu
                .iter()
                .zip(&gpu)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            assert_eq!(
                bad, 0,
                "decode stage not bit-exact for K={}.{} cb={codebook:?}",
                k.ka,
                if k.half { 5 } else { 0 }
            );
        }
    }

    /// Real-pack parity + throughput (Issue 001 T7b record): the 5 sample
    /// classes from the §15 CPU bench, tier-2 (FMA-contraction class) compare
    /// vs the CPU parallel arm, GPU wall + kernel-only timing. Opt-in via
    /// `EXL3_PACK_DIR`; skips loudly when unset (the T5 convention).
    #[test]
    #[ignore = "needs a real EXL3 pack on disk (EXL3_PACK_DIR) + a GPU"]
    fn real_pack_gpu_parity_and_throughput() {
        use riir_infer_core::quant::exl3_pack::Exl3Pack;

        let dir = std::env::var("EXL3_PACK_DIR").unwrap_or_default();
        if dir.is_empty() {
            eprintln!("SKIPPED: EXL3_PACK_DIR not set");
            return;
        }
        let pack = Exl3Pack::open(std::path::Path::new(&dir)).unwrap();
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let classes = [
            "self_attn.o_proj",
            "mlp.down_proj",
            "linear_attn.out_proj",
            "mlp.gate_proj",
            "lm_head",
        ];
        for class in classes {
            let Some(plan) = pack
                .plans()
                .iter()
                .filter(|p| p.key.contains(class))
                .min_by_key(|p| p.in_features as u64 * p.out_features as u64)
            else {
                eprintln!("class {class}: none");
                continue;
            };
            let layer = pack.layer(&plan.key).unwrap();

            // Warmup pass (JIT) + the timed pair.
            let _ = Exl3DequantCubeCL::dequant_layer_f32::<ActiveRuntime>(&client, &layer).unwrap();
            let (gpu, timing) =
                Exl3DequantCubeCL::dequant_layer_f32_timed::<ActiveRuntime>(&client, &layer, 4)
                    .unwrap();

            let t_cpu = std::time::Instant::now();
            let cpu = layer.dequantize_f32_parallel();
            let cpu_secs = t_cpu.elapsed().as_secs_f64();

            let mw_wall = timing.weights as f64 / timing.wall_secs / 1e6;
            let mw_kernel = timing
                .kernel_secs.map_or(f64::NAN, |k| timing.weights as f64 / k / 1e6);
            eprintln!(
                "{class}: in={} out={} K={}.{} weights={} | GPU wall {:.3}s ({:.1} Mw/s) \
                 kernel ~{:.3}s ({:.1} Mw/s) | CPU-parallel {:.3}s ({:.1} Mw/s, {:.1}x)",
                plan.in_features,
                plan.out_features,
                plan.k.ka,
                if plan.k.half { 5 } else { 0 },
                timing.weights,
                timing.wall_secs,
                mw_wall,
                timing.kernel_secs.unwrap_or(f64::NAN),
                mw_kernel,
                cpu_secs,
                timing.weights as f64 / cpu_secs / 1e6,
                cpu_secs / timing.kernel_secs.unwrap_or(f64::NAN),
            );
            // Tier-2 oracle: the FMA-contraction divergence class.
            assert_w_within_contraction_class(&cpu, &gpu, class);
        }
    }

    /// Real-pack tier-1 probe: the decode stage bit-exact on the pack's own
    /// trellis bytes — the §12.7-era consistency check for whatever pack sits
    /// under `EXL3_PACK_DIR`, on the smallest o_proj-style layer.
    #[test]
    #[ignore = "needs a real EXL3 pack on disk (EXL3_PACK_DIR) + a GPU"]
    fn real_pack_decode_bit_exact_probe() {
        use riir_infer_core::quant::exl3::{decode_tile_rot, ring_pos_to_tile_element};
        use riir_infer_core::quant::exl3_pack::Exl3Pack;

        let dir = std::env::var("EXL3_PACK_DIR").unwrap_or_default();
        if dir.is_empty() {
            eprintln!("SKIPPED: EXL3_PACK_DIR not set");
            return;
        }
        let pack = Exl3Pack::open(std::path::Path::new(&dir)).unwrap();
        // Smallest layer ≤ 64M weights (bounded w_rot readback, ≤ 256 MB f32).
        let plan = pack
            .plans()
            .iter()
            .filter(|p| p.in_features as u64 * p.out_features as u64 <= 64_000_000)
            .min_by_key(|p| p.in_features as u64 * p.out_features as u64)
            .expect("pack carries a layer within the probe budget");
        let layer = pack.layer(&plan.key).unwrap();
        let (in_f, out_f) = (layer.in_features, layer.out_features);
        let (k, codebook) = (layer.k, layer.codebook);

        // CPU decode of the first probe_a rows' tiles — ALL columns (the
        // probe region is probe_a·16 × out_f).
        let tile_bytes = k.words_per_tile() * 2;
        let cols_tiles = out_f / 16;
        let probe_a = 8.min(in_f / 16);
        let mut cpu = vec![0.0f32; probe_a * 16 * out_f];
        let mut tile = [0.0f32; 256];
        for a in 0..probe_a {
            for c in 0..cols_tiles {
                let off = (a * cols_tiles + c) * tile_bytes;
                decode_tile_rot(&mut tile, &layer.trellis_bytes()[off..off + tile_bytes], k, codebook);
                for (p, &v) in tile.iter().enumerate() {
                    let (r, co) = ring_pos_to_tile_element(p);
                    cpu[(a * 16 + r) * out_f + c * 16 + co] = v;
                }
            }
        }

        // GPU decode of the same tiles: launch over the first probe_a rows'
        // tiles only is awkward — decode the WHOLE layer and compare the probe
        // region (the full readback is bounded by the chunk budget anyway).
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();
        let trellis = trellis_as_u32(layer.trellis_bytes());
        let trellis_h = client.create_from_slice(u32::as_bytes(&trellis));
        let lut = codebook_lut(codebook);
        let lut_h = client.create_from_slice(f32::as_bytes(&lut[..]));
        let elems = in_f * out_f;
        let w_rot_h = client.empty(elems * 4);
        let total = ((in_f / 16) * (out_f / 16) * 256) as u32;
        let (n_wg, n_threads) = stride_grid(total);
        unsafe {
            exl3_trellis_decode::launch_unchecked::<ActiveRuntime>(
                &client,
                CubeCount::Static(n_wg, 1, 1),
                CubeDim::new_1d(EXL3_THREADS),
                BufferArg::from_raw_parts(trellis_h, trellis.len()),
                BufferArg::from_raw_parts(lut_h, lut.len()),
                BufferArg::from_raw_parts(w_rot_h.clone(), elems),
                0,
                (out_f / 16) as u32,
                (out_f / 16) as u32,
                out_f as u32,
                k.stream_bits_per_tile() as u32 >> 5,
                k.stream_bits_per_tile() as u32,
                k.ka as u32,
                u32::from(k.half),
                total,
                n_threads,
            );
        }
        let gpu = f32::from_bytes(&client.read_one(w_rot_h).unwrap()).to_vec();
        let probe_len = probe_a * 16 * out_f;
        let bad = cpu
            .iter()
            .zip(&gpu[..probe_len])
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        eprintln!(
            "decode probe on {} ({}x{}): {}/{} probe elements bit-exact",
            plan.key,
            in_f,
            out_f,
            probe_len - bad,
            probe_len
        );
        assert_eq!(bad, 0, "{}: decode stage not bit-exact vs the reference tile decoder", plan.key);
    }

    /// T7c-1a gate (§17.3 gate 1, synthetic half): v2 word-aligned extraction
    /// is BIT-EXACT vs v1 AND vs the CPU reference tile decoder, across
    /// integer/half K + codebooks (the always-green arm; the full-pack gate
    /// is T7c-1c, with the real-pack bench asserting v1-vs-v2 per layer).
    #[test]
    fn gpu_decode_v2_bit_exact_vs_v1_and_cpu() {
        use riir_infer_core::quant::exl3::{decode_tile_rot, ring_pos_to_tile_element};

        for (k, codebook) in [
            (Exl3K { ka: 3, half: false }, Exl3Codebook::Cb0),
            (Exl3K { ka: 5, half: false }, Exl3Codebook::Cb1Mcg),
            (Exl3K { ka: 4, half: true }, Exl3Codebook::Cb2Mul1),
            (Exl3K { ka: 2, half: false }, Exl3Codebook::Cb0),
        ] {
            let (in_f, out_f) = (384usize, 256usize);
            let synth = SynthLayer::new(in_f, out_f, k, codebook, false, 11);

            // CPU w_rot via the reference tile decoder + placement.
            let mut cpu = vec![0.0f32; in_f * out_f];
            let tile_bytes = k.words_per_tile() * 2;
            let cols_tiles = out_f / 16;
            let mut tile = [0.0f32; 256];
            for a in 0..in_f / 16 {
                for c in 0..cols_tiles {
                    let off = (a * cols_tiles + c) * tile_bytes;
                    decode_tile_rot(&mut tile, &synth.trellis[off..off + tile_bytes], k, codebook);
                    for (p, &v) in tile.iter().enumerate() {
                        let (r, co) = ring_pos_to_tile_element(p);
                        cpu[(a * 16 + r) * out_f + c * 16 + co] = v;
                    }
                }
            }

            let ctx = CubeCLContext::new().expect("CubeCL should initialize");
            let client = ctx.client();
            let trellis = trellis_as_u32(&synth.trellis);
            let trellis_h = client.create_from_slice(u32::as_bytes(&trellis));
            let lut = codebook_lut(codebook);
            let lut_h = client.create_from_slice(f32::as_bytes(&lut[..]));
            let elems = in_f * out_f;
            let total = ((in_f / 16) * (out_f / 16) * 256) as u32;
            let (n_wg, n_threads) = stride_grid(total);

            let run = |use_v2: bool| -> Vec<f32> {
                let w_rot_h = client.empty(elems * 4);
                unsafe {
                    let args = (
                        BufferArg::from_raw_parts(trellis_h.clone(), trellis.len()),
                        BufferArg::from_raw_parts(lut_h.clone(), lut.len()),
                        BufferArg::from_raw_parts(w_rot_h.clone(), elems),
                        0u32,
                        (out_f / 16) as u32,
                        (out_f / 16) as u32,
                        out_f as u32,
                        k.stream_bits_per_tile() as u32 >> 5,
                        k.stream_bits_per_tile() as u32,
                        k.ka as u32,
                        u32::from(k.half),
                        total,
                        n_threads,
                    );
                    if use_v2 {
                        exl3_trellis_decode_v2::launch_unchecked::<ActiveRuntime>(
                            &client,
                            CubeCount::Static(n_wg, 1, 1),
                            CubeDim::new_1d(EXL3_THREADS),
                            args.0,
                            args.1,
                            args.2,
                            args.3,
                            args.4,
                            args.5,
                            args.6,
                            args.7,
                            args.8,
                            args.9,
                            args.10,
                            args.11,
                            args.12,
                        );
                    } else {
                        exl3_trellis_decode::launch_unchecked::<ActiveRuntime>(
                            &client,
                            CubeCount::Static(n_wg, 1, 1),
                            CubeDim::new_1d(EXL3_THREADS),
                            args.0,
                            args.1,
                            args.2,
                            args.3,
                            args.4,
                            args.5,
                            args.6,
                            args.7,
                            args.8,
                            args.9,
                            args.10,
                            args.11,
                            args.12,
                        );
                    }
                }
                f32::from_bytes(&client.read_one(w_rot_h).unwrap()).to_vec()
            };

            let v1 = run(false);
            let v2 = run(true);
            let what = format!("K={}.{} cb={codebook:?}", k.ka, if k.half { 5 } else { 0 });
            let bad_v1v2 = v1.iter().zip(&v2).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
            assert_eq!(bad_v1v2, 0, "{what}: v2 decode not bit-exact vs v1");
            let bad_cpu = cpu.iter().zip(&v2).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
            assert_eq!(bad_cpu, 0, "{what}: v2 decode not bit-exact vs the CPU reference tile decoder");
        }
    }

    /// T7c-1b (§17.2): the discriminating decode bench on the REAL pack —
    /// arms A1 (v1) / A2 (v2) / A3 (v2-no-LUT) over the 5 sample classes,
    /// Gw/s kernel-only (reps-differential), v2-vs-v1 BIT-EXACT asserted per
    /// layer (the §17.3 gate-1 shape at bench scope; the FULL-pack gate is
    /// T7c-1c). NO throughput assert — the kill criterion (§17.2) records a
    /// bound, it does not fail the lane. Opt-in via `EXL3_PACK_DIR`.
    #[test]
    #[ignore = "needs a real EXL3 pack on disk (EXL3_PACK_DIR) + a GPU"]
    fn real_pack_decode_bench_arms() {
        use riir_infer_core::quant::exl3_pack::Exl3Pack;

        let dir = std::env::var("EXL3_PACK_DIR").unwrap_or_default();
        if dir.is_empty() {
            eprintln!("SKIPPED: EXL3_PACK_DIR not set");
            return;
        }
        let pack = Exl3Pack::open(std::path::Path::new(&dir)).unwrap();
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let classes = [
            "self_attn.o_proj",
            "mlp.down_proj",
            "linear_attn.out_proj",
            "mlp.gate_proj",
            "lm_head",
        ];
        const REPS: usize = 8;

        let mut rows: Vec<(&str, u64, f64, f64, f64)> = Vec::new();
        for class in classes {
            let Some(plan) = pack
                .plans()
                .iter()
                .filter(|p| p.key.contains(class))
                .min_by_key(|p| p.in_features as u64 * p.out_features as u64)
            else {
                eprintln!("class {class}: none");
                continue;
            };
            let key = plan.key.as_str();
            let layer = pack.layer(key).unwrap();

            // Interleaved rounds × arms, per-arm BEST kernel_secs (the
            // Issue-723 best-of convention: GPU clock ramp makes single
            // pairs oscillate wildly — measured: v1 gate_proj 62.4 → 9.1 Gw/s
            // between consecutive single-pair runs on a quiet box).
            const ROUNDS: usize = 5;
            let arms = [DecodeArm::V1, DecodeArm::V2, DecodeArm::V2NoLut];
            let mut best = [f64::INFINITY; 3];
            let mut v1_out: Option<Vec<f32>> = None;
            let mut v2_out: Option<Vec<f32>> = None;
            for _ in 0..ROUNDS {
                for (i, &arm) in arms.iter().enumerate() {
                    let (w, t) = Exl3DequantCubeCL::decode_only_layer_timed::<ActiveRuntime>(
                        &client, &layer, arm, REPS,
                    )
                    .unwrap();
                    if let Some(s) = t.kernel_secs {
                        best[i] = best[i].min(s);
                    }
                    match arm {
                        DecodeArm::V1 => v1_out = Some(w),
                        DecodeArm::V2 => v2_out = Some(w),
                        DecodeArm::V2NoLut => {}
                    }
                }
            }

            // §17.3 gate 1 at bench scope: v2 vs v1 BIT-EXACT on this layer.
            let (v1, v2) = (v1_out.unwrap(), v2_out.unwrap());
            let bad = v1.iter().zip(&v2).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
            assert_eq!(bad, 0, "{key}: v2 decode not bit-exact vs v1 on the real pack");

            let weights = (layer.in_features * layer.out_features) as u64;
            let gw = |secs: f64| weights as f64 / secs / 1e9;
            rows.push((key, weights, gw(best[0]), gw(best[1]), gw(best[2])));
        }

        eprintln!("\n=== T7c-1b discriminating decode bench (REPS={REPS}) ===");
        eprintln!("| layer | weights | A1 v1 Gw/s | A2 v2 Gw/s | A3 v2-noLUT Gw/s | v2/v1 |");
        eprintln!("|---|---:|---:|---:|---:|---:|");
        for (key, w, a1, a2, a3) in &rows {
            eprintln!("| {key} | {w} | {a1:.1} | {a2:.1} | {a3:.1} | {:.2}x |", a2 / a1.max(1e-9));
        }
        let total_w: u64 = rows.iter().map(|r| r.1).sum();
        let k1: f64 = rows.iter().map(|r| r.2 * r.1 as f64).sum::<f64>() / total_w as f64;
        let k2: f64 = rows.iter().map(|r| r.3 * r.1 as f64).sum::<f64>() / total_w as f64;
        let k3: f64 = rows.iter().map(|r| r.4 * r.1 as f64).sum::<f64>() / total_w as f64;
        eprintln!(
            "weight-weighted mean: A1 {k1:.1} / A2 {k2:.1} / A3 {k3:.1} Gw/s; A2/A1 {:.2}x",
            k2 / k1.max(1e-9)
        );
    }

    /// T7c-1d (plan 003): the STABLE harness replacing T7c-1b's noisy
    /// differential-only readout — sync-bracketed sampling, min/median/p90,
    /// the ≥5 ms work floor (TooFastToTime), the 10% instability gate, and
    /// the 15% cross-method agreement gate. Prints the full stable table +
    /// the per-arm verdicts; NO throughput assert (the §17.2 kill-criterion
    /// language records bounds, it does not fail the lane). Opt-in via
    /// `EXL3_PACK_DIR` + a GPU.
    #[test]
    #[ignore = "needs a real EXL3 pack on disk (EXL3_PACK_DIR) + a GPU"]
    fn real_pack_decode_bench_stable() {
        use riir_infer_core::quant::exl3_pack::Exl3Pack;

        let dir = std::env::var("EXL3_PACK_DIR").unwrap_or_default();
        if dir.is_empty() {
            eprintln!("SKIPPED: EXL3_PACK_DIR not set");
            return;
        }
        let pack = Exl3Pack::open(std::path::Path::new(&dir)).unwrap();
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let classes = [
            "self_attn.o_proj",
            "mlp.down_proj",
            "linear_attn.out_proj",
            "mlp.gate_proj",
            "lm_head",
        ];
        // Reps scaled so even the smallest class clears the 5 ms/pass floor;
        // the harness re-refuses with TooFastToTime if a row still lands short
        // (the caller's signal to raise reps — the error names the floor).
        const REPS: usize = 64;
        const SAMPLES: usize = 30;

        eprintln!(
            "\n=== T7c-1d stable decode bench (REPS={REPS} SAMPLES={SAMPLES}) ==="
        );
        eprintln!("| layer | weights | arm | peak Gw/s | median | p90 | spread | cross-agree | verdict |" );
        eprintln!("|---|---:|---|---:|---:|---:|---:|---|---|");

        for class in classes {
            let Some(plan) = pack
                .plans()
                .iter()
                .filter(|p| p.key.contains(class))
                .min_by_key(|p| p.in_features as u64 * p.out_features as u64)
            else {
                eprintln!("class {class}: none");
                continue;
            };
            let layer = pack.layer(&plan.key).unwrap();

            // §17.3 gate 1 at bench scope (re-asserted every run): v2 vs v1.
            let (v1, _) = Exl3DequantCubeCL::decode_only_layer::<ActiveRuntime>(
                &client, &layer, DecodeArm::V1, 1,
            )
            .unwrap();
            let (v2, _) = Exl3DequantCubeCL::decode_only_layer::<ActiveRuntime>(
                &client, &layer, DecodeArm::V2, 1,
            )
            .unwrap();
            let bad = v1.iter().zip(&v2).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
            assert_eq!(bad, 0, "{}: v2 not bit-exact vs v1", plan.key);

            for (name, arm) in [
                ("A1-v1", DecodeArm::V1),
                ("A2-v2", DecodeArm::V2),
                ("A3-noLUT", DecodeArm::V2NoLut),
            ] {
                match Exl3DequantCubeCL::bench_arm_stable::<ActiveRuntime>(
                    &client, &layer, arm, REPS, SAMPLES,
                ) {
                    Ok(b) => eprintln!(
                        "| {} | {} | {name} | {:.1} | {:.1} | {:.1} | {:.1}% | {} | {} |",
                        plan.key,
                        b.weights,
                        b.gw_peak(),
                        b.gw_median(),
                        b.weights as f64 / b.p90_secs / 1e9,
                        b.spread * 100.0,
                        if b.cross_agrees { "YES" } else { "no" },
                        if b.unstable { "UNSTABLE" } else { "ok" },
                    ),
                    Err(Exl3DequantError::TooFastToTime { pass_secs, reps }) => eprintln!(
                        "| {} | — | {name} | — | — | — | — | — | REFUSED: pass {pass_secs:.3}ms < 5ms floor at reps={reps} — raise reps |",
                        plan.key
                    ),
                    Err(e) => panic!("bench_arm_stable failed: {e}"),
                }
            }
        }
    }

    /// T7c-1c gate (§17.5): the FULL-PACK bit-exact gate — EVERY layer in
    /// the real pack, never a sample (§17.3 gate 1 at whole-pack scope).
    /// Three comparisons per layer: v2-vs-v1 (the literal gate-1 pair),
    /// v2-vs-CPU and v1-vs-CPU against `decode_w_rot_f32` (the T5 oracle
    /// reference, decode-only). The synthetic fixtures cover K 2/3/4.5/5 —
    /// this pack spans K 3.0–6.0, so combinations like K6 and the half-
    /// integers are exercised HERE for the first time; that coverage gap is
    /// the reason a whole-pack gate exists. Prints the per-K×codebook
    /// coverage table (a green run over a shrunken pack proves nothing —
    /// coverage floors pinned below) and asserts ZERO bit mismatches. Opt-in
    /// via `EXL3_PACK_DIR` + a GPU (~50 min on the 4090: every layer decoded
    /// three times).
    #[test]
    #[ignore = "needs a real EXL3 pack on disk (EXL3_PACK_DIR) + a GPU + ~1 h"]
    fn real_pack_v2_bit_exact_full() {
        use riir_infer_core::quant::exl3_pack::Exl3Pack;
        use std::collections::BTreeMap;

        let dir = std::env::var("EXL3_PACK_DIR").unwrap_or_default();
        if dir.is_empty() {
            eprintln!("SKIPPED: EXL3_PACK_DIR not set");
            return;
        }
        let pack = Exl3Pack::open(std::path::Path::new(&dir)).unwrap();
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let plans = pack.plans();
        // Coverage floor (the blindness guard): the T5-measured pack carries
        // 573 layer groups — a loader regression that detects fewer must RED
        // here, not print a green zero over an empty population.
        assert!(
            plans.len() >= 500,
            "pack exposes {} layer plans, below the T5-measured 573-group floor — loader regression?",
            plans.len()
        );

        let k_of = |k: Exl3K| if k.half { format!("K{}.5", k.ka) } else { format!("K{}", k.ka) };
        let cb_of = |cb: Exl3Codebook| match cb {
            Exl3Codebook::Cb0 => "Cb0",
            Exl3Codebook::Cb1Mcg => "Cb1Mcg",
            Exl3Codebook::Cb2Mul1 => "Cb2Mul1",
        };
        let mut coverage: BTreeMap<String, (usize, u64)> = BTreeMap::new();
        let mut total_weights = 0u64;
        let mut bad_layers = 0usize;
        let mut failures: Vec<String> = Vec::new();
        let t0 = std::time::Instant::now();

        eprintln!("\n=== T7c-1c full-pack bit-exact gate: {} layers ===", plans.len());
        for (i, plan) in plans.iter().enumerate() {
            let layer = pack.layer(&plan.key).unwrap();
            let n = (layer.in_features as u64) * (layer.out_features as u64);
            let (v1, _) = Exl3DequantCubeCL::decode_only_layer::<ActiveRuntime>(
                &client, &layer, DecodeArm::V1, 1,
            )
            .unwrap();
            let (v2, _) = Exl3DequantCubeCL::decode_only_layer::<ActiveRuntime>(
                &client, &layer, DecodeArm::V2, 1,
            )
            .unwrap();
            let bad_v2v1 = v1.iter().zip(&v2).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
            drop(v1); // peak host memory: 2 × largest layer (lm_head ⇒ ~10 GiB)
            let cpu = layer.decode_w_rot_f32();
            let bad_v2cpu = v2.iter().zip(&cpu).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
            drop(v2);
            drop(cpu);

            let entry = coverage
                .entry(format!("{}|{}", k_of(plan.k), cb_of(plan.codebook)))
                .or_insert((0usize, 0u64));
            entry.0 += 1;
            entry.1 += n;
            total_weights += n;

            eprintln!(
                "[{}/{}] {} {} {} {}w v2v1={} v2cpu={}",
                i + 1,
                plans.len(),
                plan.key,
                k_of(plan.k),
                cb_of(plan.codebook),
                n,
                bad_v2v1,
                bad_v2cpu
            );
            if bad_v2v1 > 0 || bad_v2cpu > 0 {
                bad_layers += 1;
                // Re-decode the failing pair only when a failure exists, to
                // name the first divergent element + bit patterns (the
                // reproduction datum for a targeted probe).
                if failures.len() < 20 {
                    let (v1, _) = Exl3DequantCubeCL::decode_only_layer::<ActiveRuntime>(
                        &client, &layer, DecodeArm::V1, 1,
                    )
                    .unwrap();
                    let (v2, _) = Exl3DequantCubeCL::decode_only_layer::<ActiveRuntime>(
                        &client, &layer, DecodeArm::V2, 1,
                    )
                    .unwrap();
                    let cpu = layer.decode_w_rot_f32();
                    let trip = |what: &str, a: &[f32], b: &[f32]| -> String {
                        match a.iter().zip(b).position(|(x, y)| x.to_bits() != y.to_bits()) {
                            Some(idx) => format!(
                                "{}: first mismatch at elem {idx} (in {}, out {}): {:#x} vs {:#x}",
                                what,
                                idx / layer.out_features,
                                idx % layer.out_features,
                                a[idx].to_bits(),
                                b[idx].to_bits()
                            ),
                            None => format!("{what}: no mismatch on re-decode (was {} — nondeterministic!)",
                                if what.contains("v2cpu") { bad_v2cpu } else { bad_v2v1 }),
                        }
                    };
                    failures.push(format!(
                        "{} {} {} {}w:\n  {}\n  {}",
                        plan.key,
                        k_of(plan.k),
                        cb_of(plan.codebook),
                        n,
                        trip("v2-vs-v1", &v2, &v1),
                        trip("v2-vs-cpu", &v2, &cpu),
                    ));
                }
            }
        }

        eprintln!("\n=== coverage (K|codebook: layers, weights) ===");
        for (key, (layers, weights)) in &coverage {
            eprintln!("  {key}: {layers} layers, {weights} weights");
        }
        eprintln!(
            "total: {} layers, {total_weights} weights, {bad_layers} bad, wall {:.1} s",
            plans.len(),
            t0.elapsed().as_secs_f64()
        );

        // Coverage floors: the T5-measured pack is ~26.0–26.5 G quantized
        // weights over ≥3 K×codebook classes — a run that saw less did not
        // measure the pack.
        assert!(
            total_weights >= 26_000_000_000,
            "compared only {total_weights} weights, below the T5-measured ~26 G floor"
        );
        assert!(
            coverage.len() >= 3,
            "only {} K×codebook classes seen, below the mixed-K recipe floor of 3",
            coverage.len()
        );
        assert!(
            failures.is_empty(),
            "v2 decode DIVERGES on {bad_layers}/{} layers:\n{}",
            plans.len(),
            failures.join("\n")
        );
    }
}
