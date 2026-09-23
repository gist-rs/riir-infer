//! Issue 771 T2c-a (Plan 562): the Q8-KV prefill flash arm — the
//! m4-prefill-engine transplant (Research 360), tolerance-class, DEFAULT-OFF.
//!
//! ## The design (and where it diverges from m4)
//!
//! m4-prefill-engine quantizes K/V to Q8_0 once and dequantizes at tile load
//! into threadgroup memory. Our prefill flash family is the plane idiom
//! (register-resident, zero smem, zero barriers —
//! `qwen_attention_prefill_tiled_f32`), so the adaptation is
//! **dequant-in-register at load**: each lane owns dims `[lane*8, lane*8+8)`
//! which sit entirely inside ONE Q8_0 block (block = `lane/4`; the lane's 8
//! bytes start at word `(lane%4)*2` of the block's 8 words), so a K or V row
//! load drops from 32 B f32 to 8 B i8 + one f32 scale shared by 4 lanes.
//!
//! ## Layout contract (decode-compatible strides, separate K/V buffers)
//!
//! Same per-position strides as the decode-side `Q8KVBuffers` (Plan 106 T2.9),
//! with the f32 scale stored directly (internal GPU-side format — the CPU
//! `BlockQ8_0` wire format keeps its f16 scale; nothing crosses a wire here):
//!
//! ```text
//! qs:     [n_pos × kv_stride_q8] u32,  kv_stride_q8     = n_kv × n_blocks × 8
//! scales: [n_pos × kv_scale_stride] f32, kv_scale_stride = n_kv × n_blocks
//! ```
//!
//! head_dim 256 → n_blocks 8. `d = max_abs / 127`, `q = round(v / d)` clamped
//! to ±127, all-zero block → q = 0 (the CPU `quantize_row_q8_0` contract,
//! `riir_infer_core::quant::q8kv`).
//!
//! ## Numerics: TOLERANCE ARM (NOT bit-identical — by construction)
//!
//! Q8-KV error rides ON TOP of the tiled kernel's documented FP-equivalent
//! class. The behavior gate is the argmax-flip sweep + the tolerance band
//! (Issue 771 flash rule); it breaks no f32 anchor because it never claims
//! one. `RIIR_Q8KV_PREFILL=1` (or the setter) routes it at
//! `p >= PREFILL_Q8KV_MIN_P`; the serving arms (m16/cmma) are untouched when
//! the toggle is off.
//!
//! ## Honest cost note (chunked prefill)
//!
//! The arm quantizes the full `[0, base_pos+p)` cache range per layer call —
//! chunked prefill (chunk ≤ 4096) therefore requantizes the prefix each chunk.
//! That O(P)-per-call quantize pass is part of what the A/B prices; a
//! persistent per-layer Q8 mirror cache is the follow-up if the arm wins and
//! the requantize term prices measurably (Plan 562 non-goals).
#[cfg(feature = "cubecl_runtime")]
use cubecl::prelude::*;
#[cfg(feature = "cubecl_runtime")]
use cubecl::server::Handle;

/// Per-forward Q8-KV scratch: one K/V qs+scales pair sized for `rows` cache
/// rows, reused across layers within a forward (attention consumes the
/// quantized rows within the same layer call). Allocated lazily at the first
/// Q8-arm dispatch — a default build never pays it.
// Issue 936: gated with the module's only consumer lane — the ternary batched-prefill
// ladder in `ternary_deltanet_gpu_forward` (`all(cubecl_runtime, ternary_gemm_batched,
// ternary_attention_batched_prefill)`; the module gate below already supplies
// `cubecl_runtime`). Ungated, the fields + `buffer_bytes` read as dead on every
// lane that compiles this module without that ladder (the riir-clippy consumer set).
#[cfg(all(feature = "ternary_gemm_batched", feature = "ternary_attention_batched_prefill"))]
#[derive(Debug)]
pub struct Q8PrefillScratch {
    // pub (not pub(crate)) — the one consumer (ternary_deltanet_gpu_forward,
    // still in the engine gpu crate until Plan 610 S4b) constructs the struct
    // through the module re-export; pub(crate) is invisible across the crate
    // boundary (the S3 widening class).
    pub key_qs: Handle,
    pub key_scales: Handle,
    pub value_qs: Handle,
    pub value_scales: Handle,
    /// Rows the buffers are sized for.
    pub rows: usize,
}

#[cfg(all(feature = "ternary_gemm_batched", feature = "ternary_attention_batched_prefill"))]
impl Q8PrefillScratch {
    /// `(qs_bytes, scales_bytes)` per buffer at `(rows, n_kv, head_dim)`.
    pub fn buffer_bytes(rows: usize, n_kv: usize, head_dim: usize) -> (usize, usize) {
        let blocks_per_row = n_kv * (head_dim / 32);
        (
            rows * blocks_per_row * 8 * core::mem::size_of::<u32>(),
            rows * blocks_per_row * core::mem::size_of::<f32>(),
        )
    }
}

// ---------------------------------------------------------------------------
// Kernel-side helpers
// ---------------------------------------------------------------------------
//
// Local copies of the decode kernel's byte-extraction pair
// (`attention_q8kv_cubecl.rs` `get_byte_q8`/`sign_extend_i8` — 3-liners,
// deliberately NOT imported: cross-module `#[cube]` fn calls go through
// macro codegen this crate has never exercised; the duplication is cheaper
// than the risk. If the pair ever grows, revisit).

/// Sign-extend a byte to f32 and apply the block scale.
///
/// Two if-blocks (no conditional expressions as values) — the CubeCL
/// NativeExpand macro bug workaround the decode kernel documents.
#[cfg(feature = "cubecl_runtime")]
#[cube]
fn q8_dequant_byte(word: u32, idx: u32, scale: f32) -> f32 {
    let b = (word >> (idx * 8u32)) & 0xFFu32;
    let mut v = b as f32;
    if b >= 128u32 {
        v = b as f32 - f32::new(256.0f32);
    }
    v * scale
}

// ---------------------------------------------------------------------------
// T1.1: the KV quantizer
// ---------------------------------------------------------------------------

/// Quantize post-RoPE K (or V) cache rows to Q8_0 side buffers.
///
/// One thread per (position, kv_head, block): pass 1 finds the block max-abs
/// over 32 f32 loads, pass 2 re-reads (L1-hot) and packs 32 quantized bytes
/// into 8 u32 words. Chunk-local indexing: the launcher slices all buffers at
/// row granularity, so `pos` here is chunk-relative.
///
/// # Buffer contract (per launch)
/// - `src`: `[rows × n_kv × head_dim]` f32 (row-major cache rows).
/// - `qs_out`: `[rows × n_kv × n_blocks × 8]` u32.
/// - `scales_out`: `[rows × n_kv × n_blocks]` f32.
/// - `params`: `[head_dim, n_kv, n_blocks, rows]`.
#[cfg(feature = "cubecl_runtime")]
#[allow(clippy::assign_op_pattern, reason = "CubeCL macro expansion")]
#[cube(launch_unchecked)]
fn qwen_kv_quantize_q8_batched_f32(
    src: &[f32],
    qs_out: &mut [u32],
    scales_out: &mut [f32],
    params: &[f32],
) {
    let head_dim = params[0usize] as usize;
    let n_kv = params[1usize] as usize;
    let n_blocks = params[2usize] as usize;
    let rows = params[3usize] as usize;

    let idx = ABSOLUTE_POS;
    let blocks_per_row = n_kv * n_blocks;
    if idx >= rows * blocks_per_row {
        terminate!();
    }

    let pos = idx / blocks_per_row;
    let within_row = idx % blocks_per_row;
    let kv_head = within_row / n_blocks;
    let block = within_row % n_blocks;

    let b_base = pos * n_kv * head_dim + kv_head * head_dim + block * 32usize;

    // Pass 1: block max-abs (32 fixed loads, branch-free compare form — the
    // `gemm_ternary_cmma_i8_cubecl` row-max idiom).
    let zero = f32::new(0.0f32);
    let mut mx = zero;
    let mut e = 0usize;
    while e < 32usize {
        let v = src[b_base + e];
        let a = if v < zero { zero - v } else { v };
        mx = if a > mx { a } else { mx };
        e += 1usize;
    }

    // d = max_abs / 127; all-zero block → q = 0 (inv 0, d stored as 1.0 —
    // never read when q is all-zero, matching the CPU reference's d=0 slot).
    let d = if mx > zero {
        mx / f32::new(127.0f32)
    } else {
        f32::new(1.0f32)
    };
    let inv = if mx > zero {
        f32::new(127.0f32) / mx
    } else {
        zero
    };
    scales_out[pos * blocks_per_row + within_row] = d;

    // Pass 2: re-read (L1-hot), quantize, pack 4 bytes per word × 8 words.
    // Fixed bounds throughout — no dynamically-indexed register arrays (the
    // Batch-49 local-memory-demotion class).
    //
    // Rounding: explicit half-away-from-zero via a sign branch + the f32→i32
    // converting cast. NOT `.round()` — cubecl must SYNTHESIZE round (WGSL
    // has no op) and the lowering measured a truncation-class bias on
    // negatives: the readback gate (bench_836 t836_q8kv_quantizer_readback)
    // caught ~1-quantum diffs vs the CPU replica on ~half the elements
    // (max ≈ 1.0·d, mean ≈ 0.27·d) — exactly the floor(x+0.5) signature.
    // The f32→i32 convert is specced truncating on every backend, so the
    // kernel and the CPU replica agree bit-exactly.
    let half = f32::new(0.5f32);
    let hi127 = f32::new(127.0f32);
    let lo127 = f32::new(-127.0f32);
    let out_base = (pos * blocks_per_row + within_row) * 8usize;
    let mut w = 0usize;
    while w < 8usize {
        let mut word = 0u32;
        let mut j = 0usize;
        while j < 4usize {
            let x = src[b_base + w * 4usize + j];
            let xf = x * inv;
            let biased = if xf < zero { xf - half } else { xf + half };
            let q_i = i32::cast_from(biased);
            let qf = q_i as f32;
            let qf = if qf > hi127 {
                hi127
            } else if qf < lo127 {
                lo127
            } else {
                qf
            };
            // Same-width i32→u32 casts are bit-preserving; & 0xFF extracts
            // the two's-complement byte (the i8-GEMM pack idiom).
            word |= (u32::cast_from(i32::cast_from(qf)) & 0xFFu32) << ((j * 8) as u32);
            j += 1usize;
        }
        qs_out[out_base + w] = word;
        w += 1usize;
    }
}

/// Launch the KV quantizer over cache rows `[0, n_rows)`.
///
/// Row-granular chunking keeps every launch's buffer slices contiguous; at
/// Bonsai geometry one launch covers up to 262k rows (65535 workgroups ×
/// 256 threads ÷ n_kv/n_blocks), so the loop body runs once for every shape
/// the family serves — the guard exists for the 65535 contract, not for
/// expected iteration.
///
/// # Safety
/// Caller owns the handle-size contract: `src` must hold exactly
/// `n_rows × n_kv × head_dim` f32; `qs_out` `n_rows × n_kv × (head_dim/32) × 8`
/// u32; `scales_out` `n_rows × n_kv × (head_dim/32)` f32 (the kernel is
/// `launch_unchecked` — caller-owns-correctness, the family convention).
#[cfg(feature = "cubecl_runtime")]
#[allow(clippy::too_many_arguments, reason = "GPU kernel launch")]
pub unsafe fn launch_kv_quantize_q8<R: Runtime>(
    client: &ComputeClient<R>,
    src: Handle,
    qs_out: Handle,
    scales_out: Handle,
    n_rows: usize,
    n_kv: usize,
    head_dim: usize,
) {
    const MAX_WG_X: u32 = 65535;
    const THREADS: u32 = 256;

    let n_blocks = head_dim / 32;
    let params: [f32; 4] = [
        head_dim as f32,
        n_kv as f32,
        n_blocks as f32,
        n_rows as f32,
    ];
    let params_handle = crate::params_cache::params_handle(client, f32::as_bytes(&params));

    let blocks_per_row = n_kv * n_blocks;
    let rows_per_launch = MAX_WG_X as usize * THREADS as usize / blocks_per_row;
    let mut done = 0usize;
    while done < n_rows {
        let rows = (n_rows - done).min(rows_per_launch);
        let row_f = head_dim * n_kv; // f32s per row
        unsafe {
            qwen_kv_quantize_q8_batched_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(
                    (rows * blocks_per_row).div_ceil(THREADS as usize) as u32,
                    1,
                    1,
                ),
                CubeDim::new_1d(THREADS),
                BufferArg::from_raw_parts(
                    src.clone().offset_start((done * row_f * 4) as u64),
                    rows * row_f,
                ),
                BufferArg::from_raw_parts(
                    qs_out
                        .clone()
                        .offset_start((done * blocks_per_row * 8 * 4) as u64),
                    rows * blocks_per_row * 8,
                ),
                BufferArg::from_raw_parts(
                    scales_out
                        .clone()
                        .offset_start((done * blocks_per_row * 4) as u64),
                    rows * blocks_per_row,
                ),
                BufferArg::from_raw_parts(params_handle.clone(), 4),
            );
        }
        done += rows;
    }
}

// ---------------------------------------------------------------------------
// T1.2: the Q8 tiled flash attention kernel
// ---------------------------------------------------------------------------

/// Queries per cube — same shape as the f32 tiled kernel (`TILED_Q_PER_CUBE`).
const Q8_TILED_Q_PER_CUBE: u32 = 8;

/// Plane-per-query tiled causal flash prefill reading Q8_0 K/V — the T2c-a
/// arm of `qwen_attention_prefill_tiled_f32` (see the module doc for the
/// layout + numerics contract).
///
/// `params`: `[head_dim, n_head, n_kv_head, p, scale, q_offset, q_tiles,
/// n_blocks]` — one wider than the f32 kernel (n_blocks drives the q8
/// strides).
#[cfg(feature = "cubecl_runtime")]
#[allow(clippy::assign_op_pattern, reason = "CubeCL macro expansion")]
#[cube(launch_unchecked)]
fn qwen_attention_prefill_tiled_q8_f32(
    query: &[f32],
    key_qs: &[u32],
    key_scales: &[f32],
    value_qs: &[u32],
    value_scales: &[f32],
    gate: &[f32],
    attn_out: &mut [f32],
    params: &[f32],
) {
    let head_dim = params[0usize] as u32;
    let n_head = params[1usize] as u32;
    let n_kv_head = params[2usize] as u32;
    let p = params[3usize] as u32;
    let scale = params[4usize];
    let q_offset = params[5usize] as u32;
    let q_tiles = params[6usize] as u32;
    let n_blocks = params[7usize] as u32;

    let cube_id = CUBE_POS_X;
    let head_idx = cube_id / q_tiles;
    let q_tile = cube_id % q_tiles;

    let pl = UNIT_POS / 32u32;
    let lane = UNIT_POS_PLANE;

    let q_pos = q_tile * Q8_TILED_Q_PER_CUBE + pl;
    let active = q_pos < p;

    // GQA + strides (identical to the f32 kernel).
    let kv_group = head_idx * n_kv_head / n_head;
    let q_stride = n_head * head_dim;
    let q_pos_abs = q_pos + q_offset;
    let q_off = (q_pos * q_stride + head_idx * head_dim) as usize;
    let dims_base = (lane * 8u32) as usize;

    // Q8 strides + this lane's (block, word-pair) decomposition.
    let u32_per_head = n_blocks * 8u32;
    let q8_stride = n_kv_head * u32_per_head;
    let scale_stride = n_kv_head * n_blocks;
    let blk = lane >> 2u32;
    let wpair = (lane & 3u32) * 2u32;

    let q0 = if active { query[q_off + dims_base] } else { f32::new(0.0f32) };
    let q1 = if active { query[q_off + dims_base + 1usize] } else { f32::new(0.0f32) };
    let q2 = if active { query[q_off + dims_base + 2usize] } else { f32::new(0.0f32) };
    let q3 = if active { query[q_off + dims_base + 3usize] } else { f32::new(0.0f32) };
    let q4 = if active { query[q_off + dims_base + 4usize] } else { f32::new(0.0f32) };
    let q5 = if active { query[q_off + dims_base + 5usize] } else { f32::new(0.0f32) };
    let q6 = if active { query[q_off + dims_base + 6usize] } else { f32::new(0.0f32) };
    let q7 = if active { query[q_off + dims_base + 7usize] } else { f32::new(0.0f32) };

    // Online softmax state, per lane (uniform across the plane after every
    // plane_sum broadcast).
    let mut run_max = f32::new(-1e30f32);
    let mut run_sum = f32::new(0.0f32);
    let mut o0 = f32::new(0.0f32);
    let mut o1 = f32::new(0.0f32);
    let mut o2 = f32::new(0.0f32);
    let mut o3 = f32::new(0.0f32);
    let mut o4 = f32::new(0.0f32);
    let mut o5 = f32::new(0.0f32);
    let mut o6 = f32::new(0.0f32);
    let mut o7 = f32::new(0.0f32);

    // Uniform loop bound: the max causal position across the cube's 8 queries
    // (+1) — identical to the f32 kernel.
    let q_last = q_tile * Q8_TILED_Q_PER_CUBE + (Q8_TILED_Q_PER_CUBE - 1u32);
    let n_loop = if q_last < p { q_last } else { p - 1u32 } + q_offset + 1u32;

    let mut pos = 0u32;
    while pos < n_loop {
        // Q8_0 K row: one scale + two words per lane.
        let scale_idx = (pos * scale_stride + kv_group * n_blocks + blk) as usize;
        let k_scale = key_scales[scale_idx];
        let qs_base = (pos * q8_stride + kv_group * u32_per_head + blk * 8u32 + wpair) as usize;
        let kw0 = key_qs[qs_base];
        let kw1 = key_qs[qs_base + 1usize];

        let k0 = q8_dequant_byte(kw0, 0u32, k_scale);
        let k1 = q8_dequant_byte(kw0, 1u32, k_scale);
        let k2 = q8_dequant_byte(kw0, 2u32, k_scale);
        let k3 = q8_dequant_byte(kw0, 3u32, k_scale);
        let k4 = q8_dequant_byte(kw1, 0u32, k_scale);
        let k5 = q8_dequant_byte(kw1, 1u32, k_scale);
        let k6 = q8_dequant_byte(kw1, 2u32, k_scale);
        let k7 = q8_dequant_byte(kw1, 3u32, k_scale);

        let partial = q0 * k0 + q1 * k1 + q2 * k2 + q3 * k3
            + q4 * k4 + q5 * k5 + q6 * k6 + q7 * k7;
        // Reduce + broadcast within this plane's 32 lanes (tree order — the
        // documented FP-equivalent-not-bit-identical class).
        let score = plane_sum(partial) * scale;

        let in_causal = active && pos <= q_pos_abs;
        let masked = if in_causal { score } else { f32::new(-1e30f32) };

        let new_max = if masked > run_max { masked } else { run_max };
        let w = (masked - new_max).exp();
        let correction = (run_max - new_max).exp();
        run_sum = run_sum * correction + w;

        // Q8_0 V row — same decomposition as K.
        let v_scale = value_scales[scale_idx];
        let vw0 = value_qs[qs_base];
        let vw1 = value_qs[qs_base + 1usize];

        let v0 = q8_dequant_byte(vw0, 0u32, v_scale);
        let v1 = q8_dequant_byte(vw0, 1u32, v_scale);
        let v2 = q8_dequant_byte(vw0, 2u32, v_scale);
        let v3 = q8_dequant_byte(vw0, 3u32, v_scale);
        let v4 = q8_dequant_byte(vw1, 0u32, v_scale);
        let v5 = q8_dequant_byte(vw1, 1u32, v_scale);
        let v6 = q8_dequant_byte(vw1, 2u32, v_scale);
        let v7 = q8_dequant_byte(vw1, 3u32, v_scale);

        o0 = o0 * correction + w * v0;
        o1 = o1 * correction + w * v1;
        o2 = o2 * correction + w * v2;
        o3 = o3 * correction + w * v3;
        o4 = o4 * correction + w * v4;
        o5 = o5 * correction + w * v5;
        o6 = o6 * correction + w * v6;
        o7 = o7 * correction + w * v7;
        run_max = new_max;

        pos += 1u32;
    }

    if active {
        let inv_sum = f32::new(1.0f32) / run_sum;
        let g_off = q_off + dims_base;
        let g0 = gate[g_off];
        let g1 = gate[g_off + 1usize];
        let g2 = gate[g_off + 2usize];
        let g3 = gate[g_off + 3usize];
        let g4 = gate[g_off + 4usize];
        let g5 = gate[g_off + 5usize];
        let g6 = gate[g_off + 6usize];
        let g7 = gate[g_off + 7usize];
        let neg0 = f32::new(0.0f32) - g0;
        let neg1 = f32::new(0.0f32) - g1;
        let neg2 = f32::new(0.0f32) - g2;
        let neg3 = f32::new(0.0f32) - g3;
        let neg4 = f32::new(0.0f32) - g4;
        let neg5 = f32::new(0.0f32) - g5;
        let neg6 = f32::new(0.0f32) - g6;
        let neg7 = f32::new(0.0f32) - g7;
        attn_out[g_off] = o0 * inv_sum / (f32::new(1.0f32) + neg0.exp());
        attn_out[g_off + 1usize] = o1 * inv_sum / (f32::new(1.0f32) + neg1.exp());
        attn_out[g_off + 2usize] = o2 * inv_sum / (f32::new(1.0f32) + neg2.exp());
        attn_out[g_off + 3usize] = o3 * inv_sum / (f32::new(1.0f32) + neg3.exp());
        attn_out[g_off + 4usize] = o4 * inv_sum / (f32::new(1.0f32) + neg4.exp());
        attn_out[g_off + 5usize] = o5 * inv_sum / (f32::new(1.0f32) + neg5.exp());
        attn_out[g_off + 6usize] = o6 * inv_sum / (f32::new(1.0f32) + neg6.exp());
        attn_out[g_off + 7usize] = o7 * inv_sum / (f32::new(1.0f32) + neg7.exp());
    }
}

/// Launch the Q8 tiled flash prefill. Same chunked `base_pos` contract as
/// [`crate::QwenAttentionPrefillTiledCubeCL`]; the qs/scales handles must hold
/// `(base_pos + p)` rows in the module-doc layout.
#[cfg(feature = "cubecl_runtime")]
pub struct QwenAttentionPrefillTiledQ8CubeCL;

#[cfg(feature = "cubecl_runtime")]
impl QwenAttentionPrefillTiledQ8CubeCL {
    /// # Safety
    /// Same contract as [`crate::QwenAttentionPrefillTiledCubeCL::launch`] —
    /// identical handle shapes and chunked `base_pos` semantics; `head_dim`
    /// must be a multiple of 32 (256 for Bonsai); the q8 handles must cover
    /// `(base_pos + p)` rows.
    #[allow(clippy::too_many_arguments, reason = "GPU kernel launch")]
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        query_handle: Handle,
        key_qs: Handle,
        key_scales: Handle,
        value_qs: Handle,
        value_scales: Handle,
        gate_handle: Handle,
        attn_out_handle: Handle,
        head_dim: usize,
        n_head: usize,
        n_kv_head: usize,
        p: usize,
        base_pos: usize,
    ) {
        const MAX_WG_X: u32 = 65535;
        debug_assert_eq!(
            head_dim % 32,
            0,
            "q8 tiled flash kernel requires head_dim multiple of 32"
        );
        let n_blocks = (head_dim / 32) as u32;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let q_tiles_total = p.div_ceil(Q8_TILED_Q_PER_CUBE as usize).max(1);
        let tiles_per_chunk = (MAX_WG_X as usize / n_head.max(1) / 8).max(1);
        let kv_rows = base_pos + p;
        let q8_stride = n_kv_head * (head_dim / 32) * 8;
        let scale_stride = n_kv_head * (head_dim / 32);
        let mut t0 = 0usize;
        while t0 < p {
            let tiles_left = q_tiles_total - t0.div_ceil(Q8_TILED_Q_PER_CUBE as usize);
            let tiles = tiles_per_chunk.min(tiles_left);
            let tc = (tiles * Q8_TILED_Q_PER_CUBE as usize).min(p - t0);
            let params: [f32; 8] = [
                head_dim as f32,
                n_head as f32,
                n_kv_head as f32,
                tc as f32,
                scale,
                (base_pos + t0) as f32,
                tiles as f32,
                n_blocks as f32,
            ];
            let params_handle =
                crate::params_cache::params_handle(client, f32::as_bytes(&params));
            let q_len = tc * n_head * head_dim;
            let q_slice = query_handle.clone().offset_start((t0 * n_head * head_dim * 4) as u64);
            let g_slice = gate_handle.clone().offset_start((t0 * n_head * head_dim * 4) as u64);
            let o_slice =
                attn_out_handle.clone().offset_start((t0 * n_head * head_dim * 4) as u64);
            let n_cubes = n_head * tiles;
            unsafe {
                qwen_attention_prefill_tiled_q8_f32::launch_unchecked::<R>(
                    client,
                    CubeCount::Static(n_cubes as u32, 1, 1),
                    CubeDim::new_1d(256),
                    BufferArg::from_raw_parts(q_slice, q_len),
                    BufferArg::from_raw_parts(key_qs.clone(), kv_rows * q8_stride),
                    BufferArg::from_raw_parts(key_scales.clone(), kv_rows * scale_stride),
                    BufferArg::from_raw_parts(value_qs.clone(), kv_rows * q8_stride),
                    BufferArg::from_raw_parts(value_scales.clone(), kv_rows * scale_stride),
                    BufferArg::from_raw_parts(g_slice, q_len),
                    BufferArg::from_raw_parts(o_slice, q_len),
                    BufferArg::from_raw_parts(params_handle, 8),
                );
            }
            t0 += tc;
        }
    }
}
