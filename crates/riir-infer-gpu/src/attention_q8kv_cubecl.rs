//! CubeCL flash attention kernel with inline Q8_0 KV dequantization (Plan 106 T2.9).
//!
//! Implements the same online softmax flash attention as `attention_cubecl.rs` but
//! with Q8_0 quantized KV cache. During Q·K scoring and value accumulation, i8
//! values are dequantized inline — no intermediate f32 KV materialization.
//!
//! # Q8_0 Block Format
//!
//! Each block: 32 values, 34 bytes (2 byte f16 scale + 32 × i8).
//! Dequant: `value = d * qs[i]`.
//! For head_dim=256: 8 blocks per head per position.
//!
//! # Storage Layout (two-buffer pattern, matching Q4_K)
//!
//! Two separate GPU arrays:
//!
//! | Buffer | Type | Per (pos, kv_head) | Combined Layout |
//! |--------|------|--------------------|-----------------|
//! | `kv_qs` | `Array<u32>` | 8 blocks × 8 u32s = 64 u32s | `keys_qs \|\| values_qs` |
//! | `kv_scales` | `Array<f32>` | 8 blocks × 1 f32 = 8 f32s | `keys_scales \|\| values_scales` |
//!
//! Strides (Gemma 2 2B, n_kv_head=4):
//! - `kv_stride_q8` = n_kv_head × 8_blocks × 8_u32s = 256 u32s per position
//! - `kv_scale_stride` = n_kv_head × 8_blocks = 32 f32s per position
//!
//! # Algorithm
//!
//! Identical to `attention_decode_f32` but with inline Q8_0 dequant:
//!
//! 1. Phase 1: Each thread computes softcapped Q·K score via inline dequant.
//! 2. Phases 2–3: Unrolled parallel max reduction, exp computation (unchanged).
//! 3. Phase 4: Weighted value accumulation via inline dequant.
//! 4. Phases 5–6: Unrolled parallel sum reduction, online softmax update (unchanged).
//!
//! # CubeCL Constraints
//!
//! Same workarounds as `attention_cubecl.rs`:
//! - 5 Array parameters (query, kv_qs, kv_scales, sink_kv, attn_out) — the
//!   pre-0.11 "exactly 4 arrays" limit is gone (Bench 636 migration; other
//!   kernels in this crate bind 5-7 arrays).
//! - `#[cube(launch_unchecked)]` with hardcoded Gemma 2 2B constants.
//!
//! # Sink Guard (Issue 716)
//!
//! Massive activations (MA) are sparse channels (1-2 features at 100-300×
//! typical) concentrated on sink tokens (position 0, delimiters). Per-32-block
//! absmax lets one MA channel set the block scale, collapsing the row's other
//! ~30 channels in that block to ~1 quant step. Sink rows also attract
//! disproportionate attention mass, so the V-row error flows directly into
//! the attention output (KV-side twin of the Research 085/086 weight-side
//! collapse; see Research 487).
//!
//! Mitigation (T3 winner): the first `sink_rows` positions are stored
//! LOSSLESSLY as f32 in a sidecar (`sink_kv`, layout `[sink_keys ||
//! sink_values]`, each half `sink_rows × n_kv_head × head_dim`). The kernel
//! reads the sidecar for `pos < sink_rows` in both the Q·K scoring and the
//! V-accumulation phases. `sink_kv` bound with length 0 ⇒ `sink_rows = 0` ⇒
//! neither branch is ever taken — behaviorally identical to the pre-716
//! kernel. Cost: `sink_rows × n_kv_head × head_dim × 2 × 4B` (S=4 → 32 KB
//! per layer, Gemma 2 2B ×26 layers ≈ 832 KB total — noise vs the 3.56× Q8
//! memory win).
//! - No conditional expressions as values — use `if { }` statements.
//! - Unrolled reductions (8 hardcoded steps), `sync_cube()` between steps.
//! - `UNIT_POS` is `u32`, `ABSOLUTE_POS` is `usize` — cast for indexing.
//!
//! # Dispatch
//!
//! | CubeDim | CubeCount | Responsibility |
//! |---------|-----------|----------------|
//! | `new_1d(256)` | `(n_head, 1, 1)` | 1 workgroup/head |

#[cfg(feature = "cubecl_runtime")]
use cubecl::prelude::*;

#[cfg(feature = "cubecl_runtime")]
use cubecl::server::Handle;

use bytemuck::Zeroable;
use riir_infer_core::quant::q8kv::{BlockQ8_0, Q8_BLOCK_SIZE, quantize_row_q8_0};

// ---------------------------------------------------------------------------
// CPU helpers
// ---------------------------------------------------------------------------

/// Convert IEEE 754 f16 bit pattern to f32.
///
/// Uses the `half` crate for correct conversion including subnormals (Issue 593).
#[cfg(feature = "cubecl_runtime")]
fn f16_bits_to_f32(bits: u16) -> f32 {
    half::f16::from_bits(bits).to_f32()
}

// ---------------------------------------------------------------------------
// CubeCL helper functions
// ---------------------------------------------------------------------------

/// Extract byte at index (0–3) from a u32 word (little-endian).
#[cfg(feature = "cubecl_runtime")]
#[cube]
fn get_byte_q8(word: u32, idx: u32) -> u32 {
    (word >> (idx * 8u32)) & 0xFFu32
}

/// Sign-extend a u8 byte value to f32.
///
/// For values 0–127: return as positive f32.
/// For values 128–255: subtract 256 to get -128 to -1 as f32.
///
/// Uses two if-blocks (no conditional expressions as values) to avoid
/// CubeCL v0.10 NativeExpand macro bug.
#[cfg(feature = "cubecl_runtime")]
#[cube]
fn sign_extend_i8(byte_val: u32) -> f32 {
    let mut result = byte_val as f32;
    if byte_val >= 128u32 {
        result = byte_val as f32 - f32::new(256.0f32);
    }
    result
}

// ---------------------------------------------------------------------------
// Flash attention decode kernel — Q8_0 inline KV dequant
// ---------------------------------------------------------------------------

/// CubeCL flash attention decode with online softmax, softcapping, and Q8_0 KV dequant.
///
/// Same algorithm as `attention_decode_f32` but reads quantized KV from two
/// buffers (`kv_qs` + `kv_scales`) and dequantizes inline during scoring.
///
/// ## 4-Array Parameter Layout
///
/// - `query`: `[f32; n_head × head_dim]` — query vectors (2048 for Gemma 2 2B).
/// - `kv_qs`: `[u32; 2 × n_positions × kv_stride_q8]` — packed i8 values.
///   Combined: `keys_qs || values_qs`.
/// - `kv_scales`: `[f32; 2 × n_positions × kv_scale_stride]` — per-block f32 scales.
///   Combined: `keys_scales || values_scales`.
/// - `sink_kv`: `[f32; 2 × sink_rows × n_kv_head × head_dim]` — lossless f32
///   sidecar for the first `sink_rows` positions (Issue 716; `sink_keys ||
///   sink_values`). Bind length 0 to disable the guard.
/// - `attn_out`: `[f32; n_head × head_dim]` — output (2048 for Gemma 2 2B).
///
/// ## GQA Mapping (Gemma 2 2B)
///
/// n_head=8, n_kv_head=4 → 2 query heads per KV head.
///
/// ## Dimensions
///
/// ```text
/// kv_stride_q8    = n_kv_head × 8_blocks × 8_u32s = 256 u32s
/// kv_scale_stride  = n_kv_head × 8_blocks = 32 f32s
/// kv_qs_half       = kv_qs.len() / 2
/// kv_scales_half   = kv_scales.len() / 2
/// n_positions      = kv_scales_half / kv_scale_stride
/// sink_rows        = (sink_kv.len() / 2) / (n_kv_head × head_dim)
/// ```
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn attention_decode_q8kv(
    query: &[f32],
    kv_qs: &[u32],
    kv_scales: &[f32],
    sink_kv: &[f32],
    attn_out: &mut [f32],
) {
    // ── Gemma 2 2B constants ──
    let head_dim = 256u32;
    let n_head = 8u32;
    let n_kv_head = 4u32;
    let n_blocks = 8u32; // head_dim / Q8_BLOCK_SIZE = 256 / 32
    let u32_per_block = 8u32; // Q8_BLOCK_SIZE / 4 = 32 / 4
    let u32_per_head = n_blocks * u32_per_block; // 64 u32s per (pos, kv_head)
    let kv_stride_q8 = n_kv_head * u32_per_head; // 256 u32s per position
    let kv_scale_stride = n_kv_head * n_blocks; // 32 f32s per position
    let softcap = f32::new(50.0f32);
    let scale = f32::new(0.0625f32); // 1/√256
    let cube_size = 256u32;

    // ── Derive dimensions from array lengths ──
    let kv_qs_half = kv_qs.len() as u32 / 2u32;
    let kv_scales_half = kv_scales.len() as u32 / 2u32;
    let n_positions = kv_scales_half / kv_scale_stride;

    // ── Issue 716 sink sidecar geometry ──
    // sink_kv = [sink_keys || sink_values], each half sink_rows × n_kv_head ×
    // head_dim f32s. A zero-length binding ⇒ sink_rows = 0 ⇒ neither guard
    // branch is ever taken (behaviorally identical to the pre-716 kernel).
    let kv_f32_stride = n_kv_head * head_dim; // 1024 f32s per position
    let sink_half = sink_kv.len() as u32 / 2u32;
    let sink_rows = sink_half / kv_f32_stride;

    // ── Workgroup/thread assignment ──
    let head_idx = ABSOLUTE_POS as u32 / cube_size;
    let tid = UNIT_POS;

    // Guard: skip if overdispatched beyond n_head
    if head_idx >= n_head {
        terminate!();
    }

    let head_off = head_idx * head_dim;

    // Guard: no positions — write zeros and exit
    if n_positions == 0u32 {
        if tid < head_dim {
            attn_out[(head_off + tid) as usize] = f32::new(0.0f32);
        }
        terminate!();
    }

    // GQA: compute which KV head this query head attends to
    let kv_group = head_idx * n_kv_head / n_head;

    // Whether this thread owns a valid output dimension
    let valid_dim = tid < head_dim;

    // ── Shared memory for reductions (1 KB) ──
    let mut smem = Shared::<[f32]>::new_slice(256usize);

    // ── Online softmax running state (per-thread) ──
    let mut running_max = f32::new(-1e30f32);
    let mut running_sum = f32::new(0.0f32);
    let mut running_out = f32::new(0.0f32);

    // ── Process KV positions in tiles of 256 ──
    let n_tiles = n_positions.div_ceil(cube_size);

    let mut tile = 0u32;
    while tile < n_tiles {
        let tile_base = tile * cube_size;
        let pos = tile_base + tid;
        let valid_pos = pos < n_positions;

        // ── Phase 1: Softcapped Q·K score with inline Q8_0 dequant ──
        let mut my_score = f32::new(-1e30f32);
        if valid_pos {
            let mut dot = f32::new(0.0f32);
            if pos < sink_rows {
                // Issue 716 sink guard: lossless f32 sidecar row (keys half).
                let k_base = pos * kv_f32_stride + kv_group * head_dim;
                let mut dim = 0u32;
                while dim < head_dim {
                    dot += query[(head_off + dim) as usize] * sink_kv[(k_base + dim) as usize];
                    dim += 1u32;
                }
            } else {
                let mut block_idx = 0u32;
                while block_idx < n_blocks {
                    // Read scale for this block (key half)
                    let scale_idx = pos * kv_scale_stride + kv_group * n_blocks + block_idx;
                    let block_scale = kv_scales[scale_idx as usize];

                    // Base offset for this block's packed i8 values
                    let qs_block_base =
                        pos * kv_stride_q8 + kv_group * u32_per_head + block_idx * u32_per_block;

                    let mut e = 0u32;
                    while e < 32u32 {
                        let word_idx = qs_block_base + e / 4u32;
                        let byte_in_word = e % 4u32;
                        let word = kv_qs[word_idx as usize];
                        let byte_val = get_byte_q8(word, byte_in_word);
                        let i8_val = sign_extend_i8(byte_val);
                        let k_val = block_scale * i8_val;

                        let dim = block_idx * 32u32 + e;
                        dot += query[(head_off + dim) as usize] * k_val;
                        e += 1u32;
                    }
                    block_idx += 1u32;
                }
            }
            let raw = dot * scale;
            my_score = softcap * f32::tanh(raw / softcap);
        }
        smem[tid as usize] = my_score;
        sync_cube();

        // ── Phase 2: Unrolled parallel max reduction → tile_max ──
        if tid < 128u32 && smem[(tid + 128u32) as usize] > smem[tid as usize] {
            smem[tid as usize] = smem[(tid + 128u32) as usize];
        }
        sync_cube();
        if tid < 64u32 && smem[(tid + 64u32) as usize] > smem[tid as usize] {
            smem[tid as usize] = smem[(tid + 64u32) as usize];
        }
        sync_cube();
        if tid < 32u32 && smem[(tid + 32u32) as usize] > smem[tid as usize] {
            smem[tid as usize] = smem[(tid + 32u32) as usize];
        }
        sync_cube();
        if tid < 16u32 && smem[(tid + 16u32) as usize] > smem[tid as usize] {
            smem[tid as usize] = smem[(tid + 16u32) as usize];
        }
        sync_cube();
        if tid < 8u32 && smem[(tid + 8u32) as usize] > smem[tid as usize] {
            smem[tid as usize] = smem[(tid + 8u32) as usize];
        }
        sync_cube();
        if tid < 4u32 && smem[(tid + 4u32) as usize] > smem[tid as usize] {
            smem[tid as usize] = smem[(tid + 4u32) as usize];
        }
        sync_cube();
        if tid < 2u32 && smem[(tid + 2u32) as usize] > smem[tid as usize] {
            smem[tid as usize] = smem[(tid + 2u32) as usize];
        }
        sync_cube();
        if tid < 1u32 && smem[1usize] > smem[0usize] {
            smem[0usize] = smem[1usize];
        }
        sync_cube();
        let tile_max = smem[0usize];
        sync_cube();

        // ── Phase 3: Compute exp(score - tile_max) ──
        let mut my_exp = f32::new(0.0f32);
        if valid_pos {
            my_exp = f32::exp(my_score - tile_max);
        }
        smem[tid as usize] = my_exp;
        sync_cube();

        // ── Phase 4: Weighted value accumulation with inline Q8_0 dequant ──
        let mut tile_val = f32::new(0.0f32);
        if valid_dim {
            // Precompute block/element for this thread's output dimension
            let v_block_idx = tid / 32u32;
            let v_element = tid % 32u32;
            let v_word_offset = v_element / 4u32;
            let v_byte_in_word = v_element % 4u32;

            let mut i = 0u32;
            while i < cube_size {
                let pos_i = tile_base + i;
                if pos_i < n_positions {
                    // Q8_0 dequant read (also valid — just dead — for sink
                    // rows, whose quantized rows still sit in the buffers).
                    let v_scale_idx = kv_scales_half
                        + pos_i * kv_scale_stride
                        + kv_group * n_blocks
                        + v_block_idx;
                    let v_scale = kv_scales[v_scale_idx as usize];

                    let v_qs_word_idx = kv_qs_half
                        + pos_i * kv_stride_q8
                        + kv_group * u32_per_head
                        + v_block_idx * u32_per_block
                        + v_word_offset;
                    let v_qs_word = kv_qs[v_qs_word_idx as usize];
                    let v_byte = get_byte_q8(v_qs_word, v_byte_in_word);
                    let v_i8 = sign_extend_i8(v_byte);
                    let mut v_val = v_scale * v_i8;

                    // Issue 716 sink guard: lossless f32 sidecar row (values
                    // half) OVERRIDES the quantized read for the sink span.
                    if pos_i < sink_rows {
                        v_val = sink_kv[(sink_half
                            + pos_i * kv_f32_stride
                            + kv_group * head_dim
                            + tid)
                            as usize];
                    }

                    tile_val += smem[i as usize] * v_val;
                }
                i += 1u32;
            }
        }

        // ── Phase 5: Unrolled parallel sum reduction → tile_sum ──
        sync_cube();
        if tid < 128u32 {
            smem[tid as usize] = smem[tid as usize] + smem[(tid + 128u32) as usize];
        }
        sync_cube();
        if tid < 64u32 {
            smem[tid as usize] = smem[tid as usize] + smem[(tid + 64u32) as usize];
        }
        sync_cube();
        if tid < 32u32 {
            smem[tid as usize] = smem[tid as usize] + smem[(tid + 32u32) as usize];
        }
        sync_cube();
        if tid < 16u32 {
            smem[tid as usize] = smem[tid as usize] + smem[(tid + 16u32) as usize];
        }
        sync_cube();
        if tid < 8u32 {
            smem[tid as usize] = smem[tid as usize] + smem[(tid + 8u32) as usize];
        }
        sync_cube();
        if tid < 4u32 {
            smem[tid as usize] = smem[tid as usize] + smem[(tid + 4u32) as usize];
        }
        sync_cube();
        if tid < 2u32 {
            smem[tid as usize] = smem[tid as usize] + smem[(tid + 2u32) as usize];
        }
        sync_cube();
        if tid < 1u32 {
            smem[0usize] = smem[0usize] + smem[1usize];
        }
        sync_cube();
        let tile_sum = smem[0usize];

        // ── Phase 6: Online softmax update ──
        let mut new_max = running_max;
        if tile_max > new_max {
            new_max = tile_max;
        }
        let prev_corr = f32::exp(running_max - new_max);
        let curr_corr = f32::exp(tile_max - new_max);

        running_sum = running_sum * prev_corr + tile_sum * curr_corr;
        running_out = running_out * prev_corr + tile_val * curr_corr;
        running_max = new_max;

        sync_cube();
        tile += 1u32;
    }

    // ── Final: Normalize and write output ──
    if valid_dim {
        attn_out[(head_off + tid) as usize] = running_out / running_sum;
    }
}

// ---------------------------------------------------------------------------
// Public types: Q8_0 KV buffer pair
// ---------------------------------------------------------------------------

/// Q8_0 KV cache buffer pair for attention dispatch.
///
/// Two separate GPU-ready arrays following the Q4_K two-buffer pattern:
/// - `kv_qs`: packed i8 quantized values as u32 words (4 i8s per u32)
/// - `kv_scales`: per-block f32 scales (CPU pre-decoded f16→f32)
///
/// # Combined Layout
///
/// ```text
/// kv_qs:    [keys_qs(n_pos × kv_stride_q8) || values_qs(n_pos × kv_stride_q8)]
/// kv_scales: [keys_scales(n_pos × kv_scale_stride) || values_scales(n_pos × kv_scale_stride)]
/// ```
///
/// Where (Gemma 2 2B, n_kv_head=4, head_dim=256):
/// - `kv_stride_q8 = n_kv_head × 8_blocks × 8_u32s = 256` u32s per position
/// - `kv_scale_stride = n_kv_head × 8_blocks = 32` f32s per position
#[cfg(feature = "cubecl_runtime")]
#[derive(Clone, Debug)]
pub struct Q8KVBuffers {
    /// Packed i8 quantized values as u32 words.
    /// Combined layout: `[keys_qs || values_qs]`.
    pub kv_qs: Vec<u32>,
    /// Per-block f32 scales (CPU pre-decoded f16→f32).
    /// Combined layout: `[keys_scales || values_scales]`.
    pub kv_scales: Vec<f32>,
    /// Lossless f32 sidecar for the first `sink_rows` positions (Issue 716):
    /// `[sink_keys || sink_values]`, each half `sink_rows × n_kv_head × head_dim`.
    /// Empty when the sink guard is off — the quantized rows 0..sink_rows are
    /// still present in `kv_qs`/`kv_scales` (dead data under the guard, ~1 KB/
    /// layer at S=4) so the q8 layout never shifts.
    pub sink_kv: Vec<f32>,
}

#[cfg(feature = "cubecl_runtime")]
impl Q8KVBuffers {
    /// Quantize f32 K/V rows to Q8_0 and pack into buffer pair.
    ///
    /// # Arguments
    ///
    /// - `keys`: `[n_pos × n_kv_head × head_dim]` f32 key vectors (row-major).
    /// - `values`: `[n_pos × n_kv_head × head_dim]` f32 value vectors (row-major).
    /// - `n_kv_head`: Number of KV heads (4 for Gemma 2 2B).
    /// - `head_dim`: Head dimension (256 for Gemma 2 2B, must be multiple of 32).
    ///
    /// # Returns
    ///
    /// `Q8KVBuffers` with combined layout ready for GPU upload.
    ///
    /// # Panics
    ///
    /// Panics if `head_dim` is not a multiple of `Q8_BLOCK_SIZE` (32),
    /// or if input lengths don't match expected dimensions.
    pub fn quantize_kv(keys: &[f32], values: &[f32], n_kv_head: usize, head_dim: usize) -> Self {
        assert!(
            head_dim.is_multiple_of(Q8_BLOCK_SIZE),
            "head_dim ({head_dim}) must be multiple of {Q8_BLOCK_SIZE}"
        );

        let n_blocks = head_dim / Q8_BLOCK_SIZE; // 8 for head_dim=256
        let u32_per_head = n_blocks * (Q8_BLOCK_SIZE / 4); // 64 u32s per (pos, kv_head)
        let scales_per_head = n_blocks; // 8 f32s per (pos, kv_head)

        let kv_stride_q8 = n_kv_head * u32_per_head; // 256 u32s per position
        let kv_scale_stride = n_kv_head * scales_per_head; // 32 f32s per position

        let kv_len = keys.len();
        let n_positions = kv_len / (n_kv_head * head_dim);
        assert_eq!(
            values.len(),
            kv_len,
            "keys and values must have same length"
        );

        let kv_qs_half = n_positions * kv_stride_q8;
        let kv_scales_half = n_positions * kv_scale_stride;

        let total_qs = 2 * kv_qs_half;
        let total_scales = 2 * kv_scales_half;

        let mut kv_qs = vec![0u32; total_qs];
        let mut kv_scales = vec![0.0f32; total_scales];

        // Quantize keys (first half) and values (second half)
        Self::quantize_kv_half(
            keys,
            n_positions,
            n_kv_head,
            head_dim,
            n_blocks,
            u32_per_head,
            &mut kv_qs,
            &mut kv_scales,
            0, // qs_offset
            0, // scales_offset
            kv_stride_q8,
            kv_scale_stride,
        );
        Self::quantize_kv_half(
            values,
            n_positions,
            n_kv_head,
            head_dim,
            n_blocks,
            u32_per_head,
            &mut kv_qs,
            &mut kv_scales,
            kv_qs_half,     // qs_offset
            kv_scales_half, // scales_offset
            kv_stride_q8,
            kv_scale_stride,
        );

        Self { kv_qs, kv_scales, sink_kv: Vec::new() }
    }

    /// Quantize with the Issue 716 sink guard: the first `sink_rows` positions
    /// are additionally stored losslessly as f32 in `sink_kv` (the q8 buffers
    /// are byte-identical to `quantize_kv`'s — the kernel simply never reads
    /// the quantized rows 0..sink_rows when the sidecar is bound).
    ///
    /// `sink_rows = 0` returns exactly `quantize_kv`'s output. Requires
    /// `sink_rows <= n_positions`.
    #[cfg(feature = "q8kv_sink_guard")]
    pub fn quantize_kv_with_sink(
        keys: &[f32],
        values: &[f32],
        n_kv_head: usize,
        head_dim: usize,
        sink_rows: usize,
    ) -> Self {
        let mut bufs = Self::quantize_kv(keys, values, n_kv_head, head_dim);
        let n_positions = keys.len() / (n_kv_head * head_dim);
        assert!(
            sink_rows <= n_positions,
            "sink_rows ({sink_rows}) must be <= n_positions ({n_positions})"
        );
        let stride = n_kv_head * head_dim;
        let mut sink_kv = Vec::with_capacity(2 * sink_rows * stride);
        sink_kv.extend_from_slice(&keys[..sink_rows * stride]);
        sink_kv.extend_from_slice(&values[..sink_rows * stride]);
        bufs.sink_kv = sink_kv;
        bufs
    }

    /// Quantize one half (keys or values) into the combined buffers.
    #[allow(clippy::too_many_arguments)]
    fn quantize_kv_half(
        data: &[f32],
        n_positions: usize,
        n_kv_head: usize,
        head_dim: usize,
        n_blocks: usize,
        u32_per_head: usize,
        kv_qs: &mut [u32],
        kv_scales: &mut [f32],
        qs_offset: usize,
        scales_offset: usize,
        kv_stride_q8: usize,
        kv_scale_stride: usize,
    ) {
        // Issue 710 H13: one scratch row-buffer hoisted through the loop — the
        // per-(pos, kv_head) `vec![BlockQ8_0::zeroed(); n_blocks]` was
        // 2 × n_positions × n_kv_head heap allocs per cache build.
        // `quantize_row_q8_0` fully overwrites all n_blocks (head_dim is a
        // multiple of Q8_BLOCK_SIZE), so the zero-fill was never observable.
        let mut blocks = vec![BlockQ8_0::zeroed(); n_blocks];
        for pos in 0..n_positions {
            for kv_head in 0..n_kv_head {
                let data_off = pos * n_kv_head * head_dim + kv_head * head_dim;
                let row = &data[data_off..data_off + head_dim];

                // Quantize to Q8_0 blocks
                quantize_row_q8_0(row, &mut blocks);

                // Pack into output buffers
                let qs_base = qs_offset + pos * kv_stride_q8 + kv_head * u32_per_head;
                let scales_base = scales_offset + pos * kv_scale_stride + kv_head * n_blocks;

                for (block_idx, block) in blocks.iter().enumerate() {
                    // Scale: f16 bits → f32
                    let scale_f32 = f16_bits_to_f32(block.d);
                    kv_scales[scales_base + block_idx] = scale_f32;

                    // Pack i8 values into u32 words (4 i8s per u32, little-endian)
                    let qs_block_base = qs_base + block_idx * (Q8_BLOCK_SIZE / 4);
                    for word_idx in 0..(Q8_BLOCK_SIZE / 4) {
                        let mut word: u32 = 0;
                        for byte_idx in 0..4 {
                            let element_idx = word_idx * 4 + byte_idx;
                            let byte_val = block.qs[element_idx] as u8 as u32;
                            word |= byte_val << (byte_idx * 8);
                        }
                        kv_qs[qs_block_base + word_idx] = word;
                    }
                }
            }
        }
    }

    /// Dequantize Q8_0 KV buffers back to f32 (for testing/verification).
    ///
    /// Returns `(keys, values)` as separate f32 vectors.
    pub fn dequantize_kv(
        &self,
        n_positions: usize,
        n_kv_head: usize,
        head_dim: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let n_blocks = head_dim / Q8_BLOCK_SIZE;
        let u32_per_head = n_blocks * (Q8_BLOCK_SIZE / 4);
        let kv_stride_q8 = n_kv_head * u32_per_head;
        let kv_scale_stride = n_kv_head * n_blocks;

        let kv_qs_half = n_positions * kv_stride_q8;
        let kv_scales_half = n_positions * kv_scale_stride;

        let mut keys = vec![0.0f32; n_positions * n_kv_head * head_dim];
        let mut values = vec![0.0f32; n_positions * n_kv_head * head_dim];

        Self::dequantize_half(
            &self.kv_qs,
            &self.kv_scales,
            n_positions,
            n_kv_head,
            head_dim,
            n_blocks,
            &mut keys,
            0,
            0,
            kv_stride_q8,
            kv_scale_stride,
        );
        Self::dequantize_half(
            &self.kv_qs,
            &self.kv_scales,
            n_positions,
            n_kv_head,
            head_dim,
            n_blocks,
            &mut values,
            kv_qs_half,
            kv_scales_half,
            kv_stride_q8,
            kv_scale_stride,
        );

        // Issue 716: when the sink sidecar is present, the EFFECTIVE KV for
        // rows 0..sink_rows is the lossless f32 sidecar, not the quantized
        // (dead-data) rows. Overwrite so dequantize reflects what the guarded
        // kernel actually consumes.
        if !self.sink_kv.is_empty() {
            let stride = n_kv_head * head_dim;
            let sink_rows = self.sink_kv.len() / (2 * stride);
            let sink_half = sink_rows * stride;
            for pos in 0..sink_rows {
                let off = pos * stride;
                keys[off..off + stride].copy_from_slice(&self.sink_kv[off..off + stride]);
                values[off..off + stride]
                    .copy_from_slice(&self.sink_kv[sink_half + off..sink_half + off + stride]);
            }
        }

        (keys, values)
    }

    /// Dequantize one half of the combined buffers to f32.
    #[allow(clippy::too_many_arguments)]
    fn dequantize_half(
        kv_qs: &[u32],
        kv_scales: &[f32],
        n_positions: usize,
        n_kv_head: usize,
        head_dim: usize,
        n_blocks: usize,
        output: &mut [f32],
        qs_offset: usize,
        scales_offset: usize,
        kv_stride_q8: usize,
        kv_scale_stride: usize,
    ) {
        let u32_per_head = n_blocks * (Q8_BLOCK_SIZE / 4);

        for pos in 0..n_positions {
            for kv_head in 0..n_kv_head {
                let out_off = pos * n_kv_head * head_dim + kv_head * head_dim;
                let qs_base = qs_offset + pos * kv_stride_q8 + kv_head * u32_per_head;
                let scales_base = scales_offset + pos * kv_scale_stride + kv_head * n_blocks;

                for block_idx in 0..n_blocks {
                    let block_scale = kv_scales[scales_base + block_idx];
                    let qs_block_base = qs_base + block_idx * (Q8_BLOCK_SIZE / 4);

                    for element in 0..Q8_BLOCK_SIZE {
                        let word_idx = qs_block_base + element / 4;
                        let byte_in_word = element % 4;
                        let word = kv_qs[word_idx];
                        let byte_val = ((word >> (byte_in_word * 8)) & 0xFF) as i8;
                        let dim = block_idx * Q8_BLOCK_SIZE + element;
                        output[out_off + dim] = block_scale * byte_val as f32;
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Public launcher
// ---------------------------------------------------------------------------

/// CubeCL flash attention decode launcher with Q8_0 KV dequant.
///
/// Provides the public API for launching the CubeCL flash attention kernel
/// with inline Q8_0 KV dequantization for Gemma 2 decode.
///
/// # Example
///
/// ```rust,ignore
/// let ctx = CubeCLContext::new()?;
/// let client = ctx.client();
///
/// let params = AttentionParams { n_positions: 512, ..Default::default() };
/// let q8_bufs = Q8KVBuffers::quantize_kv(&keys, &values, 4, 256);
///
/// let kv_qs_handle = client.create_from_slice(bytemuck::cast_slice::<u32, u8>(&q8_bufs.kv_qs));
/// let kv_scales_handle = client.create_from_slice(f32::as_bytes(&q8_bufs.kv_scales));
///
/// AttentionQ8KVCubeCL::launch::<ActiveRuntime>(
///     &client, q_handle, kv_qs_handle, kv_scales_handle, out_handle, &params,
/// );
/// ```
#[cfg(feature = "cubecl_runtime")]
pub struct AttentionQ8KVCubeCL;

#[cfg(feature = "cubecl_runtime")]
use super::AttentionParams;

#[cfg(feature = "cubecl_runtime")]
impl AttentionQ8KVCubeCL {
    /// Launch flash attention decode kernel with Q8_0 inline KV dequant.
    ///
    /// Single dispatch for all 8 query heads. Each workgroup (256 threads)
    /// handles one head with tiled KV scan, online softmax, and inline
    /// Q8_0 dequantization during scoring and value accumulation.
    ///
    /// Dispatch: `(n_head, 1, 1)` workgroups of 256 threads each.
    ///
    /// # Buffer sizes
    ///
    /// - `query_handle`: `n_head × head_dim = 2048` f32 elements
    /// - `kv_qs_handle`: `2 × n_positions × kv_stride_q8` u32 elements
    ///   (keys_qs || values_qs, use `Q8KVBuffers::quantize_kv`)
    /// - `kv_scales_handle`: `2 × n_positions × kv_scale_stride` f32 elements
    ///   (keys_scales || values_scales, use `Q8KVBuffers::quantize_kv`)
    /// - `attn_out_handle`: `n_head × head_dim = 2048` f32 elements
    ///
    /// **`n_positions` is the single source of truth for BOTH buffer-length
    /// args (Issue 710 H11):** the kernel re-derives `n_positions` from the
    /// `kv_scales` length arg (`len/2/kv_scale_stride`, kernel L152-154), so
    /// the caller's ACTUAL buffer allocations MUST match `params.n_positions`
    /// exactly — under-sized allocations read OOB under `launch_unchecked`
    /// (caller-owns-correctness), oversized ones silently extend context.
    ///
    /// # Panics
    ///
    /// Panics if params don't match Gemma 2 2B constants (n_head=8, n_kv_head=4,
    /// head_dim=256).
    pub fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        query_handle: Handle,
        kv_qs_handle: Handle,
        kv_scales_handle: Handle,
        attn_out_handle: Handle,
        params: &AttentionParams,
    ) {
        assert_eq!(params.n_head, 8, "Only Gemma 2 2B n_head=8 supported");
        assert_eq!(params.n_kv_head, 4, "Only Gemma 2 2B n_kv_head=4 supported");
        assert_eq!(
            params.head_dim, 256,
            "Only Gemma 2 2B head_dim=256 supported"
        );

        let n_head = params.n_head as u32;
        let q_len = params.n_head * params.head_dim;

        // Q8_0 strides: n_kv_head × n_blocks × words_per_block
        let kv_stride_q8 = params.n_kv_head * 8 * 8; // 4 × 8 × 8 = 256 u32s
        let kv_scale_stride = params.n_kv_head * 8; // 4 × 8 = 32 f32s
        let total_qs_len = 2 * params.n_positions * kv_stride_q8;
        let total_scales_len = 2 * params.n_positions * kv_scale_stride;

        // Issue 515 T2 sweep: this kernel derives `n_positions` from
        // `kv_scales.len()` (and offsets from `kv_qs.len()`) — the BOUND
        // BUFFER's size, not the bound length. Guard both so a pooled or
        // reused buffer cannot silently change the kernel's shape idea.
        crate::cubecl_runtime::assert_binding_derives_units(
            &kv_qs_handle,
            2 * kv_stride_q8,
            params.n_positions,
            "AttentionQ8KV::launch kv_qs",
        );
        crate::cubecl_runtime::assert_binding_derives_units(
            &kv_scales_handle,
            2 * kv_scale_stride,
            params.n_positions,
            "AttentionQ8KV::launch kv_scales",
        );

        // Issue 716: bind a 1-f32 dummy sink sidecar. The kernel derives
        // sink_rows from the sidecar's ALLOCATION size (cubecl's
        // `register_buffer` uses `handle.size_in_used()`, NOT the bound length
        // — the Bench 642 lesson), so the dummy must be SMALLER than one sink
        // row: 1 f32 → len 1 → 1/2/1024 = 0 → sink_rows = 0 → neither guard
        // branch is ever taken (behaviorally identical to the pre-716 kernel).
        // Never read at that length; the 4-byte alloc is noise against the
        // per-call kv_qs/kv_scales uploads beside it.
        let no_sink_handle = client.empty(core::mem::size_of::<f32>());
        // 515 T2: pin the dummy at sink_rows == 0 — if the pool ever rounds
        // the 4-byte alloc up past one sink row (2 × n_kv_head × head_dim
        // f32s), the kernel would silently take the sink branch and read the
        // dummy; this assert turns that into a loud refusal.
        crate::cubecl_runtime::assert_binding_derives_units(
            &no_sink_handle,
            2 * params.n_kv_head * params.head_dim,
            0,
            "AttentionQ8KV::launch no_sink dummy",
        );
        // SAFETY: Caller guarantees correct buffer sizes.
        unsafe {
            attention_decode_q8kv::launch_unchecked::<R>(
                client,
                CubeCount::Static(n_head, 1, 1),
                CubeDim::new_1d(256),
                BufferArg::from_raw_parts(query_handle, q_len),
                BufferArg::from_raw_parts(kv_qs_handle, total_qs_len),
                BufferArg::from_raw_parts(kv_scales_handle, total_scales_len),
                BufferArg::from_raw_parts(no_sink_handle, 1),
                BufferArg::from_raw_parts(attn_out_handle, q_len),
            );
        }
    }

    /// Launch with the Issue 716 sink guard: the first `sink_rows` KV
    /// positions are read losslessly from the f32 `sink_kv_handle` sidecar
    /// (`[sink_keys || sink_values]`, produced by
    /// `Q8KVBuffers::quantize_kv_with_sink`) instead of the Q8_0 buffers.
    ///
    /// Same dispatch geometry as [`Self::launch`]. The `sink_kv_handle`
    /// **allocation** must be EXACTLY `2 × sink_rows × n_kv_head × head_dim`
    /// f32 elements — cubecl's `register_buffer` derives the kernel-visible
    /// length from the handle's allocation size, not the bound length (the
    /// Bench 642 lesson), so an oversized allocation silently inflates
    /// `sink_rows` (caller-owns-correctness, Issue 710 H11).
    ///
    /// # Panics
    ///
    /// Panics if params don't match Gemma 2 2B constants (n_head=8,
    /// n_kv_head=4, head_dim=256).
    #[cfg(feature = "q8kv_sink_guard")]
    pub fn launch_with_sink<R: Runtime>(
        client: &ComputeClient<R>,
        query_handle: Handle,
        kv_qs_handle: Handle,
        kv_scales_handle: Handle,
        sink_kv_handle: Handle,
        attn_out_handle: Handle,
        sink_rows: usize,
        params: &AttentionParams,
    ) {
        assert_eq!(params.n_head, 8, "Only Gemma 2 2B n_head=8 supported");
        assert_eq!(params.n_kv_head, 4, "Only Gemma 2 2B n_kv_head=4 supported");
        assert_eq!(
            params.head_dim, 256,
            "Only Gemma 2 2B head_dim=256 supported"
        );
        assert!(
            sink_rows <= params.n_positions,
            "sink_rows ({sink_rows}) must be <= n_positions ({})",
            params.n_positions
        );

        let n_head = params.n_head as u32;
        let q_len = params.n_head * params.head_dim;

        let kv_stride_q8 = params.n_kv_head * 8 * 8;
        let kv_scale_stride = params.n_kv_head * 8;
        let total_qs_len = 2 * params.n_positions * kv_stride_q8;
        let total_scales_len = 2 * params.n_positions * kv_scale_stride;
        let total_sink_len = 2 * sink_rows * params.n_kv_head * params.head_dim;

        // Issue 515 T2 sweep: same derived-dimension guards as `launch`, plus
        // the sink sidecar — `sink_rows` is derived from `sink_kv.len()` on
        // the device, so an oversized allocation silently inflates it (the
        // exact hazard the doc above names; caller-owns-correctness becomes
        // launcher-refuses-loudly).
        crate::cubecl_runtime::assert_binding_derives_units(
            &kv_qs_handle,
            2 * kv_stride_q8,
            params.n_positions,
            "AttentionQ8KV::launch_with_sink kv_qs",
        );
        crate::cubecl_runtime::assert_binding_derives_units(
            &kv_scales_handle,
            2 * kv_scale_stride,
            params.n_positions,
            "AttentionQ8KV::launch_with_sink kv_scales",
        );
        crate::cubecl_runtime::assert_binding_derives_units(
            &sink_kv_handle,
            2 * params.n_kv_head * params.head_dim,
            sink_rows,
            "AttentionQ8KV::launch_with_sink sink_kv",
        );

        // SAFETY: Caller guarantees correct buffer sizes.
        unsafe {
            attention_decode_q8kv::launch_unchecked::<R>(
                client,
                CubeCount::Static(n_head, 1, 1),
                CubeDim::new_1d(256),
                BufferArg::from_raw_parts(query_handle, q_len),
                BufferArg::from_raw_parts(kv_qs_handle, total_qs_len),
                BufferArg::from_raw_parts(kv_scales_handle, total_scales_len),
                BufferArg::from_raw_parts(sink_kv_handle, total_sink_len),
                BufferArg::from_raw_parts(attn_out_handle, q_len),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "cubecl_runtime"))]
mod tests {
    use super::*;
    use crate::attention_cubecl::AttentionCubeCL;
    use crate::cubecl_runtime::CubeCLContext;

    use crate::cubecl_runtime::ActiveRuntime;

    /// Gemma 2 2B test constants.
    const N_HEAD: usize = 8;
    const N_KV_HEAD: usize = 4;
    const HEAD_DIM: usize = 256;
    const SOFTCAP: f32 = 50.0;
    const SCALE: f32 = 0.0625; // 1/√256

    /// Generate deterministic test data using sin/cos patterns.
    fn make_test_data(n_positions: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let q_len = N_HEAD * HEAD_DIM;
        let kv_len = n_positions * N_KV_HEAD * HEAD_DIM;

        let query: Vec<f32> = (0..q_len)
            .map(|i| ((i as f32) * 0.01).sin() * 0.5)
            .collect();
        let keys: Vec<f32> = (0..kv_len)
            .map(|i| ((i as f32) * 0.02).cos() * 0.3)
            .collect();
        let values: Vec<f32> = (0..kv_len)
            .map(|i| ((i as f32) * 0.03).sin() * 0.2)
            .collect();

        (query, keys, values)
    }

    /// Run f32 attention on GPU and return output.
    fn run_f32_attention(
        client: &ComputeClient<ActiveRuntime>,
        query: &[f32],
        keys: &[f32],
        values: &[f32],
        n_positions: usize,
    ) -> Vec<f32> {
        let params = AttentionParams {
            n_head: N_HEAD,
            n_kv_head: N_KV_HEAD,
            head_dim: HEAD_DIM,
            n_positions,
            softcap: SOFTCAP,
            scale: SCALE,
        };

        let combined_kv = params.combine_kv(keys, values);
        let q_len = N_HEAD * HEAD_DIM;

        let query_handle = client.create_from_slice(f32::as_bytes(query));
        let kv_handle = client.create_from_slice(f32::as_bytes(&combined_kv));
        let out_handle = client.empty(q_len * core::mem::size_of::<f32>());

        AttentionCubeCL::launch::<ActiveRuntime>(
            client,
            query_handle,
            kv_handle,
            out_handle.clone(),
            &params,
        );

        let bytes = client.read_one(out_handle).expect("should read f32 output");
        f32::from_bytes(&bytes).to_vec()
    }

    /// Run Q8_0 attention on GPU and return output.
    fn run_q8kv_attention(
        client: &ComputeClient<ActiveRuntime>,
        query: &[f32],
        keys: &[f32],
        values: &[f32],
        n_positions: usize,
    ) -> Vec<f32> {
        let params = AttentionParams {
            n_head: N_HEAD,
            n_kv_head: N_KV_HEAD,
            head_dim: HEAD_DIM,
            n_positions,
            softcap: SOFTCAP,
            scale: SCALE,
        };

        // Quantize KV to Q8_0
        let q8_bufs = Q8KVBuffers::quantize_kv(keys, values, N_KV_HEAD, HEAD_DIM);

        let q_len = N_HEAD * HEAD_DIM;
        let query_handle = client.create_from_slice(f32::as_bytes(query));
        let kv_qs_bytes = bytemuck::cast_slice::<u32, u8>(&q8_bufs.kv_qs);
        let kv_qs_handle = client.create_from_slice(kv_qs_bytes);
        let kv_scales_handle = client.create_from_slice(f32::as_bytes(&q8_bufs.kv_scales));
        let out_handle = client.empty(q_len * core::mem::size_of::<f32>());

        AttentionQ8KVCubeCL::launch::<ActiveRuntime>(
            client,
            query_handle,
            kv_qs_handle,
            kv_scales_handle,
            out_handle.clone(),
            &params,
        );

        let bytes = client
            .read_one(out_handle)
            .expect("should read q8kv output");
        f32::from_bytes(&bytes).to_vec()
    }

    /// Run guarded Q8_0 attention (Issue 716): first `sink_rows` positions
    /// served losslessly from the f32 sidecar. `sink_rows = 0` takes the
    /// zero-length-sidecar path (any valid f32 handle, never read).
    #[cfg(feature = "q8kv_sink_guard")]
    fn run_q8kv_attention_with_sink(
        client: &ComputeClient<ActiveRuntime>,
        query: &[f32],
        keys: &[f32],
        values: &[f32],
        n_positions: usize,
        sink_rows: usize,
    ) -> Vec<f32> {
        let params = AttentionParams {
            n_head: N_HEAD,
            n_kv_head: N_KV_HEAD,
            head_dim: HEAD_DIM,
            n_positions,
            softcap: SOFTCAP,
            scale: SCALE,
        };

        let q8_bufs = Q8KVBuffers::quantize_kv_with_sink(keys, values, N_KV_HEAD, HEAD_DIM, sink_rows);

        let q_len = N_HEAD * HEAD_DIM;
        let query_handle = client.create_from_slice(f32::as_bytes(query));
        let kv_qs_handle = client.create_from_slice(bytemuck::cast_slice::<u32, u8>(&q8_bufs.kv_qs));
        let kv_scales_handle = client.create_from_slice(f32::as_bytes(&q8_bufs.kv_scales));
        // The sink handle's ALLOCATION size is the kernel-visible length
        // (Bench 642 lesson) — bind a 1-f32 dummy when the sidecar is empty
        // (1/2/1024 = 0 → sink_rows = 0, never read).
        let sink_handle = if q8_bufs.sink_kv.is_empty() {
            client.empty(core::mem::size_of::<f32>())
        } else {
            client.create_from_slice(f32::as_bytes(&q8_bufs.sink_kv))
        };
        let out_handle = client.empty(q_len * core::mem::size_of::<f32>());

        AttentionQ8KVCubeCL::launch_with_sink::<ActiveRuntime>(
            client,
            query_handle,
            kv_qs_handle,
            kv_scales_handle,
            sink_handle,
            out_handle.clone(),
            sink_rows,
            &params,
        );

        let bytes = client
            .read_one(out_handle)
            .expect("should read q8kv sink output");
        f32::from_bytes(&bytes).to_vec()
    }

    /// Compare Q8_0 attention output against f32 reference and report max error.
    fn compare_q8kv_vs_f32(
        client: &ComputeClient<ActiveRuntime>,
        query: &[f32],
        keys: &[f32],
        values: &[f32],
        n_positions: usize,
        tolerance: f32,
    ) {
        let f32_output = run_f32_attention(client, query, keys, values, n_positions);
        let q8_output = run_q8kv_attention(client, query, keys, values, n_positions);

        assert_eq!(f32_output.len(), q8_output.len(), "output length mismatch");

        let mut max_err = 0.0f32;
        let mut max_err_idx = 0usize;
        for (i, (&f_ref, &q_val)) in f32_output.iter().zip(q8_output.iter()).enumerate() {
            let err = (f_ref - q_val).abs();
            if err > max_err {
                max_err = err;
                max_err_idx = i;
            }
            assert!(
                err < tolerance,
                "element {i}: f32={f_ref}, q8={q_val}, err={err} (tolerance {tolerance})"
            );
        }

        println!(
            "q8kv attention (n_pos={n_positions}): max_error = {max_err:.6} at idx {max_err_idx}"
        );
    }

    /// Single KV position: output should closely match f32 attention.
    #[test]
    fn test_attention_q8kv_single_position() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let n_positions = 1;
        let (query, keys, values) = make_test_data(n_positions);
        compare_q8kv_vs_f32(&client, &query, &keys, &values, n_positions, 0.05);
    }

    /// Small context (10 positions): tests multi-position softmax with Q8_0 dequant.
    #[test]
    fn test_attention_q8kv_small_context() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let n_positions = 10;
        let (query, keys, values) = make_test_data(n_positions);
        compare_q8kv_vs_f32(&client, &query, &keys, &values, n_positions, 0.1);
    }

    /// Realistic Gemma 2 decode dimensions (32 positions).
    #[test]
    fn test_attention_q8kv_gemma2_realistic() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let n_positions = 32;
        let (query, keys, values) = make_test_data(n_positions);
        compare_q8kv_vs_f32(&client, &query, &keys, &values, n_positions, 0.15);
    }

    /// Multi-tile path — n_positions ≥ 512 forces n_tiles ≥ 2 (tile = 256),
    /// the FIRST test coverage of the Phase 6 online-softmax cross-tile
    /// rescale (`prev_corr`/`curr_corr`) and the tail-tile Phase 4 guard —
    /// production decode with context > 256 tokens runs exactly this path
    /// (riir-ai Issue 710 H10).
    ///
    /// VALIDATED 2026-08-16 in a GPU-exclusive window (4090, DX12/wgpu-30):
    /// PASSED at max_error 4.3e-5 vs f32 reference (tol 0.15) — the ignore
    /// removed. NOTE: this test passes while the kimi backward parity suite
    /// diverges on the same box (Issue 711) — the q8kv attention kernels are
    /// numerically clean on DX12; the divergence is specific to the kimi
    /// GEMV/GEMM path.
    #[test]
    fn test_attention_q8kv_multi_tile() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let n_positions = 512; // 2 full tiles — exercises cross-tile rescale
        let (query, keys, values) = make_test_data(n_positions);
        compare_q8kv_vs_f32(&client, &query, &keys, &values, n_positions, 0.15);
    }

    /// Quantize→dequant roundtrip: verify Q8_0 preserves attention quality.
    ///
    /// Quantizes K/V to Q8_0, dequantizes back to f32, then runs f32 attention
    /// on both original and roundtripped data. The roundtripped attention output
    /// should be close to the original.
    #[test]
    fn test_q8kv_quantize_roundtrip() {
        let n_positions = 16;
        let (query, keys, values) = make_test_data(n_positions);

        // Quantize and dequantize
        let q8_bufs = Q8KVBuffers::quantize_kv(&keys, &values, N_KV_HEAD, HEAD_DIM);
        let (deq_keys, deq_values) = q8_bufs.dequantize_kv(n_positions, N_KV_HEAD, HEAD_DIM);

        // Verify quantization error is bounded
        let mut max_key_err = 0.0f32;
        let mut max_val_err = 0.0f32;
        for (&orig, &deq) in keys.iter().zip(deq_keys.iter()) {
            let err = (orig - deq).abs();
            if err > max_key_err {
                max_key_err = err;
            }
        }
        for (&orig, &deq) in values.iter().zip(deq_values.iter()) {
            let err = (orig - deq).abs();
            if err > max_val_err {
                max_val_err = err;
            }
        }

        // Q8_0 quantization error should be < 1/127 of value range
        // Test data range is roughly [-0.5, 0.5], so max error ~ 0.5/127 ≈ 0.004
        println!(
            "Q8_0 roundtrip (n_pos={n_positions}): max_key_err={max_key_err:.6}, max_val_err={max_val_err:.6}"
        );
        assert!(
            max_key_err < 0.01,
            "Key quantization error too large: {max_key_err}"
        );
        assert!(
            max_val_err < 0.01,
            "Value quantization error too large: {max_val_err}"
        );

        // Run attention on both original and roundtripped data via GPU
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let orig_output = run_f32_attention(&client, &query, &keys, &values, n_positions);
        let roundtrip_output =
            run_f32_attention(&client, &query, &deq_keys, &deq_values, n_positions);

        let mut max_attn_err = 0.0f32;
        for (&orig, &rt) in orig_output.iter().zip(roundtrip_output.iter()) {
            let err = (orig - rt).abs();
            if err > max_attn_err {
                max_attn_err = err;
            }
        }

        println!("Q8_0 attention roundtrip (n_pos={n_positions}): max_attn_err={max_attn_err:.6}");
        assert!(
            max_attn_err < 0.05,
            "Attention roundtrip error too large: {max_attn_err}"
        );
    }

    // ── Issue 716: massive-activation (MA) failure mode + sink guard ──────

    /// MA channels: one in block 0 (dim 5), one in block 4 (dim 130) — the
    /// sparse 1-2 channel outliers of Sun et al. 2402.17762 / 2608.12149.
    const MA_DIMS: [usize; 2] = [5, 130];

    /// True for dims sharing a 32-block with an MA channel (the poisoned
    /// blocks: neighbor channels collapse to ~1 quant step).
    fn is_poisoned_dim(dim: usize) -> bool {
        MA_DIMS.iter().any(|&ma| dim / Q8_BLOCK_SIZE == ma / Q8_BLOCK_SIZE)
    }

    /// Build `make_test_data` with massive activations injected at
    /// `ma_positions`: each MA channel is set to `ma_mult` × the typical row
    /// magnitude (keys ±0.3, values ±0.2) in every kv_head of that position.
    fn make_test_data_with_ma(
        n_positions: usize,
        ma_positions: &[usize],
        ma_mult: f32,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let (query, mut keys, mut values) = make_test_data(n_positions);
        for &pos in ma_positions {
            assert!(pos < n_positions, "MA position {pos} out of range");
            for dim in MA_DIMS {
                for kv_head in 0..N_KV_HEAD {
                    let base = pos * N_KV_HEAD * HEAD_DIM + kv_head * HEAD_DIM + dim;
                    keys[base] = keys[base].signum() * ma_mult * 0.3;
                    values[base] = values[base].signum() * ma_mult * 0.2;
                }
            }
        }
        (query, keys, values)
    }

    /// CPU reference decode attention (all heads), mirroring the GPU kernel
    /// math (scale + softcap tanh + softmax + weighted V sum).
    fn cpu_attention_output(
        query: &[f32],
        keys: &[f32],
        values: &[f32],
        n_positions: usize,
    ) -> Vec<f32> {
        cpu_attention_output_rotated_sink(query, keys, values, n_positions, 0)
    }

    /// CPU reference attention; when `sink_rows > 0` the first `sink_rows`
    /// positions are treated as Hadamard-ROTATED rows (Issue 716 T2 variant
    /// (c)): scoring uses `HQ·K̃` and the sink V contribution is accumulated
    /// in rotated space then un-rotated (`H` symmetric unitary).
    fn cpu_attention_output_rotated_sink(
        query: &[f32],
        keys: &[f32],
        values: &[f32],
        n_positions: usize,
        sink_rows: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; N_HEAD * HEAD_DIM];
        for head in 0..N_HEAD {
            let kv_group = head * N_KV_HEAD / N_HEAD;
            let q_off = head * HEAD_DIM;

            // Rotated query slice for sink-row scoring (only if rotated sinks exist).
            let mut hq = [0.0f32; 256];
            if sink_rows > 0 {
                hq.copy_from_slice(&query[q_off..q_off + HEAD_DIM]);
                fwht256(&mut hq);
            }

            let mut scores = vec![0.0f32; n_positions];
            for (pos, score) in scores.iter_mut().enumerate() {
                let row = pos * N_KV_HEAD * HEAD_DIM + kv_group * HEAD_DIM;
                let mut dot = 0.0f32;
                if pos < sink_rows {
                    for d in 0..HEAD_DIM {
                        dot += hq[d] * keys[row + d];
                    }
                } else {
                    for d in 0..HEAD_DIM {
                        dot += query[q_off + d] * keys[row + d];
                    }
                }
                *score = SOFTCAP * ((dot * SCALE) / SOFTCAP).tanh();
            }

            let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let weights: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
            let sum: f32 = weights.iter().sum();

            // Sink contributions accumulate in rotated space, then un-rotate.
            let mut sink_acc = [0.0f32; 256];
            for (pos, &w_raw) in weights.iter().enumerate() {
                let row = pos * N_KV_HEAD * HEAD_DIM + kv_group * HEAD_DIM;
                let w = w_raw / sum;
                if pos < sink_rows {
                    for d in 0..HEAD_DIM {
                        sink_acc[d] += w * values[row + d];
                    }
                } else {
                    for d in 0..HEAD_DIM {
                        out[q_off + d] += w * values[row + d];
                    }
                }
            }
            if sink_rows > 0 {
                fwht256(&mut sink_acc);
                for d in 0..HEAD_DIM {
                    out[q_off + d] += sink_acc[d];
                }
            }
        }
        out
    }

    /// Mean attention weight of `pos` across query heads (CPU diagnostic).
    fn cpu_attention_sink_weight(
        query: &[f32],
        keys: &[f32],
        n_positions: usize,
        pos_of_interest: usize,
    ) -> f32 {
        let mut total = 0.0f32;
        for head in 0..N_HEAD {
            let kv_group = head * N_KV_HEAD / N_HEAD;
            let q_off = head * HEAD_DIM;
            let mut scores = vec![0.0f32; n_positions];
            for (pos, score) in scores.iter_mut().enumerate() {
                let row = pos * N_KV_HEAD * HEAD_DIM + kv_group * HEAD_DIM;
                let mut dot = 0.0f32;
                for d in 0..HEAD_DIM {
                    dot += query[q_off + d] * keys[row + d];
                }
                *score = SOFTCAP * ((dot * SCALE) / SOFTCAP).tanh();
            }
            let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let sum: f32 = scores.iter().map(|s| (s - max).exp()).sum();
            total += (scores[pos_of_interest] - max).exp() / sum;
        }
        total / N_HEAD as f32
    }

    /// In-place fast Walsh-Hadamard transform, 256 elements, unitary
    /// (includes the 1/√256 = 1/16 scaling; `H` symmetric orthogonal).
    fn fwht256(v: &mut [f32]) {
        assert_eq!(v.len(), 256, "fwht256 expects exactly 256 elements");
        let mut h = 1usize;
        while h < 256 {
            let mut i = 0usize;
            while i < 256 {
                let mut j = i;
                while j < i + h {
                    let a = v[j];
                    let b = v[j + h];
                    v[j] = a + b;
                    v[j + h] = a - b;
                    j += 1;
                }
                i += h * 2;
            }
            h *= 2;
        }
        for x in v.iter_mut() {
            *x /= 16.0;
        }
    }

    /// Per-element error report of q8kv vs f32 attention (GPU), split by
    /// poisoned-block dims vs clean dims (Issue 716 T1 signature).
    struct MaErrorReport {
        max_err: f32,
        mean_err: f32,
        max_err_poisoned: f32,
        max_err_clean: f32,
    }

    fn measure_q8kv_error(
        client: &ComputeClient<ActiveRuntime>,
        query: &[f32],
        keys: &[f32],
        values: &[f32],
        n_positions: usize,
    ) -> MaErrorReport {
        let f32_output = run_f32_attention(client, query, keys, values, n_positions);
        let q8_output = run_q8kv_attention(client, query, keys, values, n_positions);
        let mut report = MaErrorReport {
            max_err: 0.0,
            mean_err: 0.0,
            max_err_poisoned: 0.0,
            max_err_clean: 0.0,
        };
        let mut sum = 0.0f32;
        for (i, (&f_ref, &q_val)) in f32_output.iter().zip(q8_output.iter()).enumerate() {
            let err = (f_ref - q_val).abs();
            let dim = i % HEAD_DIM;
            report.max_err = report.max_err.max(err);
            sum += err;
            if is_poisoned_dim(dim) {
                report.max_err_poisoned = report.max_err_poisoned.max(err);
            } else {
                report.max_err_clean = report.max_err_clean.max(err);
            }
        }
        report.mean_err = sum / f32_output.len() as f32;
        report
    }

    /// T1 (CPU half): one MA channel poisons its 32-block — neighbor channels
    /// collapse to ~1 quant step (the KV-side twin of Research 085/086
    /// weight-side outlier→scale collapse). Absolute error vs the row's
    /// typical magnitude (relative error is meaningless on the zero-crossing
    /// sin/cos pattern).
    #[test]
    fn test_q8kv_ma_row_collapse_cpu() {
        let n_positions = 8;
        let (_query, keys, values) = make_test_data_with_ma(n_positions, &[0], 100.0);
        let q8_bufs = Q8KVBuffers::quantize_kv(&keys, &values, N_KV_HEAD, HEAD_DIM);
        let (deq_keys, _) = q8_bufs.dequantize_kv(n_positions, N_KV_HEAD, HEAD_DIM);

        // Sink row (pos 0, kv_head 0): absolute error on non-MA channels.
        let mut max_abs_poisoned = 0.0f32;
        let mut max_abs_clean = 0.0f32;
        for dim in 0..HEAD_DIM {
            if MA_DIMS.contains(&dim) {
                continue;
            }
            let err = (keys[dim] - deq_keys[dim]).abs();
            if is_poisoned_dim(dim) {
                max_abs_poisoned = max_abs_poisoned.max(err);
            } else {
                max_abs_clean = max_abs_clean.max(err);
            }
        }
        // Expected: poisoned-block step = MA/127 = 30/127 ≈ 0.236 (err ≤ 0.118);
        // clean-block step = 0.3/127 ≈ 0.0024 (err ≤ 0.0012) — ~100× collapse.
        println!(
            "MA row collapse (x100): abs_err poisoned-block={max_abs_poisoned:.4} clean-block={max_abs_clean:.6}"
        );
        assert!(
            max_abs_poisoned > 10.0 * max_abs_clean,
            "neighbor-channel collapse missing: poisoned={max_abs_poisoned} clean={max_abs_clean}"
        );
        // And the collapse is at the ~1-quant-step scale of the MA block:
        // err ≈ half the poisoned block's step (MA × 0.3 / 127 / 2).
        let poisoned_step = 100.0 * 0.3 / 127.0;
        assert!(
            max_abs_poisoned > poisoned_step * 0.25 && max_abs_poisoned <= poisoned_step,
            "poisoned-block error should sit at ~half the MA-block quant step ({poisoned_step:.4}), got {max_abs_poisoned:.4}"
        );
    }

    /// T1 (GPU half): the paper's failure mode is LIVE here — attention
    /// output error inflates by orders of magnitude vs uniform data,
    /// concentrated at the poisoned block dims. Measured numbers recorded in
    /// Bench 691.
    #[test]
    fn test_q8kv_ma_output_error_inflation() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();
        let n_positions = 32;

        let (q_u, k_u, v_u) = make_test_data(n_positions);
        let uniform = measure_q8kv_error(&client, &q_u, &k_u, &v_u, n_positions);

        let (q100, k100, v100) = make_test_data_with_ma(n_positions, &[0, 7], 100.0);
        let ma100 = measure_q8kv_error(&client, &q100, &k100, &v100, n_positions);

        let (q300, k300, v300) = make_test_data_with_ma(n_positions, &[0, 7], 300.0);
        let ma300 = measure_q8kv_error(&client, &q300, &k300, &v300, n_positions);

        let w100 = cpu_attention_sink_weight(&q100, &k100, n_positions, 0);
        let w300 = cpu_attention_sink_weight(&q300, &k300, n_positions, 0);
        println!(
            "Issue 716 T1 (n_pos={n_positions}): uniform  max={:.6} mean={:.7}",
            uniform.max_err, uniform.mean_err
        );
        println!(
            "  MA x100 (sink w={w100:.3}): max={:.6} mean={:.7} poisoned={:.6} clean={:.6}",
            ma100.max_err, ma100.mean_err, ma100.max_err_poisoned, ma100.max_err_clean
        );
        println!(
            "  MA x300 (sink w={w300:.3}): max={:.6} mean={:.7} poisoned={:.6} clean={:.6}",
            ma300.max_err, ma300.mean_err, ma300.max_err_poisoned, ma300.max_err_clean
        );

        assert!(
            ma100.max_err > 5.0 * uniform.max_err,
            "MA x100 failure mode not live: {} vs uniform {}",
            ma100.max_err,
            uniform.max_err
        );
        assert!(
            ma300.max_err > 10.0 * uniform.max_err,
            "MA x300 failure mode not live: {} vs uniform {}",
            ma300.max_err,
            uniform.max_err
        );
        assert!(
            ma300.max_err_poisoned > ma300.max_err_clean,
            "error not concentrated at poisoned dims: {} vs {}",
            ma300.max_err_poisoned,
            ma300.max_err_clean
        );
    }

    /// T2 variant (c), CPU-simulated quality axis: Hadamard-rotating ONLY the
    /// sink rows spreads the MA across the head (per-block absmax ≈ MA/16)
    /// so neighbors regain ~√N/2 resolution — a real improvement over plain
    /// q8, but nonzero, and its kernel cost strictly exceeds (b)'s branch
    /// (rotated-Q buffer + separate sink accumulator + un-rotate phase vs a
    /// per-position f32 read). Records the T3 decision input.
    #[test]
    fn test_q8kv_hadamard_sink_simulation_cpu() {
        let n_positions = 32;
        let sink_rows = 4;
        let ma_positions: Vec<usize> = (0..sink_rows).collect();
        let (query, keys, values) = make_test_data_with_ma(n_positions, &ma_positions, 300.0);

        let reference = cpu_attention_output(&query, &keys, &values, n_positions);

        // (a) plain q8, CPU-simulated.
        let q8 = Q8KVBuffers::quantize_kv(&keys, &values, N_KV_HEAD, HEAD_DIM);
        let (dk, dv) = q8.dequantize_kv(n_positions, N_KV_HEAD, HEAD_DIM);
        let out_a = cpu_attention_output(&query, &dk, &dv, n_positions);
        let err_a = reference
            .iter()
            .zip(out_a.iter())
            .fold(0.0f32, |m, (&r, &o)| m.max((r - o).abs()));

        // (c) Hadamard-rotated sink rows, quantized, rotation-aware attention.
        let mut rk = keys.clone();
        let mut rv = values.clone();
        for pos in 0..sink_rows {
            for kv_head in 0..N_KV_HEAD {
                let off = pos * N_KV_HEAD * HEAD_DIM + kv_head * HEAD_DIM;
                fwht256(&mut rk[off..off + HEAD_DIM]);
                fwht256(&mut rv[off..off + HEAD_DIM]);
            }
        }
        let q8c = Q8KVBuffers::quantize_kv(&rk, &rv, N_KV_HEAD, HEAD_DIM);
        let (drk, drv) = q8c.dequantize_kv(n_positions, N_KV_HEAD, HEAD_DIM);
        let out_c = cpu_attention_output_rotated_sink(&query, &drk, &drv, n_positions, sink_rows);
        let err_c = reference
            .iter()
            .zip(out_c.iter())
            .fold(0.0f32, |m, (&r, &o)| m.max((r - o).abs()));

        println!(
            "Issue 716 T2(c) CPU sim (MA x300, S={sink_rows}): (a) plain q8 err={err_a:.6}, (c) hadamard err={err_c:.6}"
        );
        assert!(
            err_c < err_a * 0.5,
            "Hadamard must reduce sink-row error (else the simulation is wrong): {err_c} vs {err_a}"
        );
        // (c) reduces but does NOT eliminate sink error; (b)'s sidecar is
        // EXACT on sink rows (proven on GPU in the guard tests) — the T3 tie-break.
        assert!(err_c > 0.0);
    }

    /// T4 G1: with the guard (S=4) covering the MA positions, error returns
    /// to non-sink q8 baseline levels — the MA error is gone.
    #[test]
    #[cfg(feature = "q8kv_sink_guard")]
    fn test_q8kv_sink_guard_restores_accuracy() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();
        let n_positions = 32;

        // MA at pos 0 + pos 3 — both inside the S=4 guard span.
        let (q, k, v) = make_test_data_with_ma(n_positions, &[0, 3], 300.0);
        let unguarded = measure_q8kv_error(&client, &q, &k, &v, n_positions);

        let f32_out = run_f32_attention(&client, &q, &k, &v, n_positions);
        let guard_out = run_q8kv_attention_with_sink(&client, &q, &k, &v, n_positions, 4);
        let guard_max = f32_out
            .iter()
            .zip(guard_out.iter())
            .fold(0.0f32, |m, (&a, &b)| m.max((a - b).abs()));

        let (qu, ku, vu) = make_test_data(n_positions);
        let uniform = measure_q8kv_error(&client, &qu, &ku, &vu, n_positions);

        println!(
            "Issue 716 T4 G1 (S=4, MA x300): unguarded max={:.6} → guarded max={:.6} (uniform baseline {:.6})",
            unguarded.max_err, guard_max, uniform.max_err
        );
        assert!(
            guard_max < unguarded.max_err * 0.25,
            "sink guard must remove the bulk of the MA error: {guard_max} vs {}",
            unguarded.max_err
        );
        assert!(
            guard_max <= uniform.max_err * 3.0,
            "guarded error should sit at the non-sink q8 baseline: {guard_max} vs {}",
            uniform.max_err
        );
    }

    /// T4: `sink_rows = 0` is bit-identical to the unguarded launch.
    #[test]
    #[cfg(feature = "q8kv_sink_guard")]
    fn test_q8kv_sink_guard_zero_rows_bit_identical() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();
        let n_positions = 32;
        let (query, keys, values) = make_test_data(n_positions);

        let a = run_q8kv_attention(&client, &query, &keys, &values, n_positions);
        let b = run_q8kv_attention_with_sink(&client, &query, &keys, &values, n_positions, 0);
        assert_eq!(a.len(), b.len());
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                x.to_bits() == y.to_bits(),
                "sink_rows=0 must be bit-identical (idx {i}: {x} vs {y})"
            );
        }
    }

    /// Sanity: `sink_rows = n_positions` (everything sidecar) tracks the f32
    /// kernel within fp noise (same online-softmax structure).
    #[test]
    #[cfg(feature = "q8kv_sink_guard")]
    fn test_q8kv_sink_guard_full_sidecar_matches_f32() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();
        let n_positions = 32;
        let (query, keys, values) = make_test_data(n_positions);

        let f32_out = run_f32_attention(&client, &query, &keys, &values, n_positions);
        let sink_out =
            run_q8kv_attention_with_sink(&client, &query, &keys, &values, n_positions, n_positions);
        let max = f32_out
            .iter()
            .zip(sink_out.iter())
            .fold(0.0f32, |m, (&a, &b)| m.max((a - b).abs()));
        println!("full sidecar vs f32 kernel: max diff {max:.8}");
        assert!(max < 1e-4, "all-sink path must match the f32 kernel, got {max}");
    }

    /// T4 G2 (release-only): the guard branch is perf-neutral — f32 sidecar
    /// reads REPLACE byte-dequant loops on the guard span, plus one compare
    /// per position.
    #[test]
    #[cfg_attr(debug_assertions, ignore)]
    #[cfg(feature = "q8kv_sink_guard")]
    fn test_q8kv_sink_guard_perf_neutral() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();
        let n_positions = 512;
        let (query, keys, values) = make_test_data(n_positions);

        // Warm both paths.
        let _ = run_q8kv_attention(&client, &query, &keys, &values, n_positions);
        let _ = run_q8kv_attention_with_sink(&client, &query, &keys, &values, n_positions, 4);

        let iters = 50;
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            let _ = run_q8kv_attention(&client, &query, &keys, &values, n_positions);
        }
        let base = t0.elapsed();

        let t1 = std::time::Instant::now();
        for _ in 0..iters {
            let _ = run_q8kv_attention_with_sink(&client, &query, &keys, &values, n_positions, 4);
        }
        let sink = t1.elapsed();

        println!("Issue 716 G2: base={base:?} sink(S=4)={sink:?} for {iters} iters @ n_pos={n_positions}");
        assert!(
            sink.as_secs_f64() < base.as_secs_f64() * 1.3 + 5e-3,
            "sink guard must be perf-neutral: base {base:?} vs sink {sink:?}"
        );
    }
}
