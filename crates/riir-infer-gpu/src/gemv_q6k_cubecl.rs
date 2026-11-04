//! The Q6_K in-kernel-dequant GEMV for the dense-qwen38 Metal tail
//! (issue 1150 follow-up 3 — the tail_logits lane, bench 938's top lead).
//!
//! # Why
//!
//! The Metal dense lane host-dequants `output.weight` (Q6_K) to f32 at load —
//! 10.7 GB of GPU-resident f32 the tail GEMV re-reads EVERY ingest chunk
//! (~470 GB/s ≈ 1.7× off the bandwidth floor ⇒ 22.8 s of a 25 s
//! synchronized pass at pp512; bench 938). The CUDA reference
//! (`qwen38_dense_cudarc`) keeps lm_head PACKED (`QuantW`) and GEMVs it
//! in-kernel; this module is the CubeCL/Metal twin for the tail row: the
//! packed payload is 208 payload-bytes + an f32 `d` sidecar per 256-value
//! block ≈ **2.1 GB resident — 5× less traffic on the lane's dominant stage**.
//!
//! # Numerics class (disclosed)
//!
//! f32 activations × in-kernel-dequanted Q6_K weights, f32 accumulate — the
//! lane's documented f32-activation class (MORE precise than the cudarc
//! int8-activation path, never bit-equal to it). The products `d·sc·q·x` are
//! identical to the incumbent's dequant-then-GEMV; the SUMMATION ORDER
//! differs (per-lane block partials + one plane tree vs the incumbent's
//! row-parallel tree) — logits move at the 1e-5 re-association class. Opt-in
//! behind `RIIR_QWEN38_LMHEAD_Q6K=1` (promote-only env law; off = the
//! incumbent f32 path byte-identical). Unit gate:
//! `tests/gemv_q6k_cubecl_g1` vs the CPU canonical `gemv_q6_k`.
//!
//! # Byte layout (the CPU dequantizer's, verbatim — `BlockQ6K` repr(C))
//!
//! Per 256-value super-block: `ql[128]` (nibble pairs) @0, `qh[64]` (2 high
//! bits per value, 4-per-byte) @128, `scales[16]` (i8) @192, `d` (f16) @208.
//! The host upload repacks each 210-byte block into 52 u32 words
//! (ql words 0..32, qh 32..48, sc 48..52) + one f32 in the d sidecar — the
//! kernel never sees the dead bytes or an f16 (the Q4_K tile GEMM's
//! pre-extracted-scale precedent).
//!
//! # Dispatch
//!
//! One plane (32 lanes) per output row, 8 planes per 256-thread workgroup.
//! Each lane walks ALL of its row's blocks, holding a per-lane partial; ONE
//! `plane_sum` at the end (the GemvCubeCL plane pattern — the per-block
//! reduction would cost 40 tree passes per row). Lane l of a block computes
//! the l-th value of each 32-value quarter (q1..q4 × two halves = 8 FMAs),
//! so per block the plane covers all 256 values exactly once.

use cubecl::prelude::*;
use cubecl::server::Handle;

/// QK_K — the Q6_K super-block width.
const QK_K: usize = 256;
/// u32 words per block's packed payload (208 bytes / 4).
const BLOCK_WORDS: usize = 52;

#[cube(launch_unchecked)]
fn q6k_gemv_tail_f32(
    w: &[u32],
    d_sc: &[f32],
    x: &[f32],
    out: &mut [f32],
    params: &[f32], // [m, n]
) {
    let m = params[0usize] as usize;
    let n = params[1usize] as usize;
    let nb = n / QK_K; // blocks per row

    let row = (CUBE_POS_X as usize * 8usize) + ((UNIT_POS / 32u32) as usize);
    let lane = (UNIT_POS % 32u32) as usize;
    if row >= m {
        terminate!();
    }

    let row_w = row * nb * BLOCK_WORDS;
    let row_d = row * nb;
    let mut acc = f32::new(0.0f32);

    let mut b = 0usize;
    while b < nb {
        let wbase = row_w + b * BLOCK_WORDS;
        let d = d_sc[row_d + b];
        let xb = b * QK_K;

        // Two 128-value halves; per half l ∈ [0,32) → q1..q4 at
        // +l, +l+32, +l+64, +l+96 with scale indices is+{0,2,4,6}
        // (is = l/16), the CPU mapping verbatim. Byte j of the payload:
        // word j>>2, shift (j&3)*8 — inlined (the #[cube] expansion
        // resolves plain fns with slice args only via module paths).
        let mut half = 0usize;
        while half < 2usize {
            let ql_off = half * 64usize;
            let qh_off = half * 32usize;
            let sc_off = half * 8usize;
            let hbase = xb + half * 128usize;

            let j0 = ql_off + lane;
            let ql0 = (w[wbase + (j0 >> 2)] >> (((j0 & 3usize) * 8usize) as u32)) & 0xFFu32;
            let j1 = ql_off + lane + 32usize;
            let ql1 = (w[wbase + (j1 >> 2)] >> (((j1 & 3usize) * 8usize) as u32)) & 0xFFu32;
            let jh = 128usize + qh_off + lane;
            let qhb = (w[wbase + (jh >> 2)] >> (((jh & 3usize) * 8usize) as u32)) & 0xFFu32;

            let is = lane / 16usize;
            let js0 = 192usize + sc_off + is;
            let s0 = (w[wbase + (js0 >> 2)] >> (((js0 & 3usize) * 8usize) as u32)) & 0xFFu32;
            let js1 = 192usize + sc_off + is + 2usize;
            let s1 = (w[wbase + (js1 >> 2)] >> (((js1 & 3usize) * 8usize) as u32)) & 0xFFu32;
            let js2 = 192usize + sc_off + is + 4usize;
            let s2 = (w[wbase + (js2 >> 2)] >> (((js2 & 3usize) * 8usize) as u32)) & 0xFFu32;
            let js3 = 192usize + sc_off + is + 6usize;
            let s3 = (w[wbase + (js3 >> 2)] >> (((js3 & 3usize) * 8usize) as u32)) & 0xFFu32;

            // i8 scales: sign-extend the u8 payload.
            let mut i0 = (s0 & 0x7Fu32) as i32;
            if (s0 >> 7u32) == 1u32 {
                i0 -= 128i32;
            }
            let mut i1 = (s1 & 0x7Fu32) as i32;
            if (s1 >> 7u32) == 1u32 {
                i1 -= 128i32;
            }
            let mut i2 = (s2 & 0x7Fu32) as i32;
            if (s2 >> 7u32) == 1u32 {
                i2 -= 128i32;
            }
            let mut i3 = (s3 & 0x7Fu32) as i32;
            if (s3 >> 7u32) == 1u32 {
                i3 -= 128i32;
            }

            let q1 = ((ql0 & 0x0Fu32) | ((qhb & 3u32) << 4u32)) as i32 - 32i32;
            let q2 = ((ql1 & 0x0Fu32) | (((qhb >> 2u32) & 3u32) << 4u32)) as i32 - 32i32;
            let q3 = ((ql0 >> 4u32) | (((qhb >> 4u32) & 3u32) << 4u32)) as i32 - 32i32;
            let q4 = ((ql1 >> 4u32) | (((qhb >> 6u32) & 3u32) << 4u32)) as i32 - 32i32;

            let x0 = x[hbase + lane];
            let x1 = x[hbase + lane + 32usize];
            let x2 = x[hbase + lane + 64usize];
            let x3 = x[hbase + lane + 96usize];

            let d0 = d * (i0 as f32);
            let d1 = d * (i1 as f32);
            let d2 = d * (i2 as f32);
            let d3 = d * (i3 as f32);

            acc += d0 * (q1 as f32) * x0;
            acc += d1 * (q2 as f32) * x1;
            acc += d2 * (q3 as f32) * x2;
            acc += d3 * (q4 as f32) * x3;

            half += 1usize;
        }
        b += 1usize;
    }

    // One plane reduction for the whole row.
    let dot = plane_sum(acc);
    if lane == 0usize {
        out[row] = dot;
    }
}

/// Launcher for the Q6_K tail GEMV (full-vocab chunk pass, output overwritten).
#[cfg(feature = "cubecl_runtime")]
pub struct GemvQ6KCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl GemvQ6KCubeCL {
    /// # Safety
    /// - `w`: `m * (n/256) * 52` u32 payload words (row-major blocks, the
    ///   host-repacked ql/qh/sc payload — see the module doc).
    /// - `d_sc`: `m * (n/256)` f32 super-block scales.
    /// - `x`: `n` f32 activations; `out`: `m` f32 (overwritten).
    #[allow(
        clippy::too_many_arguments,
        reason = "GPU kernel launch/dispatch: many buffer handles are inherent to the fused-kernel interface"
    )]
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        w_handle: Handle,
        d_sc_handle: Handle,
        x_handle: Handle,
        out_handle: Handle,
        m: usize,
        n: usize,
    ) {
        debug_assert!(
            n.is_multiple_of(QK_K),
            "q6k gemv requires n % 256 == 0, got {n}"
        );
        let params: [f32; 2] = [m as f32, n as f32];
        let params_handle = crate::params_cache::params_handle(client, f32::as_bytes(&params));
        let planes_per_wg = 8usize;
        let num_wg = m.div_ceil(planes_per_wg).max(1) as u32;
        unsafe {
            q6k_gemv_tail_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(num_wg, 1, 1),
                CubeDim::new_1d(256u32),
                BufferArg::from_raw_parts(w_handle, m * (n / QK_K) * BLOCK_WORDS),
                BufferArg::from_raw_parts(d_sc_handle, m * (n / QK_K)),
                BufferArg::from_raw_parts(x_handle, n),
                BufferArg::from_raw_parts(out_handle, m),
                BufferArg::from_raw_parts(params_handle, 2),
            );
        }
    }
}

/// Host-side repack: the GGUF's 210-byte `BlockQ6K` run → (payload words, d
/// sidecar). Payload per block = ql[128] + qh[64] + scales[16] = 208 bytes =
/// 52 u32 words; `d` (f16) converts on the host once.
///
/// # Panics
/// If `blocks` is empty or `rows * nb` exceeds the slice (caller sizes both
/// from the same GGUF tensor).
#[must_use]
pub fn repack_q6k_payload(
    blocks: &[riir_infer_core::quant::q6k::BlockQ6K],
    rows: usize,
) -> (Vec<u32>, Vec<f32>) {
    let nb = blocks.len().div_ceil(rows.max(1));
    let mut w = vec![0u32; rows * nb * BLOCK_WORDS];
    let mut d = vec![0.0f32; rows * nb];
    for (i, block) in blocks.iter().take(rows * nb).enumerate() {
        let base = i * BLOCK_WORDS;
        // 208 payload bytes → 52 little-endian u32 words.
        let bytes: [u8; 208] = std::array::from_fn(|j| {
            if j < 128 {
                block.ql[j]
            } else if j < 192 {
                block.qh[j - 128]
            } else {
                block.scales[j - 192].to_ne_bytes()[0]
            }
        });
        for (k, word) in w[base..base + BLOCK_WORDS].iter_mut().enumerate() {
            let b0 = bytes[k * 4] as u32;
            let b1 = bytes[k * 4 + 1] as u32;
            let b2 = bytes[k * 4 + 2] as u32;
            let b3 = bytes[k * 4 + 3] as u32;
            *word = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
        }
        d[i] = f32::from(half::f16::from_bits(block.d));
    }
    (w, d)
}
