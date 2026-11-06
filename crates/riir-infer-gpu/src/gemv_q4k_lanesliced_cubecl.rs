//! Lane-sliced Q4_K GEMV — the load-time repack price cell (Issue 1178 T2).
//!
//! uzu's law (Research 378 / T0 verdict item c): W4 decode GEMV runs on a
//! LANE-SLICED weight layout, not the GGUF block layout. This module repacks
//! Q4_K blocks at load into a layout where each SIMD lane's nibbles live in
//! their own private u32 stream, and pre-decodes the per-sub-block scale/min
//! into f32 factor pairs (A = d·sc, B = dmin·min), so the kernel per element
//! is `A·nib·x − B·x` — no scale decode, no cross-lane word sharing, and
//! nibbles never touch the dequant path.
//!
//! # Layout (per row `r`, block `b`, lane `j` in 0..32)
//!
//! Element `e = 32·c + j` of block `b` (sub-block `c` = e/32, position `j`)
//! packs into bit range `[4c, 4c)` of word
//! `sliced[((r·32 + j)·B + b)]` — one u32 per (row, lane, block), 8 nibbles
//! low-to-high by sub-block. Factors: `ab[(r·B + b)·16 + c·2] = A`,
//! `+1 = B` (interleaved f32 pairs).
//!
//! Payload: sliced words equal the qs bytes (128 B/block, unchanged); the
//! factor table adds 64 B/block (+44%) — the price the cell charges against
//! removing ~4 ALU ops per element and the cross-lane fetch sharing.
//!
//! # Price-cell posture
//!
//! Opt-in `q4k_lanesliced_decode_gemv`; decode GEMV routes to this variant,
//! prefill `gemm()` keeps the plain `Q4KHandle` (the handle carries both —
//! memory cost ≈ 1.44× quant weights, disclosed). G1 greedy-walk pin +
//! decode A/B on bmo decide any production join (T3 GOAT).

#[cfg(feature = "cubecl_runtime")]
use cubecl::prelude::*;

#[cfg(feature = "cubecl_runtime")]
use cubecl::server::Handle;

use riir_infer_core::quant::q4k::BlockQ4K;

/// Q4_K super-block size (matches `gemv_q4k_cubecl::Q4K_BLOCK_SIZE`).
const BLOCK: usize = 256;
/// Sub-blocks per super-block.
const SUBS: usize = 8;
/// LANES per plane (Metal subgroup).
const LANES: usize = 32;

/// Robust f16→f32 for the repack — the canonical decode via the `half`
/// crate (identical to the CPU dequant's `f32::from(f16::from_bits(..))`).
fn half_to_f32(h: u16) -> f32 {
    f32::from(half::f16::from_bits(h))
}

/// 6-bit scale/min decode (CPU twin of `riir_infer_core::quant::q4k::get_scale_min_k4`
/// — the CANONICAL GGUF packing; the repacked factors must match it exactly).
fn scale_min_u(packed: &[u8; 12], j: usize) -> (u8, u8) {
    if j < 4 {
        (packed[j] & 63, packed[j + 4] & 63)
    } else {
        let sc = (packed[j + 4] & 0x0F) | ((packed[j - 4] >> 6) << 4);
        let m = (packed[j + 4] >> 4) | ((packed[j] >> 6) << 4);
        (sc, m)
    }
}

/// Repacked weights + pre-decoded factor table.
#[cfg(feature = "cubecl_runtime")]
#[derive(Clone)]
pub struct LaneSlicedQ4KHandle {
    /// One u32 per (row, lane, block): 8 nibbles, sub-block c at bits [4c,4c).
    pub sliced_words: Handle,
    /// Interleaved (A, B) f32 pairs: `ab[(r·B + b)·16 + c·2 (+1)]`.
    pub ab: Handle,
    pub m: usize,
    pub n: usize,
}

#[cfg(feature = "cubecl_runtime")]
impl LaneSlicedQ4KHandle {
    /// Repack `blocks` ([m × n/256] Q4_K blocks) into the lane-sliced layout
    /// and upload. CPU cost is one pass over the nibbles (~0.3-0.5 s per
    /// 100 MB matrix, single-thread; a price-cell disclosure, not a serving
    /// path — load happens once).
    pub fn from_blocks(
        client: &ComputeClient<crate::cubecl_runtime::ActiveRuntime>,
        blocks: &[BlockQ4K],
        m: usize,
        n: usize,
    ) -> Self {
        assert!(n.is_multiple_of(BLOCK), "n ({n}) must be multiple of {BLOCK}");
        let bpr = n / BLOCK; // blocks per row
        assert_eq!(blocks.len(), m * bpr, "blocks len mismatch");

        // sliced[r][j][b] — one u32 per (row, lane, block).
        let mut sliced = vec![0u32; m * LANES * bpr];
        // ab[r][b][c] = (A, B) interleaved.
        let mut ab = vec![0f32; m * bpr * SUBS * 2];

        for r in 0..m {
            for b in 0..bpr {
                let blk = &blocks[r * bpr + b];
                let d = half_to_f32(blk.d);
                let dmin = half_to_f32(blk.dmin);
                // nibble of element e = 32c + j: byte qs[(c/2)*32 + j],
                // low if c even, high if c odd.
                for j in 0..LANES {
                    let mut word = 0u32;
                    for c in 0..SUBS {
                        let byte = blk.qs[(c / 2) * 32 + j];
                        let nib = if c % 2 == 0 {
                            (byte & 0x0F) as u32
                        } else {
                            ((byte >> 4) & 0x0F) as u32
                        };
                        word |= nib << (4 * c);
                        // Factor decode: the canonical GGUF Q4_K packing
                        // (get_scale_min_k4 twin above); factors (d·sc, dmin·min)
                        // in the kernel's own f32 order.
                        let (sc, mn) = scale_min_u(&blk.scales, c);
                        let idx = (r * bpr + b) * SUBS * 2 + c * 2;
                        ab[idx] = d * sc as f32;
                        ab[idx + 1] = dmin * mn as f32;
                    }
                    sliced[(r * LANES + j) * bpr + b] = word;
                }
            }
        }

        let sliced_words = client.create_from_slice(u32::as_bytes(&sliced));
        let ab_handle = client.create_from_slice(f32::as_bytes(&ab));
        Self {
            sliced_words,
            ab: ab_handle,
            m,
            n,
        }
    }

    /// Blocks per row.
    #[inline]
    pub fn blocks_per_row(&self) -> usize {
        self.n / BLOCK
    }

    /// Launch the lane-sliced GEMV: `output[M] = Σ sliced·x` with factors.
    ///
    /// One plane per output row (Metal subgroup 32); lane `j` walks its own
    /// private word stream — every load serves exactly one lane.
    ///
    /// # Safety
    ///
    /// Buffer lengths must match the layout invariants (`m`, `n` multiples).
    pub unsafe fn launch<R: Runtime>(
        &self,
        client: &ComputeClient<R>,
        input_handle: Handle,
        output_handle: Handle,
    ) {
        let bpr = self.blocks_per_row() as u32;
        let wg_size = 256u32;
        let plane_size = 32u32;
        let rows_per_wg = wg_size / plane_size; // 8
        let num_wg = (self.m as u32).div_ceil(rows_per_wg).max(1);

        unsafe {
            gemv_q4k_lanesliced::launch_unchecked::<R>(
                client,
                CubeCount::Static(num_wg, 1, 1),
                CubeDim::new_1d(wg_size),
                BufferArg::from_raw_parts(
                    self.sliced_words.clone(),
                    self.m * LANES * self.blocks_per_row(),
                ),
                BufferArg::from_raw_parts(self.ab.clone(), self.m * self.blocks_per_row() * SUBS * 2),
                BufferArg::from_raw_parts(input_handle, self.n),
                BufferArg::from_raw_parts(output_handle, self.m),
            );
        }
        let _ = bpr;
    }
}

/// One plane owns one output row; lane j streams its private u32 chain.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn gemv_q4k_lanesliced(
    sliced_words: &[u32],
    ab: &[f32],
    input: &[f32],
    output: &mut [f32],
) {
    let n = input.len() as u32;
    let m = output.len() as u32;
    let bpr = n / (BLOCK as u32);

    let row = ABSOLUTE_POS_X / PLANE_DIM;
    let lane = UNIT_POS_PLANE;

    if row >= m {
        terminate!();
    }

    let mut partial = f32::new(0.0f32);
    let base = (row * (PLANE_DIM as u32) + lane) * bpr;

    let mut b = 0u32;
    while b < bpr {
        let word = sliced_words[(base + b) as usize];
        let col0 = b * (BLOCK as u32) + lane;
        let abase = (row * bpr + b) * (SUBS as u32) * 2u32;
        let mut c = 0u32;
        while c < (SUBS as u32) {
            let nib = (word >> (4 * c)) & 0xFu32;
            let x = input[(col0 + c * (LANES as u32)) as usize];
            let a = ab[(abase + c * 2u32) as usize];
            let bb = ab[(abase + c * 2u32 + 1u32) as usize];
            partial += a * (nib as f32) * x - bb * x;
            c += 1u32;
        }
        b += 1u32;
    }

    let result = plane_sum(partial);
    if lane == 0u32 {
        output[row as usize] = result;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scale_decode_cpu_twin_matches_gguf_semantics() {
        // All-uniform scales via the CANONICAL encoder (mirror of
        // get_scale_min_k4's inverse): every sub-block sc = 3, min = 5.
        let mut sc = [0u8; 12];
        for j in 0..8 {
            let (s, m) = (3u8, 5u8);
            if j < 4 {
                sc[j] |= s;
                sc[j + 4] |= m;
            } else {
                sc[j + 4] |= (s & 0x0F) | ((m & 0x0F) << 4);
                sc[j - 4] |= (s >> 4) << 6;
                sc[j] |= (m >> 4) << 6;
            }
        }
        for j in 0..8 {
            let (d_s, d_m) = scale_min_u(&sc, j);
            assert_eq!(d_s, 3, "scale[{j}]");
            assert_eq!(d_m, 5, "min[{j}]");
        }
    }

    #[test]
    fn half_to_f32_roundtrips_normals_and_zero() {
        assert_eq!(half_to_f32(0), 0.0);
        assert_eq!(half_to_f32(0x3C00), 1.0);
        assert_eq!(half_to_f32(0xC000), -2.0);
        let h: u16 = 0x4248; // 2^1 · (1 + 584/1024) = 3.140625
        assert!((half_to_f32(h) - 3.140625).abs() < 1e-6);
    }

    #[test]
    fn repack_layout_nibble_at_expected_bit() {
        // 1 row, 1 block, n = 256. Element e = 32c + j must appear at bits
        // [4c..4c+4) of sliced word (row, j, block 0).
        let mut blocks = vec![BlockQ4K {
            d: half::f16::from_f32(1.0).to_bits(),
            dmin: half::f16::from_f32(0.0).to_bits(),
            scales: [0u8; 12],
            qs: [0u8; 128],
        }];
        // Set element e = 32*2 + 7 (c=2, j=7) nibble = 9 → byte qs[(2/2)*32+7]
        // low nibble (c even) = qs[39].
        blocks[0].qs[32 + 7] = 0x09;
        let blk = &blocks[0];
        let (c, j) = (2usize, 7usize);
        let byte = blk.qs[(c / 2) * 32 + j];
        let nib = (byte & 0x0F) as u32;
        assert_eq!(nib, 9);
        let word = nib << (4 * c);
        assert_eq!((word >> (4 * c)) & 0xF, 9);
    }
}
