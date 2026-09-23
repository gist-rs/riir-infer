//! Issue 726 T3 — Form C per-row int8 requant of fused ternary weights.
//!
//! Recipe (issue §"T2/T3 design contract", measured P10/P11):
//! `s_row = max group scale in row o`; `ws[o] = s_row/127` (fp16);
//! `q[o][i] = round(127 · sign_i · scale_g / s_row)` (int8). The ANE program
//! dequants via `constexpr_blockwise_shift_scale` (`w = q · ws`, per output
//! channel), so the reconstruction error per weight is ≤ `s_row/254` and the
//! measured projection-level cosine with realistic scaled-ternary weights is
//! **1.000011** (P11) — clears the G1 bar (0.999) and the stretch (0.9999).
//!
//! Input format: the Q2_0-style bit-planes the GPU path already uses
//! (`pos_bits`/`neg_bits` u64 blocks + per-group f16 scales — the
//! `TernaryGroupWeights`/`TernaryHandle` layout), so the requant consumes
//! the exact weights already resident at model load. Output: the two Form C
//! blobs (`weight_data.bin` int8 payload + `weight_scale.bin` fp16 payload).
//!
//! Pure CPU, platform-independent (unit-tested everywhere); the ANE-side
//! consumption is macOS-only.

/// Ternary group size (Q2_0 / `TernaryGroupWeights` — 128 weights/group).
pub const TERNARY_GROUP_SIZE: usize = 128;

/// The Form C requant of one fused weight matrix.
///
/// `pos_bits`/`neg_bits`: `[rows * blocks64]` u64 (columns packed 64/word,
/// row-major). `group_scale`: `[rows * groups_per_row]` f16 bits (the
/// `TernaryHandle::group_scale_f32` values are the f32 decodes of these).
/// Returns `(int8 payload [rows*cols], fp16 row-scale bits [rows])`.
pub fn requant_per_row_int8(
    pos_bits: &[u64],
    neg_bits: &[u64],
    group_scale: &[half::f16],
    rows: usize,
    cols: usize,
) -> (Vec<i8>, Vec<u16>) {
    let blocks64 = cols.div_ceil(64);
    let groups = cols.div_ceil(TERNARY_GROUP_SIZE);
    debug_assert_eq!(pos_bits.len(), rows * blocks64);
    debug_assert_eq!(neg_bits.len(), rows * blocks64);
    debug_assert_eq!(group_scale.len(), rows * groups);

    let mut q = vec![0i8; rows * cols];
    let mut ws = vec![0u16; rows];

    for o in 0..rows {
        let row_pos = &pos_bits[o * blocks64..(o + 1) * blocks64];
        let row_neg = &neg_bits[o * blocks64..(o + 1) * blocks64];
        let row_scales = &group_scale[o * groups..(o + 1) * groups];

        // s_row = max |w| over the row = max group scale over groups that
        // contain at least one nonzero bit (a zero weight contributes 0).
        let mut s_row = 0.0f32;
        for (g, &s_bits) in row_scales.iter().enumerate() {
            let col_lo = g * TERNARY_GROUP_SIZE;
            let col_hi = ((g + 1) * TERNARY_GROUP_SIZE).min(cols);
            let mut any = false;
            for c in col_lo..col_hi {
                let w = c / 64;
                let bit = 1u64 << (c % 64);
                if row_pos[w] & bit != 0 || row_neg[w] & bit != 0 {
                    any = true;
                    break;
                }
            }
            if any {
                s_row = s_row.max(s_bits.to_f32());
            }
        }

        // Zero row: any ws works (q is all zero); 1.0 keeps the blob sane.
        let ws_row = if s_row > 0.0 { s_row } else { 1.0 };
        ws[o] = half::f16::from_f32(ws_row / 127.0).to_bits();

        for c in 0..cols {
            let w = c / 64;
            let bit = 1u64 << (c % 64);
            let pos = row_pos[w] & bit != 0;
            let neg = row_neg[w] & bit != 0;
            if !pos && !neg {
                continue; // zero weight → q = 0
            }
            let sign = if pos { 1.0f32 } else { -1.0f32 };
            let s_g = row_scales[c / TERNARY_GROUP_SIZE].to_f32();
            let scaled = (127.0 * sign * s_g / ws_row).round().clamp(-127.0, 127.0);
            q[o * cols + c] = scaled as i8;
        }
    }
    (q, ws)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f16v(f: f32) -> half::f16 {
        half::f16::from_f32(f)
    }

    /// Round through fp16 (the ws path is fp16 — expectations must follow).
    fn f16r(f: f32) -> f32 {
        half::f16::from_f32(f).to_f32()
    }

    /// 2 rows × 8 cols, one group per row (group size capped at cols for the
    /// test — the production path uses 128; bit indexing is group-agnostic
    /// since scales are looked up by c/GROUP which stays 0 here).
    #[test]
    fn requant_uniform_scale_is_exact() {
        // cols = 128 so the group arithmetic matches production constants.
        let rows: usize = 2;
        let cols: usize = 128;
        let blocks64 = cols.div_ceil(64);
        let mut pos = vec![0u64; rows * blocks64];
        let mut neg = vec![0u64; rows * blocks64];
        // Row 0: + at col 0, − at col 63, + at col 64, − at col 127.
        pos[0] |= 1 << 0;
        neg[0] |= 1 << 63;
        pos[1] |= 1 << 0; // row 0, word 1 (cols 64..128)
        neg[1] |= 1 << 63;
        // Row 1: + at col 5, zero elsewhere.
        pos[blocks64] |= 1 << 5;
        let scales = vec![f16v(0.03125); rows]; // 2^-5 — exact in fp16
        let (q, ws) = requant_per_row_int8(&pos, &neg, &scales, rows, cols);
        // Uniform group scale ⇒ q hits ±127 exactly, ws = s/127.
        assert_eq!(q[0], 127);
        assert_eq!(q[63], -127);
        assert_eq!(q[64], 127);
        assert_eq!(q[127], -127);
        assert_eq!(q[cols + 5], 127);
        assert_eq!(q[cols], 0); // zero weight stays 0
        let ws0 = half::f16::from_bits(ws[0]).to_f32();
        assert_eq!(ws0, f16r(0.03125 / 127.0));
        // At uniform scale q hits ±127 exactly, so the only reconstruction
        // error is the fp16 rounding of ws (relative ~2^-11).
        assert!((q[0] as f32 * ws0 - 0.03125).abs() < 0.03125 * 1e-3);
        assert!((q[63] as f32 * ws0 + 0.03125).abs() < 0.03125 * 1e-3);
    }

    #[test]
    fn requant_two_groups_takes_row_max() {
        // cols = 256 → 2 groups/row; group scales differ (both exact fp16).
        let rows: usize = 1;
        let cols: usize = 256;
        let blocks64 = cols.div_ceil(64);
        let mut pos = vec![0u64; blocks64];
        let neg = vec![0u64; blocks64];
        pos[0] |= 1 << 0; // group 0, scale 0.0625
        pos[3] |= 1 << 0; // group 1 (col 192), scale 0.015625
        let scales = vec![f16v(0.0625), f16v(0.015625)];
        let (q, ws) = requant_per_row_int8(&pos, &neg, &scales, rows, cols);
        let s_row = 0.0625f32;
        // Group 0 (the max) hits ±127 exactly.
        assert_eq!(q[0], 127);
        // Group 1 quantizes relative to s_row: 127 · 0.015625/0.0625 = 31.75
        // → round = 32.
        assert_eq!(q[192], 32);
        let ws0 = half::f16::from_bits(ws[0]).to_f32();
        assert_eq!(ws0, f16r(s_row / 127.0));
        // Per-weight error bound: |w − q·ws| ≤ s_row/254.
        let err = (0.015625 - 32.0 * ws0).abs();
        assert!(err <= s_row / 254.0 + 1e-9, "err {err}");
    }

    #[test]
    fn requant_empty_groups_do_not_inflate_s_row() {
        // A group whose bits are all zero carries a LARGE scale — it must
        // NOT set s_row (the quantization reference is the largest |w| that
        // actually appears).
        let cols: usize = 256;
        let blocks64 = cols.div_ceil(64);
        let mut pos = vec![0u64; blocks64];
        let neg = vec![0u64; blocks64];
        pos[0] |= 1 << 0; // only group 0 populated, tiny scale
        let scales = vec![f16v(0.001), f16v(0.05)]; // group 1 empty + big
        let (q, ws) = requant_per_row_int8(&pos, &neg, &scales, 1, cols);
        assert_eq!(q[0], 127); // exact against the ROW max (0.001), not 0.05
        let ws0 = half::f16::from_bits(ws[0]).to_f32();
        assert_eq!(ws0, f16r(0.001 / 127.0));
    }

    #[test]
    fn requant_all_zero_row_is_safe() {
        let (q, ws) = requant_per_row_int8(&[0u64; 2], &[0u64; 2], &[f16v(0.5); 1], 1, 128);
        assert!(q.iter().all(|&v| v == 0));
        // Zero-row guard: ws = f16(1/127) (ws_row=1 through the /127
        // formula) — q·ws = 0 regardless of the scale value.
        assert_eq!(half::f16::from_bits(ws[0]).to_f32(), f16r(1.0 / 127.0));
    }

    #[test]
    fn requant_error_bound_holds_on_realistic_scales() {
        // P11's measured regime: scales spanning ~0.002–0.061 (BitNet-style
        // learned magnitudes). Deterministic sweep over a full group.
        let cols: usize = 128 * 8; // 8 groups
        let rows: usize = 4;
        let blocks64 = cols.div_ceil(64);
        let groups = cols / 128;
        let mut pos = vec![0u64; rows * blocks64];
        let mut neg = vec![0u64; rows * blocks64];
        let mut scales = vec![half::f16::from_f32(0.0); rows * groups];
        let mut seed = 0x12345678u64;
        let mut next = || {
            // xorshift — deterministic pseudo-random
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for r in 0..rows {
            for g in 0..groups {
                // scale in [0.002, 0.061], snapped to fp16.
                let t = (next() % 1000) as f32 / 1000.0;
                let s = 0.002 + t * 0.059;
                scales[r * groups + g] = f16v(s);
            }
            for c in 0..cols {
                match next() % 3 {
                    0 => pos[r * blocks64 + c / 64] |= 1 << (c % 64),
                    1 => neg[r * blocks64 + c / 64] |= 1 << (c % 64),
                    _ => {}
                }
            }
        }
        let (q, ws) = requant_per_row_int8(&pos, &neg, &scales, rows, cols);
        // The per-weight bound: |w − q·ws| ≤ s_row/254 (+ fp16 rounding of
        // ws — one ulp of a value ≤ 0.061/127 ≈ 4.8e-4 is ~2.9e-7).
        for r in 0..rows {
            let s_row = (0..groups)
                .map(|g| scales[r * groups + g].to_f32())
                .fold(0.0f32, f32::max);
            let ws_r = half::f16::from_bits(ws[r]).to_f32();
            for c in 0..cols {
                let w64 = c / 64;
                let bit = 1u64 << (c % 64);
                let pos_b = pos[r * blocks64 + w64] & bit != 0;
                let neg_b = neg[r * blocks64 + w64] & bit != 0;
                let w = if pos_b {
                    scales[r * groups + c / 128].to_f32()
                } else if neg_b {
                    -scales[r * groups + c / 128].to_f32()
                } else {
                    0.0
                };
                let recon = q[r * cols + c] as f32 * ws_r;
                let err = (w - recon).abs();
                assert!(
                    err <= s_row / 254.0 + 1e-6,
                    "row {r} col {c}: |{w} − {recon}| = {err} > {s_row}/254"
                );
            }
        }
    }
}
