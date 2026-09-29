//! Plan 614 / Issue 026 — the DQ phase-sensitivity fake-quant instrument.
//!
//! One shared, PURE spec of the two activation grids + the A8 control grid, the
//! host reference every kernel is gated against (G-i3), and the process knobs +
//! PER-PHASE launch counters (D5). The kernels live beside their lanes: the CUDA
//! blockwise kernel in `cudarc_kernels`, the CubeCL twin in `dq_fakequant_cubecl`.
//!
//! The spec is deliberately UNGATED (pure f32 math over slices — the runner's
//! unit tests + the G-i3 oracle compile at default features), while the kernels
//! and the injection sites are `dq_phase_bench`-gated (default-off, the G4 law:
//! the hot path is untouched when the feature is off).
//!
//! FROZEN by Plan 614's pre-registration (commit d3300f2 — the definition text
//! D1/D2 is the contract; edits here that change NUMERICS invalidate the
//! freeze and require a re-review).

/// The activation grid under test (Plan 614 D1).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DqGrid {
    /// A2 — the asymmetric 4-level activation grid (027-T0 SHAPE; gates nothing
    /// in 027): per-128 block, `d = sign(argmax|a|) · f16(amax/2)`, tie-break
    /// LOWEST INDEX WINS (D1 — parallel reductions on two backends can
    /// otherwise pick different signs), codes on `d·{−1, 0, +1, +2}`:
    /// `q = clamp(round(a/d), −1, +2)`, `aq = q·d`.
    A2,
    /// A4 (primary tier) — per-32 affine 16-level: `min`/`max` in f32 per
    /// block; `s = f16((max−min)/15)`; `q = clamp(round((a−min)/s), 0, 15)`;
    /// `aq = q·s + min` (min in f32). No secondary scale quantization.
    A4,
    /// A8 — the CONTROL grid (D1 fallback lane): per-128 symmetric 8-bit,
    /// `d = f16(amax/127)`, `q = clamp(round(a/d), −127, 127)`, `aq = q·d`.
    /// Emulates the shipping prefill's dynamic q8 class on the decode side so
    /// the `dec_a8` cell can bound the base-asymmetry floor.
    A8,
}

impl DqGrid {
    pub const fn block(self) -> usize {
        match self {
            DqGrid::A2 | DqGrid::A8 => 128,
            DqGrid::A4 => 32,
        }
    }
}

/// Which phase(s) carry the fake-quant passes (Plan 614 D1 matrix rows).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DqPhaseArm {
    Off,
    PrefillOnly,
    DecodeOnly,
    Both,
}

impl DqPhaseArm {
    pub const fn prefill_armed(self) -> bool {
        matches!(self, DqPhaseArm::PrefillOnly | DqPhaseArm::Both)
    }
    pub const fn decode_armed(self) -> bool {
        matches!(self, DqPhaseArm::DecodeOnly | DqPhaseArm::Both)
    }
}

/// Round half away from zero (the RTN encoder convention — Plan 614 D1's
/// bit-exact spec; `f32::round` is exactly this).
#[inline]
pub fn round_half_away(x: f32) -> f32 {
    x.round()
}

/// The A2 grid math for one 128-element block (host reference; kernels mirror
/// this bit-exactly — G-i3).
///
/// `d = sign(argmax|a|) · f16(amax/2)` — the argmax tie-break is LOWEST INDEX
/// (a strict `>` scan from index 0 gives exactly that: the first max wins).
#[inline]
pub fn a2_quant_dequant_block(block: &[f32], out: &mut [f32]) {
    debug_assert_eq!(block.len(), out.len());
    // amax + first-index argmax (strict > keeps the FIRST maximum).
    let mut amax = 0f32;
    let mut aidx = 0usize;
    for (i, &a) in block.iter().enumerate() {
        let m = a.abs();
        if m > amax {
            amax = m;
            aidx = i;
        }
    }
    if amax == 0.0 {
        // All-zero block: guarded, never NaN (D1).
        out.fill(0.0);
        return;
    }
    let d = f16_round(block[aidx].signum() * amax / 2.0);
    if d == 0.0 || !d.is_finite() {
        // amax/2 underflowed f16 (subnormal-to-zero): the block is at the f16
        // floor; quantizing would collapse it to zeros. Emit the nearest
        // representable grid (all-zero) — the kernel must take the SAME branch
        // (G-i3 cases include it via the subnormal fixture).
        out.fill(0.0);
        return;
    }
    for (o, &a) in out.iter_mut().zip(block.iter()) {
        let q = (a / d).round().clamp(-1.0, 2.0);
        *o = q * d;
    }
}

/// The A4 grid math for one 32-element block (host reference).
#[inline]
pub fn a4_quant_dequant_block(block: &[f32], out: &mut [f32]) {
    debug_assert_eq!(block.len(), out.len());
    let mut mn = f32::INFINITY;
    let mut mx = f32::NEG_INFINITY;
    for &a in block {
        if a < mn {
            mn = a;
        }
        if a > mx {
            mx = a;
        }
    }
    let range = mx - mn;
    if range <= 0.0 || !range.is_finite() {
        // Constant block (range 0 / non-finite guard): any grid point
        // reproduces a constant exactly — emit min (== max == every element
        // when finite; the non-finite case emits the input unchanged, which
        // the kernel mirrors and G-i3's fixture carries).
        if range == 0.0 && mn.is_finite() {
            out.fill(mn);
        } else {
            out.copy_from_slice(block);
        }
        return;
    }
    let s = f16_round(range / 15.0);
    if s == 0.0 || !s.is_finite() {
        out.copy_from_slice(block);
        return;
    }
    for (o, &a) in out.iter_mut().zip(block.iter()) {
        let q = ((a - mn) / s).round().clamp(0.0, 15.0);
        *o = q * s + mn;
    }
}

/// The A8 control grid math for one 128-element block (host reference).
#[inline]
pub fn a8_quant_dequant_block(block: &[f32], out: &mut [f32]) {
    debug_assert_eq!(block.len(), out.len());
    let mut amax = 0f32;
    for &a in block {
        let m = a.abs();
        if m > amax {
            amax = m;
        }
    }
    if amax == 0.0 {
        out.fill(0.0);
        return;
    }
    let d = f16_round(amax / 127.0);
    if d == 0.0 || !d.is_finite() {
        out.fill(0.0);
        return;
    }
    for (o, &a) in out.iter_mut().zip(block.iter()) {
        let q = (a / d).round().clamp(-127.0, 127.0);
        *o = q * d;
    }
}

/// f32 → f16 → f32 rounding (the grids' scale precision — D1's spec).
/// Implemented in pure integer bit math — the EXACT mirror of the CUDA and
/// CubeCL kernels' `f16_round` (G-i3's one-reference law: the host oracle
/// must be the same algorithm, not a third implementation). Equality with
/// `half::f16::from_f32(x).to_f32()` is pinned by the
/// `f16_round_matches_half_crate` test (run under cubecl_runtime, where the
/// `half` dep exists).
#[inline]
pub fn f16_round(x: f32) -> f32 {
    let bits = x.to_bits();
    let sign = bits & 0x8000_0000u32;
    let exp_f = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x007F_FFFF;
    if exp_f == 0xFF {
        // NaN/Inf never reach a scale (guards); mirror ±Inf.
        return f32::from_bits(sign | 0x7F80_0000);
    }
    if exp_f == 0 {
        // Subnormal f32: |x| < 2^-126 — below half the f16 subnormal step;
        // RNE gives ±0.
        return f32::from_bits(sign);
    }
    let e = exp_f - 127 + 15;
    if e >= 0x1F {
        return f32::from_bits(sign | 0x7F80_0000); // overflow → ±Inf
    }
    if e <= 0 {
        // f16-subnormal range: round x to the nearest multiple of 2^-24.
        // x = m24 × 2^(exp_f-150), m24 = mant|0x800000 → x/2^-24 = m24 >> (126-exp_f).
        let m24 = mant | 0x0080_0000u32;
        let sh = (126 - exp_f) as u32;
        if sh > 24 {
            // x/2^-24 < 0.5 for every m24 < 2^24 once sh > 24 — RNE gives 0.
            return f32::from_bits(sign);
        }
        let kept = m24 >> sh;
        let rem = m24 & ((1u32 << sh) - 1);
        let half = 1u32 << (sh - 1);
        let mut k = kept;
        if rem > half || (rem == half && (kept & 1) == 1) {
            k += 1;
        }
        if k >= 1024 {
            // Rounded up into the smallest f16 normal: 2^-14.
            return f32::from_bits(sign | (113u32 << 23));
        }
        if k == 0 {
            return f32::from_bits(sign);
        }
        // k × 2^-24, constructed from bits (exact; 1 ≤ k < 1024).
        let mut kk = k;
        let mut p = 0u32;
        while kk > 1 {
            kk >>= 1;
            p += 1;
        }
        let mant23 = k << (23 - p);
        let exp32 = p + 127 - 24;
        return f32::from_bits(sign | (exp32 << 23) | (mant23 & 0x007F_FFFF));
    }
    // Normal: round the 23-bit mantissa to 10 bits (RNE).
    let mut h10 = mant >> 13;
    let rem = mant & 0x1FFF;
    let half = 0x1000u32;
    let mut e2 = e;
    if rem > half || (rem == half && (h10 & 1) == 1) {
        h10 += 1;
        if h10 == 0x400 {
            h10 = 0;
            e2 = e + 1;
        }
    }
    if e2 >= 0x1F {
        return f32::from_bits(sign | 0x7F80_0000);
    }
    let exp32 = (e2 - 15 + 127) as u32;
    f32::from_bits(sign | (exp32 << 23) | (h10 << 13))
}

/// The full host-reference pass over a buffer viewed as `[rows][dim]`
/// row-major (rows = tokens). Blocks NEVER cross a row boundary — D1: per
/// token along the reduction dimension, no cross-token statistics. A row's
/// ragged tail (final partial block) is quantized as its own block. This is
/// the G-i3 oracle; both kernels mirror it exactly.
pub fn host_quant_dequant(grid: DqGrid, buf: &[f32], out: &mut [f32], dim: usize) {
    assert_eq!(buf.len(), out.len());
    assert!(buf.len().is_multiple_of(dim), "buffer must be a whole number of rows");
    for (src, dst) in buf.chunks(dim).zip(out.chunks_mut(dim)) {
        for (sb, db) in src.chunks(grid.block()).zip(dst.chunks_mut(grid.block())) {
            match grid {
                DqGrid::A2 => a2_quant_dequant_block(sb, db),
                DqGrid::A4 => a4_quant_dequant_block(sb, db),
                DqGrid::A8 => a8_quant_dequant_block(sb, db),
            }
        }
    }
}

// ─── Knobs + counters (D5; the AtomicBool/setter house pattern) ─────────────
//
// PER-PHASE launch counters (round-2 verdict fix): a pf-only cell that fires
// during decode (or vice versa) is INSTRUMENT-FAIL, never a quiet asymmetry.
// The expected-count arithmetic is FROZEN in the plan: prefill = 256/chunk
// ((3 sites × 64 layers) + (site 4 × 16 attention layers) + (site 5 × 48 GDN
// layers)); decode = 256 × (n_generated − 1) per item.

static FQ_ARM: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);
static FQ_GRID: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);
static FQ_PREFILL_LAUNCHES: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
static FQ_DECODE_LAUNCHES: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

const ARM_OFF: u8 = 0;
const ARM_PREFILL: u8 = 1;
const ARM_DECODE: u8 = 2;
const ARM_BOTH: u8 = 3;
const GRID_A2: u8 = 0;
const GRID_A4: u8 = 1;
const GRID_A8: u8 = 2;

/// Configure the instrument (the runner calls this per cell; the env vars
/// `RIIR_DQ_FQ_PHASE` / `RIIR_DQ_FQ_GRID` are convenience wrappers for
/// one-process-per-cell runs).
pub fn set_arm(arm: DqPhaseArm) {
    let v = match arm {
        DqPhaseArm::Off => ARM_OFF,
        DqPhaseArm::PrefillOnly => ARM_PREFILL,
        DqPhaseArm::DecodeOnly => ARM_DECODE,
        DqPhaseArm::Both => ARM_BOTH,
    };
    FQ_ARM.store(v, std::sync::atomic::Ordering::Relaxed);
}

pub fn set_grid(grid: DqGrid) {
    let v = match grid {
        DqGrid::A2 => GRID_A2,
        DqGrid::A4 => GRID_A4,
        DqGrid::A8 => GRID_A8,
    };
    FQ_GRID.store(v, std::sync::atomic::Ordering::Relaxed);
}

pub fn current_arm() -> DqPhaseArm {
    match FQ_ARM.load(std::sync::atomic::Ordering::Relaxed) {
        ARM_PREFILL => DqPhaseArm::PrefillOnly,
        ARM_DECODE => DqPhaseArm::DecodeOnly,
        ARM_BOTH => DqPhaseArm::Both,
        _ => DqPhaseArm::Off,
    }
}

pub fn current_grid() -> DqGrid {
    match FQ_GRID.load(std::sync::atomic::Ordering::Relaxed) {
        GRID_A4 => DqGrid::A4,
        GRID_A8 => DqGrid::A8,
        _ => DqGrid::A2,
    }
}

/// Read the env knobs (`RIIR_DQ_FQ_PHASE=off|prefill|decode|both`,
/// `RIIR_DQ_FQ_GRID=a2|a4|a8`). Unset/unknown = Off/A2 — the fail-closed
/// default (an unset instrument never injects).
pub fn apply_env() {
    let arm = std::env::var("RIIR_DQ_FQ_PHASE").unwrap_or_default();
    let arm = match arm.trim().to_ascii_lowercase().as_str() {
        "prefill" => DqPhaseArm::PrefillOnly,
        "decode" => DqPhaseArm::DecodeOnly,
        "both" => DqPhaseArm::Both,
        _ => DqPhaseArm::Off,
    };
    set_arm(arm);
    let grid = std::env::var("RIIR_DQ_FQ_GRID").unwrap_or_default();
    let grid = match grid.trim().to_ascii_lowercase().as_str() {
        "a4" => DqGrid::A4,
        "a8" => DqGrid::A8,
        _ => DqGrid::A2,
    };
    set_grid(grid);
}

/// The per-phase launch counters (D5's phase-leak detection).
pub fn fq_prefill_launches() -> u64 {
    FQ_PREFILL_LAUNCHES.load(std::sync::atomic::Ordering::Relaxed)
}
pub fn fq_decode_launches() -> u64 {
    FQ_DECODE_LAUNCHES.load(std::sync::atomic::Ordering::Relaxed)
}
pub fn fq_reset_counters() {
    FQ_PREFILL_LAUNCHES.store(0, std::sync::atomic::Ordering::Relaxed);
    FQ_DECODE_LAUNCHES.store(0, std::sync::atomic::Ordering::Relaxed);
}

#[inline]
#[cfg_attr(
    all(target_os = "macos", feature = "dq_phase_bench"),
    allow(dead_code, reason = "the prefill lane is CUDA-only (not-macos) — the caller exists on the 4090")
)]
pub(crate) fn note_prefill_launch() {
    FQ_PREFILL_LAUNCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}
#[inline]
pub(crate) fn note_decode_launch() {
    FQ_DECODE_LAUNCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// Whether the prefill-side injections should fire right now.
#[inline]
pub fn prefill_armed() -> bool {
    current_arm().prefill_armed()
}

/// Whether the decode-side injections should fire right now.
#[inline]
pub fn decode_armed() -> bool {
    current_arm().decode_armed()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn expected_pass(grid: DqGrid, buf: &[f32], dim: usize) -> Vec<f32> {
        let mut out = vec![0f32; buf.len()];
        host_quant_dequant(grid, buf, &mut out, dim);
        out
    }

    #[test]
    fn a2_grid_points_and_tie_break() {
        // A block whose max-magnitude element is NEGATIVE: the sign steers the
        // double step to the negative side (d < 0 → +2d is the most negative).
        let mut block = vec![0.1f32; 128];
        block[7] = -1.0; // first strict max at index 7 (tie-break: index 3 == 1.0 loses)
        block[3] = -1.0; // TIE — lowest index (3) must win the argmax
        block[40] = 0.5;
        let mut out = vec![0f32; 128];
        a2_quant_dequant_block(&block, &mut out);
        let d = f16_round(-1.0f32 / 2.0);
        assert_eq!(out[3], 2.0 * d); // the argmax element lands on +2d (D1)
        assert_eq!(out[7], 2.0 * d); // the tied element quantizes identically
        // 0.5 sits ON the grid: 0.5 = -1 × d (d = -0.5) → reproduced exactly.
        assert_eq!(out[40], 0.5);
        assert_eq!(out[0], 0.0); // 0.1/d rounds to 0
    }

    #[test]
    fn a2_positive_sign_side() {
        let mut block = vec![0.0f32; 128];
        block[10] = 2.0;
        block[20] = -0.9;
        let mut out = vec![0f32; 128];
        a2_quant_dequant_block(&block, &mut out);
        let d = f16_round(1.0f32);
        assert_eq!(out[10], 2.0 * d);
        assert_eq!(out[20], -d); // -0.9/d = -0.9 → round = -1
        assert!((out[20] + 1.0).abs() < 1e-6);
    }

    #[test]
    fn zero_block_is_zero_never_nan() {
        let block = vec![0f32; 128];
        let mut out = vec![f32::NAN; 128];
        a2_quant_dequant_block(&block, &mut out);
        assert!(out.iter().all(|&x| x == 0.0));
        let block4 = vec![0f32; 32];
        let mut out4 = vec![f32::NAN; 32];
        a4_quant_dequant_block(&block4, &mut out4);
        assert!(out4.iter().all(|&x| x == 0.0));
        let mut out8 = vec![f32::NAN; 128];
        a8_quant_dequant_block(&block, &mut out8);
        assert!(out8.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn a4_affine_levels_and_clamp() {
        // 32 elements spanning [0, 15]: s = f16(1.0); q = clamp(round(a), 0, 15).
        let block: Vec<f32> = (0..32).map(|i| i as f32 % 16.0).collect();
        let mut out = vec![0f32; 32];
        a4_quant_dequant_block(&block, &mut out);
        let s = f16_round(15.0f32 / 15.0);
        for (i, (&a, &o)) in block.iter().zip(out.iter()).enumerate() {
            let q = ((a - 0.0) / s).round().clamp(0.0, 15.0);
            assert_eq!(o, q * s, "index {i}");
        }
    }

    #[test]
    fn a4_constant_block_reproduces_exactly() {
        let block = vec![0.3f32; 32];
        let mut out = vec![0f32; 32];
        a4_quant_dequant_block(&block, &mut out);
        assert!(out.iter().all(|&x| x == 0.3));
    }

    #[test]
    fn a8_control_symmetric_int8() {
        let mut block = vec![0.01f32; 128];
        block[0] = 1.27;
        block[1] = -1.27;
        let mut out = vec![0f32; 128];
        a8_quant_dequant_block(&block, &mut out);
        let d = f16_round(1.27f32 / 127.0);
        assert!((out[0] - 127.0 * d).abs() < 1e-9);
        assert!((out[1] + 127.0 * d).abs() < 1e-9);
        // 0.01/d ≈ 1.0 → q=1 → d
        assert!((out[2] - d).abs() < 1e-9);
    }

    #[test]
    fn host_pass_blocks_never_cross_token_boundary() {
        // Two 128-element tokens: the second token's block statistics must not
        // see the first token's values. A2 sign comes from each token's own
        // argmax.
        let mut buf = vec![0f32; 256];
        buf[5] = -2.0; // token 0: negative max
        buf[128 + 9] = 2.0; // token 1: positive max
        let out = expected_pass(DqGrid::A2, &buf, 128);
        let d_neg = f16_round(-1.0f32);
        let d_pos = f16_round(1.0f32);
        assert_eq!(out[5], 2.0 * d_neg);
        assert_eq!(out[128 + 9], 2.0 * d_pos);
    }

    #[test]
    fn ragged_tail_is_own_block() {
        // 100 elements under A4 (32-blocks): the tail of 4 elements is its own
        // block.
        let mut buf = vec![0.5f32; 100];
        buf[99] = 3.0;
        let out = expected_pass(DqGrid::A4, &buf, 100);
        // The tail block [96..100) has min .5 max 3.0; s = f16(2.5/15) is not
        // an exact divisor, so the max endpoint lands within ONE grid step
        // (the clamp), and the min endpoint exactly.
        assert_eq!(out[96], 0.5);
        let s = f16_round(2.5f32 / 15.0);
        assert!((out[99] - 3.0).abs() <= s + 1e-9);
    }

    #[test]
    fn subnormal_f16_scale_guard() {
        // amax/2 in the f16-subnormal range: the guard branch emits zeros and
        // never NaN.
        let mut block = vec![1e-8f32; 128];
        block[0] = 2e-8;
        let mut out = vec![f32::NAN; 128];
        a2_quant_dequant_block(&block, &mut out);
        assert!(out.iter().all(|x| x.is_finite()));
    }

    /// G-i3's foundation: the pure-Rust `f16_round` (the host oracle, and the
    /// exact algorithm both kernels mirror) equals the `half` crate's
    /// conversion on every float class — normals across exponents, RNE ties,
    /// subnormals, under/overflow, zero, sign symmetry.
    #[cfg(feature = "cubecl_runtime")]
    #[test]
    fn f16_round_matches_half_crate() {
        let mut cases: Vec<f32> = vec![0.0, -0.0, 1.0, -1.0, 0.5, 2.0 / 3.0, 1e-3, 1e3];
        // Every exponent boundary ±1 in the f16 normal range, mantissa 0 and
        // mantissa all-ones and a tie pattern.
        for e in (-15i32..=16).step_by(1) {
            for m in [0u32, 0x7FFF, 0x1001, 0x1000, 0x0FFF] {
                let bits = (((e + 127) as u32) << 23) | m;
                cases.push(f32::from_bits(bits));
                cases.push(f32::from_bits(bits | 0x8000_0000));
            }
        }
        // Subnormal-range values and the exact halfway points of the f16
        // subnormal grid (2^-25 boundaries).
        for i in 1..64u32 {
            cases.push(i as f32 * 2f32.powi(-25));
            cases.push(-(i as f32) * 2f32.powi(-25));
        }
        // Underflow / overflow.
        cases.push(1e-45);
        cases.push(-1e-45);
        cases.push(7e4);
        cases.push(-7e4);
        cases.push(f32::MAX);
        for &x in &cases {
            assert_eq!(
                f16_round(x).to_bits(),
                half::f16::from_f32(x).to_f32().to_bits(),
                "x={x:e} (bits {:#x})",
                x.to_bits()
            );
        }
    }
}
