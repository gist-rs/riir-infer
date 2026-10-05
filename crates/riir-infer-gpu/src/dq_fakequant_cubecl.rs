//! Plan 614 / Issue 026 — the CubeCL twin of the DQ fake-quant kernels
//! (the decode lane's on-device quantize→dequantize; G-i3 gates it against
//! the SAME host reference as the CUDA kernel — proving the two lanes agree).
//!
//! One workgroup per block (`CUBE_POS_X` = the workgroup/block index,
//! `UNIT_POS` = the thread within the block): load the block into registers,
//! reduce in shared memory (A2: (|a|, idx) with lowest-index tie-break; A4:
//! min/max; A8: amax), round the scale through the BIT-EXACT f16 emulation,
//! quantize from each thread's saved original, write back IN PLACE. Rows are
//! tokens; blocks never cross a row boundary (D1: per token along the
//! reduction dim). Scalars travel in a params buffer (the rmsnorm house
//! pattern — CubeCL v0.10 u32-casting issues).
//!
//! The f32→f16→f32 round-trip is integer bit operations (cubecl's in-kernel
//! `FloatBits` reinterpret) and equals `half::f16::from_f32(x).to_f32()` on
//! every input class (normal, subnormal, overflow, zero) — pinned by the
//! G-i3 unit test against the host reference (which uses `half` directly).

#[cfg(feature = "cubecl_runtime")]
use cubecl::prelude::*;
#[cfg(feature = "cubecl_runtime")]
use cubecl::server::Handle;

use crate::dq_fakequant::DqGrid;

// ---------------------------------------------------------------------------
// The bit-exact f32 → f16 → f32 round-trip
// ---------------------------------------------------------------------------

/// IEEE-754 binary16 conversion with round-to-nearest-even, then immediate
/// widening back to f32 — the scale-precision step of every grid (D1).
#[cfg(feature = "cubecl_runtime")]
#[allow(clippy::collapsible_if, reason = "CubeCL v0.10: nested-if statement form is required — no conditional expressions as values")]
#[cube]
pub(crate) fn f16_round_bits(bits: u32) -> f32 {
    // IEEE-754 binary16 RNE round-trip == half::f16::from_f32(x).to_f32().
    // (CubeCL: no `return` in #[cube] fns — every path assigns `r`; nested-if
    // form avoids conditional-expression values.)
    let sign = bits & 0x8000_0000u32;
    let exp_f = ((bits >> 23) & 0xFFu32) as i32;
    let mant = bits & 0x007F_FFFFu32;
    let mut r = f32::from_bits(bits);
    if exp_f == 0xFF {
        // NaN / Inf never reach a scale (host guards); emit ±Inf.
        r = f32::from_bits(sign | 0x7F80_0000u32);
    }
    if exp_f == 0 {
        // Subnormal f32 input: |x| < 2^-126 — below half the f16 subnormal
        // step — RNE gives ±0.
        r = f32::from_bits(sign);
    }
    if exp_f > 0 {
        if exp_f < 0xFF {
            let e = exp_f - 127 + 15;
            if e >= 0x1F {
                // Overflow: f16 rounds large normals up to ±Inf.
                r = f32::from_bits(sign | 0x7F80_0000u32);
            }
            if e < 0x1F {
                if e > 0 {
                    // Normal: round the 23-bit mantissa to 10 bits (RNE).
                    let mut h10 = mant >> 13;
                    let rem = mant & 0x1FFFu32;
                    let half = 0x1000u32;
                    let mut e2 = e;
                    if rem > half {
                        h10 = h10 + 1u32;
                    }
                    if rem == half {
                        if (h10 & 1u32) == 1u32 {
                            h10 = h10 + 1u32;
                        }
                    }
                    if h10 == 0x400u32 {
                        h10 = 0;
                        e2 = e + 1;
                    }
                    if e2 >= 0x1F {
                        r = f32::from_bits(sign | 0x7F80_0000u32);
                    }
                    if e2 < 0x1F {
                        let exp32 = (e2 - 15 + 127) as u32;
                        r = f32::from_bits(sign | (exp32 << 23) | (h10 << 13));
                    }
                }
                if e <= 0 {
                    // f16-subnormal range: round to the nearest multiple of
                    // 2^-24 (x = m24 × 2^(exp_f-150), m24 = mant|0x800000, so
                    // x/2^-24 = m24 >> (126-exp_f)).
                    let m24 = mant | 0x0080_0000u32;
                    let sh = (126 - exp_f) as u32;
                    if sh > 24 {
                        // x/2^-24 < 0.5 for every m24 < 2^24 once sh > 24.
                        r = f32::from_bits(sign);
                    }
                    if sh <= 24 {
                        let kept = m24 >> sh;
                        let rem = m24 & ((1u32 << sh) - 1u32);
                        let half = 1u32 << (sh - 1);
                        let mut k = kept;
                        if rem > half {
                            k = kept + 1u32;
                        }
                        if rem == half {
                            if (kept & 1u32) == 1u32 {
                                k = kept + 1u32;
                            }
                        }
                        if k >= 1024u32 {
                            // Rounded up into the smallest f16 NORMAL: 2^-14.
                            r = f32::from_bits(sign | (113u32 << 23));
                        }
                        if k < 1024u32 {
                            if k == 0u32 {
                                r = f32::from_bits(sign);
                            }
                            if k > 0u32 {
                                // k × 2^-24 from bits (leading bit p,
                                // normalized mantissa, exp = p - 24 + 127).
                                let mut kk = k;
                                let mut p = 0u32;
                                while kk > 1u32 {
                                    kk = kk >> 1;
                                    p = p + 1u32;
                                }
                                let mant23 = k << (23u32 - p);
                                let exp32 = p + 127u32 - 24u32;
                                r = f32::from_bits(sign | (exp32 << 23) | (mant23 & 0x007F_FFFFu32));
                            }
                        }
                    }
                }
            }
        }
    }
    r
}

/// Round half AWAY FROM ZERO (the D1 frozen spec — Rust `f32::round`, the RTN
/// encoder convention). The GPU's native `round` is half-to-EVEN, which
/// diverges on every exact .5 ratio; this explicit form matches the host
/// oracle bit-exactly (pinned by G-i3's tie fixtures).
#[cfg(feature = "cubecl_runtime")]
#[cube]
pub(crate) fn round_half_away(x: f32) -> f32 {
    // ±0 passes through with its sign (0.0 × negative d must stay -0.0).
    let mut r = x;
    if x != f32::new(0.0f32) {
        let ax = x.abs();
        let ar = (ax + f32::new(0.5f32)).floor();
        r = ar;
        if x < f32::new(0.0f32) {
            r = -ar;
        }
    }
    r
}

// ---------------------------------------------------------------------------
// A2 — asymmetric 4-level, per-128, d = sign(argmax|a|)·f16(amax/2)
// ---------------------------------------------------------------------------

/// One workgroup per 128-block (grid = rows × blocks_per_row, `CUBE_POS_X` =
/// the flat block index). Workgroup size MUST be 128.
#[cfg(feature = "cubecl_runtime")]
#[allow(
    clippy::collapsible_if,
    reason = "CubeCL v0.10: nested-if statement form is required — no conditional expressions as values"
)]
#[cube(launch_unchecked)]
pub(crate) fn dq_fq_a2(buf: &mut [f32], params: &[f32]) {
    // params: [dim, blocks_per_row, rows, elem_offset] — the offset (Issue
    // 033 KV lane) shifts every access into a larger slab; the activation
    // callers pass 0.
    let dim = params[0usize] as u32;
    let bpr = params[1usize] as u32;
    let rows = params[2usize] as u32;
    let off = params[3usize] as u32;
    let wid = CUBE_POS_X;
    let tid = UNIT_POS;
    let row = wid / bpr;
    let blk = wid % bpr;
    let in_grid = row < rows;
    let row_base = row * dim;
    let base = row_base + blk * 128u32;
    let row_end = row_base + dim;
    let in_blk = base < row_end;
    let n_here = if base + 128u32 <= row_end {
        128u32
    } else {
        row_end - base
    };
    let live = in_grid && in_blk && tid < n_here;

    // EVERY thread participates in the shared-memory init (out-of-range lanes
    // contribute m = 0 — a ragged tail's unwritten slot must never inject
    // garbage into the reduction; the G-i3 ragged fixture pins this).
    let orig = if live {
        buf[(off + base + tid) as usize]
    } else {
        f32::new(0.0f32)
    };
    let m = orig.abs();

    let mut sm = Shared::<[f32]>::new_slice(128usize);
    let mut si = Shared::<[u32]>::new_slice(128usize);
    sm[tid as usize] = m;
    si[tid as usize] = tid;
    sync_cube();

    // (|a|, idx) argmax reduce; ties → SMALLER index (D1's tie-break).
    let mut stride = 64usize;
    while stride > 0 {
        if (tid as usize) < stride {
            let om = sm[tid as usize + stride];
            let oi = si[tid as usize + stride];
            let mym = sm[tid as usize];
            let myi = si[tid as usize];
            if om > mym {
                sm[tid as usize] = om;
                si[tid as usize] = oi;
            }
            if om == mym {
                if oi < myi {
                    sm[tid as usize] = om;
                    si[tid as usize] = oi;
                }
            }
        }
        sync_cube();
        stride /= 2;
    }

    if live {
        let amax = sm[0];
        if amax == f32::new(0.0f32) {
            buf[(off + base + tid) as usize] = f32::new(0.0f32);
            terminate!();
        }
        // The winning index's SIGNED value supplies the sign (lowest index on
        // ties, by construction). buf[widx] is still original — no writes yet.
        let widx = si[0] as usize;
        let sign_val = buf[(off + base + widx as u32) as usize];
        let mut sgn = f32::new(1.0f32);
        if sign_val < f32::new(0.0f32) {
            sgn = f32::new(-1.0f32);
        }
        let d = f16_round_bits((sgn * amax / f32::new(2.0f32)).to_bits());
        if d == f32::new(0.0f32) {
            buf[(off + base + tid) as usize] = f32::new(0.0f32);
            terminate!();
        }
        let q = round_half_away(orig / d).clamp(f32::new(-1.0f32), f32::new(2.0f32));
        buf[(off + base + tid) as usize] = q * d;
    }
}

// ---------------------------------------------------------------------------
// A4 — affine 16-level, per-32, s = f16((max−min)/15)
// ---------------------------------------------------------------------------

/// One workgroup per 32-block. Workgroup size MUST be 32.
#[cfg(feature = "cubecl_runtime")]
#[allow(
    clippy::neg_cmp_op_on_partial_ord,
    reason = "the guard must also catch NaN (range NaN → constant-block branch); CubeCL has no partial_cmp"
)]
#[cube(launch_unchecked)]
pub(crate) fn dq_fq_a4(buf: &mut [f32], params: &[f32]) {
    let dim = params[0usize] as u32;
    let bpr = params[1usize] as u32;
    let rows = params[2usize] as u32;
    let off = params[3usize] as u32;
    let wid = CUBE_POS_X;
    let tid = UNIT_POS;
    let row = wid / bpr;
    let blk = wid % bpr;
    let in_grid = row < rows;
    let row_base = row * dim;
    let base = row_base + blk * 32u32;
    let row_end = row_base + dim;
    let in_blk = base < row_end;
    let n_here = if base + 32u32 <= row_end {
        32u32
    } else {
        row_end - base
    };
    let live = in_grid && in_blk && tid < n_here;

    // EVERY thread inits the shared slots (out-of-range lanes mirror the
    // live min/max — contributing neutrally: for min use +INF-analog, for max
    // -INF-analog, so they can never win; orig stays 0 for the pass-through).
    let orig = if live {
        buf[(off + base + tid) as usize]
    } else {
        f32::new(0.0f32)
    };
    let mut smn = Shared::<[f32]>::new_slice(32usize);
    let mut smx = Shared::<[f32]>::new_slice(32usize);
    if live {
        smn[tid as usize] = orig;
        smx[tid as usize] = orig;
    }
    if !live {
        smn[tid as usize] = f32::new(1e30f32);
        smx[tid as usize] = f32::new(-1e30f32);
    }
    sync_cube();

    let mut stride = 16usize;
    while stride > 0 {
        if (tid as usize) < stride {
            let on = smn[tid as usize + stride];
            let ox = smx[tid as usize + stride];
            if on < smn[tid as usize] {
                smn[tid as usize] = on;
            }
            if ox > smx[tid as usize] {
                smx[tid as usize] = ox;
            }
        }
        sync_cube();
        stride /= 2;
    }

    if live {
        let mn = smn[0];
        let mx = smx[0];
        let range = mx - mn;
        if !(range > f32::new(0.0f32)) {
            // Constant block: reproduces exactly.
            buf[(off + base + tid) as usize] = mn;
            terminate!();
        }
        let s = f16_round_bits((range / f32::new(15.0f32)).to_bits());
        if s == f32::new(0.0f32) {
            buf[(off + base + tid) as usize] = orig;
            terminate!();
        }
        let q = round_half_away((orig - mn) / s).clamp(f32::new(0.0f32), f32::new(15.0f32));
        buf[(off + base + tid) as usize] = q * s + mn;
    }
}

// ---------------------------------------------------------------------------
// A8 — symmetric int8 control, per-128, d = f16(amax/127)
// ---------------------------------------------------------------------------

/// One workgroup per 128-block. Workgroup size MUST be 128.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
pub(crate) fn dq_fq_a8(buf: &mut [f32], params: &[f32]) {
    let dim = params[0usize] as u32;
    let bpr = params[1usize] as u32;
    let rows = params[2usize] as u32;
    let off = params[3usize] as u32;
    let wid = CUBE_POS_X;
    let tid = UNIT_POS;
    let row = wid / bpr;
    let blk = wid % bpr;
    let in_grid = row < rows;
    let row_base = row * dim;
    let base = row_base + blk * 128u32;
    let row_end = row_base + dim;
    let in_blk = base < row_end;
    let n_here = if base + 128u32 <= row_end {
        128u32
    } else {
        row_end - base
    };
    let live = in_grid && in_blk && tid < n_here;

    let orig = if live {
        buf[(off + base + tid) as usize]
    } else {
        f32::new(0.0f32)
    };
    let m = orig.abs();

    let mut sm = Shared::<[f32]>::new_slice(128usize);
    sm[tid as usize] = m;
    sync_cube();

    let mut stride = 64usize;
    while stride > 0 {
        if (tid as usize) < stride {
            let om = sm[tid as usize + stride];
            if om > sm[tid as usize] {
                sm[tid as usize] = om;
            }
        }
        sync_cube();
        stride /= 2;
    }

    if live {
        let amax = sm[0];
        if amax == f32::new(0.0f32) {
            buf[(off + base + tid) as usize] = f32::new(0.0f32);
            terminate!();
        }
        let d = f16_round_bits((amax / f32::new(127.0f32)).to_bits());
        if d == f32::new(0.0f32) {
            buf[(off + base + tid) as usize] = f32::new(0.0f32);
            terminate!();
        }
        let q = round_half_away(orig / d).clamp(f32::new(-127.0f32), f32::new(127.0f32));
        buf[(off + base + tid) as usize] = q * d;
    }
}

// ---------------------------------------------------------------------------
// The launcher (the decode injection sites call this once per pass)
// ---------------------------------------------------------------------------

/// In-place fake-quant of `buf` viewed as `[rows][dim]` row-major (decode:
/// rows = 1). No dispatch unless the decode arm is armed. Bumps the decode
/// launch counter on every fired pass (D5's phase-leak detection).
#[cfg(feature = "cubecl_runtime")]
#[cfg_attr(
    not(feature = "dq_phase_bench"),
    allow(dead_code, reason = "the only callers are the dq_phase_bench-gated Plan-614 injection sites")
)]
pub(crate) fn launch_decode_pass<R: Runtime>(
    client: &ComputeClient<R>,
    buf: Handle,
    dim: usize,
    rows: usize,
    grid: DqGrid,
) {
    use crate::dq_fakequant::{decode_armed, note_decode_launch};

    if !decode_armed() {
        return;
    }
    let block = grid.block();
    let blocks_per_row = dim.div_ceil(block);
    let total_blocks = (blocks_per_row * rows) as u32;
    let params: [f32; 4] = [dim as f32, blocks_per_row as f32, rows as f32, 0.0f32];
    let params_handle = crate::params_cache::params_handle(client, f32::as_bytes(&params));
    let count = CubeCount::Static(total_blocks.max(1), 1, 1);
    let cube_dim = if block == 32 {
        CubeDim::new_1d(32)
    } else {
        CubeDim::new_1d(128)
    };
    let n = dim * rows;
    unsafe {
        match grid {
            DqGrid::A2 => dq_fq_a2::launch_unchecked::<R>(
                client,
                count,
                cube_dim,
                BufferArg::from_raw_parts(buf.clone(), n),
                BufferArg::from_raw_parts(params_handle, 3),
            ),
            DqGrid::A4 => dq_fq_a4::launch_unchecked::<R>(
                client,
                count,
                cube_dim,
                BufferArg::from_raw_parts(buf.clone(), n),
                BufferArg::from_raw_parts(params_handle, 3),
            ),
            DqGrid::A8 => dq_fq_a8::launch_unchecked::<R>(
                client,
                count,
                cube_dim,
                BufferArg::from_raw_parts(buf.clone(), n),
                BufferArg::from_raw_parts(params_handle, 3),
            ),
        }
    }
    note_decode_launch();
}

/// Issue 033 — the KV-STORE axis (CubeCL runtime): in-place fake-quant of
/// the row region at element offset `off_elems` inside the cache slab
/// (`buf` viewed as `[off_elems .. off_elems + rows*dim]`). Gated on the
/// KV arm; bumps the KV launch counter. The kernels index through
/// `params[3]`, so the SAME compiled kernels serve the activation lane
/// (offset 0) and this one — no second kernel set.
#[cfg(feature = "cubecl_runtime")]
#[cfg_attr(
    not(feature = "dq_phase_bench"),
    allow(dead_code, reason = "the only callers are the dq_phase_bench-gated KV injection sites (Issue 033)")
)]
pub(crate) fn launch_kv_pass<R: Runtime>(
    client: &ComputeClient<R>,
    buf: Handle,
    dim: usize,
    rows: usize,
    off_elems: usize,
    grid: DqGrid,
) {
    use crate::dq_fakequant::{kv_armed, note_kv_launch};

    if !kv_armed() {
        return;
    }
    let block = grid.block();
    let blocks_per_row = dim.div_ceil(block);
    let total_blocks = (blocks_per_row * rows) as u32;
    let params: [f32; 4] = [
        dim as f32,
        blocks_per_row as f32,
        rows as f32,
        off_elems as f32,
    ];
    let params_handle = crate::params_cache::params_handle(client, f32::as_bytes(&params));
    let count = CubeCount::Static(total_blocks.max(1), 1, 1);
    let cube_dim = if block == 32 {
        CubeDim::new_1d(32)
    } else {
        CubeDim::new_1d(128)
    };
    // The arg spans the offset + the quantized region (the kernel indexes
    // up to off + rows*dim - 1).
    let n = off_elems + dim * rows;
    unsafe {
        match grid {
            DqGrid::A2 => dq_fq_a2::launch_unchecked::<R>(
                client,
                count,
                cube_dim,
                BufferArg::from_raw_parts(buf.clone(), n),
                BufferArg::from_raw_parts(params_handle, 4),
            ),
            DqGrid::A4 => dq_fq_a4::launch_unchecked::<R>(
                client,
                count,
                cube_dim,
                BufferArg::from_raw_parts(buf.clone(), n),
                BufferArg::from_raw_parts(params_handle, 4),
            ),
            DqGrid::A8 => dq_fq_a8::launch_unchecked::<R>(
                client,
                count,
                cube_dim,
                BufferArg::from_raw_parts(buf.clone(), n),
                BufferArg::from_raw_parts(params_handle, 4),
            ),
        }
    }
    note_kv_launch();
}

// ---------------------------------------------------------------------------
// G-i3 test support (feature-gated with the module; used by
// tests/dq_fakequant_gi3.rs — NOT part of the instrument's hot path)
// ---------------------------------------------------------------------------

/// Run one fake-quant pass over `buf` viewed as `[rows][dim]` REGARDLESS of
/// the phase knobs, WITHOUT bumping the launch counters (the G-i3 oracle
/// exercises the kernels, not the phase wiring). Test-only.
#[cfg(feature = "cubecl_runtime")]
pub fn test_support_launch<R: Runtime>(
    client: &ComputeClient<R>,
    buf: &Handle,
    dim: usize,
    rows: usize,
    grid: DqGrid,
) {
    let block = grid.block();
    let blocks_per_row = dim.div_ceil(block);
    let total_blocks = (blocks_per_row * rows) as u32;
    let params: [f32; 4] = [dim as f32, blocks_per_row as f32, rows as f32, 0.0f32];
    let params_handle = crate::params_cache::params_handle(client, f32::as_bytes(&params));
    let count = CubeCount::Static(total_blocks.max(1), 1, 1);
    let cube_dim = if block == 32 {
        CubeDim::new_1d(32)
    } else {
        CubeDim::new_1d(128)
    };
    let n = dim * rows;
    unsafe {
        match grid {
            DqGrid::A2 => dq_fq_a2::launch_unchecked::<R>(
                client,
                count,
                cube_dim,
                BufferArg::from_raw_parts(buf.clone(), n),
                BufferArg::from_raw_parts(params_handle, 4),
            ),
            DqGrid::A4 => dq_fq_a4::launch_unchecked::<R>(
                client,
                count,
                cube_dim,
                BufferArg::from_raw_parts(buf.clone(), n),
                BufferArg::from_raw_parts(params_handle, 4),
            ),
            DqGrid::A8 => dq_fq_a8::launch_unchecked::<R>(
                client,
                count,
                cube_dim,
                BufferArg::from_raw_parts(buf.clone(), n),
                BufferArg::from_raw_parts(params_handle, 4),
            ),
        }
    }
}
