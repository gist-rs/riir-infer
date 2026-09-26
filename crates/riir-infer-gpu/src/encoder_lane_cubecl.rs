//! The encoder-lane support kernels (plan 611 S3 — T7 op-layer unification).
//!
//! The four non-matmul, non-norm ops the laya `Backend` attention block
//! composes: rotate-half RoPE, the head split/merge permutations, and the
//! embedding row gather. These are the serving lane's kernels — the engine
//! keeps its own fused families (`rope_geglu`, the causal attention set);
//! what lands here is what the `Backend` trait op order needs, callable
//! from both sides like every S1–S3 kernel.
//!
//! All four follow the S1b/S2 launcher discipline: dims ride the params
//! buffer as f32-encoded usize ([`f32_exact`], exactness bounded at 2²⁴),
//! WHOLE parent binds (no byte-offset handle views — the wgpu 32-byte
//! storage-bind alignment finding), `debug_assert_binding_at_least` guards
//! every declared length, and the dispatch covers the OUTPUT extent with
//! in-kernel bounds checks.
//!
//! # The RoPE pairing (why one thread owns a PAIR)
//!
//! Rotate-half swaps two positions per cos/sin lane: `o[j] = q1·c − q2·s`,
//! `o[j + half] = q2·c + q1·s`. A thread-per-element kernel would race —
//! the thread at `j` reads `q[j + half]` while the thread at `j + half`
//! writes it. One thread per (head, position, lane) PAIR reads both values
//! before writing either: race-free by construction, and the expressions
//! are the CPU lane's verbatim (`ops::apply_rope_inplace`), so parity is
//! exact modulo compiler contraction.

#[cfg(feature = "cubecl_runtime")]
use cubecl::prelude::*;

#[cfg(feature = "cubecl_runtime")]
use cubecl::server::Handle;

#[cfg(feature = "cubecl_runtime")]
use crate::cubecl_runtime::{debug_assert_binding_at_least, f32_exact};

/// The elementwise per-launch workgroup count (the elementwise module's
/// helper, mirrored — this file's dispatches are output-extent-shaped like
/// the elementwise family's, not row-shaped like the norms family's).
#[cfg(feature = "cubecl_runtime")]
fn wg_count(n: usize) -> u32 {
    n.div_ceil(256).max(1) as u32
}

// ---------------------------------------------------------------------------
// Rotate-half RoPE — in place, one thread per (head, position, lane) pair
// ---------------------------------------------------------------------------

/// Rotate-half RoPE on `[heads, seq, hd]` with `[seq, hd]` cos/sin rows,
/// in place (plan 611 S3). `params` = `[seq, hd]`; `hd` must be even.
///
/// Pairwise, race-free: thread tid → (h, pos, j) with `j < hd/2`, reads
/// `q1 = q[base + j]` / `q2 = q[base + hd/2 + j]`, writes both rotated
/// values — the CPU lane's expressions verbatim.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn rope_rotate_half_inplace_f32(q: &mut [f32], cos: &[f32], sin: &[f32], params: &[f32]) {
    let heads = params[0usize] as u32;
    let seq = params[1usize] as u32;
    let hd = params[2usize] as u32;
    let half = hd / 2u32;
    let pairs_per_head = seq * half;
    let tid = ABSOLUTE_POS as u32;
    let total = heads * pairs_per_head;

    if tid < total {
        let h = tid / pairs_per_head;
        let rem = tid % pairs_per_head;
        let pos = rem / half;
        let j = rem % half;
        let base = (h * seq + pos) * hd;
        let crow = pos * hd + j;
        let c = cos[crow as usize];
        let s = sin[crow as usize];
        let q1 = q[(base + j) as usize];
        let q2 = q[(base + half + j) as usize];
        q[(base + j) as usize] = q1 * c - q2 * s;
        q[(base + half + j) as usize] = q2 * c + q1 * s;
    }
}

/// Launcher for [`rope_rotate_half_inplace_f32`] (plan 611 S3).
///
/// # Safety
///
/// `q_handle` must back `heads * seq * hd` f32; `cos_handle`/`sin_handle`
/// must each back `seq * hd` f32; `hd` must be even and > 0.
#[cfg(feature = "cubecl_runtime")]
pub struct RopeRotateHalfCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl RopeRotateHalfCubeCL {
    /// # Safety
    ///
    /// See the struct doc — every extent is asserted here.
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        q_handle: Handle,
        heads: usize,
        seq: usize,
        hd: usize,
        cos_handle: Handle,
        sin_handle: Handle,
    ) {
        debug_assert_binding_at_least(&q_handle, heads * seq * hd, "Rope::q");
        debug_assert_binding_at_least(&cos_handle, seq * hd, "Rope::cos");
        debug_assert_binding_at_least(&sin_handle, seq * hd, "Rope::sin");
        assert!(heads > 0 && seq > 0, "rope: degenerate shape");
        assert!(
            hd > 0 && hd.is_multiple_of(2),
            "rope: hd must be even and > 0"
        );
        let params: &[f32] = &[f32_exact(heads), f32_exact(seq), f32_exact(hd)];
        let params_handle = crate::params_cache::params_handle(client, f32::as_bytes(params));
        let pairs = heads * seq * (hd / 2);
        // SAFETY: extents asserted above; one thread per pair, disjoint
        // writes, both reads before either write.
        unsafe {
            rope_rotate_half_inplace_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(wg_count(pairs), 1, 1),
                CubeDim::new_1d(256),
                BufferArg::from_raw_parts(q_handle, heads * seq * hd),
                BufferArg::from_raw_parts(cos_handle, seq * hd),
                BufferArg::from_raw_parts(sin_handle, seq * hd),
                BufferArg::from_raw_parts(params_handle, 3),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// split_heads / merge_heads — pure permutations
// ---------------------------------------------------------------------------

/// `[seq, row_stride]` at column `off` → `[heads, seq, hd]`:
/// `out[(h·seq + s)·hd + i] = src[s·row_stride + off + h·hd + i]` (plan
/// 611 S3). `params` = `[row_stride, off, seq, heads, hd]`.
///
/// The fused `[seq, 3d]` projection's contiguous-thirds split (off 0/d/2d);
/// `off` also carries the packed forward's qkv row-offset seam (the qkv
/// slab starts at element `qkv_off` — the caller folds it into `off`).
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn split_heads_f32(src: &[f32], out: &mut [f32], params: &[f32]) {
    let row_stride = params[0usize] as u32;
    let off = params[1usize] as u32;
    let seq = params[2usize] as u32;
    let hd = params[4usize] as u32;
    let tid = ABSOLUTE_POS as u32;
    let n = out.len() as u32;

    if tid < n {
        let row = tid / hd;
        let i = tid % hd;
        let s = row % seq;
        let h = row / seq;
        out[tid as usize] = src[(s * row_stride + off + h * hd + i) as usize];
    }
}

/// Launcher for [`split_heads_f32`] (plan 611 S3).
///
/// # Safety
///
/// `out_handle` must back `heads * seq * hd` f32; `src_handle` must back at
/// least `(seq − 1)·row_stride + off + heads·hd` f32.
#[cfg(feature = "cubecl_runtime")]
pub struct SplitHeadsCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl SplitHeadsCubeCL {
    /// # Safety
    ///
    /// See the struct doc — every extent is asserted here.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        src_handle: Handle,
        src_len: usize,
        out_handle: Handle,
        out_len: usize,
        row_stride: usize,
        off: usize,
        seq: usize,
        heads: usize,
        hd: usize,
    ) {
        let want = heads * seq * hd;
        assert_eq!(out_len, want, "split_heads: out extent");
        let src_need = (seq - 1) * row_stride + off + heads * hd;
        assert!(
            src_len >= src_need,
            "split_heads: src extent ({src_len} < {src_need})"
        );
        debug_assert_binding_at_least(&src_handle, src_len, "SplitHeads::src");
        debug_assert_binding_at_least(&out_handle, out_len, "SplitHeads::out");
        assert!(
            seq > 0 && heads > 0 && hd > 0,
            "split_heads: degenerate shape"
        );
        let params: &[f32] = &[
            f32_exact(row_stride),
            f32_exact(off),
            f32_exact(seq),
            f32_exact(heads),
            f32_exact(hd),
        ];
        let params_handle = crate::params_cache::params_handle(client, f32::as_bytes(params));
        // SAFETY: extents asserted above; the kernel bounds-checks tid < n.
        unsafe {
            split_heads_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(wg_count(out_len), 1, 1),
                CubeDim::new_1d(256),
                BufferArg::from_raw_parts(src_handle, src_len),
                BufferArg::from_raw_parts(out_handle, out_len),
                BufferArg::from_raw_parts(params_handle, 5),
            );
        }
    }
}

/// `[heads, seq, hd]` → `[seq, heads·hd]`:
/// `out[s·(heads·hd) + h·hd + i] = src[(h·seq + s)·hd + i]` (plan 611 S3).
/// `params` = `[seq, heads, hd]`.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn merge_heads_f32(src: &[f32], out: &mut [f32], params: &[f32]) {
    let seq = params[0usize] as u32;
    let heads = params[1usize] as u32;
    let hd = params[2usize] as u32;
    let d = heads * hd;
    let tid = ABSOLUTE_POS as u32;
    let n = out.len() as u32;

    if tid < n {
        let s = tid / d;
        let r = tid % d;
        let h = r / hd;
        out[tid as usize] = src[((h * seq + s) * hd + (r % hd)) as usize];
    }
}

/// Launcher for [`merge_heads_f32`] (plan 611 S3).
///
/// # Safety
///
/// `out_handle` must back `seq * heads * hd` f32; `src_handle` must back
/// `heads * seq * hd` f32.
#[cfg(feature = "cubecl_runtime")]
pub struct MergeHeadsCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl MergeHeadsCubeCL {
    /// # Safety
    ///
    /// See the struct doc — every extent is asserted here.
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        src_handle: Handle,
        out_handle: Handle,
        out_len: usize,
        seq: usize,
        heads: usize,
        hd: usize,
    ) {
        assert_eq!(out_len, seq * heads * hd, "merge_heads: out extent");
        debug_assert_binding_at_least(&src_handle, heads * seq * hd, "MergeHeads::src");
        debug_assert_binding_at_least(&out_handle, out_len, "MergeHeads::out");
        assert!(
            seq > 0 && heads > 0 && hd > 0,
            "merge_heads: degenerate shape"
        );
        let params: &[f32] = &[f32_exact(seq), f32_exact(heads), f32_exact(hd)];
        let params_handle = crate::params_cache::params_handle(client, f32::as_bytes(params));
        // SAFETY: extents asserted above; the kernel bounds-checks tid < n.
        unsafe {
            merge_heads_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(wg_count(out_len), 1, 1),
                CubeDim::new_1d(256),
                BufferArg::from_raw_parts(src_handle, heads * seq * hd),
                BufferArg::from_raw_parts(out_handle, out_len),
                BufferArg::from_raw_parts(params_handle, 3),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// gather_rows — the embedding row gather
// ---------------------------------------------------------------------------

/// `out[r·d + i] = x[rows[r]·d + i]` (plan 611 S3). `params` = `[d]`;
/// `rows` is the u32 index buffer (the host ids, uploaded by the launcher's
/// caller).
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn gather_rows_f32(x: &[f32], rows: &[u32], out: &mut [f32], params: &[f32]) {
    let d = params[0usize] as u32;
    let tid = ABSOLUTE_POS as u32;
    let n = out.len() as u32;

    if tid < n {
        let r = tid / d;
        let row = rows[r as usize];
        out[tid as usize] = x[(row * d + (tid % d)) as usize];
    }
}

/// Launcher for [`gather_rows_f32`] (plan 611 S3).
///
/// `rows_handle` is a **u32** buffer (one index per row) — the caller
/// uploads the host ids; this launcher validates only the shape contract
/// (every index must land inside `x`), scanning the host slice the caller
/// passes alongside.
///
/// # Safety
///
/// `out_handle` must back `rows.len() * d` f32; `x_handle` must back
/// `max(rows)·d + d` f32 — validated against the `rows_host` slice;
/// `rows_handle` must back `rows_host.len()` u32.
#[cfg(feature = "cubecl_runtime")]
pub struct GatherRowsCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl GatherRowsCubeCL {
    /// # Safety
    ///
    /// See the struct doc — every extent is asserted here.
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        x_handle: Handle,
        x_len: usize,
        rows_handle: Handle,
        rows_host: &[usize],
        out_handle: Handle,
        out_len: usize,
        d: usize,
    ) {
        assert!(d > 0, "gather_rows: degenerate shape");
        assert_eq!(out_len, rows_host.len() * d, "gather_rows: out extent");
        for (r, &row) in rows_host.iter().enumerate() {
            assert!(
                (row + 1) * d <= x_len,
                "gather_rows: row {r} index {row} out of x extent"
            );
        }
        debug_assert_binding_at_least(&x_handle, x_len, "GatherRows::x");
        debug_assert_binding_at_least(&out_handle, out_len, "GatherRows::out");
        // SAFETY: u32 binding size — width differs from the f32 helper, so
        // the byte check is spelled here rather than reusing it.
        debug_assert!(
            rows_handle.size_in_used() >= (rows_host.len() * 4) as u64,
            "GatherRows::rows: binding too small"
        );
        let params: &[f32] = &[f32_exact(d)];
        let params_handle = crate::params_cache::params_handle(client, f32::as_bytes(params));
        // SAFETY: extents asserted above; the kernel bounds-checks tid < n.
        unsafe {
            gather_rows_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(wg_count(out_len), 1, 1),
                CubeDim::new_1d(256),
                BufferArg::from_raw_parts(x_handle, x_len),
                BufferArg::from_raw_parts(rows_handle, rows_host.len()),
                BufferArg::from_raw_parts(out_handle, out_len),
                BufferArg::from_raw_parts(params_handle, 1),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Tests — op parity vs the host references (the ops.rs semantics verbatim)
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "cubecl_runtime"))]
mod tests {
    use crate::cubecl_runtime::{ActiveRuntime, CubeCLContext};

    use super::*;

    fn noise(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((s >> 8) as f32) / 8_388_608.0 - 1.0
            })
            .collect()
    }

    fn run_and_read(
        client: &ComputeClient<ActiveRuntime>,
        launch: impl FnOnce(),
        out_handle: &cubecl::server::Handle,
        len: usize,
    ) -> Vec<f32> {
        launch();
        let bytes = client.read_one(out_handle.clone()).expect("read out");
        f32::from_bytes(&bytes)[..len].to_vec()
    }

    /// Rope vs the CPU lane's pairwise form — exact modulo contraction
    /// (the 1e-5 bound catches a wrong-form rope: swapped halves, missing
    /// rotate, cos/sin column mixups all land far outside it).
    #[test]
    fn test_rope_rotate_half_matches_cpu() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();
        let (heads, seq, hd) = (3usize, 9usize, 16usize);
        let q = noise(heads * seq * hd, 0x5150);
        let cos = noise(seq * hd, 0xc051);
        let sin = noise(seq * hd, 0x51b3);
        let mut want = q.clone();
        let half = hd / 2;
        for h in 0..heads {
            for pos in 0..seq {
                let base = (h * seq + pos) * hd;
                for j in 0..half {
                    let c = cos[pos * hd + j];
                    let s = sin[pos * hd + j];
                    let q1 = want[base + j];
                    let q2 = want[base + half + j];
                    want[base + j] = q1 * c - q2 * s;
                    want[base + half + j] = q2 * c + q1 * s;
                }
            }
        }
        let q_h = client.create_from_slice(f32::as_bytes(&q));
        let cos_h = client.create_from_slice(f32::as_bytes(&cos));
        let sin_h = client.create_from_slice(f32::as_bytes(&sin));
        let got = run_and_read(
            &client,
            || unsafe {
                RopeRotateHalfCubeCL::launch::<ActiveRuntime>(
                    &client,
                    q_h.clone(),
                    heads,
                    seq,
                    hd,
                    cos_h.clone(),
                    sin_h.clone(),
                )
            },
            &q_h,
            q.len(),
        );
        let max_err = want
            .iter()
            .zip(&got)
            .map(|(w, g)| (w - g).abs())
            .fold(0.0f32, f32::max);
        println!("rope ({heads}×{seq}×{hd}): max_err = {max_err:.3e}");
        assert!(max_err < 1e-5, "rope diverged: {max_err:.3e}");
    }

    /// split + merge round-trip = identity, and split alone vs the host
    /// permutation (the head-lane copies must be EXACT — they move bytes).
    #[test]
    fn test_split_merge_heads_match_cpu() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();
        let (seq, heads, hd) = (11usize, 4usize, 8usize);
        let row_stride = heads * hd + 3; // deliberately ragged
        let off = 2usize;
        let src = noise(seq * row_stride, 0x5e17);
        let src_h = client.create_from_slice(f32::as_bytes(&src));
        let split_len = heads * seq * hd;
        let split_h = client.empty(split_len * core::mem::size_of::<f32>());
        let got_split = run_and_read(
            &client,
            || unsafe {
                SplitHeadsCubeCL::launch::<ActiveRuntime>(
                    &client,
                    src_h.clone(),
                    src.len(),
                    split_h.clone(),
                    split_len,
                    row_stride,
                    off,
                    seq,
                    heads,
                    hd,
                )
            },
            &split_h,
            split_len,
        );
        let mut want_split = vec![0f32; split_len];
        for h in 0..heads {
            for s in 0..seq {
                let begin = s * row_stride + off + h * hd;
                let dst = (h * seq + s) * hd;
                want_split[dst..dst + hd].copy_from_slice(&src[begin..begin + hd]);
            }
        }
        assert_eq!(got_split, want_split, "split_heads must be exact");

        let d = heads * hd;
        let merge_len = seq * d;
        let merge_h = client.empty(merge_len * core::mem::size_of::<f32>());
        let got_merge = run_and_read(
            &client,
            || unsafe {
                MergeHeadsCubeCL::launch::<ActiveRuntime>(
                    &client,
                    split_h.clone(),
                    merge_h.clone(),
                    merge_len,
                    seq,
                    heads,
                    hd,
                )
            },
            &merge_h,
            merge_len,
        );
        let mut want_merge = vec![0f32; merge_len];
        for h in 0..heads {
            for s in 0..seq {
                let sb = (h * seq + s) * hd;
                let db = s * d + h * hd;
                want_merge[db..db + hd].copy_from_slice(&want_split[sb..sb + hd]);
            }
        }
        assert_eq!(got_merge, want_merge, "merge_heads must be exact");
    }

    /// gather_rows vs the host copy — exact, including a repeated and an
    /// out-of-order row index.
    #[test]
    fn test_gather_rows_matches_cpu() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();
        let (table_rows, d) = (17usize, 12usize);
        let x = noise(table_rows * d, 0x6a7e);
        let rows: Vec<usize> = vec![3, 0, 3, 16, 7];
        let out_len = rows.len() * d;
        let x_h = client.create_from_slice(f32::as_bytes(&x));
        let rows_u32: Vec<u32> = rows.iter().map(|&r| r as u32).collect();
        let rows_h = client.create_from_slice(u32::as_bytes(&rows_u32));
        let out_h = client.empty(out_len * core::mem::size_of::<f32>());
        let got = run_and_read(
            &client,
            || unsafe {
                GatherRowsCubeCL::launch::<ActiveRuntime>(
                    &client,
                    x_h.clone(),
                    x.len(),
                    rows_h.clone(),
                    &rows,
                    out_h.clone(),
                    out_len,
                    d,
                )
            },
            &out_h,
            out_len,
        );
        let mut want = vec![0f32; out_len];
        for (r, &row) in rows.iter().enumerate() {
            want[r * d..(r + 1) * d].copy_from_slice(&x[row * d..(row + 1) * d]);
        }
        assert_eq!(got, want, "gather_rows must be exact");
    }
}
