//! Plan 602 Phase B (B1) — CubeCL/Metal Hadamard rotation kernels for the
//! Bonsai-2 folded-ternary runtime (the M3 mirror of Issue 980 T4's cudarc
//! lane, `deltanet_rotation_cudarc.rs`).
//!
//! The CPU contract lives in `riir-infer-core::deltanet::rotation`
//! (`rotate_forward_inplace` / `rotate_inverse_inplace` /
//! `permute_gdn_v_grouped_inplace`); every kernel here is a numeric twin of
//! those functions, dispatched through the CubeCL (Metal/wgpu) client so the
//! folded matmuls consume rotated activations GPU-resident. The rotation runs
//! between the RMSNorm and the int8 activation quantize — it commutes with
//! neither.
//!
//! Kernel set (only dispatched when the loaded model declares
//! `prism.hadamard`; the split, non-fused set — the cudarc K1–K5 fusion
//! escalation is a named follow-up rung in Plan 602 B5, not the baseline):
//!
//! | kernel | CPU twin | used at |
//! |---|---|---|
//! | `fwht_rotate_forward_f32` | `rotate_forward_inplace` | every folded matmul input (sign→FWHT) |
//! | `fwht_rotate_inverse_f32` | `rotate_inverse_inplace` | token-embedding lookup (FWHT→sign) |
//! | `fwht_forward_copy_f32` | memcpy + forward | whole-prefill staging (the T4-ALT twin) |
//! | `gdn_v_permute_f32` | `permute_gdn_v_grouped_inplace` | `ssm_out` input (tiled→grouped heads) |
//!
//! The single-vector and batched ([p × width]) forms are ONE kernel each:
//! signs index by `base % width`, so a decode-width vector (total == width)
//! and a prefill batch tile the same dispatch.
//!
//! Numeric parity: the FWHT applies the per-butterfly `1/√2` (the CPU/fork
//! unitary map), not a single end-stage `1/√n` — matching the CPU lane's
//! rounding shape and the cudarc twins' exact arithmetic (each element is
//! `(a ± b) * 1/√2`; no add-mul pair exists for the compiler to contract,
//! so the split is bit-stable on Metal).
//!
//! The butterfly body is deliberately INLINE in each FWHT kernel (three
//! mechanical copies): a `#[cube]` helper cannot take a `Shared` argument
//! (`Shared` is not `LaunchArg`), and a `macro_rules!` inside a `#[cube]`
//! body would expand after the cube pass. The three copies carry this
//! comment as the dedup marker.
//!
//! The dense escape-set GEMV (Bonsai-2's dense `ssm_alpha/beta`, 48×5120) is
//! deliberately NOT re-implemented here: `GemvCubeCL::launch_plane`
//! (`gemv_cubecl.rs`) is the same fp32 row-GEMV over dense weights, and the
//! Phase-B2 wiring consumes it directly (substrate-first — no second GEMV).

#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemv"))]
use cubecl::prelude::*;

#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemv"))]
use cubecl::server::Handle;

#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemv"))]
use riir_infer_core::deltanet::rotation::TernaryRotationConfig;

/// The FWHT kernels' shared-memory bound (one Hadamard block per cube,
/// 1024 f32 slots). `prism.hadamard` block sizes above this refuse at
/// [`RotationTablesCubeCL::build`] (Bonsai-2 is 1024).
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemv"))]
pub(crate) const MAX_FWHT_BLOCK: usize = 1024;

/// Threads (units) per cube — the repo-wide CubeCL dispatch width.
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemv"))]
const FWHT_THREADS: u32 = 256;

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

/// In-place forward rotation on a folded matmul's input:
/// `x ← (1/√n)·H_n·(S⊙x)` per block — sign FIRST, Hadamard SECOND
/// (`rotate_forward_inplace`'s order). Signs index by `base % width`, so one
/// dispatch serves both the decode vector (total == width) and a prefill
/// `[p × width]` batch.
///
/// Dispatch: one cube per Hadamard block, `CubeDim::new_1d(256)`.
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemv"))]
// clippy: the butterfly constant must be a literal for the cube macro; it is
// exactly `FRAC_1_SQRT_2` at f32 precision (the cudarc twin's value).
#[allow(clippy::approx_constant)]
#[cube(launch_unchecked)]
fn fwht_rotate_forward_f32(x: &mut [f32], signs: &[f32], params: &[f32]) {
    let total = x.len() as u32;
    let width = params[0usize] as u32;
    let hblock = params[1usize] as u32;
    let threads = 256u32;
    let base = CUBE_POS_X * hblock;
    if base >= total {
        terminate!();
    }
    let row_off = base % width;
    let tid = UNIT_POS;
    let mut smem = Shared::<[f32]>::new_slice(MAX_FWHT_BLOCK);
    let mut i = tid;
    while i < hblock {
        smem[i as usize] = x[(base + i) as usize] * signs[(row_off + i) as usize];
        i += threads;
    }
    sync_cube();
    // INLINE butterfly (dedup marker: fwht_butterfly_copy)
    let inv_sqrt2 = f32::new(0.70710678f32);
    let mut step = 2u32;
    while step <= hblock {
        let half = step / 2u32;
        let mut bi = tid;
        while bi < hblock / 2u32 {
            let blk = (bi / half) * step;
            let off = bi % half;
            let a = smem[(blk + off) as usize];
            let b = smem[(blk + off + half) as usize];
            smem[(blk + off) as usize] = (a + b) * inv_sqrt2;
            smem[(blk + off + half) as usize] = (a - b) * inv_sqrt2;
            bi += threads;
        }
        sync_cube();
        step = step * 2u32;
    }
    let mut j = tid;
    while j < hblock {
        x[(base + j) as usize] = smem[j as usize];
        j += threads;
    }
}

/// In-place inverse rotation for the token-embedding lookup result:
/// `z ← S⊙((1/√n)·H_n·z)` per block — Hadamard FIRST, sign SECOND
/// (`rotate_inverse_inplace`'s order; composing with the forward gives
/// `S·H·H·S = I` per block).
///
/// Dispatch: one cube per Hadamard block, `CubeDim::new_1d(256)`.
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemv"))]
#[allow(clippy::approx_constant)] // FRAC_1_SQRT_2 at f32 precision, literal for the cube macro
#[cube(launch_unchecked)]
fn fwht_rotate_inverse_f32(x: &mut [f32], signs: &[f32], params: &[f32]) {
    let total = x.len() as u32;
    let width = params[0usize] as u32;
    let hblock = params[1usize] as u32;
    let threads = 256u32;
    let base = CUBE_POS_X * hblock;
    if base >= total {
        terminate!();
    }
    let row_off = base % width;
    let tid = UNIT_POS;
    let mut smem = Shared::<[f32]>::new_slice(MAX_FWHT_BLOCK);
    let mut i = tid;
    while i < hblock {
        smem[i as usize] = x[(base + i) as usize];
        i += threads;
    }
    sync_cube();
    // INLINE butterfly (dedup marker: fwht_butterfly_copy)
    let inv_sqrt2 = f32::new(0.70710678f32);
    let mut step = 2u32;
    while step <= hblock {
        let half = step / 2u32;
        let mut bi = tid;
        while bi < hblock / 2u32 {
            let blk = (bi / half) * step;
            let off = bi % half;
            let a = smem[(blk + off) as usize];
            let b = smem[(blk + off + half) as usize];
            smem[(blk + off) as usize] = (a + b) * inv_sqrt2;
            smem[(blk + off + half) as usize] = (a - b) * inv_sqrt2;
            bi += threads;
        }
        sync_cube();
        step = step * 2u32;
    }
    let mut j = tid;
    while j < hblock {
        x[(base + j) as usize] = smem[j as usize] * signs[(row_off + j) as usize];
        j += threads;
    }
}

/// Copy-rotate staging twin (the cudarc T4-ALT `fwht_forward_copy_batched`):
/// reads the PRIMAL `src`, writes the rotated `dst` — one memory pass where
/// copy + in-place rotate costs two, and no `&mut` alias on the destination.
/// Mathematically identical to copy + `fwht_rotate_forward_f32` (sign first,
/// Hadamard second).
///
/// Dispatch: one cube per Hadamard block, `CubeDim::new_1d(256)`.
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemv"))]
#[allow(clippy::approx_constant)] // FRAC_1_SQRT_2 at f32 precision, literal for the cube macro
#[cube(launch_unchecked)]
fn fwht_forward_copy_f32(src: &[f32], dst: &mut [f32], signs: &[f32], params: &[f32]) {
    let total = dst.len() as u32;
    let width = params[0usize] as u32;
    let hblock = params[1usize] as u32;
    let threads = 256u32;
    let base = CUBE_POS_X * hblock;
    if base >= total {
        terminate!();
    }
    let row_off = base % width;
    let tid = UNIT_POS;
    let mut smem = Shared::<[f32]>::new_slice(MAX_FWHT_BLOCK);
    let mut i = tid;
    while i < hblock {
        smem[i as usize] = src[(base + i) as usize] * signs[(row_off + i) as usize];
        i += threads;
    }
    sync_cube();
    // INLINE butterfly (dedup marker: fwht_butterfly_copy)
    let inv_sqrt2 = f32::new(0.70710678f32);
    let mut step = 2u32;
    while step <= hblock {
        let half = step / 2u32;
        let mut bi = tid;
        while bi < hblock / 2u32 {
            let blk = (bi / half) * step;
            let off = bi % half;
            let a = smem[(blk + off) as usize];
            let b = smem[(blk + off + half) as usize];
            smem[(blk + off) as usize] = (a + b) * inv_sqrt2;
            smem[(blk + off + half) as usize] = (a - b) * inv_sqrt2;
            bi += threads;
        }
        sync_cube();
        step = step * 2u32;
    }
    let mut j = tid;
    while j < hblock {
        dst[(base + j) as usize] = smem[j as usize];
        j += threads;
    }
}

/// `gdn_v_grouped` head permute, tiled `[hd, nk, rep]` → grouped
/// `[hd, rep, nk]` (`permute_gdn_v_grouped_inplace`'s whole-head-block
/// gather, expressed per element). `tmp` is the source copy (the caller
/// stages `x` into it first — no in-place race); `x` is the destination.
/// Batched by construction: the row index is `i / v_dim`.
///
/// Dispatch: `ceil(total/256)` cubes, `CubeDim::new_1d(256)`.
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemv"))]
#[cube(launch_unchecked)]
fn gdn_v_permute_f32(tmp: &[f32], x: &mut [f32], params: &[f32]) {
    let total = x.len();
    let v_dim = params[0usize] as usize;
    let hd = params[1usize] as usize;
    let n_k = params[2usize] as usize;
    let rep = params[3usize] as usize;
    let i = ABSOLUTE_POS;
    if i >= total {
        terminate!();
    }
    let row = i / v_dim;
    let j = i % v_dim;
    let head = j / hd; // grouped head index: nk * rep + r
    let off = j % hd;
    let nk = head / rep;
    let r = head % rep;
    let src_head = r * n_k + nk; // tiled head index
    x[i] = tmp[row * v_dim + src_head * hd + off];
}

// ---------------------------------------------------------------------------
// Launcher
// ---------------------------------------------------------------------------

/// CubeCL (Metal) Hadamard-rotation launchers — the split-kernel set only
/// (the fused K1–K5 escalation is Plan 602 B5's named follow-up rung).
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemv"))]
pub struct RotationCubeCL;

#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemv"))]
impl RotationCubeCL {
    /// In-place forward rotation over `total` elements (a width vector or a
    /// `[p × width]` batch).
    ///
    /// # Safety
    ///
    /// - `x_handle`: `total` f32 elements (mutated in place)
    /// - `signs_handle`: `width` f32 elements (+1.0/-1.0)
    /// - `total` must be a multiple of `hblock`; `hblock` a power of two
    ///   `<= 1024`; `width` a multiple of `hblock`
    pub unsafe fn launch_forward<R: Runtime>(
        client: &ComputeClient<R>,
        x_handle: Handle,
        signs_handle: Handle,
        total: usize,
        width: usize,
        hblock: usize,
    ) {
        let params: &[f32] = &[width as f32, hblock as f32];
        let params_handle = crate::params_cache::params_handle(client, f32::as_bytes(params));
        let n_cubes = (total / hblock.max(1)) as u32;
        unsafe {
            fwht_rotate_forward_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(n_cubes, 1, 1),
                CubeDim::new_1d(FWHT_THREADS),
                BufferArg::from_raw_parts(x_handle, total),
                BufferArg::from_raw_parts(signs_handle, width),
                BufferArg::from_raw_parts(params_handle, 2),
            );
        }
    }

    /// In-place inverse rotation (the embedding-lookup twin).
    ///
    /// # Safety
    ///
    /// Same contract as [`Self::launch_forward`].
    pub unsafe fn launch_inverse<R: Runtime>(
        client: &ComputeClient<R>,
        x_handle: Handle,
        signs_handle: Handle,
        total: usize,
        width: usize,
        hblock: usize,
    ) {
        let params: &[f32] = &[width as f32, hblock as f32];
        let params_handle = crate::params_cache::params_handle(client, f32::as_bytes(params));
        let n_cubes = (total / hblock.max(1)) as u32;
        unsafe {
            fwht_rotate_inverse_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(n_cubes, 1, 1),
                CubeDim::new_1d(FWHT_THREADS),
                BufferArg::from_raw_parts(x_handle, total),
                BufferArg::from_raw_parts(signs_handle, width),
                BufferArg::from_raw_parts(params_handle, 2),
            );
        }
    }

    /// Copy-rotate staging: reads `src_handle` (primal), writes `dst_handle`
    /// (rotated) — `memcpy + launch_forward` in one memory pass.
    ///
    /// # Safety
    ///
    /// - `src_handle` / `dst_handle`: `total` f32 elements each
    /// - rest as [`Self::launch_forward`]
    pub unsafe fn launch_forward_copy<R: Runtime>(
        client: &ComputeClient<R>,
        src_handle: Handle,
        dst_handle: Handle,
        signs_handle: Handle,
        total: usize,
        width: usize,
        hblock: usize,
    ) {
        let params: &[f32] = &[width as f32, hblock as f32];
        let params_handle = crate::params_cache::params_handle(client, f32::as_bytes(params));
        let n_cubes = (total / hblock.max(1)) as u32;
        unsafe {
            fwht_forward_copy_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(n_cubes, 1, 1),
                CubeDim::new_1d(FWHT_THREADS),
                BufferArg::from_raw_parts(src_handle, total),
                BufferArg::from_raw_parts(dst_handle, total),
                BufferArg::from_raw_parts(signs_handle, width),
                BufferArg::from_raw_parts(params_handle, 2),
            );
        }
    }

    /// `gdn_v_grouped` tiled→grouped head permute over a `[p × v_dim]`
    /// batch (`p == 1` for decode). `tmp_handle` is the staged source copy
    /// of `x`; `x_handle` is the in-place destination.
    ///
    /// # Safety
    ///
    /// - `tmp_handle` / `x_handle`: `total = p * v_dim` f32 elements each
    /// - `v_dim == n_v_heads * head_dim`; `rep == n_v_heads / n_k_groups`
    pub unsafe fn launch_gdn_v_permute<R: Runtime>(
        client: &ComputeClient<R>,
        tmp_handle: Handle,
        x_handle: Handle,
        total: usize,
        v_dim: usize,
        hd: usize,
        n_k: usize,
        rep: usize,
    ) {
        let params: &[f32] = &[v_dim as f32, hd as f32, n_k as f32, rep as f32];
        let params_handle = crate::params_cache::params_handle(client, f32::as_bytes(params));
        let n_wg = total.div_ceil(FWHT_THREADS as usize) as u32;
        unsafe {
            gdn_v_permute_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(n_wg, 1, 1),
                CubeDim::new_1d(FWHT_THREADS),
                BufferArg::from_raw_parts(tmp_handle, total),
                BufferArg::from_raw_parts(x_handle, total),
                BufferArg::from_raw_parts(params_handle, 4),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Sign tables
// ---------------------------------------------------------------------------

/// GPU-resident rotation tables for one folded model: the width-keyed sign
/// vectors uploaded as f32 (+1.0/-1.0) plus the parsed geometry — the
/// CubeCL mirror of the cudarc lane's `RotationTables`.
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemv"))]
pub struct RotationTablesCubeCL {
    pub block_size: usize,
    pub signs: Vec<(usize, Handle)>,
    pub gdn_v_grouped: bool,
    pub gdn_v_heads: usize,
    pub gdn_k_groups: usize,
    pub inverse_embedding: bool,
}

#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemv"))]
impl RotationTablesCubeCL {
    /// Build from the parsed CPU config: upload the sign vectors as f32.
    /// Errors (not panics) on a block size this kernel set cannot serve —
    /// the loader already refused non-power-of-2 sizes, this bounds the
    /// shared-memory window.
    pub fn build<R: Runtime>(
        client: &ComputeClient<R>,
        cfg: &TernaryRotationConfig,
    ) -> Result<Self, String> {
        if cfg.block_size > MAX_FWHT_BLOCK || !cfg.block_size.is_power_of_two() {
            return Err(format!(
                "prism.hadamard block_size {} exceeds the CubeCL FWHT kernel's power-of-2 \
                 shared-memory bound ({MAX_FWHT_BLOCK})",
                cfg.block_size
            ));
        }
        let mut signs = Vec::with_capacity(cfg.signs.len());
        for (width, v) in &cfg.signs {
            let row: Vec<f32> = v.iter().map(|&s| s as f32).collect();
            debug_assert_eq!(row.len(), *width);
            signs.push((*width, client.create_from_slice(f32::as_bytes(&row))));
        }
        Ok(Self {
            block_size: cfg.block_size,
            signs,
            gdn_v_grouped: cfg.gdn_v_grouped,
            gdn_v_heads: cfg.gdn_v_heads,
            gdn_k_groups: cfg.gdn_k_groups,
            inverse_embedding: cfg.inverse_embedding,
        })
    }

    /// The sign vector for a folded input width (the loader validated the
    /// key set — a miss here is a wiring bug, not a model state).
    pub fn signs_for_width(&self, width: usize) -> &Handle {
        self.signs
            .iter()
            .find(|(w, _)| *w == width)
            .map(|(_, s)| s)
            .unwrap_or_else(|| panic!("rotation sign table missing width {width} (loader validated the set — this is a wiring bug)"))
    }
}

// ---------------------------------------------------------------------------
// Tests — every kernel against its CPU twin (Plan 602 B1's G1 shape)
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "cubecl_runtime", feature = "ternary_gemv"))]
mod tests {
    use super::*;
    use crate::cubecl_runtime::{ActiveRuntime, CubeCLContext};
    use riir_infer_core::deltanet::rotation::{
        permute_gdn_v_grouped_inplace, rotate_forward_inplace,
    };

    /// Deterministic LCG — mixed-magnitude signed values, no external deps
    /// (the cudarc twin's generator, so fixtures match across lanes).
    struct Lcg(u64);
    impl Lcg {
        fn next_f32(&mut self, lo: f32, hi: f32) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let bits = ((self.0 >> 33) as f32) / (u32::MAX >> 1) as f32;
            lo + bits * (hi - lo)
        }
    }

    fn signs_vec(n: usize, seed: u64) -> Vec<i8> {
        let mut lcg = Lcg(seed);
        (0..n).map(|_| if lcg.next_f32(-1.0, 1.0) >= 0.0 { 1 } else { -1 }).collect()
    }

    /// Forward rotation: GPU must be BIT-IDENTICAL to the CPU twin — the
    /// per-butterfly `(a±b)·1/√2` shape leaves no contraction room, and B4's
    /// Metal-vs-CPU parity gate inherits this assertion's shape.
    #[test]
    fn fwht_forward_matches_cpu_bitexact() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        for &(width, hblock) in &[(5120usize, 1024usize), (6144, 1024), (5120, 512)] {
            let mut lcg = Lcg(0x602 + width as u64);
            let x: Vec<f32> = (0..width).map(|_| lcg.next_f32(-2.0, 2.0)).collect();
            let signs = signs_vec(width, 0x5117 + width as u64);

            let mut cpu = x.clone();
            rotate_forward_inplace(&mut cpu, Some(&signs), hblock);

            let x_handle = client.create_from_slice(f32::as_bytes(&x));
            let signs_f32: Vec<f32> = signs.iter().map(|&s| s as f32).collect();
            let signs_handle = client.create_from_slice(f32::as_bytes(&signs_f32));
            unsafe {
                RotationCubeCL::launch_forward::<ActiveRuntime>(
                    &client,
                    x_handle.clone(),
                    signs_handle,
                    width,
                    width,
                    hblock,
                );
            }
            let bytes = client.read_one(x_handle).expect("read rotated");
            let gpu = f32::from_bytes(&bytes);

            assert_eq!(gpu.len(), width, "width {width}");
            for (i, (&c, &g)) in cpu.iter().zip(gpu.iter()).enumerate() {
                assert!(
                    c.to_bits() == g.to_bits(),
                    "width {width} hblock {hblock} elem {i}: cpu {c} (bits {:x}) vs gpu {g} (bits {:x})",
                    c.to_bits(),
                    g.to_bits()
                );
            }
        }
    }

    /// Forward then inverse composes to the identity per block (S·H·H·S = I).
    #[test]
    fn fwht_forward_then_inverse_is_identity() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let width = 5120usize;
        let hblock = 1024usize;
        let mut lcg = Lcg(0x1d2c);
        let x: Vec<f32> = (0..width).map(|_| lcg.next_f32(-3.0, 3.0)).collect();
        let signs = signs_vec(width, 0x7ee5);

        let x_handle = client.create_from_slice(f32::as_bytes(&x));
        let signs_f32: Vec<f32> = signs.iter().map(|&s| s as f32).collect();
        let signs_handle = client.create_from_slice(f32::as_bytes(&signs_f32));
        unsafe {
            RotationCubeCL::launch_forward::<ActiveRuntime>(
                &client,
                x_handle.clone(),
                signs_handle.clone(),
                width,
                width,
                hblock,
            );
            RotationCubeCL::launch_inverse::<ActiveRuntime>(
                &client,
                x_handle.clone(),
                signs_handle,
                width,
                width,
                hblock,
            );
        }
        let bytes = client.read_one(x_handle).expect("read roundtrip");
        let gpu = f32::from_bytes(&bytes);
        for (i, (&orig, &got)) in x.iter().zip(gpu.iter()).enumerate() {
            assert!(
                (orig - got).abs() <= 1e-4,
                "roundtrip elem {i}: {orig} vs {got}"
            );
        }
    }

    /// The batched form is the same kernel: a 3-row `[p × width]` batch must
    /// equal the CPU twin applied row-wise (row signs repeat via `% width`).
    #[test]
    fn fwht_forward_batched_matches_cpu() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let width = 5120usize;
        let hblock = 1024usize;
        let rows = 3usize;
        let total = rows * width;
        let mut lcg = Lcg(0xba7c);
        let x: Vec<f32> = (0..total).map(|_| lcg.next_f32(-1.5, 1.5)).collect();
        let signs = signs_vec(width, 0x5eed);

        let mut cpu = x.clone();
        for r in 0..rows {
            rotate_forward_inplace(&mut cpu[r * width..(r + 1) * width], Some(&signs), hblock);
        }

        let x_handle = client.create_from_slice(f32::as_bytes(&x));
        let signs_f32: Vec<f32> = signs.iter().map(|&s| s as f32).collect();
        let signs_handle = client.create_from_slice(f32::as_bytes(&signs_f32));
        unsafe {
            RotationCubeCL::launch_forward::<ActiveRuntime>(
                &client,
                x_handle.clone(),
                signs_handle,
                total,
                width,
                hblock,
            );
        }
        let bytes = client.read_one(x_handle).expect("read batched");
        let gpu = f32::from_bytes(&bytes);
        for (i, (&c, &g)) in cpu.iter().zip(gpu.iter()).enumerate() {
            assert!(
                c.to_bits() == g.to_bits(),
                "batched elem {i}: cpu {c} vs gpu {g}"
            );
        }
    }

    /// The copy-rotate staging twin equals copy + in-place forward.
    #[test]
    fn fwht_forward_copy_matches_inplace() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let width = 6144usize;
        let hblock = 1024usize;
        let mut lcg = Lcg(0xc0ffee);
        let src: Vec<f32> = (0..width).map(|_| lcg.next_f32(-2.0, 2.0)).collect();
        let signs = signs_vec(width, 0xd57);

        let mut cpu = src.clone();
        rotate_forward_inplace(&mut cpu, Some(&signs), hblock);

        let src_handle = client.create_from_slice(f32::as_bytes(&src));
        let dst_handle = client.empty(width * core::mem::size_of::<f32>());
        let signs_f32: Vec<f32> = signs.iter().map(|&s| s as f32).collect();
        let signs_handle = client.create_from_slice(f32::as_bytes(&signs_f32));
        unsafe {
            RotationCubeCL::launch_forward_copy::<ActiveRuntime>(
                &client,
                src_handle,
                dst_handle.clone(),
                signs_handle,
                width,
                width,
                hblock,
            );
        }
        let bytes = client.read_one(dst_handle).expect("read staged");
        let gpu = f32::from_bytes(&bytes);
        for (i, (&c, &g)) in cpu.iter().zip(gpu.iter()).enumerate() {
            assert!(
                c.to_bits() == g.to_bits(),
                "copy-rotate elem {i}: cpu {c} vs gpu {g}"
            );
        }
    }

    /// The gdn_v permute: GPU equals the CPU twin on a small grouped-V
    /// geometry (batched row + the whole-head gather).
    #[test]
    fn gdn_v_permute_matches_cpu() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        // n_v = 8 heads, n_k = 2 groups → rep = 4; hd = 16 → v_dim = 128.
        let n_v = 8usize;
        let n_k = 2usize;
        let hd = 16usize;
        let v_dim = n_v * hd;
        let rows = 2usize; // batched (decode is rows == 1)
        let total = rows * v_dim;
        let rep = n_v / n_k;

        let mut lcg = Lcg(0x9e7);
        let x: Vec<f32> = (0..total).map(|_| lcg.next_f32(-2.0, 2.0)).collect();

        let mut cpu = x.clone();
        let mut tmp = vec![0.0f32; total];
        for r in 0..rows {
            let (xr, tr) = (
                &mut cpu[r * v_dim..(r + 1) * v_dim],
                &mut tmp[r * v_dim..(r + 1) * v_dim],
            );
            permute_gdn_v_grouped_inplace(xr, tr, n_v, n_k);
        }

        // GPU: tmp = staged source copy, x = destination (in-place contract).
        let tmp_handle = client.create_from_slice(f32::as_bytes(&x));
        let x_handle = client.create_from_slice(f32::as_bytes(&x));
        unsafe {
            RotationCubeCL::launch_gdn_v_permute::<ActiveRuntime>(
                &client,
                tmp_handle,
                x_handle.clone(),
                total,
                v_dim,
                hd,
                n_k,
                rep,
            );
        }
        let bytes = client.read_one(x_handle).expect("read permuted");
        let gpu = f32::from_bytes(&bytes);
        for (i, (&c, &g)) in cpu.iter().zip(gpu.iter()).enumerate() {
            assert!(
                c.to_bits() == g.to_bits(),
                "permute elem {i}: cpu {c} vs gpu {g}"
            );
        }
    }

    /// Tables build: validates the block-size bound + uploads width-keyed
    /// signs; `signs_for_width` resolves each key and misses loudly.
    #[test]
    fn tables_build_and_signs_for_width() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let signs_5120 = signs_vec(5120, 0xa11);
        let signs_6144 = signs_vec(6144, 0xb22);
        let signs_17408 = signs_vec(17408, 0xc33);
        let cfg = TernaryRotationConfig {
            block_size: 1024,
            signs: vec![
                (5120usize, signs_5120.clone()),
                (6144, signs_6144.clone()),
                (17408, signs_17408.clone()),
            ],
            gdn_v_grouped: true,
            gdn_v_heads: 48,
            gdn_k_groups: 16,
            inverse_embedding: true,
        };
        let tables = RotationTablesCubeCL::build::<ActiveRuntime>(&client, &cfg)
            .expect("tables build within bound");
        assert_eq!(tables.block_size, 1024);
        assert_eq!(tables.signs.len(), 3);
        for w in [5120usize, 6144, 17408] {
            let h = tables.signs_for_width(w);
            let bytes = client.read_one(h.clone()).expect("read signs");
            let gpu = f32::from_bytes(&bytes);
            let want: Vec<f32> = match w {
                5120 => signs_5120.clone(),
                6144 => signs_6144.clone(),
                _ => signs_17408.clone(),
            }
            .iter()
            .map(|&s| s as f32)
            .collect();
            assert_eq!(gpu.len(), w);
            assert_eq!(gpu, want, "width {w} sign upload");
        }

        // The bound refuses (not panics) on an oversized block.
        let bad = TernaryRotationConfig {
            block_size: 2048,
            ..cfg
        };
        assert!(RotationTablesCubeCL::build::<ActiveRuntime>(&client, &bad).is_err());
    }
}
