//! CubeCL tiled matrix transpose — `dst[cols, rows] = src[rows, cols]^T`
//! (riir-train Issue 572).
//!
//! # Why this exists
//!
//! The Gemma-2 training lane pre-transposes every frozen base weight so the
//! backward GEMV (`gemv_batched_plane_f32`) can read each weight row
//! contiguously. That pre-transposition is a **bandwidth** decision, not an
//! algebraic one — `w_t^T` is the original matrix — and it costs a second full
//! copy of the model on the device: 9.74 GB for Gemma-2-2B at f32, which does
//! not fit beside the weights on a 24 GB discrete card.
//!
//! Transposing **on demand into a reused scratch handle** removes that copy.
//! The op has to be a GPU kernel to be worth doing: the CPU round trip the
//! setup path uses (`read_handle` → transpose → `create_from_slice`) is
//! ~600 MB of PCIe traffic per layer, i.e. ~15.6 GB per step at 26 layers.
//!
//! # Why tiled rather than one thread per element
//!
//! A transpose reads along one axis and writes along the other, so a naive
//! `dst[c * rows + r] = src[r * cols + c]` has one side strided by a full row.
//! At `cols = 2304` an f32 load touches one 32-byte sector per 4 useful bytes —
//! ~8× the traffic on the strided side. The classic fix is a shared-memory
//! tile: load a `TILE × TILE` block with coalesced reads, store it back with
//! coalesced writes, and pay the stride only inside shared memory.
//!
//! The shared tile is padded to `TILE × (TILE + 1)` so the transposed read
//! `tile[lc * (TILE + 1) + lr]` walks a stride that is coprime with the bank
//! count — the standard bank-conflict avoidance for this kernel.
//!
//! # Relationship to `gpu_transpose.rs`
//!
//! [`crate::gpu_transpose`] is a raw-`wgpu` tiled transpose written for the
//! same class of problem (LM-head weight transpose) and wired to nothing. It
//! takes `wgpu::Buffer`s; every training and inference path in this workspace
//! moves `cubecl::server::Handle`s. This module is the CubeCL-native form, so
//! callers need no `Handle` ↔ `Buffer` bridge.

#[cfg(feature = "cubecl_runtime")]
use cubecl::prelude::*;

#[cfg(feature = "cubecl_runtime")]
use cubecl::server::Handle;

/// Side of the square shared-memory tile, in elements.
///
/// 32 matches the plane/warp width on every backend this crate targets, so a
/// tile row is exactly one coalesced 128-byte transaction.
#[cfg(feature = "cubecl_runtime")]
pub const TRANSPOSE_TILE: u32 = 32;

/// Threads per workgroup. One workgroup owns one `TRANSPOSE_TILE²` tile, so
/// each thread moves `TRANSPOSE_TILE² / TRANSPOSE_WG` elements.
#[cfg(feature = "cubecl_runtime")]
pub const TRANSPOSE_WG: u32 = 256;

/// Tiled transpose kernel: `dst[c * rows + r] = src[r * cols + c]`.
///
/// # Dispatch
///
/// `CubeCount::Static(ceil(cols / TILE), ceil(rows / TILE), 1)` workgroups of
/// `TRANSPOSE_WG` threads. `CUBE_POS_X` selects the column block, `CUBE_POS_Y`
/// the row block — a **2-D** grid because a 1-D one overflows the 65535-per-
/// dimension limit on the tied embedding (`256000 × 2304` = 576 000 tiles).
///
/// # Params
///
/// `params[0] = rows as f32`, `params[1] = cols as f32`. Both ride in params
/// rather than being derived from `src.len()` / `dst.len()`: a pooled scratch
/// handle is routinely **oversized** relative to the matrix living in it, and
/// `len()` reports the whole allocation (the Issue 697/698 G3 mis-stride, one
/// kernel over). Each dimension is well under f32's 2^24 exact-integer bound
/// even though their product is not, so the f32 params carry no rounding.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn transpose_tiled_f32(src: &[f32], dst: &mut [f32], params: &[f32]) {
    let rows = params[0usize] as u32;
    let cols = params[1usize] as u32;

    let tile_dim = 32u32;
    let tile_stride = 33u32; // TILE + 1 — pad away shared-memory bank conflicts.
    let wg = 256u32;

    let row_block = CUBE_POS_Y * tile_dim;
    let col_block = CUBE_POS_X * tile_dim;

    let mut tile = Shared::<[f32]>::new_slice(1056usize); // 32 * 33

    // ── Load: consecutive units read consecutive columns → coalesced. ──
    // ALL units participate; the bounds test zero-fills rather than
    // terminating, because every unit must reach `sync_cube()`.
    let mut chunk = 0u32;
    while chunk < tile_dim * tile_dim {
        let local = UNIT_POS + chunk;
        let lr = local / tile_dim;
        let lc = local % tile_dim;
        let r = row_block + lr;
        let c = col_block + lc;

        let mut v = f32::new(0.0f32);
        if r < rows && c < cols {
            v = src[(r * cols + c) as usize];
        }
        tile[(lr * tile_stride + lc) as usize] = v;

        chunk += wg;
    }

    sync_cube();

    // ── Store: consecutive units write consecutive rows of `dst` → coalesced.
    // `lr` now indexes the OUTPUT row (a source column) and `lc` the output
    // column (a source row), so the shared read strides by `tile_stride`.
    let mut chunk_out = 0u32;
    while chunk_out < tile_dim * tile_dim {
        let local = UNIT_POS + chunk_out;
        let lr = local / tile_dim;
        let lc = local % tile_dim;
        let c = col_block + lr;
        let r = row_block + lc;

        if c < cols && r < rows {
            dst[(c * rows + r) as usize] = tile[(lc * tile_stride + lr) as usize];
        }

        chunk_out += wg;
    }
}

/// Launcher for the tiled f32 transpose.
///
/// See [`transpose_tiled_f32`] for the dispatch geometry and the reason the
/// shape rides in `params`.
#[cfg(feature = "cubecl_runtime")]
pub struct TransposeCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl TransposeCubeCL {
    /// Launch `dst[cols, rows] = src[rows, cols]^T`.
    ///
    /// `dst` is **written in full** over its first `rows * cols` elements, so a
    /// reused scratch handle needs no clearing between calls.
    ///
    /// # Safety
    ///
    /// - `src` must have at least `rows * cols` f32 elements in `[rows, cols]`
    ///   row-major layout.
    /// - `dst` must have at least `rows * cols` f32 elements; it is read back as
    ///   `[cols, rows]` row-major.
    /// - `rows` and `cols` must both be > 0.
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        src_handle: Handle,
        dst_handle: Handle,
        rows: usize,
        cols: usize,
    ) {
        let tiles_x = (cols as u32).div_ceil(TRANSPOSE_TILE).max(1);
        let tiles_y = (rows as u32).div_ceil(TRANSPOSE_TILE).max(1);

        let params: &[f32] = &[rows as f32, cols as f32];
        let params_handle = client.create_from_slice(f32::as_bytes(params));

        // SAFETY: caller guarantees both handles hold >= rows * cols f32.
        unsafe {
            transpose_tiled_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(tiles_x, tiles_y, 1),
                CubeDim::new_1d(TRANSPOSE_WG),
                BufferArg::from_raw_parts(src_handle, rows * cols),
                BufferArg::from_raw_parts(dst_handle, rows * cols),
                BufferArg::from_raw_parts(params_handle, 2),
            );
        }
    }
}
