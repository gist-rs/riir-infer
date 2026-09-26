//! CubeCL tiled matmul kernel for prefill (Plan 106 T2.4).
//!
//! Implements `C[M,P] = A[M,N] × B^T[P,N]` using CubeCL's `#[cube]` DSL.
//! B is stored `[P,N]` row-major (transposed access — no physical transpose).
//!
//! # Variants
//!
//! - **Tiled**: 16×16 shared memory tiled matmul. Each workgroup computes a 16×16
//!   output tile. Threads cooperatively load A and B tiles into shared memory,
//!   then accumulate the dot product across k-dimension tiles.
//!   Matches `matmul_transb.wgsl` algorithm exactly.
//!
//! **Shipped (Bench 416 Phase 2, 2026-07-09; CubeCL 0.11 re-verified 2026-08-13):
//! - **CMMA**: 8×8×8 cooperative matrix multiply-accumulate using Metal `simdgroup_matrix`.
//!   The `cubecl::frontend::cmma` module is public (`cmma::load` + offset-position sub-slice API
//!   via `Slice::__to_raw_parts`), and Metal3 advertises 4 wmma configs including `(f32,f32,f32)`
//!   and `(f16,f16,f32)`. Runtime detection via `MatmulSwapAb::cmma_available`. The swap_ab CMMA
//!   kernel is at `matmul_swap_ab_cubecl.rs::matmul_swap_ab_cmma_f32` — **44–61% speedup on Gemma2
//!   MLP projection shapes** (re-measured on CubeCL 0.11 after the `to_slice()`/`to_slice_mut()`
//!   API-removal fix; bit-identical correctness). This standard kernel does NOT use CMMA
//!   (it uses scalar 16×16 tiling); CMMA is only in the swap_ab variant.
//!
//! # Dispatch
//!
//! | Variant | CubeDim       | CubeCount                    | Output block |
//! |---------|---------------|------------------------------|-------------|
//! | Tiled   | `new_1d(256)` | `ceil(M/16), ceil(P/16), 1` | 16×16       |
//!
//! 2D dispatch (X = M-tiles, Y = P-tiles) keeps each axis under wgpu's
//! 65535-workgroup-per-dimension cap, allowing up to `65535²` workgroups
//! (Issue 309 — was a flat 1D `(wg_x*wg_y, 1, 1)` that capped at 65535 total).
//!
//! # CODA Hook (Track 3)
//!
//! The accumulator stays in registers before global write. Track 3 epilogue
//! visitors can consume the accumulator before writeback, enabling fused
//! matmul + activation + residual in a single dispatch.
//!
//! # TransB Convention
//!
//! B is stored `[P, N]` row-major. The multiplication computes:
//! ```text
//! C[i,j] = Σ_k A[i,k] × B[j,k]   (dot product of A row i and B row j)
//! ```
//! This avoids physically transposing B — we simply access B by row.

#[cfg(feature = "cubecl_runtime")]
use cubecl::prelude::*;

#[cfg(feature = "cubecl_runtime")]
use cubecl::server::Handle;

#[cfg(feature = "cubecl_runtime")]
use crate::cubecl_runtime::{debug_assert_binding_at_least, f32_exact};

// ---------------------------------------------------------------------------
// Tiled matmul kernel — matches matmul_transb.wgsl
// ---------------------------------------------------------------------------

/// CubeCL tiled matmul with transposed B: `C[M,P] = A[M,N] × B^T[P,N]`.
///
/// B is stored `[P,N]` row-major. The kernel accesses `B[j,k]` directly,
/// equivalent to multiplying by `B^T` without physical transposition.
///
/// ## Dimensions
///
/// Derived from array lengths at runtime (no scalar parameters — avoids
/// CubeCL v0.10 macro expansion bug with `launch_unchecked`):
///
/// ```text
/// a.len() = M×N,  b.len() = P×N,  out.len() = M×P
/// N² = a.len() × b.len() / out.len()
/// N = isqrt(N²),  M = a.len() / N,  P = out.len() / M
/// ```
///
/// ## Algorithm
///
/// Each workgroup (256 threads, 1D layout via `CubeDim::new_1d(256)`)
/// computes a 16×16 output tile. The 2D workgroup position comes directly
/// from the 2D dispatch grid (`CUBE_POS_X` = M-tile, `CUBE_POS_Y` = P-tile):
///
/// 1. Workgroup tile position = `(CUBE_POS_X, CUBE_POS_Y)` (native 2D dispatch).
/// 2. Compute 2D local position from `UNIT_POS / 16` and `UNIT_POS % 16`.
/// 3. All 256 threads cooperatively load tiles of A and B into shared memory:
///    - `tile_a[local_row * 16 + local_col] ← A[row, k_base + local_col]`
///    - `tile_b[local_col * 16 + local_row] ← B[col, k_base + local_row]`
///      (swapped indices for transB convention)
/// 4. `sync_cube()` barrier.
/// 5. Each in-bounds thread accumulates one output element over the k-tile.
/// 6. `sync_cube()` barrier.
/// 7. Repeat for `ceil(N/16)` k-tiles.
/// 8. Write output for valid positions only.
///
/// All threads participate in tile loading (even out-of-bounds threads)
/// to avoid shared-memory gaps.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn matmul_tiled_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    let tile = 16u32;
    // cube_size (256) is implicit in CubeDim::new_1d(256) and UNIT_POS's range.

    // Derive dimensions from array lengths:
    //   a.len()=M*N, b.len()=P*N, out.len()=M*P
    //   N² = a_len * b_len / out_len (multiply first to avoid truncation)
    let a_len = a.len() as u32;
    let b_len = b.len() as u32;
    let out_len = out.len() as u32;

    // Newton's integer sqrt for N — convergence-based, no counter.
    // n > n_sq / n ⟺ n² > n_sq (avoids u32 overflow in the loop test).
    //
    // u64 intermediate for the product: `a_len * b_len` overflows u32 for
    // Gemma2-class shapes (e.g. 18432 * 21233664 ≈ 3.91e11 > u32::MAX). The
    // quotient N² always fits in u32 for any realistic N (N ≤ 65535 → N² <
    // 4.29e9), so the cast back to u32 is safe. See Issue 376.
    let n_sq = ((a_len as u64) * (b_len as u64) / (out_len as u64)) as u32;
    let mut n = n_sq;
    if n_sq > 1u32 {
        n = n_sq.div_ceil(2u32);
        while n > n_sq / n {
            n = (n + n_sq / n) / 2u32;
        }
    }

    let m = a_len / n;
    let p = out_len / m;

    // 2D dispatch (Issue 309): the dispatch grid is `CubeCount::Static(wg_x, wg_y, 1)`
    // where `wg_x = ceil(M/16)` and `wg_y = ceil(P/16)`. The cube (workgroup)
    // position is therefore natively 2D — `CUBE_POS_X` is the M-tile index and
    // `CUBE_POS_Y` is the P-tile index. This replaces the old flat-`wg_id`
    // derivation (`wg_id / num_tiles_p`, `wg_id % num_tiles_p`) which was only
    // valid under the old 1D dispatch and broke for N > 4096 (65535 cap).
    let wg_row = CUBE_POS_X;
    let wg_col = CUBE_POS_Y;
    let local_row = UNIT_POS / tile;
    let local_col = UNIT_POS % tile;
    let row = wg_row * tile + local_row;
    let col = wg_col * tile + local_col;

    // Accumulator for this thread's output element.
    let mut sum = f32::new(0.0f32);

    // Shared memory for 16×16 tiles of A and B.
    let mut tile_a = Shared::<[f32]>::new_slice(256usize);
    let mut tile_b = Shared::<[f32]>::new_slice(256usize);

    let num_k_tiles = n.div_ceil(tile);

    let mut k_tile = 0u32;
    while k_tile < num_k_tiles {
        let k_base = k_tile * tile;

        // Cooperative load: ALL 256 threads must participate.
        // tile_a[local_row * 16 + local_col] ← A[row, k_base + local_col]
        // tile_b[local_col * 16 + local_row] ← B[col, k_base + local_row]
        //   (transB: B stored [P,N] row-major, swapped indices for dot product)
        let smem_a = (local_row * tile + local_col) as usize;
        let smem_b = (local_col * tile + local_row) as usize;

        let a_k = k_base + local_col;
        if row < m && a_k < n {
            tile_a[smem_a] = a[(row * n + a_k) as usize];
        } else {
            tile_a[smem_a] = f32::new(0.0f32);
        }

        let b_k = k_base + local_row;
        if col < p && b_k < n {
            tile_b[smem_b] = b[(col * n + b_k) as usize];
        } else {
            tile_b[smem_b] = f32::new(0.0f32);
        }

        sync_cube();

        // Only in-bounds threads accumulate.
        if row < m && col < p {
            let mut k = 0u32;
            while k < tile {
                sum += tile_a[(local_row * tile + k) as usize]
                    * tile_b[(local_col * tile + k) as usize];
                k += 1u32;
            }
        }

        sync_cube();
        k_tile += 1u32;
    }

    // Write output for valid positions only.
    if row < m && col < p {
        out[(row * p + col) as usize] = sum;
    }
}

// ---------------------------------------------------------------------------
// The encoder-lane family (plan 611 S2): offset + head-batched tiled
// kernels over WHOLE parent binds.
// ---------------------------------------------------------------------------
//
// The laya `Backend` trait's matmul ops carry ELEMENT OFFSETS into parent
// buffers and bind WHOLE parents — the S1b finding (wgpu's 32-byte
// `min_storage_buffer_offset_alignment` vs the forward's element-arbitrary
// offsets) means the slab cannot ride a byte-offset handle view; the
// offsets ride the params buffer instead, exactly like the S1b elementwise
// family. Dims are therefore EXPLICIT params too: the derivation-from-
// lengths trick above (`matmul_tiled_f32`) only works when the binds are
// the exact logical slabs, which whole-parent binds are not.
//
// Two kernels cover the trait's four offset ops:
//
// - `matmul_batched_transb_off_f32` — `out[h·m·m + i·m + j] = Σ_t
//   a[a_off + h·m·k + i·k + t] · b[b_off + h·m·k + j·k + t]` — the score
//   shape (`matmul_kt` at heads=1, `matmul_kt_heads` at heads=N; B is the
//   [m×k] KEY matrix indexed by its row).
// - `matmul_batched_rr_off_f32` — `out[h·m·n + i·n + j] = Σ_t
//   a[a_off + h·m·k + i·k + t] · b[b_off + h·k·n + t·n + j]` — the plain
//   row-major × row-major shape (`matmul` at heads=1, `matmul_heads` at
//   heads=N).
//
// The head batch rides the DISPATCH z axis (`CUBE_POS_Z`) — one launch for
// all heads, no host loop, no per-head cache lookups. The whole-batch
// destination is ONE `client.empty` slot on the backend side (the packed
// scores parent never fragments).
//
// Tile/dispatch geometry is the shipped transB kernel's: 16×16 smem tiles,
// 256-thread cubes, `CubeCount::Static(ceil(m/16), ceil(p/16), heads)`.

/// Batched transB tiled matmul at element offsets (plan 611 S2).
///
/// One cube per 16×16 output tile per head. `params` = `[m, k, a_off,
/// b_off, out_off]` (f32-encoded usize). Per-head slabs: `a`/`b` are
/// `[heads × m × k]` at their base offsets, `out` is `[heads × m × m]`.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn matmul_batched_transb_off_f32(a: &[f32], b: &[f32], out: &mut [f32], params: &[f32]) {
    let m = params[0usize] as u32;
    let k = params[1usize] as u32;
    let a_off = params[2usize] as u32;
    let b_off = params[3usize] as u32;
    let out_off = params[4usize] as u32;

    let tile = 16u32;
    let h = CUBE_POS_Z;
    let wg_row = CUBE_POS_X;
    let wg_col = CUBE_POS_Y;
    let local_row = UNIT_POS / tile;
    let local_col = UNIT_POS % tile;
    let row = wg_row * tile + local_row;
    let col = wg_col * tile + local_col;

    // Per-head slab bases (q/k share the [heads, m, k] shape; out is
    // [heads, m, m]).
    let a_base = a_off + h * m * k;
    let b_base = b_off + h * m * k;
    let o_base = out_off + h * m * m;

    let mut sum = f32::new(0.0f32);
    let mut tile_a = Shared::<[f32]>::new_slice(256usize);
    let mut tile_b = Shared::<[f32]>::new_slice(256usize);

    let num_k_tiles = k.div_ceil(tile);
    let mut k_tile = 0u32;
    while k_tile < num_k_tiles {
        let k_base = k_tile * tile;
        let smem_a = (local_row * tile + local_col) as usize;
        let smem_b = (local_col * tile + local_row) as usize;

        // A row-major [m, k]: tile_a[slot] ← A[row, k_base + local_col].
        let a_k = k_base + local_col;
        if row < m && a_k < k {
            tile_a[smem_a] = a[(a_base + row * k + a_k) as usize];
        } else {
            tile_a[smem_a] = f32::new(0.0f32);
        }

        // B row-major [m, k] (the key matrix): tile_b[slot] ← B[col,
        // k_base + local_row] — the transB dot-product pairing.
        let b_k = k_base + local_row;
        if col < m && b_k < k {
            tile_b[smem_b] = b[(b_base + col * k + b_k) as usize];
        } else {
            tile_b[smem_b] = f32::new(0.0f32);
        }

        sync_cube();

        if row < m && col < m {
            let mut t = 0u32;
            while t < tile {
                sum += tile_a[(local_row * tile + t) as usize]
                    * tile_b[(local_col * tile + t) as usize];
                t += 1u32;
            }
        }

        sync_cube();
        k_tile += 1u32;
    }

    if row < m && col < m {
        out[(o_base + row * m + col) as usize] = sum;
    }
}

/// Batched row-major × row-major tiled matmul at element offsets (plan 611
/// S2).
///
/// One cube per 16×16 output tile per head. `params` = `[m, k, n, a_off,
/// b_off, out_off]`. Per-head slabs: `a` is `[heads × m × k]`, `b` is
/// `[heads × k × n]`, `out` is `[heads × m × n]`.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn matmul_batched_rr_off_f32(a: &[f32], b: &[f32], out: &mut [f32], params: &[f32]) {
    let m = params[0usize] as u32;
    let k = params[1usize] as u32;
    let n = params[2usize] as u32;
    let a_off = params[3usize] as u32;
    let b_off = params[4usize] as u32;
    let out_off = params[5usize] as u32;

    let tile = 16u32;
    let h = CUBE_POS_Z;
    let wg_row = CUBE_POS_X;
    let wg_col = CUBE_POS_Y;
    let local_row = UNIT_POS / tile;
    let local_col = UNIT_POS % tile;
    let row = wg_row * tile + local_row;
    let col = wg_col * tile + local_col;

    let a_base = a_off + h * m * k;
    let b_base = b_off + h * k * n;
    let o_base = out_off + h * m * n;

    let mut sum = f32::new(0.0f32);
    let mut tile_a = Shared::<[f32]>::new_slice(256usize);
    let mut tile_b = Shared::<[f32]>::new_slice(256usize);

    let num_k_tiles = k.div_ceil(tile);
    let mut k_tile = 0u32;
    while k_tile < num_k_tiles {
        let k_base = k_tile * tile;
        let smem_a = (local_row * tile + local_col) as usize;
        let smem_b = (local_col * tile + local_row) as usize;

        // A row-major [m, k] — same load as the transB kernel.
        let a_k = k_base + local_col;
        if row < m && a_k < k {
            tile_a[smem_a] = a[(a_base + row * k + a_k) as usize];
        } else {
            tile_a[smem_a] = f32::new(0.0f32);
        }

        // B row-major [k, n]: tile_b[slot] ← B[k_base + local_row, col].
        // The smem slot [local_col·16 + local_row] then reads as
        // B[k_base + t, col] in the accumulate loop — the transpose-for-free
        // bank layout, no Bᵀ materialization anywhere.
        let b_t = k_base + local_row;
        if col < n && b_t < k {
            tile_b[smem_b] = b[(b_base + b_t * n + col) as usize];
        } else {
            tile_b[smem_b] = f32::new(0.0f32);
        }

        sync_cube();

        if row < m && col < n {
            let mut t = 0u32;
            while t < tile {
                sum += tile_a[(local_row * tile + t) as usize]
                    * tile_b[(local_col * tile + t) as usize];
                t += 1u32;
            }
        }

        sync_cube();
        k_tile += 1u32;
    }

    if row < m && col < n {
        out[(o_base + row * n + col) as usize] = sum;
    }
}

// ---------------------------------------------------------------------------
// Public launcher
// ---------------------------------------------------------------------------

/// CubeCL tiled matmul launcher.
///
/// Provides the public API for launching CubeCL matmul kernels.
/// Currently supports the shared-memory tiled variant (16×16 tiles).
///
/// # Example
///
/// ```rust,ignore
/// let ctx = CubeCLContext::new()?;
/// let client = ctx.client();
///
/// // C[128, 2048] = A[128, 2304] × B^T[2048, 2304]
/// let (m, n, p) = (128, 2304, 2048);
/// MatmulCubeCL::launch::<ActiveRuntime>(
///     &client, a_handle, b_handle, out_handle, m, n, p,
/// );
/// ```
#[cfg(feature = "cubecl_runtime")]
pub struct MatmulCubeCL;

#[cfg(feature = "cubecl_runtime")]
#[allow(dead_code)] // Used in T2.6 forward pass wiring
impl MatmulCubeCL {
    /// Launch tiled matmul kernel: `C[M,P] = A[M,N] × B^T[P,N]`.
    ///
    /// B is stored `[P,N]` row-major. The kernel accesses B rows directly
    /// (equivalent to multiplying by B^T).
    ///
    /// Uses 16×16 shared memory tiling with 256-thread workgroups.
    /// Dispatch: 2D `CubeCount::Static(ceil(M/16), ceil(P/16), 1)` — each axis
    /// independently capped at 65535 workgroups (Issue 309; was a flat 1D
    /// dispatch that capped total workgroups at 65535, crashing for N ≥ 4096).
    ///
    /// # Safety
    ///
    /// Buffer handles must have correct sizes:
    /// - `a_handle`: M×N f32 elements
    /// - `b_handle`: P×N f32 elements
    /// - `out_handle`: M×P f32 elements
    pub fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        a_handle: Handle,
        b_handle: Handle,
        out_handle: Handle,
        m: usize,
        n: usize,
        p: usize,
    ) {
        unsafe { Self::launch_tiled::<R>(client, a_handle, b_handle, out_handle, m, n, p) };
    }

    /// Launch 16×16 shared-memory tiled matmul kernel.
    ///
    /// Each workgroup computes a 16×16 tile of the output.
    /// 256 threads per workgroup. k-dimension accumulated in tiles.
    ///
    /// # Safety
    ///
    /// Buffer handles must have correct sizes:
    /// - `a_handle`: M×N f32 elements
    /// - `b_handle`: P×N f32 elements
    /// - `out_handle`: M×P f32 elements
    pub unsafe fn launch_tiled<R: Runtime>(
        client: &ComputeClient<R>,
        a_handle: Handle,
        b_handle: Handle,
        out_handle: Handle,
        m: usize,
        n: usize,
        p: usize,
    ) {
        let wg_x = (m as u32).div_ceil(16).max(1);
        let wg_y = (p as u32).div_ceil(16).max(1);

        // SAFETY: Caller guarantees correct buffer sizes.
        //
        // 2D dispatch (Issue 309): each axis capped at 65535 independently,
        // allowing up to 65535² ≈ 4.3B workgroups. The kernel derives its 2D
        // tile position from `CUBE_POS_X` (M-tile) and `CUBE_POS_Y` (P-tile)
        // directly, so it is correct regardless of the flat linearization the
        // backend chooses for the dispatch grid.
        unsafe {
            matmul_tiled_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(wg_x, wg_y, 1),
                CubeDim::new_1d(256),
                BufferArg::from_raw_parts(a_handle, m * n),
                BufferArg::from_raw_parts(b_handle, p * n),
                BufferArg::from_raw_parts(out_handle, m * p),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Encoder-lane launchers (plan 611 S2)
// ---------------------------------------------------------------------------

/// Launcher for [`matmul_batched_transb_off_f32`] — the score shape over
/// whole parent binds: `heads` slabs of `out[m×m] = a[m×k] @ b[m×k]ᵀ` at
/// element base offsets (plan 611 S2).
///
/// `matmul_kt` calls it with heads=1; `matmul_kt_heads` with heads=N and
/// zero offsets. Dispatch `Static(ceil(m/16), ceil(m/16), heads)`.
#[cfg(feature = "cubecl_runtime")]
pub struct MatmulTransbOffCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl MatmulTransbOffCubeCL {
    /// Launch the batched transB matmul over whole parent binds.
    ///
    /// # Safety
    ///
    /// Handles must back at least the declared lengths, with
    /// `a_off + heads·m·k ≤ a_len`, `b_off + heads·m·k ≤ b_len`,
    /// `out_off + heads·m·m ≤ out_len`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        a_handle: Handle,
        a_len: usize,
        b_handle: Handle,
        b_len: usize,
        out_handle: Handle,
        out_len: usize,
        heads: usize,
        m: usize,
        k: usize,
        a_off: usize,
        b_off: usize,
        out_off: usize,
    ) {
        assert!(
            heads >= 1 && m >= 1 && k >= 1,
            "transb-off: degenerate shape"
        );
        assert!(
            a_off + heads * m * k <= a_len,
            "transb-off: a extent ({a_off} + {} > {a_len})",
            heads * m * k
        );
        assert!(
            b_off + heads * m * k <= b_len,
            "transb-off: b extent ({b_off} + {} > {b_len})",
            heads * m * k
        );
        assert!(
            out_off + heads * m * m <= out_len,
            "transb-off: out extent ({out_off} + {} > {out_len})",
            heads * m * m
        );
        debug_assert_binding_at_least(&a_handle, a_len, "TransbOff::a");
        debug_assert_binding_at_least(&b_handle, b_len, "TransbOff::b");
        debug_assert_binding_at_least(&out_handle, out_len, "TransbOff::out");
        let params: &[f32] = &[
            f32_exact(m),
            f32_exact(k),
            f32_exact(a_off),
            f32_exact(b_off),
            f32_exact(out_off),
        ];
        let params_handle = crate::params_cache::params_handle(client, f32::as_bytes(params));
        let wg = (m as u32).div_ceil(16).max(1);
        // SAFETY: extents asserted above; the kernel bounds-checks row/col.
        unsafe {
            matmul_batched_transb_off_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(wg, wg, heads as u32),
                CubeDim::new_1d(256),
                BufferArg::from_raw_parts(a_handle, a_len),
                BufferArg::from_raw_parts(b_handle, b_len),
                BufferArg::from_raw_parts(out_handle, out_len),
                BufferArg::from_raw_parts(params_handle, params.len()),
            );
        }
    }
}

/// Launcher for [`matmul_batched_rr_off_f32`] — the plain row-major ×
/// row-major shape over whole parent binds: `heads` slabs of
/// `out[m×n] = a[m×k] @ b[k×n]` at element base offsets (plan 611 S2).
///
/// `matmul` calls it with heads=1; `matmul_heads` with heads=N and zero
/// offsets. Dispatch `Static(ceil(m/16), ceil(n/16), heads)`.
#[cfg(feature = "cubecl_runtime")]
pub struct MatmulRrOffCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl MatmulRrOffCubeCL {
    /// Launch the batched row-major matmul over whole parent binds.
    ///
    /// # Safety
    ///
    /// Handles must back at least the declared lengths, with
    /// `a_off + heads·m·k ≤ a_len`, `b_off + heads·k·n ≤ b_len`,
    /// `out_off + heads·m·n ≤ out_len`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        a_handle: Handle,
        a_len: usize,
        b_handle: Handle,
        b_len: usize,
        out_handle: Handle,
        out_len: usize,
        heads: usize,
        m: usize,
        k: usize,
        n: usize,
        a_off: usize,
        b_off: usize,
        out_off: usize,
    ) {
        assert!(
            heads >= 1 && m >= 1 && k >= 1 && n >= 1,
            "rr-off: degenerate shape"
        );
        assert!(
            a_off + heads * m * k <= a_len,
            "rr-off: a extent ({a_off} + {} > {a_len})",
            heads * m * k
        );
        assert!(
            b_off + heads * k * n <= b_len,
            "rr-off: b extent ({b_off} + {} > {b_len})",
            heads * k * n
        );
        assert!(
            out_off + heads * m * n <= out_len,
            "rr-off: out extent ({out_off} + {} > {out_len})",
            heads * m * n
        );
        debug_assert_binding_at_least(&a_handle, a_len, "RrOff::a");
        debug_assert_binding_at_least(&b_handle, b_len, "RrOff::b");
        debug_assert_binding_at_least(&out_handle, out_len, "RrOff::out");
        let params: &[f32] = &[
            f32_exact(m),
            f32_exact(k),
            f32_exact(n),
            f32_exact(a_off),
            f32_exact(b_off),
            f32_exact(out_off),
        ];
        let params_handle = crate::params_cache::params_handle(client, f32::as_bytes(params));
        let wg_x = (m as u32).div_ceil(16).max(1);
        let wg_y = (n as u32).div_ceil(16).max(1);
        // SAFETY: extents asserted above; the kernel bounds-checks row/col.
        unsafe {
            matmul_batched_rr_off_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(wg_x, wg_y, heads as u32),
                CubeDim::new_1d(256),
                BufferArg::from_raw_parts(a_handle, a_len),
                BufferArg::from_raw_parts(b_handle, b_len),
                BufferArg::from_raw_parts(out_handle, out_len),
                BufferArg::from_raw_parts(params_handle, params.len()),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "cubecl_runtime"))]
mod tests {
    use crate::cubecl_runtime::ActiveRuntime;

    use crate::cubecl_runtime::CubeCLContext;

    use super::*;

    /// Reference implementation: compute C = A × B^T on CPU.
    ///
    /// B is stored [P, N] row-major. C[i,j] = Σ_k A[i,k] × B[j,k].
    fn matmul_transb_cpu(a: &[f32], b: &[f32], m: usize, n: usize, p: usize) -> Vec<f32> {
        let mut c = vec![0.0f32; m * p];
        for i in 0..m {
            for j in 0..p {
                let mut sum = 0.0f32;
                for k in 0..n {
                    sum += a[i * n + k] * b[j * n + k];
                }
                c[i * p + j] = sum;
            }
        }
        c
    }

    /// Verify tiled matmul against CPU reference.
    fn verify_matmul(
        client: &ComputeClient<ActiveRuntime>,
        a: &[f32],
        b: &[f32],
        m: usize,
        n: usize,
        p: usize,
        tolerance: f32,
    ) {
        let expected = matmul_transb_cpu(a, b, m, n, p);

        let a_handle = client.create_from_slice(f32::as_bytes(a));
        let b_handle = client.create_from_slice(f32::as_bytes(b));
        let out_handle = client.empty(m * p * core::mem::size_of::<f32>());

        unsafe {
            MatmulCubeCL::launch_tiled::<ActiveRuntime>(
                client,
                a_handle,
                b_handle,
                out_handle.clone(),
                m,
                n,
                p,
            );
        }

        let bytes = client.read_one(out_handle).expect("should read output");
        let output = f32::from_bytes(&bytes);

        assert_eq!(output.len(), m * p, "output length mismatch");
        let mut max_err = 0.0f32;
        for (i, (&exp, &got)) in expected.iter().zip(output.iter()).enumerate() {
            let err = (exp - got).abs();
            if err > max_err {
                max_err = err;
            }
            assert!(
                err < tolerance,
                "element {i}: expected {exp}, got {got}, err = {err}"
            );
        }
        println!("matmul ({m}×{n}) × ({p}×{n})^T: max_error = {max_err}");
    }

    /// Verify 4×4 identity × identity^T → identity.
    #[test]
    fn test_matmul_tiled_identity() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let m = 4;
        let n = 4;
        let p = 4;

        let mut a = vec![0.0f32; m * n];
        let mut b = vec![0.0f32; p * n];
        for i in 0..4 {
            a[i * n + i] = 1.0;
            b[i * n + i] = 1.0;
        }

        verify_matmul(&client, &a, &b, m, n, p, 1e-5);
    }

    /// Verify 3×4 × 4^T×3 general matrix (A × A^T).
    #[test]
    fn test_matmul_tiled_general() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let m = 3;
        let n = 4;
        let p = 3;

        let a: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0, //
            5.0, 6.0, 7.0, 8.0, //
            9.0, 10.0, 11.0, 12.0,
        ];
        let b = a.clone();
        // C = A × A^T:
        // [0][0] = 1+4+9+16 = 30
        // [0][1] = 5+12+21+32 = 70
        // [0][2] = 9+20+33+48 = 110
        // [1][1] = 25+36+49+64 = 174

        verify_matmul(&client, &a, &b, m, n, p, 1e-3);
    }

    /// Verify 64×64 diagonal × identity (fills multiple 16×16 tiles exactly).
    #[test]
    fn test_matmul_tiled_large() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let m = 64;
        let n = 64;
        let p = 64;

        let mut a = vec![0.0f32; m * n];
        for i in 0..m.min(n) {
            a[i * n + i] = (i + 1) as f32;
        }
        let mut b = vec![0.0f32; p * n];
        for i in 0..p.min(n) {
            b[i * n + i] = 1.0;
        }

        verify_matmul(&client, &a, &b, m, n, p, 1e-3);
    }

    /// Verify 20×48 × 12^T×48 (partial edge tiles in M, N, and P).
    #[test]
    fn test_matmul_tiled_non_square() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let m = 20; // 16 + 4 (partial tile in M)
        let n = 48; // 3 full k-tiles
        let p = 12; // < 16 (partial tile in P)

        let a: Vec<f32> = (0..m * n).map(|i| (i as f32 * 0.01).sin()).collect();
        let b: Vec<f32> = (0..p * n).map(|i| (i as f32 * 0.02).cos()).collect();

        verify_matmul(&client, &a, &b, m, n, p, 1e-2);
    }

    /// Verify with Gemma 2 Q-projection-like dimensions (truncated for speed).
    #[test]
    fn test_matmul_tiled_gemma2_dims() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        // Q projection: A=[seq_len, n_embd], B=[q_dim, n_embd]
        // C = A × B^T → [seq_len, q_dim]
        let seq_len = 8;
        let n_embd = 64; // Truncated (full: 2304)
        let q_dim = 32; // Truncated (full: 2048)

        let a: Vec<f32> = (0..seq_len * n_embd)
            .map(|i| (i as f32 * 0.01).sin())
            .collect();
        let b: Vec<f32> = (0..q_dim * n_embd)
            .map(|i| (i as f32 * 0.02).cos())
            .collect();

        verify_matmul(&client, &a, &b, seq_len, n_embd, q_dim, 1e-2);
    }

    /// Regression for Issue 376: `a_len * b_len` must not overflow u32.
    ///
    /// Shape M=8, N=820, P=820 gives `a_len=6560`, `b_len=672400`, product =
    /// 4,410,944,000 > u32::MAX (4,294,967,296). The pre-fix kernel computed
    /// `n_sq = a_len * b_len / out_len` in u32, which wrapped to a garbage N
    /// (isqrt(115976704/6560) = 132 instead of 820), producing a totally
    /// wrong output. With the u64-intermediate fix, N=820 is derived correctly.
    ///
    /// This test runs the full kernel and checks against the CPU reference —
    /// if the overflow regresses, max_err will be O(output magnitude).
    #[test]
    fn test_matmul_tiled_u32_overflow_shape() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let m = 8;
        let n = 820;
        let p = 820;

        // Sanity: this shape DOES overflow u32 in the intermediate product.
        let a_len = (m * n) as u64;
        let b_len = (p * n) as u64;
        assert!(
            a_len * b_len > u32::MAX as u64,
            "test premise: a_len*b_len must overflow u32 (got {})",
            a_len * b_len
        );

        let a: Vec<f32> = (0..m * n).map(|i| (i as f32 * 0.001).sin()).collect();
        let b: Vec<f32> = (0..p * n).map(|i| (i as f32 * 0.0017).cos()).collect();

        // Tolerance scales with N (rounding accumulates over the contraction).
        verify_matmul(&client, &a, &b, m, n, p, 1e-1);
    }

    /// Verify 1×8 × 1^T×8 → single output element (edge case).
    #[test]
    fn test_matmul_tiled_single_element() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let m = 1;
        let n = 8;
        let p = 1;

        let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        // C[0,0] = 1+2+3+4+5+6+7+8 = 36

        verify_matmul(&client, &a, &b, m, n, p, 1e-5);
    }

    /// Regression for Issue 309: dispatch with more than 65535 total workgroups.
    ///
    /// Shape M=65536 (4096 M-tiles) × P=256 (16 P-tiles) gives 65536 total
    /// workgroups — one over the pre-Issue-309 1D cap of 65535/dim.
    ///
    /// `#[ignore]` because >65535 workgroups requires >16.7M output elements
    /// (each 16×16 tile yields 256 outputs), so the 64 MiB readback is ~3 min in
    /// debug. The 6 fast matmul tests above already validate correctness of the
    /// `CUBE_POS_X`/`CUBE_POS_Y` 2D mapping; this test specifically guards the
    /// dispatch-cap regression and is meant to run on-demand in release:
    ///
    /// ```sh
    /// cargo test -p riir-gpu --features cubecl_runtime --release \
    ///   matmul_tiled_exceeds_old_dispatch_cap -- --ignored --nocapture
    /// ```
    ///
    /// N=1 is chosen to keep the test's readback size manageable. With the
    /// Issue 376 fix, the kernel's `n_sq` derivation uses a u64 intermediate,
    /// so larger N would no longer overflow — but N=1 still keeps the output
    /// buffer small enough for the sparse-probe correctness check below.
    #[test]
    #[ignore = "slow: 64 MiB readback in debug; run with --release --ignored"]
    fn test_matmul_tiled_exceeds_old_dispatch_cap() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        // M=65536 (4096 M-tiles), P=256 (16 P-tiles) → 65536 total workgroups,
        // one over the pre-Issue-309 1D cap. N=1 keeps the readback small;
        // the Issue 376 u64 fix means larger N would also work.
        let m = 65536;
        let n = 1;
        let p = 256;
        let wg_total = (m.div_ceil(16)) * (p.div_ceil(16));
        assert_eq!(wg_total, 65536, "test premise: >65535 workgroups");

        // A[i,0] = i+1, B[j,0] = 1 → C[i,j] = (i+1) for every j.
        let a: Vec<f32> = (0..m).map(|i| (i + 1) as f32).collect();
        let b: Vec<f32> = vec![1.0f32; p];

        let a_handle = client.create_from_slice(f32::as_bytes(&a));
        let b_handle = client.create_from_slice(f32::as_bytes(&b));
        let out_handle = client.empty(m * p * core::mem::size_of::<f32>());

        // This launch would panic under the old 1D dispatch
        // (`wgpu error: dispatch group size dimension must be ≤ 65535`).
        unsafe {
            MatmulCubeCL::launch_tiled::<ActiveRuntime>(
                &client,
                a_handle,
                b_handle,
                out_handle.clone(),
                m,
                n,
                p,
            );
        }

        let bytes = client.read_one(out_handle).expect("should read output");
        let output = f32::from_bytes(&bytes);
        assert_eq!(output.len(), m * p, "output length mismatch");

        // Sparse correctness probe (64 points) — the full 16M-element scan is
        // O(seconds) even in release; the bench does exhaustive checking.
        let mut max_err = 0.0f32;
        for k in 0..64u64 {
            let seed = k.wrapping_mul(0x9E3779B97F4A7C15);
            let i = ((seed >> 16) % m as u64) as usize;
            let j = ((seed >> 32) % p as u64) as usize;
            let expected = (i + 1) as f32; // C[i,j] = A[i,0]*B[j,0] = (i+1)*1
            let got = output[i * p + j];
            let err = (got - expected).abs();
            if err > max_err {
                max_err = err;
            }
        }
        assert!(max_err < 1e-2, "sparse probe mismatch: max_err = {max_err}");
        println!(
            "matmul ({m}×{n}) × ({p}×{n})^T [{wg_total} wgs > 65535 cap]: sparse-probe max_err = {max_err}"
        );
    }

    // -------------------------------------------------------------------
    // Encoder-lane family (plan 611 S2): offset + head-batched kernels.
    // -------------------------------------------------------------------

    /// Deterministic [-1, 1) noise (the smoke test's LCG, same constant).
    fn noise(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((s >> 8) as f32) / 8_388_608.0 - 1.0
            })
            .collect()
    }

    /// `out[h][m×n] = a[h][m×k] @ b[h][k×n]` (all row-major) on the host —
    /// per-head slabs at base offsets (heads=1 degenerates to the plain
    /// matmul).
    fn matmul_rr_cpu(
        a: &[f32],
        a_off: usize,
        b: &[f32],
        b_off: usize,
        heads: usize,
        m: usize,
        k: usize,
        n: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; heads * m * n];
        for h in 0..heads {
            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0f32;
                    for t in 0..k {
                        sum += a[a_off + h * m * k + i * k + t] * b[b_off + h * k * n + t * n + j];
                    }
                    out[h * m * n + i * n + j] = sum;
                }
            }
        }
        out
    }

    /// Per-head score slabs `out[h][m×m] = a[h][m×k] @ b[h][m×k]ᵀ` on the
    /// host, at base offsets.
    fn scores_cpu(
        q: &[f32],
        q_off: usize,
        kk: &[f32],
        k_off: usize,
        heads: usize,
        m: usize,
        hd: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; heads * m * m];
        for h in 0..heads {
            for i in 0..m {
                for j in 0..m {
                    let mut sum = 0.0f32;
                    for t in 0..hd {
                        sum += q[q_off + h * m * hd + i * hd + t]
                            * kk[k_off + h * m * hd + j * hd + t];
                    }
                    out[h * m * m + i * m + j] = sum;
                }
            }
        }
        out
    }

    /// One kernel, both postures the S2 slice serves: heads=1 at non-zero
    /// offsets over PADDED parents (the whole-parent-bind shape the backend
    /// actually binds — the derivation-from-lengths kernel cannot express
    /// this) and heads=4 batched at zero offsets (the attention batch).
    #[test]
    fn test_matmul_transb_off_offsets_and_heads() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();
        let tol = 1e-3;

        // heads=1, non-zero offsets, padded parents.
        {
            let (heads, m, k) = (1usize, 37usize, 45usize);
            let (qa, ka, oa) = (11usize, 7usize, 3usize);
            let q = noise(qa + heads * m * k + 5, 0x1234);
            let kk = noise(ka + heads * m * k + 9, 0x5678);
            let q_h = client.create_from_slice(f32::as_bytes(&q));
            let k_h = client.create_from_slice(f32::as_bytes(&kk));
            let out_len = oa + heads * m * m + 13;
            let out_h = client.empty(out_len * core::mem::size_of::<f32>());
            unsafe {
                MatmulTransbOffCubeCL::launch::<ActiveRuntime>(
                    &client,
                    q_h,
                    q.len(),
                    k_h,
                    kk.len(),
                    out_h.clone(),
                    out_len,
                    heads,
                    m,
                    k,
                    qa,
                    ka,
                    oa,
                );
            }
            let bytes = client.read_one(out_h).expect("read out");
            let got = f32::from_bytes(&bytes);
            let want = scores_cpu(&q, qa, &kk, ka, heads, m, k);
            let mut max_err = 0.0f32;
            for (&w, &g) in want.iter().zip(got[oa..].iter()) {
                max_err = max_err.max((w - g).abs());
            }
            println!("transb-off offsets ({m}×{k}, heads {heads}): max_err = {max_err:.3e}");
            assert!(max_err < tol, "transb-off offsets diverged: {max_err:.3e}");
        }

        // heads=4, zero offsets — the `matmul_kt_heads` dispatch.
        {
            let (heads, m, k) = (4usize, 54usize, 64usize);
            let q = noise(heads * m * k, 0xabcd);
            let kk = noise(heads * m * k, 0xef01);
            let q_h = client.create_from_slice(f32::as_bytes(&q));
            let k_h = client.create_from_slice(f32::as_bytes(&kk));
            let out_h = client.empty(heads * m * m * core::mem::size_of::<f32>());
            unsafe {
                MatmulTransbOffCubeCL::launch::<ActiveRuntime>(
                    &client,
                    q_h,
                    q.len(),
                    k_h,
                    kk.len(),
                    out_h.clone(),
                    heads * m * m,
                    heads,
                    m,
                    k,
                    0,
                    0,
                    0,
                );
            }
            let bytes = client.read_one(out_h).expect("read out");
            let got = f32::from_bytes(&bytes);
            let want = scores_cpu(&q, 0, &kk, 0, heads, m, k);
            let mut max_err = 0.0f32;
            for (&w, &g) in want.iter().zip(got.iter()) {
                max_err = max_err.max((w - g).abs());
            }
            println!("transb-off heads ({heads}×{m}×{k}): max_err = {max_err:.3e}");
            assert!(max_err < tol, "transb-off heads diverged: {max_err:.3e}");
        }
    }

    /// The RR kernel at both postures: heads=1 at non-zero offsets (the
    /// `matmul` op) and heads=3 batched (the `matmul_heads` op).
    #[test]
    fn test_matmul_rr_off_offsets_and_heads() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();
        let tol = 1e-3;

        // heads=1, non-zero offsets over padded parents.
        {
            let (heads, m, k, n) = (1usize, 33usize, 29usize, 41usize);
            let (aa, ba, oa) = (17usize, 5usize, 23usize);
            let a = noise(aa + heads * m * k + 4, 0x0f0f);
            let b = noise(ba + heads * k * n + 8, 0x1111);
            let a_h = client.create_from_slice(f32::as_bytes(&a));
            let b_h = client.create_from_slice(f32::as_bytes(&b));
            let out_len = oa + heads * m * n + 6;
            let out_h = client.empty(out_len * core::mem::size_of::<f32>());
            unsafe {
                MatmulRrOffCubeCL::launch::<ActiveRuntime>(
                    &client,
                    a_h,
                    a.len(),
                    b_h,
                    b.len(),
                    out_h.clone(),
                    out_len,
                    heads,
                    m,
                    k,
                    n,
                    aa,
                    ba,
                    oa,
                );
            }
            let bytes = client.read_one(out_h).expect("read out");
            let got = f32::from_bytes(&bytes);
            let want = matmul_rr_cpu(&a, aa, &b, ba, heads, m, k, n);
            let mut max_err = 0.0f32;
            for (&w, &g) in want.iter().zip(got[oa..].iter()) {
                max_err = max_err.max((w - g).abs());
            }
            println!("rr-off offsets ({m}×{k}×{n}): max_err = {max_err:.3e}");
            assert!(max_err < tol, "rr-off offsets diverged: {max_err:.3e}");
        }

        // heads=3, zero offsets — the `matmul_heads` dispatch (scores @ v).
        {
            let (heads, m, k, n) = (3usize, 54usize, 54usize, 64usize);
            let a = noise(heads * m * k, 0x2222);
            let b = noise(heads * k * n, 0x3333);
            let a_h = client.create_from_slice(f32::as_bytes(&a));
            let b_h = client.create_from_slice(f32::as_bytes(&b));
            let out_h = client.empty(heads * m * n * core::mem::size_of::<f32>());
            unsafe {
                MatmulRrOffCubeCL::launch::<ActiveRuntime>(
                    &client,
                    a_h,
                    a.len(),
                    b_h,
                    b.len(),
                    out_h.clone(),
                    heads * m * n,
                    heads,
                    m,
                    k,
                    n,
                    0,
                    0,
                    0,
                );
            }
            let bytes = client.read_one(out_h).expect("read out");
            let got = f32::from_bytes(&bytes);
            let want = matmul_rr_cpu(&a, 0, &b, 0, heads, m, k, n);
            let mut max_err = 0.0f32;
            for (&w, &g) in want.iter().zip(got.iter()) {
                max_err = max_err.max((w - g).abs());
            }
            println!("rr-off heads ({heads}×{m}×{k}×{n}): max_err = {max_err:.3e}");
            assert!(max_err < tol, "rr-off heads diverged: {max_err:.3e}");
        }
    }
}
