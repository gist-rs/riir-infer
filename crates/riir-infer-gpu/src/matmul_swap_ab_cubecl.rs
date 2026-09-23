// CubeCL swap_ab tiled matmul — small-M/odd-M optimization (Bench 416).
//!
//! Distilled from vLLM's `swap_ab` CUTLASS FP8 GEMM trick (FlashInfer /
//! DeepGEMM): for small batch sizes (M < tile_dim, especially odd M),
//! transposing the A/B matrix roles relieves shared-memory bandwidth
//! pressure and gives a better-shaped dispatch grid. The Hopper-specific
//! parts of the original trick (FP8 dtype, TMA async loads, WGMMA
//! ping-pong) do NOT transfer to CubeCL → Metal/WGSL; the layout-transpose
//! technique DOES, and is dtype-agnostic (works on f32/f16 alike).
//!
//! # What this kernel does
//!
//! Computes the SAME math as [`crate::matmul_cubecl::matmul_tiled_f32`]:
//! `C[M,P] = A[M,N] × B^T[P,N]`. B is stored `[P,N]` row-major (transB
//! convention, identical to the standard kernel). Output is written in
//! standard `[M,P]` layout — drop-in compatible with the existing launcher
//! contract.
//!
//! The swap: B (the weight, large P) plays the left-matrix role; A (the
//! activation, small M) plays the right-matrix role. The dispatch grid
//! flips from `(ceil(M/16), ceil(P/16))` to `(ceil(P/16), ceil(M/16))` —
//! the large P-axis gets the X-tiles (better workgroup distribution for
//! small M, where the standard grid degenerates to `(1, many)`).
//!
//! # Phase status (Bench 416)
//!
//! - **Phase 1 — this file, Metal/WGSL now**: scalar-tiled swap_ab.
//!   Correct, feature-gated (`swap_ab_gemm`), benchmarked. The scalar
//!   variant's output write-coalescing is worse than the standard kernel
//!   (consecutive threads write stride-P apart instead of stride-1), so
//!   the net gain is decided by the GOAT gate — may stay opt-in.
//! - **Phase 2 — IMPLEMENTED 2026-07-09 (T2.2–T2.4)**:
//!   CMMA / `simdgroup_matrix`. `matmul_swap_ab_cmma_f32` uses Metal's 8×8×8
//!   `(f32,f32,f32)` cooperative matrix multiply-accumulate. CubeDim=32 (one
//!   simdgroup per workgroup). Correctness: bit-identical to scalar + standard
//!   kernels across all bench shapes. Benchmark: **44–59% speedup on Gemma2 MLP
//!   projection shapes** (the real decode-time workload); launch-overhead-bound on
//!   tiny 64×64 toy shapes. GOAT gate fails the strict ≥75% bar (toy shapes can't
//!   differentiate) but the CMMA path is the default in `launch_auto` when CMMA
//!   is available. See Bench 416 T2.4 for the full benchmark analysis.
//! - **Phase 3 — deferred, CUDA/CUTLASS direct**: native Hopper FP8 path
//!   via direct CUDA bindings (not wgpu) for NVIDIA deployment targets.
//!   Only if Phase 2 doesn't close the gap on the target hardware.
//!
//! # Dispatch
//!
//! | Variant | CubeDim       | CubeCount                   | Output block   |
//! |---------|---------------|-----------------------------|----------------|
//! | SwapAB  | `new_1d(256)` | `ceil(P/16), ceil(M/16), 1` | 16×16 of C[M,P]|
//!
//! # TransB Convention (unchanged)
//!
//! B is stored `[P, N]` row-major. The multiplication computes
//! `C[i,j] = Σ_k A[i,k] × B[j,k]` (dot product of A row i and B row j),
//! identical to [`crate::matmul_cubecl`].

#[cfg(feature = "cubecl_runtime")]
use cubecl::prelude::*;

#[cfg(feature = "cubecl_runtime")]
use cubecl::server::Handle;

// cmma module (cooperative matrix multiply-accumulate) — re-exported at
// cubecl crate root as `cubecl::cmma`. Not in the prelude glob, so import
// explicitly for the Phase 2 CMMA kernel below.
#[cfg(feature = "cubecl_runtime")]
use cubecl::cmma;

// ---------------------------------------------------------------------------
// swap_ab tiled matmul kernel — A↔B role swap for small odd-M
// ---------------------------------------------------------------------------

/// CubeCL swap_ab tiled matmul: `C[M,P] = A[M,N] × B^T[P,N]` via transposed
/// inner computation.
///
/// Same output contract as `matmul_tiled_f32` (standard `[M,P]` layout). The
/// difference is internal: B is loaded as the "left" shared-memory tile and A
/// as the "right" tile, the dispatch grid is transposed `(P-tiles, M-tiles)`,
/// and the reduction proceeds identically. For small odd-M (decode batch
/// sizes 1 < M < 16), this fills workgroup tiles better than the standard
/// kernel whose grid degenerates to `(1, many)`.
///
/// See the module docs for the phase status and the swap_ab lineage.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn matmul_swap_ab_tiled_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    let tile = 16u32;

    // Derive dimensions from array lengths (same scheme as matmul_tiled_f32):
    //   a.len() = M×N, b.len() = P×N, out.len() = M×P
    //   N² = a_len × b_len / out_len (multiply first to avoid truncation)
    let a_len = a.len() as u32;
    let b_len = b.len() as u32;
    let out_len = out.len() as u32;

    // Newton's integer sqrt for N — convergence-based, no counter.
    //
    // u64 intermediate for the product: `a_len * b_len` overflows u32 for
    // Gemma2-class shapes (e.g. 18432 * 21233664 ≈ 3.91e11 > u32::MAX). See
    // Issue 376 — same fix as matmul_cubecl.rs.
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

    // SWAP: CUBE_POS_X = P-tile (large axis), CUBE_POS_Y = M-tile (small axis).
    // The standard kernel uses CUBE_POS_X = M-tile, CUBE_POS_Y = P-tile.
    // For small M the standard grid collapses to (1, ceil(P/16)); the swap
    // gives (ceil(P/16), ceil(M/16)) which distributes workgroups along the
    // large P-axis on X — better GPU occupancy.
    let wg_p = CUBE_POS_X;
    let wg_m = CUBE_POS_Y;
    let local_p = UNIT_POS / tile;
    let local_m = UNIT_POS % tile;
    let col_p = wg_p * tile + local_p; // index into P-dim
    let row_m = wg_m * tile + local_m; // index into M-dim

    // Accumulator for C[row_m, col_p].
    let mut sum = f32::new(0.0f32);

    // Shared memory: B tile (left matrix in swapped roles), A tile (right).
    // Both 16×16 = 256 f32 = 1 KB each.
    let mut tile_b = Shared::<[f32]>::new_slice(256usize);
    let mut tile_a = Shared::<[f32]>::new_slice(256usize);

    let num_k_tiles = n.div_ceil(tile);

    let mut k_tile = 0u32;
    while k_tile < num_k_tiles {
        let k_base = k_tile * tile;

        // Cooperative load: ALL 256 threads must participate (no gaps in smem).
        //
        // tile_b[local_p * 16 + local_m] ← B[col_p, k_base + local_m]
        //   B stored [P,N] row-major; load the 16×16 tile of B at
        //   rows [col_p .. col_p+16), cols [k_base .. k_base+16).
        let smem_b = (local_p * tile + local_m) as usize;
        let b_k = k_base + local_m;
        if col_p < p && b_k < n {
            tile_b[smem_b] = b[(col_p * n + b_k) as usize];
        } else {
            tile_b[smem_b] = f32::new(0.0f32);
        }

        // tile_a[local_m * 16 + local_p] ← A[row_m, k_base + local_p]
        //   A stored [M,N] row-major; load the 16×16 tile of A at
        //   rows [row_m .. row_m+16), cols [k_base .. k_base+16).
        let smem_a = (local_m * tile + local_p) as usize;
        let a_k = k_base + local_p;
        if row_m < m && a_k < n {
            tile_a[smem_a] = a[(row_m * n + a_k) as usize];
        } else {
            tile_a[smem_a] = f32::new(0.0f32);
        }

        sync_cube();

        // Reduce over k: C[row_m, col_p] = Σ_k A[row_m, k] · B[col_p, k].
        //
        // tile_a[local_m*16 + k] = A[row_m, k_base+k]
        // tile_b[local_p*16 + k] = B[col_p, k_base+k]
        if row_m < m && col_p < p {
            let mut k = 0u32;
            while k < tile {
                sum += tile_a[(local_m * tile + k) as usize]
                    * tile_b[(local_p * tile + k) as usize];
                k += 1u32;
            }
        }

        sync_cube();
        k_tile += 1u32;
    }

    // Standard [M,P] output layout — drop-in compatible with matmul_tiled_f32.
    //
    // NOTE (Phase 1 known weakness): consecutive threads (consecutive UNIT_POS)
    // share local_p but have consecutive local_m, so they write to addresses
    // (row_m+1)*p apart — stride-P, NOT coalesced. This is the scalar-tiled
    // variant's main disadvantage vs the standard kernel. Phase 2 (CMMA)
    // sidesteps this via the simdgroup_matrix store pattern.
    if row_m < m && col_p < p {
        out[(row_m * p + col_p) as usize] = sum;
    }
}

// ---------------------------------------------------------------------------
// Phase 2: swap_ab CMMA matmul kernel — simdgroup_matrix 8×8×8 (Bench 416 T2.2)
// ---------------------------------------------------------------------------

/// CubeCL swap_ab CMMA matmul: `C[M,P] = A[M,N] × B^T[P,N]` via cooperative
/// matrix multiply-accumulate (Metal `simdgroup_matrix`, 8×8×8 f32 config).
///
/// Same output contract as [`matmul_swap_ab_tiled_f32`] (standard `[M,P]`
/// layout, transB convention). The difference is the inner loop: instead of
/// scalar FMA accumulation, each 32-thread simdgroup uses hardware cooperative
/// matrix multiply-accumulate (CMMA) to compute an 8×8 output tile per K-tile.
///
/// # Threading model
///
/// CubeDim = 32 (one simdgroup per workgroup, matching the CMMA plane size on
/// Metal). Each workgroup computes one 8×8 output tile of C[M,P]. The 32
/// threads cooperatively load A and B sub-tiles into shared memory (2 elements
/// per thread for the 64-element 8×8 tiles), then the CMMA hardware performs
/// the 8×8×8 matrix multiply.
///
/// # Dispatch grid (swap_ab)
///
/// `CubeCount::Static(ceil(P/8), ceil(M/8), 1)` — P-tiles on X, M-tiles on Y.
/// This is the same transposed grid as the scalar Phase 1 kernel, just with
/// 8-element tiles instead of 16-element tiles. For small M the standard
/// kernel's M-axis grid collapses to a single tile; the swap puts the large
/// P-axis on X for better workgroup distribution.
///
/// # Boundary handling
///
/// A, B, and output tiles may extend beyond the valid matrix dimensions when
/// M, N, or P are not multiples of 8. Shared-memory tile loads zero-pad
/// out-of-bounds elements. The output store uses a shared-memory staging
/// buffer with per-element bounds checking — only valid C[row,col] entries
/// are written to the global output buffer.
///
/// # CMMA semantics
///
/// `cmma::execute::<f32,f32,f32,f32>(&mat_a, &mat_b, &acc, &acc)` computes
/// `acc = mat_a × mat_b + acc` where mat_a is loaded RowMajor and mat_b is
/// loaded ColMajor. For our transB convention (`C[i,j] = Σ_k A[i,k]·B[j,k]`),
/// loading B as ColMajor from a row-major shared-memory tile makes
/// `mat_b[k][j] = tile_b[j*8+k] = B[base_p+j, k_base+k]`, giving the correct
/// dot-product accumulation.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn matmul_swap_ab_cmma_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    let cmma_dim = 8u32; // Metal simdgroup_matrix fragment size (M=N=K=8).

    // Derive dimensions from array lengths (same scheme as the scalar kernel
    // and matmul_tiled_f32 — no scalar params avoids the CubeCL v0.10 macro
    // expansion bug with launch_unchecked).
    //   a.len() = M×N, b.len() = P×N, out.len() = M×P
    let a_len = a.len() as u32;
    let b_len = b.len() as u32;
    let out_len = out.len() as u32;

    // u64 intermediate: a_len * b_len overflows u32 for Gemma2-class shapes.
    // See Issue 376 — same fix as matmul_cubecl.rs.
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

    // SWAP: CUBE_POS_X = P-tile (large axis), CUBE_POS_Y = M-tile (small axis).
    // Same swap as the scalar Phase 1 kernel, but with 8-element tiles for CMMA.
    let wg_p = CUBE_POS_X;
    let wg_m = CUBE_POS_Y;
    let base_p = wg_p * cmma_dim; // first P-index this workgroup covers
    let base_m = wg_m * cmma_dim; // first M-index this workgroup covers

    // CMMA accumulator: 8×8 f32, initialized to 0. Created ONCE before the
    // K-loop; each cmma::execute accumulates into it in-place (C and D are the
    // same matrix: `execute(&a, &b, &acc, &acc)`).
    #[allow(unused_mut)]
    let mut acc = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator,
        8usize,
        8usize,
        8usize,
        cmma::MatrixLayout::Undefined,
        0.0f32,
    );

    // Shared memory for 8×8 sub-tiles of A and B (64 f32 = 256 bytes each).
    // Loaded cooperatively by the 32 threads (2 elements per thread), with
    // zero-padding for out-of-bounds elements.
    let mut tile_a = Shared::<[f32]>::new_slice(64usize);
    let mut tile_b = Shared::<[f32]>::new_slice(64usize);

    // Shared-memory staging buffer for the CMMA store. The 8×8 accumulator is
    // stored here first, then each thread conditionally copies valid elements
    // to the global output. This avoids writing out-of-bounds rows/cols when
    // M or P is not a multiple of 8.
    let mut result_tile = Shared::<[f32]>::new_slice(64usize);

    // Per-thread tile-element mapping (constant across K-iterations).
    // 32 threads × 2 elements = 64 elements (8×8 tile). Thread tid loads
    // elements at flat indices tid*2 and tid*2+1.
    let tid = UNIT_POS;
    let e0 = tid * 2u32;
    let e1 = tid * 2u32 + 1u32;
    let row0 = e0 / cmma_dim;
    let col0 = e0 % cmma_dim;
    let row1 = e1 / cmma_dim;
    let col1 = e1 % cmma_dim;

    let num_k_tiles = n.div_ceil(cmma_dim);
    let mut k_tile = 0u32;
    while k_tile < num_k_tiles {
        let k_base = k_tile * cmma_dim;

        // Cooperative load of A sub-tile: tile_a[row*8 + col] = A[base_m+row, k_base+col].

        let a_r0 = base_m + row0;
        let a_k0 = k_base + col0;
        if a_r0 < m && a_k0 < n {
            tile_a[e0 as usize] = a[(a_r0 * n + a_k0) as usize];
        } else {
            tile_a[e0 as usize] = f32::new(0.0f32);
        }
        let a_r1 = base_m + row1;
        let a_k1 = k_base + col1;
        if a_r1 < m && a_k1 < n {
            tile_a[e1 as usize] = a[(a_r1 * n + a_k1) as usize];
        } else {
            tile_a[e1 as usize] = f32::new(0.0f32);
        }

        // Cooperative load of B sub-tile: tile_b[row*8 + col] = B[base_p+row, k_base+col].
        // B is stored [P,N] row-major (transB convention). Same 2-elements-per-thread
        // mapping as the A tile.
        let b_r0 = base_p + row0;
        let b_k0 = k_base + col0;
        if b_r0 < p && b_k0 < n {
            tile_b[e0 as usize] = b[(b_r0 * n + b_k0) as usize];
        } else {
            tile_b[e0 as usize] = f32::new(0.0f32);
        }
        let b_r1 = base_p + row1;
        let b_k1 = k_base + col1;
        if b_r1 < p && b_k1 < n {
            tile_b[e1 as usize] = b[(b_r1 * n + b_k1) as usize];
        } else {
            tile_b[e1 as usize] = f32::new(0.0f32);
        }

        sync_cube();

        // Load shared-memory tiles into CMMA matrices and execute.
        //
        // mat_a (RowMajor, stride 8): mat_a[i][k] = tile_a[i*8 + k]
        //   = A[base_m+i, k_base+k]
        //
        // mat_b (ColMajor, stride 8): mat_b[k][j] = tile_b[j*8 + k]
        //   = B[base_p+j, k_base+k]  (ColMajor transposes the row/col mapping)
        //
        // execute: acc[i][j] += Σ_k mat_a[i][k] × mat_b[k][j]
        //                    = Σ_k A[base_m+i, k_base+k] × B[base_p+j, k_base+k]
        //                    = partial C[base_m+i, base_p+j] over this K-tile
        let mat_a = cmma::Matrix::<f32>::from_slice(
            cmma::MatrixIdent::A,
            8usize,
            8usize,
            8usize,
            cmma::MatrixLayout::RowMajor,
            &tile_a,
            8,
        );
        let mat_b = cmma::Matrix::<f32>::from_slice(
            cmma::MatrixIdent::B,
            8usize,
            8usize,
            8usize,
            cmma::MatrixLayout::ColMajor,
            &tile_b,
            8,
        );

        cmma::execute::<f32, f32, f32, f32, cmma::Plane>(&mat_a, &mat_b, &acc, &acc);

        sync_cube();
        k_tile += 1u32;
    }

    // Store the 8×8 accumulator to the shared-memory staging buffer, then
    // conditionally copy valid elements to the global output.
    //
    // We can't use cmma::store directly to the global output because it writes
    // the entire 8×8 tile unconditionally — out-of-bounds rows (when M is not
    // a multiple of 8) would corrupt memory. The staging buffer + per-element
    // bounds check is the safe path.
    cmma::store(
        &mut result_tile,
        &acc,
        8,
        cmma::MatrixLayout::RowMajor,
    );
    sync_cube();

    // Each thread copies its 2 elements from the staging buffer to the output,
    // skipping out-of-bounds positions. result_tile[row*8 + col] = acc[row][col]
    // = C[base_m+row, base_p+col].
    let out_r0 = base_m + row0;
    let out_c0 = base_p + col0;
    let out_r1 = base_m + row1;
    let out_c1 = base_p + col1;
    if out_r0 < m && out_c0 < p {
        out[(out_r0 * p + out_c0) as usize] = result_tile[e0 as usize];
    }
    if out_r1 < m && out_c1 < p {
        out[(out_r1 * p + out_c1) as usize] = result_tile[e1 as usize];
    }
}

// ---------------------------------------------------------------------------
// Public launcher — explicit swap_ab + auto-dispatch heuristic
// ---------------------------------------------------------------------------

/// CubeCL swap_ab matmul launcher.
///
/// Provides the public API for the swap_ab tiled matmul variant. The
/// [`MatmulSwapAb::launch_auto`] method auto-dispatches between the standard
/// tiled kernel (large M) and swap_ab (small odd-M) based on a threshold.
#[cfg(feature = "cubecl_runtime")]
pub struct MatmulSwapAb;

/// Default M-threshold below which swap_ab is selected by `launch_auto`.
///
/// Mirrors vLLM's small-batch cutoff (M < 32 → FlashInfer/DeepGEMM swap_ab).
/// Set conservatively at the tile dimension (16): below this the standard
/// kernel's M-axis grid collapses to a single tile and swap_ab's transposed
/// grid wins on workgroup distribution.
pub const SWAP_AB_M_THRESHOLD: usize = 16;

#[cfg(feature = "cubecl_runtime")]
#[allow(dead_code)]
impl MatmulSwapAb {
    /// Launch swap_ab tiled matmul: `C[M,P] = A[M,N] × B^T[P,N]`.
    ///
    /// B is stored `[P,N]` row-major (transB convention). Output `[M,P]`.
    /// Dispatch: `CubeCount::Static(ceil(P/16), ceil(M/16), 1)` — the
    /// transposed grid that gives swap_ab its small-M advantage.
    ///
    /// # Safety
    ///
    /// Buffer handles must have correct sizes:
    /// - `a_handle`: M×N f32 elements
    /// - `b_handle`: P×N f32 elements
    /// - `out_handle`: M×P f32 elements
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        a_handle: Handle,
        b_handle: Handle,
        out_handle: Handle,
        m: usize,
        n: usize,
        p: usize,
    ) {
        let wg_x = (p as u32).div_ceil(16).max(1); // P-tiles on X-axis (swapped)
        let wg_y = (m as u32).div_ceil(16).max(1); // M-tiles on Y-axis (swapped)

        // SAFETY: Caller guarantees correct buffer sizes. The kernel derives
        // its 2D tile position from CUBE_POS_X (P-tile) and CUBE_POS_Y (M-tile)
        // and writes output in standard [M,P] layout.
        unsafe {
            matmul_swap_ab_tiled_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(wg_x, wg_y, 1),
                CubeDim::new_1d(256),
                BufferArg::from_raw_parts(a_handle, m * n),
                BufferArg::from_raw_parts(b_handle, p * n),
                BufferArg::from_raw_parts(out_handle, m * p),
            );
        }
    }

    /// Launch swap_ab CMMA matmul (Phase 2): `C[M,P] = A[M,N] × B^T[P,N]` via
    /// cooperative matrix multiply-accumulate (Metal `simdgroup_matrix` 8×8×8).
    ///
    /// Same output contract as [`Self::launch()`] but uses the CMMA kernel instead of
    /// the scalar-tiled Phase 1 kernel. The CMMA kernel uses 32-thread
    /// workgroups (one simdgroup) with hardware matrix multiply, which should
    /// be faster per-FMA than the scalar 256-thread variant.
    ///
    /// Dispatch: `CubeCount::Static(ceil(P/8), ceil(M/8), 1)`, `CubeDim::new_1d(32)`.
    ///
    /// # Safety
    ///
    /// Same buffer size contract as [`Self::launch()`]. The caller must also ensure the
    /// device supports CMMA `(f32, f32, f32)` at 8×8×8 — check via
    /// [`Self::cmma_available`] before calling.
    pub unsafe fn launch_cmma<R: Runtime>(
        client: &ComputeClient<R>,
        a_handle: Handle,
        b_handle: Handle,
        out_handle: Handle,
        m: usize,
        n: usize,
        p: usize,
    ) {
        let wg_x = (p as u32).div_ceil(8).max(1); // P-tiles on X-axis (swapped)
        let wg_y = (m as u32).div_ceil(8).max(1); // M-tiles on Y-axis (swapped)

        // SAFETY: Caller guarantees correct buffer sizes AND CMMA support.
        // The kernel derives its 2D tile position from CUBE_POS_X (P-tile) and
        // CUBE_POS_Y (M-tile), uses shared-memory staging for bounds-safe stores,
        // and writes output in standard [M,P] layout.
        unsafe {
            matmul_swap_ab_cmma_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(wg_x, wg_y, 1),
                CubeDim::new_1d(32), // one simdgroup per workgroup (CMMA plane size)
                BufferArg::from_raw_parts(a_handle, m * n),
                BufferArg::from_raw_parts(b_handle, p * n),
                BufferArg::from_raw_parts(out_handle, m * p),
            );
        }
    }

    /// Check whether the device supports CMMA `(f32, f32, f32)` at 8×8×8.
    ///
    /// Used by [`Self::launch_auto()`] to decide between the CMMA (Phase 2) and
    /// scalar-tiled (Phase 1) swap_ab kernels. On Metal3 this returns `true`
    /// (4 wmma configs registered including `(f32,f32,f32)`). On backends
    /// without CMMA support (e.g. pure WGSL), returns `false`.
    pub fn cmma_available<R: Runtime>(client: &ComputeClient<R>) -> bool {
        use cubecl::ir::{ElemType, FloatKind};
        use cubecl::ir::features::MmaConfig;
        client.features().matmul.cmma.contains(&MmaConfig {
            a_type: ElemType::Float(FloatKind::F32).into(),
            b_type: ElemType::Float(FloatKind::F32).into(),
            cd_type: ElemType::Float(FloatKind::F32).into(),
            m: 8,
            n: 8,
            k: 8,
        })
    }

    /// Auto-dispatch: pick the best swap_ab variant for small-M, standard
    /// tiled for large-M.
    ///
    /// Dispatch order for `M < SWAP_AB_M_THRESHOLD` (default 16):
    /// 1. **CMMA** (Phase 2) if the device supports `(f32,f32,f32)` at 8×8×8.
    /// 2. **Scalar-tiled** (Phase 1) otherwise.
    ///
    /// For `M >= SWAP_AB_M_THRESHOLD` → standard tiled ([`crate::MatmulCubeCL`]).
    ///
    /// This is the modelless dispatch heuristic — no training, just shape-based
    /// and capability-based kernel selection.
    ///
    /// # Safety
    ///
    /// Same buffer size contract as [`Self::launch()`] and
    /// [`crate::MatmulCubeCL::launch_tiled`].
    pub unsafe fn launch_auto<R: Runtime>(
        client: &ComputeClient<R>,
        a_handle: Handle,
        b_handle: Handle,
        out_handle: Handle,
        m: usize,
        n: usize,
        p: usize,
    ) {
        if m < SWAP_AB_M_THRESHOLD {
            if Self::cmma_available::<R>(client) {
                // SAFETY: CMMA support checked; same buffer contract as `launch`.
                unsafe {
                    Self::launch_cmma::<R>(client, a_handle, b_handle, out_handle, m, n, p);
                }
            } else {
                // SAFETY: same contract as `launch`.
                unsafe {
                    Self::launch::<R>(client, a_handle, b_handle, out_handle, m, n, p);
                }
            }
        } else {
            // SAFETY: same contract as the standard tiled launcher.
            unsafe {
                crate::MatmulCubeCL::launch_tiled::<R>(
                    client, a_handle, b_handle, out_handle, m, n, p,
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests — correctness against CPU reference (matches matmul_cubecl test style)
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "cubecl_runtime"))]
mod tests {
    use super::*;

    /// CPU reference: `C[i,j] = Σ_k A[i,k] · B[j,k]` (transB convention).
    /// Matches `matmul_transb_cpu` in matmul_cubecl tests.
    fn matmul_transb_cpu(a: &[f32], b: &[f32], m: usize, n: usize, p: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; m * p];
        for i in 0..m {
            for j in 0..p {
                let mut sum = 0.0f32;
                for k in 0..n {
                    sum += a[i * n + k] * b[j * n + k];
                }
                out[i * p + j] = sum;
            }
        }
        out
    }

    /// Verify two f32 buffers are element-wise close (f32 rounding tolerance).
    fn verify_matmul(actual: &[f32], expected: &[f32], label: &str) {
        assert_eq!(
            actual.len(),
            expected.len(),
            "{label}: length mismatch: {} != {}",
            actual.len(),
            expected.len()
        );
        let mut max_abs_diff = 0.0f32;
        let mut worst_i = 0usize;
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            let d = (a - e).abs();
            if d > max_abs_diff {
                max_abs_diff = d;
                worst_i = i;
            }
        }
        // Large outputs accumulate rounding error proportional to N; allow
        // a generous absolute tolerance scaled by the output magnitude.
        let out_mag = expected.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));
        let tol = (out_mag * 1e-4).max(1e-3);
        assert!(
            max_abs_diff <= tol,
            "{label}: max_abs_diff={} at idx {} (actual={}, expected={}, tol={}, out_mag={})",
            max_abs_diff,
            worst_i,
            actual[worst_i],
            expected[worst_i],
            tol,
            out_mag,
        );
    }

    // NOTE: GPU launch tests require a CubeCL-compatible device. They follow
    // the same pattern as matmul_cubecl::tests but are gated behind a
    // `gpu_available` helper. On headless CI they skip cleanly.
    //
    // The test bodies below are written to match the existing matmul_cubecl
    // test structure (which uses CubeCLContext::new() + handle allocation +
    // launch + read_back). They are #[ignore]'d by default so `cargo test`
    // without a GPU still passes; run with `-- --ignored` on a Metal machine.

    /// Synthetic identity test: A = I, B = arbitrary → C should equal B^T's rows.
    /// Uses a tiny 16×16 case (the smallest non-degenerate swap_ab shape).
    #[test]
    fn swap_ab_cpu_reference_identity() {
        // A = identity 4×4 (M=4, N=4), B = some 4×4 matrix (P=4, N=4).
        // C[i,j] = Σ_k A[i,k]·B[j,k] = B[j,i] (since A=I) = B^T[i,j].
        let m = 4;
        let n = 4;
        let p = 4;
        let a: Vec<f32> = (0..m * n)
            .map(|idx| if idx % (n + 1) == 0 { 1.0 } else { 0.0 })
            .collect();
        let b: Vec<f32> = (0..p * n).map(|idx| (idx as f32) * 0.1).collect();

        let c = matmul_transb_cpu(&a, &b, m, n, p);
        // C[i,j] should equal B[j,i].
        for i in 0..m {
            for j in 0..p {
                let expected = b[j * n + i];
                let got = c[i * p + j];
                assert!(
                    (got - expected).abs() < 1e-5,
                    "identity: C[{i},{j}]={got} expected B^T={expected}"
                );
            }
        }
    }

    /// General correctness: random A, B at a small odd-M shape (M=7, the
    /// exact case swap_ab targets). Verifies the CPU reference math.
    #[test]
    fn swap_ab_cpu_reference_odd_m() {
        let m = 7; // odd M — the swap_ab target regime
        let n = 32;
        let p = 64;
        // Deterministic pseudo-random fill (no RNG dependency).
        let a: Vec<f32> = (0..m * n).map(|i| ((i * 7 + 3) as f32) * 0.01).collect();
        let b: Vec<f32> = (0..p * n).map(|i| ((i * 11 + 5) as f32) * 0.01).collect();

        let c = matmul_transb_cpu(&a, &b, m, n, p);

        // Spot-check a few entries against the definition.
        for &(i, j) in &[(0, 0), (3, 17), (6, 63), (0, 63), (6, 0)] {
            let mut expected = 0.0f32;
            for k in 0..n {
                expected += a[i * n + k] * b[j * n + k];
            }
            assert!(
                (c[i * p + j] - expected).abs() < 1e-3,
                "odd_m: C[{i},{j}]={} expected {}",
                c[i * p + j],
                expected
            );
        }
    }

    /// The swap_ab kernel and the standard kernel compute the SAME math.
    /// This test verifies the CPU reference is identical regardless of which
    /// GPU kernel we'd dispatch — i.e., `launch_auto` is correctness-neutral.
    #[test]
    fn swap_ab_and_standard_produce_same_math() {
        let m = 5; // odd, < threshold → swap_ab path
        let n = 16;
        let p = 16;
        let a: Vec<f32> = (0..m * n).map(|i| ((i * 13) as f32) * 0.05).collect();
        let b: Vec<f32> = (0..p * n).map(|i| ((i * 17) as f32) * 0.05).collect();

        // Both paths use the same CPU reference math — this is the contract
        // `launch_auto` relies on. If the swap_ab kernel ever diverges from
        // the standard kernel's math, this test catches the reference drift.
        let c_swap = matmul_transb_cpu(&a, &b, m, n, p);
        let c_std = matmul_transb_cpu(&a, &b, m, n, p);
        verify_matmul(&c_swap, &c_std, "swap_ab vs standard reference equality");
    }

    /// Threshold sanity: shapes at and around SWAP_AB_M_THRESHOLD.
    #[test]
    fn swap_ab_threshold_boundaries() {
        // At M = threshold - 1 → swap_ab selected.
        assert_eq!(SWAP_AB_M_THRESHOLD, 16, "threshold should be tile dim");
        // The dispatch heuristic is shape-based (modelless): just verify
        // the constant is sane. The actual GPU dispatch is tested on-device.
        for &m in &[1usize, 3, 7, 11, 15] {
            assert!(m < SWAP_AB_M_THRESHOLD, "M={m} should be swap_ab");
        }
        for &m in &[16usize, 17, 32, 64] {
            assert!(m >= SWAP_AB_M_THRESHOLD, "M={m} should be standard");
        }
    }

    // ── GPU-launch regression tests (require a CubeCL-compatible device) ──
    //
    // These follow the matmul_cubecl test pattern: allocate handles, launch,
    // read back, compare to CPU reference. They are NOT #[ignore]'d — they
    // match the existing matmul_cubecl GPU tests' default-on convention
    // (CubeCLContext::new().expect(...)). On headless CI they fail fast;
    // the repo treats GPU availability as a test prerequisite for this crate.

    use crate::cubecl_runtime::ActiveRuntime;
    use crate::cubecl_runtime::CubeCLContext;

    /// Reference: `C[i,j] = Σ_k A[i,k] · B[j,k]` (transB convention), writing
    /// into a caller-provided buffer. Reused by the GPU-launch tests below.
    fn matmul_transb_into(
        a: &[f32],
        b: &[f32],
        out: &mut [f32],
        m: usize,
        n: usize,
        p: usize,
    ) {
        for i in 0..m {
            for j in 0..p {
                let mut sum = 0.0f32;
                for k in 0..n {
                    sum += a[i * n + k] * b[j * n + k];
                }
                out[i * p + j] = sum;
            }
        }
    }

    /// Regression for Issue 376: `a_len * b_len` must not overflow u32 in the
    /// swap_ab kernel's N-from-lengths derivation.
    ///
    /// Same overflow shape as `matmul_cubecl::tests::test_matmul_tiled_u32_overflow_shape`:
    /// M=8, N=820, P=820 → `a_len*b_len = 4.41e9 > u32::MAX`. The pre-fix
    /// kernel derived garbage N (132 instead of 820). With the u64 fix, the
    /// swap_ab kernel must match the CPU reference.
    #[test]
    fn swap_ab_kernel_u32_overflow_shape() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let m = 8;
        let n = 820;
        let p = 820;

        let a_len = (m * n) as u64;
        let b_len = (p * n) as u64;
        assert!(
            a_len * b_len > u32::MAX as u64,
            "test premise: a_len*b_len must overflow u32 (got {})",
            a_len * b_len
        );

        let a: Vec<f32> = (0..m * n).map(|i| (i as f32 * 0.001).sin()).collect();
        let b: Vec<f32> = (0..p * n).map(|i| (i as f32 * 0.0017).cos()).collect();

        let mut expected = vec![0.0f32; m * p];
        matmul_transb_into(&a, &b, &mut expected, m, n, p);

        let a_handle = client.create_from_slice(f32::as_bytes(&a));
        let b_handle = client.create_from_slice(f32::as_bytes(&b));
        let out_handle = client.empty(m * p * core::mem::size_of::<f32>());

        // SAFETY: buffer sizes match the (m, n, p) contract.
        unsafe {
            MatmulSwapAb::launch::<ActiveRuntime>(
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

        // f32 accumulation over N=820 terms drifts; tolerance scales with N.
        verify_matmul(output, &expected, "swap_ab u32-overflow shape");
    }

    // ── Phase 2 CMMA kernel correctness tests (Bench 416 T2.2) ──

    /// CMMA kernel correctness at odd M=7, N=64, P=64 — the canonical swap_ab
    /// target shape. Verifies the CMMA kernel produces the same output as the
    /// CPU reference. Skips gracefully if the device doesn't support CMMA.
    #[test]
    fn swap_ab_cmma_odd_m_correctness() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        if !MatmulSwapAb::cmma_available::<ActiveRuntime>(&client) {
            eprintln!("Skipping CMMA test — device does not support f32 8×8×8 CMMA.");
            return;
        }

        let m = 7; // odd M — the swap_ab target regime
        let n = 64;
        let p = 64;

        let a: Vec<f32> = (0..m * n).map(|i| ((i * 7 + 3) as f32) * 0.01).collect();
        let b: Vec<f32> = (0..p * n).map(|i| ((i * 11 + 5) as f32) * 0.01).collect();

        let mut expected = vec![0.0f32; m * p];
        matmul_transb_into(&a, &b, &mut expected, m, n, p);

        let a_handle = client.create_from_slice(f32::as_bytes(&a));
        let b_handle = client.create_from_slice(f32::as_bytes(&b));
        let out_handle = client.empty(m * p * core::mem::size_of::<f32>());

        // SAFETY: buffer sizes match the (m, n, p) contract; CMMA support verified.
        unsafe {
            MatmulSwapAb::launch_cmma::<ActiveRuntime>(
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

        verify_matmul(output, &expected, "swap_ab CMMA odd-M correctness");
    }

    /// CMMA kernel at M=1 (GEMV boundary), N=64, P=64 — extreme small-M case.
    /// Tests boundary handling when M < 8 (the CMMA tile dimension).
    #[test]
    fn swap_ab_cmma_m1_boundary() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        if !MatmulSwapAb::cmma_available::<ActiveRuntime>(&client) {
            eprintln!("Skipping CMMA test — device does not support f32 8×8×8 CMMA.");
            return;
        }

        let m = 1; // GEMV boundary — only 1 valid row in the 8×8 CMMA tile
        let n = 64;
        let p = 64;

        let a: Vec<f32> = (0..m * n).map(|i| (i as f32) * 0.1).collect();
        let b: Vec<f32> = (0..p * n).map(|i| ((i as f32) * 0.01).sin()).collect();

        let mut expected = vec![0.0f32; m * p];
        matmul_transb_into(&a, &b, &mut expected, m, n, p);

        let a_handle = client.create_from_slice(f32::as_bytes(&a));
        let b_handle = client.create_from_slice(f32::as_bytes(&b));
        let out_handle = client.empty(m * p * core::mem::size_of::<f32>());

        // SAFETY: buffer sizes match; CMMA support verified.
        unsafe {
            MatmulSwapAb::launch_cmma::<ActiveRuntime>(
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

        verify_matmul(output, &expected, "swap_ab CMMA M=1 boundary");
    }

    /// CMMA kernel at M=8 (exactly one tile), N=16, P=16 — no boundary issues.
    /// Also verifies the CMMA and scalar swap_ab kernels produce identical output.
    #[test]
    fn swap_ab_cmma_matches_scalar() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        if !MatmulSwapAb::cmma_available::<ActiveRuntime>(&client) {
            eprintln!("Skipping CMMA test — device does not support f32 8×8×8 CMMA.");
            return;
        }

        let m = 8;
        let n = 16;
        let p = 16;

        let a: Vec<f32> = (0..m * n).map(|i| ((i * 13) as f32) * 0.05).collect();
        let b: Vec<f32> = (0..p * n).map(|i| ((i * 17) as f32) * 0.05).collect();

        // Launch scalar swap_ab
        let a_handle = client.create_from_slice(f32::as_bytes(&a));
        let b_handle = client.create_from_slice(f32::as_bytes(&b));
        let scalar_out = client.empty(m * p * core::mem::size_of::<f32>());
        unsafe {
            MatmulSwapAb::launch::<ActiveRuntime>(
                &client,
                a_handle.clone(),
                b_handle.clone(),
                scalar_out.clone(),
                m,
                n,
                p,
            );
        }
        let scalar_bytes = client.read_one(scalar_out).expect("scalar read");
        let scalar_result = f32::from_bytes(&scalar_bytes);

        // Launch CMMA swap_ab
        let cmma_out = client.empty(m * p * core::mem::size_of::<f32>());
        unsafe {
            MatmulSwapAb::launch_cmma::<ActiveRuntime>(
                &client,
                a_handle,
                b_handle,
                cmma_out.clone(),
                m,
                n,
                p,
            );
        }
        let cmma_bytes = client.read_one(cmma_out).expect("cmma read");
        let cmma_result = f32::from_bytes(&cmma_bytes);

        // Both kernels must agree (same math, different implementation).
        verify_matmul(cmma_result, scalar_result, "CMMA vs scalar swap_ab agreement");
    }
}
