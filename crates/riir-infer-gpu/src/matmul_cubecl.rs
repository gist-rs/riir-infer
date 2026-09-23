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
}
