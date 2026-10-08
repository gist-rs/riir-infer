//! CubeCL tiled matmul with an f16 weight operand — the eDLM lane's GEMM
//! (Issue 1005 T8), generic beyond it: `C[M,P] = A[M,N](f32) × B[P,N](f16)^T`.
//!
//! B is stored `[P,N]` row-major f16 — the transB convention every weight in
//! this crate already uses (GGUF row-major, `matmul_tiled_f32`-compatible).
//! The eDLM upload path dequants Q8_0 to f32 EXACTLY on the host (the same
//! `dequant_f16_to_f32` the CPU streaming path reads) then rounds to f16 once
//! at load; inside the kernel each f16 weight is cast back to f32 and the
//! accumulation stays f32 — the f16 GEMV family's law. The only numerics face
//! vs an f32-resident GEMM is the f16 weight rounding itself (rel ≤ 2^-11 per
//! weight), disclosed and gated by the lane's tolerance law, never
//! bit-identity.
//!
//! # Algorithm
//!
//! `matmul_cubecl::matmul_tiled_f32` verbatim: 16×16 shared-memory tiles,
//! 256-thread cubes, 2D dispatch (`CUBE_POS_X` = M-tile, `CUBE_POS_Y` =
//! P-tile), buffer-derived dimensions (Newton isqrt for N), bounds-checked
//! loads and writes — the only delta is the cast at the B tile load.
//!
//! # Dispatch
//!
//! | CubeDim       | CubeCount                    | Output block |
//! |---------------|------------------------------|--------------|
//! | `new_1d(256)` | `ceil(M/16), ceil(P/16), 1`  | 16×16        |

#[cfg(feature = "edlm_gpu")]
use cubecl::prelude::*;

#[cfg(feature = "edlm_gpu")]
use cubecl::server::Handle;

#[cfg(feature = "edlm_gpu")]
use half::f16 as half_f16;

#[cfg(feature = "edlm_gpu")]
use crate::cubecl_runtime::assert_binding_derives_units;

/// Tiled matmul with an f16 B operand: `C[M,P] = A[M,N] × B^T[P,N]`.
///
/// Dimensions are derived from the array lengths exactly like
/// `matmul_tiled_f32` (`a.len()=M·N`, `b.len()=P·N`, `out.len()=M·P`;
/// Newton isqrt for N with the u64 intermediate — Issue 376).
#[cfg(feature = "edlm_gpu")]
#[cube(launch_unchecked)]
fn matmul_tiled_f16b_f32(a: &[f32], b: &[half_f16], out: &mut [f32]) {
    let tile = 16u32;

    let a_len = a.len() as u32;
    let b_len = b.len() as u32;
    let out_len = out.len() as u32;

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

    let wg_row = CUBE_POS_X;
    let wg_col = CUBE_POS_Y;
    let local_row = UNIT_POS / tile;
    let local_col = UNIT_POS % tile;
    let row = wg_row * tile + local_row;
    let col = wg_col * tile + local_col;

    let mut sum = f32::new(0.0f32);

    let mut tile_a = Shared::<[f32]>::new_slice(256usize);
    let mut tile_b = Shared::<[f32]>::new_slice(256usize);

    let num_k_tiles = n.div_ceil(tile);

    let mut k_tile = 0u32;
    while k_tile < num_k_tiles {
        let k_base = k_tile * tile;

        let smem_a = (local_row * tile + local_col) as usize;
        let smem_b = (local_col * tile + local_row) as usize;

        let a_k = k_base + local_col;
        if row < m && a_k < n {
            tile_a[smem_a] = a[(row * n + a_k) as usize];
        } else {
            tile_a[smem_a] = f32::new(0.0f32);
        }

        // transB: B stored [P,N] row-major, swapped smem indices for the dot
        // product; the f16 weight is cast to f32 AT THE LOAD (accumulation
        // stays f32 — the f16 GEMV family's numerics law).
        let b_k = k_base + local_row;
        if col < p && b_k < n {
            tile_b[smem_b] = f32::cast_from(b[(col * n + b_k) as usize]);
        } else {
            tile_b[smem_b] = f32::new(0.0f32);
        }

        sync_cube();

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

    if row < m && col < p {
        out[(row * p + col) as usize] = sum;
    }
}

/// Launcher for [`matmul_tiled_f16b_f32`].
///
/// # Safety
///
/// Buffer handles must have correct sizes (asserted):
/// - `a_handle`: M×N f32 elements
/// - `b_handle`: P×N f16 elements
/// - `out_handle`: M×P f32 elements
#[cfg(feature = "edlm_gpu")]
pub struct MatmulF16bCubeCL;

#[cfg(feature = "edlm_gpu")]
impl MatmulF16bCubeCL {
    /// Launch the f16-B tiled matmul: `out[M,P] = a[M,N] × b[P,N]^T`.
    pub fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        a_handle: Handle,
        b_handle: Handle,
        out_handle: Handle,
        m: usize,
        n: usize,
        p: usize,
    ) {
        assert!(m > 0 && n > 0 && p > 0, "matmul dims must be positive");
        // The kernel derives its shape from the BOUND buffers — an oversized
        // binding silently changes the kernel's idea of its own shape (the
        // `.issues/515` identically-zero-output class); guard all three. The
        // standard guard divides by 4 (it assumes f32 handles), so the f16 B
        // buffer gets the byte-exact manual form of the same check.
        assert_binding_derives_units(&a_handle, n, m, "MatmulF16bCubeCL a");
        assert_eq!(
            b_handle.size_in_used(),
            (p * n * core::mem::size_of::<half_f16>()) as u64,
            "MatmulF16bCubeCL b: handle backs {} bytes, want P*N f16 = {}",
            b_handle.size_in_used(),
            p * n * 2
        );
        assert_binding_derives_units(&out_handle, p, m, "MatmulF16bCubeCL out");

        let wg_x = (m as u32).div_ceil(16).max(1);
        let wg_y = (p as u32).div_ceil(16).max(1);

        // SAFETY: caller guarantees correct buffer sizes (asserted above).
        unsafe {
            matmul_tiled_f16b_f32::launch_unchecked::<R>(
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

#[cfg(all(test, feature = "edlm_gpu"))]
mod tests {
    use super::*;
    use crate::ActiveRuntime;
    use crate::cubecl_runtime::{CubeCLContext, create_f32, read_f32};

    struct Lcg(u64);
    impl Lcg {
        fn next_f32(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        }
    }

    /// CPU reference: the same f16-rounded weights, f32 accumulation in the
    /// plain k order (the accumulation ORDER differs from the tiled kernel —
    /// tolerance-class, never bit-identity).
    fn cpu_matmul(a: &[f32], b_f16: &[half_f16], m: usize, n: usize, p: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; m * p];
        for i in 0..m {
            for j in 0..p {
                let mut acc = 0.0f32;
                for k in 0..n {
                    acc += a[i * n + k] * f32::from(b_f16[j * n + k]);
                }
                out[i * p + j] = acc;
            }
        }
        out
    }

    fn roundtrip_case(m: usize, n: usize, p: usize) {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let mut r = Lcg(0xA11CE);
        let a: Vec<f32> = (0..m * n).map(|_| r.next_f32() * 0.5).collect();
        let b_f32: Vec<f32> = (0..p * n).map(|_| r.next_f32() * 0.5).collect();
        let b_f16: Vec<half_f16> = b_f32.iter().map(|&v| half_f16::from_f32(v)).collect();

        let a_h = create_f32(&client, &a);
        let b_h = client.create_from_slice(bytemuck::cast_slice::<half_f16, u8>(&b_f16));
        let out_h = client.empty(m * p * core::mem::size_of::<f32>());
        MatmulF16bCubeCL::launch::<ActiveRuntime>(&client, a_h, b_h, out_h.clone(), m, n, p);
        let got = read_f32(&client, out_h).expect("read");
        let want = cpu_matmul(&a, &b_f16, m, n, p);

        // Same products, different accumulation order (tiled k-chunks vs
        // sequential k): tolerance, with slack for the N-iteration depth.
        let worst = got
            .iter()
            .zip(&want)
            .map(|(g, w)| (g - w).abs())
            .fold(0.0f32, f32::max);
        let denom = want.iter().map(|w| w.abs()).fold(0.0f32, f32::max);
        let rel = worst / denom.max(1e-9);
        assert!(
            rel < 1e-4,
            "matmul f16b [{m}x{n}]x[{p}x{n}] rel drift {rel} (abs {worst})"
        );
    }

    #[test]
    fn matmul_f16b_matches_cpu_reference() {
        roundtrip_case(37, 96, 80); // odd M (bounds-check arm)
        roundtrip_case(16, 64, 32); // exact tiles
        roundtrip_case(1, 128, 48); // single row
        roundtrip_case(129, 256, 272); // multi-tile M and P
    }
}
