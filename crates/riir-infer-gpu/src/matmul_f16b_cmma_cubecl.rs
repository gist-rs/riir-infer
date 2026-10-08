//! NVIDIA cooperative-matrix dense GEMM with an f16 weight operand — the
//! eDLM lane's tensor-core projection kernel (Issue 1005 T8 follow-up),
//! generic beyond it: `C[M,P] = A[M,N](f32) × B[P,N](f16)^T`.
//!
//! ## Why
//!
//! [`crate::matmul_f16b_cubecl::MatmulF16bCubeCL`] is the `matmul_tiled_f32`
//! shape — one output per thread, scalar FFMA — the exact shape katgpt-rs
//! Issue 734 T6 measured saturating at ~24% of fp32 peak on the 4090 with the
//! wall at scalar-ALU issue. The only way past it is the tensor cores, and
//! this lane's operands are ALREADY f16 (the f16-resident upload law): the
//! `f16×f16→f32 @ 16×16×16` cooperative-matrix shape is the native fit, with
//! f32 accumulation exact (the f16 GEMV family's numerics law).
//!
//! ## Design
//!
//! [`crate::gemm_ternary_cmma16_cubecl`] verbatim shape (the T6-proven
//! 2×2 sub-tile grid): one 32-thread workgroup (one subgroup — the native
//! cooperative-matrix `units_per_block`) computes a **32×32 output tile**;
//! per K-step it stages `a0/a1` (A rows 0/16, f32→f16 AT THE LOAD) +
//! `b0/b1` (B rows 0/16, f16 direct) into shared, then 4 `cmma::execute`
//! (f16 inputs, f32 accumulators). Each staged tile feeds 2 executes — the
//! T6 lesson that staging ALU work must stay well under the tensor
//! throughput. Dispatch mirrors the f16b kernel's convention (CUBE_POS_X =
//! M-tile, CUBE_POS_Y = P-tile), dims passed EXPLICITLY (the launcher
//! asserts the handles back exactly those sizes — the `.issues/515`
//! binding-derived-shape guard in its manual form).
//!
//! ## Numerics (disclosed faces vs the scalar kernel)
//!
//! The scalar f16b kernel keeps A in f32 through the product; the tensor
//! core consumes f16 A — activations round at ~2^-11 relative, a SECOND
//! rounding face beside the f16 weight rounding. Both faces are covered by
//! the lane's tolerance law (argmax exact + 0.015 on the real model; the
//! tiny-weights pipeline gate re-pinned per posture in
//! `edlm_cubecl::tests`). The GOAT lane (`tests/edlm_matmul_cmma_goat.rs`)
//! measures kernel drift + speedup interleaved (the sequential-A/B law).
//!
//! Kill-switch: `EDLM_GPU_CMMA=0` restores the scalar kernel at the eDLM
//! dispatch sites (`edlm_cubecl`); the kernel itself is independently
//! unit-tested either way.

#![allow(clippy::too_many_arguments)]

#[cfg(feature = "edlm_gpu")]
use cubecl::cmma;

#[cfg(feature = "edlm_gpu")]
use cubecl::prelude::*;

#[cfg(feature = "edlm_gpu")]
use cubecl::server::Handle;

// f16 is the CubeCL primitive type — same import the Issue 655 f16 simdgroup
// kernel uses.
#[cfg(feature = "edlm_gpu")]
use half::f16;

#[cfg(feature = "edlm_gpu")]
use crate::cubecl_runtime::assert_binding_derives_units;

/// Cooperative-matrix tile edge (NVIDIA: 16).
#[cfg(feature = "edlm_gpu")]
const CMMA16: u32 = 16;
/// Row sub-tiles per workgroup (output tile = 2×2 sub-tiles = 32×32).
#[cfg(feature = "edlm_gpu")]
const RT: u32 = 2;
#[cfg(feature = "edlm_gpu")]
const CT: u32 = 2;

/// Cooperative-matrix dense matmul with an f16 B operand:
/// `out[M,P] = a[M,N] × b[P,N]^T` via f16 16×16×16 tensor-core mma, f32
/// accumulators. Out-of-range m/p/k stage ZERO (dead accumulator rows/cols
/// are never written back — the guarded writeback makes them invisible).
#[cfg(feature = "edlm_gpu")]
#[cube(launch_unchecked)]
fn matmul_f16b_cmma_f32(
    a: &[f32],
    b: &[f16],
    out: &mut [f32],
    m: u32,
    n: u32,
    p: u32,
) {
    // X = M-tiles, Y = P-tiles (the f16b kernel's dispatch convention).
    let base_m = CUBE_POS_X * (RT * CMMA16);
    let base_p = CUBE_POS_Y * (CT * CMMA16);

    if base_m >= m || base_p >= p {
        terminate!();
    }

    // Shared f16 staging — one buffer per matrix (cubecl's cmma::from_slice
    // cannot sub-slice, the Issue 645 "Follow-ups" note).
    let mut a0 = Shared::<[f16]>::new_slice(256usize);
    let mut a1 = Shared::<[f16]>::new_slice(256usize);
    let mut b0 = Shared::<[f16]>::new_slice(256usize);
    let mut b1 = Shared::<[f16]>::new_slice(256usize);

    // f32 result staging (cmma::store writes the full tile unconditionally).
    let mut r00 = Shared::<[f32]>::new_slice(256usize);
    let mut r01 = Shared::<[f32]>::new_slice(256usize);
    let mut r10 = Shared::<[f32]>::new_slice(256usize);
    let mut r11 = Shared::<[f32]>::new_slice(256usize);

    // Accumulators (f32 — exact accumulation).
    #[allow(unused_mut)]
    let mut acc00 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator,
        16usize,
        16usize,
        16usize,
        cmma::MatrixLayout::Undefined,
        0.0f32,
    );
    #[allow(unused_mut)]
    let mut acc01 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator,
        16usize,
        16usize,
        16usize,
        cmma::MatrixLayout::Undefined,
        0.0f32,
    );
    #[allow(unused_mut)]
    let mut acc10 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator,
        16usize,
        16usize,
        16usize,
        cmma::MatrixLayout::Undefined,
        0.0f32,
    );
    #[allow(unused_mut)]
    let mut acc11 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator,
        16usize,
        16usize,
        16usize,
        cmma::MatrixLayout::Undefined,
        0.0f32,
    );

    let tid = UNIT_POS; // 0..32

    let num_k = n.div_ceil(CMMA16);
    let mut kt = 0u32;
    while kt < num_k {
        let k_base = kt * CMMA16;

        // ── Stage: 8 elements of each tile per thread (4 × 256 / 32). ──
        // e = row_in * 16 + col_in within each 16×16 tile; col_in indexes k.
        // Statement-form loads (the cube macro's expansion — rvalue `if`
        // does not lower).
        let zero_f = f32::new(0.0f32);
        for s in 0u32..8u32 {
            let e = tid + s * 32u32;
            let row_in = e / CMMA16;
            let k = k_base + (e % CMMA16);
            let k_ok = k < n;

            // A tiles — m rows base_m + {0,16} + row_in, f32→f16 at the load.
            let a_r0 = base_m + row_in;
            if a_r0 < m && k_ok {
                a0[e as usize] = f16::cast_from(a[(a_r0 * n + k) as usize]);
            } else {
                a0[e as usize] = f16::cast_from(zero_f);
            }
            let a_r1 = a_r0 + CMMA16;
            if a_r1 < m && k_ok {
                a1[e as usize] = f16::cast_from(a[(a_r1 * n + k) as usize]);
            } else {
                a1[e as usize] = f16::cast_from(zero_f);
            }

            // B tiles — p rows base_p + {0,16} + row_in (row_in doubles as
            // the p index within the tile), f16 direct. The [p][k] staging
            // read as ColMajor B = the [k][p] matrix (the cmma16 pattern).
            let b_r0 = base_p + row_in;
            if b_r0 < p && k_ok {
                b0[e as usize] = b[(b_r0 * n + k) as usize];
            } else {
                b0[e as usize] = f16::cast_from(zero_f);
            }
            let b_r1 = b_r0 + CMMA16;
            if b_r1 < p && k_ok {
                b1[e as usize] = b[(b_r1 * n + k) as usize];
            } else {
                b1[e as usize] = f16::cast_from(zero_f);
            }
        }

        sync_cube();

        // ── MMA: 4 executes (each staged tile feeds 2). ──
        let ma0 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &a0,
            16,
        );
        let ma1 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &a1,
            16,
        );
        let mb0 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::ColMajor,
            &b0,
            16,
        );
        let mb1 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::ColMajor,
            &b1,
            16,
        );

        cmma::execute::<f16, f16, f32, f32, cmma::Plane>(&ma0, &mb0, &acc00, &acc00);
        cmma::execute::<f16, f16, f32, f32, cmma::Plane>(&ma0, &mb1, &acc01, &acc01);
        cmma::execute::<f16, f16, f32, f32, cmma::Plane>(&ma1, &mb0, &acc10, &acc10);
        cmma::execute::<f16, f16, f32, f32, cmma::Plane>(&ma1, &mb1, &acc11, &acc11);

        // Barrier before the next staging overwrites the shared tiles — the
        // cooperative-matrix loads read shared memory, and a racing thread
        // must not clobber a lane's fragment source.
        sync_cube();

        kt += 1u32;
    }

    // ── Store accumulators to shared, then guarded writeback. ──
    cmma::store(&mut r00, &acc00, 16, cmma::MatrixLayout::RowMajor);
    cmma::store(&mut r01, &acc01, 16, cmma::MatrixLayout::RowMajor);
    cmma::store(&mut r10, &acc10, 16, cmma::MatrixLayout::RowMajor);
    cmma::store(&mut r11, &acc11, 16, cmma::MatrixLayout::RowMajor);
    sync_cube();

    // 32 outputs per thread (4 tiles × 8 elements). Staging layout:
    // r[rsub*2+csub][row_in * 16 + p_in], RowMajor over (m, p).
    for i in 0u32..8u32 {
        let e = tid + i * 32u32;
        let row_in = e / CMMA16;
        let p_in = e % CMMA16;
        let row = base_m + row_in;
        let col = base_p + p_in;
        if row < m && col < p {
            out[(row * p + col) as usize] = r00[e as usize];
        }
        let row16 = row + CMMA16;
        if row16 < m && col < p {
            out[(row16 * p + col) as usize] = r10[e as usize];
        }
        let col16 = col + CMMA16;
        if row < m && col16 < p {
            out[(row * p + col16) as usize] = r01[e as usize];
        }
        if row16 < m && col16 < p {
            out[(row16 * p + col16) as usize] = r11[e as usize];
        }
    }
}

/// Launcher for [`matmul_f16b_cmma_f32`] — the scalar
/// [`crate::matmul_f16b_cubecl::MatmulF16bCubeCL`] drop-in (same argument
/// order, same output layout).
///
/// # Safety
///
/// Buffer handles must have correct sizes (asserted):
/// - `a_handle`: M×N f32 elements
/// - `b_handle`: P×N f16 elements
/// - `out_handle`: M×P f32 elements
#[cfg(feature = "edlm_gpu")]
pub struct MatmulF16bCmmaCubeCL;

#[cfg(feature = "edlm_gpu")]
impl MatmulF16bCmmaCubeCL {
    /// Launch the cooperative-matrix f16-B matmul:
    /// `out[M,P] = a[M,N] × b[P,N]^T`.
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
        // The .issues/515 class: an oversized binding silently changes the
        // kernel's idea of its own shape — guard all three (the f16 B buffer
        // gets the byte-exact manual form, the standard guard assumes f32).
        assert_binding_derives_units(&a_handle, n, m, "MatmulF16bCmmaCubeCL a");
        assert_eq!(
            b_handle.size_in_used(),
            (p * n * core::mem::size_of::<f16>()) as u64,
            "MatmulF16bCmmaCubeCL b: handle backs {} bytes, want P*N f16 = {}",
            b_handle.size_in_used(),
            p * n * 2
        );
        assert_binding_derives_units(&out_handle, p, m, "MatmulF16bCmmaCubeCL out");

        let cubes_x = ((m as u32).div_ceil(RT * CMMA16)).max(1);
        let cubes_y = ((p as u32).div_ceil(CT * CMMA16)).max(1);

        // SAFETY: caller guarantees correct buffer sizes (asserted above).
        unsafe {
            matmul_f16b_cmma_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(cubes_x, cubes_y, 1),
                CubeDim::new_1d(32),
                BufferArg::from_raw_parts(a_handle, m * n),
                BufferArg::from_raw_parts(b_handle, p * n),
                BufferArg::from_raw_parts(out_handle, m * p),
                m as u32,
                n as u32,
                p as u32,
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

    /// CPU reference on the SAME f16-rounded operands the kernel sees
    /// (A rounds to f16 at the kernel's load — the oracle rounds first so
    /// ONLY the accumulation order differs, tolerance-class).
    fn cpu_matmul(a: &[f32], b_f16: &[f16], m: usize, n: usize, p: usize) -> Vec<f32> {
        let a16: Vec<f16> = a.iter().map(|&v| f16::from_f32(v)).collect();
        let mut out = vec![0.0f32; m * p];
        for i in 0..m {
            for j in 0..p {
                let mut acc = 0.0f32;
                for k in 0..n {
                    acc += f32::from(a16[i * n + k]) * f32::from(b_f16[j * n + k]);
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
        let b_f16: Vec<f16> = b_f32.iter().map(|&v| f16::from_f32(v)).collect();

        let a_h = create_f32(&client, &a);
        let b_h = client.create_from_slice(bytemuck::cast_slice::<f16, u8>(&b_f16));
        let out_h = client.empty(m * p * core::mem::size_of::<f32>());
        MatmulF16bCmmaCubeCL::launch::<ActiveRuntime>(&client, a_h, b_h, out_h.clone(), m, n, p);
        let got = read_f32(&client, out_h).expect("read");
        let want = cpu_matmul(&a, &b_f16, m, n, p);

        // Same f16-rounded products, different accumulation order (16-wide
        // tensor-core k-steps vs sequential k): the cmma16 tolerance class
        // (~1e-3), with slack for the N-iteration depth.
        let worst = got
            .iter()
            .zip(&want)
            .map(|(g, w)| (g - w).abs())
            .fold(0.0f32, f32::max);
        let denom = want.iter().map(|w| w.abs()).fold(0.0f32, f32::max);
        let rel = worst / denom.max(1e-9);
        assert!(
            rel < 3e-3,
            "matmul f16b cmma [{m}x{n}]x[{p}x{n}] rel drift {rel} (abs {worst})"
        );
    }

    #[test]
    fn matmul_f16b_cmma_matches_cpu_reference() {
        roundtrip_case(37, 96, 80); // odd M and P (bounds arms)
        roundtrip_case(32, 64, 32); // exact 32×32 tiles
        roundtrip_case(1, 128, 48); // single row
        roundtrip_case(129, 256, 272); // multi-tile M and P, odd edges
        roundtrip_case(96, 47, 80); // partial K tile (n not 16-aligned)
    }
}
