//! NVIDIA cooperative-matrix (VK_KHR_cooperative_matrix) ternary GEMM for
//! PREFILL — Issue 734 T6.
//!
//! ## Why this kernel exists
//!
//! The workgroup-tiled f32 kernel ([`crate::GemmTernaryTiledCubeCL`]) saturates
//! at ~20 TFLOPS on the 4090 (24% of fp32 peak) — Issue 734 T6's variant sweep
//! measured that bank-conflict fixes (x-remap, stride-33 padding) and deeper
//! register blocking (8×8) do NOT move it: the wall is scalar-ALU instruction
//! issue, not LDS conflicts. The only way past it is the tensor cores.
//!
//! Issue 734 T2 recorded "cubecl-wgpu panics on CoopMma" — that verdict read
//! the **WGSL** compiler (which cannot express cooperative matrices). The 4090
//! box runs the **`wgpu<spirv>`** runtime (the vendored cubecl-wgpu `spirv`
//! feature), whose `cubecl-spirv` compiler FULLY implements
//! `Operation::CoopMma` (Fill/Load/Execute/Store → `OpCooperativeMatrixMulAddKHR`),
//! and the vendored runtime already enables VK_KHR_cooperative_matrix + queries
//! the shapes (the `cmma_probe_734` output). The advertised set on this box:
//!
//! - `f16×f16→f32 @ 16×16×16` ← this kernel (best float precision: 10 mantissa
//!   bits — the weight `sign×scale` product and activations each round at
//!   ~2^-11 relative)
//! - `bf16×bf16→f32 @ 16×16×16`
//! - `i8×i8→i32 @ 16×16×32`, fp8 shapes, tensor addressing (future levers)
//!
//! ## Design
//!
//! One 32-thread workgroup (one subgroup — the native cooperative-matrix
//! `units_per_block`) computes a **32×32 output tile** (R=C=2 sub-tiles of
//! 16×16): per K-step, stage `a0/a1` (weight rows × 16 cols, dequantized to
//! f16) + `b0/b1` (16 tokens × 16 cols, f32→f16) into shared, then 4
//! `cmma::execute` (f16 inputs, f32 accumulators). The 2×2 sub-tile grid
//! halves the staging work per mma (each staged tile feeds 2 executes) —
//! the T6 sweep's lesson that staging ALU work must stay well under the
//! tensor throughput.
//!
//! Numerics: same contract family as the tiled kernel but at f16 input
//! precision — a **relative-error gate** (~1e-3 class), never bit-identity.
//! The f32 accumulation inside the mma is exact.

#![allow(clippy::too_many_arguments)]

#[cfg(feature = "cubecl_runtime")]
use cubecl::cmma;

#[cfg(feature = "cubecl_runtime")]
use cubecl::prelude::*;

#[cfg(feature = "cubecl_runtime")]
use cubecl::server::Handle;

#[cfg(feature = "cubecl_runtime")]
use crate::gemv_ternary_cubecl::TernaryHandle;

// f16 is the CubeCL primitive type — same import the Issue 655 f16 simdgroup
// kernel uses.
#[cfg(feature = "cubecl_runtime")]
use half::f16;

/// Cooperative-matrix tile edge (NVIDIA: 16).
const CMMA16: u32 = 16;
/// Row sub-tiles per workgroup (output tile = RT×CT sub-tiles = 32×32).
const RT: u32 = 2;
/// Token sub-tiles per workgroup.
const CT: u32 = 2;

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------

/// Cooperative-matrix ternary GEMM: `output[p × m] = dequant(W[m × n]) @
/// input[p × n]^T` via f16 16×16×16 tensor-core mma.
///
/// Workgroup = 32 threads (one subgroup). Staging: per K-step each thread
/// writes 8 elements of each of the 4 f16 tiles (a0/a1/b0/b1); the mma block
/// loads 4 matrices from shared and runs 4 executes into 4 f32 accumulators.
/// Out-of-range rows/tokens stage CLAMPED data (their accumulator rows/cols
/// are never written back — the tiled64 kernel's pattern).
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn gemm_ternary_cmma16_f16(
    pos_bits_u32: &[u32],
    neg_bits_u32: &[u32],
    group_scale_f32: &[f32],
    input_batch: &[f32],
    output_batch: &mut [f32],
    blocks64: u32,
    groups_per_row: u32,
    n: u32,
    m: u32,
    p_tokens: u32,
) {
    let words_per_row = blocks64 * 2u32;

    // X = token tiles, Y = row tiles (the swap_ab dispatch pattern — better
    // distribution when m is small).
    let base_tok = CUBE_POS_X * (CT * CMMA16);
    let base_row = CUBE_POS_Y * (RT * CMMA16);

    if base_row >= m || base_tok >= p_tokens {
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
        // e = row_in * 16 + col_in within each 16×16 tile.
        for s in 0u32..8u32 {
            let e = tid + s * 32u32;
            let row_in = e / CMMA16;
            let col = k_base + (e % CMMA16);
            let one = f32::new(1.0f32);
            let zero = f32::new(0.0f32);
            let bitp = col % 32u32;
            let word_off = col / 32u32;

            // A tiles — rows base_row + {0,16} + row_in.
            let row_r0 = base_row + row_in;
            let row_c0 = if row_r0 < m { row_r0 } else { m - 1u32 };
            let posw0 = pos_bits_u32[(row_c0 * words_per_row + word_off) as usize];
            let negw0 = neg_bits_u32[(row_c0 * words_per_row + word_off) as usize];
            let sign0 = select((posw0 >> bitp) & 1u32 != 0u32, one, zero)
                - select((negw0 >> bitp) & 1u32 != 0u32, one, zero);
            let sc0 = group_scale_f32[(row_c0 * groups_per_row + (col / 128u32)) as usize];
            a0[e as usize] = f16::cast_from(sign0 * sc0);

            let row_r1 = row_r0 + CMMA16;
            let row_c1 = if row_r1 < m { row_r1 } else { m - 1u32 };
            let posw1 = pos_bits_u32[(row_c1 * words_per_row + word_off) as usize];
            let negw1 = neg_bits_u32[(row_c1 * words_per_row + word_off) as usize];
            let sign1 = select((posw1 >> bitp) & 1u32 != 0u32, one, zero)
                - select((negw1 >> bitp) & 1u32 != 0u32, one, zero);
            let sc1 = group_scale_f32[(row_c1 * groups_per_row + (col / 128u32)) as usize];
            a1[e as usize] = f16::cast_from(sign1 * sc1);

            // B tiles — tokens base_tok + {0,16} + row_in (row_in doubles as
            // the token index within the tile).
            let tok_c0 = base_tok + row_in;
            let tok_x0 = if tok_c0 < p_tokens { tok_c0 } else { p_tokens - 1u32 };
            b0[e as usize] = f16::cast_from(input_batch[(tok_x0 * n + col) as usize]);

            let tok_c1 = tok_c0 + CMMA16;
            let tok_x1 = if tok_c1 < p_tokens { tok_c1 } else { p_tokens - 1u32 };
            b1[e as usize] = f16::cast_from(input_batch[(tok_x1 * n + col) as usize]);
        }

        sync_cube();

        // ── MMA: 4 executes (each staged tile feeds 2). B is ColMajor over
        //    the [tok][col] layout = X^T — the same convention as the
        //    simdgroup kernels. ──
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
    // r[rsub*2+csub][row_in * 16 + tok_in], RowMajor over (weight row, token).
    for i in 0u32..8u32 {
        let e = tid + i * 32u32;
        let row_in = e / CMMA16;
        let tok_in = e % CMMA16;
        let row = base_row + row_in;
        let tok = base_tok + tok_in;
        if row < m && tok < p_tokens {
            output_batch[(tok * m + row) as usize] = r00[e as usize];
        }
        let row16 = row + CMMA16;
        if row16 < m && tok < p_tokens {
            output_batch[(tok * m + row16) as usize] = r10[e as usize];
        }
        let tok16 = tok + CMMA16;
        if row < m && tok16 < p_tokens {
            output_batch[(tok16 * m + row) as usize] = r01[e as usize];
        }
        if row16 < m && tok16 < p_tokens {
            output_batch[(tok16 * m + row16) as usize] = r11[e as usize];
        }
    }
}

// ---------------------------------------------------------------------------
// K-blocked variant (Issue 734 T6): 4 k-steps per barrier pair.
// ---------------------------------------------------------------------------

/// K-block size — stages 4 k-steps (64 cols) of both matrix families per
/// barrier pair, then runs 16 mma. 4× fewer control barriers per mma than
/// [`gemm_ternary_cmma16_f16`] — tests the barrier-bound hypothesis.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn gemm_ternary_cmma16_f16_kb4(
    pos_bits_u32: &[u32],
    neg_bits_u32: &[u32],
    group_scale_f32: &[f32],
    input_batch: &[f32],
    output_batch: &mut [f32],
    blocks64: u32,
    groups_per_row: u32,
    n: u32,
    m: u32,
    p_tokens: u32,
) {
    let words_per_row = blocks64 * 2u32;

    let base_tok = CUBE_POS_X * (CT * CMMA16);
    let base_row = CUBE_POS_Y * (RT * CMMA16);

    if base_row >= m || base_tok >= p_tokens {
        terminate!();
    }

    // One f16 buffer per family; tiles addressed via `slice` sub-views.
    // Layout: [kk][tile][16×16] — tile (kk, t) at offset (kk*2 + t)*256.
    let mut a_all = Shared::<[f16]>::new_slice(2048usize);
    let mut b_all = Shared::<[f16]>::new_slice(2048usize);

    let mut r00 = Shared::<[f32]>::new_slice(256usize);
    let mut r01 = Shared::<[f32]>::new_slice(256usize);
    let mut r10 = Shared::<[f32]>::new_slice(256usize);
    let mut r11 = Shared::<[f32]>::new_slice(256usize);

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
    let one = f32::new(1.0f32);
    let zero = f32::new(0.0f32);

    let num_kb = n.div_ceil(CMMA16 * 4u32);
    let mut kb = 0u32;
    while kb < num_kb {
        let k_base = kb * (CMMA16 * 4u32);

        // ── Stage A: 2048 elements / 32 threads = 64 each. ──
        for s in 0u32..64u32 {
            let e_idx = tid + s * 32u32;
            let kk = e_idx / 512u32;
            let r = (e_idx % 512u32) / 256u32;
            let e = e_idx % 256u32;
            let row_in = e / CMMA16;
            let col = k_base + kk * CMMA16 + (e % CMMA16);
            let row = base_row + r * CMMA16 + row_in;
            let row_c = if row < m { row } else { m - 1u32 };
            if col < n {
                let posw = pos_bits_u32[(row_c * words_per_row + (col / 32u32)) as usize];
                let negw = neg_bits_u32[(row_c * words_per_row + (col / 32u32)) as usize];
                let bitp = col % 32u32;
                let sign = select((posw >> bitp) & 1u32 != 0u32, one, zero)
                    - select((negw >> bitp) & 1u32 != 0u32, one, zero);
                let sc = group_scale_f32[(row_c * groups_per_row + (col / 128u32)) as usize];
                a_all[e_idx as usize] = f16::cast_from(sign * sc);
            } else {
                a_all[e_idx as usize] = f16::cast_from(zero);
            }
        }

        // ── Stage B: 2048 elements / 32 threads = 64 each. ──
        for s in 0u32..64u32 {
            let e_idx = tid + s * 32u32;
            let kk = e_idx / 512u32;
            let c = (e_idx % 512u32) / 256u32;
            let e = e_idx % 256u32;
            let tok_in = e / CMMA16;
            let col = k_base + kk * CMMA16 + (e % CMMA16);
            let tok = base_tok + c * CMMA16 + tok_in;
            let tok_c = if tok < p_tokens { tok } else { p_tokens - 1u32 };
            if col < n {
                b_all[e_idx as usize] = f16::cast_from(input_batch[(tok_c * n + col) as usize]);
            } else {
                b_all[e_idx as usize] = f16::cast_from(zero);
            }
        }

        sync_cube();

        // ── MMA: 4 k-steps × 4 pairs = 16 executes per barrier pair. ──
        let mut kk = 0u32;
        while kk < 4u32 {
            let ao0 = kk * 512u32;
            let ao1 = ao0 + 256u32;
            let bo0 = kk * 512u32;
            let bo1 = bo0 + 256u32;
            let ma0 = cmma::Matrix::<f16>::from_slice(
                cmma::MatrixIdent::A,
                16usize,
                16usize,
                16usize,
                cmma::MatrixLayout::RowMajor,
                a_all.slice(ao0 as usize, (ao0 + 256u32) as usize),
                16,
            );
            let ma1 = cmma::Matrix::<f16>::from_slice(
                cmma::MatrixIdent::A,
                16usize,
                16usize,
                16usize,
                cmma::MatrixLayout::RowMajor,
                a_all.slice(ao1 as usize, (ao1 + 256u32) as usize),
                16,
            );
            let mb0 = cmma::Matrix::<f16>::from_slice(
                cmma::MatrixIdent::B,
                16usize,
                16usize,
                16usize,
                cmma::MatrixLayout::ColMajor,
                b_all.slice(bo0 as usize, (bo0 + 256u32) as usize),
                16,
            );
            let mb1 = cmma::Matrix::<f16>::from_slice(
                cmma::MatrixIdent::B,
                16usize,
                16usize,
                16usize,
                cmma::MatrixLayout::ColMajor,
                b_all.slice(bo1 as usize, (bo1 + 256u32) as usize),
                16,
            );

            cmma::execute::<f16, f16, f32, f32, cmma::Plane>(&ma0, &mb0, &acc00, &acc00);
            cmma::execute::<f16, f16, f32, f32, cmma::Plane>(&ma0, &mb1, &acc01, &acc01);
            cmma::execute::<f16, f16, f32, f32, cmma::Plane>(&ma1, &mb0, &acc10, &acc10);
            cmma::execute::<f16, f16, f32, f32, cmma::Plane>(&ma1, &mb1, &acc11, &acc11);

            kk += 1u32;
        }

        sync_cube();

        kb += 1u32;
    }

    cmma::store(&mut r00, &acc00, 16, cmma::MatrixLayout::RowMajor);
    cmma::store(&mut r01, &acc01, 16, cmma::MatrixLayout::RowMajor);
    cmma::store(&mut r10, &acc10, 16, cmma::MatrixLayout::RowMajor);
    cmma::store(&mut r11, &acc11, 16, cmma::MatrixLayout::RowMajor);
    sync_cube();

    for i in 0u32..8u32 {
        let e = tid + i * 32u32;
        let row_in = e / CMMA16;
        let tok_in = e % CMMA16;
        let row = base_row + row_in;
        let tok = base_tok + tok_in;
        if row < m && tok < p_tokens {
            output_batch[(tok * m + row) as usize] = r00[e as usize];
        }
        let row16 = row + CMMA16;
        if row16 < m && tok < p_tokens {
            output_batch[(tok * m + row16) as usize] = r10[e as usize];
        }
        let tok16 = tok + CMMA16;
        if row < m && tok16 < p_tokens {
            output_batch[(tok16 * m + row) as usize] = r01[e as usize];
        }
        if row16 < m && tok16 < p_tokens {
            output_batch[(tok16 * m + row16) as usize] = r11[e as usize];
        }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Cooperative-matrix ternary GEMM launcher (Issue 734 T6).
///
/// Computes `output_batch[p_tokens × m] = dequant_ternary(weight) @ input_batch
/// [p_tokens × n]^T` via the NVIDIA cooperative-matrix f16 16×16×16 tensor-core
/// path. One 32-thread workgroup per 32×32 output tile.
#[cfg(feature = "cubecl_runtime")]
pub struct GemmTernaryCmma16CubeCL;

#[cfg(feature = "cubecl_runtime")]
impl GemmTernaryCmma16CubeCL {
    /// Launch the cooperative-matrix kernel.
    ///
    /// # Safety
    ///
    /// - `input_handle` must hold `p_tokens × handle.n` f32 elements
    /// - `output_handle` must hold `p_tokens × handle.m` f32 elements
    /// - `p_tokens > 0`
    /// - the device must support cmma `(f16, f16, f32)` at 16×16×16 — call
    ///   [`Self::f16_available`]
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        handle: &TernaryHandle,
        input_handle: Handle,
        output_handle: Handle,
        p_tokens: usize,
    ) {
        let n = handle.n as u32;
        let m = handle.m as u32;
        debug_assert!(
            n.is_multiple_of(CMMA16),
            "n must be a multiple of {CMMA16} (the ternary group size 128 guarantees it)"
        );
        let blocks64 = handle.blocks64 as u32;
        let groups_per_row = handle.groups_per_row as u32;
        let p = p_tokens as u32;

        let num_wg_x = p.div_ceil(CT * CMMA16).max(1);
        let num_wg_y = m.div_ceil(RT * CMMA16).max(1);

        let pos_len = handle.m * handle.blocks64 * 2;
        let neg_len = pos_len;
        let scale_len = handle.m * handle.groups_per_row;

        unsafe {
            gemm_ternary_cmma16_f16::launch_unchecked::<R>(
                client,
                CubeCount::Static(num_wg_x, num_wg_y, 1),
                CubeDim::new_1d(32), // one subgroup per workgroup
                BufferArg::from_raw_parts(handle.pos_bits_u32.clone(), pos_len),
                BufferArg::from_raw_parts(handle.neg_bits_u32.clone(), neg_len),
                BufferArg::from_raw_parts(handle.group_scale_f32.clone(), scale_len),
                BufferArg::from_raw_parts(input_handle, p_tokens * handle.n),
                BufferArg::from_raw_parts(output_handle, p_tokens * handle.m),
                blocks64,
                groups_per_row,
                n,
                m,
                p,
            );
        }
    }

    /// Check whether the device supports cmma `(f16, f16, f32)` at 16×16×16
    /// (the NVIDIA cooperative-matrix signature — NOT Metal's 8×8×8).
    pub fn f16_available<R: Runtime>(client: &ComputeClient<R>) -> bool {
        use cubecl::ir::features::MmaConfig;
        use cubecl::ir::{ElemType, FloatKind};

        client.features().matmul.cmma.contains(&MmaConfig {
            a_type: ElemType::Float(FloatKind::F16).into(),
            b_type: ElemType::Float(FloatKind::F16).into(),
            cd_type: ElemType::Float(FloatKind::F32).into(),
            m: 16,
            n: 16,
            k: 16,
        })
    }

    /// Launch the K-blocked (KB=4) variant — 4× fewer barriers per mma.
    ///
    /// # Safety
    ///
    /// Same contract as [`Self::launch`].
    pub unsafe fn launch_kb4<R: Runtime>(
        client: &ComputeClient<R>,
        handle: &TernaryHandle,
        input_handle: Handle,
        output_handle: Handle,
        p_tokens: usize,
    ) {
        let n = handle.n as u32;
        let m = handle.m as u32;
        debug_assert!(
            n.is_multiple_of(CMMA16),
            "n must be a multiple of {CMMA16} (the ternary group size 128 guarantees it)"
        );
        let blocks64 = handle.blocks64 as u32;
        let groups_per_row = handle.groups_per_row as u32;
        let p = p_tokens as u32;

        let num_wg_x = p.div_ceil(CT * CMMA16).max(1);
        let num_wg_y = m.div_ceil(RT * CMMA16).max(1);

        let pos_len = handle.m * handle.blocks64 * 2;
        let neg_len = pos_len;
        let scale_len = handle.m * handle.groups_per_row;

        unsafe {
            gemm_ternary_cmma16_f16_kb4::launch_unchecked::<R>(
                client,
                CubeCount::Static(num_wg_x, num_wg_y, 1),
                CubeDim::new_1d(32),
                BufferArg::from_raw_parts(handle.pos_bits_u32.clone(), pos_len),
                BufferArg::from_raw_parts(handle.neg_bits_u32.clone(), neg_len),
                BufferArg::from_raw_parts(handle.group_scale_f32.clone(), scale_len),
                BufferArg::from_raw_parts(input_handle, p_tokens * handle.n),
                BufferArg::from_raw_parts(output_handle, p_tokens * handle.m),
                blocks64,
                groups_per_row,
                n,
                m,
                p,
            );
        }
    }

    /// Launch the sg8 variant - 256-thread workgroups (8 subgroups), 128x64
    /// output tiles. Halves activation re-read traffic again vs sg4.
    ///
    /// # Safety
    ///
    /// Same contract as [`Self::launch`].
    pub unsafe fn launch_sg8<R: Runtime>(
        client: &ComputeClient<R>,
        handle: &TernaryHandle,
        input_handle: Handle,
        output_handle: Handle,
        p_tokens: usize,
    ) {
        let n = handle.n as u32;
        let m = handle.m as u32;
        debug_assert!(
            n.is_multiple_of(CMMA16),
            "n must be a multiple of {CMMA16} (the ternary group size 128 guarantees it)"
        );
        let blocks64 = handle.blocks64 as u32;
        let groups_per_row = handle.groups_per_row as u32;
        let p = p_tokens as u32;

        let num_wg_x = p.div_ceil(64).max(1);
        let num_wg_y = m.div_ceil(128).max(1);

        let pos_len = handle.m * handle.blocks64 * 2;
        let neg_len = pos_len;
        let scale_len = handle.m * handle.groups_per_row;

        unsafe {
            gemm_ternary_cmma16_f16_sg8::launch_unchecked::<R>(
                client,
                CubeCount::Static(num_wg_x, num_wg_y, 1),
                CubeDim::new_1d(256), // 8 subgroups
                BufferArg::from_raw_parts(handle.pos_bits_u32.clone(), pos_len),
                BufferArg::from_raw_parts(handle.neg_bits_u32.clone(), neg_len),
                BufferArg::from_raw_parts(handle.group_scale_f32.clone(), scale_len),
                BufferArg::from_raw_parts(input_handle, p_tokens * handle.n),
                BufferArg::from_raw_parts(output_handle, p_tokens * handle.m),
                blocks64,
                groups_per_row,
                n,
                m,
                p,
            );
        }
    }

    /// Launch the sg4 variant - 128-thread workgroups (4 subgroups), 64x64
    /// output tiles. Halves activation re-read traffic + 4x mma per barrier
    /// vs the 32-thread v1.
    ///
    /// # Safety
    ///
    /// Same contract as [`Self::launch`].
    pub unsafe fn launch_sg4<R: Runtime>(
        client: &ComputeClient<R>,
        handle: &TernaryHandle,
        input_handle: Handle,
        output_handle: Handle,
        p_tokens: usize,
    ) {
        let n = handle.n as u32;
        let m = handle.m as u32;
        debug_assert!(
            n.is_multiple_of(CMMA16),
            "n must be a multiple of {CMMA16} (the ternary group size 128 guarantees it)"
        );
        let blocks64 = handle.blocks64 as u32;
        let groups_per_row = handle.groups_per_row as u32;
        let p = p_tokens as u32;

        let num_wg_x = p.div_ceil(64).max(1);
        let num_wg_y = m.div_ceil(64).max(1);

        let pos_len = handle.m * handle.blocks64 * 2;
        let neg_len = pos_len;
        let scale_len = handle.m * handle.groups_per_row;

        unsafe {
            gemm_ternary_cmma16_f16_sg4::launch_unchecked::<R>(
                client,
                CubeCount::Static(num_wg_x, num_wg_y, 1),
                CubeDim::new_1d(128), // 4 subgroups
                BufferArg::from_raw_parts(handle.pos_bits_u32.clone(), pos_len),
                BufferArg::from_raw_parts(handle.neg_bits_u32.clone(), neg_len),
                BufferArg::from_raw_parts(handle.group_scale_f32.clone(), scale_len),
                BufferArg::from_raw_parts(input_handle, p_tokens * handle.n),
                BufferArg::from_raw_parts(output_handle, p_tokens * handle.m),
                blocks64,
                groups_per_row,
                n,
                m,
                p,
            );
        }
    }
}
/// Issue 734 T6 sg4 kernel — 4-subgroup workgroup, 64×64 output tile.
///
/// The v1 kernel's wall (measured): activation re-read traffic at R=2
/// (32-row tiles re-read the [P×n] input m/32 = 544×) + per-k-step
/// barriers with only 4 mma each. This variant quadruples the tile via
/// 4 subgroups (one 16-row A sub-tile each, divergently selected),
/// halving activation traffic AND quadrupling mma per barrier.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn gemm_ternary_cmma16_f16_sg4(
    pos_bits_u32: &[u32],
    neg_bits_u32: &[u32],
    group_scale_f32: &[f32],
    input_batch: &[f32],
    output_batch: &mut [f32],
    blocks64: u32,
    groups_per_row: u32,
    n: u32,
    m: u32,
    p_tokens: u32,
) {
    let words_per_row = blocks64 * 2u32;

    let base_tok = CUBE_POS_X * 64u32;
    let base_row = CUBE_POS_Y * 64u32;
    if base_row >= m || base_tok >= p_tokens {
        terminate!();
    }

    // 4 A tiles (one per row sub-tile) + 4 B tiles (one per token sub-tile).
    let mut a0 = Shared::<[f16]>::new_slice(256usize);
    let mut a1 = Shared::<[f16]>::new_slice(256usize);
    let mut a2 = Shared::<[f16]>::new_slice(256usize);
    let mut a3 = Shared::<[f16]>::new_slice(256usize);
    let mut b0 = Shared::<[f16]>::new_slice(256usize);
    let mut b1 = Shared::<[f16]>::new_slice(256usize);
    let mut b2 = Shared::<[f16]>::new_slice(256usize);
    let mut b3 = Shared::<[f16]>::new_slice(256usize);
    // 16 f32 result tiles (64 rows x 64 tokens).
    let mut r00 = Shared::<[f32]>::new_slice(256usize);
    let mut r01 = Shared::<[f32]>::new_slice(256usize);
    let mut r02 = Shared::<[f32]>::new_slice(256usize);
    let mut r03 = Shared::<[f32]>::new_slice(256usize);
    let mut r10 = Shared::<[f32]>::new_slice(256usize);
    let mut r11 = Shared::<[f32]>::new_slice(256usize);
    let mut r12 = Shared::<[f32]>::new_slice(256usize);
    let mut r13 = Shared::<[f32]>::new_slice(256usize);
    let mut r20 = Shared::<[f32]>::new_slice(256usize);
    let mut r21 = Shared::<[f32]>::new_slice(256usize);
    let mut r22 = Shared::<[f32]>::new_slice(256usize);
    let mut r23 = Shared::<[f32]>::new_slice(256usize);
    let mut r30 = Shared::<[f32]>::new_slice(256usize);
    let mut r31 = Shared::<[f32]>::new_slice(256usize);
    let mut r32 = Shared::<[f32]>::new_slice(256usize);
    let mut r33 = Shared::<[f32]>::new_slice(256usize);

    // Per-thread accumulators: MY subgroup's 4 token sub-tiles (each thread
    // belongs to exactly one sg branch, so 4 accs live at any thread).
    #[allow(unused_mut)]
    let mut acc0 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator,
        16usize, 16usize, 16usize,
        cmma::MatrixLayout::Undefined,
        0.0f32,
    );
    #[allow(unused_mut)]
    let mut acc1 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator,
        16usize, 16usize, 16usize,
        cmma::MatrixLayout::Undefined,
        0.0f32,
    );
    #[allow(unused_mut)]
    let mut acc2 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator,
        16usize, 16usize, 16usize,
        cmma::MatrixLayout::Undefined,
        0.0f32,
    );
    #[allow(unused_mut)]
    let mut acc3 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator,
        16usize, 16usize, 16usize,
        cmma::MatrixLayout::Undefined,
        0.0f32,
    );

    let tid = UNIT_POS; // 0..128
    let one = f32::new(1.0f32);
    let zero = f32::new(0.0f32);

    let num_k = n.div_ceil(CMMA16);
    let mut kt = 0u32;
    while kt < num_k {
        let k_base = kt * CMMA16;

        // ── Stage: 8 tiles × 256 elems / 128 threads = 2 per tile per thread. ──
        // Hand-unrolled per tile (no inner runtime loop).
        let row_in0 = (tid % 128u32) / CMMA16;
        let colbit0 = ((tid % 128u32) % CMMA16) + k_base;
        let row_in1 = (tid % 128u32 + 128u32) / CMMA16;
        let colbit1 = ((tid % 128u32 + 128u32) % CMMA16) + k_base;
        let bitp0 = colbit0 % 32u32;
        let word_off0 = colbit0 / 32u32;
        let bitp1 = colbit1 % 32u32;
        let word_off1 = colbit1 / 32u32;
        // Tile pair 0.
        {
            let row = base_row  + row_in0;
            let row_c = if row < m { row } else { m - 1u32 };
            let posw = pos_bits_u32[(row_c * words_per_row + word_off0) as usize];
            let negw = neg_bits_u32[(row_c * words_per_row + word_off0) as usize];
            let sign = select((posw >> bitp0) & 1u32 != 0u32, one, zero)
                - select((negw >> bitp0) & 1u32 != 0u32, one, zero);
            let sc = group_scale_f32[(row_c * groups_per_row + (colbit0 / 128u32)) as usize];
            a0[(tid % 128u32 ) as usize] = f16::cast_from(sign * sc);
            let tok = base_tok  + row_in0;
            let tok_c = if tok < p_tokens { tok } else { p_tokens - 1u32 };
            b0[(tid % 128u32 ) as usize] = f16::cast_from(input_batch[(tok_c * n + colbit0) as usize]);
        }
        {
            let row = base_row  + row_in1;
            let row_c = if row < m { row } else { m - 1u32 };
            let posw = pos_bits_u32[(row_c * words_per_row + word_off1) as usize];
            let negw = neg_bits_u32[(row_c * words_per_row + word_off1) as usize];
            let sign = select((posw >> bitp1) & 1u32 != 0u32, one, zero)
                - select((negw >> bitp1) & 1u32 != 0u32, one, zero);
            let sc = group_scale_f32[(row_c * groups_per_row + (colbit1 / 128u32)) as usize];
            a0[(tid % 128u32 + 128u32) as usize] = f16::cast_from(sign * sc);
            let tok = base_tok  + row_in1;
            let tok_c = if tok < p_tokens { tok } else { p_tokens - 1u32 };
            b0[(tid % 128u32 + 128u32) as usize] = f16::cast_from(input_batch[(tok_c * n + colbit1) as usize]);
        }
        // Tile pair 1.
        {
            let row = base_row + 16u32 + row_in0;
            let row_c = if row < m { row } else { m - 1u32 };
            let posw = pos_bits_u32[(row_c * words_per_row + word_off0) as usize];
            let negw = neg_bits_u32[(row_c * words_per_row + word_off0) as usize];
            let sign = select((posw >> bitp0) & 1u32 != 0u32, one, zero)
                - select((negw >> bitp0) & 1u32 != 0u32, one, zero);
            let sc = group_scale_f32[(row_c * groups_per_row + (colbit0 / 128u32)) as usize];
            a1[(tid % 128u32 ) as usize] = f16::cast_from(sign * sc);
            let tok = base_tok + 16u32 + row_in0;
            let tok_c = if tok < p_tokens { tok } else { p_tokens - 1u32 };
            b1[(tid % 128u32 ) as usize] = f16::cast_from(input_batch[(tok_c * n + colbit0) as usize]);
        }
        {
            let row = base_row + 16u32 + row_in1;
            let row_c = if row < m { row } else { m - 1u32 };
            let posw = pos_bits_u32[(row_c * words_per_row + word_off1) as usize];
            let negw = neg_bits_u32[(row_c * words_per_row + word_off1) as usize];
            let sign = select((posw >> bitp1) & 1u32 != 0u32, one, zero)
                - select((negw >> bitp1) & 1u32 != 0u32, one, zero);
            let sc = group_scale_f32[(row_c * groups_per_row + (colbit1 / 128u32)) as usize];
            a1[(tid % 128u32 + 128u32) as usize] = f16::cast_from(sign * sc);
            let tok = base_tok + 16u32 + row_in1;
            let tok_c = if tok < p_tokens { tok } else { p_tokens - 1u32 };
            b1[(tid % 128u32 + 128u32) as usize] = f16::cast_from(input_batch[(tok_c * n + colbit1) as usize]);
        }
        // Tile pair 2.
        {
            let row = base_row + 32u32 + row_in0;
            let row_c = if row < m { row } else { m - 1u32 };
            let posw = pos_bits_u32[(row_c * words_per_row + word_off0) as usize];
            let negw = neg_bits_u32[(row_c * words_per_row + word_off0) as usize];
            let sign = select((posw >> bitp0) & 1u32 != 0u32, one, zero)
                - select((negw >> bitp0) & 1u32 != 0u32, one, zero);
            let sc = group_scale_f32[(row_c * groups_per_row + (colbit0 / 128u32)) as usize];
            a2[(tid % 128u32 ) as usize] = f16::cast_from(sign * sc);
            let tok = base_tok + 32u32 + row_in0;
            let tok_c = if tok < p_tokens { tok } else { p_tokens - 1u32 };
            b2[(tid % 128u32 ) as usize] = f16::cast_from(input_batch[(tok_c * n + colbit0) as usize]);
        }
        {
            let row = base_row + 32u32 + row_in1;
            let row_c = if row < m { row } else { m - 1u32 };
            let posw = pos_bits_u32[(row_c * words_per_row + word_off1) as usize];
            let negw = neg_bits_u32[(row_c * words_per_row + word_off1) as usize];
            let sign = select((posw >> bitp1) & 1u32 != 0u32, one, zero)
                - select((negw >> bitp1) & 1u32 != 0u32, one, zero);
            let sc = group_scale_f32[(row_c * groups_per_row + (colbit1 / 128u32)) as usize];
            a2[(tid % 128u32 + 128u32) as usize] = f16::cast_from(sign * sc);
            let tok = base_tok + 32u32 + row_in1;
            let tok_c = if tok < p_tokens { tok } else { p_tokens - 1u32 };
            b2[(tid % 128u32 + 128u32) as usize] = f16::cast_from(input_batch[(tok_c * n + colbit1) as usize]);
        }
        // Tile pair 3.
        {
            let row = base_row + 48u32 + row_in0;
            let row_c = if row < m { row } else { m - 1u32 };
            let posw = pos_bits_u32[(row_c * words_per_row + word_off0) as usize];
            let negw = neg_bits_u32[(row_c * words_per_row + word_off0) as usize];
            let sign = select((posw >> bitp0) & 1u32 != 0u32, one, zero)
                - select((negw >> bitp0) & 1u32 != 0u32, one, zero);
            let sc = group_scale_f32[(row_c * groups_per_row + (colbit0 / 128u32)) as usize];
            a3[(tid % 128u32 ) as usize] = f16::cast_from(sign * sc);
            let tok = base_tok + 48u32 + row_in0;
            let tok_c = if tok < p_tokens { tok } else { p_tokens - 1u32 };
            b3[(tid % 128u32 ) as usize] = f16::cast_from(input_batch[(tok_c * n + colbit0) as usize]);
        }
        {
            let row = base_row + 48u32 + row_in1;
            let row_c = if row < m { row } else { m - 1u32 };
            let posw = pos_bits_u32[(row_c * words_per_row + word_off1) as usize];
            let negw = neg_bits_u32[(row_c * words_per_row + word_off1) as usize];
            let sign = select((posw >> bitp1) & 1u32 != 0u32, one, zero)
                - select((negw >> bitp1) & 1u32 != 0u32, one, zero);
            let sc = group_scale_f32[(row_c * groups_per_row + (colbit1 / 128u32)) as usize];
            a3[(tid % 128u32 + 128u32) as usize] = f16::cast_from(sign * sc);
            let tok = base_tok + 48u32 + row_in1;
            let tok_c = if tok < p_tokens { tok } else { p_tokens - 1u32 };
            b3[(tid % 128u32 + 128u32) as usize] = f16::cast_from(input_batch[(tok_c * n + colbit1) as usize]);
        }

        sync_cube();

        // ── MMA: 4 B fragments (uniform) + divergent A per subgroup. ──
        let mb0 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize, 16usize, 16usize,
            cmma::MatrixLayout::ColMajor,
            &b0,
            16,
        );
        let mb1 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize, 16usize, 16usize,
            cmma::MatrixLayout::ColMajor,
            &b1,
            16,
        );
        let mb2 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize, 16usize, 16usize,
            cmma::MatrixLayout::ColMajor,
            &b2,
            16,
        );
        let mb3 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize, 16usize, 16usize,
            cmma::MatrixLayout::ColMajor,
            &b3,
            16,
        );
        let sg = tid / 32u32;
        if sg == 0u32 {
            let ma = cmma::Matrix::<f16>::from_slice(
                cmma::MatrixIdent::A,
                16usize, 16usize, 16usize,
                cmma::MatrixLayout::RowMajor,
                &a0,
                16,
            );
            cmma::execute::<f16, f16, f32, f32, cmma::Plane>(&ma, &mb0, &acc0, &acc0);
            cmma::execute::<f16, f16, f32, f32, cmma::Plane>(&ma, &mb1, &acc1, &acc1);
            cmma::execute::<f16, f16, f32, f32, cmma::Plane>(&ma, &mb2, &acc2, &acc2);
            cmma::execute::<f16, f16, f32, f32, cmma::Plane>(&ma, &mb3, &acc3, &acc3);
        }
        else if sg == 1u32 {
            let ma = cmma::Matrix::<f16>::from_slice(
                cmma::MatrixIdent::A,
                16usize, 16usize, 16usize,
                cmma::MatrixLayout::RowMajor,
                &a1,
                16,
            );
            cmma::execute::<f16, f16, f32, f32, cmma::Plane>(&ma, &mb0, &acc0, &acc0);
            cmma::execute::<f16, f16, f32, f32, cmma::Plane>(&ma, &mb1, &acc1, &acc1);
            cmma::execute::<f16, f16, f32, f32, cmma::Plane>(&ma, &mb2, &acc2, &acc2);
            cmma::execute::<f16, f16, f32, f32, cmma::Plane>(&ma, &mb3, &acc3, &acc3);
        }
        else if sg == 2u32 {
            let ma = cmma::Matrix::<f16>::from_slice(
                cmma::MatrixIdent::A,
                16usize, 16usize, 16usize,
                cmma::MatrixLayout::RowMajor,
                &a2,
                16,
            );
            cmma::execute::<f16, f16, f32, f32, cmma::Plane>(&ma, &mb0, &acc0, &acc0);
            cmma::execute::<f16, f16, f32, f32, cmma::Plane>(&ma, &mb1, &acc1, &acc1);
            cmma::execute::<f16, f16, f32, f32, cmma::Plane>(&ma, &mb2, &acc2, &acc2);
            cmma::execute::<f16, f16, f32, f32, cmma::Plane>(&ma, &mb3, &acc3, &acc3);
        }
        else if sg == 3u32 {
            let ma = cmma::Matrix::<f16>::from_slice(
                cmma::MatrixIdent::A,
                16usize, 16usize, 16usize,
                cmma::MatrixLayout::RowMajor,
                &a3,
                16,
            );
            cmma::execute::<f16, f16, f32, f32, cmma::Plane>(&ma, &mb0, &acc0, &acc0);
            cmma::execute::<f16, f16, f32, f32, cmma::Plane>(&ma, &mb1, &acc1, &acc1);
            cmma::execute::<f16, f16, f32, f32, cmma::Plane>(&ma, &mb2, &acc2, &acc2);
            cmma::execute::<f16, f16, f32, f32, cmma::Plane>(&ma, &mb3, &acc3, &acc3);
        }

        sync_cube();

        kt += 1u32;
    }

    // ── Store: each sg stores its 4 accs (divergent), then all threads copy out. ──
    {
        let sg = tid / 32u32;
        if sg == 0u32 {
            cmma::store(&mut r00, &acc0, 16, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut r01, &acc1, 16, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut r02, &acc2, 16, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut r03, &acc3, 16, cmma::MatrixLayout::RowMajor);
        }
        else if sg == 1u32 {
            cmma::store(&mut r10, &acc0, 16, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut r11, &acc1, 16, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut r12, &acc2, 16, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut r13, &acc3, 16, cmma::MatrixLayout::RowMajor);
        }
        else if sg == 2u32 {
            cmma::store(&mut r20, &acc0, 16, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut r21, &acc1, 16, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut r22, &acc2, 16, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut r23, &acc3, 16, cmma::MatrixLayout::RowMajor);
        }
        else if sg == 3u32 {
            cmma::store(&mut r30, &acc0, 16, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut r31, &acc1, 16, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut r32, &acc2, 16, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut r33, &acc3, 16, cmma::MatrixLayout::RowMajor);
        }
    }
    sync_cube();

    // Writeback: 16 tiles x 256 = 4096 elems / 128 threads = 32 each.
    // Thread t covers elements [t*32, t*32+32) of the concatenated tiles:
    // tile = E/256 (sg*4+c), rem = E%256.
    for i in 0u32..32u32 {
        let eo = tid * 32u32 + i;
        let tile = eo / 256u32;
        let rem = eo % 256u32;
        let row_in = rem / CMMA16;
        let tok_in = rem % CMMA16;
        let row = base_row + (tile / 4u32) * CMMA16 + row_in;
        let tok = base_tok + (tile % 4u32) * CMMA16 + tok_in;
        if row < m && tok < p_tokens {
            output_batch[(tok * m + row) as usize] = select(tile == 0u32, r00[rem as usize], select(tile == 1u32, r01[rem as usize], select(tile == 2u32, r02[rem as usize], select(tile == 3u32, r03[rem as usize], select(tile == 4u32, r10[rem as usize], select(tile == 5u32, r11[rem as usize], select(tile == 6u32, r12[rem as usize], select(tile == 7u32, r13[rem as usize], select(tile == 8u32, r20[rem as usize], select(tile == 9u32, r21[rem as usize], select(tile == 10u32, r22[rem as usize], select(tile == 11u32, r23[rem as usize], select(tile == 12u32, r30[rem as usize], select(tile == 13u32, r31[rem as usize], select(tile == 14u32, r32[rem as usize], r33[rem as usize])))))))))))))));
        }
    }
}
/// Issue 734 T6 sg4 kernel — 4-subgroup workgroup, 64×64 output tile.
///
/// The v1 kernel's wall (measured): activation re-read traffic at R=2
/// (32-row tiles re-read the [P×n] input m/32 = 544×) + per-k-step
/// barriers with only 4 mma each. This variant quadruples the tile via
/// 4 subgroups (one 16-row A sub-tile each, divergently selected),
/// halving activation traffic AND quadrupling mma per barrier.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn gemm_ternary_cmma16_f16_sg8(
    pos_bits_u32: &[u32],
    neg_bits_u32: &[u32],
    group_scale_f32: &[f32],
    input_batch: &[f32],
    output_batch: &mut [f32],
    blocks64: u32,
    groups_per_row: u32,
    n: u32,
    m: u32,
    p_tokens: u32,
) {
    let words_per_row = blocks64 * 2u32;

    let base_tok = CUBE_POS_X * 64u32;
    let base_row = CUBE_POS_Y * 128u32;
    if base_row >= m || base_tok >= p_tokens {
        terminate!();
    }

    // 4 A tiles (one per row sub-tile) + 4 B tiles (one per token sub-tile).
    let mut a0 = Shared::<[f16]>::new_slice(256usize);
    let mut a1 = Shared::<[f16]>::new_slice(256usize);
    let mut a2 = Shared::<[f16]>::new_slice(256usize);
    let mut a3 = Shared::<[f16]>::new_slice(256usize);
    let mut a4 = Shared::<[f16]>::new_slice(256usize);
    let mut a5 = Shared::<[f16]>::new_slice(256usize);
    let mut a6 = Shared::<[f16]>::new_slice(256usize);
    let mut a7 = Shared::<[f16]>::new_slice(256usize);
    let mut b0 = Shared::<[f16]>::new_slice(256usize);
    let mut b1 = Shared::<[f16]>::new_slice(256usize);
    let mut b2 = Shared::<[f16]>::new_slice(256usize);
    let mut b3 = Shared::<[f16]>::new_slice(256usize);
    // 16 f32 result tiles (64 rows x 64 tokens).
    let mut r00 = Shared::<[f32]>::new_slice(256usize);
    let mut r01 = Shared::<[f32]>::new_slice(256usize);
    let mut r02 = Shared::<[f32]>::new_slice(256usize);
    let mut r03 = Shared::<[f32]>::new_slice(256usize);
    let mut r10 = Shared::<[f32]>::new_slice(256usize);
    let mut r11 = Shared::<[f32]>::new_slice(256usize);
    let mut r12 = Shared::<[f32]>::new_slice(256usize);
    let mut r13 = Shared::<[f32]>::new_slice(256usize);
    let mut r20 = Shared::<[f32]>::new_slice(256usize);
    let mut r21 = Shared::<[f32]>::new_slice(256usize);
    let mut r22 = Shared::<[f32]>::new_slice(256usize);
    let mut r23 = Shared::<[f32]>::new_slice(256usize);
    let mut r30 = Shared::<[f32]>::new_slice(256usize);
    let mut r31 = Shared::<[f32]>::new_slice(256usize);
    let mut r32 = Shared::<[f32]>::new_slice(256usize);
    let mut r33 = Shared::<[f32]>::new_slice(256usize);
    let mut r40 = Shared::<[f32]>::new_slice(256usize);
    let mut r41 = Shared::<[f32]>::new_slice(256usize);
    let mut r42 = Shared::<[f32]>::new_slice(256usize);
    let mut r43 = Shared::<[f32]>::new_slice(256usize);
    let mut r50 = Shared::<[f32]>::new_slice(256usize);
    let mut r51 = Shared::<[f32]>::new_slice(256usize);
    let mut r52 = Shared::<[f32]>::new_slice(256usize);
    let mut r53 = Shared::<[f32]>::new_slice(256usize);
    let mut r60 = Shared::<[f32]>::new_slice(256usize);
    let mut r61 = Shared::<[f32]>::new_slice(256usize);
    let mut r62 = Shared::<[f32]>::new_slice(256usize);
    let mut r63 = Shared::<[f32]>::new_slice(256usize);
    let mut r70 = Shared::<[f32]>::new_slice(256usize);
    let mut r71 = Shared::<[f32]>::new_slice(256usize);
    let mut r72 = Shared::<[f32]>::new_slice(256usize);
    let mut r73 = Shared::<[f32]>::new_slice(256usize);

    // Per-thread accumulators: MY subgroup's 4 token sub-tiles (each thread
    // belongs to exactly one sg branch, so 4 accs live at any thread).
    #[allow(unused_mut)]
    let mut acc0 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator,
        16usize, 16usize, 16usize,
        cmma::MatrixLayout::Undefined,
        0.0f32,
    );
    #[allow(unused_mut)]
    let mut acc1 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator,
        16usize, 16usize, 16usize,
        cmma::MatrixLayout::Undefined,
        0.0f32,
    );
    #[allow(unused_mut)]
    let mut acc2 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator,
        16usize, 16usize, 16usize,
        cmma::MatrixLayout::Undefined,
        0.0f32,
    );
    #[allow(unused_mut)]
    let mut acc3 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator,
        16usize, 16usize, 16usize,
        cmma::MatrixLayout::Undefined,
        0.0f32,
    );

    let tid = UNIT_POS; // 0..128
    let one = f32::new(1.0f32);
    let zero = f32::new(0.0f32);

    let num_k = n.div_ceil(CMMA16);
    let mut kt = 0u32;
    while kt < num_k {
        let k_base = kt * CMMA16;

        // ── Stage: 16 tiles × 256 elems / 256 threads = 1 per tile per thread. ──
        let row_in = tid / CMMA16;
        let colbit = (tid % CMMA16) + k_base;
        let bitp = colbit % 32u32;
        let word_off = colbit / 32u32;
        {
            let row = base_row  + row_in;
            let row_c = if row < m { row } else { m - 1u32 };
            let posw = pos_bits_u32[(row_c * words_per_row + word_off) as usize];
            let negw = neg_bits_u32[(row_c * words_per_row + word_off) as usize];
            let sign = select((posw >> bitp) & 1u32 != 0u32, one, zero)
                - select((negw >> bitp) & 1u32 != 0u32, one, zero);
            let sc = group_scale_f32[(row_c * groups_per_row + (colbit / 128u32)) as usize];
            a0[tid as usize] = f16::cast_from(sign * sc);
            let tok = base_tok  + row_in;
            let tok_c = if tok < p_tokens { tok } else { p_tokens - 1u32 };
            b0[tid as usize] = f16::cast_from(input_batch[(tok_c * n + colbit) as usize]);
        }
        {
            let row = base_row + 16u32 + row_in;
            let row_c = if row < m { row } else { m - 1u32 };
            let posw = pos_bits_u32[(row_c * words_per_row + word_off) as usize];
            let negw = neg_bits_u32[(row_c * words_per_row + word_off) as usize];
            let sign = select((posw >> bitp) & 1u32 != 0u32, one, zero)
                - select((negw >> bitp) & 1u32 != 0u32, one, zero);
            let sc = group_scale_f32[(row_c * groups_per_row + (colbit / 128u32)) as usize];
            a1[tid as usize] = f16::cast_from(sign * sc);
            let tok = base_tok + 16u32 + row_in;
            let tok_c = if tok < p_tokens { tok } else { p_tokens - 1u32 };
            b1[tid as usize] = f16::cast_from(input_batch[(tok_c * n + colbit) as usize]);
        }
        {
            let row = base_row + 32u32 + row_in;
            let row_c = if row < m { row } else { m - 1u32 };
            let posw = pos_bits_u32[(row_c * words_per_row + word_off) as usize];
            let negw = neg_bits_u32[(row_c * words_per_row + word_off) as usize];
            let sign = select((posw >> bitp) & 1u32 != 0u32, one, zero)
                - select((negw >> bitp) & 1u32 != 0u32, one, zero);
            let sc = group_scale_f32[(row_c * groups_per_row + (colbit / 128u32)) as usize];
            a2[tid as usize] = f16::cast_from(sign * sc);
            let tok = base_tok + 32u32 + row_in;
            let tok_c = if tok < p_tokens { tok } else { p_tokens - 1u32 };
            b2[tid as usize] = f16::cast_from(input_batch[(tok_c * n + colbit) as usize]);
        }
        {
            let row = base_row + 48u32 + row_in;
            let row_c = if row < m { row } else { m - 1u32 };
            let posw = pos_bits_u32[(row_c * words_per_row + word_off) as usize];
            let negw = neg_bits_u32[(row_c * words_per_row + word_off) as usize];
            let sign = select((posw >> bitp) & 1u32 != 0u32, one, zero)
                - select((negw >> bitp) & 1u32 != 0u32, one, zero);
            let sc = group_scale_f32[(row_c * groups_per_row + (colbit / 128u32)) as usize];
            a3[tid as usize] = f16::cast_from(sign * sc);
            let tok = base_tok + 48u32 + row_in;
            let tok_c = if tok < p_tokens { tok } else { p_tokens - 1u32 };
            b3[tid as usize] = f16::cast_from(input_batch[(tok_c * n + colbit) as usize]);
        }
        {
            let row = base_row + 4u32 * CMMA16 + row_in;
            let row_c = if row < m { row } else { m - 1u32 };
            let posw = pos_bits_u32[(row_c * words_per_row + word_off) as usize];
            let negw = neg_bits_u32[(row_c * words_per_row + word_off) as usize];
            let sign = select((posw >> bitp) & 1u32 != 0u32, one, zero)
                - select((negw >> bitp) & 1u32 != 0u32, one, zero);
            let sc = group_scale_f32[(row_c * groups_per_row + (colbit / 128u32)) as usize];
            a4[tid as usize] = f16::cast_from(sign * sc);
        }
        {
            let row = base_row + 5u32 * CMMA16 + row_in;
            let row_c = if row < m { row } else { m - 1u32 };
            let posw = pos_bits_u32[(row_c * words_per_row + word_off) as usize];
            let negw = neg_bits_u32[(row_c * words_per_row + word_off) as usize];
            let sign = select((posw >> bitp) & 1u32 != 0u32, one, zero)
                - select((negw >> bitp) & 1u32 != 0u32, one, zero);
            let sc = group_scale_f32[(row_c * groups_per_row + (colbit / 128u32)) as usize];
            a5[tid as usize] = f16::cast_from(sign * sc);
        }
        {
            let row = base_row + 6u32 * CMMA16 + row_in;
            let row_c = if row < m { row } else { m - 1u32 };
            let posw = pos_bits_u32[(row_c * words_per_row + word_off) as usize];
            let negw = neg_bits_u32[(row_c * words_per_row + word_off) as usize];
            let sign = select((posw >> bitp) & 1u32 != 0u32, one, zero)
                - select((negw >> bitp) & 1u32 != 0u32, one, zero);
            let sc = group_scale_f32[(row_c * groups_per_row + (colbit / 128u32)) as usize];
            a6[tid as usize] = f16::cast_from(sign * sc);
        }
        {
            let row = base_row + 7u32 * CMMA16 + row_in;
            let row_c = if row < m { row } else { m - 1u32 };
            let posw = pos_bits_u32[(row_c * words_per_row + word_off) as usize];
            let negw = neg_bits_u32[(row_c * words_per_row + word_off) as usize];
            let sign = select((posw >> bitp) & 1u32 != 0u32, one, zero)
                - select((negw >> bitp) & 1u32 != 0u32, one, zero);
            let sc = group_scale_f32[(row_c * groups_per_row + (colbit / 128u32)) as usize];
            a7[tid as usize] = f16::cast_from(sign * sc);
        }

        sync_cube();

        // ── MMA: 4 B fragments (uniform) + divergent A per subgroup. ──
        let mb0 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize, 16usize, 16usize,
            cmma::MatrixLayout::ColMajor,
            &b0,
            16,
        );
        let mb1 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize, 16usize, 16usize,
            cmma::MatrixLayout::ColMajor,
            &b1,
            16,
        );
        let mb2 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize, 16usize, 16usize,
            cmma::MatrixLayout::ColMajor,
            &b2,
            16,
        );
        let mb3 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize, 16usize, 16usize,
            cmma::MatrixLayout::ColMajor,
            &b3,
            16,
        );
        let sg = tid / 32u32;
        if sg == 0u32 {
            let ma = cmma::Matrix::<f16>::from_slice(
                cmma::MatrixIdent::A,
                16usize, 16usize, 16usize,
                cmma::MatrixLayout::RowMajor,
                &a0,
                16,
            );
            cmma::execute::<f16, f16, f32, f32, cmma::Plane>(&ma, &mb0, &acc0, &acc0);
            cmma::execute::<f16, f16, f32, f32, cmma::Plane>(&ma, &mb1, &acc1, &acc1);
            cmma::execute::<f16, f16, f32, f32, cmma::Plane>(&ma, &mb2, &acc2, &acc2);
            cmma::execute::<f16, f16, f32, f32, cmma::Plane>(&ma, &mb3, &acc3, &acc3);
        }
        else if sg == 1u32 {
            let ma = cmma::Matrix::<f16>::from_slice(
                cmma::MatrixIdent::A,
                16usize, 16usize, 16usize,
                cmma::MatrixLayout::RowMajor,
                &a1,
                16,
            );
            cmma::execute::<f16, f16, f32, f32, cmma::Plane>(&ma, &mb0, &acc0, &acc0);
            cmma::execute::<f16, f16, f32, f32, cmma::Plane>(&ma, &mb1, &acc1, &acc1);
            cmma::execute::<f16, f16, f32, f32, cmma::Plane>(&ma, &mb2, &acc2, &acc2);
            cmma::execute::<f16, f16, f32, f32, cmma::Plane>(&ma, &mb3, &acc3, &acc3);
        }
        else if sg == 2u32 {
            let ma = cmma::Matrix::<f16>::from_slice(
                cmma::MatrixIdent::A,
                16usize, 16usize, 16usize,
                cmma::MatrixLayout::RowMajor,
                &a2,
                16,
            );
            cmma::execute::<f16, f16, f32, f32, cmma::Plane>(&ma, &mb0, &acc0, &acc0);
            cmma::execute::<f16, f16, f32, f32, cmma::Plane>(&ma, &mb1, &acc1, &acc1);
            cmma::execute::<f16, f16, f32, f32, cmma::Plane>(&ma, &mb2, &acc2, &acc2);
            cmma::execute::<f16, f16, f32, f32, cmma::Plane>(&ma, &mb3, &acc3, &acc3);
        }
        else if sg == 3u32 {
            let ma = cmma::Matrix::<f16>::from_slice(
                cmma::MatrixIdent::A,
                16usize, 16usize, 16usize,
                cmma::MatrixLayout::RowMajor,
                &a3,
                16,
            );
            cmma::execute::<f16, f16, f32, f32, cmma::Plane>(&ma, &mb0, &acc0, &acc0);
            cmma::execute::<f16, f16, f32, f32, cmma::Plane>(&ma, &mb1, &acc1, &acc1);
            cmma::execute::<f16, f16, f32, f32, cmma::Plane>(&ma, &mb2, &acc2, &acc2);
            cmma::execute::<f16, f16, f32, f32, cmma::Plane>(&ma, &mb3, &acc3, &acc3);
        }
        else if sg == 4u32 {
            let ma = cmma::Matrix::<f16>::from_slice(
                cmma::MatrixIdent::A,
                16usize, 16usize, 16usize,
                cmma::MatrixLayout::RowMajor,
                &a4,
                16,
            );
            cmma::execute::<f16, f16, f32, f32, cmma::Plane>(&ma, &mb0, &acc0, &acc0);
            cmma::execute::<f16, f16, f32, f32, cmma::Plane>(&ma, &mb1, &acc1, &acc1);
            cmma::execute::<f16, f16, f32, f32, cmma::Plane>(&ma, &mb2, &acc2, &acc2);
            cmma::execute::<f16, f16, f32, f32, cmma::Plane>(&ma, &mb3, &acc3, &acc3);
        }
        else if sg == 5u32 {
            let ma = cmma::Matrix::<f16>::from_slice(
                cmma::MatrixIdent::A,
                16usize, 16usize, 16usize,
                cmma::MatrixLayout::RowMajor,
                &a5,
                16,
            );
            cmma::execute::<f16, f16, f32, f32, cmma::Plane>(&ma, &mb0, &acc0, &acc0);
            cmma::execute::<f16, f16, f32, f32, cmma::Plane>(&ma, &mb1, &acc1, &acc1);
            cmma::execute::<f16, f16, f32, f32, cmma::Plane>(&ma, &mb2, &acc2, &acc2);
            cmma::execute::<f16, f16, f32, f32, cmma::Plane>(&ma, &mb3, &acc3, &acc3);
        }
        else if sg == 6u32 {
            let ma = cmma::Matrix::<f16>::from_slice(
                cmma::MatrixIdent::A,
                16usize, 16usize, 16usize,
                cmma::MatrixLayout::RowMajor,
                &a6,
                16,
            );
            cmma::execute::<f16, f16, f32, f32, cmma::Plane>(&ma, &mb0, &acc0, &acc0);
            cmma::execute::<f16, f16, f32, f32, cmma::Plane>(&ma, &mb1, &acc1, &acc1);
            cmma::execute::<f16, f16, f32, f32, cmma::Plane>(&ma, &mb2, &acc2, &acc2);
            cmma::execute::<f16, f16, f32, f32, cmma::Plane>(&ma, &mb3, &acc3, &acc3);
        }
        else if sg == 7u32 {
            let ma = cmma::Matrix::<f16>::from_slice(
                cmma::MatrixIdent::A,
                16usize, 16usize, 16usize,
                cmma::MatrixLayout::RowMajor,
                &a7,
                16,
            );
            cmma::execute::<f16, f16, f32, f32, cmma::Plane>(&ma, &mb0, &acc0, &acc0);
            cmma::execute::<f16, f16, f32, f32, cmma::Plane>(&ma, &mb1, &acc1, &acc1);
            cmma::execute::<f16, f16, f32, f32, cmma::Plane>(&ma, &mb2, &acc2, &acc2);
            cmma::execute::<f16, f16, f32, f32, cmma::Plane>(&ma, &mb3, &acc3, &acc3);
        }

        sync_cube();

        kt += 1u32;
    }

    // ── Store: each sg stores its 4 accs (divergent), then all threads copy out. ──
    {
        let sg = tid / 32u32;
        if sg == 0u32 {
            cmma::store(&mut r00, &acc0, 16, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut r01, &acc1, 16, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut r02, &acc2, 16, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut r03, &acc3, 16, cmma::MatrixLayout::RowMajor);
        }
        else if sg == 1u32 {
            cmma::store(&mut r10, &acc0, 16, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut r11, &acc1, 16, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut r12, &acc2, 16, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut r13, &acc3, 16, cmma::MatrixLayout::RowMajor);
        }
        else if sg == 2u32 {
            cmma::store(&mut r20, &acc0, 16, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut r21, &acc1, 16, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut r22, &acc2, 16, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut r23, &acc3, 16, cmma::MatrixLayout::RowMajor);
        }
        else if sg == 3u32 {
            cmma::store(&mut r30, &acc0, 16, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut r31, &acc1, 16, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut r32, &acc2, 16, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut r33, &acc3, 16, cmma::MatrixLayout::RowMajor);
        }
        else if sg == 4u32 {
            cmma::store(&mut r40, &acc0, 16, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut r41, &acc1, 16, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut r42, &acc2, 16, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut r43, &acc3, 16, cmma::MatrixLayout::RowMajor);
        }
        else if sg == 5u32 {
            cmma::store(&mut r50, &acc0, 16, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut r51, &acc1, 16, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut r52, &acc2, 16, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut r53, &acc3, 16, cmma::MatrixLayout::RowMajor);
        }
        else if sg == 6u32 {
            cmma::store(&mut r60, &acc0, 16, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut r61, &acc1, 16, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut r62, &acc2, 16, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut r63, &acc3, 16, cmma::MatrixLayout::RowMajor);
        }
        else if sg == 7u32 {
            cmma::store(&mut r70, &acc0, 16, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut r71, &acc1, 16, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut r72, &acc2, 16, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut r73, &acc3, 16, cmma::MatrixLayout::RowMajor);
        }
    }
    sync_cube();

    // Writeback: 16 tiles x 256 = 4096 elems / 128 threads = 32 each.
    // Thread t covers elements [t*32, t*32+32) of the concatenated tiles:
    // tile = E/256 (sg*4+c), rem = E%256.
    for i in 0u32..32u32 {
        let eo = tid * 32u32 + i;
        let tile = eo / 256u32;
        let rem = eo % 256u32;
        let row_in = rem / CMMA16;
        let tok_in = rem % CMMA16;
        let row = base_row + (tile / 4u32) * CMMA16 + row_in;
        let tok = base_tok + (tile % 4u32) * CMMA16 + tok_in;
        if row < m && tok < p_tokens {
            output_batch[(tok * m + row) as usize] = select(tile == 0u32, r00[rem as usize], select(tile == 1u32, r01[rem as usize], select(tile == 2u32, r02[rem as usize], select(tile == 3u32, r03[rem as usize], select(tile == 4u32, r10[rem as usize], select(tile == 5u32, r11[rem as usize], select(tile == 6u32, r12[rem as usize], select(tile == 7u32, r13[rem as usize], select(tile == 8u32, r20[rem as usize], select(tile == 9u32, r21[rem as usize], select(tile == 10u32, r22[rem as usize], select(tile == 11u32, r23[rem as usize], select(tile == 12u32, r30[rem as usize], select(tile == 13u32, r31[rem as usize], select(tile == 14u32, r32[rem as usize], select(tile == 15u32, r33[rem as usize], select(tile == 16u32, r40[rem as usize], select(tile == 17u32, r41[rem as usize], select(tile == 18u32, r42[rem as usize], select(tile == 19u32, r43[rem as usize], select(tile == 20u32, r50[rem as usize], select(tile == 21u32, r51[rem as usize], select(tile == 22u32, r52[rem as usize], select(tile == 23u32, r53[rem as usize], select(tile == 24u32, r60[rem as usize], select(tile == 25u32, r61[rem as usize], select(tile == 26u32, r62[rem as usize], select(tile == 27u32, r63[rem as usize], select(tile == 28u32, r70[rem as usize], select(tile == 29u32, r71[rem as usize], select(tile == 30u32, r72[rem as usize], select(tile == 31u32, r73[rem as usize], r73[rem as usize]))))))))))))))))))))))))))))))));
        }
    }
}
