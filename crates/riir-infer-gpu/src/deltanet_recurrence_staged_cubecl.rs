//! DeltaNet multi-token recurrence with threadgroup-staged k/q/v
//! (riir-ai Issue 1004 R1 — the MTPLX 2.12.0 distill's first residual rung).
//!
//! [`crate::deltanet_cubecl::DeltanetRecurrenceMultiTokenCubeCL`] runs one
//! cube per `(head, row)`, one plane wide. The 128 row-cubes of a head all
//! read the SAME k/q vectors from global memory on every token (DRAM once,
//! L2-amplified ~128x), and each read sits on the token loop's dependent
//! chain.
//!
//! This kernel packs `rows` row-planes into one cube and stages a block of
//! `tb` tokens' k/q (plus the cube's `rows` v elements and the head's β/α)
//! into threadgroup memory with cooperative coalesced loads, one barrier per
//! block. The hot loop then reads staging only.
//!
//! **Bit-identity is by construction, not by tolerance:** each plane still
//! owns one state row in 4 registers per lane, and the per-token arithmetic
//! is the identical expression sequence in the identical order, including
//! both `plane_sum` reductions. Staging changes only WHERE an operand is read
//! from, never its value. `tests::staged_is_bit_identical_to_multi_token`
//! pins that against the shipping kernel.
//!
//! Same constraint as the shipping kernel: `head_dim == 128`.

#[cfg(feature = "deltanet_recurrence_smem_staged")]
use cubecl::prelude::*;
#[cfg(feature = "deltanet_recurrence_smem_staged")]
use cubecl::server::Handle;

/// Multi-token recurrence, `rows` row-planes per cube, `tb` tokens staged per
/// block. See the module docs for the bit-identity argument.
#[cfg(feature = "deltanet_recurrence_smem_staged")]
#[cube(launch_unchecked)]
fn deltanet_recurrence_multi_token_staged_f32(
    qkvx: &[f32],
    beta: &[f32],
    decay: &[f32],
    params: &[f32],
    state: &mut [f32],
    output: &mut [f32],
    #[comptime] rows: usize,
    #[comptime] tb: usize,
) {
    let head_dim_f32 = params[0usize];
    let head_dim = head_dim_f32 as u32;
    let n_head = params[1usize] as u32;
    let p = params[2usize] as u32;
    let v_dim = params[3usize] as u32; // n_head * head_dim

    let rows_u = rows as u32;
    let tb_u = tb as u32;

    // One cube per (head, block of `rows` rows); one plane per row.
    let head = CUBE_POS_X;
    let row_base = CUBE_POS_Y * rows_u;
    let local_row = UNIT_POS_Y;
    let row = row_base + local_row;

    // No early `terminate!()`: every unit must reach every `sync_cube()`. The
    // launcher sizes the grid exactly (`n_head` x `head_dim / rows`), so the
    // whole cube is in range or out of range together.
    let in_range = head < n_head && row < head_dim;

    let lane = UNIT_POS_X; // == UNIT_POS_PLANE: a plane is one y-row of 32
    let stride = PLANE_DIM;
    let tid = UNIT_POS; // linear unit index within the cube
    let n_threads = CUBE_DIM;

    let qkvx_stride = 3u32 * v_dim;
    let head_off = head * head_dim;
    let k_base = v_dim + head_off;
    let v_base = 2u32 * v_dim + head_off;
    let row_off = (head * head_dim * head_dim + row * head_dim) as usize;

    let c0 = lane as usize;
    let c1 = (lane + stride) as usize;
    let c2 = (lane + 2u32 * stride) as usize;
    let c3 = (lane + 3u32 * stride) as usize;

    let mut smem_k = Shared::<[f32]>::new_slice(comptime!(tb * 128usize));
    let mut smem_q = Shared::<[f32]>::new_slice(comptime!(tb * 128usize));
    let mut smem_v = Shared::<[f32]>::new_slice(comptime!(tb * rows));
    let mut smem_beta = Shared::<[f32]>::new_slice(tb);
    let mut smem_decay = Shared::<[f32]>::new_slice(tb);

    let mut s0 = f32::new(0.0f32);
    let mut s1 = f32::new(0.0f32);
    let mut s2 = f32::new(0.0f32);
    let mut s3 = f32::new(0.0f32);
    if in_range {
        s0 = state[row_off + c0];
        s1 = state[row_off + c1];
        s2 = state[row_off + c2];
        s3 = state[row_off + c3];
    }

    let scale = f32::new(1.0f32) / head_dim_f32.sqrt();

    let mut t0 = 0u32;
    while t0 < p {
        let remaining = p - t0;
        let n_blk = if remaining < tb_u { remaining } else { tb_u };

        if in_range {
            // ── Cooperative coalesced loads: k/q for n_blk tokens ──
            let n_kq = n_blk * head_dim;
            let mut i = tid;
            while i < n_kq {
                let tt = i / head_dim;
                let c = i - tt * head_dim;
                let t_row = (t0 + tt) * qkvx_stride;
                smem_k[i as usize] = qkvx[(t_row + k_base + c) as usize];
                smem_q[i as usize] = qkvx[(t_row + head_off + c) as usize];
                i += n_threads;
            }
            // ── v: the cube's `rows` elements per token ──
            let n_v = n_blk * rows_u;
            let mut j = tid;
            while j < n_v {
                let tt = j / rows_u;
                let r = j - tt * rows_u;
                smem_v[j as usize] =
                    qkvx[((t0 + tt) * qkvx_stride + v_base + row_base + r) as usize];
                j += n_threads;
            }
            // ── β/α: one per token for this head ──
            if tid < n_blk {
                let s_idx = ((t0 + tid) * n_head + head) as usize;
                smem_beta[tid as usize] = beta[s_idx];
                smem_decay[tid as usize] = decay[s_idx];
            }
        }
        sync_cube();

        if in_range {
            let mut tt = 0u32;
            while tt < n_blk {
                let kq = (tt * head_dim) as usize;
                let beta_val = smem_beta[tt as usize];
                let decay_val = smem_decay[tt as usize];

                let k0 = smem_k[kq + c0];
                let k1 = smem_k[kq + c1];
                let k2 = smem_k[kq + c2];
                let k3 = smem_k[kq + c3];

                // Step 1: decay (registers).
                s0 *= decay_val;
                s1 *= decay_val;
                s2 *= decay_val;
                s3 *= decay_val;

                // Step 2: kv_mem = Σ_c S[row, c] · k[c].
                let acc_k = s0 * k0 + s1 * k1 + s2 * k2 + s3 * k3;
                let kv_mem_row = plane_sum(acc_k);

                // Step 3: delta.
                let v_row = smem_v[(tt * rows_u + local_row) as usize];
                let delta_row = beta_val * (v_row - kv_mem_row);

                // Step 4: rank-1 update (registers).
                s0 += k0 * delta_row;
                s1 += k1 * delta_row;
                s2 += k2 * delta_row;
                s3 += k3 * delta_row;

                // Step 5: out[row] = (Σ_c S[row, c] · q[c]) / sqrt(d).
                let q0 = smem_q[kq + c0];
                let q1 = smem_q[kq + c1];
                let q2 = smem_q[kq + c2];
                let q3 = smem_q[kq + c3];
                let acc_q = s0 * q0 + s1 * q1 + s2 * q2 + s3 * q3;
                let dot = plane_sum(acc_q);

                if lane == 0u32 {
                    output[((t0 + tt) * v_dim + head * head_dim + row) as usize] = dot * scale;
                }

                tt += 1;
            }
        }
        // The next block's loads overwrite the staging this block read.
        sync_cube();

        t0 += tb_u;
    }

    if in_range {
        state[row_off + c0] = s0;
        state[row_off + c1] = s1;
        state[row_off + c2] = s2;
        state[row_off + c3] = s3;
    }
}

/// Launcher for the staged multi-token recurrence (riir-ai Issue 1004 R1).
#[cfg(feature = "deltanet_recurrence_smem_staged")]
pub struct DeltanetRecurrenceStagedCubeCL;

#[cfg(feature = "deltanet_recurrence_smem_staged")]
impl DeltanetRecurrenceStagedCubeCL {
    /// Plane width assumed by the kernel. 32 on Metal and CUDA.
    pub const PLANE: usize = 32;
    /// Columns each lane holds in registers (`head_dim / PLANE`).
    pub const COLS_PER_LANE: usize = 4;
    /// Default row-planes per cube.
    pub const DEFAULT_ROWS: usize = 8;
    /// Default tokens staged per block.
    pub const DEFAULT_TB: usize = 16;
    /// Metal's threadgroup-memory ceiling; the staging must fit under it.
    pub const SMEM_LIMIT_BYTES: usize = 32 * 1024;

    /// Whether the kernel serves this `head_dim` (128 only).
    #[inline]
    #[must_use]
    pub fn supports(head_dim: usize) -> bool {
        head_dim == Self::PLANE * Self::COLS_PER_LANE
    }

    /// Threadgroup bytes the staging takes for a `(rows, tb)` shape.
    #[inline]
    #[must_use]
    pub const fn smem_bytes(rows: usize, tb: usize) -> usize {
        (2 * tb * 128 + tb * rows + 2 * tb) * core::mem::size_of::<f32>()
    }

    /// Whether `(rows, tb)` is a valid shape: `rows` divides 128, the cube
    /// stays within 1024 units, and the staging fits threadgroup memory.
    #[inline]
    #[must_use]
    pub const fn valid_shape(rows: usize, tb: usize) -> bool {
        rows > 0
            && tb > 0
            && 128 % rows == 0
            && rows * Self::PLANE <= 1024
            && Self::smem_bytes(rows, tb) <= Self::SMEM_LIMIT_BYTES
    }

    /// Process ALL `p` tokens of a layer in ONE dispatch at the default
    /// `(rows, tb)` — bit-identical to the shipping multi-token kernel.
    ///
    /// # Safety
    ///
    /// Same contract as
    /// [`crate::deltanet_cubecl::DeltanetRecurrenceMultiTokenCubeCL::launch_with_gpu_handles`].
    #[allow(
        clippy::too_many_arguments,
        reason = "GPU kernel launch/dispatch: many buffer handles are inherent to the fused-kernel interface"
    )]
    pub unsafe fn launch_with_gpu_handles<R: Runtime>(
        client: &ComputeClient<R>,
        qkvx_handle: Handle,
        beta_handle: Handle,
        decay_handle: Handle,
        state_handle: Handle,
        output_handle: Handle,
        n_head: usize,
        head_dim: usize,
        p: usize,
    ) {
        unsafe {
            Self::launch_shape(
                client,
                qkvx_handle,
                beta_handle,
                decay_handle,
                state_handle,
                output_handle,
                n_head,
                head_dim,
                p,
                Self::DEFAULT_ROWS,
                Self::DEFAULT_TB,
            );
        }
    }

    /// [`Self::launch_with_gpu_handles`] at an explicit `(rows, tb)` — the
    /// shape sweep's entry point.
    ///
    /// # Safety
    ///
    /// As [`Self::launch_with_gpu_handles`], plus `valid_shape(rows, tb)`.
    #[allow(
        clippy::too_many_arguments,
        reason = "GPU kernel launch/dispatch: many buffer handles are inherent to the fused-kernel interface"
    )]
    pub unsafe fn launch_shape<R: Runtime>(
        client: &ComputeClient<R>,
        qkvx_handle: Handle,
        beta_handle: Handle,
        decay_handle: Handle,
        state_handle: Handle,
        output_handle: Handle,
        n_head: usize,
        head_dim: usize,
        p: usize,
        rows: usize,
        tb: usize,
    ) {
        debug_assert!(
            Self::supports(head_dim),
            "staged recurrence requires head_dim == {}, got {head_dim}",
            Self::PLANE * Self::COLS_PER_LANE
        );
        debug_assert!(
            Self::valid_shape(rows, tb),
            "invalid staging shape rows={rows} tb={tb}"
        );

        let v_dim = n_head * head_dim;
        let params: [f32; 4] = [head_dim as f32, n_head as f32, p as f32, v_dim as f32];
        let params_handle = crate::params_cache::params_handle(client, f32::as_bytes(&params));
        let qkvx_len = p * 3 * v_dim;
        let state_len = n_head * head_dim * head_dim;
        let output_len = p * v_dim;

        unsafe {
            deltanet_recurrence_multi_token_staged_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(n_head as u32, (head_dim / rows) as u32, 1),
                // x = lane within the plane, y = the cube's row-plane.
                CubeDim::new_2d(Self::PLANE as u32, rows as u32),
                BufferArg::from_raw_parts(qkvx_handle, qkvx_len),
                BufferArg::from_raw_parts(beta_handle, p * n_head),
                BufferArg::from_raw_parts(decay_handle, p * n_head),
                BufferArg::from_raw_parts(params_handle, 4),
                BufferArg::from_raw_parts(state_handle, state_len),
                BufferArg::from_raw_parts(output_handle, output_len),
                rows,
                tb,
            );
        }
    }
}

#[cfg(all(test, feature = "deltanet_recurrence_smem_staged"))]
mod tests {
    use super::*;
    use crate::cubecl_runtime::{ActiveRuntime, CubeCLContext};
    use crate::deltanet_cubecl::DeltanetRecurrenceMultiTokenCubeCL;

    /// Deterministic pseudo-random data in a realistic range (LCG, no RNG dep).
    fn lcg_fill(seed: u64, n: usize, lo: f32, hi: f32) -> Vec<f32> {
        let mut x = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        (0..n)
            .map(|_| {
                x = x
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let u = (x >> 40) as f32 / (1u64 << 24) as f32;
                lo + (hi - lo) * u
            })
            .collect()
    }

    struct Case {
        qkvx: Vec<f32>,
        beta: Vec<f32>,
        decay: Vec<f32>,
        state: Vec<f32>,
    }

    fn case(n_head: usize, p: usize, seed: u64) -> Case {
        let v_dim = n_head * 128;
        Case {
            // l2-normalized-scale q/k (~1/sqrt(128) per element) and O(1) v.
            qkvx: lcg_fill(seed, p * 3 * v_dim, -0.18, 0.18),
            beta: lcg_fill(seed ^ 0xB, p * n_head, 0.05, 0.95),
            decay: lcg_fill(seed ^ 0xD, p * n_head, 0.80, 0.999),
            // A non-zero carried state exercises the register load.
            state: lcg_fill(seed ^ 0x5, n_head * 128 * 128, -0.05, 0.05),
        }
    }

    /// Run one arm; `staged = None` is the shipping multi-token kernel.
    fn run(
        client: &ComputeClient<ActiveRuntime>,
        c: &Case,
        n_head: usize,
        p: usize,
        staged: Option<(usize, usize)>,
    ) -> (Vec<u32>, Vec<u32>) {
        let qkvx = client.create_from_slice(f32::as_bytes(&c.qkvx));
        let beta = client.create_from_slice(f32::as_bytes(&c.beta));
        let decay = client.create_from_slice(f32::as_bytes(&c.decay));
        let state = client.create_from_slice(f32::as_bytes(&c.state));
        let output = client.create_from_slice(f32::as_bytes(&vec![0.0f32; p * n_head * 128]));
        unsafe {
            match staged {
                None => {
                    DeltanetRecurrenceMultiTokenCubeCL::launch_with_gpu_handles::<ActiveRuntime>(
                        client,
                        qkvx,
                        beta,
                        decay,
                        state.clone(),
                        output.clone(),
                        n_head,
                        128,
                        p,
                    )
                }
                Some((rows, tb)) => DeltanetRecurrenceStagedCubeCL::launch_shape::<ActiveRuntime>(
                    client,
                    qkvx,
                    beta,
                    decay,
                    state.clone(),
                    output.clone(),
                    n_head,
                    128,
                    p,
                    rows,
                    tb,
                ),
            }
        }
        let bits = |h| {
            f32::from_bytes(&client.read_one_unchecked(h))
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<u32>>()
        };
        (bits(output), bits(state))
    }

    #[test]
    fn shape_validity() {
        assert!(DeltanetRecurrenceStagedCubeCL::valid_shape(8, 16));
        assert!(DeltanetRecurrenceStagedCubeCL::valid_shape(4, 16));
        assert!(!DeltanetRecurrenceStagedCubeCL::valid_shape(3, 16)); // 128 % 3 != 0
        assert!(!DeltanetRecurrenceStagedCubeCL::valid_shape(64, 1)); // 2048 units
        assert!(!DeltanetRecurrenceStagedCubeCL::valid_shape(8, 32)); // 33.8 KB > 32 KB
        assert_eq!(
            DeltanetRecurrenceStagedCubeCL::smem_bytes(8, 16),
            (4096 + 128 + 32) * 4
        );
    }

    /// G1: bit-identical output AND carried state vs the shipping kernel, over
    /// every shape the sweep uses and token counts that hit a full block, a
    /// partial tail block, a single token, and a sub-block run.
    #[test]
    fn staged_is_bit_identical_to_multi_token() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();
        let n_head = 4; // small but > 1: catches head-indexing mistakes
        for &p in &[1usize, 7, 16, 63, 64, 130] {
            let c = case(n_head, p, 0x1004 + p as u64);
            let (ref_out, ref_state) = run(&client, &c, n_head, p, None);
            assert!(
                ref_out.iter().any(|&b| b != 0),
                "reference output is all zero at p={p}"
            );
            for &(rows, tb) in &[(1usize, 1usize), (4, 16), (8, 16), (8, 8), (16, 8), (2, 24)] {
                let (out, st) = run(&client, &c, n_head, p, Some((rows, tb)));
                let out_diff = out.iter().zip(&ref_out).filter(|(a, b)| a != b).count();
                let st_diff = st.iter().zip(&ref_state).filter(|(a, b)| a != b).count();
                assert_eq!(
                    (out_diff, st_diff),
                    (0, 0),
                    "rows={rows} tb={tb} p={p}: {out_diff} output / {st_diff} state words differ"
                );
            }
        }
    }
}
