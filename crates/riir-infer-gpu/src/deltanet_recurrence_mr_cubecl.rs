//! DeltaNet multi-token recurrence, multi-plane launch geometry
//! (riir-ai Plan 607 C1 — T2, the 4090 r4w4-class rung's CubeCL twin).
//!
//! [`crate::deltanet_cubecl::DeltanetRecurrenceMultiTokenCubeCL`] launches
//! one cube per `(head, row)` — 32 lanes, one t-serial chain per cube, two
//! `plane_sum` serializations per token ON that chain. The 4090 lane's
//! multi-row ladder (Issue 904, `prefill_cuda_deltanet.rs` REC_MR_GEOMETRIES)
//! packs independent chains per scheduling group to hide exactly that
//! latency; B1 found zero `recmr` analog in the CubeCL lane.
//!
//! This kernel packs `planes` row-planes into one cube (`CubeDim` =
//! `PLANE × planes`), one plane (simdgroup) per state row — B1's geometry:
//! `CubeCount::Static(n_head, head_dim / planes, 1)`, ladder `{1, 2, 4, 8}`,
//! `1` = the incumbent's shape. No smem, no barriers (the CUDA ladder is
//! launch-geometry only; so is this).
//!
//! **Bit-identity is by construction:** each plane's chain is the incumbent's
//! op sequence verbatim (same lane-owned columns, same `plane_sum` reductions,
//! same order) — merging planes changes only WHICH row a plane owns and how
//! many planes share a cube. No cross-plane reduction, no new cube barrier,
//! no reorder. On SIMT hardware each unit executes its own plane's chain, so
//! the per-warp instruction stream is byte-for-byte the incumbent's; the
//! probed axes are threadgroup granularity (32 → 64/128/256 lanes per cube)
//! and per-cube k/q/β/α load sharing (B1's secondary CUDA win). The literal
//! CUDA r-axis (multiple rows per LANE, `#[unroll]`-interleaved in one
//! stream) is the named follow-up arm if G2 shows the wall is per-chain
//! latency — not built here (C1 = B1's spec).
//!
//! **Scoring is an Ultra cell** (Plan 607 D2; the M3 keeps the light half —
//! this module's G1 correctness test IS the M3-side deliverable). The env
//! arm defaults OFF; promotion to the CUDA lane's default-ON shape is E1,
//! GOAT-gated.
//!
//! Same constraint as the shipping kernel: `head_dim == 128`.

#[cfg(all(
    feature = "cubecl_runtime",
    feature = "ternary_deltanet_chunked_prefill"
))]
use cubecl::prelude::*;
#[cfg(all(
    feature = "cubecl_runtime",
    feature = "ternary_deltanet_chunked_prefill"
))]
use cubecl::server::Handle;

/// Multi-token recurrence, `planes` row-planes per cube, one plane per row.
/// See the module docs for the bit-identity argument.
#[cfg(all(
    feature = "cubecl_runtime",
    feature = "ternary_deltanet_chunked_prefill"
))]
#[cube(launch_unchecked)]
fn deltanet_recurrence_multi_token_mr_f32(
    qkvx: &[f32],
    beta: &[f32],
    decay: &[f32],
    params: &[f32],
    state: &mut [f32],
    output: &mut [f32],
    #[comptime] planes: usize,
) {
    let head_dim_f32 = params[0usize];
    let head_dim = head_dim_f32 as u32;
    let n_head = params[1usize] as u32;
    let p = params[2usize] as u32;
    let v_dim = params[3usize] as u32; // n_head * head_dim

    let planes_u = planes as u32;

    // One cube per (head, block of `planes` rows); one plane per row.
    let head = CUBE_POS_X;
    let row = CUBE_POS_Y * planes_u + PLANE_POS;

    // No `sync_cube()` in this kernel (no staging), so an early terminate is
    // safe — and the launcher sizes the grid exactly
    // (`n_head × head_dim / planes`), so out-of-range planes cannot occur for
    // a valid ladder arm; the guard stays as belt-and-braces.
    if head >= n_head {
        terminate!();
    }
    if row >= head_dim {
        terminate!();
    }

    let lane = UNIT_POS_PLANE;
    let stride = PLANE_DIM;

    let qkvx_stride = 3u32 * v_dim; // per-token qkvx row
    let head_off = head * head_dim; // h*d within a q/k/v block
    let k_base = v_dim + head_off; // K block base within a token row
    let v_base = 2u32 * v_dim + head_off;
    // Base of THIS row within the [head_dim × head_dim] state block for this head.
    let row_off = (head * head_dim * head_dim + row * head_dim) as usize;

    // Column indices owned by this lane: lane, lane+32, lane+64, lane+96.
    let c0 = lane as usize;
    let c1 = (lane + stride) as usize;
    let c2 = (lane + 2u32 * stride) as usize;
    let c3 = (lane + 3u32 * stride) as usize;

    // ── Load the state row into registers ONCE — held for all p tokens ──
    let mut s0 = state[row_off + c0];
    let mut s1 = state[row_off + c1];
    let mut s2 = state[row_off + c2];
    let mut s3 = state[row_off + c3];

    let scale = f32::new(1.0f32) / head_dim_f32.sqrt();

    let mut t = 0u32;
    while t < p {
        let t_row = (t * qkvx_stride) as usize;
        let scalar_idx = (t * n_head + head) as usize;
        let beta_val = beta[scalar_idx];
        let decay_val = decay[scalar_idx];

        let k0 = qkvx[t_row + k_base as usize + c0];
        let k1 = qkvx[t_row + k_base as usize + c1];
        let k2 = qkvx[t_row + k_base as usize + c2];
        let k3 = qkvx[t_row + k_base as usize + c3];

        // Step 1: decay (registers).
        s0 *= decay_val;
        s1 *= decay_val;
        s2 *= decay_val;
        s3 *= decay_val;

        // Step 2: retrieve kv_mem = Σ_c S[row, c] · k[c] (plane covers the row).
        let acc_k = s0 * k0 + s1 * k1 + s2 * k2 + s3 * k3;
        let kv_mem_row = plane_sum(acc_k);

        // Step 3: delta.
        let v_row = qkvx[t_row + v_base as usize + row as usize];
        let delta_row = beta_val * (v_row - kv_mem_row);

        // Step 4: rank-1 update (registers).
        s0 += k0 * delta_row;
        s1 += k1 * delta_row;
        s2 += k2 * delta_row;
        s3 += k3 * delta_row;

        // Step 5: read out[row] = (Σ_c S[row, c] · q[c]) / sqrt(d).
        let q0 = qkvx[t_row + head_off as usize + c0];
        let q1 = qkvx[t_row + head_off as usize + c1];
        let q2 = qkvx[t_row + head_off as usize + c2];
        let q3 = qkvx[t_row + head_off as usize + c3];
        let acc_q = s0 * q0 + s1 * q1 + s2 * q2 + s3 * q3;
        let dot = plane_sum(acc_q);

        if lane == 0u32 {
            output[(t * v_dim + head * head_dim + row) as usize] = dot * scale;
        }

        t += 1;
    }

    // ── Single global write pass: the updated row ──
    state[row_off + c0] = s0;
    state[row_off + c1] = s1;
    state[row_off + c2] = s2;
    state[row_off + c3] = s3;
}

/// Dispatches through the multi-plane arm so far — the vacuous guard (the
/// Bench-768 lesson: a toggle that silently never reaches the kernel would
/// otherwise read as a tie).
#[cfg(all(
    feature = "cubecl_runtime",
    feature = "ternary_deltanet_chunked_prefill"
))]
static REC_MR_LAUNCHES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Total multi-plane-recurrence dispatches in this process.
#[cfg(all(
    feature = "cubecl_runtime",
    feature = "ternary_deltanet_chunked_prefill"
))]
#[must_use]
pub fn rec_mr_launch_count() -> usize {
    REC_MR_LAUNCHES.load(std::sync::atomic::Ordering::Relaxed)
}

/// Plan 607 B1 — the rows-per-cube ladder. `1` is the incumbent's shape (the
/// in-ladder control: must stay bit-identical AND behaviorally equivalent).
pub const REC_MR_PLANES_LADDER: [usize; 4] = [1, 2, 4, 8];

const NO_REC_MR_OVERRIDE: usize = usize::MAX - 1;

/// Runtime override for one-binary A/B (`Some(planes)` selects a geometry,
/// `None` forces the incumbent; env/default resolve again only when the
/// override is reset — there is no reset, process-wide by design, matching
/// the CUDA lane's `set_rec_mr`).
#[cfg(all(
    feature = "cubecl_runtime",
    feature = "ternary_deltanet_chunked_prefill"
))]
static REC_MR_OVERRIDE: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(NO_REC_MR_OVERRIDE);

/// Env arm `RIIR_GDN_REC_MR` — DEFAULT-OFF (Plan 607 C1: Phase C arms are
/// opt-in; the CUDA lane's default-ON shape is the E1 promotion target, not
/// the C1 posture). `1..=4` selects [`REC_MR_PLANES_LADDER`]`[n - 1]`;
/// `0|off|false` (and unset) is the incumbent; garbage fails OPEN to the
/// incumbent with a warning (the Bench-896 protocol).
#[cfg(all(
    feature = "cubecl_runtime",
    feature = "ternary_deltanet_chunked_prefill"
))]
static REC_MR_ENV: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();

#[cfg(all(
    feature = "cubecl_runtime",
    feature = "ternary_deltanet_chunked_prefill"
))]
pub(crate) fn rec_mr_planes() -> Option<usize> {
    let over = REC_MR_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed);
    if over != NO_REC_MR_OVERRIDE {
        // `usize::MAX` = the force-off sentinel; anything else is a planes
        // value (the caller validated it at set time).
        return (over != usize::MAX).then_some(over);
    }
    *REC_MR_ENV.get_or_init(|| match std::env::var("RIIR_GDN_REC_MR") {
        Err(_) => None,
        Ok(s) => {
            let t = s.trim().to_lowercase();
            if matches!(t.as_str(), "0" | "off" | "false") {
                None
            } else {
                match t.parse::<usize>() {
                    Ok(n) if (1..=REC_MR_PLANES_LADDER.len()).contains(&n) => {
                        Some(REC_MR_PLANES_LADDER[n - 1])
                    }
                    _ => {
                        eprintln!(
                            "[rec-mr] RIIR_GDN_REC_MR={s:?} is not an arm in 1..={} nor off/false — failing OPEN to the incumbent kernel",
                            REC_MR_PLANES_LADDER.len()
                        );
                        None
                    }
                }
            }
        }
    })
}

/// Runtime override for one-binary A/B (`Some(planes)` selects a geometry,
/// `None` forces the incumbent). Invalid geometries are refused loud — a
/// silently-ignored override would poison an A/B.
#[cfg(all(
    feature = "cubecl_runtime",
    feature = "ternary_deltanet_chunked_prefill"
))]
pub fn set_rec_mr(planes: Option<usize>) {
    match planes {
        Some(p) => {
            assert!(
                matches!(p, 1 | 2 | 4 | 8),
                "set_rec_mr: {p} is not a REC_MR_PLANES_LADDER entry"
            );
            REC_MR_OVERRIDE.store(p, std::sync::atomic::Ordering::Relaxed);
        }
        None => REC_MR_OVERRIDE.store(usize::MAX, std::sync::atomic::Ordering::Relaxed),
    }
}

/// Launcher for the multi-plane recurrence (Plan 607 C1).
#[cfg(all(
    feature = "cubecl_runtime",
    feature = "ternary_deltanet_chunked_prefill"
))]
pub struct DeltanetRecurrenceMultiTokenMRCubeCL;

#[cfg(all(
    feature = "cubecl_runtime",
    feature = "ternary_deltanet_chunked_prefill"
))]
impl DeltanetRecurrenceMultiTokenMRCubeCL {
    /// Plane width assumed by the kernel. 32 on Metal and CUDA.
    pub const PLANE: usize = 32;
    /// Columns each lane holds in registers (`head_dim / PLANE`).
    pub const COLS_PER_LANE: usize = 4;

    /// Whether the kernel serves this `head_dim` (128 only — same
    /// register-blocked unroll as the incumbent multi-token kernel).
    #[inline]
    #[must_use]
    pub fn supports(head_dim: usize) -> bool {
        head_dim == Self::PLANE * Self::COLS_PER_LANE
    }

    /// Whether `planes` is a ladder arm (each divides `head_dim == 128`, and
    /// `planes * PLANE ≤ 1024` holds trivially).
    #[inline]
    #[must_use]
    pub const fn valid_planes(planes: usize) -> bool {
        matches!(planes, 1 | 2 | 4 | 8)
    }

    /// Process ALL `p` tokens of a layer in ONE dispatch at `planes` rows per
    /// cube — bit-identical to the incumbent multi-token kernel at every
    /// ladder arm (each plane's chain IS the incumbent's chain; only the
    /// cube geometry differs).
    ///
    /// # Safety
    ///
    /// Same contract as
    /// [`crate::deltanet_cubecl::DeltanetRecurrenceMultiTokenCubeCL::launch_with_gpu_handles`],
    /// plus `valid_planes(planes)`.
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
        planes: usize,
    ) {
        debug_assert!(
            Self::supports(head_dim),
            "multi-plane recurrence requires head_dim == {}, got {head_dim}",
            Self::PLANE * Self::COLS_PER_LANE
        );
        debug_assert!(
            Self::valid_planes(planes),
            "multi-plane recurrence requires a REC_MR_PLANES_LADDER arm, got {planes}"
        );

        let v_dim = n_head * head_dim;
        let params: [f32; 4] = [head_dim as f32, n_head as f32, p as f32, v_dim as f32];
        let params_handle = crate::params_cache::params_handle(client, f32::as_bytes(&params));
        let qkvx_len = p * 3 * v_dim;
        let state_len = n_head * head_dim * head_dim;
        let output_len = p * v_dim;

        REC_MR_LAUNCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        unsafe {
            deltanet_recurrence_multi_token_mr_f32::launch_unchecked::<R>(
                client,
                // One cube per (head, block of `planes` rows); one plane per row.
                CubeCount::Static(n_head as u32, (head_dim / planes) as u32, 1),
                CubeDim::new_1d((Self::PLANE * planes) as u32),
                BufferArg::from_raw_parts(qkvx_handle, qkvx_len),
                BufferArg::from_raw_parts(beta_handle, p * n_head),
                BufferArg::from_raw_parts(decay_handle, p * n_head),
                BufferArg::from_raw_parts(params_handle, 4),
                BufferArg::from_raw_parts(state_handle, state_len),
                BufferArg::from_raw_parts(output_handle, output_len),
                planes,
            );
        }
    }
}

#[cfg(all(
    test,
    feature = "cubecl_runtime",
    feature = "ternary_deltanet_chunked_prefill",
    any(feature = "ternary_gemv", feature = "deltanet_inference")
))]
mod tests {
    use super::*;
    use crate::cubecl_runtime::{ActiveRuntime, CubeCLContext};
    use crate::deltanet_cubecl::DeltanetRecurrenceMultiTokenCubeCL;

    /// Deterministic pseudo-random data in a realistic range (LCG, no RNG dep)
    /// — the staged twin's fixture recipe, kept identical so the two twins'
    /// G1 evidence stays comparable.
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

    /// The standard case (staged twin's ranges) and the B1 G1 case set's
    /// adversarial arm: beta pushed toward 1 (aggressive delta) and decay
    /// edges (near-0 forget / near-1 hold) over a non-zero carried state.
    fn case(n_head: usize, p: usize, seed: u64, adversarial: bool) -> Case {
        let v_dim = n_head * 128;
        if adversarial {
            Case {
                qkvx: lcg_fill(seed, p * 3 * v_dim, -0.35, 0.35),
                beta: lcg_fill(seed ^ 0xB, p * n_head, 0.95, 1.0),
                decay: {
                    // Half the scalars near 0 (hard forget), half near 1 (hold).
                    let d = lcg_fill(seed ^ 0xD, p * n_head, 0.0, 1.0);
                    d.into_iter()
                        .map(|x| {
                            if x < 0.5 {
                                1e-3 + x * 0.01
                            } else {
                                0.999 - x * 0.001
                            }
                        })
                        .collect()
                },
                state: lcg_fill(seed ^ 0x5, n_head * 128 * 128, -0.5, 0.5),
            }
        } else {
            Case {
                // l2-normalized-scale q/k (~1/sqrt(128) per element), O(1) v.
                qkvx: lcg_fill(seed, p * 3 * v_dim, -0.18, 0.18),
                beta: lcg_fill(seed ^ 0xB, p * n_head, 0.05, 0.95),
                decay: lcg_fill(seed ^ 0xD, p * n_head, 0.80, 0.999),
                // A non-zero carried state exercises the register load (the
                // B1 case set's nonzero-base_pos-decay-states arm).
                state: lcg_fill(seed ^ 0x5, n_head * 128 * 128, -0.05, 0.05),
            }
        }
    }

    /// Run one arm; `mr = None` is the shipping multi-token kernel.
    fn run(
        client: &ComputeClient<ActiveRuntime>,
        c: &Case,
        n_head: usize,
        p: usize,
        mr: Option<usize>,
    ) -> (Vec<u32>, Vec<u32>) {
        let qkvx = client.create_from_slice(f32::as_bytes(&c.qkvx));
        let beta = client.create_from_slice(f32::as_bytes(&c.beta));
        let decay = client.create_from_slice(f32::as_bytes(&c.decay));
        let state = client.create_from_slice(f32::as_bytes(&c.state));
        let output = client.create_from_slice(f32::as_bytes(&vec![0.0f32; p * n_head * 128]));
        unsafe {
            match mr {
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
                Some(planes) => {
                    DeltanetRecurrenceMultiTokenMRCubeCL::launch_with_gpu_handles::<ActiveRuntime>(
                        client,
                        qkvx,
                        beta,
                        decay,
                        state.clone(),
                        output.clone(),
                        n_head,
                        128,
                        p,
                        planes,
                    )
                }
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
    fn mr_ladder_validity() {
        for planes in REC_MR_PLANES_LADDER {
            assert!(DeltanetRecurrenceMultiTokenMRCubeCL::valid_planes(planes));
            assert_eq!(128 % planes, 0, "ladder arm {planes} must divide head_dim");
        }
        assert!(!DeltanetRecurrenceMultiTokenMRCubeCL::valid_planes(3));
        assert!(!DeltanetRecurrenceMultiTokenMRCubeCL::valid_planes(0));
        assert!(!DeltanetRecurrenceMultiTokenMRCubeCL::valid_planes(16));
        assert!(DeltanetRecurrenceMultiTokenMRCubeCL::supports(128));
        assert!(!DeltanetRecurrenceMultiTokenMRCubeCL::supports(64));
    }

    /// G1 (Plan 607 D1, M3 half): bit-identical output AND carried state vs
    /// the shipping multi-token kernel, over every ladder arm × the B1 case
    /// set — short-p, p not a multiple of anything in particular, nonzero
    /// carried decay states, and the adversarial β/decay arm. `planes = 1`
    /// is the in-ladder control (the incumbent's exact shape).
    #[test]
    fn mr_is_bit_identical_to_multi_token() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();
        let n_head = 4; // small but > 1: catches head-indexing mistakes
        let mut launches_before = rec_mr_launch_count();
        for adversarial in [false, true] {
            for &p in &[1usize, 7, 16, 63, 64, 130] {
                let c = case(n_head, p, 0x607 + p as u64, adversarial);
                let (ref_out, ref_state) = run(&client, &c, n_head, p, None);
                assert!(
                    ref_out.iter().any(|&b| b != 0),
                    "reference output is all zero at p={p} adversarial={adversarial}"
                );
                for planes in REC_MR_PLANES_LADDER {
                    let (out, st) = run(&client, &c, n_head, p, Some(planes));
                    let out_diff = out.iter().zip(&ref_out).filter(|(a, b)| a != b).count();
                    let st_diff = st.iter().zip(&ref_state).filter(|(a, b)| a != b).count();
                    assert_eq!(
                        (out_diff, st_diff),
                        (0, 0),
                        "planes={planes} p={p} adversarial={adversarial}: \
                         {out_diff} output / {st_diff} state words differ"
                    );
                }
            }
        }
        let launches_after = rec_mr_launch_count();
        assert!(
            launches_after > launches_before,
            "the vacuous guard: the G1 sweep must have dispatched the MR kernel"
        );
        launches_before = launches_after;
        let _ = launches_before;
    }

    /// G2 (Plan 607 D2, isolated-kernel half): the incumbent multi-token
    /// kernel vs the MR ladder at the Bonsai GDN layer shape (n_v=48, hd=128),
    /// interleaved median-of-9 — the CUDA twin's probe discipline
    /// (`prefill_cuda_deltanet.rs::test_rec_mr_timing_probe`) ported to the
    /// CubeCL/Metal lane. p=128 is the PRODUCTION launch shape (the league
    /// prefill chunk; the A2 census's frame); p=2048 is the long-launch
    /// reference matching the CUDA twin's table. Inputs upload ONCE per
    /// shape; each timed call is launch + in-order-queue drain (a 4-byte
    /// read of a dedicated handle — wgpu queues are in-order, so the read
    /// waits for every prior enqueue; its fixed ~µs cost is arm-neutral).
    ///
    /// EXCLUSIVE-GPU cell: run with `--ignored --release` on a verified-idle
    /// box (Plan 607's Ultra binding — bmo/amo).
    #[test]
    #[ignore = "kernel-only timing probe - exclusive GPU window required; Ultra-only per Plan 607"]
    fn test_rec_mr_timing_probe() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();
        let (n_v, hd) = (48usize, 128usize);
        let v_dim = n_v * hd;
        // The drain handle: created once, read (4 bytes) after every launch.
        let sync_handle = client.create_from_slice(f32::as_bytes(&[1.0f32]));

        for p in [128usize, 2048usize] {
            const ROUNDS: usize = 9;
            // Same LCG recipe + ranges as the CUDA twin's probe so the two
            // lanes' tables stay comparable.
            let mut seed = 0xDEFA_CE90_4001u64;
            let mut next = move || {
                seed = seed
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                (seed >> 40) as f32 / 16_777_216.0
            };
            let beta: Vec<f32> = (0..p * n_v).map(|_| 0.05 + next() * 0.90).collect();
            let decay: Vec<f32> = (0..p * n_v).map(|_| 0.5 + next() * 0.499).collect();
            let qkvx: Vec<f32> = (0..p * 3 * v_dim).map(|_| next() - 0.5).collect();
            let state: Vec<f32> = (0..n_v * hd * hd).map(|_| next() - 0.5).collect();

            let qkvx_h = client.create_from_slice(f32::as_bytes(&qkvx));
            let beta_h = client.create_from_slice(f32::as_bytes(&beta));
            let decay_h = client.create_from_slice(f32::as_bytes(&decay));
            let state_h = client.create_from_slice(f32::as_bytes(&state));
            let output_h = client.create_from_slice(f32::as_bytes(&vec![0.0f32; p * v_dim]));

            let timed = |arm: Option<usize>| -> f64 {
                let t0 = std::time::Instant::now();
                unsafe {
                    match arm {
                        Some(planes) => {
                            DeltanetRecurrenceMultiTokenMRCubeCL::launch_with_gpu_handles::<
                                ActiveRuntime,
                            >(
                                &client,
                                qkvx_h.clone(),
                                beta_h.clone(),
                                decay_h.clone(),
                                state_h.clone(),
                                output_h.clone(),
                                n_v,
                                hd,
                                p,
                                planes,
                            )
                        }
                        None => DeltanetRecurrenceMultiTokenCubeCL::launch_with_gpu_handles::<
                            ActiveRuntime,
                        >(
                            &client,
                            qkvx_h.clone(),
                            beta_h.clone(),
                            decay_h.clone(),
                            state_h.clone(),
                            output_h.clone(),
                            n_v,
                            hd,
                            p,
                        ),
                    }
                }
                // In-order drain: this read executes only after the launch.
                let _ = f32::from_bytes(&client.read_one_unchecked(sync_handle.clone()));
                t0.elapsed().as_secs_f64() * 1e3
            };

            // Warmup: every arm once, then the baseline hard (JIT + caches).
            for planes in REC_MR_PLANES_LADDER {
                let _ = timed(Some(planes));
            }
            for _ in 0..3 {
                let _ = timed(None);
            }

            let n_arms = REC_MR_PLANES_LADDER.len();
            let mut samples: Vec<Vec<f64>> = vec![Vec::with_capacity(ROUNDS); n_arms + 1];
            for _ in 0..ROUNDS {
                samples[0].push(timed(None)); // baseline interleaved every round
                for (a, &planes) in REC_MR_PLANES_LADDER.iter().enumerate() {
                    samples[a + 1].push(timed(Some(planes)));
                }
            }
            for row in &mut samples {
                row.sort_by(|x, y| x.total_cmp(y));
            }
            let med: Vec<f64> = samples.iter().map(|r| r[ROUNDS / 2]).collect();
            println!(
                "[rec-mr-timing-cubecl] n_v={n_v} hd={hd} p={p} (median of {ROUNDS}, interleaved):"
            );
            println!("  planes=1-equivalent baseline: {:8.3} ms (1.000x)", med[0]);
            for (a, m) in med.iter().enumerate().skip(1) {
                println!(
                    "  planes={:<2}: {:8.3} ms ({:.3}x)",
                    REC_MR_PLANES_LADDER[a - 1],
                    m,
                    med[0] / m,
                );
            }
        }
    }
}
