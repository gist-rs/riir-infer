//! Plan 622 T1.4 — the per-key-bias cost gate at the FA prefill arm (the
//! row-6 DEFAULT prefill kernel, the arm T1.0 named for the measured claim):
//! the biased-call latency ratio vs the incumbent at league shapes, in
//! RELEASE, GPU exclusive.
//!
//! Three arms, one fixture, interleaved A/B/A/B timing (the
//! sequential-A/B-measures-the-box lesson; medians over 48 interleaved
//! iterations): incumbent (`kb=None`, the dummies) / armed all-zero /
//! armed varying (the Phase-2-class real table). The GATE, settled by
//! measurement twice (2026-10-23): the f32 wire measured 1.09–1.13 by cell
//! (ratchet 1.15); the f16 bias-table lever then LANDED — first cut
//! (branchy bit-manip decode) measured WORSE than f32 (1.24–1.29), the
//! one-instruction `cvt.f32.f16` decode cut the armed tax to
//! **1.013–1.033** by cell over three verdict-stable runs — so BOTH
//! lengths now carry the re-pinned measured-class ratchet **≤ 1.05**
//! (~1.7% headroom over the observed max; every demoted implementation
//! — f32 direct 1.0875+, smem stage 1.1657, bit-manip decode 1.2375 —
//! still reds). Any promotion cites this record; owner-gated.
//!
//! Box provenance is PRINTED, not asserted (the Windows sibling of
//! `bench_preflight.sh` does not run here): quote the `PROVENANCE:` line
//! beside any published number.
//!
//! G4 (launch path alloc-free) is BY CONSTRUCTION at this seam: the armed
//! path adds one `CudaView` slice (a borrow), one `i32`, and two `.arg`
//! pushes to each launcher — no device or host allocation (the counting
//! allocator would measure the harness, not the seam; the T1.2 launchers'
//! diff is the review surface).
//!
//! Run (4090): `cargo test -p riir-infer-gpu --release --features
//! ternary_gemv_cuda_raw,ternary_gemm_batched --test
//! plan622_keybias_fa_cost_gate -- --ignored --nocapture`

#![cfg(all(
    feature = "ternary_gemv_cuda_raw",
    feature = "ternary_gemm_batched",
    feature = "cubecl_runtime",
    not(target_os = "macos")
))]

use cudarc::driver::safe::CudaSlice;
use riir_infer_gpu::cudarc_kernels::attention::KeyBiasLayerArgs;
use riir_infer_gpu::fa_scratch_rows;
use riir_infer_gpu::prefill_cuda_attention::CudaAttnKernels;

/// The FA geometry (the only shape `fa_launch_check` accepts).
const HEAD_DIM: usize = 256;
const N_HEAD: usize = 24;
const N_KV: usize = 4;

const ITERS: usize = 48;
const WARMUP: usize = 4;

fn lcg(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 33) as f32 / (u32::MAX >> 1) as f32) * 0.5 - 0.25
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

#[test]
#[ignore = "cost gate: 4090 workstation, release, GPU exclusive"]
fn fa_kb_cost_gate_league_shapes() {
    if cudarc::driver::safe::CudaContext::new(0).is_err() {
        eprintln!("[skip] no CUDA device");
        return;
    }
    let ctx = cudarc::driver::safe::CudaContext::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();
    let kernels = CudaAttnKernels::new_standalone().expect("compile").0;

    // nvrtc is cached per process, not per crate build — one throwaway
    // launch warms the module compile so the timed arms measure kernels.
    let warm = std::time::Instant::now();

    for &p in &[2048usize, 4096] {
        let base_pos = 64usize;
        let kv_len = base_pos + p;
        let padded = fa_scratch_rows(kv_len);
        let q_len = p * N_HEAD * HEAD_DIM;

        let mut state: u64 = 0x622;
        let query: Vec<f32> = (0..q_len).map(|_| lcg(&mut state)).collect();
        let gate: Vec<f32> = (0..q_len).map(|_| lcg(&mut state)).collect();
        let k32: Vec<f32> = (0..padded * N_KV * HEAD_DIM)
            .map(|_| lcg(&mut state))
            .collect();
        let v32: Vec<f32> = (0..padded * N_KV * HEAD_DIM)
            .map(|_| lcg(&mut state))
            .collect();

        let q_dev = stream.clone_htod(&query).unwrap();
        let g_dev = stream.clone_htod(&gate).unwrap();
        let (kh, vh) = unsafe {
            (
                stream.alloc::<u16>(padded * N_KV * HEAD_DIM).unwrap(),
                stream.alloc::<u16>(padded * N_KV * HEAD_DIM).unwrap(),
            )
        };
        // f32 staging buffers for the conversion pass.
        let k_tmp = stream.clone_htod(&k32).unwrap();
        let v_tmp = stream.clone_htod(&v32).unwrap();
        unsafe {
            kernels
                .launch_kv_f32_to_f16(&stream, &k_tmp, &v_tmp, &kh, &vh, kv_len, padded, N_KV)
                .expect("kv f16");
        }
        let out = stream.alloc_zeros::<f32>(q_len).unwrap();

        // Armed tables: all-zero + a varying one (the real-class shape).
        let mut varying = vec![0.0f32; N_HEAD * kv_len];
        for h in 0..N_HEAD {
            for k in 0..kv_len {
                varying[h * kv_len + k] = ((h * 7 + k) % 11) as f32 * 0.3 - 1.5;
            }
        }
        let bias_zero = stream.clone_htod(&vec![0u16; N_HEAD * kv_len]).unwrap();
        let bias_var = stream
            .clone_htod(
                &varying
                    .iter()
                    .map(|&x| half::f16::from_f32(x).to_bits())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let enabled = stream.clone_htod(&[1i32]).unwrap();

        let launch = |out: &CudaSlice<f32>, kb: Option<&KeyBiasLayerArgs<'_>>| unsafe {
            kernels
                .launch_attention_fa(
                    &stream, &q_dev, &kh, &vh, &g_dev, out, HEAD_DIM, N_HEAD, N_KV, p, base_pos,
                    None, kb,
                )
                .expect("launch fa");
        };

        let zero_kb = KeyBiasLayerArgs {
            bias: bias_zero.slice(0..bias_zero.len()),
            stride: kv_len,
            enabled: &enabled,
        };
        let var_kb = KeyBiasLayerArgs {
            bias: bias_var.slice(0..bias_var.len()),
            stride: kv_len,
            enabled: &enabled,
        };

        // Warm the module compile (nvrtc) OUTSIDE the timed arms.
        launch(&out, None);
        launch(&out, Some(&zero_kb));
        launch(&out, Some(&var_kb));
        stream.synchronize().unwrap();
        eprintln!(
            "[p={p}] module warm (first-call nvrtc) {:?}",
            warm.elapsed()
        );

        // Interleaved A/B/C with ROTATED orderings — the fixed first-arm
        // order lets the box's idle-clock ramp tax one arm every triplet
        // (the sequential-timing lesson, one level down: WHICH arm eats the
        // ramp must not be constant). Iteration i uses rotation i % 3; each
        // arm still runs once per iteration, so the medians stay paired.
        let arms: [u8; 3] = [0, 1, 2]; // 0 = incumbent, 1 = zero, 2 = varying
        let mut t_inc: Vec<f64> = Vec::with_capacity(ITERS);
        let mut t_zero: Vec<f64> = Vec::with_capacity(ITERS);
        let mut t_var: Vec<f64> = Vec::with_capacity(ITERS);
        for _ in 0..WARMUP {
            launch(&out, None);
            launch(&out, Some(&zero_kb));
            launch(&out, Some(&var_kb));
        }
        stream.synchronize().unwrap();
        for i in 0..ITERS {
            for a in arms.iter().copied() {
                let rot = arms[(a as usize + i) % 3];
                let t = std::time::Instant::now();
                match rot {
                    0 => launch(&out, None),
                    1 => launch(&out, Some(&zero_kb)),
                    _ => launch(&out, Some(&var_kb)),
                }
                stream.synchronize().unwrap();
                let dt = t.elapsed().as_secs_f64() * 1e3;
                match rot {
                    0 => t_inc.push(dt),
                    1 => t_zero.push(dt),
                    _ => t_var.push(dt),
                }
            }
        }

        let (m_inc, m_zero, m_var) = (median(&mut t_inc), median(&mut t_zero), median(&mut t_var));
        let r_zero = m_zero / m_inc;
        let r_var = m_var / m_inc;
        eprintln!(
            "[p={p}] PROVENANCE: 4090 (RTX 4090, Windows, release, GPU idle — desktop composition only), rotated-order interleaved medians over {ITERS} iters after {WARMUP} warmup"
        );
        eprintln!(
            "[p={p}] incumbent {m_inc:.3} ms | zero-armed {m_zero:.3} ms (ratio {r_zero:.4}) | varying-armed {m_var:.3} ms (ratio {r_var:.4})"
        );
        // The plan's G2 bar, settled by measurement twice. (1) 2026-10-23,
        // f32 wire, six runs, THREE read-path implementations: the armed tax
        // was ~9–13% at BOTH league lengths (lim-guarded direct read
        // 2048→1.1234 / 4096→1.0875; per-tile smem stage 2048→1.1657 —
        // strictly worse, REVERTED; clamped-unconditional direct read
        // 2048→1.1012..1.1234 / 4096→1.1258..1.1338), ratchet 1.15. (2)
        // SAME DAY, the named demotion lever LANDED — the f16 bias table —
        // and beat its ~6% prediction: the first cut decoded with the
        // branchy bit-manip form and measured WORSE than f32 (1.2375/1.2857
        // — the ~10-op decode sits after the KQ mma where only the loads
        // hoist); the one-instruction `cvt.f32.f16` decode cut the tax to
        // 1.013–1.033 by cell over THREE verdict-stable runs (2048:
        // 1.0186–1.0260; 4096: 1.0129–1.0326). Zero-vs-varying still reads
        // the same addresses (the ±1–2% inter-run spread is the noise
        // floor). Ratchet re-pinned DOWNWARD per the landing law to 1.05:
        // ~1.7% headroom over the measured max, and every demoted
        // implementation still reds (f32 direct 1.0875+, smem stage
        // 1.1657, bit-manip decode 1.2375). The armed posture is the
        // Phase-2 pruned-prefill admission price, never paid on the
        // default path (flag-off byte-identity is the kb gate's law);
        // any promotion cites this record and is OWNER-GATED.
        assert!(
            r_zero <= 1.05,
            "kb cost gate: zero-armed ratio {r_zero:.4} > 1.05 at p={p}"
        );
        assert!(
            r_var <= 1.05,
            "kb cost gate: varying-armed ratio {r_var:.4} > 1.05 at p={p}"
        );
    }
}
