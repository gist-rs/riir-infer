//! Plan 620 Phase D — the FA prefill si-hook gate (the decode hook's
//! `test_attention_decode_si_flag_off_bit_identity_and_affine`, ported to
//! the `att_pf_fa` class).
//!
//! Six launches over one fixture, the decode test's matrix:
//! `base` (si=None, the dummies) / `flag-off` (session-shaped pointers,
//! disabled) / `identity` (a=1, m=0, inv=1, ARMED) / `affine` (a=2, m=1,
//! inv=0.5 — `a·inv == 1` exactly, so every in-range non-sink score shifts
//! by exactly +1.0) / and the mid-context pair (eager @ q_offset=64 vs the
//! `_dp` graph twin). The sharp teeth are bit-exact: flag-off == base and
//! identity == base (the single-rounding intrinsics make the identity
//! affine exact, and the softmax collapses any ±0.0 from the `+0.0`), and
//! the devpos twin == the eager twin at the same q_offset. The coarse teeth
//! are the CPU twins (plain + affine, the same f16-rounded operands) at a
//! tolerance bar — the FA kernel is f16-operand tolerance-class, so the bar
//! is wide; an O(1) si-algebra error (a mis-indexed dist, a missed sink, a
//! wrong head window) moves scores by O(1) and blows through it
//! decisively, while f16 rounding noise sits orders below it.

#![cfg(all(
    feature = "ternary_gemv_cuda_raw",
    feature = "ternary_gemm_batched",
    feature = "cubecl_runtime",
    not(target_os = "macos")
))]

use cudarc::driver::safe::CudaSlice;
use riir_infer_gpu::cudarc_kernels::attention::SiDecodeLayerArgs;
use riir_infer_gpu::fa_scratch_rows;
use riir_infer_gpu::prefill_cuda_attention::CudaAttnKernels;

/// The FA geometry (the only shape `fa_launch_check` accepts): DKQ=DV=256,
/// GQA exactly 6:1.
const HEAD_DIM: usize = 256;
const N_HEAD: usize = 24;
const N_KV: usize = 4;
/// p = FA_MIN_P: the engagement-predicate boundary this hook is scoped to.
const P: usize = 128;
/// Mid-context: dist > 0 and the kv==0 sink both live.
const BASE_POS: usize = 64;
const KV_LEN: usize = BASE_POS + P;
/// The CPU-twin tolerance bar — the FA kernel's f16-operand/f16-accumulator
/// class; an O(1) si-algebra error is unmissable at it, rounding noise sits
/// ~10x below.
const TOL: f32 = 3e-2;

fn lcg(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 33) as f32 / (u32::MAX >> 1) as f32) * 0.5 - 0.25
}

/// The CPU twin of the FA kernel's output for one posture: f32 softmax over
/// the f16-rounded operands (`cvt.rn` semantics via the `half` crate), the
/// si affine per (qp, kv) when armed, the sigmoid gate in the epilogue.
/// f32 accumulation — the kernel's f16 P/V accumulation is noise the bar
/// absorbs.
///
/// CAUSAL, like the kernel (`lim = q_offset + j + 1`; the twin computes the
/// base-0 posture only, so the limit is `kv <= j`): keys past the limit are
/// masked to -inf (exp → 0). The birth version scored every KV_LEN key
/// unmasked — an O(1) divergence from the kernel at every j > 0 row, caught
/// by the Plan-622 kb gate's port before this twin had ever executed its
/// CPU-twin asserts on a CUDA box (the bit-identity arms were unaffected —
/// they never read this function).
#[allow(clippy::too_many_arguments)]
fn cpu_fa_reference(
    query: &[f32],
    kh: &[f32],
    vh: &[f32],
    gate: &[f32],
    armed: bool,
    a: f32,
    m: f32,
    inv: f32,
) -> Vec<f32> {
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let mut out = vec![0.0f32; P * N_HEAD * HEAD_DIM];
    for j in 0..P {
        for h in 0..N_HEAD {
            let kv_group = h * N_KV / N_HEAD;
            let mut max_score = f32::NEG_INFINITY;
            let mut scores = vec![0.0f32; KV_LEN];
            for (kv, s) in scores.iter_mut().enumerate() {
                let mut dot = 0.0f32;
                let q_off = (j * N_HEAD + h) * HEAD_DIM;
                let k_off = (kv * N_KV + kv_group) * HEAD_DIM;
                for d in 0..HEAD_DIM {
                    dot += query[q_off + d] * kh[k_off + d];
                }
                let raw = dot * scale;
                // Sink carve-out at key 0; dist = (q_offset + j) - kv.
                // Causal mask first (the kernel's lim at q_offset = 0 is
                // j + 1): masked keys underflow exp to exactly 0.
                *s = if kv > j {
                    f32::NEG_INFINITY
                } else {
                    match kv {
                        0 => raw,
                        _ if armed => a * (raw * inv) + m,
                        _ => raw,
                    }
                };
                max_score = max_score.max(*s);
            }
            let mut sum = 0.0f32;
            for s in scores.iter_mut() {
                *s = if s.is_finite() {
                    (*s - max_score).exp()
                } else {
                    0.0
                };
                sum += *s;
            }
            let inv_sum = 1.0 / sum;
            let o_off = (j * N_HEAD + h) * HEAD_DIM;
            for d in 0..HEAD_DIM {
                let mut acc = 0.0f32;
                for kv in 0..KV_LEN {
                    acc += scores[kv] * inv_sum * vh[(kv * N_KV + kv_group) * HEAD_DIM + d];
                }
                let sig = 1.0 / (1.0 + (-gate[o_off + d]).exp());
                out[o_off + d] = acc * sig;
            }
        }
    }
    out
}

fn max_rel(gpu: &[f32], cpu: &[f32]) -> f32 {
    gpu.iter()
        .zip(cpu.iter())
        .map(|(g, c)| (g - c).abs() / c.abs().max(1e-6))
        .fold(0.0f32, f32::max)
}

#[test]
fn test_prefill_fa_si_flag_off_identity_and_affine() {
    let Ok((kernels, stream)) = CudaAttnKernels::new_standalone() else {
        eprintln!("[skip] no CUDA device (CudaAttnKernels::new_standalone failed)");
        return;
    };
    // SAFETY (per launch): every buffer below is allocated to the shape the
    // launcher contract names (the f16 scratch comes from the real
    // conversion pass); head_dim == 256 and n_head/n_kv == 6 hold by the
    // consts. One unsafe block wraps the launch sequence.
    unsafe {
        let q_len = P * N_HEAD * HEAD_DIM;
        let kv_elems = KV_LEN * N_KV * HEAD_DIM;
        let padded = fa_scratch_rows(KV_LEN);

        // Deterministic fixture with real spread.
        let mut st: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut query = vec![0.0f32; q_len];
        let mut kcache = vec![0.0f32; kv_elems];
        let mut vcache = vec![0.0f32; kv_elems];
        let mut gate = vec![0.0f32; q_len];
        for x in query.iter_mut() {
            *x = lcg(&mut st);
        }
        for x in kcache.iter_mut() {
            *x = lcg(&mut st);
        }
        for x in vcache.iter_mut() {
            *x = lcg(&mut st);
        }
        for x in gate.iter_mut() {
            *x = lcg(&mut st);
        }
        // The operands f16-rounded exactly as the kernel reads them (`cvt.rn`).
        let kh_host: Vec<f32> = kcache
            .iter()
            .map(|&x| half::f16::from_f32(x).to_f32())
            .collect();
        let vh_host: Vec<f32> = vcache
            .iter()
            .map(|&x| half::f16::from_f32(x).to_f32())
            .collect();

        let q_dev: CudaSlice<f32> = stream.clone_htod(&query).unwrap();
        let k_dev = stream.clone_htod(&kcache).unwrap();
        let v_dev = stream.clone_htod(&vcache).unwrap();
        let gate_dev = stream.clone_htod(&gate).unwrap();
        let kh = stream.alloc_zeros::<u16>(padded * N_KV * HEAD_DIM).unwrap();
        let vh = stream.alloc_zeros::<u16>(padded * N_KV * HEAD_DIM).unwrap();
        let pos_dev = stream.clone_htod(&[BASE_POS as i32]).unwrap();

        kernels
            .launch_kv_f32_to_f16(&stream, &k_dev, &v_dev, &kh, &vh, KV_LEN, padded, N_KV)
            .expect("kv f16 conversion");

        // Session-shaped buffers: the LUTs cover every dist the causal shape
        // can reach; the affine arm uses the constant map a=2, m=1 with
        // inv=0.5 — a·inv == 1.0 exactly (power-of-two), so the armed score
        // is `raw + 1.0` at every non-sink key.
        let max_dist = KV_LEN + 8;
        let lut_identity: Vec<f32> = (0..=max_dist).flat_map(|_| [1.0f32, 0.0]).collect();
        let lut_affine: Vec<f32> = (0..=max_dist).flat_map(|_| [2.0f32, 1.0]).collect();
        let lut_identity_dev = stream.clone_htod(&lut_identity).unwrap();
        let lut_affine_dev = stream.clone_htod(&lut_affine).unwrap();
        let inv_half_dev = stream.clone_htod(&vec![0.5f32; N_HEAD]).unwrap();
        let inv_identity_dev = stream.clone_htod(&vec![1.0f32; N_HEAD]).unwrap();
        let disabled = stream.clone_htod(&[0i32]).unwrap();
        let enabled = stream.clone_htod(&[1i32]).unwrap();

        let launch_eager =
            |out: &CudaSlice<f32>, base_pos: usize, si: Option<&SiDecodeLayerArgs<'_>>| {
                kernels
                    .launch_attention_fa(
                        &stream, &q_dev, &kh, &vh, &gate_dev, out, HEAD_DIM, N_HEAD, N_KV, P,
                        base_pos, si,
                        // Plan 622 — the kb hook: the incumbent posture (the
                        // Plan-620 si gate owns this harness).
                        None,
                    )
                    .expect("launch_attention_fa");
                stream.synchronize().expect("sync");
                let mut host = vec![0f32; q_len];
                stream.memcpy_dtoh(out, &mut host).unwrap();
                host
            };
        let affine_args = || SiDecodeLayerArgs {
            lut: &lut_affine_dev,
            inv_sigma: inv_half_dev.slice(0..N_HEAD),
            enabled: &enabled,
        };

        let outs: Vec<CudaSlice<f32>> = (0..6)
            .map(|_| stream.alloc_zeros::<f32>(q_len).unwrap())
            .collect();

        let base = launch_eager(&outs[0], 0, None);
        let flag_off = launch_eager(
            &outs[1],
            0,
            Some(&SiDecodeLayerArgs {
                lut: &lut_identity_dev,
                inv_sigma: inv_identity_dev.slice(0..N_HEAD),
                enabled: &disabled,
            }),
        );
        let identity = launch_eager(
            &outs[2],
            0,
            Some(&SiDecodeLayerArgs {
                lut: &lut_identity_dev,
                inv_sigma: inv_identity_dev.slice(0..N_HEAD),
                enabled: &enabled,
            }),
        );
        let affine = launch_eager(&outs[3], 0, Some(&affine_args()));
        // Mid-context runs (q_offset = 64): eager + the devpos graph twin.
        let mid = launch_eager(&outs[4], BASE_POS, Some(&affine_args()));
        kernels
            .launch_attention_fa_devpos(
                &stream,
                &q_dev,
                &kh,
                &vh,
                &gate_dev,
                &outs[5],
                HEAD_DIM,
                N_HEAD,
                N_KV,
                P,
                &pos_dev,
                Some(&affine_args()),
                // Plan 622 — the kb hook: the incumbent posture.
                None,
            )
            .expect("launch_attention_fa_devpos");
        stream.synchronize().expect("sync");
        let mut mid_dp = vec![0f32; q_len];
        stream.memcpy_dtoh(&outs[5], &mut mid_dp).unwrap();

        // (1) Flag-off bit-identity — session-shaped pointers, disabled.
        for (i, (b, d)) in base.iter().zip(flag_off.iter()).enumerate() {
            assert_eq!(
                b.to_bits(),
                d.to_bits(),
                "flag-off bit-identity broke at {i}: {b:?} vs {d:?}"
            );
        }
        // (2) Identity schedule armed — bit-identical too.
        for (i, (b, n)) in base.iter().zip(identity.iter()).enumerate() {
            assert_eq!(
                b.to_bits(),
                n.to_bits(),
                "identity-LUT wiring broke at {i}: {b:?} vs {n:?}"
            );
        }
        // (3) The affine moved the result (the hook is live on the prefill
        // path).
        assert!(
            base.iter()
                .zip(affine.iter())
                .any(|(b, a)| b.to_bits() != a.to_bits()),
            "armed affine schedule produced no output change — the prefill hook is inert"
        );
        // (4) The eager and devpos twins agree bit-for-bit at q_offset = 64.
        for (i, (e, d)) in mid.iter().zip(mid_dp.iter()).enumerate() {
            assert_eq!(
                e.to_bits(),
                d.to_bits(),
                "devpos twin diverged from eager at {i}: {e:?} vs {d:?}"
            );
        }
        // (5) + (6) The CPU twins at the f16 tolerance bar — plain validates
        // the fixture machinery; affine validates the prefill si math
        // (per-element dist, the kv==0 sink, the per-head inv window).
        let cpu_plain = cpu_fa_reference(&query, &kh_host, &vh_host, &gate, false, 0.0, 0.0, 1.0);
        let cpu_affine = cpu_fa_reference(&query, &kh_host, &vh_host, &gate, true, 2.0, 1.0, 0.5);
        let rel_plain = max_rel(&base, &cpu_plain);
        let rel_affine = max_rel(&affine, &cpu_affine);
        eprintln!(
            "[prefill_fa_si] rel_plain={rel_plain:.4e} rel_affine={rel_affine:.4e} (f16-operand class, TOL={TOL})"
        );
        assert!(
            rel_plain <= TOL,
            "plain CPU twin mismatch: max_rel={rel_plain:.6} > {TOL}"
        );
        assert!(
            rel_affine <= TOL,
            "affine CPU twin mismatch (the prefill si math): max_rel={rel_affine:.6} > {TOL}"
        );
    }
}
