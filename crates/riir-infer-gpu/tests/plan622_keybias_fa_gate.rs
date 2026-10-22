//! Plan 622 T1.2/T1.4 — the FA prefill kb-bias gate: the bit-identity +
//! staging-correctness teeth for the per-key attention-logit bias at the
//! `att_pf_fa` class, ported from the Plan-620 si-gate matrix.
//!
//! Arms over one fixture (mid-context `base_pos = 70` — deliberately NOT
//! 32-tile-aligned, so the causal limits and the bias table cut through KV
//! tiles and the last tile's fragment indexes run past the causal limit —
//! the clamped-index read's inert adds live exactly there):
//! `base` (kb=None, the dummies) / `flag_off` (session-shaped pointers,
//! disabled) / `zero_armed` / `moved` (a varying ±1.5 table, run TWICE) /
//! `moved_pad` (the same values, stride = KV_LEN + 40 with garbage pad) /
//! `moved_dp` (the devpos graph twin). The sharp teeth are bit-exact:
//! flag-off == base and zero-armed == base (the softmax collapses the ±0.0
//! the `+0.0` add can produce, and the causal overwrite makes the
//! out-of-range adds inert), moved run-1 == run-2 (the armed path's
//! run-to-run race guard — any future smem staging or caching lane shows
//! here first), moved_pad == moved (stride independence — a broken guard
//! would read the garbage pad), and the devpos twin == the eager twin
//! (both entries share the body). The coarse teeth are the CPU twins
//! (plain + biased, the same f16-rounded operands, causally masked like
//! the kernel) at the bench_946 calibrated bars — a mis-indexed bias read
//! (wrong head, wrong key offset) moves scores O(1) and blows through.

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

/// The FA geometry (the only shape `fa_launch_check` accepts): DKQ=DV=256,
/// GQA exactly 6:1.
const HEAD_DIM: usize = 256;
const N_HEAD: usize = 24;
const N_KV: usize = 4;
/// p = FA_MIN_P: the engagement-predicate boundary this hook is scoped to.
const P: usize = 128;
/// Mid-context, NOT tile-aligned: kv_len = 198 → 7 KV tiles, the last one
/// partial (6 real rows + a zero tail) — the exact shape that stresses the
/// stage's per-tile guards.
const BASE_POS: usize = 70;
const KV_LEN: usize = BASE_POS + P;
/// The CPU-twin tolerance bars — the bench_946 G1 calibrated metric for
/// this exact kernel class (the f16-operand/f16-VKQ FA class measured
/// ~1e-2 worst-case at its 4090 promotion): scaled = |a-b| /
/// max(|b|, ABS_BAR/REL_BAR), so the bar reads |a-b| <= max(REL*|b|,
/// ABS_BAR). A per-element rel with a 1e-6 floor is meaningless here —
/// attention outputs cancel to ~0 on hundreds of elements and the f16
/// accumulation noise (~2e-4 absolute) then reads as rel ~ O(10) with the
/// kernel perfectly healthy (measured: rel_plain=6.67 at the FIRST 4090
/// run of this gate, every bit-identity arm green). A structural kb error
/// (a mis-indexed stage row) moves scores O(1) — unmissable at these
/// bars; the noise class sits 5x below them.
const REL_BAR: f32 = 1e-1;
const ABS_BAR: f32 = 5e-2;

fn lcg(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 33) as f32 / (u32::MAX >> 1) as f32) * 0.5 - 0.25
}

/// The kb table's f16 wire (the T1.4 lever): host f32 rows → IEEE halves,
/// RN-even — the same conversion the session's `write_table` does (`half`
/// and the crate's `kv_f16_bits::f32_to_f16` agree bit-for-bit).
fn f16wire(v: &[f32]) -> Vec<u16> {
    v.iter()
        .map(|&x| half::f16::from_f32(x).to_bits())
        .collect()
}

/// The CPU twin of the FA kernel's output for one posture: f32 softmax over
/// the f16-rounded operands (`cvt.rn` semantics via the `half` crate), the
/// kb bias added to every in-range score when armed, CAUSALLY MASKED like
/// the kernel (`lim = q_offset + j + 1`; masked keys underflow exp to 0),
/// the sigmoid gate in the epilogue. f32 accumulation — the kernel's f16
/// P/V accumulation is noise the bar absorbs. `bias` is the head-major
/// `[n_head][KV_LEN]` table (the test's stride-exact layout).
#[allow(clippy::too_many_arguments)]
fn cpu_fa_reference(
    query: &[f32],
    kh: &[f32],
    vh: &[f32],
    gate: &[f32],
    bias: Option<&[f32]>,
    q_offset: usize,
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
                let biased = match bias {
                    Some(b) => raw + b[h * KV_LEN + kv],
                    None => raw,
                };
                // Causal: the kernel masks kv >= q_offset + j + 1 to -1e30f.
                *s = if kv > q_offset + j {
                    f32::NEG_INFINITY
                } else {
                    biased
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

/// The bench_946 G1 metric: (max, mean) scaled diff, scaled =
/// |a-b| / max(|b|, ABS_BAR/REL_BAR) — the bar check is `max <= REL_BAR`.
fn err_stats(a: &[f32], b: &[f32]) -> (f32, f64) {
    let denom_floor = ABS_BAR / REL_BAR;
    let mut max = 0f32;
    let mut sum = 0f64;
    let mut n = 0usize;
    for i in 0..a.len().min(b.len()) {
        if !a[i].is_finite() || !b[i].is_finite() {
            max = f32::INFINITY;
            continue;
        }
        let d = (a[i] - b[i]).abs();
        let denom = b[i].abs().max(denom_floor);
        let s = d / denom;
        if s > max {
            max = s;
        }
        sum += s as f64;
        n += 1;
    }
    (max, if n > 0 { sum / n as f64 } else { 0.0 })
}

#[test]
fn test_prefill_fa_kb_flag_off_zero_stage_and_stride() {
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
        let mut st: u64 = 0x6226_2262_2622_6221;
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

        // The kb tables. `moved` is the varying ±1.5 real-class shape
        // (stride-exact); `moved_pad` carries the SAME values at the same
        // slots in a stride = KV_LEN + 40 table whose pad band is GARBAGE —
        // the stage must never read it (stride independence THROUGH the
        // stage is the graphs-law tooth: a captured graph bakes a
        // max-sized stride).
        let mut moved_tbl = vec![0.0f32; N_HEAD * KV_LEN];
        for h in 0..N_HEAD {
            for k in 0..KV_LEN {
                moved_tbl[h * KV_LEN + k] = ((h * 7 + k) % 11) as f32 * 0.3 - 1.5;
            }
        }
        let pad_stride = KV_LEN + 40;
        let mut moved_pad_tbl = vec![0.0f32; N_HEAD * pad_stride];
        for h in 0..N_HEAD {
            for k in 0..KV_LEN {
                moved_pad_tbl[h * pad_stride + k] = moved_tbl[h * KV_LEN + k];
            }
            for k in KV_LEN..pad_stride {
                moved_pad_tbl[h * pad_stride + k] = lcg(&mut st);
            }
        }
        let bias_zero = stream.clone_htod(&vec![0u16; N_HEAD * KV_LEN]).unwrap();
        let bias_moved = stream.clone_htod(&f16wire(&moved_tbl)).unwrap();
        let bias_moved_pad = stream.clone_htod(&f16wire(&moved_pad_tbl)).unwrap();
        let disabled = stream.clone_htod(&[0i32]).unwrap();
        let enabled = stream.clone_htod(&[1i32]).unwrap();

        let launch_eager = |out: &CudaSlice<f32>,
                            base_pos: usize,
                            kb: Option<&KeyBiasLayerArgs<'_>>| {
            kernels
                .launch_attention_fa(
                    &stream, &q_dev, &kh, &vh, &gate_dev, out, HEAD_DIM, N_HEAD, N_KV, P, base_pos,
                    // Plan 620 — the si hook: the incumbent posture (this
                    // gate owns the kb arm).
                    None, kb,
                )
                .expect("launch_attention_fa");
            stream.synchronize().expect("sync");
            let mut host = vec![0f32; q_len];
            stream.memcpy_dtoh(out, &mut host).unwrap();
            host
        };

        let flag_off_kb = KeyBiasLayerArgs {
            bias: bias_moved.slice(0..bias_moved.len()),
            stride: KV_LEN,
            enabled: &disabled,
        };
        let zero_kb = KeyBiasLayerArgs {
            bias: bias_zero.slice(0..bias_zero.len()),
            stride: KV_LEN,
            enabled: &enabled,
        };
        let moved_kb = KeyBiasLayerArgs {
            bias: bias_moved.slice(0..bias_moved.len()),
            stride: KV_LEN,
            enabled: &enabled,
        };
        let moved_pad_kb = KeyBiasLayerArgs {
            bias: bias_moved_pad.slice(0..bias_moved_pad.len()),
            stride: pad_stride,
            enabled: &enabled,
        };

        let outs: Vec<CudaSlice<f32>> = (0..6)
            .map(|_| stream.alloc_zeros::<f32>(q_len).unwrap())
            .collect();

        let base = launch_eager(&outs[0], 0, None);
        let flag_off = launch_eager(&outs[1], 0, Some(&flag_off_kb));
        let zero = launch_eager(&outs[2], 0, Some(&zero_kb));
        let moved = launch_eager(&outs[3], BASE_POS, Some(&moved_kb));
        // Run-twice determinism — the T1.4 smem stage's double-buffer race
        // guard (a WAR-parity bug shows as run-to-run bit drift).
        let moved_again = launch_eager(&outs[3], BASE_POS, Some(&moved_kb));
        let moved_pad = launch_eager(&outs[4], BASE_POS, Some(&moved_pad_kb));
        // The devpos graph twin (q_offset from the device pos buffer).
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
                None,
                Some(&moved_kb),
            )
            .expect("launch_attention_fa_devpos");
        stream.synchronize().expect("sync");
        let mut moved_dp = vec![0f32; q_len];
        stream.memcpy_dtoh(&outs[5], &mut moved_dp).unwrap();

        // (1) Flag-off bit-identity — session-shaped pointers, disabled.
        for (i, (b, d)) in base.iter().zip(flag_off.iter()).enumerate() {
            assert_eq!(
                b.to_bits(),
                d.to_bits(),
                "kb flag-off bit-identity broke at {i}: {b:?} vs {d:?}"
            );
        }
        // (2) Zero-armed bit-identity — the staged +0.0 add collapses in the
        // softmax (exp(x) == exp(x + 0) to the bit).
        for (i, (b, z)) in base.iter().zip(zero.iter()).enumerate() {
            assert_eq!(
                b.to_bits(),
                z.to_bits(),
                "kb zero-armed bit-identity broke at {i}: {b:?} vs {z:?}"
            );
        }
        // (3) The bias moved the result (the hook is live through the stage).
        assert!(
            base.iter()
                .zip(moved.iter())
                .any(|(b, m)| b.to_bits() != m.to_bits()),
            "armed varying bias produced no output change — the prefill kb hook is inert"
        );
        // (4) Run-twice determinism — the armed path's race guard.
        for (i, (m, r)) in moved.iter().zip(moved_again.iter()).enumerate() {
            assert_eq!(
                m.to_bits(),
                r.to_bits(),
                "kb armed run-to-run drift at {i} — the read path races: {m:?} vs {r:?}"
            );
        }
        // (5) Stride independence — the garbage pad is never read (the
        // clamped index keeps every read inside [0, kv_len) of the row).
        for (i, (m, p)) in moved.iter().zip(moved_pad.iter()).enumerate() {
            assert_eq!(
                m.to_bits(),
                p.to_bits(),
                "kb stride-pad divergence at {i} — the stage reads past kv_len: {m:?} vs {p:?}"
            );
        }
        // (6) The eager and devpos twins agree bit-for-bit (both entries run
        // the same staged body).
        for (i, (m, d)) in moved.iter().zip(moved_dp.iter()).enumerate() {
            assert_eq!(
                m.to_bits(),
                d.to_bits(),
                "kb devpos twin diverged from eager at {i}: {m:?} vs {d:?}"
            );
        }
        // (7) + (8) The CPU twins at the bench_946 calibrated bars — plain
        // validates the fixture machinery (causal mask included); biased
        // validates the prefill kb math (per-head rows, per-key slots, the
        // in-range predicate, the stride arithmetic through the stage).
        let cpu_plain = cpu_fa_reference(&query, &kh_host, &vh_host, &gate, None, 0);
        let cpu_moved = cpu_fa_reference(
            &query,
            &kh_host,
            &vh_host,
            &gate,
            Some(&moved_tbl),
            BASE_POS,
        );
        let (mx_plain, mean_plain) = err_stats(&base, &cpu_plain);
        let (mx_moved, mean_moved) = err_stats(&moved, &cpu_moved);
        eprintln!(
            "[prefill_fa_kb] scaled_plain max={mx_plain:.3e} mean={mean_plain:.3e} | scaled_moved max={mx_moved:.3e} mean={mean_moved:.3e} (bars: REL={REL_BAR:.0e}, ABS={ABS_BAR:.0e})"
        );
        assert!(
            mx_plain <= REL_BAR,
            "plain CPU twin mismatch: max scaled {mx_plain:.6} > {REL_BAR}"
        );
        assert!(
            mx_moved <= REL_BAR,
            "biased CPU twin mismatch (the prefill kb staging math): max scaled {mx_moved:.6} > {REL_BAR}"
        );
    }
}
