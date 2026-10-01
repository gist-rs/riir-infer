//! The Q8 residency bit-identity gate (Plan 616 Phase 1): a weight
//! carried as RAW blocked Q8_0 bytes must produce byte-identical
//! forwards to the same values widened on host — through the CPU lane
//! (the once-only host widening) AND through the Metal lane (the
//! `q8_widen_t` load kernel dequant-transposing on device).
//!
//! The construction mirrors the q8 artifact converter's own proof: every
//! GEMM tensor's Q8 payload is the SERIALIZED fake-quant of the seed
//! fill, so the Q8 carrier's decoded values are exactly the Dense
//! twin's payload — the two encoders differ in CARRIER only, never in
//! values. Any byte difference in the outputs is therefore a defect of
//! the carrier path (the widen arithmetic, the transpose, the cache),
//! never of the test data.
//!
//! The geometry exercises every GEMM op family (`matmul_w`,
//! `matmul_w_accum` incl. its fold + unfused arms, `matmul_w_glu`), the
//! packed mixed-plan fallback, the k-tail block (`act_head.0.weight` is
//! [256, d+4] — k % 32 != 0), the head's eight plain GEMMs, the warm
//! path AND the first-miss path (no warm call), and the MPS arm (the
//! default row floor is 0, so the dense batch-1 shapes dispatch MPS on
//! the q8-loaded `Wᵀ`).
//!
//! Run: `cargo test --release --features laya-riir-metal --test q8_widen_identity`
#![cfg(all(target_os = "macos", feature = "laya-riir-metal"))]

use std::collections::HashMap;

use riir_infer_laya::laya::config::EncoderConfig;
use riir_infer_laya::laya::riir::backend::{Backend, Cpu};
use riir_infer_laya::laya::riir::encoder::Encoder;
use riir_infer_laya::laya::riir::fake_quant::{
    BLOCK, fake_quant_q8, q8_quant_of, q8_scale_bits, q8_scale_f32,
};
use riir_infer_laya::laya::riir::head::{Head, HeadScratch};
use riir_infer_laya::laya::riir::metal::Metal;
use riir_infer_laya::laya::riir::weights::{RawQ8, WeightData, Weights, q8_blocked_len};

/// Deterministic xorshift fill — the gate compares two carriers of the
/// SAME values, so coverage is all the data needs.
fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s % 2000) as f32 / 1000.0 - 1.0
        })
        .collect()
}

/// The fake-quant of `data` — the exact f32 values a Q8 artifact tensor
/// with these weights decodes to (the converter's own round-trip law).
fn fake_quantized(data: &[f32]) -> Vec<f32> {
    let mut q = data.to_vec();
    fake_quant_q8(&mut q);
    q
}

/// Serialize the fake-quantized values with the converter's exact block
/// layout — `decode(bytes) == fake_quantized(data)` by construction.
fn q8_bytes_for(data: &[f32]) -> Vec<u8> {
    let q = fake_quantized(data);
    let mut out = Vec::with_capacity(q8_blocked_len(q.len()));
    for block in q.chunks(BLOCK) {
        let amax = block.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
        let bits = q8_scale_bits(amax);
        let d = q8_scale_f32(bits);
        out.extend_from_slice(&bits.to_le_bytes());
        for &x in block {
            out.push(q8_quant_of(x, d) as u8);
        }
    }
    out
}

fn numel(shape: &[usize]) -> usize {
    shape.iter().product()
}

/// The Q8-carried tensor (raw blocked bytes, never widened at parse).
fn q8_tensor(shape: Vec<usize>, seed: u64) -> Weights {
    let n = numel(&shape);
    let raw = RawQ8::new(n, q8_bytes_for(&fill(n, seed))).expect("blocked layout");
    Weights {
        shape,
        data: WeightData::Q8(raw),
    }
}

/// The Dense twin: the widened f32 the Q8 carrier decodes to, exactly.
fn dense_twin(shape: Vec<usize>, seed: u64) -> Weights {
    let n = numel(&shape);
    Weights {
        shape,
        data: WeightData::F32(fake_quantized(&fill(n, seed))),
    }
}

/// A plain f32 tensor (norms/biases — identical in both carriers).
fn f32_tensor(shape: Vec<usize>, seed: u64) -> Weights {
    let n = numel(&shape);
    Weights {
        shape,
        data: WeightData::F32(fill(n, seed)),
    }
}

/// d=64 (hd 64, one head), 4 layers (full / sliding / sliding / full) —
/// the packed-equivalence gate's geometry, so the Q8 gate walks the same
/// op stream plus a head.
fn test_config() -> EncoderConfig {
    EncoderConfig {
        hidden: 64,
        layers: 4,
        heads: 1,
        intermediate: 32,
        vocab: 97,
        eps: 1e-5,
        global_every: 3,
        local_attention: 8, // window 4
        rope_theta_full: 10_000.0,
        rope_theta_slide: 16_000.0,
        sliding: vec![false, true, true, false],
        hidden_activation: "gelu".into(),
    }
}

/// One map builder, both carriers: `gemm` picks the carrier for every
/// GEMM-shaped tensor (the Q8 map carries RAW bytes; the Dense map the
/// exact widened values); the 1D tensors are identical f32 in both.
fn build_maps() -> (HashMap<String, Weights>, HashMap<String, Weights>) {
    let cfg = test_config();
    let d = cfg.hidden;
    let i = cfg.intermediate;
    let mut q8: HashMap<String, Weights> = HashMap::new();
    let mut dn: HashMap<String, Weights> = HashMap::new();
    let mut seed = 1u64;
    let mut next = || {
        seed += 7;
        seed
    };
    let mut put = |name: String, shape: Vec<usize>, gemm: bool| {
        let s = next();
        if gemm {
            q8.insert(name.clone(), q8_tensor(shape.clone(), s));
            dn.insert(name, dense_twin(shape, s));
        } else {
            // Weights is not Clone (the Q8 carrier holds a OnceLock) —
            // the f32 tensor is deterministic in (shape, seed), so build
            // it twice.
            q8.insert(name.clone(), f32_tensor(shape.clone(), s));
            dn.insert(name, f32_tensor(shape, s));
        }
    };

    put(
        "encoder.embeddings.tok_embeddings.weight".into(),
        vec![cfg.vocab, d],
        true, // host gather consumes it — exercises into_f32 at from_map
    );
    put("encoder.embeddings.norm.weight".into(), vec![d], false);
    for idx in 0..cfg.layers {
        if idx != 0 {
            put(
                format!("encoder.layers.{idx}.attn_norm.weight"),
                vec![d],
                false,
            );
        }
        put(
            format!("encoder.layers.{idx}.attn.Wqkv.weight"),
            vec![3 * d, d],
            true,
        );
        put(
            format!("encoder.layers.{idx}.attn.Wo.weight"),
            vec![d, d],
            true,
        );
        put(
            format!("encoder.layers.{idx}.mlp.Wi.weight"),
            vec![2 * i, d],
            true,
        );
        put(
            format!("encoder.layers.{idx}.mlp.Wo.weight"),
            vec![d, i],
            true,
        );
        put(
            format!("encoder.layers.{idx}.mlp_norm.weight"),
            vec![d],
            false,
        );
    }
    put("encoder.final_norm.weight".into(), vec![d], false);

    // The head: 2 layers × 4 GEMMs (in_proj [3d,d], out [d,d], l1 [4d,d],
    // l2 [d,4d]) + the scorer/act stacks. `act_head.0.weight` is [256,
    // d+4] — its k = 68 carries the k-tail block (68 = 2·32 + 4).
    for idx in 0..2 {
        put(
            format!("head.layers.{idx}.self_attn.in_proj_weight"),
            vec![3 * d, d],
            true,
        );
        put(
            format!("head.layers.{idx}.self_attn.in_proj_bias"),
            vec![3 * d],
            false,
        );
        put(
            format!("head.layers.{idx}.self_attn.out_proj.weight"),
            vec![d, d],
            true,
        );
        put(
            format!("head.layers.{idx}.self_attn.out_proj.bias"),
            vec![d],
            false,
        );
        put(format!("head.layers.{idx}.norm1.weight"), vec![d], false);
        put(format!("head.layers.{idx}.norm1.bias"), vec![d], false);
        put(format!("head.layers.{idx}.norm2.weight"), vec![d], false);
        put(format!("head.layers.{idx}.norm2.bias"), vec![d], false);
        put(
            format!("head.layers.{idx}.linear1.weight"),
            vec![4 * d, d],
            true,
        );
        put(
            format!("head.layers.{idx}.linear1.bias"),
            vec![4 * d],
            false,
        );
        put(
            format!("head.layers.{idx}.linear2.weight"),
            vec![d, 4 * d],
            true,
        );
        put(format!("head.layers.{idx}.linear2.bias"), vec![d], false);
    }
    put("type_emb.weight".into(), vec![3, d], true);
    put("scorer.0.weight".into(), vec![d], false);
    put("scorer.0.bias".into(), vec![d], false);
    put("scorer.1.weight".into(), vec![d, d], true);
    put("scorer.1.bias".into(), vec![d], false);
    put("scorer.3.weight".into(), vec![1, d], true);
    put("scorer.3.bias".into(), vec![1], false);
    put("act_head.0.weight".into(), vec![256, d + 4], true);
    put("act_head.0.bias".into(), vec![256], false);
    put("act_head.2.weight".into(), vec![2, 256], true);
    put("act_head.2.bias".into(), vec![2], false);

    (q8, dn)
}

fn ids_for(total: usize, vocab: usize) -> Vec<u32> {
    fill(total, 4242)
        .into_iter()
        .map(|v| ((v + 1.0) * 0.5 * (vocab - 1) as f32) as u32)
        .collect()
}

/// The kill-switch must be unset — under `LAYA_Q8_HOST_F32=1` the Q8
/// carrier widens at load and this whole gate would compare Dense to
/// Dense (a green zero wearing the gate's name).
fn require_kill_switch_unset() {
    assert!(
        std::env::var("LAYA_Q8_HOST_F32").as_deref() != Ok("1"),
        "LAYA_Q8_HOST_F32=1 forces the host widen at load — unset it before \
         running the Q8 identity gate (it would otherwise prove nothing)"
    );
}

#[test]
fn q8_carrier_matches_dense_cpu_bit_identically() {
    require_kill_switch_unset();
    let cfg = test_config();
    let (mut mq, _) = build_maps();
    let (_, mut md) = build_maps();
    let enc_q = Encoder::from_map(&mut mq, cfg.clone(), "q8-cpu").expect("q8 encoder loads");
    let enc_d = Encoder::from_map(&mut md, cfg, "dense-cpu").expect("dense encoder loads");
    let (mut hq_map, _) = build_maps();
    let (_, mut hd_map) = build_maps();
    let head_q = Head::from_map(&mut hq_map, "q8-cpu", 64, 1e-5).expect("q8 head");
    let head_d = Head::from_map(&mut hd_map, "dense-cpu", 64, 1e-5).expect("dense head");

    let ids = ids_for(23, 97);
    let b = Cpu;
    let hq = enc_q.forward(&b, &ids).expect("q8 forward");
    let hd = enc_d.forward(&b, &ids).expect("dense forward");
    assert_eq!(hq, hd, "encoder outputs must be bit-identical on CPU");

    let mut sq = HeadScratch::new();
    let mut sd = HeadScratch::new();
    let mut h1 = hq.clone();
    let mut h2 = hd;
    let oq = head_q
        .forward(&b, &mut h1, 1, &[0, 22], &mut sq)
        .expect("q8 head");
    let od = head_d
        .forward(&b, &mut h2, 1, &[0, 22], &mut sd)
        .expect("dense head");
    assert_eq!(oq.logits, od.logits, "head logits bit-identical on CPU");
    assert_eq!(
        oq.act_probabilities, od.act_probabilities,
        "act probabilities bit-identical on CPU"
    );
}

#[test]
fn q8_carrier_matches_dense_metal_bit_identically() {
    require_kill_switch_unset();
    let cfg = test_config();
    let (mut mq, _) = build_maps();
    let (_, mut md) = build_maps();
    let enc_q = Encoder::from_map(&mut mq, cfg.clone(), "q8-m").expect("q8 encoder");
    let enc_d = Encoder::from_map(&mut md, cfg, "dense-m").expect("dense encoder");
    let (mut hq_map, _) = build_maps();
    let (_, mut hd_map) = build_maps();
    let head_q = Head::from_map(&mut hq_map, "q8-m", 64, 1e-5).expect("q8 head");
    let head_d = Head::from_map(&mut hd_map, "dense-m", 64, 1e-5).expect("dense head");

    let m = Metal::new().expect("metal backend");
    // The default posture: folds ON (the split-K epilogue rungs) + MPS ON
    // (row floor 0 → the dense batch-1 shapes dispatch MPS over the
    // q8-loaded Wᵀ).
    enc_q.warm(&m);
    head_q.warm(&m);
    enc_d.warm(&m);
    head_d.warm(&m);

    // The PACKED form: mixed row segments exercise matmul_w_accum's
    // mixed-plan fallback (the unfused then_add stream) beside the fold.
    let seqs = [3usize, 17, 1, 2];
    let total: usize = seqs.iter().sum();
    let ids = ids_for(total, 97);

    m.begin_pass();
    let out_q = enc_q
        .forward_packed(&m, &ids, &seqs)
        .expect("q8 packed forward");
    let mut host_q = vec![0f32; total * 64];
    m.download_into(&out_q, &mut host_q);

    m.begin_pass();
    let out_d = enc_d
        .forward_packed(&m, &ids, &seqs)
        .expect("dense packed forward");
    let mut host_d = vec![0f32; total * 64];
    m.download_into(&out_d, &mut host_d);

    assert_eq!(
        host_q, host_d,
        "packed encoder outputs bit-identical (Metal)"
    );

    // The head, per question, through the enqueue/read/act stream (its
    // eight plain GEMMs ride the Q8 carrier too).
    for (qi, &seq) in seqs.iter().enumerate() {
        let off: usize = seqs[..qi].iter().sum();
        let markers = [0usize, seq - 1];
        let mut sq = HeadScratch::new();
        let mut sd = HeadScratch::new();
        let mut hq = host_q[off * 64..(off + seq) * 64].to_vec();
        let mut hd = host_d[off * 64..(off + seq) * 64].to_vec();
        m.begin_pass();
        let oq = head_q
            .forward(&m, &mut hq, qi % 3, &markers, &mut sq)
            .expect("q8 head");
        let od = head_d
            .forward(&m, &mut hd, qi % 3, &markers, &mut sd)
            .expect("dense head");
        assert_eq!(oq.logits, od.logits, "q{qi} logits bit-identical (Metal)");
        assert_eq!(
            oq.act_probabilities, od.act_probabilities,
            "q{qi} act probabilities bit-identical (Metal)"
        );
    }
}

#[test]
fn q8_first_miss_matches_warmed_dense_metal() {
    require_kill_switch_unset();
    let cfg = test_config();
    // NO warm_weight_2d_q8 call — the first forward's weight_t_buf_q8
    // miss builds the Wᵀ in-flight (the un-warmed path must serve too).
    let (mut mq, _) = build_maps();
    let (_, mut md) = build_maps();
    let enc_q = Encoder::from_map(&mut mq, cfg.clone(), "q8-fm").expect("q8 encoder");
    let enc_d = Encoder::from_map(&mut md, cfg, "dense-fm").expect("dense encoder");

    let m = Metal::new().expect("metal backend");
    let ids = ids_for(19, 97);

    m.begin_pass();
    let out_q = enc_q.forward(&m, &ids).expect("q8 forward");
    let mut host_q = vec![0f32; 19 * 64];
    m.download_into(&out_q, &mut host_q);

    m.begin_pass();
    let out_d = enc_d.forward(&m, &ids).expect("dense forward");
    let mut host_d = vec![0f32; 19 * 64];
    m.download_into(&out_d, &mut host_d);

    assert_eq!(host_q, host_d, "first-miss q8 forward bit-identical");
}

#[test]
fn q8_unfused_stream_matches_dense_metal() {
    require_kill_switch_unset();
    let cfg = test_config();
    let (mut mq, _) = build_maps();
    let (_, mut md) = build_maps();
    let enc_q = Encoder::from_map(&mut mq, cfg.clone(), "q8-uf").expect("q8 encoder");
    let enc_d = Encoder::from_map(&mut md, cfg, "dense-uf").expect("dense encoder");

    // Folds OFF: the residual add and the GLU ride the unfused
    // then_add / then_glu streams on the Q8-loaded Wᵀ.
    let m = Metal::new()
        .expect("metal backend")
        .with_folds(false, false);
    enc_q.warm(&m);
    enc_d.warm(&m);

    let ids = ids_for(19, 97);
    m.begin_pass();
    let out_q = enc_q.forward(&m, &ids).expect("q8 forward");
    let mut host_q = vec![0f32; 19 * 64];
    m.download_into(&out_q, &mut host_q);

    m.begin_pass();
    let out_d = enc_d.forward(&m, &ids).expect("dense forward");
    let mut host_d = vec![0f32; 19 * 64];
    m.download_into(&out_d, &mut host_d);

    assert_eq!(host_q, host_d, "unfused-stream q8 forward bit-identical");
}

/// The load kernel directly: `q8_widen_t`'s Wᵀ read out through an
/// identity-matmul (`I_k @ Wᵀ == Wᵀ`) and compared element-wise against
/// the host widen transposed — the sharpest unit gate, so a divergence
/// is diagnosable at the kernel without re-deriving it from a red
/// forward. The shape carries the k-tail (k = 68).
#[test]
fn q8_widen_kernel_transpose_matches_host() {
    require_kill_switch_unset();
    const N: usize = 3;
    const K: usize = 68; // 2 full blocks + a 4-wide tail
    let data: Vec<f32> = fill(N * K, 77);
    let raw = RawQ8::new(N * K, q8_bytes_for(&data)).expect("blocked layout");

    let m = Metal::new().expect("metal backend");
    m.warm_weight_2d_q8(&raw, N, K);

    // Identity activations: dst = I_K @ Wᵀ = Wᵀ (row-major [k, n]).
    let mut a = vec![0f32; K * K];
    for i in 0..K {
        a[i * K + i] = 1.0;
    }
    m.begin_pass();
    let mut got = vec![0f32; K * N];
    m.matmul_w_q8(&a, K, K, &raw, N, &mut got);
    let mut host_got = vec![0f32; K * N];
    m.download_into(&got, &mut host_got);

    // The host widen transposed (the values the buffer must hold).
    let wide = raw.wide();
    let mut expect = vec![0f32; K * N];
    for (row, src) in wide.as_chunks::<K>().0.iter().enumerate() {
        for (kk, v) in src.iter().enumerate() {
            expect[kk * N + row] = *v;
        }
    }
    assert_eq!(
        host_got, expect,
        "the device Wᵀ is the host widen transposed"
    );
}

// ── The live artifact probe (#[ignore]d — real checkpoint, ~447 MB
// artifact + Metal warm, minutes) ─────────────────────────────────────

/// The REAL english q8 artifact through BOTH load paths in ONE process:
/// the Phase 1 default (raw retained, device widen) vs the kill-switch
/// (`LAYA_Q8_HOST_F32=1` — today's widen-at-load). The served answers
/// must be BYTE-identical; the probe prints both postures' peak RSS so
/// the host-residency claim lands beside the proof. Run:
///
/// ```sh
/// cargo test --release --features laya-riir-metal --test q8_widen_identity \
///   live_q8_artifact -- --ignored --nocapture
/// ```
#[test]
#[ignore]
fn live_q8_artifact_device_widen_matches_host_widen() {
    use riir_infer_laya::laya::config::Checkpoint;
    use riir_infer_laya::laya::riir::agent::{DeviceKind, RiirAgent};
    use riir_infer_laya::laya::weights::weights_root;

    require_kill_switch_unset();
    // SAFETY: sequential env flips driving the two load postures; no
    // other thread reads these during the probe.
    unsafe { std::env::set_var("LAYA_WEIGHTS_VARIANT", "q8") };
    let root = weights_root();

    let rss_mib = |label: &str| {
        // The task's own footprint (mach_task_basic_info.resident_size) —
        // the honest single-process host-residency readout.
        // SAFETY: an FFI probe (mach task_info) on the test's own task;
        // the struct/flavor pair is the documented MACH_TASK_BASIC_INFO.
        unsafe extern "C" {
            static mach_task_self_: u32;
            fn task_info(target: u32, flavor: u32, info: *mut u8, count: *mut u32) -> i32;
        }
        #[repr(C)]
        #[derive(Default)]
        struct TimeVal {
            sec: i32,
            usec: i32,
        }
        #[repr(C)]
        #[derive(Default)]
        struct BasicInfo {
            virtual_size: u64,
            resident_size: u64,
            resident_size_max: u64,
            user_time: TimeVal,
            system_time: TimeVal,
            policy: i32,
            suspend_count: i32,
        }
        let mut info = BasicInfo::default();
        let mut count = (std::mem::size_of::<BasicInfo>() / 4) as u32; // 12
        // MACH_TASK_BASIC_INFO = 20; target = mach_task_self().
        let rc = unsafe {
            task_info(
                mach_task_self_,
                20,
                &mut info as *mut BasicInfo as *mut u8,
                &mut count,
            )
        };
        assert_eq!(rc, 0, "task_info failed");
        let r = info.resident_size / (1024 * 1024);
        let p = info.resident_size_max / (1024 * 1024);
        println!("[{label}] rss {r} MiB · peak {p} MiB");
        (r, p)
    };

    let ck = Checkpoint::English;
    let answers;
    {
        let agent_device = RiirAgent::load_with_device(&root, ck, DeviceKind::Metal)
            .expect("device-widen agent loads");
        rss_mib("device-widen (raw retained, alone)");
        let state = serde_json::json!(
            "The quarterly revenue forecast beat every analyst estimate. Shares \
             rose after the central bank held rates steady."
        );
        let qdef = serde_json::json!({
            "type": "score",
            "instructions": "Rate the market sentiment from 0 (very negative) to 4 (very positive).",
            "criteria": ["very negative", "negative", "neutral", "positive", "very positive"]
        });
        let fwd = agent_device
            .forward_question(&state, &qdef)
            .expect("forward runs");
        println!(
            "[device] probs {:?} conf {:.6} act {:?}",
            fwd.probs, fwd.confidence, fwd.act_probabilities
        );
        answers = fwd;
        // DROP before the second load: the agent's Vecs munmap, so the
        // host posture's reading below is its own, not a sum of both.
    }

    unsafe { std::env::set_var("LAYA_Q8_HOST_F32", "1") };
    let agent_host =
        RiirAgent::load_with_device(&root, ck, DeviceKind::Metal).expect("host-widen agent loads");
    rss_mib("host-widen (kill-switch, alone after drop)");
    unsafe { std::env::remove_var("LAYA_Q8_HOST_F32") };

    let state = serde_json::json!(
        "The quarterly revenue forecast beat every analyst estimate. Shares \
         rose after the central bank held rates steady."
    );
    let qdef = serde_json::json!({
        "type": "score",
        "instructions": "Rate the market sentiment from 0 (very negative) to 4 (very positive).",
        "criteria": ["very negative", "negative", "neutral", "positive", "very positive"]
    });
    let fwd_host = agent_host
        .forward_question(&state, &qdef)
        .expect("forward runs");
    println!(
        "[host] probs {:?} conf {:.6} act {:?}",
        fwd_host.probs, fwd_host.confidence, fwd_host.act_probabilities
    );
    assert_eq!(answers.probs, fwd_host.probs, "probabilities bit-identical");
    assert_eq!(
        answers.act_probabilities, fwd_host.act_probabilities,
        "act probabilities bit-identical"
    );
    assert_eq!(
        answers.confidence, fwd_host.confidence,
        "confidence bit-identical"
    );
    println!("LIVE PROBE: the q8 artifact's two load paths agree byte-for-byte");
}
