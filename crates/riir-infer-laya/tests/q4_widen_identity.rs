//! The Q4 residency bit-identity gate (Plan 616 Phase 3 — the
//! [`q8_widen_identity`] battery's 4-bit twin): a weight carried as RAW
//! blocked Q4_0 bytes must produce byte-identical forwards to the same
//! values widened on host — through the CPU lane (the once-only host
//! widening) AND through the Metal lane (the device-resident RAW bytes +
//! the derived `sgemm_q4`/`sgemm_xwide_q4`/`sgemm_splitk_q4` staging by
//! default; the `q4_widen_t` load kernel under `LAYA_Q8_DEVICE_F32=1`).
//!
//! AT ITS OWN FIDELITY, never vs F16: the Dense twin carries the
//! fake-quant-Q4 DECODED values (the converter's round-trip law), so the
//! two encoders differ in CARRIER only, never in values. Any byte
//! difference in the outputs is a defect of the carrier path (the nibble
//! widen arithmetic, the derived staging decode, the caches), never of
//! the test data. The Q4 grid's distance FROM F16 is the separate D1
//! probe's subject (instinct issue 018's own law).
//!
//! THE DISPATCH-TREE LAW (the q8 gate's, shared): the bit-identity pairs
//! run on a `.with_mps(false)` instance — both arms then share one
//! dispatch tree and byte-equality is the carrier proof. The derived q4
//! kernels are the q8 MSL texts with ONLY the decode block swapped (the
//! format seam asserts its own tokens at generation), so a red here
//! localizes to the decode or the seam.
//!
//! Run: `cargo test --release --features laya-riir-metal --test q4_widen_identity`
#![cfg(all(target_os = "macos", feature = "laya-riir-metal"))]

use std::collections::HashMap;

use riir_infer_laya::laya::config::EncoderConfig;
use riir_infer_laya::laya::riir::backend::{Backend, Cpu};
use riir_infer_laya::laya::riir::encoder::Encoder;
use riir_infer_laya::laya::riir::fake_quant::{
    BLOCK, fake_quant_q4, q4_quant_of, q4_scale_bits, q4_scale_f32,
};
use riir_infer_laya::laya::riir::head::{Head, HeadScratch};
use riir_infer_laya::laya::riir::metal::Metal;
use riir_infer_laya::laya::riir::weights::{RawQ4, WeightData, Weights, q4_blocked_len};

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

/// The fake-quant of `data` at the Q4 grid — the exact f32 values a Q4
/// artifact tensor with these weights decodes to (the converter's own
/// round-trip law).
fn fake_quantized(data: &[f32]) -> Vec<f32> {
    let mut q = data.to_vec();
    fake_quant_q4(&mut q);
    q
}

/// Serialize the fake-quantized values with the converter's exact block
/// layout (GGML nibble order: even element low, odd element high) —
/// `decode(bytes) == fake_quantized(data)` by construction.
fn q4_bytes_for(data: &[f32]) -> Vec<u8> {
    let q = fake_quantized(data);
    let mut out = Vec::with_capacity(q4_blocked_len(q.len()));
    for block in q.chunks(BLOCK) {
        let amax = block.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
        let bits = q4_scale_bits(amax);
        let d = q4_scale_f32(bits);
        out.extend_from_slice(&bits.to_le_bytes());
        for pair in block.chunks(2) {
            let lo = q4_quant_of(pair[0], d) as u8;
            let hi = pair.get(1).map(|&v| q4_quant_of(v, d) as u8).unwrap_or(0);
            out.push((hi & 0x0F) << 4 | (lo & 0x0F));
        }
    }
    out
}

fn numel(shape: &[usize]) -> usize {
    shape.iter().product()
}

/// The Q4-carried tensor (raw blocked bytes, never widened at parse).
fn q4_tensor(shape: Vec<usize>, seed: u64) -> Weights {
    let n = numel(&shape);
    let raw = RawQ4::new(n, q4_bytes_for(&fill(n, seed))).expect("blocked layout");
    Weights {
        shape,
        data: WeightData::Q4(raw),
    }
}

/// The Dense twin: the widened f32 the Q4 carrier decodes to, exactly.
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
/// the packed-equivalence gate's geometry, so the Q4 gate walks the same
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
/// GEMM-shaped tensor (the Q4 map carries RAW bytes; the Dense map the
/// exact widened values); the 1D tensors are identical f32 in both.
fn build_maps() -> (HashMap<String, Weights>, HashMap<String, Weights>) {
    let cfg = test_config();
    let d = cfg.hidden;
    let i = cfg.intermediate;
    let mut q4: HashMap<String, Weights> = HashMap::new();
    let mut dn: HashMap<String, Weights> = HashMap::new();
    let mut seed = 1u64;
    let mut next = || {
        seed += 7;
        seed
    };
    let mut put = |name: String, shape: Vec<usize>, gemm: bool| {
        let s = next();
        if gemm {
            q4.insert(name.clone(), q4_tensor(shape.clone(), s));
            dn.insert(name, dense_twin(shape, s));
        } else {
            // Weights is not Clone (the Q4 carrier holds a OnceLock) —
            // the f32 tensor is deterministic in (shape, seed), so build
            // it twice.
            q4.insert(name.clone(), f32_tensor(shape.clone(), s));
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

    // The head: 2 layers × 4 GEMMs + the scorer/act stacks.
    // `act_head.0.weight` is [256, d+4] — its k = 68 carries the k-tail
    // block (68 = 2·32 + 4).
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

    (q4, dn)
}

fn ids_for(total: usize, vocab: usize) -> Vec<u32> {
    fill(total, 4242)
        .into_iter()
        .map(|v| ((v + 1.0) * 0.5 * (vocab - 1) as f32) as u32)
        .collect()
}

/// The kill switches must be unset — under `LAYA_Q8_HOST_F32=1` the Q4
/// carrier widens at load and this whole gate would compare Dense to
/// Dense (a green zero wearing the gate's name); under
/// `LAYA_Q8_DEVICE_F32=1` the default tests would exercise the load-kernel
/// path instead of the fused staging.
fn require_kill_switch_unset() {
    assert!(
        std::env::var("LAYA_Q8_HOST_F32").as_deref() != Ok("1"),
        "LAYA_Q8_HOST_F32=1 forces the host widen at load — unset it before \
         running the Q4 identity gate (it would otherwise prove nothing)"
    );
    assert!(
        std::env::var("LAYA_Q8_DEVICE_F32").as_deref() != Ok("1"),
        "LAYA_Q8_DEVICE_F32=1 forces the device-F32 posture — unset it so the \
         default tests exercise the fused staging (the kill-switch has its own \
         dedicated arms)"
    );
}

/// The env-flip lock (the q8 gate's): the tests that flip
/// `LAYA_Q8_DEVICE_F32` hold it so two flippers cannot interleave.
fn env_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
    LOCK.get_or_init(|| std::sync::Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[test]
fn q4_carrier_matches_dense_cpu_bit_identically() {
    require_kill_switch_unset();
    let cfg = test_config();
    let (mut mq, _) = build_maps();
    let (_, mut md) = build_maps();
    let enc_q = Encoder::from_map(&mut mq, cfg.clone(), "q4-cpu").expect("q4 encoder loads");
    let enc_d = Encoder::from_map(&mut md, cfg, "dense-cpu").expect("dense encoder loads");
    let (mut hq_map, _) = build_maps();
    let (_, mut hd_map) = build_maps();
    let head_q = Head::from_map(&mut hq_map, "q4-cpu", 64, 1e-5).expect("q4 head");
    let head_d = Head::from_map(&mut hd_map, "dense-cpu", 64, 1e-5).expect("dense head");

    let ids = ids_for(23, 97);
    let b = Cpu;
    let hq = enc_q.forward(&b, &ids).expect("q4 forward");
    let hd = enc_d.forward(&b, &ids).expect("dense forward");
    assert_eq!(hq, hd, "encoder outputs must be bit-identical on CPU");

    let mut sq = HeadScratch::new();
    let mut sd = HeadScratch::new();
    let mut h1 = hq.clone();
    let mut h2 = hd;
    let oq = head_q
        .forward(&b, &mut h1, 1, &[0, 22], &mut sq)
        .expect("q4 head");
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
fn q4_carrier_matches_dense_metal_bit_identically() {
    require_kill_switch_unset();
    let cfg = test_config();
    let (mut mq, _) = build_maps();
    let (_, mut md) = build_maps();
    let enc_q = Encoder::from_map(&mut mq, cfg.clone(), "q4-m").expect("q4 encoder");
    let enc_d = Encoder::from_map(&mut md, cfg, "dense-m").expect("dense encoder");
    let (mut hq_map, _) = build_maps();
    let (_, mut hd_map) = build_maps();
    let head_q = Head::from_map(&mut hq_map, "q4-m", 64, 1e-5).expect("q4 head");
    let head_d = Head::from_map(&mut hd_map, "dense-m", 64, 1e-5).expect("dense head");

    // The bit-identity pair shares ONE dispatch tree: MPS-off (the q8
    // gate's law — under the default posture the dense shapes differ by
    // the option-(i) T13-class accumulation-order delta).
    let m = Metal::new().expect("metal backend").with_mps(false);
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
        .expect("q4 packed forward");
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
        "packed encoder outputs bit-identical (Metal, q4 carrier)"
    );

    // The head, per question (its eight plain GEMMs ride the Q4 carrier
    // too — including the k-tail act_head GEMM).
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
            .expect("q4 head");
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
fn q4_first_miss_matches_warmed_dense_metal() {
    require_kill_switch_unset();
    let cfg = test_config();
    // NO warm call — the first forward's resident-buffer miss uploads
    // in-flight (the un-warmed path must serve too).
    let (mut mq, _) = build_maps();
    let (_, mut md) = build_maps();
    let enc_q = Encoder::from_map(&mut mq, cfg.clone(), "q4-fm").expect("q4 encoder");
    let enc_d = Encoder::from_map(&mut md, cfg, "dense-fm").expect("dense encoder");

    let m = Metal::new().expect("metal backend");
    let ids = ids_for(19, 97);

    m.begin_pass();
    let out_q = enc_q.forward(&m, &ids).expect("q4 forward");
    let mut host_q = vec![0f32; 19 * 64];
    m.download_into(&out_q, &mut host_q);

    m.begin_pass();
    let out_d = enc_d.forward(&m, &ids).expect("dense forward");
    let mut host_d = vec![0f32; 19 * 64];
    m.download_into(&out_d, &mut host_d);

    assert_eq!(host_q, host_d, "first-miss q4 forward bit-identical");
}

#[test]
fn q4_unfused_stream_matches_dense_metal() {
    require_kill_switch_unset();
    let cfg = test_config();
    let (mut mq, _) = build_maps();
    let (_, mut md) = build_maps();
    let enc_q = Encoder::from_map(&mut mq, cfg.clone(), "q4-uf").expect("q4 encoder");
    let enc_d = Encoder::from_map(&mut md, cfg, "dense-uf").expect("dense encoder");

    // Folds OFF: the residual add and the GLU ride the unfused
    // then_add / then_glu streams on the Q4 raw bytes.
    let m = Metal::new()
        .expect("metal backend")
        .with_folds(false, false);
    enc_q.warm(&m);
    enc_d.warm(&m);

    let ids = ids_for(19, 97);
    m.begin_pass();
    let out_q = enc_q.forward(&m, &ids).expect("q4 forward");
    let mut host_q = vec![0f32; 19 * 64];
    m.download_into(&out_q, &mut host_q);

    m.begin_pass();
    let out_d = enc_d.forward(&m, &ids).expect("dense forward");
    let mut host_d = vec![0f32; 19 * 64];
    m.download_into(&out_d, &mut host_d);

    assert_eq!(host_q, host_d, "unfused-stream q4 forward bit-identical");
}

/// Resident vs kill-switch (the load kernel + MPS over the F32 `Wᵀ`),
/// in ONE process — the q8 gate's posture gate at Q4 fidelity: the
/// dispatch trees differ on the dense shapes by design (option (i)), so
/// the hidden-state delta is MEASURED and asserted G5-class, and the
/// counters pin the mechanism (the fused arm never widens; the
/// kill-switch arm never fuses).
#[test]
fn q4_resident_matches_device_f32_posture() {
    let _env = env_lock();
    require_kill_switch_unset();
    let cfg = test_config();
    let (mut mq, _) = build_maps();
    let (mut mp1, _) = build_maps();
    let enc_q = Encoder::from_map(&mut mq, cfg.clone(), "q4-res").expect("q4 encoder");
    let enc_p1 = Encoder::from_map(&mut mp1, cfg, "q4-p1").expect("q4 encoder");

    let m = Metal::new().expect("metal backend");
    let seqs = [3usize, 17, 1, 2, 8];
    let total: usize = seqs.iter().sum();
    let ids = ids_for(total, 97);

    // Arm A: the default device-resident fused posture.
    enc_q.warm(&m);
    m.begin_pass();
    let out_q = enc_q.forward_packed(&m, &ids, &seqs).expect("q4 forward");
    let mut host_q = vec![0f32; total * 64];
    m.download_into(&out_q, &mut host_q);
    let fused_after_a = m.q4_fused_dispatches();
    assert!(
        fused_after_a > 0,
        "the resident arm must dispatch the fused staging kernels"
    );
    assert_eq!(
        m.q4_widen_dispatches(),
        0,
        "the resident arm must never build the widened Wᵀ"
    );

    // Arm B: the kill-switch (load kernel + MPS over the F32 Wᵀ).
    // SAFETY: sequential single-threaded flip under the env lock;
    // restored before the test returns.
    unsafe { std::env::set_var("LAYA_Q8_DEVICE_F32", "1") };
    enc_p1.warm(&m);
    m.begin_pass();
    let out_p1 = enc_p1.forward_packed(&m, &ids, &seqs).expect("q4 forward");
    let mut host_p1 = vec![0f32; total * 64];
    m.download_into(&out_p1, &mut host_p1);
    // SAFETY: restore before the counter assertions (see the set above).
    unsafe { std::env::remove_var("LAYA_Q8_DEVICE_F32") };
    let widen_total = m.q4_widen_dispatches();
    assert!(
        widen_total > 0,
        "the kill-switch arm must dispatch the load kernel at warm"
    );
    assert_eq!(
        m.q4_fused_dispatches(),
        fused_after_a,
        "the kill-switch arm must stay off the fused path"
    );

    // The drift: measured, printed, asserted G5-class (the q8 gate's own
    // budget on the raw hidden states).
    let max_diff: f32 = host_q
        .iter()
        .zip(host_p1.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    let scale: f32 = host_q.iter().fold(0.0f32, |a, v| a.max(v.abs())).max(1e-9);
    println!(
        "q4 resident vs device-f32 (packed hidden states): max abs {max_diff:.3e} · scale {scale:.3e}"
    );
    assert!(
        max_diff <= 1e-4 * scale.max(1e-3),
        "the option-(i) dispatch delta must stay G5-class: {max_diff:.4e}"
    );
}

/// The load kernel directly (kill-switch posture): `q4_widen_t`'s Wᵀ
/// read out through an identity matmul (`I_k @ Wᵀ == Wᵀ`) and compared
/// element-wise against the host nibble widen transposed — the sharpest
/// unit gate. The shape carries the k-tail (k = 68; its tail block is
/// byte-aligned, never split).
#[test]
fn q4_widen_kernel_transpose_matches_host() {
    let _env = env_lock();
    require_kill_switch_unset();
    const N: usize = 3;
    const K: usize = 68; // 2 full blocks + a 4-wide tail
    let data: Vec<f32> = fill(N * K, 77);
    let raw = RawQ4::new(N * K, q4_bytes_for(&data)).expect("blocked layout");

    let m = Metal::new().expect("metal backend");
    // SAFETY: sequential single-threaded flip under the env lock;
    // restored before the test returns.
    unsafe { std::env::set_var("LAYA_Q8_DEVICE_F32", "1") };
    m.warm_weight_2d_q4(&raw, N, K);
    assert!(
        m.q4_widen_dispatches() > 0,
        "the kill-switch warm must dispatch the load kernel"
    );

    // Identity activations: dst = I_K @ Wᵀ = Wᵀ (row-major [k, n]).
    let mut a = vec![0f32; K * K];
    for i in 0..K {
        a[i * K + i] = 1.0;
    }
    m.begin_pass();
    let mut got = vec![0f32; K * N];
    m.matmul_w_q4(&a, K, K, &raw, N, &mut got);
    // SAFETY: restore before the assertion tail (see the set above).
    unsafe { std::env::remove_var("LAYA_Q8_DEVICE_F32") };
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
        "the device Wᵀ is the host widen transposed (q4)"
    );
}
