//! The Q8 residency bit-identity gate (Plan 616 Phase 1 + 2): a weight
//! carried as RAW blocked Q8_0 bytes must produce byte-identical
//! forwards to the same values widened on host — through the CPU lane
//! (the once-only host widening) AND through the Metal lane (Phase 2's
//! device-resident RAW bytes + the fused `sgemm_q8`/`sgemm_xwide_q8`/
//! `sgemm_splitk_q8` staging by default; the `q8_widen_t` load kernel
//! under `LAYA_Q8_DEVICE_F32=1`).
//!
//! THE DISPATCH-TREE LAW (Phase 2, option (i)): under the default
//! posture the dense weight shapes MPS serves on the F32 `Wᵀ` run OUR
//! fused instances instead — a T13-class accumulation-order change, the
//! priced cost of the memory win. So the BIT-identity pairs here run on
//! a `.with_mps(false)` instance: both arms then share one dispatch
//! tree and byte-equality is the carrier proof. The default-posture
//! delta is bounded by the same G5-class budget the split rule already
//! carries, and the live probe measures it explicitly.
//!
//! The construction mirrors the q8 artifact converter's own proof: every
//! GEMM tensor's Q8 payload is the SERIALIZED fake-quant of the seed
//! fill, so the Q8 carrier's decoded values are exactly the Dense
//! twin's payload — the two encoders differ in CARRIER only, never in
//! values. Any byte difference in the outputs is therefore a defect of
//! the carrier path (the widen arithmetic, the fused staging, the
//! caches), never of the test data.
//!
//! The geometry exercises every GEMM op family (`matmul_w`,
//! `matmul_w_accum` incl. its fold + unfused arms, `matmul_w_glu`), the
//! packed mixed-plan fallback, the k-tail block (`act_head.0.weight` is
//! [256, d+4] — k % 32 != 0), the head's eight plain GEMMs, the warm
//! path AND the first-miss path (no warm call).
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

/// The kill switches must be unset — under `LAYA_Q8_HOST_F32=1` the Q8
/// carrier widens at load and this whole gate would compare Dense to
/// Dense (a green zero wearing the gate's name); under
/// `LAYA_Q8_DEVICE_F32=1` the default tests would exercise the Phase 1
/// load-kernel path instead of the Phase 2 fused staging.
fn require_kill_switch_unset() {
    assert!(
        std::env::var("LAYA_Q8_HOST_F32").as_deref() != Ok("1"),
        "LAYA_Q8_HOST_F32=1 forces the host widen at load — unset it before \
         running the Q8 identity gate (it would otherwise prove nothing)"
    );
    assert!(
        std::env::var("LAYA_Q8_DEVICE_F32").as_deref() != Ok("1"),
        "LAYA_Q8_DEVICE_F32=1 forces the Phase 1 device-F32 posture — unset \
         it so the default tests exercise the fused staging (the kill-switch \
         has its own dedicated arms)"
    );
}

/// The env-flip lock: the tests that flip `LAYA_Q8_DEVICE_F32` hold it
/// so two flippers cannot interleave their windows. The per-call live
/// read makes a flip a re-route, never a bit change (both paths are
/// bit-identical per tree), so the non-flipping tests need no lock.
fn env_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
    LOCK.get_or_init(|| std::sync::Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
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

    let m = Metal::new()
        .expect("metal backend")
        // The bit-identity pair shares ONE dispatch tree: MPS-off re-bases
        // the split rule to DEFAULT for both arms (the f32 twin and the
        // fused-q8 carrier) and keeps MPS off the dense shapes — under the
        // default posture the two trees differ there by design (option
        // (i), the T13-class accumulation-order delta; the module doc
        // carries the law).
        .with_mps(false);
    // The default posture: folds ON (the split-K epilogue rungs) on the
    // shared tree.
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

/// Phase 2 vs Phase 1, byte-for-byte, in ONE process: the device-resident
/// fused posture (default) vs `LAYA_Q8_DEVICE_F32=1` (the load kernel +
/// MPS over the F32 `Wᵀ`). The two postures' DISPATCH TREES differ on the
/// dense shapes by design (option (i)) — this gate pins the DELTA as
/// G5-class: the split/small shapes stay bit-identical (same kernels,
/// same slice chains), the dense shapes may move last bits, and the
/// ENCODER+HEAD answers are asserted equal at a drift budget two orders
/// under the G5 gate (measured, printed). The counters pin the mechanism:
/// the fused arm dispatches fused and never widens; the kill-switch arm
/// widens and never fuses. (The MPS-ON dense shapes of the kill-switch
/// arm are the T13 posture — the widen output is the same f32 buffer MPS
/// consumed all along.)
#[test]
fn q8_resident_matches_device_f32_posture() {
    let _env = env_lock();
    require_kill_switch_unset();
    let cfg = test_config();
    let (mut mq, _) = build_maps();
    let (mut mp1, _) = build_maps();
    let enc_q = Encoder::from_map(&mut mq, cfg.clone(), "q8-res").expect("q8 encoder");
    let enc_p1 = Encoder::from_map(&mut mp1, cfg, "q8-p1").expect("q8 encoder");

    let m = Metal::new().expect("metal backend");
    let seqs = [3usize, 17, 1, 2, 8];
    let total: usize = seqs.iter().sum();
    let ids = ids_for(total, 97);

    // Arm A: the default device-resident fused posture.
    enc_q.warm(&m);
    m.begin_pass();
    let out_q = enc_q.forward_packed(&m, &ids, &seqs).expect("q8 forward");
    let mut host_q = vec![0f32; total * 64];
    m.download_into(&out_q, &mut host_q);
    let fused_after_a = m.q8_fused_dispatches();
    assert!(
        fused_after_a > 0,
        "the resident arm must dispatch the fused staging kernels"
    );
    assert_eq!(
        m.q8_widen_dispatches(),
        0,
        "the resident arm must never build the widened Wᵀ"
    );

    // Arm B: the Phase 1 kill-switch (load kernel + MPS over the F32 Wᵀ).
    // SAFETY: sequential single-threaded flip under the env lock;
    // restored before the test returns.
    unsafe { std::env::set_var("LAYA_Q8_DEVICE_F32", "1") };
    enc_p1.warm(&m);
    m.begin_pass();
    let out_p1 = enc_p1.forward_packed(&m, &ids, &seqs).expect("q8 forward");
    let mut host_p1 = vec![0f32; total * 64];
    m.download_into(&out_p1, &mut host_p1);
    // SAFETY: restore before the counter assertions (see the set above).
    unsafe { std::env::remove_var("LAYA_Q8_DEVICE_F32") };
    let widen_total = m.q8_widen_dispatches();
    assert!(
        widen_total > 0,
        "the kill-switch arm must dispatch the load kernel at warm"
    );
    assert_eq!(
        m.q8_fused_dispatches(),
        fused_after_a,
        "the kill-switch arm must stay off the fused path"
    );

    // The drift: split/small shapes bit-identical is NOT asserted here —
    // the packed forward mixes segments and the per-element diffs are
    // what the budget bounds. Measure, print, assert two orders under
    // the G5 gate (1e-3 on probs; here the raw hidden states).
    let max_diff: f32 = host_q
        .iter()
        .zip(host_p1.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    let scale: f32 = host_q.iter().fold(0.0f32, |a, v| a.max(v.abs())).max(1e-9);
    println!(
        "resident vs device-f32 (packed hidden states): max abs {max_diff:.3e} · scale {scale:.3e}"
    );
    assert!(
        max_diff <= 1e-4 * scale.max(1e-3),
        "the option-(i) dispatch delta must stay G5-class: {max_diff:.4e}"
    );
}

/// The load kernel directly (Phase 1 posture, under the kill-switch):
/// `q8_widen_t`'s Wᵀ read out through an identity-matmul
/// (`I_k @ Wᵀ == Wᵀ`) and compared element-wise against the host widen
/// transposed — the sharpest unit gate, so a divergence is diagnosable
/// at the kernel without re-deriving it from a red forward. The shape
/// carries the k-tail (k = 68). (Under the default Phase 2 posture the
/// fused GEMM computes the same product from the raw bytes — this gate
/// pins the LOAD KERNEL itself.)
#[test]
fn q8_widen_kernel_transpose_matches_host() {
    let _env = env_lock();
    require_kill_switch_unset();
    const N: usize = 3;
    const K: usize = 68; // 2 full blocks + a 4-wide tail
    let data: Vec<f32> = fill(N * K, 77);
    let raw = RawQ8::new(N * K, q8_bytes_for(&data)).expect("blocked layout");

    let m = Metal::new().expect("metal backend");
    // SAFETY: sequential single-threaded flip under the env lock;
    // restored before the test returns.
    unsafe { std::env::set_var("LAYA_Q8_DEVICE_F32", "1") };
    m.warm_weight_2d_q8(&raw, N, K);
    assert!(
        m.q8_widen_dispatches() > 0,
        "the kill-switch warm must dispatch the load kernel"
    );

    // Identity activations: dst = I_K @ Wᵀ = Wᵀ (row-major [k, n]).
    let mut a = vec![0f32; K * K];
    for i in 0..K {
        a[i * K + i] = 1.0;
    }
    m.begin_pass();
    let mut got = vec![0f32; K * N];
    m.matmul_w_q8(&a, K, K, &raw, N, &mut got);
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
        "the device Wᵀ is the host widen transposed"
    );
}

// ── The live artifact probe (#[ignore]d — real checkpoint, ~447 MB
// artifact + Metal warm, minutes) ─────────────────────────────────────

/// The REAL english q8 artifact through the THREE load postures in ONE
/// process, plus the Phase 2 measurement columns:
///
/// - Arm A: the DEFAULT device-resident FUSED posture (raw bytes on the
///   device, the fused staging kernels, MPS off on the weight shapes).
/// - Arm B: `LAYA_Q8_DEVICE_F32=1` — Phase 1 (load kernel → F32 `Wᵀ`,
///   MPS over the dense shapes).
/// - Arm C: `LAYA_Q8_HOST_F32=1` — Phase 0 (host widen at load; the f32
///   carrier, MPS over the dense shapes).
///
/// THE DISPATCH-TREE LAW (Plan 616 option (i)): A and B differ on the
/// dense weight shapes by design — our fused instances vs MPS, the same
/// T13-class accumulation-order change T13 itself re-seated when MPS was
/// promoted. B and C share ONE tree (the widened Wᵀ is a plain f32
/// buffer to MPS) and must be BYTE-identical — the Phase 1 gate, kept.
/// The A-vs-B delta is MEASURED and asserted two orders under the G5
/// gate; the per-device determinism of the fused posture is pinned by a
/// byte-identical double run of Arm A. RSS prints per arm (sequential
/// loads with drops); the paired timing A/B interleaves A and B
/// position-balanced (the Issue-021 latency law — quote the box state).
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
    use std::time::Instant;

    require_kill_switch_unset();
    // SAFETY: sequential env flips driving the three load postures; no
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
    let state = serde_json::json!(
        "The quarterly revenue forecast beat every analyst estimate. Shares \
         rose after the central bank held rates steady."
    );
    let qdef = serde_json::json!({
        "type": "score",
        "instructions": "Rate the market sentiment from 0 (very negative) to 4 (very positive).",
        "criteria": ["very negative", "negative", "neutral", "positive", "very positive"]
    });
    let ask = |agent: &RiirAgent| agent.forward_question(&state, &qdef).expect("forward runs");
    let show = |label: &str, probs: &[f32], conf: f64, act: &[f32]| {
        println!("[{label}] probs {probs:?} conf {conf:.6} act {act:?}");
    };

    // ── Arm A: the default fused-resident posture, run TWICE — the
    // fused path's per-device determinism pin.
    let (probs_a, act_a, conf_a);
    {
        let agent = RiirAgent::load_with_device(&root, ck, DeviceKind::Metal)
            .expect("fused-resident agent loads");
        rss_mib("A fused-resident (alone)");
        let f1 = ask(&agent);
        let f2 = ask(&agent);
        show("A.1 fused", &f1.probs, f1.confidence, &f1.act_probabilities);
        show(
            "A.2 fused (determinism)",
            &f2.probs,
            f2.confidence,
            &f2.act_probabilities,
        );
        assert_eq!(
            f1.probs, f2.probs,
            "the fused posture must be run-to-run byte-identical"
        );
        assert_eq!(f1.act_probabilities, f2.act_probabilities);
        assert_eq!(f1.confidence, f2.confidence);
        probs_a = f1.probs.clone();
        act_a = f1.act_probabilities.clone();
        conf_a = f1.confidence;
        // DROP before the next load: the agent's Vecs munmap, so the
        // next posture's reading is its own, not a sum of both.
    }

    // ── Arm B: the Phase 1 kill-switch.
    unsafe { std::env::set_var("LAYA_Q8_DEVICE_F32", "1") };
    let (probs_b, act_b, conf_b);
    let mut agent_b;
    {
        let agent = RiirAgent::load_with_device(&root, ck, DeviceKind::Metal)
            .expect("device-f32 agent loads");
        rss_mib("B device-f32 / Phase 1 (alone after drop)");
        let f = ask(&agent);
        show("B device-f32", &f.probs, f.confidence, &f.act_probabilities);
        probs_b = f.probs.clone();
        act_b = f.act_probabilities.clone();
        conf_b = f.confidence;
        agent_b = agent; // kept alive for the timing A/B below
    }
    unsafe { std::env::remove_var("LAYA_Q8_DEVICE_F32") };

    // The option-(i) dispatch delta: measured, bounded two orders under
    // the G5 gate. (T13 precedent: the MPS promotion itself re-seated
    // the same class; Phase 3 adoption re-seats the q8-posture cells at
    // these bytes. The SHIPPED F16 posture is untouched — no frozen bit
    // moves for the seated cells.)
    let drift = |label: &str, x: &[f32], y: &[f32]| {
        let d = x
            .iter()
            .zip(y.iter())
            .map(|(p, q)| (p - q).abs())
            .fold(0.0f32, f32::max);
        println!("drift {label}: max abs {d:.3e}");
        d
    };
    let d_probs = drift("A-vs-B probs", &probs_a, &probs_b);
    let d_act = drift("A-vs-B act", &act_a, &act_b);
    let d_conf = (conf_a - conf_b).abs();
    println!("drift A-vs-B conf: max abs {d_conf:.3e}");
    assert!(
        d_probs <= 1e-5,
        "probs drift must stay two orders under G5: {d_probs:.4e}"
    );
    assert!(
        d_act <= 1e-5,
        "act drift must stay two orders under G5: {d_act:.4e}"
    );
    assert!(
        d_conf <= 1e-5,
        "conf drift must stay two orders under G5: {d_conf:.4e}"
    );

    // ── Arm C: the Phase 0 host-widen — byte-identical to B (one tree).
    unsafe { std::env::set_var("LAYA_Q8_HOST_F32", "1") };
    {
        let agent = RiirAgent::load_with_device(&root, ck, DeviceKind::Metal)
            .expect("host-widen agent loads");
        rss_mib("C host-widen / Phase 0 (alone after drop)");
        let f = ask(&agent);
        show("C host-widen", &f.probs, f.confidence, &f.act_probabilities);
        assert_eq!(
            probs_b, f.probs,
            "B vs C probabilities bit-identical (one tree)"
        );
        assert_eq!(act_b, f.act_probabilities, "B vs C act bit-identical");
        assert_eq!(conf_b, f.confidence, "B vs C confidence bit-identical");
    }
    unsafe { std::env::remove_var("LAYA_Q8_HOST_F32") };

    // ── The paired timing A/B (position-balanced interleave, both agents
    // warm; the Issue-021 law — quote the box state in the record).
    println!("timing A/B (fused-resident A vs device-f32 B), position-balanced, 8 rounds:");
    let mut a_ms: Vec<f64> = Vec::new();
    let mut b_ms: Vec<f64> = Vec::new();
    let mut agent_a =
        RiirAgent::load_with_device(&root, ck, DeviceKind::Metal).expect("fused agent for timing");
    let _warm = (ask(&agent_a), ask(&agent_b));
    for r in 0..8 {
        let (first, second) = if r % 2 == 0 {
            (&mut agent_a, &mut agent_b)
        } else {
            (&mut agent_b, &mut agent_a)
        };
        let t = Instant::now();
        let _ = ask(first);
        let df = t.elapsed().as_secs_f64() * 1e3;
        let t = Instant::now();
        let _ = ask(second);
        let ds = t.elapsed().as_secs_f64() * 1e3;
        if r % 2 == 0 {
            a_ms.push(df);
            b_ms.push(ds);
        } else {
            b_ms.push(df);
            a_ms.push(ds);
        }
    }
    let median = |v: &mut Vec<f64>| {
        v.sort_by(|x, y| x.partial_cmp(y).unwrap());
        v[v.len() / 2]
    };
    let (ma, mb) = (median(&mut a_ms), median(&mut b_ms));
    println!(
        "paired medians: fused-resident {ma:.1} ms · device-f32 {mb:.1} ms · ratio {:.3}",
        ma / mb
    );
    println!(
        "LIVE PROBE: fused-resident deterministic ×2; Phase-1 tree byte-identical \
         (B==C); the option-(i) dispatch delta bounded at {d_probs:.2e} probs"
    );
}

/// The DEVICE-residency column (Plan 616 Phase 2's headline): load the
/// REAL english q8 weight map directly (the agent's own load path minus
/// the tokenizer), warm one Metal instance per posture on the FULL
/// encoder weight set, and read [`Metal::device_allocated_bytes`] — the
/// GPU pool this process charged. Expected: the fused-resident posture
/// holds the raw blocked bytes (numel × 1.0625 — Phase 0's 0.448 GB
/// derivation) where the kill-switch posture holds the widened F32 `Wᵀ`
/// (numel × 4 — the measured 1.685 GB). Instances are sequential
/// (one-live-instance); the load-kernel dispatches of the kill-switch
/// arm are the posture's own warm cost. #[ignore]d — the ~447 MB
/// artifact + warm, ~a minute.
///
/// ```sh
/// cargo test --release --features laya-riir-metal --test q8_widen_identity \
///   live_q8_device_bytes -- --ignored --nocapture
/// ```
#[test]
#[ignore]
fn live_q8_device_bytes_resident_vs_widened() {
    use riir_infer_laya::laya::config::{Checkpoint, load_checkpoint_configs};
    use riir_infer_laya::laya::riir::backend::Backend;
    use riir_infer_laya::laya::riir::encoder::Encoder;
    use riir_infer_laya::laya::riir::q8_artifact::resolve_weights_file;
    use riir_infer_laya::laya::riir::weights as w;

    require_kill_switch_unset();
    // SAFETY: sequential env flips driving the two postures; no other
    // thread reads these during the probe.
    unsafe { std::env::set_var("LAYA_WEIGHTS_VARIANT", "q8") };
    let root = riir_infer_laya::laya::weights::weights_root();
    let ck = Checkpoint::English;
    let name = ck.subfolder();
    let dir = root.join(name);
    let (weights_path, q8) = resolve_weights_file(&dir, name).expect("weights file");
    assert!(q8, "the probe needs the q8 variant resolved");
    let (_, enc_cfg) = load_checkpoint_configs(&dir, name).expect("configs");

    let warm_probe = |label: &str| {
        let mut map = w::load(&weights_path, name).expect("weight map");
        let enc = Encoder::from_map(&mut map, enc_cfg.clone(), name).expect("encoder");
        let m = Metal::new().expect("metal backend");
        enc.warm(&m);
        // Drain the warm's widen dispatches before reading (the serial
        // queue does not need it for the byte count, but the residency
        // number should describe SETTLED state).
        m.begin_pass();
        let bytes = m.device_allocated_bytes();
        println!(
            "[{label}] device_allocated = {:.1} MiB",
            bytes as f64 / 1048576.0
        );
        m.begin_pass();
        bytes
    };

    let resident = warm_probe("A fused-resident (default)");

    // SAFETY: sequential single-threaded flip; restored below.
    unsafe { std::env::set_var("LAYA_Q8_DEVICE_F32", "1") };
    let widened = warm_probe("B device-f32 / Phase 1");
    unsafe { std::env::remove_var("LAYA_Q8_DEVICE_F32") };

    let ratio = widened as f64 / resident.max(1) as f64;
    println!(
        "DEVICE RESIDENCY: widened/resident = {ratio:.2}× (widened {:.1} MiB vs raw {:.1} MiB)",
        widened as f64 / 1048576.0,
        resident as f64 / 1048576.0
    );
    // The resident posture must hold strictly less: the raw blocked
    // bytes (1.0625 B/elt) replace the widened F32 Wᵀ (4 B/elt) for the
    // 126 GEMM tensors. The 1D norms upload identically in both.
    assert!(
        resident < widened,
        "device residency must drop under the fused-resident posture"
    );
}
