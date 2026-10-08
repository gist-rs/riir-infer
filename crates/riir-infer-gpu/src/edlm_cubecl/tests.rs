//! Issue 1005 T8 tests — kernel units + the tiny-weights full-pipeline
//! parity vs the CPU T7 cached path (the lane's gate) + the real-Q8_0
//! env-gated published-sample gate (skip-loud without `EDLM_GGUF`).
//!
//! Tolerance law: GPU-vs-CPU is NEVER bit-identity (f16 weight rounding +
//! online-softmax accumulation order). The tiny parity test QUANTIZES the
//! tiny weights to the f16 grid FIRST and runs the CPU oracle on the SAME
//! rounded weights — isolating kernel/pipeline error (tight gate) from the
//! weight-rounding face (measured and disclosed on the real model).

use super::*;

use crate::cubecl_runtime::{ActiveComputeClient, CubeCLContext, create_f32, read_f32};
use riir_infer_core::transformer::edlm::{
    EdlmLayerWeights, forward_edlm_branches_cached, edlm_state_prefill,
};
use riir_infer_core::types::Config;

// ── fixtures ────────────────────────────────────────────────────────

struct Lcg(u64);
impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f32 / (1u64 << 31) as f32) - 1.0
    }
    fn mat(&mut self, rows: usize, cols: usize, scale: f32) -> Vec<f32> {
        (0..rows * cols).map(|_| self.next_f32() * scale).collect()
    }
    fn gamma(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| 1.0 + 0.1 * self.next_f32()).collect()
    }
}

/// The attention cube is head_dim 128, so the tiny config keeps the 8B
/// release's head shape at a 2-layer/512-dim body.
fn tiny_config() -> Config {
    let mut c = Config::micro();
    c.vocab_size = 64;
    c.block_size = 256;
    // Qwen3 shape law: n_embd == n_head * head_dim.
    c.n_embd = 512;
    c.n_head = 4;
    c.n_kv_head = 2;
    c.head_dim = 128;
    c.mlp_hidden = 1024;
    c.n_layer = 2;
    c.rms_norm_eps = 1e-6;
    c.rope_theta = 1_000_000.0;
    c
}

fn tiny_weights(config: &Config) -> EdlmWeights {
    let mut r = Lcg(0x5EED_1005);
    let mut layers = Vec::new();
    for _ in 0..config.n_layer {
        layers.push(EdlmLayerWeights {
            base: riir_infer_core::llama_layer::LlamaLayerWeights {
                attn_wq: r.mat(config.n_head * config.head_dim, config.n_embd, 0.2),
                attn_wk: r.mat(config.n_kv_head * config.head_dim, config.n_embd, 0.2),
                attn_wv: r.mat(config.n_kv_head * config.head_dim, config.n_embd, 0.2),
                attn_wo: r.mat(config.n_embd, config.n_head * config.head_dim, 0.2),
                gate_proj: r.mat(config.mlp_hidden, config.n_embd, 0.2),
                up_proj: r.mat(config.mlp_hidden, config.n_embd, 0.2),
                down_proj: r.mat(config.n_embd, config.mlp_hidden, 0.2),
                input_norm: r.gamma(config.n_embd),
                post_attn_norm: r.gamma(config.n_embd),
            },
            q_norm: r.gamma(config.head_dim),
            k_norm: r.gamma(config.head_dim),
        });
    }
    EdlmWeights {
        wte: r.mat(config.vocab_size, config.n_embd, 0.5),
        final_norm: r.gamma(config.n_embd),
        lm_head: None,
        layers,
        pointer: None,
    }
}

/// Round every weight to the f16 grid (the CPU oracle then runs on the SAME
/// weights the GPU holds — kernel/pipeline error is isolated from the
/// weight-rounding face).
fn quantize_to_f16_grid(w: &mut EdlmWeights) {
    let q = |v: &mut Vec<f32>| {
        for x in v.iter_mut() {
            *x = half_f16::from_f32(*x).to_f32();
        }
    };
    q(&mut w.wte);
    q(&mut w.final_norm);
    for l in &mut w.layers {
        q(&mut l.base.attn_wq);
        q(&mut l.base.attn_wk);
        q(&mut l.base.attn_wv);
        q(&mut l.base.attn_wo);
        q(&mut l.base.gate_proj);
        q(&mut l.base.up_proj);
        q(&mut l.base.down_proj);
        q(&mut l.base.input_norm);
        q(&mut l.base.post_attn_norm);
        q(&mut l.q_norm);
        q(&mut l.k_norm);
    }
}

/// state(5) | q1: [instr, opt, decide] | q2: [instr, opt, opt, opt, decide]
/// — the same layout law the core tests pin.
fn sample_encoding() -> PackedEncoding {
    let s = 0i32;
    let mut ids = Vec::new();
    let mut pos = Vec::new();
    let mut seg = Vec::new();
    for i in 0..5 {
        ids.push(i);
        pos.push(i);
        seg.push(s);
    }
    let mut decide_idx = Vec::new();
    let mut opt_idx = Vec::new();
    // q1 branch: len 3
    for bi in 0..2 {
        ids.push(10 + bi);
        pos.push(5 + bi);
        seg.push(1);
    }
    ids.push(63);
    pos.push(5 + 2);
    seg.push(1);
    decide_idx.push(ids.len() - 1);
    opt_idx.push(vec![ids.len() - 2]);
    // q2 branch: len 5
    for bi in 0..4 {
        ids.push(20 + bi);
        pos.push(5 + bi);
        seg.push(2);
    }
    ids.push(62);
    pos.push(5 + 4);
    seg.push(2);
    decide_idx.push(ids.len() - 1);
    opt_idx.push(vec![ids.len() - 3, ids.len() - 2]);
    PackedEncoding {
        ids,
        pos,
        seg,
        opt: Vec::new(),
        state_len: 5,
        state_truncated: false,
        decide_idx,
        opt_idx,
    }
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

// ── kernel units ────────────────────────────────────────────────────

#[test]
fn rmsnorm_rows_matches_cpu_reference() {
    let ctx = CubeCLContext::new().expect("CubeCL should initialize");
    let client = ctx.client();

    let mut r = Lcg(0xBEEF);
    let rows = 7usize;
    let n = 512usize;
    let eps = 1e-6f32;
    let x: Vec<f32> = (0..rows * n).map(|_| r.next_f32() * 2.0).collect();
    let gamma: Vec<f32> = (0..n).map(|_| 1.0 + 0.1 * r.next_f32()).collect();

    let mut want = x.clone();
    for chunk in want.chunks_exact_mut(n) {
        rmsnorm_with_gamma_eps(chunk, &gamma, eps as f64);
    }

    let x_h = create_f32(&client, &x);
    let g_h = create_f32(&client, &gamma);
    let out_h = client.empty(rows * n * core::mem::size_of::<f32>());
    EdlmRmsNormRowsCubeCL::launch::<ActiveRuntime>(
        &client,
        x_h,
        g_h,
        out_h.clone(),
        rows,
        n,
        eps,
    );
    let got = read_f32(&client, out_h).expect("read");
    let worst = max_abs_diff(&got, &want);
    assert!(
        worst < 1e-5,
        "rmsnorm rows drift {worst} (reduction-order class, n={n})"
    );
}

#[test]
fn attn_multi_matches_cpu_reference_with_visibility_bounds() {
    let ctx = CubeCLContext::new().expect("CubeCL should initialize");
    let client = ctx.client();

    // 8 Q heads / 2 KV heads / head_dim 128 (the kernel's cube shape).
    let n_head = 8usize;
    let n_kv = 2usize;
    let hd = 128usize;
    let sq = 5usize;
    let n_pos = 9usize;
    let kvd = n_kv * hd;
    let q_dim = n_head * hd;
    let scale = 1.0 / (hd as f32).sqrt();

    let mut r = Lcg(0xC0FFEE);
    let q: Vec<f32> = (0..sq * q_dim).map(|_| r.next_f32()).collect();
    // K rows scaled so softmax isn't degenerate.
    let keys: Vec<f32> = (0..n_pos * kvd).map(|_| 0.2 * r.next_f32()).collect();
    let values: Vec<f32> = (0..n_pos * kvd).map(|_| r.next_f32()).collect();
    // Visibility: some rows causal-with-prefix, some full — the two laws the
    // model pass emits (branch continuation + bidir state).
    let t_n: Vec<u32> = vec![1, 3, 5, 9, 9];

    // CPU reference: query row i attends keys 0..t_n(i), GQA kvh = h*n_kv/n_head.
    let mut want = vec![0.0f32; sq * q_dim];
    for qi in 0..sq {
        for h in 0..n_head {
            let kvh = h * n_kv / n_head;
            let vis = t_n[qi] as usize;
            let mut scores = Vec::with_capacity(vis);
            for j in 0..vis {
                let mut dot = 0.0f32;
                for t in 0..hd {
                    dot += q[qi * q_dim + h * hd + t] * keys[j * kvd + kvh * hd + t];
                }
                scores.push(dot * scale);
            }
            let mx = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = scores.iter().map(|&s| (s - mx).exp()).collect();
            let sum: f32 = exps.iter().sum();
            for (j, &e) in exps.iter().enumerate() {
                for d in 0..hd {
                    want[qi * q_dim + h * hd + d] +=
                        e / sum * values[j * kvd + kvh * hd + d];
                }
            }
        }
    }

    let mut kv_combined = keys;
    kv_combined.extend_from_slice(&values);
    let q_h = create_f32(&client, &q);
    let kv_h = create_f32(&client, &kv_combined);
    let tn_h = create_u32(&client, &t_n);
    let out_h = client.empty(sq * q_dim * core::mem::size_of::<f32>());
    EdlmAttnMultiCubeCL::launch::<ActiveRuntime>(
        &client,
        q_h,
        kv_h,
        tn_h,
        out_h.clone(),
        &EdlmAttnMultiParams {
            n_head,
            n_kv_head: n_kv,
            head_dim: hd,
            n_positions: n_pos,
            seq_q: sq,
            scale,
        },
    );
    let got = read_f32(&client, out_h).expect("read");

    // Partial-tile masking (row 1: vis 3 < tile 128) + multi-tile (row 3/4:
    // vis 9 > 128? no — 9 < 128; add a >128 case below) + full rows.
    let worst = max_abs_diff(&got, &want);
    assert!(
        worst < 2e-4,
        "attention drift {worst} (online-vs-plain softmax order class)"
    );
}

#[test]
fn attn_multi_multi_tile_visibility() {
    // vis > 128 exercises the tile loop's rescale path (rows 0/1) — the
    // online-softmax carry across >1 tile is THE class the single-tile test
    // above cannot see.
    let ctx = CubeCLContext::new().expect("CubeCL should initialize");
    let client = ctx.client();

    let n_head = 2usize;
    let n_kv = 1usize;
    let hd = 128usize;
    let sq = 2usize;
    let n_pos = 300usize;
    let kvd = n_kv * hd;
    let q_dim = n_head * hd;
    let scale = 1.0 / (hd as f32).sqrt();

    let mut r = Lcg(0xD00D);
    let q: Vec<f32> = (0..sq * q_dim).map(|_| r.next_f32()).collect();
    let keys: Vec<f32> = (0..n_pos * kvd).map(|_| 0.1 * r.next_f32()).collect();
    let values: Vec<f32> = (0..n_pos * kvd).map(|_| r.next_f32()).collect();
    let t_n: Vec<u32> = vec![129, 300]; // one just over a tile, one 2.34 tiles

    let mut want = vec![0.0f32; sq * q_dim];
    for qi in 0..sq {
        for h in 0..n_head {
            let kvh = h * n_kv / n_head;
            let vis = t_n[qi] as usize;
            let mut scores = Vec::with_capacity(vis);
            for j in 0..vis {
                let mut dot = 0.0f32;
                for t in 0..hd {
                    dot += q[qi * q_dim + h * hd + t] * keys[j * kvd + kvh * hd + t];
                }
                scores.push(dot * scale);
            }
            let mx = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = scores.iter().map(|&s| (s - mx).exp()).collect();
            let sum: f32 = exps.iter().sum();
            for (j, &e) in exps.iter().enumerate() {
                for d in 0..hd {
                    want[qi * q_dim + h * hd + d] += e / sum * values[j * kvd + kvh * hd + d];
                }
            }
        }
    }

    let mut kv_combined = keys;
    kv_combined.extend_from_slice(&values);
    let q_h = create_f32(&client, &q);
    let kv_h = create_f32(&client, &kv_combined);
    let tn_h = create_u32(&client, &t_n);
    let out_h = client.empty(sq * q_dim * core::mem::size_of::<f32>());
    EdlmAttnMultiCubeCL::launch::<ActiveRuntime>(
        &client,
        q_h,
        kv_h,
        tn_h,
        out_h.clone(),
        &EdlmAttnMultiParams {
            n_head,
            n_kv_head: n_kv,
            head_dim: hd,
            n_positions: n_pos,
            seq_q: sq,
            scale,
        },
    );
    let got = read_f32(&client, out_h).expect("read");
    let worst = max_abs_diff(&got, &want);
    assert!(
        worst < 2e-4,
        "multi-tile attention drift {worst} (online carry class)"
    );
}

// ── the pipeline gate ───────────────────────────────────────────────

/// Full-pipeline parity vs the CPU T7 cached path on f16-grid tiny weights,
/// BOTH state_bidir postures (the core lane's own parity law: parity holds
/// in each posture, never across).
fn tiny_pipeline_parity(state_bidir: bool) -> (f32, f32) {
    let config = tiny_config();
    let mut weights = tiny_weights(&config);
    quantize_to_f16_grid(&mut weights);
    let enc = sample_encoding();
    enc.validate().expect("fixture invariants");
    let rows = riir_infer_core::transformer::edlm::rows_of(&enc).expect("rows");

    // CPU oracle: the T7 cached path on the SAME f16-grid weights.
    let sl = enc.state_len;
    let cpu_cache = edlm_state_prefill(
        &weights,
        &config,
        &enc.ids[..sl],
        &enc.pos[..sl],
        state_bidir,
    );
    let cpu_rows =
        forward_edlm_branches_cached(&weights, &config, &enc, &rows, &cpu_cache)
            .expect("cpu branches");

    // GPU: same weights through the f16 upload path.
    let mut gpu = EdlmGpuModel::from_weights(&weights, &config).expect("gpu model");
    gpu.state_prefill(&enc.ids[..sl], &enc.pos[..sl], state_bidir)
        .expect("gpu prefill");
    let gpu_rows = gpu.forward_branches(&enc, &rows).expect("gpu branches");

    // State cache K/V parity (the prefill's own output).
    let gpu_cache = gpu.state_kv().expect("cache").clone();
    let mut kv_worst = 0.0f32;
    for (li, (c, g)) in cpu_cache.layers.iter().zip(&gpu_cache.layers).enumerate() {
        kv_worst = kv_worst.max(max_abs_diff(&c.k, &g.k));
        kv_worst = kv_worst.max(max_abs_diff(&c.v, &g.v));
        let _ = li;
    }
    // Hidden parity (the pointer head's input).
    let mut hid_worst = 0.0f32;
    for (c, g) in cpu_rows.iter().zip(&gpu_rows) {
        hid_worst = hid_worst.max(max_abs_diff(c, g));
    }
    (kv_worst, hid_worst)
}

#[test]
fn tiny_parity_vs_cpu_cached_bidir() {
    let (kv, hid) = tiny_pipeline_parity(true);
    assert!(
        kv < 5e-4 && hid < 2e-3,
        "bidir parity: kv drift {kv}, hidden drift {hid}"
    );
    println!("tiny bidir parity: kv {kv:.3e} hidden {hid:.3e}");
}

#[test]
fn tiny_parity_vs_cpu_cached_causal() {
    let (kv, hid) = tiny_pipeline_parity(false);
    assert!(
        kv < 5e-4 && hid < 2e-3,
        "causal parity: kv drift {kv}, hidden drift {hid}"
    );
    println!("tiny causal parity: kv {kv:.3e} hidden {hid:.3e}");
}

/// Stale/mismatched caches REFUSE (the T7 law carries to the GPU lane).
#[test]
fn forward_branches_refuses_without_prefill() {
    let config = tiny_config();
    let mut weights = tiny_weights(&config);
    quantize_to_f16_grid(&mut weights);
    let enc = sample_encoding();
    let rows = riir_infer_core::transformer::edlm::rows_of(&enc).expect("rows");
    let mut gpu = EdlmGpuModel::from_weights(&weights, &config).expect("gpu model");
    assert!(gpu.forward_branches(&enc, &rows).is_err(), "no cache yet");
}

// ── the real-weights gate (skip-loud without EDLM_GGUF) ────────────

/// T8's real-model gate: the published sample through the GPU lane vs the
/// CPU streaming lane — probs via the CPU pointer head. Law: argmax EXACT +
/// per-option drift < 0.015 (T6's cross-runner tolerance class), measured
/// drift PRINTED. Skip-loud without `EDLM_GGUF` (CC BY-NC weights, local
/// bench only; the Q8_0 GGUF is the M3's copy today — this box runs the
/// tiny gates above until it's copied).
#[test]
fn real_q8_parity_env_gated() {
    let path = match std::env::var("EDLM_GGUF") {
        Ok(p) if !p.is_empty() => p,
        _ => {
            eprintln!(
                "SKIP: EDLM_GGUF unset — set it to drex-dlm-Q8_0.gguf for the T8 GPU parity run"
            );
            return;
        }
    };
    let t_open = std::time::Instant::now();
    let mut gpu = EdlmGpuModel::open(std::path::Path::new(&path)).expect("open edlm gguf");
    let open_s = t_open.elapsed().as_secs_f64();

    // The CPU oracle opens the same file once more (its own mmap) and carries
    // the tokenizer + pointer both sides read.
    let cpu_core =
        riir_infer_core::transformer::edlm::EdlmGgufModel::open(std::path::Path::new(&path))
            .expect("open for cpu oracle");
    let tok = cpu_core.tokenizer().expect("gguf tokenizer");

    let record = riir_infer_core::transformer::edlm::EdlmRecord {
        state: "ticket: I was charged twice for the same order. Please refund the extra payment."
            .into(),
        questions: vec![
            riir_infer_core::transformer::edlm::EdlmQuestion {
                instr: "Which team should handle this ticket?".into(),
                options: vec![
                    "billing: Payments, charges, and refunds".into(),
                    "technical: Bugs and outages".into(),
                    "other: Anything else".into(),
                ],
            },
            riir_infer_core::transformer::edlm::EdlmQuestion {
                instr: "Does the customer explicitly ask for a refund?".into(),
                options: vec!["no".into(), "yes".into()],
            },
            riir_infer_core::transformer::edlm::EdlmQuestion {
                instr: "How urgent is this ticket?".into(),
                options: vec!["Routine".into(), "Soon".into(), "Urgent".into()],
            },
        ],
    };
    let enc = riir_infer_core::transformer::edlm::encode_packed(
        &tok,
        &record,
        riir_infer_core::transformer::edlm::EdlmLimits::serving(16_384),
        true,
        false,
    )
    .expect("encode");
    assert_eq!(enc.ids.len(), 87, "their published input-token count");
    let rows = riir_infer_core::transformer::edlm::rows_of(&enc).expect("rows");

    // CPU oracle (streaming, the T7 lane) — the core opened above.
    let t_cpu = std::time::Instant::now();
    let cpu_cache = riir_infer_core::transformer::edlm::edlm_state_prefill_streaming(
        &cpu_core,
        &enc.ids[..enc.state_len],
        &enc.pos[..enc.state_len],
        true,
    )
    .expect("cpu prefill");
    let cpu_rows = riir_infer_core::transformer::edlm::forward_edlm_branches_cached_streaming(
        &cpu_core,
        &enc,
        &rows,
        &cpu_cache,
    )
    .expect("cpu branches");
    let cpu_s = t_cpu.elapsed().as_secs_f64();

    // GPU lane.
    let sl = enc.state_len;
    let t_gpu = std::time::Instant::now();
    gpu.state_prefill(&enc.ids[..sl], &enc.pos[..sl], true)
        .expect("gpu prefill");
    let prefill_s = t_gpu.elapsed().as_secs_f64();
    let t_gb = std::time::Instant::now();
    let gpu_rows = gpu.forward_branches(&enc, &rows).expect("gpu branches");
    let branches_s = t_gb.elapsed().as_secs_f64();

    // Probs via the SAME CPU pointer head on both hidden sets.
    let pointer = gpu.pointer.as_ref().expect("pointer head in gguf");
    let n = gpu.config.n_embd;
    let read = |hidden: &[f32], q: usize| -> Vec<f32> {
        let d = &hidden[rows[q].decide * n..(rows[q].decide + 1) * n];
        let mut opts = Vec::with_capacity(rows[q].opts.len() * n);
        for &o in &rows[q].opts {
            opts.extend_from_slice(&hidden[o * n..(o + 1) * n]);
        }
        pointer.question_probs(d, &opts, rows[q].opts.len())
    };

    let published: [&[f64]; 3] = [
        &[0.9641, 0.001, 0.0349],  // team: billing / technical / other
        &[1.0 - 0.8548, 0.8548],   // refund: no / yes
        &[0.1876, 0.2169, 0.5955], // urgency: Routine / Soon / Urgent
    ];
    let names = ["team", "refund", "urgency"];
    let mut worst_gpu_vs_cpu = 0.0f32;
    let mut worst_vs_published = 0.0f64;
    for q in 0..3 {
        let cpu_p = read(&cpu_rows[q], q);
        let gpu_p = read(&gpu_rows[q], q);
        for (c, g) in cpu_p.iter().zip(&gpu_p) {
            worst_gpu_vs_cpu = worst_gpu_vs_cpu.max((c - g).abs());
        }
        for (g, w) in gpu_p.iter().zip(published[q]) {
            worst_vs_published = worst_vs_published.max((*g as f64 - w).abs());
        }
        // Argmax EXACT (the law a consumer publishes).
        let am = |p: &[f32]| {
            p.iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0
        };
        assert_eq!(am(&cpu_p), am(&gpu_p), "argmax flip on {}", names[q]);
        println!(
            "{}: gpu {:?} cpu {:?}",
            names[q],
            gpu_p.iter().map(|v| (v * 1e4).round() / 1e4).collect::<Vec<_>>(),
            cpu_p.iter().map(|v| (v * 1e4).round() / 1e4).collect::<Vec<_>>(),
        );
    }
    println!(
        "T8 GPU parity: gpu-vs-cpu prob drift {:.4}, vs-published {:.4} (gate 0.015) | \
         runtime {} | open {:.1}s, cpu {:.1}s, gpu prefill {:.2}s + branches {:.2}s",
        worst_gpu_vs_cpu,
        worst_vs_published,
        gpu.runtime_name(),
        open_s,
        cpu_s,
        prefill_s,
        branches_s,
    );
    assert!(
        worst_gpu_vs_cpu < 0.015 && worst_vs_published < 0.015,
        "T8 parity gate: gpu-vs-cpu {worst_gpu_vs_cpu}, vs-published {worst_vs_published}"
    );
    // The Issue-712 law: this test held ~17 GB of device pages — release them.
    gpu.memory_cleanup();
    crate::test_gpu_support::gpu_release_pages(&gpu_client_for_cleanup());
}

/// The heavy-test serialization gate wants a client; the model's own client
/// is private to the module — build the shared one (the same OnceLock'd
/// context, so this is the SAME client the model used).
fn gpu_client_for_cleanup() -> ActiveComputeClient {
    CubeCLContext::new().expect("CubeCL context for cleanup").client()
}
