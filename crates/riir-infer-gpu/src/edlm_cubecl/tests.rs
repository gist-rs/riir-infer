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

/// state(18) | q1: [instr, opt, decide] | q2: [instr, opt, opt, opt, decide]
/// — the same layout law the core tests pin. The state is 18 tokens ON
/// PURPOSE: the real published sample's state length, and above the GEMM
/// dispatch's M≥16 tensor-core threshold — so the mixed-posture pipeline
/// run exercises cmma (state prefill) AND scalar (the 3/5-token branch
/// rows) in one pass, exactly the shipped dispatch law.
fn sample_encoding() -> PackedEncoding {
    let s = 0i32;
    let mut ids = Vec::new();
    let mut pos = Vec::new();
    let mut seg = Vec::new();
    for i in 0..18 {
        ids.push(i);
        pos.push(i);
        seg.push(s);
    }
    let mut decide_idx = Vec::new();
    let mut opt_idx = Vec::new();
    // q1 branch: len 3
    for bi in 0..2 {
        ids.push(10 + bi);
        pos.push(18 + bi);
        seg.push(1);
    }
    ids.push(63);
    pos.push(18 + 2);
    seg.push(1);
    decide_idx.push(ids.len() - 1);
    opt_idx.push(vec![ids.len() - 2]);
    // q2 branch: len 5
    for bi in 0..4 {
        ids.push(20 + bi);
        pos.push(18 + bi);
        seg.push(2);
    }
    ids.push(62);
    pos.push(18 + 4);
    seg.push(2);
    decide_idx.push(ids.len() - 1);
    opt_idx.push(vec![ids.len() - 3, ids.len() - 2]);
    PackedEncoding {
        ids,
        pos,
        seg,
        opt: Vec::new(),
        state_len: 18,
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

/// The plain-softmax CPU oracle for the two-segment attention kernel: query
/// `i` attends the seed keys `[0, sl)` then own keys `[t_start(i),
/// t_end(i))`, GQA `kvh = h·n_kv/n_head`. Order-independent (plain softmax),
/// so it oracles the seed-then-own online accumulation exactly once for all
/// four kernel units.
#[allow(clippy::too_many_arguments)]
fn cpu_attn_reference(
    q: &[f32],
    seed_k: &[f32],
    seed_v: &[f32],
    own_k: &[f32],
    own_v: &[f32],
    sl: usize,
    t_start: &[u32],
    t_end: &[u32],
    n_head: usize,
    n_kv: usize,
    hd: usize,
) -> Vec<f32> {
    let sq = t_start.len();
    let kvd = n_kv * hd;
    let q_dim = n_head * hd;
    let scale = 1.0 / (hd as f32).sqrt();
    let mut want = vec![0.0f32; sq * q_dim];
    for qi in 0..sq {
        for h in 0..n_head {
            let kvh = h * n_kv / n_head;
            let win = (t_end[qi] - t_start[qi]) as usize;
            let vis = sl + win;
            let mut scores = Vec::with_capacity(vis);
            let mut v_rows: Vec<(&[f32], usize)> = Vec::with_capacity(vis);
            for vk in 0..vis {
                let (k_row, v_row) = if vk < sl {
                    (&seed_k[vk * kvd..], &seed_v[vk * kvd..])
                } else {
                    let j = t_start[qi] as usize + (vk - sl);
                    (&own_k[j * kvd..], &own_v[j * kvd..])
                };
                let mut dot = 0.0f32;
                for t in 0..hd {
                    dot += q[qi * q_dim + h * hd + t] * k_row[kvh * hd + t];
                }
                scores.push(dot * scale);
                v_rows.push((v_row, kvh * hd));
            }
            let mx = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = scores.iter().map(|&s| (s - mx).exp()).collect();
            let sum: f32 = exps.iter().sum();
            for (e, (v_row, vo)) in exps.iter().zip(&v_rows) {
                for d in 0..hd {
                    want[qi * q_dim + h * hd + d] += e / sum * v_row[vo + d];
                }
            }
        }
    }
    want
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

/// The fused QKV fold vs the EXACT host small-ops the v1 path ran between
/// readbacks (split → per-head qk-norm → RoPE → segment scatter). Two rows
/// (3 + 2 queries) so the own-buffer segment scatter is exercised across a
/// row seam; positions include 0 (the host rope's identity fast path — the
/// kernel's general rotation must be bit-exact identity there) and large
/// angles. V passes through EXACTLY (no norm, no rotation — bit-equal); q/k
/// carry the fold's tolerance faces (tree-order sum-of-squares + GPU
/// sin/cos).
#[test]
fn qkv_fold_matches_cpu_reference() {
    let ctx = CubeCLContext::new().expect("CubeCL should initialize");
    let client = ctx.client();

    let n_head = 2usize;
    let n_kv = 1usize;
    let hd = 128usize;
    let eps = 1e-6f32;
    let q_dim = n_head * hd;
    let kvd = n_kv * hd;
    let lq = q_dim + 2 * kvd;
    // Two rows: row A = queries 0..3 (key rows 0..3), row B = queries 3..5
    // (key rows 3..5). The fold kernel's seg input is PER QUERY = row base
    // + in-row offset — UNIQUE per query, exactly the invariant the real
    // passes produce (a repeated slot would be a scheduler-dependent
    // double-write; an unwritten slot reads pool-stale bytes).
    let sq = 5usize;
    let own_total = 5usize;
    let pos: Vec<u32> = vec![0, 5, 9, 100, 2047];
    let seg_of_q: Vec<u32> = vec![0, 1, 2, 3, 4];

    let mut r = Lcg(0xF01D);
    let qkv: Vec<f32> = (0..sq * lq).map(|_| r.next_f32() * 2.0).collect();
    let q_norm: Vec<f32> = (0..hd).map(|_| 1.0 + 0.1 * r.next_f32()).collect();
    let k_norm: Vec<f32> = (0..hd).map(|_| 1.0 + 0.1 * r.next_f32()).collect();
    let freq = RopeFreqTable::new(1_000_000.0, hd);

    // Host oracle — the v1 path's exact helpers, in its exact order.
    let mut want_q = vec![0.0f32; sq * q_dim];
    let mut want_kv = vec![0.0f32; 2 * own_total * kvd];
    for g in 0..sq {
        let row = &qkv[g * lq..(g + 1) * lq];
        let mut q_row = row[..q_dim].to_vec();
        qk_norm_inplace(&mut q_row, &q_norm, n_head, hd, eps as f64);
        apply_rope_with_freq(&mut q_row, &mut [], pos[g] as usize, hd, freq.as_slice());
        want_q[g * q_dim..(g + 1) * q_dim].copy_from_slice(&q_row);

        let mut k_row = row[q_dim..q_dim + kvd].to_vec();
        qk_norm_inplace(&mut k_row, &k_norm, n_kv, hd, eps as f64);
        apply_rope_with_freq(&mut k_row, &mut [], pos[g] as usize, hd, freq.as_slice());
        let ko = seg_of_q[g] as usize * kvd;
        want_kv[ko..ko + kvd].copy_from_slice(&k_row);
        want_kv[own_total * kvd + ko..own_total * kvd + ko + kvd]
            .copy_from_slice(&row[q_dim + kvd..lq]);
    }

    let qkv_h = create_f32(&client, &qkv);
    let qn_h = create_f32(&client, &q_norm);
    let kn_h = create_f32(&client, &k_norm);
    let pos_h = create_u32(&client, &pos);
    let seg_h = create_u32(&client, &seg_of_q);
    let freq_h = create_f32(&client, freq.as_slice());
    let qr_h = client.empty(sq * q_dim * core::mem::size_of::<f32>());
    let kv_h = client.empty(2 * own_total * kvd * core::mem::size_of::<f32>());
    EdlmQkvFoldCubeCL::launch::<ActiveRuntime>(
        &client,
        qkv_h,
        qn_h,
        kn_h,
        pos_h,
        seg_h,
        freq_h,
        qr_h.clone(),
        kv_h.clone(),
        &EdlmQkvFoldParams {
            n_head,
            n_kv_head: n_kv,
            head_dim: hd,
            eps,
            n_positions: own_total,
            seq_q: sq,
        },
    );
    let got_q = read_f32(&client, qr_h).expect("read q_rope");
    let got_kv = read_f32(&client, kv_h).expect("read kv");

    // V: bit-exact copy-through.
    let v_worst = got_kv[own_total * kvd..]
        .iter()
        .zip(&want_kv[own_total * kvd..])
        .position(|(a, b)| a.to_bits() != b.to_bits());
    assert_eq!(v_worst, None, "v copy-through not bit-exact at {v_worst:?}");

    let q_worst = max_abs_diff(&got_q, &want_q);
    let k_worst = max_abs_diff(&got_kv[..own_total * kvd], &want_kv[..own_total * kvd]);
    // The gate rides the LARGE-ANGLE face: pos=2047 · freq[0]=1.0 makes the
    // GPU sin/cos argument-reduction error the dominant term (~1e-4 class
    // on this backend) — an order above the tree-reduce face. The serving
    // state lengths (≤ 2048) see the same class; disclosed, never hidden.
    assert!(
        q_worst < 1e-3 && k_worst < 1e-3,
        "fold drift: q {q_worst} k {k_worst} (tree-reduce + sin/cos class)"
    );
    println!(
        "qkv fold drift: q {q_worst:.3e} k {k_worst:.3e} (v bit-exact; gate 1e-3, \
         pos-2047 sin/cos face dominant)"
    );
}

/// The fold kernel at the TINY PIPELINE's exact shapes (n_head 4, n_kv 2 →
/// kvd 256, lq 1024; a single 18-query row — the state pass; positions
/// 0..17). This is the scenario the whole-forward parity gates; if the
/// kernel is clean HERE but the pipeline reds, the defect is in the pass
/// wiring, not the math.
#[test]
fn qkv_fold_matches_cpu_reference_tiny_pipeline_shapes() {
    let ctx = CubeCLContext::new().expect("CubeCL should initialize");
    let client = ctx.client();

    let n_head = 4usize;
    let n_kv = 2usize;
    let hd = 128usize;
    let eps = 1e-6f32;
    let q_dim = n_head * hd;
    let kvd = n_kv * hd;
    let lq = q_dim + 2 * kvd;
    let sq = 18usize;
    let own_total = sq;
    let pos: Vec<u32> = (0..sq as u32).collect();
    // The state pass's actual mapping: one row → query j's key row is
    // seg[0] + j = j (the fold kernel's seg input is PER QUERY).
    let seg_of_q: Vec<u32> = (0..sq as u32).collect();

    let mut r = Lcg(0x51CE);
    let qkv: Vec<f32> = (0..sq * lq).map(|_| r.next_f32() * 2.0).collect();
    let q_norm: Vec<f32> = (0..hd).map(|_| 1.0 + 0.1 * r.next_f32()).collect();
    let k_norm: Vec<f32> = (0..hd).map(|_| 1.0 + 0.1 * r.next_f32()).collect();
    let freq = RopeFreqTable::new(1_000_000.0, hd);

    let mut want_q = vec![0.0f32; sq * q_dim];
    let mut want_kv = vec![0.0f32; 2 * own_total * kvd];
    for g in 0..sq {
        let row = &qkv[g * lq..(g + 1) * lq];
        let mut q_row = row[..q_dim].to_vec();
        qk_norm_inplace(&mut q_row, &q_norm, n_head, hd, eps as f64);
        apply_rope_with_freq(&mut q_row, &mut [], pos[g] as usize, hd, freq.as_slice());
        want_q[g * q_dim..(g + 1) * q_dim].copy_from_slice(&q_row);

        let mut k_row = row[q_dim..q_dim + kvd].to_vec();
        qk_norm_inplace(&mut k_row, &k_norm, n_kv, hd, eps as f64);
        apply_rope_with_freq(&mut k_row, &mut [], pos[g] as usize, hd, freq.as_slice());
        let ko = seg_of_q[g] as usize * kvd;
        want_kv[ko..ko + kvd].copy_from_slice(&k_row);
        want_kv[own_total * kvd + ko..own_total * kvd + ko + kvd]
            .copy_from_slice(&row[q_dim + kvd..lq]);
    }

    let qkv_h = create_f32(&client, &qkv);
    let qr_h = client.empty(sq * q_dim * core::mem::size_of::<f32>());
    let kv_h = client.empty(2 * own_total * kvd * core::mem::size_of::<f32>());
    EdlmQkvFoldCubeCL::launch::<ActiveRuntime>(
        &client,
        qkv_h,
        create_f32(&client, &q_norm),
        create_f32(&client, &k_norm),
        create_u32(&client, &pos),
        create_u32(&client, &seg_of_q),
        create_f32(&client, freq.as_slice()),
        qr_h.clone(),
        kv_h.clone(),
        &EdlmQkvFoldParams {
            n_head,
            n_kv_head: n_kv,
            head_dim: hd,
            eps,
            n_positions: own_total,
            seq_q: sq,
        },
    );
    let got_q = read_f32(&client, qr_h).expect("read q_rope");
    let got_kv = read_f32(&client, kv_h).expect("read kv");

    let q_worst = max_abs_diff(&got_q, &want_q);
    let k_worst = max_abs_diff(&got_kv[..own_total * kvd], &want_kv[..own_total * kvd]);
    // Small positions only (≤ 17): the sin/cos face is ulp-class here, so
    // the gate is the tree-reduce class, 1e-5.
    assert!(
        q_worst < 1e-5 && k_worst < 1e-5,
        "fold drift (tiny shapes): q {q_worst} k {k_worst}"
    );
    println!(
        "qkv fold drift (tiny shapes): q {q_worst:.3e} k {k_worst:.3e} (gate 1e-5)"
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

    let mut r = Lcg(0xC0FFEE);
    let q: Vec<f32> = (0..sq * q_dim).map(|_| r.next_f32()).collect();
    // K rows scaled so softmax isn't degenerate.
    let keys: Vec<f32> = (0..n_pos * kvd).map(|_| 0.2 * r.next_f32()).collect();
    let values: Vec<f32> = (0..n_pos * kvd).map(|_| r.next_f32()).collect();
    // Visibility: some rows causal-with-prefix, some full — the two laws the
    // model pass emits (branch continuation + bidir state). No seed: the
    // state-pass form.
    let t_end: Vec<u32> = vec![1, 3, 5, 9, 9];
    let t_start: Vec<u32> = vec![0; sq];

    let want = cpu_attn_reference(
        &q,
        &[],
        &[],
        &keys,
        &values,
        0,
        &t_start,
        &t_end,
        n_head,
        n_kv,
        hd,
    );

    let mut kv_combined = keys.clone();
    kv_combined.extend_from_slice(&values);
    let q_h = create_f32(&client, &q);
    let kv_h = create_f32(&client, &kv_combined);
    let seed_h = create_f32(&client, &[0.0]);
    let ts_h = create_u32(&client, &t_start);
    let te_h = create_u32(&client, &t_end);
    let out_h = client.empty(sq * q_dim * core::mem::size_of::<f32>());
    EdlmAttnMultiCubeCL::launch::<ActiveRuntime>(
        &client,
        q_h,
        kv_h,
        seed_h,
        ts_h,
        te_h,
        out_h.clone(),
        &EdlmAttnMultiParams {
            n_head,
            n_kv_head: n_kv,
            head_dim: hd,
            n_positions: n_pos,
            n_seed: 0,
            seq_q: sq,
            scale: 1.0 / (hd as f32).sqrt(),
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

    let mut r = Lcg(0xD00D);
    let q: Vec<f32> = (0..sq * q_dim).map(|_| r.next_f32()).collect();
    let keys: Vec<f32> = (0..n_pos * kvd).map(|_| 0.1 * r.next_f32()).collect();
    let values: Vec<f32> = (0..n_pos * kvd).map(|_| r.next_f32()).collect();
    let t_end: Vec<u32> = vec![129, 300]; // one just over a tile, one 2.34 tiles
    let t_start: Vec<u32> = vec![0; sq];

    let want = cpu_attn_reference(
        &q, &[], &[], &keys, &values, 0, &t_start, &t_end, n_head, n_kv, hd,
    );

    let mut kv_combined = keys.clone();
    kv_combined.extend_from_slice(&values);
    let q_h = create_f32(&client, &q);
    let kv_h = create_f32(&client, &kv_combined);
    let seed_h = create_f32(&client, &[0.0]);
    let ts_h = create_u32(&client, &t_start);
    let te_h = create_u32(&client, &t_end);
    let out_h = client.empty(sq * q_dim * core::mem::size_of::<f32>());
    EdlmAttnMultiCubeCL::launch::<ActiveRuntime>(
        &client,
        q_h,
        kv_h,
        seed_h,
        ts_h,
        te_h,
        out_h.clone(),
        &EdlmAttnMultiParams {
            n_head,
            n_kv_head: n_kv,
            head_dim: hd,
            n_positions: n_pos,
            n_seed: 0,
            seq_q: sq,
            scale: 1.0 / (hd as f32).sqrt(),
        },
    );
    let got = read_f32(&client, out_h).expect("read");
    let worst = max_abs_diff(&got, &want);
    assert!(
        worst < 2e-4,
        "multi-tile attention drift {worst} (online carry class)"
    );
}

#[test]
fn attn_multi_seed_segment_window() {
    // The seed segment (the device-resident state KV) + per-query own
    // windows — INCLUDING a window whose start is NOT 0 (row isolation in
    // the batched multi-branch pass: query 2 attends own [2, 5), skipping
    // own keys 0..2). The shape the old count-only t_n could not express.
    let ctx = CubeCLContext::new().expect("CubeCL should initialize");
    let client = ctx.client();

    let n_head = 2usize;
    let n_kv = 1usize;
    let hd = 128usize;
    let sq = 3usize;
    let sl = 7usize;
    let n_pos = 5usize;
    let kvd = n_kv * hd;
    let q_dim = n_head * hd;

    let mut r = Lcg(0x5EED_5EED);
    let q: Vec<f32> = (0..sq * q_dim).map(|_| r.next_f32()).collect();
    let seed_k: Vec<f32> = (0..sl * kvd).map(|_| 0.15 * r.next_f32()).collect();
    let seed_v: Vec<f32> = (0..sl * kvd).map(|_| r.next_f32()).collect();
    let keys: Vec<f32> = (0..n_pos * kvd).map(|_| 0.15 * r.next_f32()).collect();
    let values: Vec<f32> = (0..n_pos * kvd).map(|_| r.next_f32()).collect();
    let t_start: Vec<u32> = vec![0, 0, 2];
    let t_end: Vec<u32> = vec![1, 3, 5];

    let want = cpu_attn_reference(
        &q, &seed_k, &seed_v, &keys, &values, sl, &t_start, &t_end, n_head, n_kv, hd,
    );

    let mut own = keys.clone();
    own.extend_from_slice(&values);
    let mut seed = seed_k.clone();
    seed.extend_from_slice(&seed_v);
    let q_h = create_f32(&client, &q);
    let kv_h = create_f32(&client, &own);
    let seed_h = create_f32(&client, &seed);
    let ts_h = create_u32(&client, &t_start);
    let te_h = create_u32(&client, &t_end);
    let out_h = client.empty(sq * q_dim * core::mem::size_of::<f32>());
    EdlmAttnMultiCubeCL::launch::<ActiveRuntime>(
        &client,
        q_h,
        kv_h,
        seed_h,
        ts_h,
        te_h,
        out_h.clone(),
        &EdlmAttnMultiParams {
            n_head,
            n_kv_head: n_kv,
            head_dim: hd,
            n_positions: n_pos,
            n_seed: sl,
            seq_q: sq,
            scale: 1.0 / (hd as f32).sqrt(),
        },
    );
    let got = read_f32(&client, out_h).expect("read");
    let worst = max_abs_diff(&got, &want);
    assert!(
        worst < 2e-4,
        "seed+window attention drift {worst} (online carry class)"
    );
}

#[test]
fn attn_multi_seed_multi_tile_carry() {
    // The seed LONGER than one tile (200 keys) + own windows — the online
    // softmax must carry across the seed→own segment SEAM mid-tile (the
    // multi-tile rescale class of the second test, one source boundary
    // added). Query 1 sees 300 keys over the two segments.
    let ctx = CubeCLContext::new().expect("CubeCL should initialize");
    let client = ctx.client();

    let n_head = 2usize;
    let n_kv = 1usize;
    let hd = 128usize;
    let sq = 2usize;
    let sl = 200usize;
    let n_pos = 100usize;
    let kvd = n_kv * hd;
    let q_dim = n_head * hd;

    let mut r = Lcg(0xFA11_1005);
    let q: Vec<f32> = (0..sq * q_dim).map(|_| r.next_f32()).collect();
    let seed_k: Vec<f32> = (0..sl * kvd).map(|_| 0.1 * r.next_f32()).collect();
    let seed_v: Vec<f32> = (0..sl * kvd).map(|_| r.next_f32()).collect();
    let keys: Vec<f32> = (0..n_pos * kvd).map(|_| 0.1 * r.next_f32()).collect();
    let values: Vec<f32> = (0..n_pos * kvd).map(|_| r.next_f32()).collect();
    let t_start: Vec<u32> = vec![0, 0];
    let t_end: Vec<u32> = vec![50, 100];

    let want = cpu_attn_reference(
        &q, &seed_k, &seed_v, &keys, &values, sl, &t_start, &t_end, n_head, n_kv, hd,
    );

    let mut own = keys.clone();
    own.extend_from_slice(&values);
    let mut seed = seed_k.clone();
    seed.extend_from_slice(&seed_v);
    let q_h = create_f32(&client, &q);
    let kv_h = create_f32(&client, &own);
    let seed_h = create_f32(&client, &seed);
    let ts_h = create_u32(&client, &t_start);
    let te_h = create_u32(&client, &t_end);
    let out_h = client.empty(sq * q_dim * core::mem::size_of::<f32>());
    EdlmAttnMultiCubeCL::launch::<ActiveRuntime>(
        &client,
        q_h,
        kv_h,
        seed_h,
        ts_h,
        te_h,
        out_h.clone(),
        &EdlmAttnMultiParams {
            n_head,
            n_kv_head: n_kv,
            head_dim: hd,
            n_positions: n_pos,
            n_seed: sl,
            seq_q: sq,
            scale: 1.0 / (hd as f32).sqrt(),
        },
    );
    let got = read_f32(&client, out_h).expect("read");
    let worst = max_abs_diff(&got, &want);
    assert!(
        worst < 2e-4,
        "seed multi-tile carry drift {worst} (online carry class)"
    );
}

// ── the pipeline gate ───────────────────────────────────────────────

/// Full-pipeline parity vs the CPU T7 cached path on f16-grid tiny weights,
/// BOTH state_bidir postures (the core lane's own parity law: parity holds
/// in each posture, never across) × ALL GEMM postures (the tensor-core
/// kernels add the activation-rounding face — the gate re-pins per posture;
/// v1 and sg8 are the SAME drift class, sg8 just reassociates k more).
/// The CPU T7 cached path on the f16-grid tiny weights — the oracle both
/// sync postures gate against.
fn tiny_cpu_outputs(state_bidir: bool) -> (EdlmStateKv, Vec<Vec<f32>>) {
    let config = tiny_config();
    let mut weights = tiny_weights(&config);
    quantize_to_f16_grid(&mut weights);
    let enc = sample_encoding();
    enc.validate().expect("fixture invariants");
    let rows = riir_infer_core::transformer::edlm::rows_of(&enc).expect("rows");
    let sl = enc.state_len;
    let cache = edlm_state_prefill(&weights, &config, &enc.ids[..sl], &enc.pos[..sl], state_bidir);
    let out = forward_edlm_branches_cached(&weights, &config, &enc, &rows, &cache)
        .expect("cpu branches");
    (cache, out)
}

/// The GPU pipeline on the same f16-grid tiny weights, at the caller's GEMM
/// posture and SYNC posture (the fold A/B seam).
fn tiny_gpu_outputs(
    state_bidir: bool,
    posture: GemmPosture,
    fold: bool,
) -> (EdlmStateKv, Vec<Vec<f32>>) {
    let config = tiny_config();
    let mut weights = tiny_weights(&config);
    quantize_to_f16_grid(&mut weights);
    let enc = sample_encoding();
    enc.validate().expect("fixture invariants");
    let rows = riir_infer_core::transformer::edlm::rows_of(&enc).expect("rows");
    let mut gpu = EdlmGpuModel::from_weights(&weights, &config)
        .expect("gpu model")
        .with_posture(posture)
        .with_fold(fold);
    let sl = enc.state_len;
    gpu.state_prefill(&enc.ids[..sl], &enc.pos[..sl], state_bidir)
        .expect("gpu prefill");
    let out = gpu.forward_branches(&enc, &rows).expect("gpu branches");
    (gpu.state_kv().expect("cache").clone(), out)
}

fn state_kv_max_diff(a: &EdlmStateKv, b: &EdlmStateKv) -> f32 {
    let mut worst = 0.0f32;
    for (ca, cb) in a.layers.iter().zip(&b.layers) {
        worst = worst.max(max_abs_diff(&ca.k, &cb.k));
        worst = worst.max(max_abs_diff(&ca.v, &cb.v));
    }
    worst
}

fn rows_max_diff(a: &[Vec<f32>], b: &[Vec<f32>]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(ra, rb)| max_abs_diff(ra, rb))
        .fold(0.0f32, f32::max)
}

fn tiny_pipeline_parity(state_bidir: bool, posture: GemmPosture) -> (f32, f32) {
    let (cpu_cache, cpu_rows) = tiny_cpu_outputs(state_bidir);
    let (gpu_cache, gpu_rows) = tiny_gpu_outputs(state_bidir, posture, fold_env_default());

    // State cache K/V parity (the prefill's own output).
    let kv_worst = state_kv_max_diff(&cpu_cache, &gpu_cache);
    // Hidden parity (the pointer head's input).
    let hid_worst = rows_max_diff(&cpu_rows, &gpu_rows);
    (kv_worst, hid_worst)
}

const ALL_POSTURES: [GemmPosture; 3] = [
    GemmPosture::Scalar,
    GemmPosture::Cmma,
    GemmPosture::CmmaSg8,
];

fn posture_name(p: GemmPosture) -> &'static str {
    match p {
        GemmPosture::Scalar => "scalar",
        GemmPosture::Cmma => "cmma",
        GemmPosture::CmmaSg8 => "cmma-sg8",
    }
}

#[test]
fn tiny_parity_vs_cpu_cached_bidir() {
    for posture in ALL_POSTURES {
        let (kv, hid) = tiny_pipeline_parity(true, posture);
        // Per-posture kv gates: the scalar kernel's only face is the f32
        // accumulation order (~6e-5 measured); the tensor-core kernels add
        // the ACTIVATION f16 rounding face (~1.5e-3 measured — the input
        // disclosure in matmul_f16b_cmma_cubecl; sg8 shares the class). The
        // hidden gate is the binding law for all (the pointer head's input).
        let kv_gate = if posture == GemmPosture::Scalar { 5e-4 } else { 3e-3 };
        assert!(
            kv < kv_gate && hid < 2e-3,
            "bidir parity ({}): kv drift {kv} (gate {kv_gate}), hidden drift {hid}",
            posture_name(posture)
        );
        println!(
            "tiny bidir parity ({}): kv {kv:.3e} hidden {hid:.3e}",
            posture_name(posture)
        );
    }
}

#[test]
fn tiny_parity_vs_cpu_cached_causal() {
    for posture in ALL_POSTURES {
        let (kv, hid) = tiny_pipeline_parity(false, posture);
        let kv_gate = if posture == GemmPosture::Scalar { 5e-4 } else { 3e-3 };
        assert!(
            kv < kv_gate && hid < 2e-3,
            "causal parity ({}): kv drift {kv} (gate {kv_gate}), hidden drift {hid}",
            posture_name(posture)
        );
        println!(
            "tiny causal parity ({}): kv {kv:.3e} hidden {hid:.3e}",
            posture_name(posture)
        );
    }
}

/// The fold-vs-host ARMS test — the two sync postures of the SAME GPU
/// pipeline (same f16 weights, same GEMM kernels, ONE shared PassGeometry)
/// must agree within the small-ops' own tolerance class: the fold's tree
/// sum-of-squares + GPU sin/cos faces feed the attention, so the drift is
/// larger than zero but WELL inside the CPU-parity gates above (both arms
/// carry the same f16 + online-softmax faces, which CANCEL in this
/// comparison). Runs at every GEMM posture — the fold is posture-agnostic
/// by construction, and the assertion is what proves it stays that way.
#[test]
fn tiny_fold_vs_host_arms() {
    for posture in ALL_POSTURES {
        let (host_kv, host_rows) = tiny_gpu_outputs(true, posture, false);
        let (fold_kv, fold_rows) = tiny_gpu_outputs(true, posture, true);
        let kv = state_kv_max_diff(&host_kv, &fold_kv);
        let hid = rows_max_diff(&host_rows, &fold_rows);
        assert!(
            kv < 1e-3 && hid < 1e-3,
            "fold-vs-host arms ({}): kv {kv:.3e} hidden {hid:.3e} (gate 1e-3)",
            posture_name(posture)
        );
        println!(
            "tiny fold-vs-host arms ({}): kv {kv:.3e} hidden {hid:.3e} (gate 1e-3)",
            posture_name(posture)
        );
    }
}

/// Stale/mismatched caches REFUSE (the T7 law carries to the GPU lane).
#[test]
fn forward_branches_refuses_without_prefill() {
    let config = tiny_config();
    let mut weights = tiny_weights(&config);
    quantize_to_f16_grid(&mut weights);
    let enc = sample_encoding();
    let rows = riir_infer_core::transformer::edlm::rows_of(&enc).expect("rows");
    let gpu = EdlmGpuModel::from_weights(&weights, &config).expect("gpu model");
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
