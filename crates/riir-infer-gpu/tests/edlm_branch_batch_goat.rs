//! The eDLM branch-pass GOAT lane (Issue 1005 T8 follow-up): the batched
//! multi-branch forward (all rows in ONE pass, the seed KV attended
//! device-resident) vs the per-row shape (one pass per row, batched-of-1)
//! — the two halves of the landed follow-up:
//!
//! - **batching** is MEASURED here (interleaved pairs, release): the four
//!   per-layer projections run at `m = Σ branch_len` once instead of once
//!   per row, and the launch/readback count divides by the row count.
//! - **the KV carry** is disclosed ARITHMETICALLY + a measured upload
//!   bandwidth: the per-row-per-layer host copy + re-upload of the combined
//!   `[state | branch]` cache no longer happens (the seed segment stays on
//!   the device from `state_prefill`).
//!
//! Run:
//! `cargo test --release -p riir-infer-gpu --features edlm_gpu
//!  --test edlm_branch_batch_goat -- --ignored --nocapture`
//!
//! Box state is part of every number: quote the `PROVENANCE:` line + the
//! host's GPU-exclusivity state (this lane prints the runtime name; take
//! `nvidia-smi` beside the run).

#![cfg(feature = "edlm_gpu")]

use riir_infer_core::transformer::edlm::{
    BranchRow, EdlmLayerWeights, EdlmWeights, PackedEncoding,
};
use riir_infer_core::types::Config;
use riir_infer_gpu::{ActiveRuntime, CubeCLContext, EdlmGpuModel, create_f32};
use cubecl::Runtime as _;

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

/// A mid-shape body: the release's head geometry (head_dim 128, Qwen3's
/// `n_embd == n_head · head_dim`), kvd 1024 (GQA 8:8 — the KV-carry term is
/// per kv head), 8 layers, mlp 2048. The STATE length rides the real
/// serving crossover (384) and the branch fan-out is 4 rows × 64 tokens —
/// the per-row GEMMs sit BELOW the sg8 crossover's m, the batched GEMMs
/// ABOVE it, so the arm also exposes the posture interaction.
fn mid_config() -> Config {
    let mut c = Config::micro();
    c.vocab_size = 64;
    c.block_size = 2048;
    c.n_embd = 1024;
    c.n_head = 8;
    c.n_kv_head = 8;
    c.head_dim = 128;
    c.mlp_hidden = 2048;
    c.n_layer = 8;
    c.rms_norm_eps = 1e-6;
    c.rope_theta = 1_000_000.0;
    c
}

fn mid_weights(config: &Config) -> EdlmWeights {
    let mut r = Lcg(0x5EED_BA7C);
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

/// Round to the f16 grid (the same isolation law as the parity tests: the
/// CPU oracle runs the SAME rounded weights).
fn quantize_to_f16_grid(w: &mut EdlmWeights) {
    let q = |v: &mut Vec<f32>| {
        for x in v.iter_mut() {
            *x = half::f16::from_f32(*x).to_f32();
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

const STATE_LEN: usize = 384;
const N_ROWS: usize = 4;
const ROW_LEN: usize = 64;

/// state(384) | 4 branch rows × 64 tokens — the serving shape (their 384
/// attention-only crossover law), fanned out over a 4-option question.
fn mid_encoding() -> PackedEncoding {
    let mut ids = Vec::with_capacity(STATE_LEN + N_ROWS * ROW_LEN);
    let mut pos = Vec::with_capacity(ids.capacity());
    let mut seg = Vec::with_capacity(ids.capacity());
    for i in 0..STATE_LEN {
        ids.push(i % 64);
        pos.push(i);
        seg.push(0);
    }
    for r in 0..N_ROWS {
        for j in 0..ROW_LEN {
            ids.push((r * 7 + j) % 64);
            pos.push(STATE_LEN + j);
            seg.push(1 + r as i32);
        }
    }
    let mut enc = PackedEncoding {
        ids,
        pos,
        seg,
        opt: Vec::new(),
        state_len: STATE_LEN,
        state_truncated: false,
        decide_idx: Vec::new(),
        opt_idx: Vec::new(),
    };
    // The rows form (no option isolation): one decide marker at each row's
    // last token.
    for r in 0..N_ROWS {
        let end = STATE_LEN + (r + 1) * ROW_LEN;
        enc.decide_idx.push(end - 1);
        enc.opt_idx.push(Vec::new());
    }
    enc
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// The arms agree (same kernels, same per-row math — only the m-shape of
/// the GEMMs and the launch count differ). The gate is loose (1e-4); the
/// printed value is the disclosure (bit-identity is EXPECTED and prints 0).
#[test]
fn batched_matches_per_row_arms() {
    let config = mid_config();
    let mut weights = mid_weights(&config);
    quantize_to_f16_grid(&mut weights);
    let enc = mid_encoding();
    let rows = riir_infer_core::transformer::edlm::rows_of(&enc).expect("rows");
    assert_eq!(rows.len(), N_ROWS, "fixture row count");

    let mut gpu = EdlmGpuModel::from_weights(&weights, &config).expect("gpu model");
    gpu.state_prefill(&enc.ids[..STATE_LEN], &enc.pos[..STATE_LEN], true)
        .expect("gpu prefill");

    let batched = gpu.forward_branches(&enc, &rows).expect("batched");
    let mut per_row = Vec::with_capacity(N_ROWS);
    for row in &rows {
        per_row.extend(gpu.forward_branches(&enc, std::slice::from_ref(row)).expect("per-row"));
    }

    let worst = batched
        .iter()
        .zip(&per_row)
        .map(|(b, p)| max_abs_diff(b, p))
        .fold(0.0f32, f32::max);
    assert!(
        worst < 1e-4,
        "batched-vs-per-row arm drift {worst} (expected ~0 — same kernels)"
    );
    println!("batched-vs-per-row arm drift: {worst:.3e} (gate 1e-4)");
}

/// One timed forward_branches call set: `per_row` runs one call PER row
/// (the old shape's dispatch), `batched` runs ONE call over all rows.
fn timed_arm(
    gpu: &EdlmGpuModel,
    enc: &PackedEncoding,
    rows: &[BranchRow],
    per_row: bool,
) -> std::time::Duration {
    let start = std::time::Instant::now();
    if per_row {
        for row in rows {
            let _ = gpu
                .forward_branches(enc, std::slice::from_ref(row))
                .expect("per-row arm");
        }
    } else {
        let _ = gpu.forward_branches(enc, rows).expect("batched arm");
    }
    start.elapsed()
}

/// The GOAT: interleaved per-row vs batched over the mid shape. PRINTS the
/// table + verdict; asserts nothing about perf (a loaded-box bar rots) —
/// promotion evidence is the printed medians, recorded in the issue row +
/// commit. Also prints the KV-carry traffic arithmetic (the eliminated
/// per-row seed re-upload) + one measured upload-bandwidth sample to
/// convert it to milliseconds.
#[test]
#[ignore]
fn goat_branch_batch_interleaved_ab() {
    let ctx = CubeCLContext::new().expect("CubeCL should initialize");
    let client = ctx.client();
    println!("PROVENANCE: runtime {}", ActiveRuntime::name(&client));

    let config = mid_config();
    let mut weights = mid_weights(&config);
    quantize_to_f16_grid(&mut weights);
    let enc = mid_encoding();
    let rows = riir_infer_core::transformer::edlm::rows_of(&enc).expect("rows");
    assert_eq!(rows.len(), N_ROWS, "fixture row count");

    let mut gpu = EdlmGpuModel::from_weights(&weights, &config).expect("gpu model");
    gpu.state_prefill(&enc.ids[..STATE_LEN], &enc.pos[..STATE_LEN], true)
        .expect("gpu prefill");

    // The KV-carry arithmetic (the 8B shape scales it by n_embd/1024 and
    // n_layer/8): per branch row, the OLD path re-uploaded the combined
    // [state | branch] cache — 2·(sl+row_len)·kvd f32 per layer.
    let kvd = config.n_kv_head * config.head_dim;
    let old_upload_bytes_per_row =
        2 * (STATE_LEN + ROW_LEN) * kvd * core::mem::size_of::<f32>() * config.n_layer;
    // One measured upload sample of the same size (cold of the timed arms —
    // this is the bandwidth the old path paid per row).
    let probe = vec![0.0f32; old_upload_bytes_per_row / core::mem::size_of::<f32>()];
    let t_up = std::time::Instant::now();
    let probe_h = create_f32(&client, &probe);
    let upload_ms = t_up.elapsed().as_secs_f64() * 1e3;
    let gbps = old_upload_bytes_per_row as f64 / (t_up.elapsed().as_secs_f64()) / 1e9;
    println!(
        "KV carry (eliminated): {:.1} MB/branch-row of seed re-upload \
         (2·({sl}+{rl})·{kvd}·4B × {layers} layers); measured one upload of \
         that size at {gbps:.1} GB/s = {upload_ms:.2} ms/row — \
         the old path paid it ×{rows_n} rows per pass",
        old_upload_bytes_per_row as f64 / 1e6,
        sl = STATE_LEN,
        rl = ROW_LEN,
        kvd = kvd,
        layers = config.n_layer,
        gbps = gbps,
        upload_ms = upload_ms,
        rows_n = N_ROWS,
    );
    drop(probe_h);

    const PAIRS: usize = 9;
    const WARMUP: usize = 2;

    for _ in 0..WARMUP {
        timed_arm(&gpu, &enc, &rows, true);
        timed_arm(&gpu, &enc, &rows, false);
    }

    let mut per_row: Vec<u128> = Vec::with_capacity(PAIRS);
    let mut batched: Vec<u128> = Vec::with_capacity(PAIRS);
    for _ in 0..PAIRS {
        per_row.push(timed_arm(&gpu, &enc, &rows, true).as_micros());
        batched.push(timed_arm(&gpu, &enc, &rows, false).as_micros());
    }
    per_row.sort_unstable();
    batched.sort_unstable();
    let pr_med = per_row[PAIRS / 2] as f64;
    let b_med = batched[PAIRS / 2] as f64;
    println!(
        "| {:>10} | {:>12} | {:>12} | {:>8} |",
        "arm", "per-row µs", "batched µs", "speedup"
    );
    println!(
        "| {:>10} | {:>12.1} | {:>12.1} | {:>7.2}x |",
        format!("{N_ROWS}×{ROW_LEN}"),
        pr_med,
        b_med,
        pr_med / b_med
    );
    println!(
        "VERDICT: {}",
        if b_med < pr_med {
            "batched wins the median — promotion candidate"
        } else {
            "batched LOST the median — read the box state before any conclusion"
        }
    );
    gpu.memory_cleanup();
}
