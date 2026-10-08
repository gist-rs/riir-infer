//! The eDLM fold GOAT lane (Issue 1005 T8 follow-up): the DEVICE-FOLD sync
//! posture (the fused QKV fold + SiLU gate + device residual adds — zero
//! mid-pass readbacks) vs the v1 HOST small-ops posture (four readbacks per
//! layer plus the host round-trips between them) — the same GPU pipeline,
//! ONE shared [`PassGeometry`] per pass, A/B'd per instance via
//! [`EdlmGpuModel::with_fold`].
//!
//! The parity evidence lives in the lib tests (`qkv_fold_matches_cpu_reference`
//! — the fold vs the exact host helpers; `tiny_fold_vs_host_arms` — the
//! whole-forward arms at every GEMM posture); THIS lane is the measurement:
//! interleaved pairs over the mid serving shape, release profile.
//!
//! Run:
//! `cargo test --release -p riir-infer-gpu --features edlm_gpu
//!  --test edlm_fold_goat -- --ignored --nocapture`
//!
//! Box state is part of every number: quote the `PROVENANCE:` line + the
//! host's GPU-exclusivity state (this lane prints the runtime name; take
//! `nvidia-smi` beside the run).

#![cfg(feature = "edlm_gpu")]

use riir_infer_core::transformer::edlm::{BranchRow, EdlmLayerWeights, EdlmWeights, PackedEncoding};
use riir_infer_core::types::Config;
use riir_infer_gpu::EdlmGpuModel;

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

/// The branch_batch GOAT's mid shape, verbatim — the same body the batched
/// multi-branch numbers were taken on, so the fold's delta reads against a
/// known cell (head_dim 128, GQA 8:8, 8 layers, mlp 2048, state 384,
/// 4 rows × 64 tokens).
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

fn mid_models(config: &Config) -> (PackedEncoding, Vec<BranchRow>, EdlmGpuModel, EdlmGpuModel) {
    let mut weights = mid_weights(config);
    quantize_to_f16_grid(&mut weights);
    let enc = mid_encoding();
    let rows = riir_infer_core::transformer::edlm::rows_of(&enc).expect("rows");
    let host = EdlmGpuModel::from_weights(&weights, config)
        .expect("gpu model")
        .with_fold(false);
    let fold = EdlmGpuModel::from_weights(&weights, config)
        .expect("gpu model")
        .with_fold(true);
    (enc, rows, host, fold)
}

/// The two arms agree — same GEMM/attention kernels, ONE shared geometry;
/// only the small-ops' numerics faces differ (tree-order sum-of-squares +
/// GPU sin/cos vs the host's sequential sum + libm sin_cos). The gate is
/// loose (1e-3); the printed value is the disclosure.
#[test]
fn fold_matches_host_arms_mid() {
    let config = mid_config();
    let (enc, rows, mut host, mut fold) = mid_models(&config);
    host.state_prefill(&enc.ids[..STATE_LEN], &enc.pos[..STATE_LEN], true)
        .expect("host prefill");
    fold.state_prefill(&enc.ids[..STATE_LEN], &enc.pos[..STATE_LEN], true)
        .expect("fold prefill");

    let host_out = host.forward_branches(&enc, &rows).expect("host branches");
    let fold_out = fold.forward_branches(&enc, &rows).expect("fold branches");
    let worst = host_out
        .iter()
        .zip(&fold_out)
        .map(|(a, b)| max_abs_diff(a, b))
        .fold(0.0f32, f32::max);
    assert!(
        worst < 1e-3,
        "fold-vs-host arm drift {worst} at the mid shape (gate 1e-3)"
    );
    println!("mid fold-vs-host arm drift: {worst:.3e} (gate 1e-3)");
    host.memory_cleanup();
    fold.memory_cleanup();
}

/// The GOAT: interleaved fold vs host over the mid shape. PRINTS the table
/// and verdict; asserts nothing about perf (a loaded-box bar rots) —
/// promotion evidence is the printed medians, recorded in the issue row and
/// commit. Also prints the sync arithmetic: the host arm's per-pass
/// readback traffic vs the fold arm's pass-final read only.
#[test]
#[ignore]
fn goat_fold_interleaved_ab() {
    use cubecl::Runtime as _;
    use riir_infer_gpu::{ActiveRuntime, CubeCLContext};

    let ctx = CubeCLContext::new().expect("CubeCL should initialize");
    let client = ctx.client();
    println!("PROVENANCE: runtime {}", ActiveRuntime::name(&client));

    let config = mid_config();
    let (enc, rows, mut host, mut fold) = mid_models(&config);
    assert_eq!(rows.len(), N_ROWS, "fixture row count");
    assert!(!host.fold_enabled() && fold.fold_enabled(), "A/B seam");
    host.state_prefill(&enc.ids[..STATE_LEN], &enc.pos[..STATE_LEN], true)
        .expect("host prefill");
    fold.state_prefill(&enc.ids[..STATE_LEN], &enc.pos[..STATE_LEN], true)
        .expect("fold prefill");

    // The sync arithmetic (per branch pass, 8 layers, sqt = Σ branch_len =
    // 256): the host arm reads back FOUR buffers per layer (raw qkv sqt·lq,
    // attn-out sqt·q_dim, gate/up sqt·2·mlp, down sqt·n) and re-uploads the
    // folded results; the fold arm's only mid-pass traffic is ZERO — the
    // pass-final hidden readback (sqt·n) is the one read either arm pays.
    let n = config.n_embd;
    let q_dim = config.n_head * config.head_dim;
    let lq = q_dim + 2 * config.n_kv_head * config.head_dim;
    let sqt = enc.ids.len() - STATE_LEN;
    let host_read_mb = (config.n_layer
        * (sqt * lq + sqt * q_dim + sqt * 2 * config.mlp_hidden + sqt * n)
        * 4) as f64
        / 1e6;
    let fold_read_mb = (sqt * n * 4) as f64 / 1e6;
    println!(
        "sync arithmetic per branch pass: host arm ≈ {host_read_mb:.1} MB read back \
         (4 readbacks × {layers} layers) + the host small-ops round-trips; \
         fold arm ≈ {fold_read_mb:.2} MB (the pass-final hidden only)",
        layers = config.n_layer,
    );

    fn timed(
        gpu: &mut EdlmGpuModel,
        enc: &PackedEncoding,
        rows: &[BranchRow],
    ) -> std::time::Duration {
        let start = std::time::Instant::now();
        let _ = gpu.forward_branches(enc, rows).expect("timed arm");
        start.elapsed()
    }

    const PAIRS: usize = 9;
    const WARMUP: usize = 2;

    for _ in 0..WARMUP {
        timed(&mut host, &enc, &rows);
        timed(&mut fold, &enc, &rows);
    }

    let mut host_us: Vec<u128> = Vec::with_capacity(PAIRS);
    let mut fold_us: Vec<u128> = Vec::with_capacity(PAIRS);
    for _ in 0..PAIRS {
        host_us.push(timed(&mut host, &enc, &rows).as_micros());
        fold_us.push(timed(&mut fold, &enc, &rows).as_micros());
    }
    host_us.sort_unstable();
    fold_us.sort_unstable();
    let h_med = host_us[PAIRS / 2] as f64;
    let f_med = fold_us[PAIRS / 2] as f64;
    println!(
        "| {:>10} | {:>12} | {:>12} | {:>8} |",
        "arm", "host µs", "fold µs", "speedup"
    );
    println!(
        "| {:>10} | {:>12.1} | {:>12.1} | {:>7.2}x |",
        format!("{N_ROWS}×{ROW_LEN}"),
        h_med,
        f_med,
        h_med / f_med
    );
    println!(
        "VERDICT: {}",
        if f_med < h_med {
            "fold wins the median — promotion candidate"
        } else {
            "fold LOST the median — read the box state before any conclusion"
        }
    );
    host.memory_cleanup();
    fold.memory_cleanup();
}
