//! Plan 623 T2.2b/T2.2c — GPU-GPU 2-stage pipeline over REAL TCP (loopback)
//! on the real Bonsai-2 27B: the token stream and every tail logit are
//! BIT-IDENTICAL to the 1-box GPU forward, at each requested split — with
//! each stage holding ONLY its own scoped slice of the model (T2.2c: the
//! per-stage working set is printed beside the full model's).
//!
//! ```sh
//! CARGO_TARGET_DIR=/tmp/i623_dist cargo test -p riir-infer-dist --release \
//!   --features gpu,bonsai2_hadamard --test two_stage_g1_gpu -- --ignored --nocapture
//! ```
//!
//! Env: `BONSAI2_G1_MODEL` (model path), `I623_K` (comma list of splits,
//! default `1,32,63`), `I623_NEW` (tokens generated, default 16).

#![cfg(feature = "gpu")]

use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::Arc;
use std::thread;

use riir_infer_core::deltanet::{QwenDeltaNetTernaryWeights, WeightScope};
use riir_infer_core::gguf_loader::load_qwen_deltanet_ternary_weights_gguf_scoped;
use riir_infer_core::multi_host::LayerRange;
use riir_infer_core::types::Config;
use riir_infer_dist::argmax_first;
use riir_infer_dist::runner::{run_head, run_tail};
use riir_infer_dist::stage::{ArtifactIdentity, StageCompute, stage_scope};
use riir_infer_dist::stage_gpu::TernaryGpuStage;
use riir_infer_dist::transport::{Direction, PeerKey, TcpTransport};
use riir_infer_gpu::vram_budget::forward_working_set_bytes;

const FOLDED_MODEL: &str = "../../../riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf";
const PROMPT: [u32; 4] = [9707, 11, 1879, 0];

fn load(
    model: &std::path::Path,
    ctx: usize,
    scope: impl FnOnce(&Config) -> WeightScope,
) -> (Config, Arc<QwenDeltaNetTernaryWeights>) {
    let (mut config, w) =
        load_qwen_deltanet_ternary_weights_gguf_scoped(model, scope).expect("load");
    config.block_size = ctx;
    (config, Arc::new(w))
}

fn gib(c: &Config, w: &QwenDeltaNetTernaryWeights) -> f64 {
    forward_working_set_bytes(c, w) as f64 / (1u64 << 30) as f64
}

fn env_or<T: std::str::FromStr>(var: &str, default: T) -> T {
    std::env::var(var)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

#[test]
#[ignore = "requires the Metal GPU + the Bonsai-2 PQ2_0 model; run explicitly with --release"]
fn gpu_two_stage_tcp_bit_identical_to_one_box_gpu() {
    let model = std::env::var_os("BONSAI2_G1_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(FOLDED_MODEL));
    if !model.exists() {
        eprintln!("SKIP: model not found at {}", model.display());
        return;
    }
    let max_new: usize = env_or("I623_NEW", 16);
    let ks: Vec<u32> = std::env::var("I623_K")
        .unwrap_or_else(|_| "1,32,63".into())
        .split(',')
        .filter_map(|v| v.trim().parse().ok())
        .collect();

    let identity = ArtifactIdentity::of_gguf(&model).expect("identity");
    let ctx = PROMPT.len() + max_new;
    let (config, weights) = load(&model, ctx, |c| WeightScope::full(c.n_layer));
    let n = config.n_layer as u32;
    let full_gib = gib(&config, &weights);

    // 1-box GPU oracle on the full model, dropped (with the full weights)
    // before the scoped pair loads.
    let (want_toks, want_logits) = {
        let mut s =
            TernaryGpuStage::new(&config, weights.clone(), LayerRange::full(n), identity).unwrap();
        let empty: [f32; 0] = [];
        let mut logits = Vec::new();
        let mut next = 0;
        for (pos, &t) in PROMPT.iter().enumerate() {
            let l = s.step(t, pos, &empty).to_vec();
            next = argmax_first(&l);
            logits.push(l);
        }
        let mut toks = vec![next];
        let t = std::time::Instant::now();
        for i in 1..max_new {
            let l = s.step(next, PROMPT.len() + i - 1, &empty).to_vec();
            next = argmax_first(&l);
            logits.push(l);
            toks.push(next);
        }
        let tok_s = (max_new - 1) as f64 / t.elapsed().as_secs_f64();
        println!("1-box GPU stage decode: {tok_s:.2} tok/s (same per-step readback shape)");
        (toks, logits)
    };
    drop(weights);
    println!("1-box GPU stream: {want_toks:?} · working set {full_gib:.2} GiB");

    let key = PeerKey([0x62; 32]);
    for k in ks.into_iter().filter(|k| (1..n).contains(k)) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let (_, hw) = load(&model, ctx, |c| {
            stage_scope(LayerRange::new(0, k), c.n_layer as u32)
        });
        let (_, tw) = load(&model, ctx, |c| {
            stage_scope(LayerRange::new(k, n), c.n_layer as u32)
        });
        let (head_gib, tail_gib) = (gib(&config, &hw), gib(&config, &tw));
        let (tc, tk) = (config.clone(), key.clone());
        let tail = thread::spawn(move || {
            let mut stage = TernaryGpuStage::new(&tc, tw, LayerRange::new(k, n), identity).unwrap();
            let (sock, _) = listener.accept().unwrap();
            let mut link = TcpTransport::new(sock, tk, Direction::Upstream).unwrap();
            let mut seen = Vec::new();
            run_tail(&mut stage, &mut link, |_, l| seen.push(l.to_vec())).map(|_| seen)
        });
        let mut head = TernaryGpuStage::new(&config, hw, LayerRange::new(0, k), identity).unwrap();
        let mut link = TcpTransport::new(
            TcpStream::connect(addr).unwrap(),
            key.clone(),
            Direction::Downstream,
        )
        .unwrap();
        let r = run_head(&mut head, &mut link, &PROMPT, max_new).expect("head");
        let seen = tail.join().unwrap().expect("tail");
        assert_eq!(r.generated, want_toks, "split@{k}: token stream diverged");
        assert_eq!(seen.len(), want_logits.len(), "split@{k}: position count");
        for (p, (a, b)) in seen.iter().zip(&want_logits).enumerate() {
            assert!(
                a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()),
                "split@{k}: logits at pos {p} not bit-identical"
            );
        }
        println!(
            "split k={k}: bit-identical ×{} positions · decode {:.2} tok/s · working set head {head_gib:.2} + tail {tail_gib:.2} GiB (full {full_gib:.2})",
            seen.len(),
            r.decode_tok_s()
        );
        // Non-vacuity: each slice must be strictly smaller than the model.
        assert!(
            head_gib < full_gib && tail_gib < full_gib,
            "split@{k}: a stage is not scoped"
        );
    }
}
