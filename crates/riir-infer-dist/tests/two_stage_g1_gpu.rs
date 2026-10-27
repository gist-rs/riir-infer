//! Plan 623 T2.2b — GPU-GPU 2-stage pipeline over REAL TCP (loopback) on
//! the real Bonsai-2 27B: the token stream and every tail logit are
//! BIT-IDENTICAL to the 1-box GPU forward, at each requested split.
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

use riir_infer_core::gguf_loader::load_qwen_deltanet_ternary_weights_gguf;
use riir_infer_core::multi_host::LayerRange;
use riir_infer_dist::argmax_first;
use riir_infer_dist::runner::{run_head, run_tail};
use riir_infer_dist::stage::{ArtifactIdentity, StageCompute};
use riir_infer_dist::stage_gpu::TernaryGpuStage;
use riir_infer_dist::transport::{Direction, PeerKey, TcpTransport};

const FOLDED_MODEL: &str = "../../../riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf";
const PROMPT: [u32; 4] = [9707, 11, 1879, 0];

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
    let (mut config, weights) = load_qwen_deltanet_ternary_weights_gguf(&model).expect("load");
    config.block_size = PROMPT.len() + max_new;
    let n = config.n_layer as u32;
    let weights = Arc::new(weights);

    // 1-box GPU oracle (dropped before the pair: each stage uploads the
    // whole model — the T2.2c limit).
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
    println!("1-box GPU stream: {want_toks:?}");

    let key = PeerKey([0x62; 32]);
    for k in ks.into_iter().filter(|k| (1..n).contains(k)) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let (tc, tw, tk) = (config.clone(), weights.clone(), key.clone());
        let tail = thread::spawn(move || {
            let mut stage = TernaryGpuStage::new(&tc, tw, LayerRange::new(k, n), identity).unwrap();
            let (sock, _) = listener.accept().unwrap();
            let mut link = TcpTransport::new(sock, tk, Direction::Upstream).unwrap();
            let mut seen = Vec::new();
            run_tail(&mut stage, &mut link, |_, l| seen.push(l.to_vec())).map(|_| seen)
        });
        let mut head =
            TernaryGpuStage::new(&config, weights.clone(), LayerRange::new(0, k), identity)
                .unwrap();
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
            "split k={k}: bit-identical ×{} positions · decode {:.2} tok/s",
            seen.len(),
            r.decode_tok_s()
        );
    }
}
