//! Plan 623 T2.2 — the 2-stage PoC over REAL TCP sockets (loopback):
//! the pipelined token stream and every tail-side logit vector are
//! BIT-IDENTICAL to the 1-box forward, at every split k; plus the
//! security negatives (wrong key, mismatched artifact) refuse.

#[path = "../../../tests/common/synth_bonsai2.rs"]
mod synth;

use std::net::{TcpListener, TcpStream};
use std::sync::Arc;
use std::thread;

use riir_infer_core::deltanet::QwenDeltaNetTernaryWeights;
use riir_infer_core::multi_host::LayerRange;
use riir_infer_core::types::Config;
use riir_infer_dist::argmax_first;
use riir_infer_dist::runner::{run_head, run_local, run_tail};
use riir_infer_dist::stage::{ArtifactIdentity, StageCompute, TernaryCpuStage};
use riir_infer_dist::transport::{Direction, PeerKey, TcpTransport};

const PROMPT: [u32; 5] = [7, 13, 42, 100, 5];
const MAX_NEW: usize = 6;

/// `name` must be unique per call site: `write_tmp` keys on pid only, and
/// the tests in this binary run concurrently (the shared-temp-path class).
fn load(name: &str) -> (Config, Arc<QwenDeltaNetTernaryWeights>, ArtifactIdentity) {
    let bytes = synth::build_gguf(
        &synth::synth_metadata(false),
        &synth::synth_tensors_bonsai2(),
    );
    let path = synth::write_tmp(name, &bytes);
    let (c, w) = riir_infer_core::gguf_loader::load_qwen_deltanet_ternary_weights_gguf(&path)
        .expect("load synth");
    let id = ArtifactIdentity::of_gguf(&path).expect("identity");
    let _ = std::fs::remove_file(&path);
    (c, Arc::new(w), id)
}

/// 1-box reference: tokens + per-position logits.
fn reference(
    config: &Config,
    w: &Arc<QwenDeltaNetTernaryWeights>,
    id: ArtifactIdentity,
) -> (Vec<u32>, Vec<Vec<f32>>) {
    let n = config.n_layer as u32;
    let mut s = TernaryCpuStage::new(config.clone(), w.clone(), LayerRange::full(n), id).unwrap();
    let empty: [f32; 0] = [];
    let mut logits = Vec::new();
    let mut next = 0;
    for (pos, &t) in PROMPT.iter().enumerate() {
        let l = s.step(t, pos, &empty).to_vec();
        next = argmax_first(&l);
        logits.push(l);
    }
    let mut toks = vec![next];
    for i in 1..MAX_NEW {
        let l = s.step(next, PROMPT.len() + i - 1, &empty).to_vec();
        next = argmax_first(&l);
        logits.push(l);
        toks.push(next);
    }
    // run_local is the CLI's oracle — it must agree with this capture.
    let mut s2 = TernaryCpuStage::new(config.clone(), w.clone(), LayerRange::full(n), id).unwrap();
    assert_eq!(
        run_local(&mut s2, &PROMPT, MAX_NEW),
        toks,
        "run_local oracle drift"
    );
    (toks, logits)
}

struct PairOutcome {
    head: std::io::Result<Vec<u32>>,
    tail: std::io::Result<(usize, Vec<Vec<f32>>)>,
}

fn run_pair(
    config: &Config,
    w: &Arc<QwenDeltaNetTernaryWeights>,
    k: u32,
    head_key: [u8; 32],
    tail_key: [u8; 32],
    id: ArtifactIdentity,
    tail_id: ArtifactIdentity,
) -> PairOutcome {
    let n = config.n_layer as u32;
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let (tc, tw) = (config.clone(), w.clone());
    let tail = thread::spawn(move || {
        let mut stage = TernaryCpuStage::new(tc, tw, LayerRange::new(k, n), tail_id).unwrap();
        let (sock, _) = listener.accept().unwrap();
        let mut link = TcpTransport::new(sock, PeerKey(tail_key), Direction::Upstream).unwrap();
        let mut seen = Vec::new();
        run_tail(&mut stage, &mut link, |_, l| seen.push(l.to_vec())).map(|s| (s, seen))
    });
    let mut stage =
        TernaryCpuStage::new(config.clone(), w.clone(), LayerRange::new(0, k), id).unwrap();
    let sock = TcpStream::connect(addr).unwrap();
    let mut link = TcpTransport::new(sock, PeerKey(head_key), Direction::Downstream).unwrap();
    let head = run_head(&mut stage, &mut link, &PROMPT, MAX_NEW).map(|r| r.generated);
    drop(link); // a refusing head must not leave the tail blocked on read
    PairOutcome {
        head,
        tail: tail.join().unwrap(),
    }
}

#[test]
fn two_stage_tcp_stream_is_bit_identical_to_one_box_every_split() {
    let (config, w, id) = load("riir_dist_g1_split");
    let (want_toks, want_logits) = reference(&config, &w, id);
    let key = [7u8; 32];
    for k in 1..config.n_layer as u32 {
        let out = run_pair(&config, &w, k, key, key, id, id);
        let toks = out.head.expect("head");
        let (steps, logits) = out.tail.expect("tail");
        assert_eq!(toks, want_toks, "split@{k}: token stream diverged");
        assert_eq!(steps, PROMPT.len() + MAX_NEW - 1, "split@{k}: step count");
        assert_eq!(logits.len(), want_logits.len());
        for (p, (a, b)) in logits.iter().zip(&want_logits).enumerate() {
            assert!(
                a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()),
                "split@{k}: logits at pos {p} not bit-identical"
            );
        }
    }
}

#[test]
fn wrong_key_refuses_before_any_activation() {
    let (config, w, id) = load("riir_dist_g1_wrongkey");
    let out = run_pair(&config, &w, 2, [1u8; 32], [2u8; 32], id, id);
    let tail_err = out.tail.expect_err("tail must refuse a wrong-key frame");
    assert!(tail_err.to_string().contains("MAC mismatch"), "{tail_err}");
    assert!(
        out.head.is_err(),
        "head must not complete against a refusing tail"
    );
}

#[test]
fn mismatched_artifact_refuses_at_handshake() {
    let (config, w, id) = load("riir_dist_g1_artifact");
    let key = [3u8; 32];
    let foreign = ArtifactIdentity {
        artifact: [0x5A; 32],
        ..id
    };
    let out = run_pair(&config, &w, 2, key, key, id, foreign);
    let tail_err = out.tail.expect_err("tail must refuse a foreign artifact");
    assert!(
        tail_err.to_string().contains("handshake refused"),
        "{tail_err}"
    );
    assert!(out.head.is_err());
}

#[test]
fn peer_key_hex_parses_and_refuses() {
    let k = PeerKey::from_hex(&"0f".repeat(32)).unwrap();
    assert_eq!(k.0, [0x0f; 32]);
    assert!(PeerKey::from_hex("abc").is_err());
    assert!(PeerKey::from_hex(&"zz".repeat(32)).is_err());
}
