//! Plan 623 T2.2 — the 2-stage PoC over REAL TCP sockets (loopback):
//! the pipelined token stream and every tail-side logit vector are
//! BIT-IDENTICAL to the 1-box forward, at every split k; plus the
//! security negatives (wrong key, mismatched artifact) refuse.

#[path = "../../../tests/common/synth_bonsai2.rs"]
mod synth;

use std::net::{TcpListener, TcpStream};
use std::sync::Arc;
use std::thread;

use riir_infer_core::deltanet::{QwenDeltaNetTernaryWeights, WeightScope};
use riir_infer_core::multi_host::LayerRange;
use riir_infer_core::types::Config;
use riir_infer_dist::argmax_first;
use riir_infer_dist::runner::{HeadOptions, run_head_with, run_local, run_local_with, run_tail};
use riir_infer_dist::stage::{ArtifactIdentity, StageCompute, TernaryCpuStage, stage_scope};
use riir_infer_dist::transport::{Direction, PeerKey, TcpTransport};

const PROMPT: [u32; 5] = [7, 13, 42, 100, 5];
const MAX_NEW: usize = 6;

/// `name` must be unique per call site: `write_tmp` keys on pid only, and
/// the tests in this binary run concurrently (the shared-temp-path class).
fn load(name: &str) -> (Config, Arc<QwenDeltaNetTernaryWeights>, ArtifactIdentity) {
    load_scoped(name, |c| WeightScope::full(c.n_layer))
}

/// [`load`] restricted to `scope` (Plan 623 T2.2c — a stage's own slice).
fn load_scoped(
    name: &str,
    scope: impl FnOnce(&Config) -> WeightScope,
) -> (Config, Arc<QwenDeltaNetTernaryWeights>, ArtifactIdentity) {
    let bytes = synth::build_gguf(
        &synth::synth_metadata(false),
        &synth::synth_tensors_bonsai2(),
    );
    let path = synth::write_tmp(name, &bytes);
    let (c, w) =
        riir_infer_core::gguf_loader::load_qwen_deltanet_ternary_weights_gguf_scoped(&path, scope)
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
    run_pair_split(
        config,
        w,
        w,
        k,
        head_key,
        tail_key,
        id,
        tail_id,
        HeadOptions::default(),
    )
}

/// [`run_pair`] with a separate weights view per stage (scoped loads).
#[allow(clippy::too_many_arguments)]
fn run_pair_split(
    config: &Config,
    hw: &Arc<QwenDeltaNetTernaryWeights>,
    tw: &Arc<QwenDeltaNetTernaryWeights>,
    k: u32,
    head_key: [u8; 32],
    tail_key: [u8; 32],
    id: ArtifactIdentity,
    tail_id: ArtifactIdentity,
    opts: HeadOptions,
) -> PairOutcome {
    let n = config.n_layer as u32;
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let (tc, tw) = (config.clone(), tw.clone());
    let tail = thread::spawn(move || {
        let mut stage = TernaryCpuStage::new(tc, tw, LayerRange::new(k, n), tail_id).unwrap();
        let (sock, _) = listener.accept().unwrap();
        let mut link = TcpTransport::new(sock, PeerKey(tail_key), Direction::Upstream).unwrap();
        let mut seen = Vec::new();
        run_tail(&mut stage, &mut link, |_, l| seen.push(l.to_vec())).map(|s| (s, seen))
    });
    let mut stage =
        TernaryCpuStage::new(config.clone(), hw.clone(), LayerRange::new(0, k), id).unwrap();
    let sock = TcpStream::connect(addr).unwrap();
    let mut link = TcpTransport::new(sock, PeerKey(head_key), Direction::Downstream).unwrap();
    let head = run_head_with(&mut stage, &mut link, &PROMPT, MAX_NEW, opts).map(|r| r.generated);
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

/// Plan 623 T2.2c — each stage loads ONLY its own slice of the model and
/// the pipeline stays bit-identical to the full 1-box forward at every
/// split; a slice handed to the wrong window refuses at construction.
#[test]
fn scoped_stage_loads_are_bit_identical_and_refuse_the_wrong_window() {
    let (config, w, id) = load("riir_dist_g1_scoped_full");
    let (want_toks, want_logits) = reference(&config, &w, id);
    let n = config.n_layer as u32;
    let key = [9u8; 32];
    for k in 1..n {
        let (_, hw, hid) = load_scoped(&format!("riir_dist_g1_scoped_h{k}"), |c| {
            stage_scope(LayerRange::new(0, k), c.n_layer as u32)
        });
        let (_, tw, tid) = load_scoped(&format!("riir_dist_g1_scoped_t{k}"), |c| {
            stage_scope(LayerRange::new(k, n), c.n_layer as u32)
        });
        assert_eq!((hid, tid), (id, id), "scoping must not move the identity");
        // Non-vacuity: the slices really are slices.
        assert!(hw.lm_head.rows == 0 && !hw.layers[k as usize].is_resident());
        assert!(tw.wte.rows == 0 && !tw.layers[0].is_resident());
        let out = run_pair_split(
            &config,
            &hw,
            &tw,
            k,
            key,
            key,
            id,
            id,
            HeadOptions::default(),
        );
        let toks = out.head.expect("head");
        let (_, logits) = out.tail.expect("tail");
        assert_eq!(toks, want_toks, "scoped split@{k}: token stream diverged");
        for (p, (a, b)) in logits.iter().zip(&want_logits).enumerate() {
            assert!(
                a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()),
                "scoped split@{k}: logits at pos {p} not bit-identical"
            );
        }
        // A head slice cannot serve as the tail (nor the full model).
        let wrong = TernaryCpuStage::new(config.clone(), hw.clone(), LayerRange::new(k, n), id);
        assert!(wrong.is_err(), "head slice accepted as tail@{k}");
        let wrong = TernaryCpuStage::new(config.clone(), tw.clone(), LayerRange::full(n), id);
        assert!(wrong.is_err(), "tail slice accepted as the full model@{k}");
    }
}

/// Plan 623 T2.3b — chunked prompt frames (`n_tok > 1`) through
/// `StageCompute::prefill_chunk`. The CPU stage's default is the
/// per-position loop, so the token stream, the last prompt position's
/// logits and every decode logit stay bit-identical to the per-position
/// reference — at every split and chunk width (incl. one wider than the
/// prompt, and a ragged final chunk).
#[test]
fn chunked_prefill_frames_are_bit_identical_every_split() {
    let (config, w, id) = load("riir_dist_g1_chunked");
    let (want_toks, want_logits) = reference(&config, &w, id);
    let n = config.n_layer as u32;
    let key = [5u8; 32];
    // Positions the tail reports under chunking: the last prompt position,
    // then every decode step.
    let want_tail = &want_logits[PROMPT.len() - 1..];
    for chunk in [2usize, 3, 7] {
        let opts = HeadOptions {
            prefill_chunk: chunk,
            min_batched: 0,
        };
        let mut oracle =
            TernaryCpuStage::new(config.clone(), w.clone(), LayerRange::full(n), id).unwrap();
        assert_eq!(
            run_local_with(&mut oracle, &PROMPT, MAX_NEW, opts),
            want_toks,
            "chunk {chunk}: chunked oracle drift"
        );
        for k in 1..n {
            let out = run_pair_split(&config, &w, &w, k, key, key, id, id, opts);
            let toks = out.head.expect("head");
            let (steps, logits) = out.tail.expect("tail");
            assert_eq!(toks, want_toks, "chunk {chunk} split@{k}: stream diverged");
            assert_eq!(steps, PROMPT.len() + MAX_NEW - 1, "chunk {chunk} split@{k}");
            assert_eq!(logits.len(), want_tail.len(), "chunk {chunk} split@{k}");
            for (p, (a, b)) in logits.iter().zip(want_tail).enumerate() {
                assert!(
                    a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()),
                    "chunk {chunk} split@{k}: tail logits #{p} not bit-identical"
                );
            }
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

#[test]
fn prefill_width_evens_chunks_and_floors_short_prompts() {
    use riir_infer_dist::runner::prefill_width;
    let o = |prefill_chunk, min_batched| HeadOptions {
        prefill_chunk,
        min_batched,
    };
    assert_eq!(prefill_width(256, o(128, 64)), 128);
    // A ragged 130 = 128 + 2 is evened to 65 + 65.
    assert_eq!(prefill_width(130, o(128, 64)), 65);
    assert_eq!(prefill_width(300, o(128, 64)), 100);
    // Under the floor, or chunking off: per position.
    assert_eq!(prefill_width(40, o(128, 64)), 1);
    assert_eq!(prefill_width(256, o(1, 0)), 1);
    assert_eq!(prefill_width(5, o(2, 0)), 2);
    assert_eq!(prefill_width(5, o(7, 0)), 5);
}
