//! Plan 623 T2.4 — the multi-stream round-robin over REAL TCP (loopback):
//! per-stream token streams BIT-IDENTICAL to the 1-box oracle at every
//! split (the G1 bar), one stream's prefill chunks interleaving another
//! stream's decode steps (the ordering the scheduler exists for), the
//! loopback overlap read, and the tail-side refusals.

#[path = "../../../tests/common/synth_bonsai2.rs"]
mod synth;

use std::net::{TcpListener, TcpStream};
use std::sync::Arc;
use std::thread;
use std::time::Instant;

use riir_infer_core::deltanet::{QwenDeltaNetTernaryWeights, WeightScope};
use riir_infer_core::multi_host::{ActivationFrame, FrameDtype, LayerRange};
use riir_infer_core::types::Config;
use riir_infer_dist::multi::{MultiRequest, run_head_multi, run_tail_multi};
use riir_infer_dist::runner::{HeadOptions, HeadReport, run_head_with, run_local_with, run_tail};
use riir_infer_dist::stage::{ArtifactIdentity, StageCompute, TernaryCpuStage};
use riir_infer_dist::transport::{Direction, FrameKind, PeerKey, TcpTransport, Transport};

const PROMPT_A: [u32; 5] = [7, 13, 42, 100, 5];
const PROMPT_B: [u32; 8] = [3, 1, 4, 1, 5, 9, 2, 6];
const MAX_NEW: usize = 6;

/// `name` must be unique per call site: `write_tmp` keys on pid only, and
/// the tests in this binary run concurrently (the shared-temp-path class).
fn load(name: &str) -> (Config, Arc<QwenDeltaNetTernaryWeights>, ArtifactIdentity) {
    let bytes = synth::build_gguf(
        &synth::synth_metadata(false),
        &synth::synth_tensors_bonsai2(),
    );
    let path = synth::write_tmp(name, &bytes);
    let (c, w) =
        riir_infer_core::gguf_loader::load_qwen_deltanet_ternary_weights_gguf_scoped(&path, |c| {
            WeightScope::full(c.n_layer)
        })
        .expect("load synth");
    let id = ArtifactIdentity::of_gguf(&path).expect("identity");
    let _ = std::fs::remove_file(&path);
    (c, Arc::new(w), id)
}

/// 1-box oracle for one prompt at `opts`.
fn oracle(
    config: &Config,
    w: &Arc<QwenDeltaNetTernaryWeights>,
    id: ArtifactIdentity,
    prompt: &[u32],
    max_new: usize,
    opts: HeadOptions,
) -> Vec<u32> {
    let n = config.n_layer as u32;
    let mut s = TernaryCpuStage::new(config.clone(), w.clone(), LayerRange::full(n), id).unwrap();
    run_local_with(&mut s, prompt, max_new, opts)
}

/// 1-box per-position logits over `prompt` (stream 0's capture reference).
fn oracle_logits(
    config: &Config,
    w: &Arc<QwenDeltaNetTernaryWeights>,
    id: ArtifactIdentity,
    prompt: &[u32],
) -> Vec<Vec<f32>> {
    let n = config.n_layer as u32;
    let mut s = TernaryCpuStage::new(config.clone(), w.clone(), LayerRange::full(n), id).unwrap();
    let empty: [f32; 0] = [];
    let mut rows = Vec::new();
    for &t in prompt {
        rows.push(s.step(t, rows.len(), &empty).to_vec());
    }
    rows
}

type TailCapture = Vec<(u32, usize, Vec<f32>)>;

/// One multi-stream pair over loopback TCP at split `k`: stream `i` =
/// request `i`, both sides one `TernaryCpuStage` per stream (the weights
/// Arc-shared). Returns per-stream head reports + the tail's
/// `(seq, pos, logits)` capture in the order the tail saw it.
#[allow(clippy::too_many_arguments)]
fn run_multi(
    config: &Config,
    w: &Arc<QwenDeltaNetTernaryWeights>,
    id: ArtifactIdentity,
    k: u32,
    key: [u8; 32],
    prompts: &[&[u32]],
    max_new: usize,
    opts: HeadOptions,
    capture_tail: bool,
) -> (
    std::io::Result<Vec<HeadReport>>,
    std::io::Result<(usize, TailCapture)>,
) {
    let n = config.n_layer as u32;
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let (tc, tw) = (config.clone(), w.clone());
    let n_streams = prompts.len();
    let tail = thread::spawn(move || {
        let mut stages: Vec<Box<dyn StageCompute>> = (0..n_streams)
            .map(|_| {
                Box::new(
                    TernaryCpuStage::new(tc.clone(), tw.clone(), LayerRange::new(k, n), id)
                        .unwrap(),
                ) as Box<dyn StageCompute>
            })
            .collect();
        let (sock, _) = listener.accept().unwrap();
        let mut link = TcpTransport::new(sock, PeerKey(key), Direction::Upstream).unwrap();
        let mut seen: TailCapture = Vec::new();
        let mut refs: Vec<&mut dyn StageCompute> = stages
            .iter_mut()
            .map(|b| b.as_mut() as &mut dyn StageCompute)
            .collect();
        run_tail_multi(&mut refs, &mut link, |seq, pos, l| {
            if capture_tail {
                seen.push((seq, pos, l.to_vec()));
            }
        })
        .map(|steps| (steps, seen))
    });
    let mut stages: Vec<Box<dyn StageCompute>> = (0..n_streams)
        .map(|_| {
            Box::new(
                TernaryCpuStage::new(config.clone(), w.clone(), LayerRange::new(0, k), id).unwrap(),
            ) as Box<dyn StageCompute>
        })
        .collect();
    let sock = TcpStream::connect(addr).unwrap();
    let mut link = TcpTransport::new(sock, PeerKey(key), Direction::Downstream).unwrap();
    let reqs: Vec<MultiRequest> = prompts
        .iter()
        .map(|p| MultiRequest::new(p.to_vec(), max_new))
        .collect();
    let head = {
        let mut refs: Vec<&mut dyn StageCompute> = stages
            .iter_mut()
            .map(|b| b.as_mut() as &mut dyn StageCompute)
            .collect();
        run_head_multi(&mut refs, &mut link, &reqs, opts)
    };
    drop(link); // a refusing head must not leave the tail blocked on read
    (head, tail.join().unwrap())
}

#[test]
fn multi_stream_tokens_bit_identical_per_stream_every_split() {
    let (config, w, id) = load("riir_dist_multi_g1");
    let opts = HeadOptions::default(); // prompts under min_batched → per position
    let want_a = oracle(&config, &w, id, &PROMPT_A, MAX_NEW, opts);
    let want_b = oracle(&config, &w, id, &PROMPT_B, MAX_NEW, opts);
    let want_logits = oracle_logits(&config, &w, id, &PROMPT_A);
    let n = config.n_layer as u32;
    let key = [11u8; 32];
    for k in 1..n {
        let (head, tail) = run_multi(
            &config,
            &w,
            id,
            k,
            key,
            &[&PROMPT_A, &PROMPT_B],
            MAX_NEW,
            opts,
            true,
        );
        let reports = head.expect("head");
        assert_eq!(reports.len(), 2);
        assert_eq!(reports[0].generated, want_a, "split@{k}: stream 0 diverged");
        assert_eq!(reports[1].generated, want_b, "split@{k}: stream 1 diverged");
        let (steps, logits) = tail.expect("tail");
        assert_eq!(
            steps,
            PROMPT_A.len() + MAX_NEW - 1 + PROMPT_B.len() + MAX_NEW - 1,
            "split@{k}: step census"
        );
        // Stream 0's per-position logits must match the 1-box forward
        // bit-for-bit; its capture interleaves stream 1's and carries
        // decode events too, so filter to stream 0's PROMPT positions.
        let s0: Vec<(usize, &Vec<f32>)> = logits
            .iter()
            .filter(|(s, p, _)| *s == 0 && *p < PROMPT_A.len())
            .map(|(_, p, l)| (*p, l))
            .collect();
        assert_eq!(
            s0.len(),
            want_logits.len(),
            "split@{k}: stream 0 logit count"
        );
        for (p, (pos, a)) in s0.iter().enumerate() {
            assert_eq!(*pos, p, "split@{k}: stream 0 position order");
            let want = want_logits[p].as_slice();
            assert!(
                a.len() >= want.len()
                    && a[..want.len()]
                        .iter()
                        .zip(want)
                        .all(|(x, y)| x.to_bits() == y.to_bits()),
                "split@{k}: stream 0 logits at pos {p} not bit-identical"
            );
        }
    }
}

/// The scheduler's reason to exist: stream B's prompt chunks land at the
/// tail INTERLEAVED with stream A's decode steps — not strictly before
/// (the serial shape) and A keeps decoding after B's prefill ends.
#[test]
fn stream_b_prefill_interleaves_stream_a_decode() {
    let (config, w, id) = load("riir_dist_multi_interleave");
    // A: short prompt, long decode. B: long prompt (chunked), short decode.
    let prompt_a: [u32; 3] = [9, 8, 7];
    let prompt_b: [u32; 12] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];
    let opts = HeadOptions {
        prefill_chunk: 3,
        min_batched: 0,
        ..HeadOptions::default()
    };
    let (head, tail) = run_multi(
        &config,
        &w,
        id,
        2,
        [12u8; 32],
        &[&prompt_a, &prompt_b],
        10,
        opts,
        true,
    );
    let reports = head.expect("head");
    assert_eq!(
        reports[0].generated,
        oracle(&config, &w, id, &prompt_a, 10, opts)
    );
    assert_eq!(
        reports[1].generated,
        oracle(&config, &w, id, &prompt_b, 10, opts)
    );
    let (_, logits) = tail.expect("tail");
    let idx = |want: (u32, usize)| logits.iter().position(|&(s, p, _)| (s, p) == want);
    // Stream 0's chunked prefill reports only its last prompt position;
    // every decode step reports its own.
    let a_decodes: Vec<usize> = (prompt_a.len()..prompt_a.len() + 9)
        .map(|p| idx((0, p)))
        .collect::<Option<Vec<_>>>()
        .expect("stream 0 decode events");
    let b_prefill_last = idx((1, prompt_b.len() - 1)).expect("stream 1 last prefill event");
    let first_a_decode = *a_decodes.iter().min().unwrap();
    let last_a_decode = *a_decodes.iter().max().unwrap();
    assert!(
        b_prefill_last > first_a_decode,
        "stream 1 prefill finished before stream 0 decode began — no interleave"
    );
    assert!(
        last_a_decode > b_prefill_last,
        "stream 0 decode never resumed after stream 1's last prefill chunk"
    );
}

/// The degenerate 1-stream case must agree with the single-stream runner
/// (the same protocol shape, one stream).
#[test]
fn one_stream_multi_matches_run_head() {
    let (config, w, id) = load("riir_dist_multi_one");
    let n = config.n_layer as u32;
    let key = [13u8; 32];
    let opts = HeadOptions::default();
    let (head, _tail) = run_multi(&config, &w, id, 2, key, &[&PROMPT_A], MAX_NEW, opts, false);
    let reports = head.expect("head");
    assert_eq!(reports.len(), 1);
    // The single-stream pair at the same split, for the byte-compare.
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let tc = config.clone();
    let tw = w.clone();
    let tid = id;
    let tail2 = thread::spawn(move || {
        let mut stage = TernaryCpuStage::new(tc, tw, LayerRange::new(2, n), tid).unwrap();
        let (sock, _) = listener.accept().unwrap();
        let mut link = TcpTransport::new(sock, PeerKey(key), Direction::Upstream).unwrap();
        run_tail(&mut stage, &mut link, |_, _| {})
    });
    let mut single =
        TernaryCpuStage::new(config.clone(), w.clone(), LayerRange::new(0, 2), id).unwrap();
    let sock = TcpStream::connect(addr).unwrap();
    let mut link = TcpTransport::new(sock, PeerKey(key), Direction::Downstream).unwrap();
    let single_out =
        run_head_with(&mut single, &mut link, &PROMPT_A, MAX_NEW, opts).expect("single run_head");
    tail2.join().unwrap().expect("single tail");
    assert_eq!(reports[0].generated, single_out.generated);
}

/// Loopback overlap read (CPU stages, tiny compute, wire-RTT dominated):
/// two streams round-robined take clearly LESS than running them
/// serially. A RATIO bar — both arms scale with box load together; the
/// real G2 cells belong to the bin on quiet hardware.
#[test]
fn two_streams_overlap_serial_vs_interleaved() {
    let (config, w, id) = load("riir_dist_multi_overlap");
    let key = [14u8; 32];
    let opts = HeadOptions::default();
    let max_new = 32;
    // Serial: two connections, one after the other.
    let t = Instant::now();
    for _ in 0..2 {
        let (head, tail) = run_multi(&config, &w, id, 2, key, &[&PROMPT_A], max_new, opts, false);
        head.expect("serial head");
        tail.expect("serial tail");
    }
    let serial = t.elapsed();
    // Interleaved: both streams over ONE connection.
    let t = Instant::now();
    let (head, tail) = run_multi(
        &config,
        &w,
        id,
        2,
        key,
        &[&PROMPT_A, &PROMPT_A],
        max_new,
        opts,
        false,
    );
    let reports = head.expect("interleaved head");
    tail.expect("interleaved tail");
    let interl = t.elapsed();
    let want = oracle(&config, &w, id, &PROMPT_A, max_new, opts);
    assert_eq!(reports[0].generated, want, "overlap run stream 0 diverged");
    assert_eq!(reports[1].generated, want, "overlap run stream 1 diverged");
    let ratio = interl.as_secs_f64() / serial.as_secs_f64();
    println!("overlap: serial {serial:.3?} interleaved {interl:.3?} ratio {ratio:.3}");
    assert!(
        ratio < 0.85,
        "no measurable overlap: interleaved/serial {ratio:.3} (want < 0.85)"
    );
}

/// An activation naming a stream with no BeginStream refuses loud —
/// per-stream state cannot be implicitly created.
#[test]
fn activation_without_begin_stream_refuses() {
    let (config, w, id) = load("riir_dist_multi_nobegin");
    let n = config.n_layer as u32;
    let key = [15u8; 32];
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let tc = config.clone();
    let tw = w.clone();
    let tid = id;
    let tail = thread::spawn(move || {
        let mut stage = TernaryCpuStage::new(tc, tw, LayerRange::new(2, n), tid).unwrap();
        let (sock, _) = listener.accept().unwrap();
        let mut link = TcpTransport::new(sock, PeerKey(key), Direction::Upstream).unwrap();
        let mut refs: Vec<&mut dyn StageCompute> = vec![&mut stage as &mut dyn StageCompute];
        run_tail_multi(&mut refs, &mut link, |_, _, _| {})
    });
    let mut head_stage =
        TernaryCpuStage::new(config.clone(), w.clone(), LayerRange::new(0, 2), id).unwrap();
    let sock = TcpStream::connect(addr).unwrap();
    let mut link = TcpTransport::new(sock, PeerKey(key), Direction::Downstream).unwrap();
    // Legal handshake, then an activation for stream 0 WITHOUT BeginStream.
    let mut buf = Vec::new();
    riir_infer_dist::handshake::encode(head_stage.handshake(), &mut buf);
    link.send_frame(FrameKind::Handshake, &buf).unwrap();
    let kind = link.recv_frame_into(&mut buf).unwrap();
    assert_eq!(kind, FrameKind::Handshake);
    let empty: [f32; 0] = [];
    let out = head_stage.step(PROMPT_A[0], 0, &empty);
    let mut frame = Vec::new();
    ActivationFrame::encode_into(0, 0, 1, out, FrameDtype::F32, &mut frame);
    link.send_frame(FrameKind::Activation, &frame).unwrap();
    let err = tail.join().unwrap().expect_err("tail must refuse");
    assert!(
        err.to_string().contains("no BeginStream"),
        "wrong refusal: {err}"
    );
}

/// A stream cannot be opened twice — the second BeginStream refuses.
#[test]
fn duplicate_begin_stream_refuses() {
    let (config, w, id) = load("riir_dist_multi_dupbegin");
    let n = config.n_layer as u32;
    let key = [16u8; 32];
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let tc = config.clone();
    let tw = w.clone();
    let tid = id;
    let tail = thread::spawn(move || {
        let mut stage = TernaryCpuStage::new(tc, tw, LayerRange::new(2, n), tid).unwrap();
        let (sock, _) = listener.accept().unwrap();
        let mut link = TcpTransport::new(sock, PeerKey(key), Direction::Upstream).unwrap();
        let mut refs: Vec<&mut dyn StageCompute> = vec![&mut stage as &mut dyn StageCompute];
        run_tail_multi(&mut refs, &mut link, |_, _, _| {})
    });
    let head_stage =
        TernaryCpuStage::new(config.clone(), w.clone(), LayerRange::new(0, 2), id).unwrap();
    let sock = TcpStream::connect(addr).unwrap();
    let mut link = TcpTransport::new(sock, PeerKey(key), Direction::Downstream).unwrap();
    let mut buf = Vec::new();
    riir_infer_dist::handshake::encode(head_stage.handshake(), &mut buf);
    link.send_frame(FrameKind::Handshake, &buf).unwrap();
    link.recv_frame_into(&mut buf).unwrap();
    let mut begin = [0u8; 12];
    begin[..4].copy_from_slice(&0u32.to_le_bytes());
    begin[4..8].copy_from_slice(&5u32.to_le_bytes());
    begin[8..].copy_from_slice(&1u32.to_le_bytes());
    link.send_frame(FrameKind::BeginStream, &begin).unwrap();
    link.send_frame(FrameKind::BeginStream, &begin).unwrap();
    let err = tail.join().unwrap().expect_err("tail must refuse");
    assert!(
        err.to_string().contains("already open"),
        "wrong refusal: {err}"
    );
}

/// Per-stream position cursors are independent: stream 1 restarts at
/// position 0 while stream 0 is deep in its sequence — the single-stream
/// global `pos == steps` ordering would refuse this legal interleave.
#[test]
fn per_stream_positions_are_independent() {
    let (config, w, id) = load("riir_dist_multi_pos");
    let prompt_long: Vec<u32> = (100..110).collect();
    let opts = HeadOptions::default();
    let (head, tail) = run_multi(
        &config,
        &w,
        id,
        2,
        [17u8; 32],
        &[&prompt_long, &PROMPT_B],
        8,
        opts,
        true,
    );
    let reports = head.expect("head");
    assert_eq!(
        reports[0].generated,
        oracle(&config, &w, id, &prompt_long, 8, opts)
    );
    assert_eq!(
        reports[1].generated,
        oracle(&config, &w, id, &PROMPT_B, 8, opts)
    );
    let (steps, logits) = tail.expect("tail");
    assert_eq!(
        steps,
        prompt_long.len() + 7 + PROMPT_B.len() + 7,
        "tail step census"
    );
    // Only reply-expected + decode positions appear: stream 1's capture
    // starts at 0 and stream 0's ends at its last decode position.
    let s1_positions: Vec<usize> = logits
        .iter()
        .filter(|(s, _, _)| *s == 1)
        .map(|(_, p, _)| *p)
        .collect();
    assert_eq!(s1_positions.first(), Some(&0));
    let s0_positions: Vec<usize> = logits
        .iter()
        .filter(|(s, _, _)| *s == 0)
        .map(|(_, p, _)| *p)
        .collect();
    assert_eq!(
        s0_positions.last(),
        Some(&(prompt_long.len() + 6)),
        "stream 0's last captured position"
    );
}
