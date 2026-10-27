//! Plan 623 T2.4 / issue 1147 — the fused decode-batch step (`BatchN`):
//! per-stream tokens BIT-IDENTICAL through the paired path (same math,
//! same per-stream order — the G1 bar), the paired path == the single-
//! drain incumbent path on the same requests (same-binary A/B), the
//! BatchN hint census, s=1 byte-identity (G3), and the batch refusals.

#[path = "../../../tests/common/synth_bonsai2.rs"]
mod synth;

use std::cell::Cell;
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;
use std::thread;

use riir_infer_core::deltanet::{QwenDeltaNetTernaryWeights, WeightScope};
use riir_infer_core::multi_host::{ActivationFrame, FrameDtype, LayerRange};
use riir_infer_core::types::Config;
use riir_infer_dist::multi::{MultiRequest, run_head_multi, run_tail_multi};
use riir_infer_dist::runner::{HeadOptions, HeadReport, run_local_with};
use riir_infer_dist::stage::{ArtifactIdentity, StageCompute, TernaryCpuStage};
use riir_infer_dist::transport::{Direction, FrameKind, PeerKey, TcpTransport, Transport};

const PROMPT_A: [u32; 5] = [7, 13, 42, 100, 5];
const MAX_NEW: usize = 6;

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

type TailCapture = Vec<(u32, usize, Vec<f32>)>;
type CountedRun = (
    std::io::Result<Vec<HeadReport>>,
    std::io::Result<(usize, TailCapture, usize)>,
);

/// A Transport wrapper counting the BatchN hints the TAIL receives — the
/// fused-batch census without touching the protocol.
struct CountingTail {
    inner: TcpTransport,
    batches: Cell<usize>,
}

impl Transport for CountingTail {
    fn send_frame(&mut self, kind: FrameKind, payload: &[u8]) -> std::io::Result<()> {
        self.inner.send_frame(kind, payload)
    }
    fn recv_frame_into(&mut self, out: &mut Vec<u8>) -> std::io::Result<FrameKind> {
        let kind = self.inner.recv_frame_into(out)?;
        if kind == FrameKind::BatchN {
            self.batches.set(self.batches.get() + 1);
        }
        Ok(kind)
    }
}

/// One s-stream run over loopback TCP at split `k`. Returns head reports,
/// the tail's `(seq, pos, logits)` capture in arrival order, and the
/// BatchN hint count.
#[allow(clippy::too_many_arguments)]
fn run_multi_counted(
    config: &Config,
    w: &Arc<QwenDeltaNetTernaryWeights>,
    id: ArtifactIdentity,
    k: u32,
    key: [u8; 32],
    prompts: &[&[u32]],
    max_new: usize,
    opts: HeadOptions,
) -> CountedRun {
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
        let inner = TcpTransport::new(sock, PeerKey(key), Direction::Upstream).unwrap();
        let mut link = CountingTail {
            inner,
            batches: Cell::new(0),
        };
        let mut seen: TailCapture = Vec::new();
        let mut refs: Vec<&mut dyn StageCompute> = stages
            .iter_mut()
            .map(|b| b.as_mut() as &mut dyn StageCompute)
            .collect();
        let out = run_tail_multi(&mut refs, &mut link, |seq, pos, l| {
            seen.push((seq, pos, l.to_vec()));
        });
        out.map(|steps| (steps, seen, link.batches.get()))
    });
    let mut head_stages: Vec<Box<dyn StageCompute>> = (0..n_streams)
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
    let mut refs: Vec<&mut dyn StageCompute> = head_stages
        .iter_mut()
        .map(|b| b.as_mut() as &mut dyn StageCompute)
        .collect();
    let head = run_head_multi(&mut refs, &mut link, &reqs, opts);
    let tail = tail.join().unwrap();
    (head, tail)
}

fn pair_opts(pair: bool) -> HeadOptions {
    HeadOptions {
        pair_decode: pair,
        ..HeadOptions::default()
    }
}

/// Both streams through the PAIRED path land exactly the 1-box oracle's
/// tokens, and the paired run's per-stream tokens + tail capture order
/// equal the incumbent single-drain run on the same requests.
#[test]
fn paired_batch_tokens_bit_identical_and_match_incumbent_path() {
    let (config, w, id) = load("riir_dist_batch_pair_g1");
    let prompt_b: [u32; 8] = [3, 1, 4, 1, 5, 9, 2, 6];
    let prompts: [&[u32]; 2] = [&PROMPT_A, &prompt_b];

    // The incumbent path (single-drain cadence) never hints.
    let (head_i, tail_i) = run_multi_counted(
        &config,
        &w,
        id,
        2,
        [21u8; 32],
        &prompts,
        MAX_NEW,
        pair_opts(false),
    );
    let reports_i = head_i.expect("incumbent head");
    let (steps_i, seen_i, batches_i) = tail_i.expect("incumbent tail");
    assert_eq!(batches_i, 0, "incumbent path must never hint BatchN");

    // The paired path.
    let (head_p, tail_p) = run_multi_counted(
        &config,
        &w,
        id,
        2,
        [22u8; 32],
        &prompts,
        MAX_NEW,
        pair_opts(true),
    );
    let reports_p = head_p.expect("paired head");
    let (steps_p, seen_p, batches_p) = tail_p.expect("paired tail");

    for (i, r) in reports_p.iter().enumerate() {
        assert_eq!(
            r.generated,
            oracle(&config, &w, id, prompts[i], MAX_NEW, pair_opts(true)),
            "paired stream {i} diverged from the 1-box oracle"
        );
        assert_eq!(
            r.generated, reports_i[i].generated,
            "paired stream {i} diverged from the incumbent path"
        );
    }
    // Same steps, same arrival order, same logits rows — the pairing is
    // pure dispatch fusion, never a math change.
    assert_eq!(steps_p, steps_i, "step census changed");
    assert_eq!(seen_p.len(), seen_i.len(), "capture length changed");
    for ((s1, p1, l1), (s2, p2, l2)) in seen_p.iter().zip(seen_i.iter()) {
        assert_eq!(s1, s2, "capture order changed");
        assert_eq!(p1, p2, "capture order changed");
        assert_eq!(l1, l2, "capture logits diverged");
    }

    // The incumbent ran the same requests with zero hints; the paired
    // run fused at least the all-feed passes (both streams feed together
    // once both prefills are done).
    assert!(batches_p > 0, "paired path never fused a pass");
}

/// Equal prompts + chunked prefill put both streams in Feed at the same
/// pass — every decode pass after prefill pairs, so the hint census is
/// EXACTLY MAX_NEW - 1 (the last pass drains both finals).
#[test]
fn lockstep_pairing_hint_census_is_exact() {
    let (config, w, id) = load("riir_dist_batch_lockstep");
    let opts = HeadOptions {
        prefill_chunk: 4,
        min_batched: 0,
        pair_decode: true,
        ..HeadOptions::default()
    };
    let (head, tail) = run_multi_counted(
        &config,
        &w,
        id,
        2,
        [26u8; 32],
        &[&PROMPT_A, &PROMPT_A],
        MAX_NEW,
        opts,
    );
    let reports = head.expect("lockstep head");
    let (steps, _, batches) = tail.expect("lockstep tail");
    for (i, r) in reports.iter().enumerate() {
        assert_eq!(
            r.generated,
            oracle(&config, &w, id, &PROMPT_A, MAX_NEW, opts),
            "lockstep stream {i} diverged from the 1-box oracle"
        );
    }
    // Per-position steps on the tail: every feed row steps once.
    let want_steps = 2 * (PROMPT_A.len() + MAX_NEW - 1);
    assert_eq!(steps, want_steps, "lockstep step census");
    assert_eq!(batches, MAX_NEW - 1, "lockstep hint census");
}

/// s=1 is byte-identical under both arms (G3): the hint never fires at
/// one ready stream and the drain-all degenerates to the single drain.
#[test]
fn one_stream_paired_and_incumbent_are_byte_identical() {
    let (config, w, id) = load("riir_dist_batch_s1");
    let on: Vec<u32> = {
        let (head, tail) = run_multi_counted(
            &config,
            &w,
            id,
            2,
            [23u8; 32],
            &[&PROMPT_A],
            MAX_NEW,
            pair_opts(true),
        );
        let (_, _, batches) = tail.expect("s=1 paired tail");
        assert_eq!(batches, 0, "s=1 must never hint");
        head.expect("s=1 paired head")[0].generated.clone()
    };
    let off: Vec<u32> = {
        let (head, tail) = run_multi_counted(
            &config,
            &w,
            id,
            2,
            [24u8; 32],
            &[&PROMPT_A],
            MAX_NEW,
            pair_opts(false),
        );
        let _ = tail.expect("s=1 incumbent tail");
        head.expect("s=1 incumbent head")[0].generated.clone()
    };
    assert_eq!(on, off, "s=1 paired vs incumbent");
    assert_eq!(
        on,
        oracle(&config, &w, id, &PROMPT_A, MAX_NEW, pair_opts(true)),
        "s=1 vs the 1-box oracle"
    );
}

/// BatchN refusals: count 0, count over the table, a duplicated stream
/// inside one batch, and a row whose stream never began — each on a
/// fresh connection (a refusal kills the run).
#[test]
fn batch_hint_refusals() {
    let (config, w, id) = load("riir_dist_batch_refuse");
    let n = config.n_layer as u32;
    let key = [25u8; 32];
    let empty: [f32; 0] = [];

    // A legal decode feed row for stream `seq` at `pos` (the head-stage
    // step gives the row real hidden bytes; the tail never computes with
    // it — the refusal fires first — but decode_into must parse it).
    let feed_frame = |head_stage: &mut TernaryCpuStage, seq: u32, pos: usize| {
        let out = head_stage.step(PROMPT_A[0], pos, &empty);
        let mut frame = Vec::new();
        ActivationFrame::encode_into(seq, pos as u32, 1, out, FrameDtype::F32, &mut frame);
        frame
    };

    let one_arm = |name: &str, drive: &dyn Fn(&mut TcpTransport, &mut TernaryCpuStage)| {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let tc = config.clone();
        let tw = w.clone();
        let idc = id;
        let tail = thread::spawn(move || {
            let mut stages: Vec<Box<dyn StageCompute>> = (0..2)
                .map(|_| {
                    Box::new(
                        TernaryCpuStage::new(tc.clone(), tw.clone(), LayerRange::new(1, n), idc)
                            .unwrap(),
                    ) as Box<dyn StageCompute>
                })
                .collect();
            let (sock, _) = listener.accept().unwrap();
            let mut link = TcpTransport::new(sock, PeerKey(key), Direction::Upstream).unwrap();
            let mut refs: Vec<&mut dyn StageCompute> = stages
                .iter_mut()
                .map(|b| b.as_mut() as &mut dyn StageCompute)
                .collect();
            run_tail_multi(&mut refs, &mut link, |_, _, _| {})
        });
        let mut head_stage =
            TernaryCpuStage::new(config.clone(), w.clone(), LayerRange::new(0, 1), id).unwrap();
        let sock = TcpStream::connect(addr).unwrap();
        let mut link = TcpTransport::new(sock, PeerKey(key), Direction::Downstream).unwrap();
        drive(&mut link, &mut head_stage);
        let err = tail
            .join()
            .unwrap()
            .err()
            .unwrap_or_else(|| panic!("{name}: tail must refuse"));
        err.to_string()
    };

    let handshake = |link: &mut TcpTransport, head_stage: &TernaryCpuStage| {
        let mut buf = Vec::new();
        riir_infer_dist::handshake::encode(head_stage.handshake(), &mut buf);
        link.send_frame(FrameKind::Handshake, &buf).unwrap();
        let kind = link.recv_frame_into(&mut buf).unwrap();
        assert_eq!(kind, FrameKind::Handshake);
    };
    let begin = |link: &mut TcpTransport, seq: u32, plen: usize, chunk: usize| {
        let mut b = [0u8; 12];
        b[..4].copy_from_slice(&seq.to_le_bytes());
        b[4..8].copy_from_slice(&(plen as u32).to_le_bytes());
        b[8..].copy_from_slice(&(chunk as u32).to_le_bytes());
        link.send_frame(FrameKind::BeginStream, &b).unwrap();
    };
    let batch_n = |link: &mut TcpTransport, count: u32| {
        link.send_frame(FrameKind::BatchN, &count.to_le_bytes())
            .unwrap();
    };

    // 1. count 0.
    let e = one_arm("count-0", &|link, hs| {
        handshake(link, hs);
        begin(link, 0, 4, 1);
        batch_n(link, 0);
    });
    assert!(e.contains("outside"), "count-0: {e}");

    // 2. count over the table (2 stages).
    let e = one_arm("count-over", &|link, hs| {
        handshake(link, hs);
        begin(link, 0, 4, 1);
        batch_n(link, 3);
    });
    assert!(e.contains("outside"), "count-over: {e}");

    // 3. the same stream twice in one batch (row 0 is a legal feed at the
    // stream's next_pos; row 1 duplicates the seq — refused pre-compute).
    let e = one_arm("dup-seq", &|link, hs| {
        handshake(link, hs);
        begin(link, 0, 4, 1);
        let act = feed_frame(hs, 0, 0);
        batch_n(link, 2);
        link.send_frame(FrameKind::Activation, &act).unwrap();
        link.send_frame(FrameKind::Activation, &act).unwrap();
    });
    assert!(e.contains("duplicated"), "dup-seq: {e}");

    // 4. a row whose stream never began.
    let e = one_arm("no-begin", &|link, hs| {
        handshake(link, hs);
        begin(link, 0, 4, 1);
        let act0 = feed_frame(hs, 0, 0);
        let act1 = feed_frame(hs, 1, 0);
        batch_n(link, 2);
        link.send_frame(FrameKind::Activation, &act0).unwrap();
        link.send_frame(FrameKind::Activation, &act1).unwrap();
    });
    assert!(e.contains("no BeginStream"), "no-begin: {e}");

    // 5. a prefilling row (pos < prompt_len on a chunked track).
    let e = one_arm("prefill-row", &|link, hs| {
        handshake(link, hs);
        begin(link, 0, 8, 8);
        let act = feed_frame(hs, 0, 0);
        batch_n(link, 1);
        link.send_frame(FrameKind::Activation, &act).unwrap();
    });
    assert!(e.contains("prefilling"), "prefill-row: {e}");

    // 6. a multi-token row — the fused batch carries decode feeds only.
    let e = one_arm("multi-token", &|link, hs| {
        handshake(link, hs);
        begin(link, 0, 4, 1);
        let row = hs.step(PROMPT_A[0], 0, &empty).to_vec();
        let mut two_rows = row.clone();
        two_rows.extend_from_slice(&row);
        let mut f2 = Vec::new();
        ActivationFrame::encode_into(0, 0, 2, &two_rows, FrameDtype::F32, &mut f2);
        batch_n(link, 1);
        link.send_frame(FrameKind::Activation, &f2).unwrap();
    });
    assert!(e.contains("decode feeds only"), "multi-token: {e}");
}
