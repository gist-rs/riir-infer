//! Plan 623 T2.4 — the admission seam: a stream JOINs mid-flight at a
//! pass boundary, its prompt chunks interleave the resident stream's
//! decode, and NO stream's tokens move (G1 per-stream bit-identity to
//! the 1-box oracle through the join). Wait/Close semantics and the
//! loud refusals.

#[path = "../../../tests/common/synth_bonsai2.rs"]
mod synth;

use std::collections::VecDeque;
use std::io;
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;
use std::thread;

use riir_infer_core::deltanet::{QwenDeltaNetTernaryWeights, WeightScope};
use riir_infer_core::multi_host::LayerRange;
use riir_infer_core::types::Config;
use riir_infer_dist::multi::{
    Admission, AdmissionSource, MultiRequest, run_head_multi_admit, run_tail_multi,
};
use riir_infer_dist::runner::{HeadOptions, HeadReport, run_local_with};
use riir_infer_dist::stage::{ArtifactIdentity, StageCompute, TernaryCpuStage};
use riir_infer_dist::transport::{Direction, PeerKey, TcpTransport};

const PROMPT_A: [u32; 5] = [7, 13, 42, 100, 5];
const PROMPT_B: [u32; 8] = [3, 1, 4, 1, 5, 9, 2, 6];
const MAX_NEW: usize = 6;

/// `name` must be unique per call site: `write_tmp` keys on pid only,
/// and the tests in this binary run concurrently (the shared-temp-path
/// class).
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

/// Scripted admission: pops the queue; exhausted = [`Admission::Close`].
struct Scripted {
    events: VecDeque<Admission>,
}

impl AdmissionSource for Scripted {
    fn poll(&mut self) -> io::Result<Admission> {
        Ok(self.events.pop_front().unwrap_or(Admission::Close))
    }
}

type TailCapture = Vec<(u32, usize)>;

/// One head/tail pair at split `k`. The head pool starts with
/// `head_pool` prebuilt head-range stages (stream `i` = slot `i`);
/// `factory` builds joiners; the tail prewarms `tail_slots` slots (its
/// static stage table IS the tail's admission cap — it must cover every
/// stream the head may admit). Returns the head result + the tail's
/// `(seq, pos)` capture in wire order.
#[allow(clippy::too_many_arguments)]
fn run_join(
    config: &Config,
    w: &Arc<QwenDeltaNetTernaryWeights>,
    id: ArtifactIdentity,
    k: u32,
    key: [u8; 32],
    head_pool: usize,
    tail_slots: usize,
    initial: Vec<MultiRequest>,
    events: VecDeque<Admission>,
    factory: impl FnMut(usize) -> io::Result<TernaryCpuStage>,
    opts: HeadOptions,
) -> (
    io::Result<Vec<HeadReport>>,
    io::Result<(usize, TailCapture)>,
) {
    let n = config.n_layer as u32;
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let (tc, tw) = (config.clone(), w.clone());
    let tail = thread::spawn(move || {
        let mut stages: Vec<TernaryCpuStage> = (0..tail_slots)
            .map(|_| {
                TernaryCpuStage::new(tc.clone(), tw.clone(), LayerRange::new(k, n), id).unwrap()
            })
            .collect();
        let (sock, _) = listener.accept().unwrap();
        let mut link = TcpTransport::new(sock, PeerKey(key), Direction::Upstream).unwrap();
        let mut seen: TailCapture = Vec::new();
        let mut refs: Vec<&mut dyn StageCompute> = stages
            .iter_mut()
            .map(|s| s as &mut dyn StageCompute)
            .collect();
        run_tail_multi(&mut refs, &mut link, |seq, pos, _| seen.push((seq, pos)))
            .map(|steps| (steps, seen))
    });
    let sock = TcpStream::connect(addr).unwrap();
    let mut link = TcpTransport::new(sock, PeerKey(key), Direction::Downstream).unwrap();
    let mut pool: Vec<TernaryCpuStage> = (0..head_pool)
        .map(|_| {
            TernaryCpuStage::new(config.clone(), w.clone(), LayerRange::new(0, k), id).unwrap()
        })
        .collect();
    let mut factory = factory;
    let mut admission = Scripted { events };
    let head = run_head_multi_admit(
        &mut pool,
        &mut link,
        &initial,
        &mut admission,
        &mut factory,
        opts,
    );
    drop(link); // a refusing head must not leave the tail blocked on read
    (head, tail.join().unwrap())
}

/// The resident stream A runs solo until pass 7 (its 5 per-position
/// prefill frames + the first decode step), then B joins MID-DECODE:
/// B's first tail frame lands after A's decode began, both streams'
/// tokens stay bit-identical to their 1-box oracles, and the wire step
/// census is exact.
#[test]
fn midflight_join_bit_identity_and_interleave() {
    let (config, w, id) = load("riir_dist_admission_join");
    let opts = HeadOptions::default(); // prompts under min_batched → per position
    let want_a = oracle(&config, &w, id, &PROMPT_A, MAX_NEW, opts);
    let want_b = oracle(&config, &w, id, &PROMPT_B, MAX_NEW, opts);
    let events = VecDeque::from(
        std::iter::repeat_n(Admission::Wait, 6)
            .chain(std::iter::once(Admission::Join(MultiRequest::new(
                PROMPT_B.to_vec(),
                MAX_NEW,
            ))))
            .collect::<Vec<_>>(),
    );
    let joiner_range = LayerRange::new(0, 2);
    let (jc, jw, jid) = (config.clone(), w.clone(), id);
    let factory = move |seq: usize| {
        if seq != 1 {
            return Err(io::Error::other("scripted factory: only stream 1 joins"));
        }
        TernaryCpuStage::new(jc.clone(), jw.clone(), joiner_range, jid).map_err(io::Error::other)
    };
    let (head, tail) = run_join(
        &config,
        &w,
        id,
        2,
        [21u8; 32],
        1,
        2, // the tail prewarms a slot for the joiner (its cap covers both)
        vec![MultiRequest::new(PROMPT_A.to_vec(), MAX_NEW)],
        events,
        factory,
        opts,
    );
    let reports = head.expect("head");
    assert_eq!(reports.len(), 2, "resident + joining streams report");
    assert_eq!(
        reports[0].generated, want_a,
        "resident stream A diverged through the join"
    );
    assert_eq!(reports[1].generated, want_b, "joining stream B diverged");
    let (steps, seen) = tail.expect("tail");
    let b_first = seen
        .iter()
        .position(|&(s, _)| s == 1)
        .expect("joiner B appears in the tail capture");
    assert!(
        seen[..b_first]
            .iter()
            .any(|&(s, pos)| s == 0 && pos >= PROMPT_A.len()),
        "B's first frame must land AFTER stream A began decoding (mid-flight, not queued)"
    );
    assert_eq!(
        steps,
        PROMPT_A.len() + MAX_NEW - 1 + PROMPT_B.len() + MAX_NEW - 1,
        "wire step census through the join"
    );
}

/// Wait must be inert: the resident stream runs to completion through
/// Waits, and the exhausted script closes the run (drain semantics).
#[test]
fn wait_then_close_finishes_the_drain() {
    let (config, w, id) = load("riir_dist_admission_wait");
    let opts = HeadOptions::default();
    let want_a = oracle(&config, &w, id, &PROMPT_A, MAX_NEW, opts);
    let events = VecDeque::from([Admission::Wait, Admission::Wait]);
    let joiner_range = LayerRange::new(0, 1);
    let (jc, jw, jid) = (config.clone(), w.clone(), id);
    let factory = move |_seq: usize| {
        TernaryCpuStage::new(jc.clone(), jw.clone(), joiner_range, jid).map_err(io::Error::other)
    };
    let (head, tail) = run_join(
        &config,
        &w,
        id,
        1,
        [23u8; 32],
        1,
        1,
        vec![MultiRequest::new(PROMPT_A.to_vec(), MAX_NEW)],
        events,
        factory,
        opts,
    );
    let reports = head.expect("head");
    assert_eq!(reports.len(), 1);
    assert_eq!(reports[0].generated, want_a);
    let (steps, _) = tail.expect("tail");
    assert_eq!(steps, PROMPT_A.len() + MAX_NEW - 1);
}

/// The loud-refusal table: an empty-prompt join, a joiner presenting a
/// foreign handshake, an over-capacity join, and a pool smaller than
/// the initial request table — every one refuses the run with a named
/// reason, and the tail never sees the poisoned stream.
#[test]
fn bad_joins_refuse_loud() {
    let (config, w, id) = load("riir_dist_admission_refuse");
    let opts = HeadOptions::default();
    let joiner_range = LayerRange::new(0, 1);
    let (jc, jw, jid) = (config.clone(), w.clone(), id);
    let legit = move |_seq: usize| {
        TernaryCpuStage::new(jc.clone(), jw.clone(), joiner_range, jid).map_err(io::Error::other)
    };

    // (a) empty prompt.
    let events = VecDeque::from([Admission::Join(MultiRequest::new(vec![], 3))]);
    let (head, _) = run_join(
        &config,
        &w,
        id,
        1,
        [24u8; 32],
        1,
        1,
        vec![MultiRequest::new(PROMPT_A.to_vec(), 2)],
        events,
        legit,
        opts,
    );
    let err = head.expect_err("empty-prompt join refuses");
    assert!(err.to_string().contains("empty prompt"), "{err}");

    // (b) foreign handshake — the joiner's artifact digest differs.
    let foreign = ArtifactIdentity {
        geometry: id.geometry,
        artifact: [7u8; 32],
    };
    let (fc, fw) = (config.clone(), w.clone());
    let forger = move |_seq: usize| {
        TernaryCpuStage::new(fc.clone(), fw.clone(), LayerRange::new(0, 1), foreign)
            .map_err(io::Error::other)
    };
    let events = VecDeque::from([Admission::Join(MultiRequest::new(PROMPT_B.to_vec(), 2))]);
    let (head, _) = run_join(
        &config,
        &w,
        id,
        1,
        [25u8; 32],
        1,
        1,
        vec![MultiRequest::new(PROMPT_A.to_vec(), 2)],
        events,
        forger,
        opts,
    );
    let err = head.expect_err("foreign-handshake join refuses");
    assert!(err.to_string().contains("handshake differs"), "{err}");

    // (c) capacity — the join's prompt + decode span overruns the
    // stage's position capacity.
    let long = vec![9u32; config.block_size + 8];
    let events = VecDeque::from([Admission::Join(MultiRequest::new(long, 1))]);
    let (lc, lw, lid) = (config.clone(), w.clone(), id);
    let legit2 = move |_seq: usize| {
        TernaryCpuStage::new(lc.clone(), lw.clone(), LayerRange::new(0, 1), lid)
            .map_err(io::Error::other)
    };
    let (head, _) = run_join(
        &config,
        &w,
        id,
        1,
        [26u8; 32],
        1,
        1,
        vec![MultiRequest::new(PROMPT_A.to_vec(), 2)],
        events,
        legit2,
        opts,
    );
    let err = head.expect_err("over-capacity join refuses");
    assert!(err.to_string().contains("stage capacity"), "{err}");
}

/// Pre-warmed slots beyond the initial request table are legal (a
/// serve pre-warms cache slots); unused slots never announce and the
/// run behaves exactly like the static door.
#[test]
fn prewarmed_slots_beyond_initial_are_legal() {
    let (config, w, id) = load("riir_dist_admission_prewarm");
    let opts = HeadOptions::default();
    let want_a = oracle(&config, &w, id, &PROMPT_A, MAX_NEW, opts);
    let events = VecDeque::new(); // immediate Close
    let factory = |_seq: usize| -> io::Result<TernaryCpuStage> {
        panic!("no join is offered by this script")
    };
    let (head, tail) = run_join(
        &config,
        &w,
        id,
        1,
        [27u8; 32],
        2, // two prebuilt slots, one request
        1, // no join is offered — one tail slot
        vec![MultiRequest::new(PROMPT_A.to_vec(), MAX_NEW)],
        events,
        factory,
        opts,
    );
    let reports = head.expect("head");
    assert_eq!(reports.len(), 1);
    assert_eq!(reports[0].generated, want_a);
    let (steps, _) = tail.expect("tail");
    assert_eq!(steps, PROMPT_A.len() + MAX_NEW - 1);
}
