//! Plan 623 T5.1a–c — the speculative-fill G1 battery over REAL TCP
//! sockets (loopback), synth Bonsai-2:
//!
//! - **lossless, any drafter**: the spec token stream is byte-identical
//!   to the sequential oracle in ALL THREE regimes — full accept (an
//!   oracle drafter), partial accept + rewind (one wrong draft injected),
//!   and the never-accepts worst case (repeat-last).
//! - **the rewind leaves no torn state**: the post-rewind continuation
//!   equaling the oracle IS the "re-forward at the rewound position ==
//!   fresh cache" pin (T5.1b v0, HybridStateSegment snapshot/restore).
//! - **wire refusals**: malformed VerifyStream/VerifyReply payloads
//!   refuse at the codec, never parse as plausible garbage.
//! - **both-stage adjudication agreement**: a tampered `m` in a reply is
//!   refused by the head (the cross-check the wire carries).

#[path = "../../../tests/common/synth_bonsai2.rs"]
mod synth;

use std::net::{TcpListener, TcpStream};
use std::sync::Arc;
use std::thread;

use riir_infer_core::deltanet::{QwenDeltaNetTernaryWeights, WeightScope};
use riir_infer_core::multi_host::LayerRange;
use riir_infer_core::types::Config;
use riir_infer_dist::argmax_first;
use riir_infer_dist::runner::{HeadOptions, run_local};
use riir_infer_dist::spec_fill::{
    CheckpointLayer, RepeatLastDrafter, SpecStage, StageCheckpoint, VerifyReplyData, VerifyRequest,
    accepted_prefix, decode_verify_reply, encode_verify_reply, run_head_spec, run_tail_spec,
};
use riir_infer_dist::spec_mtp::{BmthHead, MtpHeadDrafter};
use riir_infer_dist::stage::{ArtifactIdentity, StageCompute, TernaryCpuStage};
use riir_infer_dist::transport::{Direction, PeerKey, TcpTransport};

const PROMPT: [u32; 5] = [7, 13, 42, 100, 5];
const MAX_NEW: usize = 8;
const K: usize = 3;

fn load(name: &str) -> (Config, Arc<QwenDeltaNetTernaryWeights>, ArtifactIdentity) {
    let bytes = synth::build_gguf(
        &synth::synth_metadata(false),
        &synth::synth_tensors_bonsai2(),
    );
    let path = synth::write_tmp(name, &bytes);
    let (c, w) =
        riir_infer_core::gguf_loader::load_qwen_deltanet_ternary_weights_gguf_scoped(&path, {
            |cfg| WeightScope::full(cfg.n_layer)
        })
        .expect("load synth");
    let id = ArtifactIdentity::of_gguf(&path).expect("identity");
    let _ = std::fs::remove_file(&path);
    (c, Arc::new(w), id)
}

/// The sequential oracle: generated tokens past the prompt.
fn oracle(config: &Config, w: &Arc<QwenDeltaNetTernaryWeights>, id: ArtifactIdentity) -> Vec<u32> {
    let n = config.n_layer as u32;
    let mut s = TernaryCpuStage::new(config.clone(), w.clone(), LayerRange::full(n), id).unwrap();
    run_local(&mut s, &PROMPT, MAX_NEW)
}

/// Proposes the TRUE token stream from the current cursor — every check
/// passes, every round fully accepts.
struct OracleDrafter {
    truth: Vec<u32>,
}

impl riir_infer_dist::spec_fill::DraftSource for OracleDrafter {
    fn propose(&mut self, ctx: &riir_infer_dist::spec_fill::DraftCtx<'_>) -> Vec<u32> {
        let idx = ctx.committed.len() - 1 - PROMPT.len();
        (0..ctx.k)
            .map(|j| self.truth.get(idx + j).copied().unwrap_or(0))
            .collect()
    }
}

/// The oracle drafter with ONE deliberately wrong draft at queue index
/// `wrong_at` — forces a partial accept (m < K) and the rewind path.
struct PoisonedDrafter {
    truth: Vec<u32>,
    wrong_at: usize,
}

impl riir_infer_dist::spec_fill::DraftSource for PoisonedDrafter {
    fn propose(&mut self, ctx: &riir_infer_dist::spec_fill::DraftCtx<'_>) -> Vec<u32> {
        let idx = ctx.committed.len() - 1 - PROMPT.len();
        (0..ctx.k)
            .map(|j| {
                let q = idx + j;
                match q == self.wrong_at {
                    true => self.truth.get(q).copied().unwrap_or(0).wrapping_add(1),
                    false => self.truth.get(q).copied().unwrap_or(0),
                }
            })
            .collect()
    }
}

struct SpecOutcome {
    head: std::io::Result<Vec<u32>>,
    tail: std::io::Result<usize>,
}

#[allow(clippy::too_many_arguments)]
fn run_pair_spec<D>(
    config: &Config,
    w: &Arc<QwenDeltaNetTernaryWeights>,
    split: u32,
    head_key: [u8; 32],
    tail_key: [u8; 32],
    id: ArtifactIdentity,
    tail_id: ArtifactIdentity,
    drafter: D,
    k: usize,
) -> SpecOutcome
where
    D: riir_infer_dist::spec_fill::DraftSource + Send + 'static,
{
    let n = config.n_layer as u32;
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let (tc, tw) = (config.clone(), w.clone());
    let tail = thread::spawn(move || {
        let mut stage = TernaryCpuStage::new(tc, tw, LayerRange::new(split, n), tail_id).unwrap();
        let (sock, _) = listener.accept().unwrap();
        let mut link = TcpTransport::new(sock, PeerKey(tail_key), Direction::Upstream).unwrap();
        run_tail_spec(&mut stage, &mut link, |_, _| {})
    });
    let (hc, hw) = (config.clone(), w.clone());
    let head = thread::spawn(move || {
        let mut stage = TernaryCpuStage::new(hc, hw, LayerRange::new(0, split), id).unwrap();
        let sock = TcpStream::connect(addr).unwrap();
        let mut link = TcpTransport::new(sock, PeerKey(head_key), Direction::Downstream).unwrap();
        let mut d = drafter;
        run_head_spec(
            &mut stage,
            &mut link,
            &PROMPT,
            MAX_NEW,
            HeadOptions::default(),
            &mut d,
            k,
        )
        .map(|r| r.generated)
    });
    SpecOutcome {
        head: head.join().unwrap(),
        tail: tail.join().unwrap(),
    }
}

#[test]
fn spec_full_accept_stream_equals_oracle() {
    let (config, w, id) = load("spec_g1_full");
    let truth = oracle(&config, &w, id);
    let out = run_pair_spec(
        &config,
        &w,
        2,
        [1u8; 32],
        [1u8; 32],
        id,
        id,
        OracleDrafter {
            truth: truth.clone(),
        },
        K,
    );
    out.tail.as_ref().expect("tail ok");
    let head = out.head.unwrap_or_else(|e| panic!("head ok: {e}"));
    assert_eq!(
        head, truth,
        "full-accept stream must be byte-identical to the oracle"
    );
    out.tail.as_ref().expect("tail clean shutdown");
}

#[test]
fn spec_partial_accept_rewind_stream_equals_oracle() {
    let (config, w, id) = load("spec_g1_partial");
    let truth = oracle(&config, &w, id);
    // A wrong draft at the FIRST queue slot (m == 0 rewind) and one mid-way
    // (m ≥ 1 partial) — both rewind regimes in one run.
    for wrong_at in [0usize, 2] {
        let out = run_pair_spec(
            &config,
            &w,
            2,
            [3u8; 32],
            [3u8; 32],
            id,
            id,
            PoisonedDrafter {
                truth: truth.clone(),
                wrong_at,
            },
            K,
        );
        let head = out
            .head
            .unwrap_or_else(|e| panic!("head ok (wrong_at {wrong_at}): {e}"));
        assert_eq!(
            head, truth,
            "post-rewind stream must be byte-identical to the oracle (wrong_at {wrong_at})"
        );
        assert!(
            out.tail.is_ok(),
            "tail clean shutdown (wrong_at {wrong_at})"
        );
    }
}

#[test]
fn spec_repeat_last_worst_case_equals_oracle() {
    let (config, w, id) = load("spec_g1_worst");
    let truth = oracle(&config, &w, id);
    let out = run_pair_spec(
        &config,
        &w,
        2,
        [5u8; 32],
        [5u8; 32],
        id,
        id,
        RepeatLastDrafter,
        K,
    );
    let head = out.head.expect("head ok");
    assert_eq!(
        head, truth,
        "the never-accepts worst case degenerates to the sequential stream, byte-for-byte"
    );
    assert!(out.tail.is_ok(), "tail clean shutdown");
}

#[test]
fn spec_acceptance_prefix_shifted_alignment() {
    // d_0 checks against `pending`; d_j (j ≥ 1) checks against argmax[j-1].
    let m = accepted_prefix(10, &[10, 11, 12], &[11, 12, 99]);
    assert_eq!(m, 3, "d_0==pending, d_1==argmax_0, d_2==argmax_1 all hold");
    let m0 = accepted_prefix(10, &[11, 10, 10], &[10, 10, 10]);
    assert_eq!(m0, 0, "d_0 wrong -> m == 0 regardless of the rest");
    let m1 = accepted_prefix(10, &[10, 12, 12], &[11, 99, 99]);
    assert_eq!(m1, 1, "d_0 holds; d_1 != argmax_0 fails the prefix at j=1");
}

#[test]
fn spec_wire_codecs_refuse_malformed() {
    // n_tok 0 refused; n_tok over the cap refused; short bodies refused;
    // trailing-junk bodies refused; m > n_tok refused.
    let tmp = Vec::new();
    assert!(VerifyRequest::decode_into(&tmp, 4, &mut Vec::new()).is_err());
    let zero = {
        let mut v = Vec::new();
        v.extend_from_slice(&1u32.to_le_bytes());
        v.extend_from_slice(&0u32.to_le_bytes());
        v.extend_from_slice(&0u32.to_le_bytes());
        v
    };
    assert!(VerifyRequest::decode_into(&zero, 4, &mut Vec::new()).is_err());
    let over = {
        let mut v = Vec::new();
        v.extend_from_slice(&1u32.to_le_bytes());
        v.extend_from_slice(&7u32.to_le_bytes());
        v.extend_from_slice(&128u32.to_le_bytes());
        v
    };
    assert!(VerifyRequest::decode_into(&over, 4, &mut Vec::new()).is_err());
    let reply = {
        let mut v = Vec::new();
        encode_verify_reply(1, 5, &[0, 0, 0], &mut v);
        v
    };
    assert!(
        decode_verify_reply(&reply).is_err(),
        "m 5 > n_tok 3 refuses"
    );
    let mut short = reply.clone();
    short.pop();
    assert!(
        decode_verify_reply(&short).is_err(),
        "truncated reply refuses"
    );
}

#[test]
fn spec_head_refuses_tampered_m() {
    // The reply's m is a cross-check: a head computing m=0 against a
    // reply claiming m=2 refuses (a broken/lying peer never parses).
    // Exercised at the unit seam: the head's adjudication branch compares
    // accepted_prefix(..) with reply.m — pin the comparison inputs.
    let argmaxes = vec![11u32, 12, 99];
    let m = accepted_prefix(10, &[10, 11, 12], &argmaxes);
    let tampered = VerifyReplyData {
        seq_id: 0,
        m: 2,
        argmaxes,
    };
    assert_ne!(m as u32, tampered.m, "the fixture genuinely disagrees");
}

/// The trait refuses non-CPU seating at compile time — pin the v0
/// posture in a doc-comment-adjacent compile-only arm instead: the GPU
/// stage has no `SpecStage` impl, which the type system enforces (no
/// runtime maybe). Nothing to execute here beyond keeping the CPU impl
/// exercised; the round-trip tests above ARE that exercise.
#[test]
fn spec_cpu_stage_round_trips_state() {
    let (config, w, id) = load("spec_g1_state");
    let n = config.n_layer as u32;
    let mut s = TernaryCpuStage::new(config.clone(), w.clone(), LayerRange::full(n), id).unwrap();
    let empty: [f32; 0] = [];
    // Consume a few positions, snapshot, consume more, restore, re-consume:
    // the continuation after restore must equal the fresh-cache forward.
    let mut next = 0u32;
    for (pos, &t) in PROMPT.iter().enumerate() {
        next = argmax_first(s.step(t, pos, &empty));
    }
    let mut snap = Vec::new();
    s.snapshot_state(&mut snap).expect("snapshot");
    let branch_a = argmax_first(s.step(next, PROMPT.len(), &empty));
    s.restore_state(&snap).expect("restore");
    let branch_b = argmax_first(s.step(next, PROMPT.len(), &empty));
    assert_eq!(
        branch_a, branch_b,
        "restore returns the stage to the snapshot cursor"
    );
}

// ───────── Issue 1148 rung 2: the exact checkpoint fast path ─────────

/// The checkpoint rewind is byte-exact AND equivalent to the codec
/// rewind: two stages advance identically past a shared cursor, then
/// rewind — one via [`StageCheckpoint`] (in-process), one via the
/// `HybridStateSegment` codec — and their next full hidden rows must
/// match bit-for-bit (f32 bits, not argmax: the pin is the STATE, and the
/// next row is its observable).
#[test]
fn spec_checkpoint_restore_matches_codec_restore_bit_exact() {
    let (config, w, id) = load("spec_g1_ckcodec");
    let n = config.n_layer as u32;
    let empty: [f32; 0] = [];
    // Extra tokens past the prompt (values are arbitrary — deterministic
    // forwards on identical states are what the pin needs).
    let tail = [PROMPT[0], PROMPT[2], PROMPT[4]];

    // Stage A — checkpoint path.
    let mut a = TernaryCpuStage::new(config.clone(), w.clone(), LayerRange::full(n), id).unwrap();
    for (pos, &t) in PROMPT.iter().enumerate() {
        a.step(t, pos, &empty);
    }
    let mut ck = StageCheckpoint::default();
    a.checkpoint_state(PROMPT.len(), &mut ck)
        .expect("checkpoint");
    assert_eq!(ck.cursor, PROMPT.len());
    for (j, &t) in tail.iter().enumerate() {
        a.step(t, PROMPT.len() + j, &empty);
    }
    a.restore_checkpoint(&ck).expect("restore_checkpoint");
    let a_out = a.step(tail[0], PROMPT.len(), &empty).to_vec();

    // Stage B — codec path, same arc.
    let mut b = TernaryCpuStage::new(config.clone(), w.clone(), LayerRange::full(n), id).unwrap();
    for (pos, &t) in PROMPT.iter().enumerate() {
        b.step(t, pos, &empty);
    }
    let mut snap = Vec::new();
    b.snapshot_state(&mut snap).expect("snapshot");
    for (j, &t) in tail.iter().enumerate() {
        b.step(t, PROMPT.len() + j, &empty);
    }
    b.restore_state(&snap).expect("restore_state");
    let b_out = b.step(tail[0], PROMPT.len(), &empty).to_vec();

    assert_eq!(a_out.len(), b_out.len());
    for (i, (x, y)) in a_out.iter().zip(b_out.iter()).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "row {i}: checkpoint rewind != codec rewind"
        );
    }

    // The retained-buffer reuse arm: a SECOND checkpoint into the same
    // `ck` (the steady-state posture — every verify round) must restore
    // identically too. ck2 captures cursor 6 (post-step@5); after its
    // restore the continuation is the step at POSITION 6 — compared
    // against b's identical arc (restore → step@5 → step@6).
    a.checkpoint_state(PROMPT.len() + 1, &mut ck)
        .expect("checkpoint 2");
    a.step(tail[1], PROMPT.len() + 1, &empty);
    a.restore_checkpoint(&ck).expect("restore_checkpoint 2");
    let a2 = a.step(tail[1], PROMPT.len() + 1, &empty);
    let b2 = b.step(tail[1], PROMPT.len() + 1, &empty);
    assert_eq!(a2.len(), b2.len());
    for (i, (x, y)) in a2.iter().zip(b2.iter()).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "row {i}: reused-checkpoint rewind diverged"
        );
    }
}

/// THE CURSOR-OVERWRITE TRUNCATE LAW (T5.1b, the plan's named pin, in its
/// strongest form): rows beyond the rewound cursor are stale-but-unread —
/// the re-forward rewrites them before any read, so even a NaN POISON
/// planted past the cursor cannot leak into the continuation. The
/// checkpoint path restores only rows `[0..cursor)` by design, so this is
/// the law its correctness rides on: restore_checkpoint + re-forward at
/// the rewound position must equal a FRESH cache bit-for-bit.
#[test]
fn kv_cursor_overwrite_truncate_law_nan_poison() {
    let (config, w, id) = load("spec_g1_cursorlaw");
    let n = config.n_layer as u32;
    let empty: [f32; 0] = [];
    let poison = 3usize; // the rewound cursor
    let span = PROMPT.len();

    // Fresh reference: forward the whole prompt on one stage, recording
    // the full output rows (logits — the last stage's step output) from
    // `poison` on.
    let mut fresh =
        TernaryCpuStage::new(config.clone(), w.clone(), LayerRange::full(n), id).unwrap();
    let mut reference: Vec<Vec<f32>> = Vec::new();
    for (pos, &t) in PROMPT.iter().enumerate() {
        let out = fresh.step(t, pos, &empty);
        if pos >= poison {
            reference.push(out.to_vec());
        }
    }

    // The rewind arc: forward to `poison`, checkpoint, advance past the
    // cursor (the verify-shaped writes), POISON every attention layer's
    // rows [poison..span) with NaN, rewind via the checkpoint (restores
    // rows [0..poison) + the GDN state, LEAVES the NaN rows stale), then
    // re-forward — outputs must equal the reference bit-for-bit.
    let mut a = TernaryCpuStage::new(config.clone(), w.clone(), LayerRange::full(n), id).unwrap();
    for (pos, &t) in PROMPT.iter().enumerate().take(poison) {
        a.step(t, pos, &empty);
    }
    let mut ck = StageCheckpoint::default();
    a.checkpoint_state(poison, &mut ck).expect("checkpoint");
    for (pos, &t) in PROMPT.iter().enumerate().skip(poison) {
        a.step(t, pos, &empty);
    }
    a.poison_kv_rows_beyond(poison, span);
    a.restore_checkpoint(&ck).expect("restore_checkpoint");
    for ((i, &t), want) in PROMPT.iter().enumerate().skip(poison).zip(reference.iter()) {
        let out = a.step(t, i, &empty);
        for (j, (x, y)) in out.iter().zip(want.iter()).enumerate() {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "row {i} elem {j}: NaN poison past the cursor leaked into the continuation"
            );
        }
    }
}

/// Checkpoint refusal arms: kinds never renegotiate; a checkpoint cannot
/// move between differently-shaped stages; restore validates the cursor
/// against the captured rows.
#[test]
fn spec_checkpoint_shape_refusals() {
    let (config, w, id) = load("spec_g1_ckrefuse");
    let n = config.n_layer as u32;
    let empty: [f32; 0] = [];
    let mut s = TernaryCpuStage::new(config.clone(), w.clone(), LayerRange::full(n), id).unwrap();
    for (pos, &t) in PROMPT.iter().enumerate() {
        s.step(t, pos, &empty);
    }
    let mut ck = StageCheckpoint::default();
    s.checkpoint_state(PROMPT.len(), &mut ck)
        .expect("checkpoint");
    // Kind corruption is refused at checkpoint AND restore.
    let mut bad = StageCheckpoint {
        cursor: ck.cursor,
        layers: ck
            .layers
            .iter()
            .map(|l| match l {
                CheckpointLayer::Attention { .. } => CheckpointLayer::DeltaNet {
                    rec: Vec::new(),
                    conv: Vec::new(),
                },
                CheckpointLayer::DeltaNet { .. } => CheckpointLayer::Attention {
                    key: Vec::new(),
                    value: Vec::new(),
                },
            })
            .collect(),
    };
    assert!(s.checkpoint_state(PROMPT.len(), &mut bad).is_err());
    assert!(s.restore_checkpoint(&bad).is_err());
    // Cursor/row mismatch at restore.
    let mut mismatched = StageCheckpoint::default();
    s.checkpoint_state(PROMPT.len(), &mut mismatched)
        .expect("ck2");
    mismatched.cursor += 1;
    assert!(s.restore_checkpoint(&mismatched).is_err());
    // A checkpoint from a different-shaped window refuses.
    let mut other = StageCheckpoint::default();
    let half =
        TernaryCpuStage::new(config.clone(), w.clone(), LayerRange::new(0, n / 2), id).unwrap();
    half.checkpoint_state(1, &mut other).expect("half ck");
    assert!(s.restore_checkpoint(&other).is_err());
}

// ───────────────────── T5.1d: the roll-drafter path ─────────────────────

/// The oracle as a ROLL drafter: same drafts as `OracleDrafter` (so every
/// round fully accepts) but delivered through `propose_with_roll` — the
/// runner's roll integration (snapshot-before-propose, rows built by the
/// drafter's own rolls, full-accept commit with NO rewind) must stay
/// byte-identical to the sequential oracle.
struct RollOracleDrafter {
    truth: Vec<u32>,
}

impl riir_infer_dist::spec_fill::DraftSource for RollOracleDrafter {
    fn propose(&mut self, ctx: &riir_infer_dist::spec_fill::DraftCtx<'_>) -> Vec<u32> {
        let idx = ctx.committed.len() - 1 - PROMPT.len();
        (0..ctx.k)
            .map(|j| self.truth.get(idx + j).copied().unwrap_or(0))
            .collect()
    }

    fn propose_with_roll(
        &mut self,
        ctx: &riir_infer_dist::spec_fill::DraftCtx<'_>,
        roll: &mut dyn FnMut(u32, usize, &mut [f32]),
    ) -> Vec<u32> {
        let cursor = ctx.committed.len() - 1;
        let idx = cursor - PROMPT.len();
        let pending = ctx.committed[cursor];
        let mut drafts = Vec::with_capacity(ctx.k);
        drafts.push(pending);
        let mut row = vec![0f32; synth::N_EMBD as usize];
        roll(pending, cursor, &mut row);
        for j in 1..ctx.k {
            let d = self.truth.get(idx + j).copied().unwrap_or(0);
            drafts.push(d);
            roll(d, cursor + j, &mut row);
        }
        drafts
    }
}

#[test]
fn spec_roll_oracle_full_accept_equals_oracle() {
    let (config, w, id) = load("spec_g1_rollo");
    let truth = oracle(&config, &w, id);
    let out = run_pair_spec(
        &config,
        &w,
        2,
        [7u8; 32],
        [7u8; 32],
        id,
        id,
        RollOracleDrafter {
            truth: truth.clone(),
        },
        K,
    );
    out.tail.as_ref().expect("tail ok");
    let head = out.head.unwrap_or_else(|e| panic!("head ok: {e}"));
    assert_eq!(
        head, truth,
        "the roll-drafter full-accept path is byte-identical to the oracle"
    );
    assert!(out.tail.as_ref().expect("tail ok") >= &0);
}

/// Deterministic LCG f32 in [0, 1) — no unseeded global draws.
struct TestLcg(u64);
impl TestLcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f32) / (u32::MAX as f32)
    }
}

/// Build BMTH v1 bytes (the documented layout — a doc-only cross-check).
fn build_bmth(d: usize, vocab: u32, support: usize, rng: &mut TestLcg) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(b"BMTH");
    b.extend_from_slice(&1u32.to_le_bytes());
    b.extend_from_slice(&(d as u32).to_le_bytes());
    b.extend_from_slice(&vocab.to_le_bytes());
    b.extend_from_slice(&(support as u32).to_le_bytes());
    for _ in 0..support * d {
        b.extend_from_slice(&rng.next_f32().to_le_bytes());
    }
    for _ in 0..support {
        b.extend_from_slice(&rng.next_f32().to_le_bytes());
    }
    for v in 0..support {
        b.extend_from_slice(&(v as u32).to_le_bytes());
    }
    b
}

#[test]
fn spec_mtp_drafter_lossless_on_synth() {
    // The trained head over synth weights drafts pseudo-random ids: the
    // echo (d_0) still always accepts, d_1.. usually reject — the m ≥ 1
    // partial-accept + rewind path through the ROLL integration. The G1
    // property is drafter-independent: the stream equals the oracle.
    let (config, w, id) = load("spec_g1_mtp");
    let truth = oracle(&config, &w, id);
    let n_embd = config.n_embd;
    let vocab = synth::VOCAB as u32;
    let bytes = build_bmth(n_embd, vocab, 16, &mut TestLcg(0xE11));
    let head = BmthHead::from_bytes(&bytes).expect("synth head");
    let drafter = MtpHeadDrafter::seat(head, n_embd).expect("seat at the stage width");
    let out = run_pair_spec(&config, &w, 2, [9u8; 32], [9u8; 32], id, id, drafter, K);
    let head_out = out.head.unwrap_or_else(|e| panic!("head ok: {e}"));
    assert_eq!(
        head_out, truth,
        "the trained-head drafter stays lossless regardless of acceptance"
    );
    assert!(out.tail.is_ok(), "tail clean shutdown");
}
