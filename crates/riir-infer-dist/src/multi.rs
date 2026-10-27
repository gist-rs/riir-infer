//! Plan 623 T2.4 — multi-stream round-robin over ONE stage link.
//!
//! The single-stream runner ([`crate::runner`]) serialises one request:
//! the head computes its step, sends, then blocks on the tail's token
//! while its stage sits idle — and the tail sits idle whenever the wire
//! does. With `s` streams round-robined over the same connection the head
//! always has a next compute while a reply is in flight and the tail
//! always has a queued frame: the pipeline bubble fills (per-token cost
//! approaches `max(head, tail + rtt)` instead of their sum), and a new
//! stream's prompt chunks interleave other streams' decode steps by
//! construction (exo's prefill-blocks-decode gap cannot form).
//!
//! Wire additions (additive — every single-stream frame is untouched):
//! - [`FrameKind::BeginStream`]: `seq_id u32 | prompt_len u32 |
//!   prefill_chunk u32` — the stream-addressed [`FrameKind::Begin`].
//! - [`FrameKind::TokenStream`]: `seq_id u32 | token u32` — the routed
//!   reply ([`FrameKind::Token`] carries no id to route by).
//!
//! Activation and prefill frames are ALREADY stream-addressed: the
//! Phase-1 `ActivationFrameHeader.seq_id` field (the single-stream
//! runner encodes it as 0). Stream id = index into the caller's stage
//! table; each stream owns ONE stage instance — the KV/GDN cache is per
//! stream, the weights stay `Arc`-shared — so the table length is the
//! admission cap and `s` streams cost `s` stage states per side.
//!
//! Scheduling: strict round-robin waves. Every pass sends at most one
//! work item per non-done stream (a prefill chunk or one decode step),
//! then the head drains exactly one reply. One reply-expected frame is
//! outstanding per stream; the tail is serial, so replies leave it
//! strictly in wire order, and the head verifies the FIFO route — a
//! reply naming any stream other than the oldest outstanding one
//! refuses loud. No send/send deadlock: the head can block sending a
//! large chunk only while the tail is still draining it, and the tail's
//! only sends are 8-byte replies, which never fill a socket buffer, with
//! a `recv` between them (the same invariant T2.3a rests on).
//!
//! Scope notes (honest residue): admission is the static request table —
//! a stream cannot JOIN mid-flight until a caller-facing admission seam
//! exists (the scheduler only ever asks "which non-done streams have
//! work", so a join is a table append at a pass boundary). The waves are
//! NOT kernel-level micro-batching: two streams' tokens still compute as
//! two `step` calls on the tail — batching them into one dispatch is a
//! forwarder rung (Phase 4 measured batch-1 decode launch-bound, so the
//! rung is real, and it is not this module's).

use std::collections::{HashMap, VecDeque};
use std::io;
use std::time::{Duration, Instant};

use riir_infer_core::multi_host::{ActivationFrame, FrameDtype};

use crate::argmax_first;
use crate::handshake;
use crate::runner::{HeadOptions, HeadReport, expect_kind, prefill_width, proto};
use crate::stage::StageCompute;
use crate::transport::{FrameKind, Transport};

/// One request of a multi-stream run.
#[derive(Debug, Clone)]
pub struct MultiRequest {
    /// Prompt token ids (non-empty).
    pub prompt: Vec<u32>,
    /// Greedy tokens to generate past the prompt (the tail samples).
    pub max_new: usize,
}

impl MultiRequest {
    pub fn new(prompt: Vec<u32>, max_new: usize) -> Self {
        Self { prompt, max_new }
    }
}

enum Phase {
    /// More prompt chunks to send; `prefill_lo` is the next chunk start.
    Prefill,
    /// One reply-expected frame is outstanding.
    Await,
    /// Ready to feed the last generated token as a decode step.
    Feed,
    /// Generated `max_new` (or hit EOS, or `max_new == 0`).
    Done,
}

struct StreamState {
    prompt: Vec<u32>,
    max_new: usize,
    chunk: usize,
    prefill_lo: usize,
    phase: Phase,
    generated: Vec<u32>,
    /// Wall clock at the first pass (after the handshake + Begins).
    t0: Instant,
    prefill: Option<Duration>,
    decode_t0: Option<Instant>,
    decode: Duration,
}

impl StreamState {
    fn report(&self) -> HeadReport {
        HeadReport {
            generated: self.generated.clone(),
            prefill: self.prefill.unwrap_or_default(),
            decode: self.decode,
        }
    }
}

/// One sendable unit of work, resolved per stream per pass.
enum Work {
    /// Prompt positions `lo..hi` (the head's own chunking).
    PrefillChunk { lo: usize, hi: usize },
    /// Feed the last generated token as one decode step.
    Feed { pos: usize, tok: u32 },
}

/// Head role over `s` streams: stage `i` serves request `i` (`seq_id` =
/// its index). Every head stage must present the SAME handshake (one
/// window per pipeline side); the tail's is verified once, as in the
/// single-stream runner.
///
/// The per-stream [`HeadReport`] walls overlap other streams' work by
/// design — `prefill` is wall-to-first-token of THAT stream, `decode`
/// first-to-last; the aggregate read is `total_generated / the caller's
/// own wall around this call`.
pub fn run_head_multi<S: StageCompute + ?Sized, T: Transport>(
    stages: &mut [&mut S],
    link: &mut T,
    reqs: &[MultiRequest],
    opts: HeadOptions,
) -> io::Result<Vec<HeadReport>> {
    if stages.len() != reqs.len() {
        return Err(proto(format!(
            "stream table {} != request count {}",
            stages.len(),
            reqs.len()
        )));
    }
    if stages.is_empty() {
        return Err(proto("no streams".into()));
    }
    let n_layer = stages[0].n_layer();
    for (i, s) in stages.iter().enumerate().skip(1) {
        if s.handshake() != stages[0].handshake() {
            return Err(proto(format!("stream {i} handshake differs from stream 0")));
        }
    }
    let mut buf = Vec::new();
    handshake::encode(stages[0].handshake(), &mut buf);
    link.send_frame(FrameKind::Handshake, &buf)?;
    let kind = link.recv_frame_into(&mut buf)?;
    expect_kind(kind, FrameKind::Handshake)?;
    let tail_hs = handshake::decode(&buf).map_err(proto)?;
    tail_hs
        .verify_pair(stages[0].handshake(), n_layer)
        .map_err(|e| proto(format!("tail handshake refused: {e}")))?;
    if !tail_hs.range.is_last(n_layer) {
        return Err(proto(
            "PoC runner is 2-stage: tail must own the last layer".into(),
        ));
    }

    let mut streams: Vec<StreamState> = reqs
        .iter()
        .enumerate()
        .map(|(i, r)| {
            if r.prompt.is_empty() {
                return Err(proto(format!("stream {i}: empty prompt")));
            }
            let needed = r.prompt.len() + r.max_new.saturating_sub(1);
            if needed > stages[i].max_positions() {
                return Err(proto(format!(
                    "stream {i} needs {needed} positions > stage capacity {}",
                    stages[i].max_positions()
                )));
            }
            let chunk = prefill_width(r.prompt.len(), opts);
            if chunk > u16::MAX as usize {
                return Err(proto(format!(
                    "stream {i}: prefill chunk {chunk} > frame n_tok cap"
                )));
            }
            Ok(StreamState {
                prompt: r.prompt.clone(),
                max_new: r.max_new,
                chunk,
                prefill_lo: 0,
                phase: Phase::Prefill,
                generated: Vec::with_capacity(r.max_new),
                t0: Instant::now(),
                prefill: None,
                decode_t0: None,
                decode: Duration::ZERO,
            })
        })
        .collect::<io::Result<_>>()?;

    // One BeginStream per stream up front, then the waves. (A stream that
    // joins later would send its BeginStream at its pass boundary — the
    // tail accepts one any time.)
    for (i, st) in streams.iter().enumerate() {
        let mut begin = [0u8; 12];
        begin[..4].copy_from_slice(&(i as u32).to_le_bytes());
        begin[4..8].copy_from_slice(&(st.prompt.len() as u32).to_le_bytes());
        begin[8..].copy_from_slice(&(st.chunk as u32).to_le_bytes());
        link.send_frame(FrameKind::BeginStream, &begin)?;
    }

    // Buffers sized to the WIDEST stream's chunk — no mid-run realloc
    // (the T2.1b posture).
    let hidden_dim = stages[0].hidden_dim();
    let max_chunk = streams.iter().map(|s| s.chunk).max().unwrap_or(1);
    let mut frame = Vec::with_capacity(20 + 4 * max_chunk * hidden_dim);
    let mut rows = Vec::new();
    let mut outstanding: VecDeque<usize> = VecDeque::new();
    let n = streams.len();
    let empty: [f32; 0] = [];

    while !streams.iter().all(|s| matches!(s.phase, Phase::Done)) {
        // One work item per stream, round-robin. The immutable view of
        // the work (`tokens`) cannot cross the state mutation — the
        // borrow ends at `send_frame`, the mutation follows it.
        for i in 0..n {
            let work = match streams[i].phase {
                Phase::Prefill => {
                    let lo = streams[i].prefill_lo;
                    let hi = (lo + streams[i].chunk).min(streams[i].prompt.len());
                    Work::PrefillChunk { lo, hi }
                }
                Phase::Feed => {
                    let pos = streams[i].prompt.len() + streams[i].generated.len() - 1;
                    let tok = *streams[i].generated.last().expect("Feed implies a token");
                    Work::Feed { pos, tok }
                }
                Phase::Await | Phase::Done => continue,
            };
            let (kind, chunked, tokens, pos, prompt_len, n_tok) = match &work {
                Work::PrefillChunk { lo, hi } => {
                    let last = *hi == streams[i].prompt.len();
                    (
                        match last {
                            true => FrameKind::Activation,
                            false => FrameKind::Prefill,
                        },
                        streams[i].chunk > 1,
                        &streams[i].prompt[*lo..*hi],
                        *lo,
                        streams[i].prompt.len(),
                        hi - lo,
                    )
                }
                // `prompt_len` is only read by the batched-prefill arm;
                // a decode step is always one position.
                Work::Feed { pos, tok } => (
                    FrameKind::Activation,
                    false,
                    std::slice::from_ref(tok),
                    *pos,
                    0,
                    1,
                ),
            };
            let hidden = match chunked {
                true => {
                    stages[i].prefill_chunk(tokens, &empty, pos, prompt_len, false, &mut rows);
                    rows.as_slice()
                }
                false => stages[i].step(tokens[0], pos, &empty),
            };
            frame.clear();
            ActivationFrame::encode_into(
                i as u32,
                pos as u32,
                n_tok as u16,
                hidden,
                FrameDtype::F32,
                &mut frame,
            );
            link.send_frame(kind, &frame)?;
            match work {
                Work::PrefillChunk { hi, .. } => {
                    if hi == streams[i].prompt.len() {
                        outstanding.push_back(i);
                        streams[i].phase = Phase::Await;
                    } else {
                        streams[i].prefill_lo = hi;
                    }
                }
                Work::Feed { .. } => {
                    outstanding.push_back(i);
                    streams[i].phase = Phase::Await;
                }
            }
        }
        // Drain exactly one reply, then re-rotate: a stream whose token
        // just landed feeds its next step while other streams' replies
        // are still in flight. Termination: a non-done stream is either
        // outstanding (this drain or a later one pops it) or has work
        // (the next pass sends it), and total work is finite.
        if let Some(expect) = outstanding.pop_front() {
            let kind = link.recv_frame_into(&mut buf)?;
            expect_kind(kind, FrameKind::TokenStream)?;
            let body: [u8; 8] = buf
                .as_slice()
                .try_into()
                .map_err(|_| proto(format!("token frame len {} != 8", buf.len())))?;
            let seq = u32::from_le_bytes(body[..4].try_into().unwrap());
            let tok = u32::from_le_bytes(body[4..].try_into().unwrap());
            if seq as usize != expect {
                return Err(proto(format!(
                    "token for stream {seq} while {expect} is oldest outstanding — reply order broken"
                )));
            }
            let s = &mut streams[expect];
            if s.prefill.is_none() {
                s.prefill = Some(s.t0.elapsed());
                s.decode_t0 = Some(Instant::now());
            }
            if s.max_new > 0 {
                s.generated.push(tok);
            }
            let finished = match s.max_new {
                0 => true,
                _ => s.generated.len() >= s.max_new || opts.eos == Some(tok),
            };
            if finished {
                s.decode = s.decode_t0.expect("first token set decode_t0").elapsed();
                s.phase = Phase::Done;
            } else {
                s.phase = Phase::Feed;
            }
        }
    }
    link.send_frame(FrameKind::Shutdown, &[])?;
    Ok(streams.iter().map(|s| s.report()).collect())
}

/// Tail role over `s` streams: connection-lifetime handshake, then any
/// interleaving of `BeginStream` / prefill / activation frames. Stream
/// `seq` computes on stage `seq`; per-stream position continuity is
/// enforced (`pos` must equal that stream's own next position — the
/// single-stream global ordering would refuse a legal interleave).
/// `on_logits(seq, pos, logits)` sees every reply-expected position and
/// every per-position decode step — the single-stream contract, tagged.
pub fn run_tail_multi<T: Transport>(
    stages: &mut [&mut dyn StageCompute],
    link: &mut T,
    mut on_logits: impl FnMut(u32, usize, &[f32]),
) -> io::Result<usize> {
    if stages.is_empty() {
        return Err(proto("no streams".into()));
    }
    let n_layer = stages[0].n_layer();
    let mut buf = Vec::new();
    let kind = link.recv_frame_into(&mut buf)?;
    expect_kind(kind, FrameKind::Handshake)?;
    let head_hs = handshake::decode(&buf).map_err(proto)?;
    stages[0]
        .handshake()
        .verify_pair(&head_hs, n_layer)
        .map_err(|e| proto(format!("head handshake refused: {e}")))?;
    if !head_hs.range.is_first() {
        return Err(proto("PoC runner is 2-stage: head must own layer 0".into()));
    }
    handshake::encode(stages[0].handshake(), &mut buf);
    link.send_frame(FrameKind::Handshake, &buf)?;

    struct Track {
        prompt_len: usize,
        chunk: usize,
        next_pos: usize,
    }
    let hidden_dim = stages[0].hidden_dim();
    let mut tracks: HashMap<u32, Track> = HashMap::new();
    let mut hidden = Vec::new();
    let mut out = Vec::new();
    let mut steps = 0usize;
    loop {
        let kind = link.recv_frame_into(&mut buf)?;
        match kind {
            FrameKind::Shutdown => return Ok(steps),
            FrameKind::BeginStream => {
                let body: [u8; 12] = buf
                    .as_slice()
                    .try_into()
                    .map_err(|_| proto(format!("begin-stream frame len {} != 12", buf.len())))?;
                let seq = u32::from_le_bytes(body[..4].try_into().unwrap());
                let prompt_len = u32::from_le_bytes(body[4..8].try_into().unwrap()) as usize;
                let chunk = u32::from_le_bytes(body[8..].try_into().unwrap()) as usize;
                if seq as usize >= stages.len() {
                    return Err(proto(format!(
                        "BeginStream seq {seq} >= {} stages",
                        stages.len()
                    )));
                }
                if prompt_len == 0 || chunk == 0 {
                    return Err(proto(format!(
                        "BeginStream seq {seq}: prompt_len {prompt_len}, chunk {chunk}"
                    )));
                }
                if tracks.contains_key(&seq) {
                    return Err(proto(format!("BeginStream seq {seq} already open")));
                }
                tracks.insert(
                    seq,
                    Track {
                        prompt_len,
                        chunk,
                        next_pos: 0,
                    },
                );
                // T2.1b posture: reserve to this stream's frame bound so a
                // later chunked frame never reallocs mid-stream.
                buf.reserve(crate::runner::frame_bound(chunk, hidden_dim));
                hidden.reserve(chunk.saturating_mul(hidden_dim));
            }
            FrameKind::Prefill | FrameKind::Activation => {
                let reply = kind == FrameKind::Activation;
                let hdr = ActivationFrame::decode_into(&buf, &mut hidden).map_err(proto)?;
                let seq = hdr.seq_id;
                let pos = hdr.pos as usize;
                let n_tok = hdr.n_tok as usize;
                let Some(track) = tracks.get_mut(&seq) else {
                    return Err(proto(format!(
                        "activation for stream {seq} with no BeginStream"
                    )));
                };
                let stage: &mut dyn StageCompute = &mut *stages[seq as usize];
                let stage_hidden = stage.hidden_dim();
                let chunked = track.chunk > 1 && pos < track.prompt_len;
                let want_tok = match chunked {
                    true => track.chunk.min(track.prompt_len - pos),
                    false => 1,
                };
                if n_tok != want_tok || hidden.len() != n_tok * stage_hidden {
                    return Err(proto(format!(
                        "stream {seq}: activation shape {n_tok}×{} != {want_tok}×{stage_hidden} at pos {pos}",
                        hidden.len() / n_tok.max(1)
                    )));
                }
                if pos != track.next_pos {
                    return Err(proto(format!(
                        "stream {seq}: activation pos {pos} != expected {}",
                        track.next_pos
                    )));
                }
                if pos + n_tok > stage.max_positions() {
                    return Err(proto(format!(
                        "stream {seq}: activation span {pos}+{n_tok} > stage capacity {}",
                        stage.max_positions()
                    )));
                }
                let logits = match chunked {
                    true => {
                        stage.prefill_chunk(&[], &hidden, pos, track.prompt_len, reply, &mut out);
                        out.as_slice()
                    }
                    false => stage.step(0, pos, &hidden),
                };
                if reply {
                    let last = pos + n_tok - 1;
                    on_logits(seq, last, logits);
                    let id = argmax_first(logits);
                    let mut reply_buf = [0u8; 8];
                    reply_buf[..4].copy_from_slice(&seq.to_le_bytes());
                    reply_buf[4..].copy_from_slice(&id.to_le_bytes());
                    link.send_frame(FrameKind::TokenStream, &reply_buf)?;
                } else if !chunked {
                    on_logits(seq, pos, logits);
                }
                track.next_pos += n_tok;
                steps += n_tok;
            }
            k => return Err(proto(format!("unexpected {k:?} frame mid-stream"))),
        }
    }
}
