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
//! Scope notes (honest residue): kernel-level micro-batching is NOT
//! here — two streams' tokens still compute as two `step` calls on the
//! tail (batching them into one dispatch is a forwarder rung — Phase 4
//! measured batch-1 decode launch-bound, so the rung is real, and it is
//! not this module's). Admission: the static door ([`run_head_multi`])
//! pins the request table; the joining door ([`run_head_multi_admit`],
//! the T2.4 admission seam) appends factory-built stages at pass
//! boundaries — the tail side needs no change (its loop already accepts
//! a `BeginStream` any time and its static table IS the tail's
//! admission cap).

use std::collections::{HashMap, VecDeque};
use std::io;
use std::time::{Duration, Instant};

use riir_infer_core::multi_host::{ActivationFrame, FrameDtype};

use crate::argmax_first;
use crate::handshake;
use crate::runner::{HeadOptions, HeadReport, expect_kind, prefill_width, proto};
use crate::stage::StageCompute;
use crate::transport::{FrameKind, Transport};

/// T3.5 scoreboard instrument (`RIIR_DIST_STEP_TIMING=1`): per-phase decode-wall
/// decomposition on each role — head {submit, finish, send, recv}, tail {recv,
/// fwd, pfwd, send} — accumulated over a 64-step window and flushed as one stderr
/// line per window (plus a forced final flush). Disarmed the only cost is one
/// `bool` load per site; `Instant::now` runs only when armed. The buckets are
/// the honest adjudication surface for the two T3.4 residues (comm-THREAD split,
/// tail logits async readback): whichever leg the scoreboard shows binding is the
/// one a follow-up arm may target.
struct StepTiming {
    armed: bool,
    t0: Instant,
    submit_ns: u64,
    finish_ns: u64,
    send_ns: u64,
    recv_ns: u64,
    fwd_ns: u64,
    pfwd_ns: u64,
    steps: u64,
}

/// T3.4 async-finish arm, PROMOTE-ONLY since the Bench-929 wire refutation
/// (demote was the old default; the ABBA cells measured −48.5% decode window
/// armed at s=2 — the drain serializes the pass, forfeiting the head⇄tail
/// pipeline). Exact-literal arm; unset or any other value = the incumbent
/// inline cadence, byte-identical to the pre-T3.4 posture.
fn async_finish_armed() -> bool {
    static ARMED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ARMED.get_or_init(|| {
        matches!(
            std::env::var("RIIR_DIST_ASYNC_FINISH").as_deref(),
            Ok("1") | Ok("true") | Ok("on")
        )
    })
}

impl StepTiming {
    fn new() -> Self {
        static ARMED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let armed = *ARMED.get_or_init(|| {
            matches!(
                std::env::var("RIIR_DIST_STEP_TIMING").as_deref(),
                Ok("1") | Ok("true") | Ok("on")
            )
        });
        Self {
            armed,
            t0: Instant::now(),
            submit_ns: 0,
            finish_ns: 0,
            send_ns: 0,
            recv_ns: 0,
            fwd_ns: 0,
            pfwd_ns: 0,
            steps: 0,
        }
    }

    fn flush(&self, role: &str, force: bool) {
        if !self.armed || self.steps == 0 || (!force && !self.steps.is_multiple_of(64)) {
            return;
        }
        let n = self.steps as f64;
        let ms = |ns: u64| ns as f64 / 1e6;
        let per = |ns: u64| ms(ns) / n;
        eprintln!(
            "step-timing[{role}]: n={} wall={:.1}ms submit={:.1} finish={:.1} send={:.1} recv={:.1} fwd={:.1} pfwd={:.1} | per-step ms: submit {:.3} finish {:.3} send {:.3} recv {:.3} fwd {:.3}",
            self.steps,
            ms(self.t0.elapsed().as_nanos() as u64),
            ms(self.submit_ns),
            ms(self.finish_ns),
            ms(self.send_ns),
            ms(self.recv_ns),
            ms(self.fwd_ns),
            ms(self.pfwd_ns),
            per(self.submit_ns),
            per(self.finish_ns),
            per(self.send_ns),
            per(self.recv_ns),
            per(self.fwd_ns),
        );
    }
}

#[inline]
fn t_now(armed: bool) -> Option<Instant> {
    armed.then(Instant::now)
}

#[inline]
fn t_add(t: Option<Instant>, bucket: &mut u64) {
    if let Some(t0) = t {
        *bucket += t0.elapsed().as_nanos() as u64;
    }
}

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

/// An admission event, polled by the joining runner at pass boundaries
/// (T2.4 admission seam).
#[derive(Debug, Clone)]
pub enum Admission {
    /// Admit this request as the next stream (`seq_id` = the current
    /// table length); the stage it runs on is built by the runner's
    /// [`StageFactory`]. A malformed join (empty prompt, over-capacity,
    /// handshake mismatch) refuses the RUN loud — validate before
    /// offering; a serve front-end owns that gate.
    Join(MultiRequest),
    /// Nothing to admit right now; keep the pipeline running.
    Wait,
    /// No further joins will ever arrive: finish the outstanding
    /// streams and return.
    Close,
}

/// The admission source. Polled at the top of every scheduling pass;
/// while the table is IDLE (every stream done, [`Admission::Close`]
/// not yet returned) an implementation should BLOCK in [`Self::poll`]
/// for its next event — the engine does not sleep for it.
pub trait AdmissionSource {
    fn poll(&mut self) -> io::Result<Admission>;
}

/// The no-join source: the static door's admission (`Close` at once).
#[derive(Debug, Default, Clone, Copy)]
pub struct ClosedAdmission;

impl AdmissionSource for ClosedAdmission {
    fn poll(&mut self) -> io::Result<Admission> {
        Ok(Admission::Close)
    }
}

/// Builds the stage instance a joining stream runs on (the runner owns
/// the stage table). Weights stay `Arc`-shared; a build allocates the
/// per-stream cache/scratch only — keep it that way, a join must not
/// re-read the artifact.
pub trait StageFactory<S: StageCompute> {
    /// `seq` is the joining stream's id (its table index).
    fn build(&mut self, seq: usize) -> io::Result<S>;
}

impl<S: StageCompute, F: FnMut(usize) -> io::Result<S>> StageFactory<S> for F {
    fn build(&mut self, seq: usize) -> io::Result<S> {
        self(seq)
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
            spec_rounds: 0,
            spec_accepted: 0,
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

/// The stage table as the wave engine sees it: index-addressed compute
/// plus the admit half of the admission seam. The static door wraps the
/// caller's borrowed slice; the joining door owns a pool and builds
/// joiners through a [`StageFactory`].
trait WaveStages {
    type Stage: ?Sized + StageCompute;
    fn table_len(&self) -> usize;
    fn stage(&mut self, i: usize) -> &mut Self::Stage;
    /// Build + append the joining stream's stage (its index is the
    /// pre-append length). Validates the handshake against table 0 and
    /// the position capacity; the engine has already capped the chunk.
    fn admit(&mut self, req: &MultiRequest, chunk: usize) -> io::Result<()>;
}

struct BorrowedTable<'a, 'b, S: ?Sized>(&'a mut [&'b mut S]);

impl<S: ?Sized + StageCompute> WaveStages for BorrowedTable<'_, '_, S> {
    type Stage = S;
    fn table_len(&self) -> usize {
        self.0.len()
    }
    fn stage(&mut self, i: usize) -> &mut S {
        &mut *self.0[i]
    }
    fn admit(&mut self, _req: &MultiRequest, _chunk: usize) -> io::Result<()> {
        Err(proto(
            "static stream table cannot admit mid-flight — use run_head_multi_admit".into(),
        ))
    }
}

struct PoolTable<'p, 'f, S: StageCompute, F: StageFactory<S>> {
    pool: &'p mut Vec<S>,
    factory: &'f mut F,
}

impl<S: StageCompute, F: StageFactory<S>> WaveStages for PoolTable<'_, '_, S, F> {
    type Stage = S;
    fn table_len(&self) -> usize {
        self.pool.len()
    }
    fn stage(&mut self, i: usize) -> &mut S {
        &mut self.pool[i]
    }
    fn admit(&mut self, req: &MultiRequest, _chunk: usize) -> io::Result<()> {
        let idx = self.pool.len();
        let stage = self.factory.build(idx)?;
        let needed = req.prompt.len() + req.max_new.saturating_sub(1);
        if needed > stage.max_positions() {
            return Err(proto(format!(
                "stream {idx} needs {needed} positions > stage capacity {}",
                stage.max_positions()
            )));
        }
        if *stage.handshake() != *self.pool[0].handshake() {
            return Err(proto(format!(
                "stream {idx} handshake differs from stream 0"
            )));
        }
        self.pool.push(stage);
        Ok(())
    }
}

/// Head role over `s` streams (the STATIC door): stage `i` serves
/// request `i` (`seq_id` = its index); the request table is fixed — no
/// mid-flight join. Every head stage must present the SAME handshake
/// (one window per pipeline side); the tail's is verified once, as in
/// the single-stream runner.
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
    run_waves(
        &mut BorrowedTable(stages),
        link,
        reqs,
        &mut ClosedAdmission,
        opts,
    )
}

/// Head role with the T2.4 admission seam: an [`Admission::Join`]
/// appends a factory-built stage at a pass boundary (the joining
/// stream's `BeginStream` rides the same boundary — the tail accepts
/// one any time; its static table is the TAIL's admission cap). `pool`
/// owns one stage per stream; pre-warmed slots beyond `initial` are
/// legal (a join takes `pool.push(factory.build(seq))`). The call
/// returns once admission closed AND every stream finished — an
/// [`AdmissionSource`] polled while the table is idle must block; the
/// engine never sleeps for it. Same report semantics as
/// [`run_head_multi`]; the vec is in stream order (joiners appended).
pub fn run_head_multi_admit<S: StageCompute, T: Transport, F: StageFactory<S>>(
    pool: &mut Vec<S>,
    link: &mut T,
    initial: &[MultiRequest],
    admission: &mut dyn AdmissionSource,
    factory: &mut F,
    opts: HeadOptions,
) -> io::Result<Vec<HeadReport>> {
    if pool.is_empty() {
        return Err(proto("no streams".into()));
    }
    if pool.len() < initial.len() {
        return Err(proto(format!(
            "stage pool {} < initial request count {}",
            pool.len(),
            initial.len()
        )));
    }
    let mut table = PoolTable { pool, factory };
    run_waves(&mut table, link, initial, admission, opts)
}

fn run_waves<T: WaveStages, L: Transport>(
    table: &mut T,
    link: &mut L,
    reqs: &[MultiRequest],
    admission: &mut dyn AdmissionSource,
    opts: HeadOptions,
) -> io::Result<Vec<HeadReport>> {
    if table.table_len() < reqs.len() {
        return Err(proto(format!(
            "stream table {} < request count {}",
            table.table_len(),
            reqs.len()
        )));
    }
    let n_layer = table.stage(0).n_layer();
    let hs0 = table.stage(0).handshake().clone();
    for i in 1..table.table_len() {
        if *table.stage(i).handshake() != hs0 {
            return Err(proto(format!("stream {i} handshake differs from stream 0")));
        }
    }
    let mut buf = Vec::new();
    handshake::encode(&hs0, &mut buf);
    link.send_frame(FrameKind::Handshake, &buf)?;
    let kind = link.recv_frame_into(&mut buf)?;
    expect_kind(kind, FrameKind::Handshake)?;
    let tail_hs = handshake::decode(&buf).map_err(proto)?;
    tail_hs
        .verify_pair(&hs0, n_layer)
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
            if needed > table.stage(i).max_positions() {
                return Err(proto(format!(
                    "stream {i} needs {needed} positions > stage capacity {}",
                    table.stage(i).max_positions()
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

    // One BeginStream per stream up front, then the waves. (A stream
    // that joins later sends its BeginStream at its pass boundary —
    // the tail accepts one any time.)
    for (i, st) in streams.iter().enumerate() {
        let mut begin = [0u8; 12];
        begin[..4].copy_from_slice(&(i as u32).to_le_bytes());
        begin[4..8].copy_from_slice(&(st.prompt.len() as u32).to_le_bytes());
        begin[8..].copy_from_slice(&(st.chunk as u32).to_le_bytes());
        link.send_frame(FrameKind::BeginStream, &begin)?;
    }

    // Buffers sized to the WIDEST stream's chunk — no mid-run realloc
    // (the T2.1b posture).
    let hidden_dim = table.stage(0).hidden_dim();
    let max_chunk = streams.iter().map(|s| s.chunk).max().unwrap_or(1);
    let mut frame = Vec::with_capacity(20 + 4 * max_chunk * hidden_dim);
    let mut rows = Vec::new();
    let mut outstanding: VecDeque<usize> = VecDeque::new();
    let empty: [f32; 0] = [];
    let mut st = StepTiming::new();
    let mut closed = false;
    loop {
        // The T2.4 admission seam: joins are table appends at pass
        // boundaries. The Admitter drains until Wait/Close; while the
        // table is idle it must block in `poll` (the engine does not
        // sleep for it).
        while !closed {
            match admission.poll()? {
                Admission::Close => closed = true,
                Admission::Wait => break,
                Admission::Join(req) => {
                    admit_stream(table, link, &mut streams, req, opts, &mut frame, &mut rows)?
                }
            }
        }
        if closed && streams.iter().all(|s| matches!(s.phase, Phase::Done)) {
            break;
        }
        // T2.4 fused batch step: when EVERY ready stream is a decode feed,
        // hint the tail (BatchN) so it submits all the chains BEFORE any
        // readback — the shared queue runs them back to back with no
        // inter-chain idle. OPT-IN (Bench 913): on a 2-box pipeline the
        // wave forfeits the head⇄tail GPU overlap (the s=2 bubble fill —
        // measured −52% on the wire); the gain is a co-located-queue one.
        // Prefill-bearing passes and solo passes ride the ordinary frames
        // byte-identically.
        let n = streams.len();
        let ready_ct = streams
            .iter()
            .filter(|s| matches!(s.phase, Phase::Prefill | Phase::Feed))
            .count();
        let all_feed = ready_ct > 1
            && streams
                .iter()
                .filter(|s| matches!(s.phase, Phase::Feed))
                .count()
                == ready_ct;
        if all_feed && opts.pair_decode {
            link.send_frame(FrameKind::BatchN, &(ready_ct as u32).to_le_bytes())?;
        }
        // Indexed on purpose: the body interleaves immutable prompt
        // slices with `phase` mutation on `streams` AND hands `i` to
        // `table.stage(i)` — an iterator would fight the borrows.
        // decode_enqueued: the T3.4 arm — streams whose chain is submitted
        // + signal armed this pass, finished/sent in the drain below.
        let mut decode_enqueued: Vec<usize> = Vec::new();
        #[allow(clippy::needless_range_loop)]
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
            // Plan-623 T3.4: decode feeds defer to the T3.4 drain ONLY on
            // an all-feed pass — the decode steady state, where the win
            // lives (stream i's readback+send overlaps stream i+1's chain
            // on the shared queue; send order stays index order, so the
            // wire order is byte-identical to the incumbent). Mixed passes
            // (a prefill chunk still working) and s=1 keep the incumbent
            // inline cadence exactly — the tail capture-order pin is the
            // contract. The demote posture and the CPU stage reduce to the
            // incumbent cadence too (schedule is a no-op; the drain's
            // step_finish then reads synchronously).
            if let Work::Feed { pos, tok } = work {
                // Wire-refuted default (Bench 929 Row 2, −48.5% decode window
                // ABBA ×2 at s=2): the deferred drain serializes each pass —
                // both head chains submit before any activation leaves, so
                // the two GPUs never overlap. PROMOTE-ONLY now: the env must
                // be the exact literal 1/true/on (the loopback rig keeps its
                // +5% single-queue class; the wire serving posture is the
                // incumbent inline cadence). The demoted path is the
                // incumbent by construction (schedule is a no-op; step_finish
                // reads synchronously), so this is byte-identical to the
                // pre-T3.4 serving posture.
                if all_feed && async_finish_armed() {
                    let t = t_now(st.armed);
                    table.stage(i).step_submit(tok, pos, &empty);
                    t_add(t, &mut st.submit_ns);
                    table.stage(i).schedule_async_finish();
                    decode_enqueued.push(i);
                    continue;
                }
                // Mixed pass: the incumbent inline cadence.
                let hidden = table.stage(i).step(tok, pos, &empty).to_vec();
                frame.clear();
                ActivationFrame::encode_into(
                    i as u32,
                    pos as u32,
                    1,
                    &hidden,
                    FrameDtype::F32,
                    &mut frame,
                );
                link.send_frame(FrameKind::Activation, &frame)?;
                outstanding.push_back(i);
                streams[i].phase = Phase::Await;
                continue;
            }
            let Work::PrefillChunk { lo, hi } = work else {
                unreachable!("the Feed arm continues above")
            };
            let last = hi == streams[i].prompt.len();
            let kind = match last {
                true => FrameKind::Activation,
                false => FrameKind::Prefill,
            };
            let prompt_len = streams[i].prompt.len();
            let pos = lo;
            let n_tok = hi - lo;
            let tokens: &[u32] = &streams[i].prompt[lo..hi];
            let hidden = if streams[i].chunk > 1 {
                table
                    .stage(i)
                    .prefill_chunk(tokens, &empty, pos, prompt_len, false, &mut rows);
                rows.as_slice()
            } else {
                // Single-position prefill chunk rides the same
                // submit+finish pair — the composition law makes it the
                // exact `step` the incumbent inline path ran.
                table.stage(i).step_submit(tokens[0], pos, &empty);
                table.stage(i).step_finish()
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
            if last {
                outstanding.push_back(i);
                streams[i].phase = Phase::Await;
            } else {
                streams[i].prefill_lo = hi;
            }
        }
        // The T3.4 drain: finish each enqueued decode stream in enqueue
        // order (the future resolves = the chain-done signal), then encode
        // + send. With the async arm DISARMED this is still correct —
        // step_finish takes the synchronous read, and the send order
        // matches the old inline cadence's outstanding order (prefill
        // frames sent above, decode frames now, indexed ascending).
        for &i in &decode_enqueued {
            let pos = streams[i].prompt.len() + streams[i].generated.len() - 1;
            let t = t_now(st.armed);
            let hidden = table.stage(i).step_finish().to_vec();
            t_add(t, &mut st.finish_ns);
            let t = t_now(st.armed);
            frame.clear();
            ActivationFrame::encode_into(
                i as u32,
                pos as u32,
                1,
                &hidden,
                FrameDtype::F32,
                &mut frame,
            );
            link.send_frame(FrameKind::Activation, &frame)?;
            t_add(t, &mut st.send_ns);
            outstanding.push_back(i);
            streams[i].phase = Phase::Await;
        }
        drop(decode_enqueued);
        // Drain the pass's replies, then re-rotate: a stream whose token
        // just landed feeds its next step while other streams' replies
        // are still in flight. The fused batch drains its WHOLE pass (the
        // tail already ran the chains back to back); the incumbent
        // single-drain cadence (the default) pipelines the two GPUs — the
        // head computes stream B's head-half under the tail's stream-A
        // chain, which IS the 2-box win (Bench 913).
        // Termination: a non-done stream is either outstanding (this drain
        // or a later one pops it) or has work (the next pass sends it),
        // and total work is finite.
        let max_drain = if opts.pair_decode { usize::MAX } else { 1 };
        for _ in 0..max_drain {
            let Some(expect) = outstanding.pop_front() else {
                break;
            };
            let t = t_now(st.armed);
            let kind = link.recv_frame_into(&mut buf)?;
            t_add(t, &mut st.recv_ns);
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
            st.steps += 1;
            st.flush("head", false);
        }
    }
    st.flush("head", true);
    link.send_frame(FrameKind::Shutdown, &[])?;
    Ok(streams.iter().map(|s| s.report()).collect())
}

/// One [`Admission::Join`] at a pass boundary: validate the request,
/// build + append the stage (table side), extend the frame reservations
/// to the joiner's bound (the T2.1b posture, joined edition — a later
/// chunked frame never reallocs mid-stream), announce the stream
/// (`BeginStream`), open its state. Refusing here fails the run loud —
/// the caller's admission front-end validates first.
#[allow(clippy::too_many_arguments)]
fn admit_stream<T: WaveStages>(
    table: &mut T,
    link: &mut impl Transport,
    streams: &mut Vec<StreamState>,
    req: MultiRequest,
    opts: HeadOptions,
    frame: &mut Vec<u8>,
    rows: &mut Vec<f32>,
) -> io::Result<()> {
    if req.prompt.is_empty() {
        return Err(proto("joining stream: empty prompt".into()));
    }
    let chunk = prefill_width(req.prompt.len(), opts);
    if chunk > u16::MAX as usize {
        return Err(proto(format!(
            "joining stream: prefill chunk {chunk} > frame n_tok cap"
        )));
    }
    let idx = streams.len();
    table.admit(&req, chunk)?;
    let hidden_dim = table.stage(0).hidden_dim();
    frame.reserve(crate::runner::frame_bound(chunk, hidden_dim));
    rows.reserve(chunk.saturating_mul(hidden_dim));
    let mut begin = [0u8; 12];
    begin[..4].copy_from_slice(&(idx as u32).to_le_bytes());
    begin[4..8].copy_from_slice(&(req.prompt.len() as u32).to_le_bytes());
    begin[8..].copy_from_slice(&(chunk as u32).to_le_bytes());
    link.send_frame(FrameKind::BeginStream, &begin)?;
    streams.push(StreamState {
        prompt: req.prompt,
        max_new: req.max_new,
        chunk,
        prefill_lo: 0,
        phase: Phase::Prefill,
        generated: Vec::with_capacity(req.max_new),
        t0: Instant::now(),
        prefill: None,
        decode_t0: None,
        decode: Duration::ZERO,
    });
    Ok(())
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
    // T2.4 fused batch: the hinted batch's (seq, pos) rows and their
    // flattened hidden rows — preallocated here, reused every batch.
    let mut batch_rows: Vec<(u32, usize)> = Vec::new();
    let mut batch_hidden: Vec<f32> = Vec::new();
    let mut batch_seen: Vec<bool> = vec![false; stages.len()];
    let mut st = StepTiming::new();
    loop {
        let t = t_now(st.armed);
        let kind = link.recv_frame_into(&mut buf)?;
        t_add(t, &mut st.recv_ns);
        match kind {
            FrameKind::Shutdown => {
                st.flush("tail", true);
                return Ok(steps);
            }
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
                        let t = t_now(st.armed);
                        stage.prefill_chunk(&[], &hidden, pos, track.prompt_len, reply, &mut out);
                        t_add(t, &mut st.pfwd_ns);
                        out.as_slice()
                    }
                    false => {
                        let t = t_now(st.armed);
                        let l = stage.step(0, pos, &hidden);
                        t_add(t, &mut st.fwd_ns);
                        l
                    }
                };
                if reply {
                    let last = pos + n_tok - 1;
                    on_logits(seq, last, logits);
                    let id = argmax_first(logits);
                    let mut reply_buf = [0u8; 8];
                    reply_buf[..4].copy_from_slice(&seq.to_le_bytes());
                    reply_buf[4..].copy_from_slice(&id.to_le_bytes());
                    let t = t_now(st.armed);
                    link.send_frame(FrameKind::TokenStream, &reply_buf)?;
                    t_add(t, &mut st.send_ns);
                    st.steps += 1;
                    st.flush("tail", false);
                } else if !chunked {
                    on_logits(seq, pos, logits);
                }
                track.next_pos += n_tok;
                steps += n_tok;
            }
            FrameKind::BatchN => {
                // T2.4 fused batch step: `count u32 LE`, then `count`
                // one-token Activation frames of DISTINCT streams. All
                // rows validate BEFORE any chain runs, then every chain
                // submits, then each finishes in FIFO order (the head's
                // outstanding order) — the shared queue runs the chains
                // back to back with no inter-chain idle.
                let count = u32::from_le_bytes(
                    buf.as_slice()
                        .try_into()
                        .map_err(|_| proto(format!("batch-n frame len {} != 4", buf.len())))?,
                ) as usize;
                if count == 0 || count > stages.len() {
                    return Err(proto(format!(
                        "BatchN count {count} outside 1..={}",
                        stages.len()
                    )));
                }
                batch_rows.clear();
                batch_hidden.clear();
                batch_seen.iter_mut().for_each(|v| *v = false);
                for row in 0..count {
                    let kind = link.recv_frame_into(&mut buf)?;
                    expect_kind(kind, FrameKind::Activation)?;
                    let hdr = ActivationFrame::decode_into(&buf, &mut hidden).map_err(proto)?;
                    let seq = hdr.seq_id;
                    let pos = hdr.pos as usize;
                    if hdr.n_tok != 1 || hidden.len() != hidden_dim {
                        return Err(proto(format!(
                            "batch row {row}: n_tok {} / {} floats != 1×{hidden_dim} — the fused batch carries decode feeds only",
                            hdr.n_tok,
                            hidden.len()
                        )));
                    }
                    if seq as usize >= stages.len() {
                        return Err(proto(format!(
                            "batch row {row}: seq {seq} >= {}",
                            stages.len()
                        )));
                    }
                    if stages[seq as usize].hidden_dim() != hidden_dim {
                        return Err(proto(format!(
                            "batch row {row}: stream {seq} hidden {} != table {hidden_dim}",
                            stages[seq as usize].hidden_dim()
                        )));
                    }
                    if batch_seen[seq as usize] {
                        return Err(proto(format!(
                            "batch row {row}: seq {seq} duplicated in one batch"
                        )));
                    }
                    batch_seen[seq as usize] = true;
                    let Some(track) = tracks.get_mut(&seq) else {
                        return Err(proto(format!(
                            "batch row {row}: stream {seq} with no BeginStream"
                        )));
                    };
                    if track.chunk > 1 && pos < track.prompt_len {
                        return Err(proto(format!(
                            "batch row {row}: stream {seq} still prefilling at pos {pos}"
                        )));
                    }
                    if pos != track.next_pos {
                        return Err(proto(format!(
                            "batch row {row}: stream {seq} activation pos {pos} != expected {}",
                            track.next_pos
                        )));
                    }
                    if pos >= stages[seq as usize].max_positions() {
                        return Err(proto(format!(
                            "batch row {row}: stream {seq} activation pos {pos} > stage capacity {}",
                            stages[seq as usize].max_positions()
                        )));
                    }
                    batch_rows.push((seq, pos));
                    batch_hidden.extend_from_slice(&hidden);
                    track.next_pos += 1;
                    steps += 1;
                }
                for (row, &(seq, pos)) in batch_rows.iter().enumerate() {
                    let stage: &mut dyn StageCompute = &mut *stages[seq as usize];
                    let lo = row * hidden_dim;
                    stage.step_submit(0, pos, &batch_hidden[lo..lo + hidden_dim]);
                }
                for &(seq, pos) in batch_rows.iter() {
                    let logits = stages[seq as usize].step_finish();
                    on_logits(seq, pos, logits);
                    let id = argmax_first(logits);
                    let mut reply_buf = [0u8; 8];
                    reply_buf[..4].copy_from_slice(&seq.to_le_bytes());
                    reply_buf[4..].copy_from_slice(&id.to_le_bytes());
                    link.send_frame(FrameKind::TokenStream, &reply_buf)?;
                }
            }
            k => return Err(proto(format!("unexpected {k:?} frame mid-stream"))),
        }
    }
}
