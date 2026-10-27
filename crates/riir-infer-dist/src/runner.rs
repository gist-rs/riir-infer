//! The two-role stage runtime (Plan 623 T2.2).
//!
//! Protocol, per connection:
//!
//! 1. head → tail `Handshake` (head's [`StageHandshake`]); tail verifies
//!    the pair (`verify_pair`: arch / geometry / artifact / dtype / range
//!    tiling) and replies with its own `Handshake`, which the head checks
//!    the same way. Any mismatch refuses LOUD before a single activation.
//! 2. head → tail `Begin` (`prompt_len`, the effective chunk width
//!    [`prefill_width`] — T2.3b).
//! 3. the prompt, in chunks of that width (1 = per position): every chunk but the last head → tail `Prefill` (no reply —
//!    the two stages overlap, T2.3a); the last chunk and every decode step:
//!    head → tail `Activation` ([`ActivationFrame`], f32, base `pos` +
//!    `n_tok` in the header) answered by tail → head `Token` (u32 LE, greedy
//!    first-index argmax over the tail's logits).
//! 4. head → tail `Shutdown`.
//!
//! With `prefill_chunk > 1` every prompt chunk rides
//! [`StageCompute::prefill_chunk`] on both stages (the GPU stage's batched
//! range prefill); decode steps always ride
//! [`StageCompute::step`]. Scope: one stream, one hop (T2.4 micro-batching
//! rides the same frames later).

use std::io;
use std::ops::Range;
use std::time::{Duration, Instant};

use riir_infer_core::multi_host::{ActivationFrame, FrameDtype};

use crate::argmax_first;
use crate::handshake;
use crate::stage::StageCompute;
use crate::transport::{self, FrameKind, Transport};

fn proto(msg: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg)
}

fn expect_kind(got: FrameKind, want: FrameKind) -> io::Result<()> {
    match got == want {
        true => Ok(()),
        false => Err(proto(format!("expected {want:?} frame, got {got:?}"))),
    }
}

/// What the head observed for one generation.
#[derive(Debug, Clone, Default)]
pub struct HeadReport {
    /// Generated token ids (excluding the prompt).
    pub generated: Vec<u32>,
    /// Wall time of prompt ingestion (all prompt positions, incl. the
    /// first sampled token's round trip).
    pub prefill: Duration,
    /// Wall time of the decode loop (`generated.len() - 1` steps).
    pub decode: Duration,
}

impl HeadReport {
    pub fn decode_tok_s(&self) -> f64 {
        let steps = self.generated.len().saturating_sub(1) as f64;
        match self.decode.as_secs_f64() > 0.0 {
            true => steps / self.decode.as_secs_f64(),
            false => 0.0,
        }
    }
}

/// Head-side knobs.
#[derive(Debug, Clone, Copy)]
pub struct HeadOptions {
    /// Requested prompt positions per prefill frame. `1` = the
    /// per-position protocol (T2.3a); `> 1` = chunked frames through
    /// [`StageCompute::prefill_chunk`] on both stages (T2.3b). The width
    /// actually used is [`prefill_width`].
    pub prefill_chunk: usize,
    /// Prompts shorter than this stay per-position. A batched chunk pays a
    /// fixed floor (measured on the M5 Ultra Metal stage, Bonsai-2 27B:
    /// ~0.9 s + 0.018 s/token vs 0.039 s/token per position → break-even
    /// ≈ 43 tokens), so a short prompt loses by batching.
    pub min_batched: usize,
    /// Stop generating once this id is produced (it is kept in the
    /// output). `None` = always generate `max_new`.
    pub eos: Option<u32>,
}

impl Default for HeadOptions {
    fn default() -> Self {
        Self {
            prefill_chunk: 1,
            min_batched: 64,
            eos: None,
        }
    }
}

/// The chunk width a prompt of `prompt_len` actually rides: `1` (per
/// position) when chunking is off or the prompt is under
/// `min_batched`; else the requested chunk EVENED over the prompt
/// (`ceil(len / ceil(len / chunk))`), so no ragged tail chunk pays the
/// batched floor for a handful of tokens. The head sends it in `Begin`;
/// the oracle uses the same rule.
pub fn prefill_width(prompt_len: usize, opts: HeadOptions) -> usize {
    let c = opts.prefill_chunk.max(1);
    match c == 1 || prompt_len < opts.min_batched.max(2) {
        true => 1,
        false => prompt_len.div_ceil(prompt_len.div_ceil(c)),
    }
}

/// Generation continues past `tok`?
fn keep_going(tok: u32, opts: HeadOptions) -> bool {
    opts.eos != Some(tok)
}

/// The prompt's prefill chunks at width `w`.
fn chunks(len: usize, w: usize) -> impl Iterator<Item = Range<usize>> {
    (0..len).step_by(w).map(move |lo| lo..(lo + w).min(len))
}

/// Wire bound of one activation frame carrying `n_rows` rows of `hidden`
/// f32: transport header + MAC + the [`ActivationFrame`] wire prefix + the
/// payload. The receiver reserves its buffers to this BEFORE the stream
/// starts (T2.1b preallocated frame buffers: the first chunked frame must
/// never be a mid-stream multi-MB realloc).
fn frame_bound(n_rows: usize, hidden: usize) -> usize {
    use riir_infer_core::multi_host::ActivationFrameHeader;
    transport::HDR_LEN
        + transport::MAC_LEN
        + ActivationFrameHeader::WIRE_PREFIX
        + 4 * n_rows * hidden
}

/// Head role: owns the token stream. Feeds `prompt`, then greedily
/// generates `max_new` tokens (the tail samples; the head embeds).
pub fn run_head<S: StageCompute + ?Sized, T: Transport>(
    stage: &mut S,
    link: &mut T,
    prompt: &[u32],
    max_new: usize,
) -> io::Result<HeadReport> {
    run_head_with(stage, link, prompt, max_new, HeadOptions::default())
}

/// [`run_head`] with explicit [`HeadOptions`].
pub fn run_head_with<S: StageCompute + ?Sized, T: Transport>(
    stage: &mut S,
    link: &mut T,
    prompt: &[u32],
    max_new: usize,
    opts: HeadOptions,
) -> io::Result<HeadReport> {
    if prompt.is_empty() {
        return Err(proto("empty prompt".into()));
    }
    // The last generated token is never fed back, so the sequence occupies
    // prompt + max_new - 1 positions.
    let needed = prompt.len() + max_new.saturating_sub(1);
    if needed > stage.max_positions() {
        return Err(proto(format!(
            "sequence needs {needed} positions > stage capacity {}",
            stage.max_positions()
        )));
    }
    let n_layer = stage.n_layer();
    let mut buf = Vec::new();
    handshake::encode(stage.handshake(), &mut buf);
    link.send_frame(FrameKind::Handshake, &buf)?;
    let kind = link.recv_frame_into(&mut buf)?;
    expect_kind(kind, FrameKind::Handshake)?;
    let tail_hs = handshake::decode(&buf).map_err(proto)?;
    tail_hs
        .verify_pair(stage.handshake(), n_layer)
        .map_err(|e| proto(format!("tail handshake refused: {e}")))?;
    if !tail_hs.range.is_last(n_layer) {
        return Err(proto(
            "PoC runner is 2-stage: tail must own the last layer".into(),
        ));
    }

    let chunk = prefill_width(prompt.len(), opts);
    if chunk > u16::MAX as usize {
        return Err(proto(format!("prefill chunk {chunk} > frame n_tok cap")));
    }
    let mut begin = [0u8; 8];
    begin[..4].copy_from_slice(&(prompt.len() as u32).to_le_bytes());
    begin[4..].copy_from_slice(&(chunk as u32).to_le_bytes());
    link.send_frame(FrameKind::Begin, &begin)?;

    let mut frame = Vec::with_capacity(20 + 4 * chunk * stage.hidden_dim());
    let mut rows = Vec::new();
    let mut report = HeadReport::default();
    let empty: [f32; 0] = [];
    let prompt_len = prompt.len();

    // `tokens` at `pos..` through this stage → one frame downstream. `kind`
    // is `Prefill` (no reply) or `Activation` (the tail answers with a
    // token). A prompt chunk under `chunk > 1` rides the batched prefill.
    let mut push = |stage: &mut S, link: &mut T, kind: FrameKind, tokens: &[u32], pos: usize| {
        let hidden = match (chunk > 1, pos < prompt_len) {
            (true, true) => {
                stage.prefill_chunk(tokens, &empty, pos, prompt_len, false, &mut rows);
                rows.as_slice()
            }
            _ => stage.step(tokens[0], pos, &empty),
        };
        frame.clear();
        ActivationFrame::encode_into(
            0,
            pos as u32,
            tokens.len() as u16,
            hidden,
            FrameDtype::F32,
            &mut frame,
        );
        link.send_frame(kind, &frame)
    };
    let mut recv_token = |link: &mut T| -> io::Result<u32> {
        let kind = link.recv_frame_into(&mut buf)?;
        expect_kind(kind, FrameKind::Token)?;
        let id: [u8; 4] = buf
            .as_slice()
            .try_into()
            .map_err(|_| proto(format!("token frame len {} != 4", buf.len())))?;
        Ok(u32::from_le_bytes(id))
    };

    // Pipelined prefill: no round trip until the last prompt chunk.
    let t0 = Instant::now();
    let last_lo = (prompt.len() - 1) / chunk * chunk;
    for c in chunks(prompt.len(), chunk) {
        let kind = match c.start == last_lo {
            true => FrameKind::Activation,
            false => FrameKind::Prefill,
        };
        push(stage, link, kind, &prompt[c.clone()], c.start)?;
    }
    let mut next = recv_token(link)?;
    report.prefill = t0.elapsed();
    if max_new > 0 {
        report.generated.push(next);
    }
    let t1 = Instant::now();
    for i in 1..max_new {
        if !keep_going(next, opts) {
            break;
        }
        push(
            stage,
            link,
            FrameKind::Activation,
            &[next],
            prompt.len() + i - 1,
        )?;
        next = recv_token(link)?;
        report.generated.push(next);
    }
    report.decode = t1.elapsed();
    link.send_frame(FrameKind::Shutdown, &[])?;
    Ok(report)
}

/// Tail role: serves one head connection until `Shutdown`. `on_logits`
/// sees every position's logits (the G1 gate captures them; the CLI
/// passes a no-op).
pub fn run_tail<S: StageCompute + ?Sized, T: Transport>(
    stage: &mut S,
    link: &mut T,
    mut on_logits: impl FnMut(usize, &[f32]),
) -> io::Result<usize> {
    let n_layer = stage.n_layer();
    let mut buf = Vec::new();
    let kind = link.recv_frame_into(&mut buf)?;
    expect_kind(kind, FrameKind::Handshake)?;
    let head_hs = handshake::decode(&buf).map_err(proto)?;
    stage
        .handshake()
        .verify_pair(&head_hs, n_layer)
        .map_err(|e| proto(format!("head handshake refused: {e}")))?;
    if !head_hs.range.is_first() {
        return Err(proto("PoC runner is 2-stage: head must own layer 0".into()));
    }
    handshake::encode(stage.handshake(), &mut buf);
    link.send_frame(FrameKind::Handshake, &buf)?;

    let hidden_dim = stage.hidden_dim();
    let mut hidden = Vec::with_capacity(hidden_dim);
    let mut out = Vec::new();
    let mut steps = 0usize;
    let kind = link.recv_frame_into(&mut buf)?;
    expect_kind(kind, FrameKind::Begin)?;
    let (prompt_len, chunk) = match buf.as_slice() {
        [a, b, c, d, e, f, g, h] => (
            u32::from_le_bytes([*a, *b, *c, *d]) as usize,
            u32::from_le_bytes([*e, *f, *g, *h]) as usize,
        ),
        _ => return Err(proto(format!("begin frame len {} != 8", buf.len()))),
    };
    if prompt_len == 0 || chunk == 0 {
        return Err(proto(format!(
            "begin: prompt_len {prompt_len}, chunk {chunk}"
        )));
    }
    // T2.1b: the activation frames are bounded by `chunk × hidden` —
    // preallocate the receive buffer and the hidden staging to the bound
    // so no frame reallocs mid-stream.
    let bound = frame_bound(chunk, hidden_dim);
    buf.reserve(bound);
    hidden.reserve(chunk.saturating_mul(hidden_dim));
    loop {
        let reply = match link.recv_frame_into(&mut buf)? {
            FrameKind::Shutdown => return Ok(steps),
            FrameKind::Activation => true,
            FrameKind::Prefill => false,
            k => return Err(proto(format!("unexpected {k:?} frame mid-stream"))),
        };
        let hdr = ActivationFrame::decode_into(&buf, &mut hidden).map_err(proto)?;
        let pos = hdr.pos as usize;
        let n_tok = hdr.n_tok as usize;
        // A prompt chunk spans exactly what the head's chunking produces;
        // every other frame is one position.
        let chunked = chunk > 1 && pos < prompt_len;
        let want_tok = match chunked {
            true => chunk.min(prompt_len - pos),
            false => 1,
        };
        if n_tok != want_tok || hidden.len() != n_tok * hidden_dim {
            return Err(proto(format!(
                "activation shape {n_tok}×{} != {want_tok}×{hidden_dim} at pos {pos}",
                hidden.len() / n_tok.max(1)
            )));
        }
        // The stage cache is position-ordered: a skipped or repeated pos
        // would silently corrupt the KV/GDN state.
        if pos != steps {
            return Err(proto(format!("activation pos {pos} != expected {steps}")));
        }
        if pos + n_tok > stage.max_positions() {
            return Err(proto(format!(
                "activation span {pos}+{n_tok} > stage capacity {}",
                stage.max_positions()
            )));
        }
        let logits = match chunked {
            true => {
                stage.prefill_chunk(&[], &hidden, pos, prompt_len, reply, &mut out);
                out.as_slice()
            }
            false => stage.step(0, pos, &hidden),
        };
        if reply {
            let last = pos + n_tok - 1;
            on_logits(last, logits);
            let id = argmax_first(logits);
            link.send_frame(FrameKind::Token, &id.to_le_bytes())?;
        } else if !chunked {
            on_logits(pos, logits);
        }
        steps += n_tok;
    }
}

/// The 1-box reference: the same greedy loop over ONE full-range stage
/// (`LayerRange::full`), no wire. The G1 oracle for [`run_head`].
pub fn run_local<S: StageCompute + ?Sized>(
    stage: &mut S,
    prompt: &[u32],
    max_new: usize,
) -> Vec<u32> {
    run_local_with(stage, prompt, max_new, HeadOptions::default())
}

/// [`run_local`] at the same prefill chunking a [`run_head_with`] uses —
/// the oracle for a chunked 2-stage run.
pub fn run_local_with<S: StageCompute + ?Sized>(
    stage: &mut S,
    prompt: &[u32],
    max_new: usize,
    opts: HeadOptions,
) -> Vec<u32> {
    let empty: [f32; 0] = [];
    let chunk = prefill_width(prompt.len(), opts);
    let mut out = Vec::with_capacity(max_new);
    let mut next = 0u32;
    match chunk {
        1 => {
            for (pos, &tok) in prompt.iter().enumerate() {
                next = argmax_first(stage.step(tok, pos, &empty));
            }
        }
        _ => {
            let mut logits = Vec::new();
            for c in chunks(prompt.len(), chunk) {
                let last = c.end == prompt.len();
                stage.prefill_chunk(
                    &prompt[c.clone()],
                    &empty,
                    c.start,
                    prompt.len(),
                    last,
                    &mut logits,
                );
            }
            next = argmax_first(&logits);
        }
    }
    if max_new > 0 {
        out.push(next);
    }
    for i in 1..max_new {
        if !keep_going(next, opts) {
            break;
        }
        next = argmax_first(stage.step(next, prompt.len() + i - 1, &empty));
        out.push(next);
    }
    out
}
