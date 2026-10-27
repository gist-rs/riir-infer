//! The two-role stage runtime (Plan 623 T2.2).
//!
//! Protocol, per connection:
//!
//! 1. head → tail `Handshake` (head's [`StageHandshake`]); tail verifies
//!    the pair (`verify_pair`: arch / geometry / artifact / dtype / range
//!    tiling) and replies with its own `Handshake`, which the head checks
//!    the same way. Any mismatch refuses LOUD before a single activation.
//! 2. per position: head → tail `Activation` ([`ActivationFrame`], f32,
//!    `pos` in the header); tail → head `Token` (u32 LE, greedy
//!    first-index argmax over the tail's logits).
//! 3. head → tail `Shutdown`.
//!
//! PoC scope: one stream, one hop, prefill token-by-token (T2.3 pipelined
//! prefill and T2.4 micro-batching ride the same frames later).

use std::io;
use std::time::{Duration, Instant};

use riir_infer_core::multi_host::{ActivationFrame, FrameDtype};

use crate::argmax_first;
use crate::handshake;
use crate::stage::StageCompute;
use crate::transport::{FrameKind, Transport};

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

/// Head role: owns the token stream. Feeds `prompt`, then greedily
/// generates `max_new` tokens (the tail samples; the head embeds).
pub fn run_head<S: StageCompute, T: Transport>(
    stage: &mut S,
    link: &mut T,
    prompt: &[u32],
    max_new: usize,
) -> io::Result<HeadReport> {
    if prompt.is_empty() {
        return Err(proto("empty prompt".into()));
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

    let mut frame = Vec::with_capacity(20 + 4 * stage.hidden_dim());
    let mut report = HeadReport::default();
    let empty: [f32; 0] = [];

    let mut push = |stage: &mut S, link: &mut T, token: u32, pos: usize| -> io::Result<u32> {
        let hidden = stage.step(token, pos, &empty);
        frame.clear();
        ActivationFrame::encode_into(0, pos as u32, 1, hidden, FrameDtype::F32, &mut frame);
        link.send_frame(FrameKind::Activation, &frame)?;
        let kind = link.recv_frame_into(&mut buf)?;
        expect_kind(kind, FrameKind::Token)?;
        let id: [u8; 4] = buf
            .as_slice()
            .try_into()
            .map_err(|_| proto(format!("token frame len {} != 4", buf.len())))?;
        Ok(u32::from_le_bytes(id))
    };

    let t0 = Instant::now();
    let mut next = 0u32;
    for (pos, &tok) in prompt.iter().enumerate() {
        next = push(stage, link, tok, pos)?;
    }
    report.prefill = t0.elapsed();
    if max_new > 0 {
        report.generated.push(next);
    }
    let t1 = Instant::now();
    for i in 1..max_new {
        next = push(stage, link, next, prompt.len() + i - 1)?;
        report.generated.push(next);
    }
    report.decode = t1.elapsed();
    link.send_frame(FrameKind::Shutdown, &[])?;
    Ok(report)
}

/// Tail role: serves one head connection until `Shutdown`. `on_logits`
/// sees every position's logits (the G1 gate captures them; the CLI
/// passes a no-op).
pub fn run_tail<S: StageCompute, T: Transport>(
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
    let mut steps = 0usize;
    loop {
        match link.recv_frame_into(&mut buf)? {
            FrameKind::Shutdown => return Ok(steps),
            FrameKind::Activation => {}
            k => return Err(proto(format!("unexpected {k:?} frame mid-stream"))),
        }
        let hdr = ActivationFrame::decode_into(&buf, &mut hidden).map_err(proto)?;
        if hidden.len() != hidden_dim || hdr.n_tok != 1 {
            return Err(proto(format!(
                "activation shape {}×{} != 1×{hidden_dim}",
                hdr.n_tok,
                hidden.len()
            )));
        }
        let pos = hdr.pos as usize;
        // The stage cache is position-ordered: a skipped or repeated pos
        // would silently corrupt the KV/GDN state.
        if pos != steps {
            return Err(proto(format!("activation pos {pos} != expected {steps}")));
        }
        let logits = stage.step(0, pos, &hidden);
        on_logits(pos, logits);
        let id = argmax_first(logits);
        link.send_frame(FrameKind::Token, &id.to_le_bytes())?;
        steps += 1;
    }
}

/// The 1-box reference: the same greedy loop over ONE full-range stage
/// (`LayerRange::full`), no wire. The G1 oracle for [`run_head`].
pub fn run_local<S: StageCompute>(stage: &mut S, prompt: &[u32], max_new: usize) -> Vec<u32> {
    let empty: [f32; 0] = [];
    let mut out = Vec::with_capacity(max_new);
    let mut next = 0u32;
    for (pos, &tok) in prompt.iter().enumerate() {
        next = argmax_first(stage.step(tok, pos, &empty));
    }
    if max_new > 0 {
        out.push(next);
    }
    for i in 1..max_new {
        next = argmax_first(stage.step(next, prompt.len() + i - 1, &empty));
        out.push(next);
    }
    out
}
