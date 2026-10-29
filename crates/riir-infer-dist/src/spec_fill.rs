//! Speculative fill over the 2-stage pipeline (Plan 623 T5.1, V0).
//!
//! The decode lever is the per-token round trip (T2.6: Tailscale −10% ≈
//! one activation→token round trip per step). Speculative fill hides it:
//! the head drafts K tokens, both stages consume all K in ONE
//! [`FrameKind::VerifyStream`] round trip, and the tail replies every
//! row's argmax — K true continuations per exchange instead of one.
//!
//! Alignment (the runner's cursor law): cursor `c` = tokens consumed; the
//! pending prediction is the token AT position `c`, not yet consumed.
//! A verify round at cursor `c` consumes drafts `d_0..d_{K-1}` at
//! positions `c..c+K-1`; the tail's `argmax_j` is the true token at
//! position `c+j+1` given the hypothesized prefix. Acceptance is therefore
//! SHIFTED: `d_0 == pending`, then `d_j == argmax_{j-1}` — with `m` the
//! leading satisfied checks, the committed sequence gains `d_0..d_{m-1}`
//! plus the correction `argmax_{m-1}` (for `m == K` the correction is
//! `argmax_{K-1}`, the true token after the full verified prefix). Every
//! committed token is a target the pipeline itself computed — lossless by
//! ADJUDICATION, not by trust; the stream is byte-identical to sequential
//! decode for ANY drafter, including the never-accepts worst case (each
//! round degenerates to one ordinary step plus a wasted round trip). The
//! G1 battery pins all three regimes.
//!
//! Rewind (T5.1b, Issue 1148 rung 2) rides an in-process exact
//! [`StageCheckpoint`]: retained-buffer copies of the live state (attention
//! rows bounded by the runner's cursor + the full GDN recurrence/conv
//! state) — memcpy-class, zero steady-state allocation, byte-exact by
//! construction, no serialization. The GDN journal's rank-1 unwind stays
//! UNWIRED here by design: it carries a measured ~6e-6 reconstruction
//! drift (Bench 885) and the lane's lossless G1 is byte-exact — an
//! approximate rewind would give up the defining property (the stream ==
//! sequential byte-for-byte, any drafter). The `HybridStateSegment`
//! codec remains the cross-process handoff format (Issue 028).
//!
//! **V0 posture (honest scope):** the verify forward rides per-position
//! [`StageCompute::step`] on both stages — bit-identical to the chunked
//! path by the T2.3b G1 proof, but not yet a single batched dispatch (the
//! GPU micro-batched verify is the forwarder rung). The v0 drafter is
//! [`RepeatLastDrafter`] (acceptance ≈ 0 by construction): this module
//! proves the MACHINERY and the lossless G1, it proves NO gain — the
//! acceptance-rate row is meaningless until a real drafter exists
//! (T5.1d, a riir-train lane).
//!
//! Protocol shape (one stream, after the ordinary prompt prefill):
//!
//! ```text
//! head: checkpoint ─ consume d_0..d_{K-1} at c..c+K-1 ─┐
//!                                                      ▼
//!                    VerifyStream { pending, drafts, rows }
//!                                                      ▼
//! tail: checkpoint ─ consume rows ── argmax ×K ── m<K? restore, steps=c
//!                                                      ▼
//!                       VerifyReply { m, argmax ×K }  (one round trip)
//!                                                      ▼
//! head: m==K? commit d_0..d_{K-1} + argmax_{K-1} (state already at c+K)
//!       m< K? restore, commit d_0..d_{m-1} (+ pending already in),
//!             resume the ORDINARY path for the re-forward (m=0: the one
//!             pending token; m≥1: the m accepted drafts) — the last
//!             reply is the next pending prediction
//! ```

use std::io;

use riir_infer_core::multi_host::{ActivationFrame, FrameDtype};

use crate::argmax_first;
use crate::runner::{HeadOptions, HeadReport, proto};
use crate::stage::StageCompute;
use crate::transport::{FrameKind, Transport};

/// Upper bound on one verify round — a sanity cap, not a tuning knob.
pub const MAX_VERIFY_K: usize = 64;

/// One layer's retained checkpoint buffer (Issue 1148 rung 2). Kinds are
/// matched to the model's layer types at checkpoint time and never
/// renegotiated — a rebuilt stage (different geometry) refuses against a
/// stale checkpoint at restore.
#[derive(Debug, Clone)]
pub enum CheckpointLayer {
    /// Attention: the LIVE rows only, `cursor * kv_dim` wide each — the
    /// rows a rewind will overwrite are stale-but-unread (the forward
    /// reads/writes positionally), so they are neither captured nor
    /// restored.
    Attention { key: Vec<f32>, value: Vec<f32> },
    /// GDN: the full fixed-size recurrence matrix + conv window
    /// (position-independent — there is nothing to shrink).
    DeltaNet { rec: Vec<f32>, conv: Vec<f32> },
}

impl CheckpointLayer {
    /// Diagnostic name for refusal messages — the kinds never renegotiate.
    pub fn kind_name(&self) -> &'static str {
        match self {
            Self::Attention { .. } => "Attention",
            Self::DeltaNet { .. } => "DeltaNet",
        }
    }
}

/// An in-process exact state checkpoint (Issue 1148 rung 2): the rewind
/// WITHOUT serialization. [`SpecStage::checkpoint_state`] fills (and
/// reuses) the retained buffers — memcpy-class, zero steady-state
/// allocation, byte-exact by construction — and
/// [`SpecStage::restore_checkpoint`] copies them back. This is the
/// spec-lane fast path; the `HybridStateSegment` codec stays the
/// cross-process handoff format (Issue 028) and the exactness oracle the
/// G1 pins compare against.
#[derive(Debug, Default, Clone)]
pub struct StageCheckpoint {
    /// Live KV row count carried by the attention buffers (positions
    /// `[0..cursor)` written). The ternary forward never advances the
    /// cache's `fill_pos` tracker (it reads/writes rows positionally), so
    /// the runner's cursor is the ONLY live bound — passed in explicitly.
    pub cursor: usize,
    pub layers: Vec<CheckpointLayer>,
}

/// A stage whose full pipeline state can be snapshotted and restored
/// byte-exactly. The CPU stage implements it over the
/// `HybridStateSegment` codec (the wire/handoff path) and over
/// [`StageCheckpoint`] (the in-process fast path); a backend without them
/// simply does not implement the trait, and the spec runners refuse to
/// seat it (capability discovery at the type level — never a runtime
/// maybe).
pub trait SpecStage: StageCompute {
    /// Encode this stage's full state (KV rows + GDN recurrence + conv
    /// windows + fill marker) into `out` (cleared first).
    fn snapshot_state(&self, out: &mut Vec<u8>) -> Result<(), String>;
    /// Restore state from a [`SpecStage::snapshot_state`] encoding. A
    /// shape mismatch refuses; nothing is resized.
    fn restore_state(&mut self, bytes: &[u8]) -> Result<(), String>;

    /// Fill `ck` with this stage's exact state at `cursor` (the count of
    /// written KV positions). Retained buffers: calling this again REUSES
    /// `ck`'s allocations (memcpy-class; zero steady-state alloc). A
    /// sliding-window (ring) layer refuses — the spec rewind is
    /// unbounded-only, the same contract the segment codec documents.
    fn checkpoint_state(&self, cursor: usize, ck: &mut StageCheckpoint) -> Result<(), String>;
    /// Copy a [`SpecStage::checkpoint_state`] checkpoint back. Byte-exact
    /// on every restored row; rows beyond `ck.cursor` are left stale (the
    /// forward never reads past the positions it has rewritten).
    fn restore_checkpoint(&mut self, ck: &StageCheckpoint) -> Result<(), String>;
}

/// What a drafter sees: the committed sequence so far (prompt +
/// generated, the pending prediction INCLUDED as the last entry) and the
/// round's requested draft count.
pub struct DraftCtx<'a> {
    pub committed: &'a [u32],
    pub k: usize,
}

/// The draft source (T5.1c). V0 ships [`RepeatLastDrafter`]; the trained
/// MTP-head drafter is T5.1d ([`crate::spec_mtp::MtpHeadDrafter`]).
pub trait DraftSource {
    fn propose(&mut self, ctx: &DraftCtx<'_>) -> Vec<u32>;

    /// Roll-aware proposal (T5.1d). The runner has ALREADY checkpointed the
    /// stage when this runs; `roll(tok, pos, out)` steps the REAL stage at
    /// `pos` and writes the resulting hidden row into `out` (hidden_dim
    /// wide), appending it to the verify request's rows. A roll drafter
    /// MUST invoke the closure exactly `ctx.k` times — its own drafts at
    /// `cursor + j` — so the rolled rows ARE the verify payload (no second
    /// forward; the rewind restores from the pre-propose checkpoint). The
    /// default ignores the closure and falls back to [`DraftSource::propose`]
    /// (token-level drafters; the runner then steps the drafts itself — the
    /// legacy path, byte-identical).
    fn propose_with_roll(
        &mut self,
        ctx: &DraftCtx<'_>,
        _roll: &mut dyn FnMut(u32, usize, &mut [f32]),
    ) -> Vec<u32> {
        self.propose(ctx)
    }
}

/// The v0 drafter: repeat the last committed token. Acceptance ≈ 0 — the
/// honest disclosure in the module doc; this proves the machinery, not a
/// gain.
#[derive(Debug, Clone, Copy, Default)]
pub struct RepeatLastDrafter;

impl DraftSource for RepeatLastDrafter {
    fn propose(&mut self, ctx: &DraftCtx<'_>) -> Vec<u32> {
        let last = ctx.committed.last().copied().unwrap_or(0);
        vec![last; ctx.k]
    }
}

/// The corpus-modal drafter — the 654 promotion gate's "cheapest real
/// alternative" comparator: the most frequent token of a fit corpus
/// (first-seen tie-break, deterministic), proposed with zero model cost
/// and zero training. Semantics mirror the v0 floor instrument's
/// `UnigramModal` (`spec_drafter_floor`, Bench 654's provenance): d_0
/// echoes pending (the accepted_prefix law), every other slot proposes
/// the modal token. A token-level drafter — the runner steps its drafts
/// itself (the legacy path, byte-identical).
#[derive(Debug, Clone, Copy)]
pub struct ModalDrafter {
    token: u32,
}

impl ModalDrafter {
    /// Fit on a token slice in stream order: strict `>` on the running
    /// best keeps the FIRST token to reach the winning count — the same
    /// first-seen tie-break the floor instrument pins.
    pub fn fit(tokens: &[u32]) -> Self {
        let mut counts: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
        let mut best = (0u32, 0u32); // (count, token)
        for &t in tokens {
            let c = counts.entry(t).and_modify(|c| *c += 1).or_insert(1);
            if *c > best.0 {
                best = (*c, t);
            }
        }
        Self { token: best.1 }
    }

    /// The fitted modal token (the harness discloses it beside the row —
    /// a modal form is artifact (whitespace runs) vs content, the floor
    /// instrument's read).
    pub fn token(&self) -> u32 {
        self.token
    }
}

impl DraftSource for ModalDrafter {
    fn propose(&mut self, ctx: &DraftCtx<'_>) -> Vec<u32> {
        let last = ctx.committed.last().copied().unwrap_or(0);
        let mut out = vec![self.token; ctx.k];
        if ctx.k > 0 {
            out[0] = last;
        }
        out
    }
}

/// Verify-round wire codec (T5.1a). The pending prediction rides the
/// request so the tail adjudicates the SAME `m` the head does — it has no
/// other way to check `d_0`.
pub struct VerifyRequest<'a> {
    pub seq_id: u32,
    /// The prediction at the round's cursor (un-consumed; the head's last
    /// committed entry).
    pub pending: u32,
    pub drafts: &'a [u32],
    /// The HEAD stage's hidden rows for the drafted positions, `K ×
    /// hidden_dim` row-major (what an [`ActivationFrame`] carries for the
    /// same positions).
    pub rows: &'a [f32],
}

impl VerifyRequest<'_> {
    pub fn encode_into(&self, out: &mut Vec<u8>) {
        out.clear();
        out.extend_from_slice(&self.seq_id.to_le_bytes());
        out.extend_from_slice(&self.pending.to_le_bytes());
        out.extend_from_slice(&(self.drafts.len() as u32).to_le_bytes());
        for &d in self.drafts {
            out.extend_from_slice(&d.to_le_bytes());
        }
        // Rows as raw LE f32 bits (the same posture an ActivationFrame
        // carries) — a per-element loop, not bytemuck: this crate keeps
        // bytemuck a DEV dependency (the zero-new-deps law).
        out.reserve(self.rows.len() * 4);
        for &v in self.rows {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }

    /// Decode into `(seq_id, pending, drafts)`; the rows land in
    /// `out_rows` (`K × hidden_dim`, cleared first).
    pub fn decode_into(
        bytes: &[u8],
        hidden_dim: usize,
        out_rows: &mut Vec<f32>,
    ) -> Result<(u32, u32, Vec<u32>), String> {
        if bytes.len() < 12 {
            return Err(format!("VerifyStream: {} bytes < 12", bytes.len()));
        }
        let seq = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        let pending = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        let n = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        if n == 0 || n > MAX_VERIFY_K {
            return Err(format!(
                "VerifyStream: n_tok {n} outside 1..={MAX_VERIFY_K}"
            ));
        }
        let drafts_end = 12 + n * 4;
        let rows_end = drafts_end + n * hidden_dim * 4;
        if bytes.len() != rows_end {
            return Err(format!(
                "VerifyStream: body {} != 12 + {n} drafts + {n}×{hidden_dim} rows",
                bytes.len()
            ));
        }
        let drafts = bytes[12..drafts_end]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes(*c))
            .collect();
        out_rows.clear();
        out_rows.reserve((rows_end - drafts_end) / 4);
        for chunk in bytes[drafts_end..rows_end].as_chunks::<4>().0 {
            out_rows.push(f32::from_le_bytes(*chunk));
        }
        Ok((seq, pending, drafts))
    }
}

/// The verify reply: the tail's per-position argmaxes plus ITS OWN
/// adjudication count `m` (the head recomputes and refuses a mismatch —
/// both sides derive `m` from the same inputs, so a disagreement is a
/// broken peer, refused loud).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifyReplyData {
    pub seq_id: u32,
    /// Accepted-prefix count the tail computed.
    pub m: u32,
    /// The tail's per-position argmaxes, K entries.
    pub argmaxes: Vec<u32>,
}

pub fn encode_verify_reply(seq_id: u32, m: u32, argmaxes: &[u32], out: &mut Vec<u8>) {
    out.clear();
    out.extend_from_slice(&seq_id.to_le_bytes());
    out.extend_from_slice(&(argmaxes.len() as u32).to_le_bytes());
    out.extend_from_slice(&m.to_le_bytes());
    for &a in argmaxes {
        out.extend_from_slice(&a.to_le_bytes());
    }
}

pub fn decode_verify_reply(bytes: &[u8]) -> Result<VerifyReplyData, String> {
    if bytes.len() < 12 {
        return Err(format!("VerifyReply: {} bytes < 12", bytes.len()));
    }
    let seq = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
    let n = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    let m = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
    if n == 0 || n > MAX_VERIFY_K {
        return Err(format!("VerifyReply: n_tok {n} outside 1..={MAX_VERIFY_K}"));
    }
    if m as usize > n {
        return Err(format!("VerifyReply: m {m} > n_tok {n}"));
    }
    let end = 12 + n * 4;
    if bytes.len() != end {
        return Err(format!(
            "VerifyReply: body {} != 12 + {n} argmaxes",
            bytes.len()
        ));
    }
    let argmaxes = bytes[12..end]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect();
    Ok(VerifyReplyData {
        seq_id: seq,
        m,
        argmaxes,
    })
}

/// The accepted-prefix count under the runner's shifted alignment:
/// `d_0 == pending`, then `d_j == argmax[j-1]`. BOTH sides compute this
/// from the same inputs; the reply carries the tail's copy as a
/// cross-check. Caps at `drafts.len()`.
pub fn accepted_prefix(pending: u32, drafts: &[u32], argmaxes: &[u32]) -> usize {
    let mut m = 0usize;
    for (j, &d) in drafts.iter().enumerate() {
        let target = match j {
            0 => pending,
            _ => argmaxes[j - 1],
        };
        match d == target {
            true => m += 1,
            false => break,
        }
    }
    m
}

/// Head role, speculative mode (T5.1c): ordinary prompt prefill (the
/// same frames as [`crate::runner::run_head_with`]), then verify rounds
/// until `max_new` tokens are committed. `k` is the draft count per
/// round (clamped to the position capacity per round; `k` ≥ 1).
///
/// The stage must implement [`SpecStage`] — a backend without state
/// snapshot/restore cannot run the lossless adjudication and is refused
/// by the type system.
pub fn run_head_spec<S, T, D>(
    stage: &mut S,
    link: &mut T,
    prompt: &[u32],
    max_new: usize,
    opts: HeadOptions,
    drafter: &mut D,
    k: usize,
) -> io::Result<HeadReport>
where
    S: StageCompute + SpecStage,
    T: Transport,
    D: DraftSource,
{
    if prompt.is_empty() {
        return Err(proto("empty prompt".into()));
    }
    if k == 0 || k > MAX_VERIFY_K {
        return Err(proto(format!("spec k {k} outside 1..={MAX_VERIFY_K}")));
    }
    let needed = prompt.len() + max_new.saturating_sub(1);
    if needed > stage.max_positions() {
        return Err(proto(format!(
            "sequence needs {needed} positions > stage capacity {}",
            stage.max_positions()
        )));
    }
    let n_layer = stage.n_layer();

    // ── handshake + Begin: the ordinary protocol's opening (mirrors
    // runner::run_head_with; the G1 battery pins both openings).
    let mut buf = Vec::new();
    let mut wire: Vec<u8> = Vec::new();
    crate::handshake::encode(stage.handshake(), &mut buf);
    link.send_frame(FrameKind::Handshake, &buf)?;
    let kind = link.recv_frame_into(&mut buf)?;
    crate::runner::expect_kind(kind, FrameKind::Handshake)?;
    let tail_hs = crate::handshake::decode(&buf).map_err(proto)?;
    tail_hs
        .verify_pair(stage.handshake(), n_layer)
        .map_err(|e| proto(format!("tail handshake refused: {e}")))?;
    if !tail_hs.range.is_last(n_layer) {
        return Err(proto(
            "spec runner is 2-stage: tail must own the last layer".into(),
        ));
    }
    let chunk = crate::runner::prefill_width(prompt.len(), opts);
    let mut begin = [0u8; 8];
    begin[..4].copy_from_slice(&(prompt.len() as u32).to_le_bytes());
    begin[4..].copy_from_slice(&(chunk as u32).to_le_bytes());
    link.send_frame(FrameKind::Begin, &begin)?;

    let mut report = HeadReport::default();
    let empty: [f32; 0] = [];
    let prompt_len = prompt.len();
    // committed = prompt + generated, the pending prediction INCLUDED as
    // the last entry (the cursor is committed.len() - 1).
    let mut committed: Vec<u32> = prompt.to_vec();
    // The pre-verify checkpoint (Issue 1148 rung 2): in-process, exact,
    // retained buffers — no serialization on the rewind path.
    let mut ck = StageCheckpoint::default();
    let mut rows: Vec<f32> = Vec::new();
    let mut chunk_rows: Vec<f32> = Vec::new();

    // ── ordinary prompt prefill (chunked width per opts), mirrors
    // runner::run_head_with's prefill loop; the last chunk takes the
    // first token reply — the pending prediction at cursor prompt_len.
    let t0 = std::time::Instant::now();
    let last_lo = (prompt.len() - 1) / chunk * chunk;
    for c in (0..prompt.len())
        .step_by(chunk)
        .map(move |lo| lo..(lo + chunk).min(prompt.len()))
    {
        let frame_kind = match c.start == last_lo {
            true => FrameKind::Activation,
            false => FrameKind::Prefill,
        };
        let hidden_out = match (chunk > 1, c.start < prompt_len) {
            (true, true) => {
                stage.prefill_chunk(
                    &prompt[c.clone()],
                    &empty,
                    c.start,
                    prompt_len,
                    false,
                    &mut chunk_rows,
                );
                chunk_rows.as_slice()
            }
            _ => stage.step(prompt[c.start], c.start, &empty),
        };
        wire.clear();
        ActivationFrame::encode_into(
            0,
            c.start as u32,
            c.len() as u16,
            hidden_out,
            FrameDtype::F32,
            &mut wire,
        );
        link.send_frame(frame_kind, &wire)?;
    }
    committed.push(recv_token(link, &mut buf)?);
    report.prefill = t0.elapsed();

    // ── verify rounds. Invariants at the top of every round:
    //   cursor        = committed.len() - 1  (the pending token is last)
    //   ck            = the pre-verify checkpoint of THIS round's cursor
    //                   when a rewind is pending
    //   pending_rewind= Some((cursor, tokens to re-forward)) after a
    //                   partial accept — served before the next draft.
    let t1 = std::time::Instant::now();
    let mut seq: u32 = 0;
    let mut pending_rewind: Option<(usize, Vec<u32>)> = None;
    while committed.len() - prompt_len < max_new {
        if let Some((pos, reforward)) = pending_rewind.take() {
            // Serve the rewind: restore, then re-forward through the
            // ORDINARY path (the tail restored before it replied, so its
            // cursor expects exactly these frames). m == 0 re-forwards the
            // pending token (already committed — not re-pushed); m ≥ 1
            // re-forwards the accepted drafts (pushed). Every reply is the
            // prediction after the fed token; the LAST one becomes the new
            // pending.
            stage.restore_checkpoint(&ck).map_err(proto)?;
            let mut last_reply = 0u32;
            for (i, &tok) in reforward.iter().enumerate() {
                let p = pos + i;
                let out = stage.step(tok, p, &empty);
                wire.clear();
                ActivationFrame::encode_into(0, p as u32, 1, out, FrameDtype::F32, &mut wire);
                link.send_frame(FrameKind::Activation, &wire)?;
                // The reply to the LAST fed token IS the new pending (for
                // m ≥ 1 it is argmax_{m-1}, the bonus) — one reply per
                // frame, no extra round trip.
                last_reply = recv_token(link, &mut buf)?;
                let already = pos + i < committed.len() && committed[pos + i] == tok;
                if !already {
                    committed.push(tok);
                }
            }
            committed.push(last_reply);
            continue;
        }

        let cursor = committed.len() - 1;
        let pending = *committed.last().expect("nonempty");
        let k_round = k.min(stage.max_positions().saturating_sub(cursor)).max(1);

        // Checkpoint BEFORE drafting (Issue 1148 rung 2): a roll drafter
        // (T5.1d) advances the stage inside propose_with_roll, so the
        // pre-verify checkpoint has to precede it. Token-level drafters
        // never touch the stage, so the checkpoint content is identical to
        // the old post-propose order — the legacy path stays byte-identical.
        // Live rows = cursor: the head has forwarded positions [0..cursor).
        stage.checkpoint_state(cursor, &mut ck).map_err(proto)?;
        rows.clear();
        let mut rolled = 0usize;
        let mut roll_fault: Option<String> = None;
        let hidden_dim = stage.hidden_dim();
        let drafts: Vec<u32> = {
            let mut roll = |tok: u32, pos: usize, out: &mut [f32]| {
                if roll_fault.is_some() {
                    return;
                }
                if pos != cursor + rolled {
                    roll_fault = Some(format!("roll contract: pos {pos} != cursor + {rolled}"));
                    return;
                }
                if out.len() != hidden_dim {
                    roll_fault = Some(format!(
                        "roll contract: out len {} != hidden_dim {hidden_dim}",
                        out.len()
                    ));
                    return;
                }
                let r = stage.step(tok, pos, &empty);
                out.copy_from_slice(r);
                rows.extend_from_slice(r);
                rolled += 1;
            };
            drafter.propose_with_roll(
                &DraftCtx {
                    committed: &committed,
                    k: k_round,
                },
                &mut roll,
            )
        };
        if let Some(e) = roll_fault {
            return Err(proto(e));
        }
        if rolled == 0 {
            // Token-level drafter: the runner consumes the drafts itself
            // (the legacy path — the rows the tail verifies).
            for (j, &d) in drafts.iter().enumerate() {
                let out = stage.step(d, cursor + j, &empty);
                rows.extend_from_slice(out);
            }
        } else if rolled != drafts.len() {
            return Err(proto(format!(
                "roll drafter contract: {rolled} rolls for {} drafts",
                drafts.len()
            )));
        }

        // One round trip.
        VerifyRequest {
            seq_id: seq,
            pending,
            drafts: &drafts,
            rows: &rows,
        }
        .encode_into(&mut wire);
        link.send_frame(FrameKind::VerifyStream, &wire)?;
        let kind = link.recv_frame_into(&mut buf)?;
        crate::runner::expect_kind(kind, FrameKind::VerifyReply)?;
        let reply = decode_verify_reply(&buf).map_err(proto)?;
        if reply.seq_id != seq {
            return Err(proto(format!("verify reply seq {} != {seq}", reply.seq_id)));
        }
        seq += 1;

        // Adjudicate (both sides derive m; a disagreement is a broken peer).
        let m = accepted_prefix(pending, &drafts, &reply.argmaxes);
        if m as u32 != reply.m {
            return Err(proto(format!(
                "verify adjudication mismatch: head m {m} != tail m {}",
                reply.m
            )));
        }
        report.spec_rounds += 1;
        report.spec_accepted += m as u64;
        if m == drafts.len() {
            // Full accept: state already sits at cursor+K on both stages.
            // d_0 == pending is ALREADY committed — push d_1..d_{K-1} plus
            // the row-(K-1) prediction as the new pending.
            committed.extend_from_slice(&drafts[1..]);
            committed.push(reply.argmaxes[m - 1]);
        } else {
            // Partial: restore on both sides; the verified prefix rides the
            // re-forward (d_0 == pending already committed — the `already`
            // check skips it); m == 0 re-forwards the pending token itself
            // (an empty re-forward would send no frames and then block on
            // the prediction reply — the deadlock class).
            let mut reforward: Vec<u32> = drafts[..m].to_vec();
            if reforward.is_empty() {
                reforward.push(pending);
            }
            pending_rewind = Some((cursor, reforward));
        }
    }
    report.decode = t1.elapsed();
    report.generated = committed[prompt_len..].to_vec();
    report.generated.truncate(max_new);
    link.send_frame(FrameKind::Shutdown, &[])?;
    Ok(report)
}

fn recv_token<T: Transport>(link: &mut T, buf: &mut Vec<u8>) -> io::Result<u32> {
    let kind = link.recv_frame_into(buf)?;
    crate::runner::expect_kind(kind, FrameKind::Token)?;
    let id: [u8; 4] = buf
        .as_slice()
        .try_into()
        .map_err(|_| proto(format!("token frame len {} != 4", buf.len())))?;
    Ok(u32::from_le_bytes(id))
}

/// Tail role, speculative mode (T5.1a/c): the ordinary serve loop PLUS
/// the [`FrameKind::VerifyStream`] arm. The ordinary arm mirrors
/// `runner::run_tail`'s loop (the G1 battery pins both loops); the verify
/// arm checkpoints, consumes the K rows per-position, computes every row's
/// argmax, adjudicates `m` (it holds `pending` from the wire), and — on a
/// partial accept — restores its pre-verify state BEFORE replying, so the
/// ordinary sequence resumes at the verify's base position with no extra
/// round trip. Requires [`SpecStage`] (capability at the type level).
pub fn run_tail_spec<S, T>(
    stage: &mut S,
    link: &mut T,
    mut on_logits: impl FnMut(usize, &[f32]),
) -> io::Result<usize>
where
    S: StageCompute + SpecStage,
    T: Transport,
{
    let n_layer = stage.n_layer();
    let hidden_dim = stage.hidden_dim();
    let mut buf = Vec::new();
    let kind = link.recv_frame_into(&mut buf)?;
    crate::runner::expect_kind(kind, FrameKind::Handshake)?;
    let head_hs = crate::handshake::decode(&buf).map_err(proto)?;
    stage
        .handshake()
        .verify_pair(&head_hs, n_layer)
        .map_err(|e| proto(format!("head handshake refused: {e}")))?;
    if !head_hs.range.is_first() {
        return Err(proto(
            "spec runner is 2-stage: head must own layer 0".into(),
        ));
    }
    crate::handshake::encode(stage.handshake(), &mut buf);
    link.send_frame(FrameKind::Handshake, &buf)?;

    let mut hidden: Vec<f32> = Vec::with_capacity(hidden_dim);
    let mut out: Vec<f32> = Vec::new();
    let mut wire: Vec<u8> = Vec::new();
    // The pre-verify checkpoint (Issue 1148 rung 2): in-process, exact,
    // retained buffers.
    let mut ck = StageCheckpoint::default();
    let mut steps = 0usize;
    let kind = link.recv_frame_into(&mut buf)?;
    crate::runner::expect_kind(kind, FrameKind::Begin)?;
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
    let bound = crate::runner::frame_bound(chunk, hidden_dim);
    buf.reserve(bound);
    hidden.reserve(chunk.saturating_mul(hidden_dim));

    loop {
        let reply = match link.recv_frame_into(&mut buf)? {
            FrameKind::Shutdown => return Ok(steps),
            FrameKind::VerifyStream => {
                let (seq, pending, drafts) =
                    VerifyRequest::decode_into(&buf, hidden_dim, &mut hidden).map_err(proto)?;
                let pos = steps;
                if pos + drafts.len() > stage.max_positions() {
                    return Err(proto(format!(
                        "verify span {pos}+{} > stage capacity {}",
                        drafts.len(),
                        stage.max_positions()
                    )));
                }
                stage.checkpoint_state(pos, &mut ck).map_err(proto)?;
                let mut argmaxes: Vec<u32> = Vec::with_capacity(drafts.len());
                for (j, row) in hidden.chunks_exact(hidden_dim).enumerate() {
                    let logits = stage.step(0, pos + j, row);
                    on_logits(pos + j, logits);
                    argmaxes.push(argmax_first(logits));
                }
                let m = accepted_prefix(pending, &drafts, &argmaxes) as u32;
                if (m as usize) < drafts.len() {
                    // Partial accept: rewind BEFORE replying — the head
                    // resumes the ordinary path at `pos` right after the
                    // reply, and this loop's `steps` must expect it.
                    stage.restore_checkpoint(&ck).map_err(proto)?;
                    steps = pos;
                } else {
                    steps = pos + drafts.len();
                }
                encode_verify_reply(seq, m, &argmaxes, &mut wire);
                link.send_frame(FrameKind::VerifyReply, &wire)?;
                continue;
            }
            FrameKind::Activation => true,
            FrameKind::Prefill => false,
            k => return Err(proto(format!("unexpected {k:?} frame mid-stream"))),
        };
        let hdr = ActivationFrame::decode_into(&buf, &mut hidden).map_err(proto)?;
        let pos = hdr.pos as usize;
        let n_tok = hdr.n_tok as usize;
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
