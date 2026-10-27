//! Multi-host pipeline-parallel inference (Plan 623 Phase 2) — the Rust
//! rewrite of exo's pipeline path, over the Phase-1 substrate in
//! [`riir_infer_core::multi_host`].
//!
//! Shape (PUSH, no hub, one one-way hop per stage boundary per token):
//!
//! ```text
//!  head (layers [0,k) + embed)  ── ActivationFrame ──▶  tail (layers [k,n) + norm + lm_head + sampler)
//!        ▲                                                      │
//!        └──────────────── 4-byte token id ◀────────────────────┘
//! ```
//!
//! - [`transport`] — the [`Transport`](transport::Transport) trait + the
//!   std TCP impl. Every frame is BLAKE3-keyed-MAC'd over
//!   `direction | kind | counter | payload` with a strict per-direction
//!   counter: no unauthenticated, replayed, reordered or reflected frame
//!   is accepted. NOT an RPC surface — the peer can only push activations
//!   and token ids.
//! - [`sockopts`] — [`dial`](sockopts::dial) / [`listen`](sockopts::listen):
//!   the std-equivalent socket setup with the T2.1b knobs (`SO_SNDBUF`/
//!   `SO_RCVBUF` sizing, named-interface bind), read-back verified.
//! - [`stage`] — the [`StageCompute`](stage::StageCompute) seam (one layer
//!   window of one model) + the CPU reference stage over the ternary
//!   DeltaNet range forward. The GPU stage plugs into the same trait.
//! - [`runner`] — the head loop (owns the token stream) and the tail
//!   serve loop (owns the sampler).
//! - [`multi`] — the multi-stream round-robin runner (T2.4): `s` streams
//!   over ONE link, one stage instance per stream, the pipeline bubble
//!   filled and prompt chunks interleaved with other streams' decode.
//!
//! - [`planner`] — the split planner (T2.5): per-host boot probes
//!   ([`planner::probe_decode_spt`]) merge into the chosen split through
//!   the pure [`planner::plan_split`] (max stage time + link hop; the
//!   memory axis refuses before timing; manual `--split` overrides).
//!
//! Correctness posture (G1): f32 activations on the wire, greedy
//! first-index argmax on the tail — the 2-stage token stream is
//! byte-identical to the 1-box forward (`tests/two_stage_g1.rs`).

pub mod handshake;
pub mod multi;
pub mod planner;
pub mod runner;
pub mod sockopts;
pub mod stage;
#[cfg(feature = "gpu")]
pub mod stage_gpu;
pub mod transport;

/// First-index strict argmax (Issue-940 determinism law: one fixed tie
/// rule, sequential scan). NaN never wins.
pub fn argmax_first(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best as u32
}
