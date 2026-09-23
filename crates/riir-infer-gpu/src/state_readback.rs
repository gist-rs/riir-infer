//! Issue 994 / riir-train Issue 452 T4 — the recurrence-state readback
//! instrument (the second named narrowing: "the same manifest pattern over
//! the DeltaNet state buffers").
//!
//! The weight-readback sibling ([`crate::weight_readback`]) eliminated the
//! weight-upload hypothesis for the ≥52K silent-corruption class: construction
//! at the suspect block size uploads bit-perfect weights. The mechanism
//! therefore lives in activations/state — this module hashes the STATE half.
//!
//! ## What is hashed (the persistent semantic state)
//!
//! - `deltanet_states[i]` / `conv_states[i]` — the per-DeltaNet-layer
//!   recurrent state and conv1d window (fixed-size, block-size-independent).
//! - the attention layers' KV caches' **valid prefix** `[0..pos)` rows
//!   (`[block_size, kvd]` row-major caches; only the written prefix is
//!   semantic — `reset_state` deliberately leaves stale rows beyond `pos`,
//!   unreachable by construction).
//! - `x` — the `[n_embd]` hidden state the final prefill chunk hands off
//!   (skipped at `pos == 0`: nothing has written it yet and the buffer's
//!   content is undefined, so hashing it would be nondeterministic noise).
//! - a synthetic `state.meta.pos` entry so a divergent position counter names
//!   itself in the diff instead of shifting every KV-prefix row.
//!
//! ## The decisive comparison (no host-side ground truth needed)
//!
//! Unlike weights, state is dynamic — but the persistent state after an
//! N-token prefill is a pure function of (weights, tokens) that CANNOT
//! legitimately depend on `config.block_size` (capacity): prefill chunking
//! depends only on N (`prefill_chunk_max` is config-independent), the
//! attention arm gates are PROMPT-level (Issue 782), KV fill is
//! absolute-position append-only, and the recurrent state carries across
//! chunks regardless of capacity. So:
//!
//! - **cross-config**: construct at a KNOWN-GOOD block size and at another
//!   block size, prefill the SAME N tokens in both, hash both post-prefill
//!   state manifests — any differing entry names a state buffer the suspect
//!   class corrupted (KV fill, recurrence state, or the hidden handoff).
//! - **determinism**: prefill → manifest → `reset_state` → prefill again →
//!   manifest; a change means a nondeterministic kernel or a
//!   pool-aliasing write (two "different" buffers sharing memory).
//! - **construction check**: the `pos == 0` manifest (zero-filled DeltaNet
//!   states + empty KV prefixes) must be identical across constructions —
//!   a non-zero or divergent zero-state names a bad pool handout at `new()`.
//!
//! Reads follow the capture tap's convention — read the WHOLE handle, hash
//! the semantic slice host-side — so the readback path itself cannot be a
//! confound. Diagnostic tooling only (the `issue994_state_readback_probe`
//! driver in riir-train); never a hot path. Cost: one state readback + BLAKE3
//! per manifest (~1.1 GB at block 8256, ~8.6 GB at 65600 on Bonsai-27B).
//!
//! ## Lane constraint
//!
//! The manifest reads the CubeCL handles directly. The cudarc whole-prefill
//! mirror lane (`prefill_cuda_full`, `ternary_gemv_cuda_raw`) keeps truth in
//! its mirrors until `sync_states` — a driver on that lane must sync first
//! (the `download_state_to_hybrid_cache` precedent). The probe's feature set
//! does not include it.

use cubecl::prelude::*;
use cubecl::server::Handle;

use riir_infer_core::types::DeltaNetLayerType;

use crate::ternary_deltanet_gpu_forward::TernaryDeltanetGpuForward;

/// One state buffer's readback identity — the same shape as the weight
/// instrument's entry, shared deliberately (one manifest digest + one diff
/// semantics for both readback families).
pub type StateBufferEntry = crate::weight_readback::WeightBufferEntry;
/// One mismatched state buffer, `(name, (hex, bytes) a, (hex, bytes) b)`.
pub type StateDiff = crate::weight_readback::WeightDiff;

pub use crate::weight_readback::{diff, manifest_digest};

fn push_hashed(
    client: &ComputeClient<crate::cubecl_runtime::ActiveRuntime>,
    name: &str,
    h: &Handle,
    limit_bytes: Option<usize>,
    out: &mut Vec<StateBufferEntry>,
) {
    let bytes = client
        .read_one(h.clone())
        .unwrap_or_else(|e| panic!("state_readback: {name} readback failed: {e}"));
    let bytes: &[u8] = match limit_bytes {
        Some(n) => &bytes[..n.min(bytes.len())],
        None => &bytes,
    };
    out.push(StateBufferEntry {
        name: name.to_string(),
        blake3_hex: blake3::hash(bytes).to_hex().to_string(),
        bytes: bytes.len(),
    });
}

/// Hash the persistent semantic state of the forward: per-DeltaNet-layer
/// recurrent + conv state, per-attention-layer KV valid prefix, the hidden
/// handoff `x` (pos > 0 only), and the position counter. Entry order is
/// deterministic (layer index, then field declaration order), so two
/// manifests compare element-wise by name via [`diff`].
pub fn manifest(fwd: &TernaryDeltanetGpuForward) -> Vec<StateBufferEntry> {
    let client = &fwd.client;
    let mut out = Vec::with_capacity(fwd.layer_types.len() * 2 + 4);

    out.push(StateBufferEntry {
        name: "state.meta.pos".to_string(),
        blake3_hex: blake3::hash(&(fwd.pos as u64).to_le_bytes())
            .to_hex()
            .to_string(),
        bytes: 8,
    });

    if fwd.pos > 0 {
        push_hashed(client, "state.hidden.x", &fwd.x, None, &mut out);
    }

    let kvd = fwd.config.n_kv_head * fwd.config.head_dim;
    let kv_prefix_bytes = fwd.pos.saturating_mul(kvd).saturating_mul(4);
    for (i, ty) in fwd.layer_types.iter().enumerate() {
        match ty {
            DeltaNetLayerType::DeltaNet => {
                if let Some(h) = &fwd.deltanet_states[i] {
                    push_hashed(
                        client,
                        &format!("state.layer{i}.deltanet_state"),
                        h,
                        None,
                        &mut out,
                    );
                }
                if let Some(h) = &fwd.conv_states[i] {
                    push_hashed(client, &format!("state.layer{i}.conv_state"), h, None, &mut out);
                }
            }
            DeltaNetLayerType::Attention => {
                if let Some(h) = &fwd.kv_key_caches[i] {
                    push_hashed(
                        client,
                        &format!("state.attn.layer{i}.k_prefix"),
                        h,
                        Some(kv_prefix_bytes),
                        &mut out,
                    );
                }
                if let Some(h) = &fwd.kv_value_caches[i] {
                    push_hashed(
                        client,
                        &format!("state.attn.layer{i}.v_prefix"),
                        h,
                        Some(kv_prefix_bytes),
                        &mut out,
                    );
                }
            }
        }
    }
    out
}
