//! Issue 994 / riir-train Issue 452 T4 — the construction weight-readback
//! instrument.
//!
//! The ≥52K-position class on the cubecl/wgpu stack can complete a forward
//! with SELF-CONSISTENT SILENTLY-WRONG activations (bit-exact in-process,
//! cos 0.18–0.42 vs an external reference; zero pool failures — Bench 600's
//! zero-failure mode). The mechanism is classified (margin-riding, box-state
//! decides the mode) but not explained. This module is the named unblock:
//! read back EVERY uploaded weight buffer and BLAKE3 it, so any run that hits
//! the silent mode gets an immediate weights-vs-not-weights answer.
//!
//! ## The decisive comparison (no host-side ground truth needed)
//!
//! Weight buffers are model-static — their content cannot legitimately depend
//! on `config.block_size`. So:
//!
//! - **cross-config**: construct at a KNOWN-GOOD block size (the ≤48K class,
//!   externally anchor-validated: Benches 598/599/601) and at the suspect
//!   class, hash both manifests — differing hashes at the same buffer names
//!   ⇒ the weights themselves were corrupted at construction (the upload /
//!   pool write is the mechanism). Identical hashes ⇒ the weights are exactly
//!   the known-good bytes and the corruption lives in activations/state.
//! - **within-run drift**: manifest → run a prefill → manifest again; a
//!   change means some kernel overwrote weight buffers mid-forward.
//!
//! Diagnostic tooling only (the `issue994_weight_readback_probe` driver in
//! riir-train); never a hot path. Cost: one full weight readback + BLAKE3 per
//! manifest (a few GiB, seconds).

use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::gemv_ternary_cubecl::TernaryHandle;
use crate::ternary_deltanet_gpu_forward::TernaryDeltanetGpuForward;

/// One weight buffer's readback identity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WeightBufferEntry {
    /// Stable name (`layer{idx}.{field}[.{sub}]` / `global.{field}[.{sub}]`).
    pub name: String,
    /// BLAKE3 of the readback bytes, hex.
    pub blake3_hex: String,
    /// Buffer size in bytes.
    pub bytes: usize,
}

fn hash_handle(
    client: &ComputeClient<crate::cubecl_runtime::ActiveRuntime>,
    prefix: &str,
    h: &Handle,
    out: &mut Vec<WeightBufferEntry>,
) {
    let bytes = client
        .read_one(h.clone())
        .unwrap_or_else(|e| panic!("weight_readback: {prefix} readback failed: {e}"));
    out.push(WeightBufferEntry {
        name: prefix.to_string(),
        blake3_hex: blake3::hash(&bytes).to_hex().to_string(),
        bytes: bytes.len(),
    });
}

/// Hash every CubeCL sub-buffer of a `TernaryHandle` (pos/neg bit planes, the
/// f32 group scales, the optional raw-f16 scales). The lazy mirror caches
/// (Metal/wgpu/CUDA mma) are deliberately skipped — they are downstream
/// copies of these handles, not sources of truth.
fn hash_ternary(
    client: &ComputeClient<crate::cubecl_runtime::ActiveRuntime>,
    prefix: &str,
    t: &TernaryHandle,
    out: &mut Vec<WeightBufferEntry>,
) {
    hash_handle(client, &format!("{prefix}.pos_bits_u32"), &t.pos_bits_u32, out);
    hash_handle(client, &format!("{prefix}.neg_bits_u32"), &t.neg_bits_u32, out);
    hash_handle(client, &format!("{prefix}.group_scale_f32"), &t.group_scale_f32, out);
    if let Some(h) = &t.group_scale_f16 {
        hash_handle(client, &format!("{prefix}.group_scale_f16"), h, out);
    }
}

/// Walk EVERY uploaded weight buffer of the forward and hash its readback.
/// Buffer order is deterministic (layer index, then field declaration order),
/// so two manifests compare element-wise by name.
pub fn manifest(fwd: &TernaryDeltanetGpuForward) -> Vec<WeightBufferEntry> {
    let client = &fwd.client;
    let mut out = Vec::with_capacity(fwd.layers.len() * 16 + 8);
    for (li, l) in fwd.layers.iter().enumerate() {
        let p = format!("layer{li}");
        #[cfg(feature = "ternary_gemm_batched")]
        {
            hash_ternary(client, &format!("{p}.in_proj_qkv"), &l.in_proj_qkv, &mut out);
            hash_ternary(client, &format!("{p}.in_proj_z"), &l.in_proj_z, &mut out);
            hash_ternary(client, &format!("{p}.in_proj_a"), &l.in_proj_a, &mut out);
            hash_ternary(client, &format!("{p}.in_proj_b"), &l.in_proj_b, &mut out);
        }
        if let Some(h) = &l.in_proj_a_f32 {
            hash_handle(client, &format!("{p}.in_proj_a_f32"), h, &mut out);
        }
        if let Some(h) = &l.in_proj_b_f32 {
            hash_handle(client, &format!("{p}.in_proj_b_f32"), h, &mut out);
        }
        hash_ternary(client, &format!("{p}.in_proj_concat"), &l.in_proj_concat, &mut out);
        hash_ternary(client, &format!("{p}.out_proj"), &l.out_proj, &mut out);
        hash_ternary(client, &format!("{p}.gate_proj"), &l.gate_proj, &mut out);
        hash_ternary(client, &format!("{p}.up_proj"), &l.up_proj, &mut out);
        hash_ternary(client, &format!("{p}.down_proj"), &l.down_proj, &mut out);
        hash_ternary(client, &format!("{p}.gate_up_proj"), &l.gate_up_proj, &mut out);
        hash_handle(client, &format!("{p}.input_norm"), &l.input_norm, &mut out);
        hash_handle(client, &format!("{p}.post_attn_norm"), &l.post_attn_norm, &mut out);
        hash_handle(client, &format!("{p}.conv1d_weight"), &l.conv1d_weight, &mut out);
        hash_handle(client, &format!("{p}.a_log"), &l.a_log, &mut out);
        hash_handle(client, &format!("{p}.dt_bias"), &l.dt_bias, &mut out);
        hash_handle(client, &format!("{p}.linear_norm"), &l.linear_norm, &mut out);
        if let Some(t) = &l.attn_wq {
            hash_ternary(client, &format!("{p}.attn_wq"), t, &mut out);
        }
        if let Some(t) = &l.attn_wkv {
            hash_ternary(client, &format!("{p}.attn_wkv"), t, &mut out);
        }
        if let Some(t) = &l.attn_wo {
            hash_ternary(client, &format!("{p}.attn_wo"), t, &mut out);
        }
        if let Some(h) = &l.attn_q_norm {
            hash_handle(client, &format!("{p}.attn_q_norm"), h, &mut out);
        }
        if let Some(h) = &l.attn_k_norm {
            hash_handle(client, &format!("{p}.attn_k_norm"), h, &mut out);
        }
    }
    hash_handle(client, "global.final_norm", &fwd.final_norm, &mut out);
    hash_ternary(client, "global.lm_head", &fwd.lm_head, &mut out);
    hash_ternary(client, "global.wte", &fwd.wte_handle, &mut out);
    out
}

/// A one-line whole-manifest digest: BLAKE3 over the per-buffer (name,
/// hash, bytes) tuples, so manifest equality is a single string compare.
pub fn manifest_digest(m: &[WeightBufferEntry]) -> String {
    let mut hasher = blake3::Hasher::new();
    for e in m {
        hasher.update(e.name.as_bytes());
        hasher.update(e.blake3_hex.as_bytes());
        hasher.update(&e.bytes.to_le_bytes());
    }
    hasher.finalize().to_hex().to_string()
}

/// One mismatched buffer: `(name, (hex, bytes) on side a, (hex, bytes) on
/// side b)` — `("(absent)", 0)` for a buffer that exists on only one side.
pub type WeightDiff = (String, (String, usize), (String, usize));

/// Element-wise comparison of two manifests by name. Returns the mismatch
/// list: one [`WeightDiff`] for every buffer whose hash differs (or that
/// exists on only one side).
pub fn diff(a: &[WeightBufferEntry], b: &[WeightBufferEntry]) -> Vec<WeightDiff> {
    use std::collections::HashMap;
    let mb: HashMap<&str, &WeightBufferEntry> =
        b.iter().map(|e| (e.name.as_str(), e)).collect();
    let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
    let mut out = Vec::new();
    for ea in a {
        seen.insert(ea.name.as_str());
        match mb.get(ea.name.as_str()) {
            Some(eb) => {
                if ea.blake3_hex != eb.blake3_hex || ea.bytes != eb.bytes {
                    out.push((
                        ea.name.clone(),
                        (ea.blake3_hex.clone(), ea.bytes),
                        (eb.blake3_hex.clone(), eb.bytes),
                    ));
                }
            }
            None => out.push((
                ea.name.clone(),
                (ea.blake3_hex.clone(), ea.bytes),
                ("(absent)".to_string(), 0),
            )),
        }
    }
    for eb in b {
        if !seen.contains(eb.name.as_str()) {
            out.push((
                eb.name.clone(),
                ("(absent)".to_string(), 0),
                (eb.blake3_hex.clone(), eb.bytes),
            ));
        }
    }
    out
}
