//! Issue 742 T4 — the single-stream prefix KV+GDN cache for TTFT (the
//! `PREFIX_CACHE=1` half of the external 381 recipe; FreeToken
//! arXiv:2608.16157's GDN-checkpoint design input, `.research/342`).
//!
//! # Mechanism
//!
//! Whole-prefix checkpoints: at a semantic anchor (a token boundary the
//! caller chooses — the doc end before the question, a turn end), snapshot
//! the FULL model recurrent state (48 GDN recurrent matrices + 48 conv
//! windows, ~151 MB + ~8 MB) into device side-buffers, keyed by the token
//! prefix. A later request that shares the prefix restores the GDN state
//! (dtod) and re-fills only the suffix — the attention KV rows `[0..P)` are
//! NOT copied: they are still byte-identical in the live cache, because
//! every state-writing forward in this engine appends at `pos >= P` and the
//! [`Qwen38PrefixCache::note_write`] lineage rule prunes any checkpoint
//! whose KV tail a younger write could have clobbered.
//!
//! This is why the cache needs no allocator changes and preserves CUDA
//! graph capture (the T4 handoff decision): the `(p, use_qg)` verify-chunk
//! graphs bake the addresses of the LIVE `state.keys`/`state.values`
//! buffers — restoring a prefix must not move them (a `KvSegmentPool`-style
//! repaged cache would invalidate every captured graph; the vLLM
//! capture×prefix-cache corruption class, §2 of the issue, is exactly the
//! hazard this design avoids — and the T4 G1 gates re-prove it with cache
//! hits under the graph arm).
//!
//! # Substrate note (consume-vs-build, the honest divergence)
//!
//! `katgpt-kv`'s `KvSegmentPool` (rolling-hash + blake3 two-phase match)
//! is the cited substrate, but it indexes WINDOWED segments (start/end
//! offsets in a sliding-window pruned pool) — the reuse shape for a paged
//! KV cache, not whole-prefix GDN checkpoints — and `katgpt-kv` is not in
//! the `ternary_gemv_cuda_raw` dep closure (adding the edge for a match
//! structure buys nothing at ≤4 entries). The match PATTERN is consumed
//! faithfully: blake3 prefix key (fast filter) + exact token compare
//! (stronger than blake3-only verify — zero collision risk by
//! construction). FreeToken's own divergence note applies verbatim: their
//! full-attn half is SGLang's radix tree deduplicating shared prefixes
//! across branches; ours is exact-match longest-prefix — equivalent for
//! single-stream reuse.
//!
//! **Serving-lane trigger (katgpt-rs Issue 771 / Bench 762):** when a
//! multi-request (continuous-batching) lane lands, consume the PUBLIC
//! radix primitive — `katgpt_kv::radix_prefix::RadixPrefixTree` + the
//! `PagedKVCache` chunk-page seam (`chunk_page_tables` /
//! `retain_chunk_pages` / `release_chunk_pages` / `adopt_chunk_pages`),
//! opt-in `radix_prefix_cache` — do NOT re-derive a private tree here.
//! Measured: hit-rate 2.45× this flat shape at equal page budget, match
//! latency 9.8× (Bench 762). This flat cache remains the correct shape
//! for single-stream lanes.
//!
//! # Correctness contract
//!
//! - `insert(tokens)` requires the caller to have filled EXACTLY `tokens`
//!   (in order, from a zeroed/reset GDN state); the snapshot is
//!   stream-ordered after those fills.
//! - Every state-writing forward MUST call [`Qwen38PrefixCache::note_write`]
//!   with its base position (wired into `forward_token`,
//!   `forward_token_graph`, `forward_verify_chunk{,_graph}`); a checkpoint
//!   with `len > pos` is pruned before the write can clobber its KV tail.
//! - `restore` returns the matched prefix length; the caller continues
//!   filling from there. The GDN copies are stream-ordered before any
//!   subsequent kernels.
//!
//! Rides `ternary_gemv_cuda_raw` (dep:cudarc; CUDA-only, non-macOS).
#![cfg(all(feature = "ternary_gemv_cuda_raw", not(target_os = "macos")))]

use std::collections::HashMap;
use std::sync::Arc;

use cudarc::driver::safe::{CudaSlice, CudaStream};

use crate::qwen38_dense_cudarc::Qwen38DecodeState;

/// One whole-prefix checkpoint: the token prefix, its blake3 key, and the
/// device-side GDN snapshots (recurrent + conv, same shapes as the live
/// state buffers).
struct PrefixEntry {
    len: usize,
    hash: [u8; 32],
    tokens: Vec<u32>,
    snap_recurrent: Vec<CudaSlice<f32>>,
    snap_conv: Vec<CudaSlice<f32>>,
}

/// blake3 of the token sequence (little-endian u32s streamed — zero-alloc).
fn prefix_hash(tokens: &[u32]) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    for t in tokens {
        h.update(&t.to_le_bytes());
    }
    *h.finalize().as_bytes()
}

/// The single-stream prefix cache (LRU, bounded).
///
/// Entries are ordered most-recently-used first; eviction drops the tail.
/// The blake3 map is rebuilt on mutation (entry count is single-digit —
/// the map is the fast filter, the exact token compare is the authority).
pub struct Qwen38PrefixCache {
    /// MRU-first entries (whole-prefix checkpoints).
    entries: Vec<PrefixEntry>,
    /// blake3(prefix tokens) → index into `entries` (rebuilt on mutation).
    by_hash: HashMap<[u8; 32], usize>,
    max_entries: usize,
    /// Diagnostics: total prunes via the lineage rule (reported by the
    /// T4 harness — a prune is the KV-validity rule firing, not an error).
    pruned: usize,
}

impl Qwen38PrefixCache {
    /// `max_entries` bounds device memory (~159 MB per checkpoint for the
    /// 27B: 48×3.1 MB recurrent + 48×160 KB conv). Env knob
    /// `QWEN38_PREFIX_CACHE_MAX` (default 4).
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: Vec::new(),
            by_hash: HashMap::new(),
            max_entries: max_entries.max(1),
            pruned: 0,
        }
    }

    /// Live entry count (MRU-first).
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether any checkpoint is live.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Total lineage prunes since construction (diagnostic).
    pub fn pruned(&self) -> usize {
        self.pruned
    }

    fn rebuild_index(&mut self) {
        self.by_hash = self
            .entries
            .iter()
            .enumerate()
            .map(|(i, e)| (e.hash, i))
            .collect();
    }

    /// Snapshot the current GDN state as a checkpoint for `tokens`.
    ///
    /// Contract: the caller has filled exactly `tokens` (in order) — the
    /// dtod copies are stream-ordered after those fills. Re-inserting an
    /// existing prefix refreshes its snapshots in place (no new buffers).
    pub fn insert(
        &mut self,
        stream: &Arc<CudaStream>,
        state: &Qwen38DecodeState,
        tokens: &[u32],
    ) -> Result<(), String> {
        if tokens.is_empty() {
            return Err("prefix insert: empty prefix".into());
        }
        let hash = prefix_hash(tokens);
        if let Some(&idx) = self.by_hash.get(&hash) {
            // Existing checkpoint: refresh the snapshots (the state may
            // legitimately differ if the fill lineage changed — same
            // tokens, different history is impossible under the reset
            // contract, but the copy is cheap and keeps the invariant
            // self-enforcing). Move to front (MRU).
            let entry = &mut self.entries[idx];
            copy_state_to_snaps(stream, state, &mut entry.snap_recurrent, &mut entry.snap_conv)?;
            let e = self.entries.remove(idx);
            self.entries.insert(0, e);
            self.rebuild_index();
            return Ok(());
        }
        // Evict LRU while at capacity (drop frees the device slices).
        while self.entries.len() >= self.max_entries {
            self.entries.pop();
            self.pruned += 1;
        }
        let mut snap_recurrent = Vec::with_capacity(state.recurrent.len());
        for r in &state.recurrent {
            snap_recurrent.push(
                stream
                    .alloc_zeros::<f32>(r.len())
                    .map_err(|e| format!("prefix alloc rec: {e}"))?,
            );
        }
        let mut snap_conv = Vec::with_capacity(state.conv.len());
        for c in &state.conv {
            snap_conv.push(
                stream
                    .alloc_zeros::<f32>(c.len())
                    .map_err(|e| format!("prefix alloc conv: {e}"))?,
            );
        }
        copy_state_to_snaps(stream, state, &mut snap_recurrent, &mut snap_conv)?;
        self.entries.insert(
            0,
            PrefixEntry {
                len: tokens.len(),
                hash,
                tokens: tokens.to_vec(),
                snap_recurrent,
                snap_conv,
            },
        );
        self.rebuild_index();
        Ok(())
    }

    /// Longest-prefix match against `prompt`: on a hit, restore the GDN
    /// state (dtod, stream-ordered) and return the matched length — the
    /// caller re-fills only `prompt[matched..]`. The attention KV rows
    /// below the matched length are already correct in the live cache
    /// (the lineage rule guarantees it); rows at/above will be
    /// overwritten-before-read by the suffix fill.
    pub fn restore(
        &mut self,
        stream: &Arc<CudaStream>,
        state: &mut Qwen38DecodeState,
        prompt: &[u32],
    ) -> Result<Option<usize>, String> {
        // Candidate lengths, longest first — the entry set is tiny, and
        // hashing each candidate prefix once is O(len) blake3 per entry.
        let mut lens: Vec<usize> = self.entries.iter().map(|e| e.len).collect();
        lens.sort_unstable_by(|a, b| b.cmp(a));
        for l in lens {
            if l > prompt.len() {
                continue;
            }
            let h = prefix_hash(&prompt[..l]);
            let Some(&idx) = self.by_hash.get(&h) else {
                continue;
            };
            // Exact verification (the authority — stronger than the hash).
            if self.entries[idx].tokens != prompt[..l] {
                continue;
            }
            {
                let entry = &self.entries[idx];
                for (r, s) in state.recurrent.iter_mut().zip(entry.snap_recurrent.iter()) {
                    stream
                        .memcpy_dtod(s, r)
                        .map_err(|e| format!("prefix restore rec: {e}"))?;
                }
                for (c, s) in state.conv.iter_mut().zip(entry.snap_conv.iter()) {
                    stream
                        .memcpy_dtod(s, c)
                        .map_err(|e| format!("prefix restore conv: {e}"))?;
                }
            }
            let e = self.entries.remove(idx);
            self.entries.insert(0, e);
            self.rebuild_index();
            return Ok(Some(l));
        }
        Ok(None)
    }

    /// The KV-lineage rule: a state write at `pos` clobbers KV rows
    /// `[pos, ..)` — every checkpoint with `len > pos` may have a stale
    /// tail and is pruned. Checkpoints at `len <= pos` keep byte-valid KV
    /// by construction (writes never touch rows below `pos`).
    pub fn note_write(&mut self, pos: usize) {
        let before = self.entries.len();
        self.entries.retain(|e| e.len <= pos);
        if self.entries.len() != before {
            self.pruned += before - self.entries.len();
            self.rebuild_index();
        }
    }
}

/// state → snapshots (dtod, stream-ordered after any prior fills).
fn copy_state_to_snaps(
    stream: &Arc<CudaStream>,
    state: &Qwen38DecodeState,
    snap_recurrent: &mut [CudaSlice<f32>],
    snap_conv: &mut [CudaSlice<f32>],
) -> Result<(), String> {
    for (s, r) in snap_recurrent.iter_mut().zip(state.recurrent.iter()) {
        stream
            .memcpy_dtod(r, s)
            .map_err(|e| format!("prefix snap rec: {e}"))?;
    }
    for (s, c) in snap_conv.iter_mut().zip(state.conv.iter()) {
        stream
            .memcpy_dtod(c, s)
            .map_err(|e| format!("prefix snap conv: {e}"))?;
    }
    Ok(())
}
