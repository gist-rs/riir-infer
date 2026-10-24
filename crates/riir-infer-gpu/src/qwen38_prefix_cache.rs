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
//! **Persistence delta (riir-infer Issue 1108 T2 — Research 014 §3.2, the
//! Locals paper's cross-restart prompt cache):** when that lane arms, the
//! second delta on top of the radix tree is persistence — the cache
//! outlives the process as a freeze/thaw fusion: entries are
//! BLAKE3-committed frozen state (radix index + whole-prefix GDN
//! checkpoints, this module's snapshot shape), thawed at serve time on
//! boot, never recomputed. The paper measures the payoff: 83% of sent
//! tokens served from memory, a 138k-token follow-up starts in 2.4 s (vs
//! ~8 min re-prefill). Class-level prior art exists (llama.cpp
//! `--prompt-cache` file persistence, the LMCache line) — the value is
//! the fusion note, a small assembly over substrate this workspace
//! already owns, not a novelty claim.
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
//!
//! # Issue 041 — the pointer-table copy path (opt-in)
//!
//! The dtod loops above submit ONE driver call per layer per direction — 96
//! submits per insert/restore cycle (48 recurrent + 48 conv), each riding
//! the µs-class driver slice. The table-copy path (`QWEN38_PREFIX_TABLE_COPY=1`)
//! replaces each 96-submit direction with ONE kernel launch: persistent
//! device pointer tables (the live state's addresses are guaranteed stable
//! by the graph-capture constraint — the same law the module doc states
//! above — so a 48-entry table built once stays valid for the state's
//! lifetime; a host-side first-address compare rebuilds it if a state is
//! ever recreated), and one `prefix_table_copy` launch per class that walks
//! `grid.y` = layer and copies `stride` elements src_table[layer] →
//! dst_table[layer]. Bit-identical to the dtod loop by construction (pure
//! Bit-identical to the dtod loop by construction (pure
//! f32 copies, element order unchanged) — gated by
//! `test_table_copy_bit_identity_against_dtod_loop`.
//!
//! # Issue 041 T4 — the snapshot arena pool
//!
//! Each entry's snapshots live in ONE contiguous buffer per class (layer
//! `i` at `[i*stride, (i+1)*stride)` — the uniform-shape insert check
//! refuses anything else, both paths). Eviction and lineage prunes return
//! the arenas to a pool (capped at `max_entries`); a geometry-matched
//! fresh insert takes one instead of paying the per-layer 96-alloc burst
//! — 2 arena allocs on a cold pool, ZERO on a pool hit (pinned by
//! `test_arena_pool_reuse_zero_allocs`). A pooled arena carries stale
//! bytes by design: the copy that follows overwrites every layer before
//! any read. Steady-state device memory stays at the `max_entries`-checkpoint
//! bound (one transient arena during the evict→take swap).
//!
//! # Issue 041 T3 — the submit-count probe
//!
//! [`submit_probe_counts`] reads the four global counters the cache paths
//! increment per driver submit: dtod memcpies, snapshot allocs, table
//! launches, table htods. One warm refresh-insert + restore cycle at the
//! 27B shape reads `[192, 0, 0, 0]` on the dtod path vs `[0, 0, 4, 0]` on
//! the table path; a fresh insert pays the T4 arena allocs (2 on a cold
//! pool, 0 on a geometry-matched pool hit) and the table path's one-time
//! table htods. The counters
//! are plain `Relaxed` adds on a path that runs ≤ ~200 submits per cycle —
//! unmeasurable against the driver slice they count.
#![cfg(all(feature = "ternary_gemv_cuda_raw", not(target_os = "macos")))]

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use cudarc::driver::safe::{CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig};
use cudarc::driver::{DevicePtr, PushKernelArg};

use crate::qwen38_dense_cudarc::Qwen38DecodeState;

/// Issue 041 T3 — submit-count probe (see the module doc). Order:
/// `[dtod_memcpies, snap_allocs, table_launches, table_htods]`.
static PROBE_DTOD_COPIES: AtomicU64 = AtomicU64::new(0);
static PROBE_SNAP_ALLOCS: AtomicU64 = AtomicU64::new(0);
static PROBE_TABLE_LAUNCHES: AtomicU64 = AtomicU64::new(0);
static PROBE_TABLE_HTODS: AtomicU64 = AtomicU64::new(0);

/// Read the submit-count probe: `[dtod_memcpies, snap_allocs,
/// table_launches, table_htods]` since process start (or the last
/// [`submit_probe_reset`]). Counts every driver submit the cache paths
/// make — the dtod loops, the fresh-insert ARENA allocs (T4: 2 per cold
/// pool miss, 0 on a geometry-matched pool hit), the table kernel
/// launches, and the pointer-table htods.
pub fn submit_probe_counts() -> [u64; 4] {
    [
        PROBE_DTOD_COPIES.load(Ordering::Relaxed),
        PROBE_SNAP_ALLOCS.load(Ordering::Relaxed),
        PROBE_TABLE_LAUNCHES.load(Ordering::Relaxed),
        PROBE_TABLE_HTODS.load(Ordering::Relaxed),
    ]
}

/// Zero the submit-count probe (test/example windowing).
pub fn submit_probe_reset() {
    PROBE_DTOD_COPIES.store(0, Ordering::Relaxed);
    PROBE_SNAP_ALLOCS.store(0, Ordering::Relaxed);
    PROBE_TABLE_LAUNCHES.store(0, Ordering::Relaxed);
    PROBE_TABLE_HTODS.store(0, Ordering::Relaxed);
}

/// One whole-prefix checkpoint: the token prefix, its blake3 key, and the
/// device-side GDN snapshots — T4 arenas: ONE contiguous buffer per class
/// (all layers back-to-back at the uniform stride; per-layer views are
/// computed at use time because cudarc views borrow their parent).
struct PrefixEntry {
    len: usize,
    hash: [u8; 32],
    tokens: Vec<u32>,
    /// `rec_layers * rec_stride` f32, layer `i` at `[i*rec_stride, (i+1)*rec_stride)`.
    snap_rec: CudaSlice<f32>,
    /// `conv_layers * conv_stride` f32, layer `i` at `[i*conv_stride, (i+1)*conv_stride)`.
    snap_conv: CudaSlice<f32>,
    /// The arena geometry (the live state's per-class layer count + stride
    /// at insert time — a pool hit must match all four).
    rec_layers: usize,
    rec_stride: usize,
    conv_layers: usize,
    conv_stride: usize,
    /// Issue 041 — device pointer tables over this entry's snapshot layers
    /// (the src side of a table-copy restore AND the dst side of a
    /// table-copy gather). Built once at entry creation when the table path
    /// is armed; `None` on the dtod path. The addresses point into the
    /// arenas above and stay valid for the entry's lifetime (T4: the arenas
    /// are recycled only after the entry is dropped).
    table_rec: Option<PtrTable>,
    table_conv: Option<PtrTable>,
}

/// T4 — a recycled arena pair (the eviction/prune return lane). Reused only
/// when the geometry matches the next fresh insert's live state exactly;
/// anything else re-allocates rather than copying wrongly.
struct RecycledArena {
    snap_rec: CudaSlice<f32>,
    snap_conv: CudaSlice<f32>,
    rec_layers: usize,
    rec_stride: usize,
    conv_layers: usize,
    conv_stride: usize,
}

impl RecycledArena {
    fn matches(
        &self,
        rec_layers: usize,
        rec_stride: usize,
        conv_layers: usize,
        conv_stride: usize,
    ) -> bool {
        self.rec_layers == rec_layers
            && self.rec_stride == rec_stride
            && self.conv_layers == conv_layers
            && self.conv_stride == conv_stride
    }
}

/// blake3 of the token sequence (little-endian u32s streamed — zero-alloc).
fn prefix_hash(tokens: &[u32]) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    for t in tokens {
        h.update(&t.to_le_bytes());
    }
    *h.finalize().as_bytes()
}

/// Issue 041 — how the table-copy path arms. Production reads the env
/// (opt-in: `QWEN38_PREFIX_TABLE_COPY=1`); tests pin the mode directly so
/// parallel tests never share env state (On/Off are test-only).
#[derive(Clone, Copy, PartialEq, Eq)]
#[cfg_attr(not(test), allow(dead_code))]
enum TableCopyMode {
    Env,
    On,
    Off,
}

impl TableCopyMode {
    fn armed(self) -> bool {
        match self {
            Self::On => true,
            Self::Off => false,
            Self::Env => std::env::var("QWEN38_PREFIX_TABLE_COPY").as_deref() == Ok("1"),
        }
    }
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
    /// Issue 041 — the table-copy engine, built lazily on first armed use.
    /// `None` unless [`TableCopyMode`] selects it.
    engine: Option<TableCopyEngine>,
    /// Arming source (env in production; direct in tests).
    table_mode: TableCopyMode,
    /// T4 — recycled snapshot arenas (eviction + lineage prunes return
    /// here; a geometry-matched fresh insert reuses one instead of paying
    /// the 96-alloc burst). Capped at `max_entries` — the pool never holds
    /// more device memory than the entries bound already allows.
    arena_pool: Vec<RecycledArena>,
}

impl Qwen38PrefixCache {
    /// `max_entries` bounds device memory (~159 MB per checkpoint for the
    /// 27B: 48×3.1 MB recurrent + 48×160 KB conv — T4: the arenas recycle
    /// through a pool capped at `max_entries`, so steady-state memory stays
    /// at this bound with one transient arena per evict→take swap). Env
    /// knob `QWEN38_PREFIX_CACHE_MAX` (default 4). The table-copy path arms
    /// via `QWEN38_PREFIX_TABLE_COPY=1` (opt-in — issue 041).
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: Vec::new(),
            by_hash: HashMap::new(),
            max_entries: max_entries.max(1),
            pruned: 0,
            engine: None,
            table_mode: TableCopyMode::Env,
            arena_pool: Vec::new(),
        }
    }

    /// Test constructor: pin the table-copy arming directly (never via env
    /// — parallel tests share the process environment).
    #[cfg(test)]
    fn with_table_mode(max_entries: usize, mode: TableCopyMode) -> Self {
        Self {
            table_mode: mode,
            ..Self::new(max_entries)
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

    /// T4 — return a dropped entry's arenas to the pool (capped: the pool
    /// never holds more arenas than `max_entries`, and the eviction→take
    /// ordering in [`Self::insert`] keeps the steady-state total at
    /// `max_entries` arenas — the pre-T4 device-memory bound, plus one
    /// transient arena during the swap).
    fn recycle_entry(&mut self, entry: PrefixEntry) {
        if self.arena_pool.len() < self.max_entries {
            self.arena_pool.push(RecycledArena {
                snap_rec: entry.snap_rec,
                snap_conv: entry.snap_conv,
                rec_layers: entry.rec_layers,
                rec_stride: entry.rec_stride,
                conv_layers: entry.conv_layers,
                conv_stride: entry.conv_stride,
            });
        }
    }

    /// T4 — take a geometry-matched arena pair from the pool, or allocate
    /// ONE buffer per class (2 allocs — the per-entry 96-alloc burst the
    /// pre-T4 path paid). A pooled arena carries stale bytes by design:
    /// every layer is fully overwritten by the copy that follows, before
    /// any read.
    fn take_arena(
        pool: &mut Vec<RecycledArena>,
        stream: &Arc<CudaStream>,
        rec_layers: usize,
        rec_stride: usize,
        conv_layers: usize,
        conv_stride: usize,
    ) -> Result<(CudaSlice<f32>, CudaSlice<f32>), String> {
        if let Some(pos) = pool
            .iter()
            .position(|a| a.matches(rec_layers, rec_stride, conv_layers, conv_stride))
        {
            let a = pool.remove(pos);
            return Ok((a.snap_rec, a.snap_conv));
        }
        PROBE_SNAP_ALLOCS.fetch_add(2, Ordering::Relaxed);
        let snap_rec = stream
            .alloc_zeros::<f32>(rec_layers * rec_stride)
            .map_err(|e| format!("prefix alloc rec arena: {e}"))?;
        let snap_conv = stream
            .alloc_zeros::<f32>(conv_layers * conv_stride)
            .map_err(|e| format!("prefix alloc conv arena: {e}"))?;
        Ok((snap_rec, snap_conv))
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
        if self.table_mode.armed() && self.engine.is_none() {
            self.engine = Some(TableCopyEngine::new(stream)?);
        }
        if let Some(engine) = self.engine.as_mut() {
            // Build/refresh the live-state tables BEFORE any entry work (the
            // state tables are the gather src).
            engine.sync_state_tables(stream, state)?;
        }
        if let Some(&idx) = self.by_hash.get(&hash) {
            // Existing checkpoint: refresh the snapshots (the state may
            // legitimately differ if the fill lineage changed — same
            // tokens, different history is impossible under the reset
            // contract, but the copy is cheap and keeps the invariant
            // self-enforcing). Move to front (MRU).
            {
                let engine = self.engine.as_ref();
                let entry = &mut self.entries[idx];
                if let Some(engine) = engine {
                    engine.ensure_entry_tables(stream, entry)?;
                    engine.copy(
                        stream,
                        ClassId::Recurrent,
                        engine.state_table(ClassId::Recurrent)?,
                        entry.table_rec.as_ref().expect("entry tables ensured"),
                    )?;
                    engine.copy(
                        stream,
                        ClassId::Conv,
                        engine.state_table(ClassId::Conv)?,
                        entry.table_conv.as_ref().expect("entry tables ensured"),
                    )?;
                } else {
                    copy_state_to_snaps(
                        stream,
                        state,
                        &mut entry.snap_rec,
                        entry.rec_stride,
                        &mut entry.snap_conv,
                        entry.conv_stride,
                    )?;
                }
            }
            let e = self.entries.remove(idx);
            self.entries.insert(0, e);
            self.rebuild_index();
            return Ok(());
        }
        // T4 — the arena layout copies one stride per class, so a
        // non-uniform live state is refused at insert (both paths; the
        // table build re-checks as defense in depth).
        let rec_layers = state.recurrent.len();
        let rec_stride = state.recurrent.first().map(|s| s.len()).unwrap_or(0);
        if state.recurrent.iter().any(|s| s.len() != rec_stride) {
            return Err(
                "prefix insert: non-uniform recurrent layer shapes are unsupported \
                 (the snapshot arena copies one stride per class)"
                    .into(),
            );
        }
        let conv_layers = state.conv.len();
        let conv_stride = state.conv.first().map(|s| s.len()).unwrap_or(0);
        if state.conv.iter().any(|s| s.len() != conv_stride) {
            return Err(
                "prefix insert: non-uniform conv layer shapes are unsupported \
                 (the snapshot arena copies one stride per class)"
                    .into(),
            );
        }
        // Evict LRU while at capacity (arenas recycle into the pool).
        while self.entries.len() >= self.max_entries {
            let e = self.entries.pop().expect("entries non-empty in loop");
            self.pruned += 1;
            self.recycle_entry(e);
        }
        let (mut snap_rec, mut snap_conv) = Self::take_arena(
            &mut self.arena_pool,
            stream,
            rec_layers,
            rec_stride,
            conv_layers,
            conv_stride,
        )?;
        // Table path: build the entry's dst tables BEFORE the gather reads
        // them; dtod path: the loop copy. Mutually exclusive.
        let mut table_rec = None;
        let mut table_conv = None;
        if let Some(engine) = self.engine.as_ref() {
            let t_rec =
                engine.upload_arena_table(stream, &snap_rec, rec_layers, rec_stride, "rec")?;
            let t_conv =
                engine.upload_arena_table(stream, &snap_conv, conv_layers, conv_stride, "conv")?;
            engine.copy(
                stream,
                ClassId::Recurrent,
                engine.state_table(ClassId::Recurrent)?,
                &t_rec,
            )?;
            engine.copy(
                stream,
                ClassId::Conv,
                engine.state_table(ClassId::Conv)?,
                &t_conv,
            )?;
            table_rec = Some(t_rec);
            table_conv = Some(t_conv);
        } else {
            copy_state_to_snaps(
                stream,
                state,
                &mut snap_rec,
                rec_stride,
                &mut snap_conv,
                conv_stride,
            )?;
        }
        self.entries.insert(
            0,
            PrefixEntry {
                len: tokens.len(),
                hash,
                tokens: tokens.to_vec(),
                snap_rec,
                snap_conv,
                rec_layers,
                rec_stride,
                conv_layers,
                conv_stride,
                table_rec,
                table_conv,
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
        if self.table_mode.armed() && self.engine.is_none() {
            self.engine = Some(TableCopyEngine::new(stream)?);
        }
        if let Some(engine) = self.engine.as_mut() {
            engine.sync_state_tables(stream, state)?;
        }
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
            // T5: an entry born on the dtod path gains its tables HERE —
            // the same lazy build the refresh path runs. Arming
            // mid-process no longer strands pre-arm checkpoints (the old
            // loud refusal was an arming-order artifact, not a shape
            // hazard — geometry drift still refuses in the table build's
            // len check and the copy's shape check).
            if let Some(engine) = self.engine.as_ref() {
                let entry = &mut self.entries[idx];
                engine.ensure_entry_tables(stream, entry)?;
            }
            {
                let engine = self.engine.as_ref();
                let entry = &self.entries[idx];
                if let Some(engine) = engine {
                    // Table path: the entry's snapshot tables are the copy
                    // SRC; the live-state tables the DST. One launch per
                    // class (issue 041).
                    let t_rec = entry.table_rec.as_ref().expect("entry tables ensured");
                    let t_conv = entry.table_conv.as_ref().expect("entry tables ensured");
                    engine.copy(
                        stream,
                        ClassId::Recurrent,
                        t_rec,
                        engine.state_table(ClassId::Recurrent)?,
                    )?;
                    engine.copy(
                        stream,
                        ClassId::Conv,
                        t_conv,
                        engine.state_table(ClassId::Conv)?,
                    )?;
                } else {
                    for (i, r) in state.recurrent.iter_mut().enumerate() {
                        PROBE_DTOD_COPIES.fetch_add(1, Ordering::Relaxed);
                        let view = entry
                            .snap_rec
                            .slice(i * entry.rec_stride..(i + 1) * entry.rec_stride);
                        stream
                            .memcpy_dtod(&view, r)
                            .map_err(|e| format!("prefix restore rec: {e}"))?;
                    }
                    for (i, c) in state.conv.iter_mut().enumerate() {
                        PROBE_DTOD_COPIES.fetch_add(1, Ordering::Relaxed);
                        let view = entry
                            .snap_conv
                            .slice(i * entry.conv_stride..(i + 1) * entry.conv_stride);
                        stream
                            .memcpy_dtod(&view, c)
                            .map_err(|e| format!("prefix restore conv: {e}"))?;
                    }
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
    /// by construction (writes never touch rows below `pos`). T4: pruned
    /// entries' arenas recycle into the pool.
    pub fn note_write(&mut self, pos: usize) {
        let before = self.entries.len();
        let mut i = 0;
        while i < self.entries.len() {
            if self.entries[i].len > pos {
                let e = self.entries.remove(i);
                self.recycle_entry(e);
            } else {
                i += 1;
            }
        }
        if self.entries.len() != before {
            self.pruned += before - self.entries.len();
            self.rebuild_index();
        }
    }
}

/// state → snapshot arenas (dtod into per-layer views, stream-ordered
/// after any prior fills). T4: the destination is ONE contiguous buffer
/// per class; layer `i` lives at `[i*stride, (i+1)*stride)`.
fn copy_state_to_snaps(
    stream: &Arc<CudaStream>,
    state: &Qwen38DecodeState,
    snap_rec: &mut CudaSlice<f32>,
    rec_stride: usize,
    snap_conv: &mut CudaSlice<f32>,
    conv_stride: usize,
) -> Result<(), String> {
    for (i, r) in state.recurrent.iter().enumerate() {
        PROBE_DTOD_COPIES.fetch_add(1, Ordering::Relaxed);
        let mut view = snap_rec.slice_mut(i * rec_stride..(i + 1) * rec_stride);
        stream
            .memcpy_dtod(r, &mut view)
            .map_err(|e| format!("prefix snap rec: {e}"))?;
    }
    for (i, c) in state.conv.iter().enumerate() {
        PROBE_DTOD_COPIES.fetch_add(1, Ordering::Relaxed);
        let mut view = snap_conv.slice_mut(i * conv_stride..(i + 1) * conv_stride);
        stream
            .memcpy_dtod(c, &mut view)
            .map_err(|e| format!("prefix snap conv: {e}"))?;
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────
// Issue 041 T1 — the pointer-table copy engine
//
// One launch per class replaces the 96-submit dtod loop: grid.y walks the
// layers, each block covers a slice of the (uniform) per-layer stride, and
// the copy reads/writes through device pointer tables. Pure f32 element
// copies in the natural order — bit-identical to the dtod loop by
// construction (gated by the test below, never weakened).
// ─────────────────────────────────────────────────────────────────────────

const PREFIX_TABLE_CUDA_SRC: &str = r#"
extern "C" __global__ void prefix_table_copy(
    const unsigned long long* __restrict__ src_table,
    const unsigned long long* __restrict__ dst_table,
    const unsigned int stride)
{
    const unsigned int layer = blockIdx.y;
    const unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i >= stride) { return; }
    const float* __restrict__ src = (const float*)src_table[layer];
    float* __restrict__ dst = (float*)dst_table[layer];
    dst[i] = src[i];
}
"#;

const TABLE_COPY_THREADS: u32 = 256;

/// The two layer classes (recurrent matrices, conv windows) — one launch
/// each per direction.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ClassId {
    Recurrent,
    Conv,
}

impl ClassId {
    fn label(self) -> &'static str {
        match self {
            Self::Recurrent => "recurrent",
            Self::Conv => "conv",
        }
    }
}

/// A device pointer table over one class of same-shaped slices, plus the
/// host-side address vector that built it (the live-state freshness probe;
/// unused for entry tables beyond the first-build record).
struct PtrTable {
    device: CudaSlice<u64>,
    /// Elements per layer (the uniform-stride assumption is checked at
    /// build time, never assumed silently).
    stride: usize,
    addrs: Vec<u64>,
}

impl PtrTable {
    fn layers(&self) -> u32 {
        self.device.len() as u32
    }
}

/// Owns the compiled kernel + the live-state tables. Built lazily on the
/// first armed insert/restore; `_module` keeps `kernel` valid (the
/// [`crate::gemv_ternary_cuda_raw::TernaryGemmCudaRaw`] precedent).
struct TableCopyEngine {
    _module: Arc<CudaModule>,
    kernel: CudaFunction,
    state_rec: Option<PtrTable>,
    state_conv: Option<PtrTable>,
}

impl TableCopyEngine {
    fn new(stream: &Arc<CudaStream>) -> Result<Self, String> {
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(
            PREFIX_TABLE_CUDA_SRC,
            cudarc::nvrtc::CompileOptions {
                arch: Some("sm_89"),
                ..Default::default()
            },
        )
        .map_err(|e| format!("prefix table nvrtc: {e}"))?;
        let ctx = stream.context();
        let module = ctx
            .load_module(ptx)
            .map_err(|e| format!("prefix table load_module: {e}"))?;
        let kernel = module
            .load_function("prefix_table_copy")
            .map_err(|e| format!("prefix table load_function: {e}"))?;
        Ok(Self {
            _module: module,
            kernel,
            state_rec: None,
            state_conv: None,
        })
    }

    /// Collect device addresses of an arena's per-layer views into a
    /// table (T4: the layers are arena slices at `i*stride`; the
    /// uniform-stride precondition is ENFORCED at insert — this builder
    /// re-checks as defense in depth). The addresses stay valid for the
    /// arena's lifetime, which is the entry's lifetime by construction
    /// (a recycled arena is overwritten fully before any read).
    fn upload_arena_table(
        &self,
        stream: &Arc<CudaStream>,
        arena: &CudaSlice<f32>,
        layers: usize,
        stride: usize,
        label: &str,
    ) -> Result<PtrTable, String> {
        if arena.len() != layers * stride {
            return Err(format!(
                "prefix table copy: non-uniform {label} arena geometry \
                 (len {} != {layers} layers × {stride})",
                arena.len()
            ));
        }
        let addrs: Vec<u64> = (0..layers)
            .map(|i| {
                arena
                    .slice(i * stride..(i + 1) * stride)
                    .device_ptr(stream)
                    .0
            })
            .collect();
        let mut device = stream
            .alloc_zeros::<u64>(addrs.len())
            .map_err(|e| format!("prefix table alloc ({label}): {e}"))?;
        stream
            .memcpy_htod(&addrs, &mut device)
            .map_err(|e| format!("prefix table htod ({label}): {e}"))?;
        PROBE_TABLE_HTODS.fetch_add(1, Ordering::Relaxed);
        Ok(PtrTable {
            device,
            stride,
            addrs,
        })
    }

    /// Rebuild the live-state tables when the state's addresses changed
    /// (a recreated state — the tables are stale otherwise). The host-side
    /// address compare is the freshness probe; a match reuses the table
    /// with zero submits.
    fn sync_state_tables(
        &mut self,
        stream: &Arc<CudaStream>,
        state: &Qwen38DecodeState,
    ) -> Result<(), String> {
        self.sync_one(stream, ClassId::Recurrent, &state.recurrent)?;
        self.sync_one(stream, ClassId::Conv, &state.conv)
    }

    fn sync_one(
        &mut self,
        stream: &Arc<CudaStream>,
        class: ClassId,
        slices: &[CudaSlice<f32>],
    ) -> Result<(), String> {
        let stride = slices.first().map(|s| s.len()).unwrap_or(0);
        if slices.iter().any(|s| s.len() != stride) {
            return Err(format!(
                "prefix table copy: non-uniform {} layer shapes are unsupported \
                 (the table path copies one stride per class)",
                class.label()
            ));
        }
        // Collect addresses FIRST (host-side, zero submits) — a warm call
        // with an unchanged state must not pay even the table re-upload.
        let addrs: Vec<u64> = slices.iter().map(|s| s.device_ptr(stream).0).collect();
        let slot = match class {
            ClassId::Recurrent => &mut self.state_rec,
            ClassId::Conv => &mut self.state_conv,
        };
        if slot.as_ref().is_some_and(|t| t.addrs == addrs) {
            return Ok(());
        }
        let mut device = stream
            .alloc_zeros::<u64>(addrs.len())
            .map_err(|e| format!("prefix table alloc ({}): {e}", class.label()))?;
        stream
            .memcpy_htod(&addrs, &mut device)
            .map_err(|e| format!("prefix table htod ({}): {e}", class.label()))?;
        PROBE_TABLE_HTODS.fetch_add(1, Ordering::Relaxed);
        *slot = Some(PtrTable {
            device,
            stride,
            addrs,
        });
        Ok(())
    }

    fn state_table(&self, class: ClassId) -> Result<&PtrTable, String> {
        match class {
            ClassId::Recurrent => self
                .state_rec
                .as_ref()
                .ok_or_else(|| "prefix table copy: state table missing".to_owned()),
            ClassId::Conv => self
                .state_conv
                .as_ref()
                .ok_or_else(|| "prefix table copy: state table missing".to_owned()),
        }
    }

    /// Ensure an entry carries tables (an entry created on the dtod path
    /// gains them lazily when the table path arms mid-process). T4: the
    /// tables point into the entry's arenas.
    fn ensure_entry_tables(
        &self,
        stream: &Arc<CudaStream>,
        entry: &mut PrefixEntry,
    ) -> Result<(), String> {
        if entry.table_rec.is_none() {
            entry.table_rec = Some(self.upload_arena_table(
                stream,
                &entry.snap_rec,
                entry.rec_layers,
                entry.rec_stride,
                "rec",
            )?);
        }
        if entry.table_conv.is_none() {
            entry.table_conv = Some(self.upload_arena_table(
                stream,
                &entry.snap_conv,
                entry.conv_layers,
                entry.conv_stride,
                "conv",
            )?);
        }
        Ok(())
    }

    /// The ONE launch: `dst_table[layer][i] = src_table[layer][i]` for
    /// every layer × element. Gather (state → entry) and scatter (entry →
    /// state) are the same kernel with the table order swapped.
    fn copy(
        &self,
        stream: &Arc<CudaStream>,
        class: ClassId,
        src: &PtrTable,
        dst: &PtrTable,
    ) -> Result<(), String> {
        if src.device.len() != dst.device.len() || src.stride != dst.stride {
            return Err(format!(
                "prefix table copy ({}): shape drift — layers {}/{} stride {}/{}; \
                 the cache entry and the live state must share one geometry",
                class.label(),
                src.device.len(),
                dst.device.len(),
                src.stride,
                dst.stride
            ));
        }
        let layers = src.layers();
        if layers == 0 || src.stride == 0 {
            return Ok(());
        }
        let stride_u = src.stride as u32;
        let gx = src.stride.div_ceil(TABLE_COPY_THREADS as usize) as u32;
        unsafe {
            stream
                .launch_builder(&self.kernel)
                .arg(&src.device)
                .arg(&dst.device)
                .arg(&stride_u)
                .launch(LaunchConfig {
                    grid_dim: (gx, layers, 1),
                    block_dim: (TABLE_COPY_THREADS, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| format!("prefix table copy launch ({}): {e}", class.label()))?;
        }
        PROBE_TABLE_LAUNCHES.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}

#[cfg(all(test, feature = "ternary_gemv_cuda_raw", not(target_os = "macos")))]
mod tests {
    use super::*;
    use cudarc::driver::safe::CudaContext;
    use std::sync::Mutex;

    /// Serializes the tests' cache ops: the T3 probe counters are global,
    /// so the probe test's count windows are only exact when no other
    /// test's insert/restore runs inside them. The tests are µs-scale —
    /// serialization costs nothing.
    static TEST_SER: Mutex<()> = Mutex::new(());

    const REC_LAYERS: usize = 4;
    const REC_LEN: usize = 512;
    const CONV_LAYERS: usize = 3;
    const CONV_LEN: usize = 128;

    fn test_stream() -> Arc<CudaStream> {
        let ctx = CudaContext::new(0).expect("CUDA context (the 4090 lane)");
        ctx.new_stream().expect("stream")
    }

    /// LCG fill — deterministic, every element in [-1, 1), no all-zero
    /// degeneracy (a zero state would pass a no-op copy).
    fn fill_class(
        stream: &Arc<CudaStream>,
        layers: usize,
        len: usize,
        seed: u64,
    ) -> Vec<CudaSlice<f32>> {
        let mut lcg = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let mut out = Vec::with_capacity(layers);
        for _ in 0..layers {
            let host: Vec<f32> = (0..len)
                .map(|_| {
                    lcg = lcg
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    ((lcg >> 33) as f32) / (1u64 << 31) as f32 - 1.0
                })
                .collect();
            let mut dev = stream.alloc_zeros::<f32>(len).expect("alloc");
            stream.memcpy_htod(&host, &mut dev).expect("htod");
            out.push(dev);
        }
        out
    }

    fn make_state(stream: &Arc<CudaStream>, seed: u64) -> Qwen38DecodeState {
        Qwen38DecodeState {
            keys: Vec::new(),
            values: Vec::new(),
            recurrent: fill_class(stream, REC_LAYERS, REC_LEN, seed),
            conv: fill_class(stream, CONV_LAYERS, CONV_LEN, seed.wrapping_add(1)),
        }
    }

    fn download_class(stream: &Arc<CudaStream>, slices: &[CudaSlice<f32>]) -> Vec<Vec<f32>> {
        slices
            .iter()
            .map(|s| {
                let mut v = vec![0.0f32; s.len()];
                stream.memcpy_dtoh(s, &mut v).expect("dtoh");
                v
            })
            .collect()
    }

    fn full_state_bytes(state: &Qwen38DecodeState, stream: &Arc<CudaStream>) -> Vec<u8> {
        let mut bytes = Vec::new();
        for v in download_class(stream, &state.recurrent) {
            for x in v {
                bytes.extend_from_slice(&x.to_bits().to_le_bytes());
            }
        }
        for v in download_class(stream, &state.conv) {
            for x in v {
                bytes.extend_from_slice(&x.to_bits().to_le_bytes());
            }
        }
        bytes
    }

    const TOKENS: [u32; 4] = [11, 22, 33, 44];

    /// T2 — the table path's insert+restore round-trip is BIT-IDENTICAL to
    /// the dtod loop's, on identical targets built from identical fills.
    #[test]
    fn test_table_copy_bit_identity_against_dtod_loop() {
        let _g = TEST_SER.lock().unwrap_or_else(|p| p.into_inner());
        let stream = test_stream();
        let state_a = make_state(&stream, 7);
        let mut c_dtod = Qwen38PrefixCache::with_table_mode(2, TableCopyMode::Off);
        let mut c_tab = Qwen38PrefixCache::with_table_mode(2, TableCopyMode::On);
        c_dtod
            .insert(&stream, &state_a, &TOKENS)
            .expect("dtod insert");
        c_tab
            .insert(&stream, &state_a, &TOKENS)
            .expect("table insert");

        // Two identical "dirty" targets — the restores must land the same
        // bytes the snapshot held (== state_a's).
        let mut b1 = make_state(&stream, 99);
        let mut b2 = make_state(&stream, 99);
        assert_eq!(c_dtod.restore(&stream, &mut b1, &TOKENS).unwrap(), Some(4));
        assert_eq!(c_tab.restore(&stream, &mut b2, &TOKENS).unwrap(), Some(4));

        let a = full_state_bytes(&state_a, &stream);
        let r1 = full_state_bytes(&b1, &stream);
        let r2 = full_state_bytes(&b2, &stream);
        assert_eq!(r1, a, "dtod round-trip must reproduce state_a");
        assert_eq!(
            r2, a,
            "table round-trip must be BIT-identical to the dtod loop"
        );
    }

    /// The refresh path (re-insert of an existing prefix), a second entry,
    /// and LRU eviction — the table path must survive the full entry
    /// lifecycle, not just one clean round-trip.
    #[test]
    fn test_table_copy_refresh_second_entry_and_eviction() {
        let _g = TEST_SER.lock().unwrap_or_else(|p| p.into_inner());
        let stream = test_stream();
        let mut c = Qwen38PrefixCache::with_table_mode(2, TableCopyMode::On);
        let s1 = make_state(&stream, 7);
        c.insert(&stream, &s1, &TOKENS).unwrap();
        c.insert(&stream, &s1, &TOKENS).unwrap(); // refresh (MRU, no new buffers)
        assert_eq!(c.len(), 1);
        let tokens2 = [55u32, 66, 77];
        let s2 = make_state(&stream, 31);
        c.insert(&stream, &s2, &tokens2).unwrap();
        assert_eq!(c.len(), 2);

        // Restore the OLDER entry — evicting nothing yet (cap 2).
        let mut t = make_state(&stream, 555);
        assert_eq!(c.restore(&stream, &mut t, &TOKENS).unwrap(), Some(4));
        assert_eq!(
            full_state_bytes(&t, &stream),
            full_state_bytes(&s1, &stream)
        );

        // A third insert evicts the LRU (tokens2 — restored most recently).
        let tokens3 = [88u32, 99];
        let s3 = make_state(&stream, 63);
        c.insert(&stream, &s3, &tokens3).unwrap();
        assert_eq!(c.len(), 2);
        let mut t3 = make_state(&stream, 777);
        assert_eq!(c.restore(&stream, &mut t3, &tokens3).unwrap(), Some(2));
        assert_eq!(
            full_state_bytes(&t3, &stream),
            full_state_bytes(&s3, &stream)
        );
    }

    /// A non-uniform class shape is refused loudly (the kernel's grid.y ×
    /// stride shape cannot express it) — never copied wrongly.
    #[test]
    fn test_table_copy_refuses_nonuniform_layers() {
        let _g = TEST_SER.lock().unwrap_or_else(|p| p.into_inner());
        let stream = test_stream();
        let mut rec = fill_class(&stream, 2, REC_LEN, 7);
        rec.push(fill_class(&stream, 1, REC_LEN * 2, 9).pop().unwrap());
        let state = Qwen38DecodeState {
            keys: Vec::new(),
            values: Vec::new(),
            recurrent: rec,
            conv: fill_class(&stream, 2, CONV_LEN, 5),
        };
        let mut c = Qwen38PrefixCache::with_table_mode(2, TableCopyMode::On);
        let err = c
            .insert(&stream, &state, &TOKENS)
            .expect_err("non-uniform shapes must refuse");
        assert!(err.contains("non-uniform"), "unexpected error: {err}");
    }

    /// T3 — the submit-count probe reads the documented counts. Fresh
    /// insert + warm refresh + restore into the live state (the production
    /// shape — the state-table freshness probe stays warm): the dtod path
    /// reads `layers(rec)+layers(conv)` dtod per direction and zero
    /// launches/htods; the table path reads 2 launches per direction, the
    /// one-time table htods (2 state + 2 entry) on the fresh insert only,
    /// and the same alloc burst both paths carry (T4's surface).
    #[test]
    fn test_submit_probe_counts() {
        let _g = TEST_SER.lock().unwrap_or_else(|p| p.into_inner());
        let stream = test_stream();
        let mut state = make_state(&stream, 21);

        // dtod arm: fresh insert + warm refresh + restore.
        submit_probe_reset();
        let mut c = Qwen38PrefixCache::with_table_mode(2, TableCopyMode::Off);
        c.insert(&stream, &state, &TOKENS).unwrap();
        c.insert(&stream, &state, &TOKENS).unwrap(); // warm refresh
        c.restore(&stream, &mut state, &TOKENS).unwrap();
        assert_eq!(
            submit_probe_counts(),
            [21, 2, 0, 0],
            "dtod: (4+3) dtod × 3 directions (fresh + refresh + restore), \
             2 arena allocs (cold pool), no table traffic"
        );

        // table arm: same cycle — into the same live state (warm state
        // tables on the restore).
        submit_probe_reset();
        let mut c2 = Qwen38PrefixCache::with_table_mode(2, TableCopyMode::On);
        c2.insert(&stream, &state, &TOKENS).unwrap();
        c2.insert(&stream, &state, &TOKENS).unwrap(); // warm refresh
        c2.restore(&stream, &mut state, &TOKENS).unwrap();
        assert_eq!(
            submit_probe_counts(),
            [0, 2, 6, 4],
            "table: 2 launches × 3 directions, 2 arena allocs (cold pool), \
             4 one-time htods (2 state + 2 entry tables)"
        );
    }

    /// T4 — the arena pool: a geometry-matched fresh insert after eviction
    /// allocates NOTHING (the pre-T4 path paid the 96-alloc burst every
    /// fresh insert), and the recycled arena round-trips bytes correctly
    /// (stale bytes are fully overwritten by the copy before any read).
    #[test]
    fn test_arena_pool_reuse_zero_allocs() {
        let _g = TEST_SER.lock().unwrap_or_else(|p| p.into_inner());
        let stream = test_stream();
        let state = make_state(&stream, 41);
        let tokens2 = [55u32, 66, 77];

        // Cold: first insert allocates the arenas. Second insert (different
        // prefix) evicts entry A into the pool, then TAKES it back — the
        // evict→take ordering means every insert after the first is a pool
        // hit (that is the steady serving cycle this gate pins).
        let mut c = Qwen38PrefixCache::with_table_mode(1, TableCopyMode::On);
        c.insert(&stream, &state, &TOKENS).unwrap();
        c.insert(&stream, &state, &tokens2).unwrap();

        // Steady state: evict + take from the pool → zero snapshot allocs.
        submit_probe_reset();
        c.insert(&stream, &state, &TOKENS).unwrap();
        let counts = submit_probe_counts();
        assert_eq!(
            counts[1], 0,
            "geometry-matched pool hit must not allocate (got {counts:?})"
        );
        assert_eq!(c.len(), 1);

        // The recycled arena round-trips: restore must land the snapshot's
        // bytes in a DIFFERENT dirty target (full byte compare).
        let mut t = make_state(&stream, 99);
        assert_eq!(c.restore(&stream, &mut t, &TOKENS).unwrap(), Some(4));
        assert_eq!(
            full_state_bytes(&t, &stream),
            full_state_bytes(&state, &stream),
            "restore through a pool-reused arena must reproduce the live state"
        );
    }

    /// T5 — a dtod-born entry restored under the table path gains its
    /// tables LAZILY (the same build the refresh path runs); arming
    /// mid-process no longer strands pre-arm checkpoints, and the
    /// lazy-armed scatter lands the snapshot bytes.
    #[test]
    fn test_table_restore_lazily_arms_dtod_entry() {
        let _g = TEST_SER.lock().unwrap_or_else(|p| p.into_inner());
        let stream = test_stream();
        let state = make_state(&stream, 55);

        // Env posture, unset: the insert runs the dtod loop.
        unsafe { std::env::remove_var("QWEN38_PREFIX_TABLE_COPY") };
        let mut c = Qwen38PrefixCache::new(2);
        c.insert(&stream, &state, &TOKENS).unwrap();
        let mut dirty = make_state(&stream, 56);
        c.restore(&stream, &mut dirty, &TOKENS).unwrap();
        assert_eq!(
            full_state_bytes(&dirty, &stream),
            full_state_bytes(&state, &stream),
            "dtod restore baseline"
        );

        // Arm mid-process: the restore must lazily build the tables —
        // never the old arming-order refusal — and land the same bytes.
        unsafe { std::env::set_var("QWEN38_PREFIX_TABLE_COPY", "1") };
        c.restore(&stream, &mut dirty, &TOKENS).unwrap();
        assert_eq!(
            full_state_bytes(&dirty, &stream),
            full_state_bytes(&state, &stream),
            "lazy-armed table restore must land the snapshot bytes"
        );
        unsafe { std::env::remove_var("QWEN38_PREFIX_TABLE_COPY") };
    }
}
