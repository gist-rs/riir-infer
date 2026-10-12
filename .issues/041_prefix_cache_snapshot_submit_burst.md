# Issue 041: prefix-cache snapshot submit burst — pointer-table gather/scatter (the #29807 analogue)

**Status:** OPEN — **T1+T2+T3+T4 DONE 2026-10-13:** the pointer-table copy path LANDED in
`qwen38_prefix_cache.rs` (module-local engine, not the rotation module — it needs
`PrefixEntry` knowledge; the divergence is deliberate) — dual device pointer tables (live
state + per-entry snapshots) + ONE `prefix_table_copy` launch per class per direction
(grid.y = layer), opt-in `QWEN38_PREFIX_TABLE_COPY=1`, dtod loop stays default. Gates
GREEN on the 4090 (5/5: bit-identity vs the dtod loop — the T2 gate —, refresh + second
entry + LRU eviction lifecycle, non-uniform shape loud refusal, the T3 submit-count probe
regression test, + the T4 pool-reuse gate); clippy clean at both postures. **T3 EXECUTED
(bench 036):** the submit probe reads `[192,0,4,0]` for a warm refresh+restore cycle
(96→2 per direction, exactly as designed); wall-time A/B at the real 27B shape
(~151 MiB/checkpoint, 41 order-rotated paired reps): enqueue **0.054×/0.045×** (18.5–22×
less host driver time — the dtod path is HOST-bound at ~8 µs/submit), sync-inclusive
**0.49×/0.48×**; stable across two runs. **T4 LANDED (same window):** per-entry snapshot
arenas (one contiguous buffer per class) + a recycled pool (cap `max_entries`) — fresh
inserts pay 2 arena allocs cold / 0 on a pool hit (was the 96-alloc burst every insert);
dtod fresh-insert wall 1.70 ms → **0.93 ms**; the warm path is untouched (bench 036
addendum). The dtod loop stays default — promotion is not gated on these benches
(serving-lane TTFT only, never a scored league lane).
Remaining: T5 (graph co-existence re-proof under the full forward — the issue-742 harness
is removed, so T5 needs a rebuilt real-model harness: GGUF load + graph capture + cache
hit under the graph arm).

## The surface (measured by Research 397, riir-ai `.research/397`)

`qwen38_prefix_cache.rs` checkpoints and restores the GDN state with per-layer
`memcpy_dtod` loops:

- `copy_state_to_snaps` = **96 separate dtod submits** per insert (48 recurrent + 48
  conv — Research 397's "128" was the 64-layer estimate; the 27B runs 48+48 per the
  module doc). A FRESH insert adds **96 `alloc_zeros` submits** on top.
- `restore` = **96 more dtod submits**.
- One insert+restore cycle ≈ **192+ submits** riding the WDDM driver slice.

Upstream precedent: llama.cpp PR **#29807** (CUDA: remove redundant copies after
SSM_SCAN — fuse the recurrent-state snapshot copy into the scan kernel; merged 10-09)
is the exact same move on the same model class (GDN/DeltaNet state writeback).

## Why the naive arena fix fails (the design dead-end, recorded)

Arena-layouting the SNAPSHOT dest alone does not reduce submits: the SOURCES
(`state.recurrent[i]`, `state.conv[i]`) are per-layer scattered `CudaSlice`s whose
addresses are baked into the captured `(p, use_qg)` verify-chunk graphs (the T4 design
constraint — restoring a prefix must not move the live buffers; the vLLM
capture×prefix-cache corruption class). An arena dest still needs one view-copy per
source = 96 submits. Only a kernel that takes MANY sources in ONE launch reduces the
count.

## The design

1. **Persistent device pointer table** for the live state: the graph-capture
   constraint GUARANTEES live addresses never move, so the table (96 ptrs = 768 B,
   one htod) is built once and reused by every insert/restore.
2. **Gather kernel** (snapshot): 2D grid — `blockIdx.y` = layer index (0..48 rec, then
   48..96 conv via one launch each or a fused 96-row y-range), `blockIdx.x` covers the
   layer element count grid-stride. Writes the contiguous arena snapshot:
   `arena[layer*stride + i] = ptr_table[layer][i]`. ONE launch (or two: rec+conv).
3. **Scatter twin** (restore): the same 2D grid reading arena → writing the live
   buffers via the pointer table. ONE launch.
4. **Persistent arena pool**: `max_entries` arenas pre-allocated (159 MB each for the
   27B), reused across inserts — kills the 96-alloc burst on fresh inserts. Eviction
   returns the arena to the pool, not the driver.
5. Total submits per cycle: **192+ → ~2-4**.

Bit-exact by construction (pure byte copy) — the G1 gate is a byte-compare against the
existing memcpy loop, no tolerance debate.

## Honest magnitude

~192 submits × µs-class driver slice ≈ **1-2 ms per insert+restore cycle**.

- Long-suffix TTFT (pp2048-class fill ≈ 540 ms): ~0.2% — noise.
- Short-suffix restore (FreeToken's scenario — restore + fill ~10 tokens): the burst is
  a meaningful fraction of a ~10 ms TTFT (up to ~20%).

This is a serving-lane (TTFT) optimization, NOT a scored league lane — the bout benches
never touch the prefix cache. Priority accordingly.

## Gates

- [x] T1 — pointer table + gather/scatter kernels — **DONE 2026-10-13, adapted**:
      module-local `TableCopyEngine` in `qwen38_prefix_cache.rs` (the engine needs
      entry-table lifetime coupling — module-local beats the rotation-module option);
      ONE kernel (`prefix_table_copy`) serves both directions (table order swapped);
      dual tables: live-state (freshness-checked per call, rebuilt only on address
      drift) + per-entry (built at entry creation, lifetime == entry).
- [x] T2 — G1: bit-identity vs the `memcpy_dtod` loop — **DONE 2026-10-13**:
      `test_table_copy_bit_identity_against_dtod_loop` (insert+restore through both
      paths into identical targets, byte-compare via `to_bits` — GREEN) + the entry
      lifecycle gate (refresh / second entry / LRU eviction) + the non-uniform refusal
      gate (the kernel's grid.y × stride shape cannot express per-layer strides — a
      non-uniform class refuses loudly, never copies wrongly).
- [x] T3 — G2: submit-count probe + wall-time A/B at insert/restore (short-suffix
      scenario FIRST — the only regime where the win is visible) — **DONE 2026-10-13**:
      module-level `submit_probe_counts()` counters (dtod memcpies / snap allocs / table
      launches / table htods; `test_submit_probe_counts` pins the warm-cycle counts at the
      test shape forever) + `examples/prefix_cache_ab.rs` (the paired instrument) —
      **bench 036**: probe `[192,0,4,0]` warm (96→2 per direction confirmed); enqueue
      0.054×/0.045×, sync 0.49×/0.48×, stable across two runs. The A/B used the real 27B
      STATE SHAPE (48×786,432 f32 rec + 48×40,960 f32 conv — the copies are shape-driven,
      not content-driven; the GGUF itself adds nothing to a byte copy). Restores targeted
      the live state object — the production shape, keeping the state-table freshness
      probe warm.
- [x] T4 — arena pool + eviction reuse; no new per-call allocs (G4 posture) —
      **DONE 2026-10-13**: per-entry snapshot arenas (ONE contiguous buffer per class;
      the uniform-shape insert check now covers BOTH paths — an arena cannot express
      per-layer strides) + `RecycledArena` pool (cap `max_entries`, evict→take ordering
      keeps steady-state memory at the pre-T4 bound with one transient arena). Fresh
      insert: 2 arena allocs cold, 0 on a geometry-matched pool hit
      (`test_arena_pool_reuse_zero_allocs` pins it + byte-compares a restore through a
      recycled arena); a pooled arena's stale bytes are fully overwritten before any
      read. Measured: dtod fresh-insert wall 1.70 ms → 0.93 ms (bench 036 addendum);
      warm cycle unchanged. The table path's entry-table htods (2 per fresh insert)
      are deliberately NOT recycled — recycling tables with arenas would save ~2 µs
      per fresh insert and cost an address-drift hazard.
- [ ] T5 — the graph-capture co-existence re-proof (cache hits under the graph arm —
      the existing T4 gate re-run; the pointer table must not change live addresses).
      Note: the table path ALIGNS with the capture constraint — it never allocates or
      moves live buffers; the gather/scatter launch is a plain stream kernel (capturable
      class) — but the proof is the existing gate re-run, not this argument.

## Refs

- riir-ai `.research/397` (opponent diff 2026-10-06→11; §League synthesis item 0)
- llama.cpp PR #29807 (`50e3e3e`, merged 2026-10-09)
- Issue 742 T4 (removed; git history + the module doc in `qwen38_prefix_cache.rs`)
- FreeToken arXiv:2608.16157 (GDN-checkpoint design input)
