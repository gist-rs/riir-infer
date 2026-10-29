# Plan 624 — Sparse prefill attention arm (Issue 1156 T2, XAttention-class block-sparse)

**Status:** In progress — zed-amo-i1156 (lease expires 2026-10-31T17:22:37Z; issue `.issues/1156_sparse_prefill_league_eval.md`)

## Why (T1-recorded basis + one correction)

T1 (bench_734 stage decomposition, league posture, amo) measured attention share
**24.6% @16K / 39.3% @32K** — rising with P; FFN 56.1/46.0%; GDN 18.6/15.2%. GO for T2.

**Correction to T1's second finding (this plan's design input):** the
"flash arms launched ZERO kernels" observation counted the WRONG three rungs
(tiled/m16/cmma). The serving arm at 16K/32K on macOS is **m32**
(`QwenAttentionPrefillTiledM32CubeCL`): it is DEFAULT-ON macOS
(Bench-845 cooled re-promotion), its length gate `TILED_FLASH_M32_MIN_P = 8192`
routes on `active_prefill_len` = the FULL prompt length, and cmma/q8kv/m64/m32_pipe
are all opt-in/default-off (verified in source 2026-10-30). tiled/m16/cmma read 0
because m32 precedes them in the ladder. **The 39.3% @32K share is the m32 flash
arm's share — the sparse arm slots ABOVE m32 and must beat m32, not a legacy path.**

Also recorded from the m32 module doc (the lifecycle lesson): isolated kernel wins
(+11–27%) failed the sustained e2e gate (0.929× @16K, the Bench-790 sustained-load
class) before the cooled re-promotion. **T2's perf gate is sustained, interleaved
e2e, never isolated kernel time.**

## Design (static, training-free, opt-in)

Block-sparse prefill attention on the ATTENTION layers only (the dispatch site is
inside `prefill_attention_layer_batched` — GDN layers never reach it). Three CubeCL
kernels in a new file `crates/riir-infer-gpu/src/qwen_attention_prefill_sparse_cubecl.rs`
(`cubecl_runtime`-gated, sibling pattern; no new cargo feature, no new deps):

1. **Score kernel** — XAttention-class antidiagonal block scoring. Per
   (head, 128-query-block, 64-KV-block): sampled antidiagonal sums of the block's
   QKᵀ, stride 8, lane-local (each lane reads its 4 Q rows' antidiagonal K partners
   directly from global; no cross-lane shuffles), `plane_sum` tree reduce.
   Causal-horizon blocks beyond a q-block's reach score 0 and are never selected.
2. **Select kernel** — per (head, q-block): iterative first-index argmax picks
   until cumulative score ≥ τ·total (deterministic tie rule: lowest block index);
   sink blocks {0, 1} (first 128 tokens) always set. Writes a u32 bitmask.
3. **Sparse attention kernel** — the m32 body VERBATIM + one addition: at KV-block
   boundaries (`pos % 64 == 0`) test the mask bit (lane-0 read + `plane_broadcast`)
   and jump `pos += 64` when unset. Whole-cube uniform decision (every 32-row
   Q-tile lies inside one 128-row mask window; q_offset is 32-aligned). Skipping a
   fully-causal-masked or unselected block is a bit-exact no-op for online softmax
   (w=0, corr=1 — the m32 doc's own argument). At τ=1.0 (all blocks set) no skip
   ever fires → the loop sequence is m32's exactly → **bit-identical to m32**.

Mask layout: `mask[q_blk × (n_head × wpr) + head × wpr + kv_blk/32]`, bit
`kv_blk % 32`; `wpr = ceil(n_kv_blocks/32)`. Scores: f32
`[q_blk × n_head × n_kv_blocks]`. Scratch = grow-only OnceLock'd buffers sized
from the KV-cache row capacity × n_head (stable addresses; sparse arm engages
eager postures only — recorded limitation, league harness is eager).

### Posture (the repo's env-arm vocabulary)

- `RIIR_PREFILL_SPARSE` = `1|on|true` arms the lane (exact-literal; junk = off;
  DEFAULT OFF — the Issue-750-T3 lossy law: opt-in, never default without the GOAT).
- `RIIR_PREFILL_SPARSE_TAU` f32 (default 0.9, clamp [0.5, 1.0]).
- Consts: `SPARSE_KV_BLOCK = 64` (the FlashMemory/BSA block vocabulary),
  `SPARSE_SINK_TOKENS = 128`, `SPARSE_Q_BLOCK = 128`, `SPARSE_STRIDE = 8`,
  `PREFILL_SPARSE_MIN_P = 16384` (the `PREFILL_Q8KV_MIN_P` slot precedent).
- Engagement: `sparse_engaged() && hd == 256 && active_prefill_len >= MIN_P` —
  top of the dispatch chain (replaces the whole dense ladder for that layer call).
- Launch counters (the vacuous-guard pattern): `sparse_score/select/attn_launch_count()`.

## Gates (pre-registered, from the issue + the lossy law)

- **G1 quality:** τ=1.0 bit-identity (kernel-level + e2e argmax equality) is the
  hard gate. τ=0.9/0.7: real-text family walk (bench_948's protocol adapted to the
  macOS CubeCL lane, 16K, depth probes at chunk boundaries) — per-family ΔNLL +
  argmax-flip table RECORDED, no aggregate-only read. The binding per-family floors
  land with T3's ΩMSR sweep (continuation-ppl axis is the known killer — bench-889).
- **G2 perf:** sparse τ=0.9 e2e prefill wall ≥ 1.00× dense (sustained, interleaved
  median-of-ratios ≥3 rounds, `bench_preflight` provenance) at 16K and 32K on amo.
- **G3 byte-identity:** env unset → the ladder is untouched (no new branch fires);
  short-prompt bypass (`< MIN_P`) byte-identical.
- **G4:** zero alloc on the engaged hot path (grow-only scratch), no new deps.

## Tasks

- [x] **T2.1** Kernel file `qwen_attention_prefill_sparse_cubecl.rs`: score + select
      + sparse-attention kernels, launchers, counters, env fns, consts; `pub mod` row
      in `riir-infer-gpu/src/lib.rs`. (riir-infer `03838f0`)
- [x] **T2.2** Dispatch wiring in `ternary_deltanet_gpu_forward.rs`
      (`prefill_attention_layer_batched` ladder top) + per-chunk scratch
      (`BatchedAttentionScratch.sparse: Option<_>` — None = zero cost, G3). (same commit)
- [x] **T2.3** Kernel tests green: τ=1.0 bit-identity vs m32 (ragged + sub-cube +
      chunked-cache shapes), select semantics (threshold/sink/identity/horizon), scorer
      structural ranking, grammar; lib 330/0 + clippy -D at the
      `qwen38_wgpu,ternary_gemm_batched,metal_tensor_gemm` posture. (same commit)
- [x] **T2.4** Quality walk instrument in riir-ai (`bench_948` corpus verbatim, macOS
      lane, p=16384, 6 families × 2 items × 4 arms interleaved per item) — **G1 PASS:
      τ=1.0 FNV/argmax/NLL identical to dense 12/12 e2e; positive control PASS; 11/12
      argmax-stable at τ=0.9, 9/12 at τ=0.7; math is the weak axis**. Record:
      `../riir-ai/.benchmarks/1151_issue1156_sparse_prefill_quality.md` (bench 1151).
- [x] **T2.5** Perf A/B cells: dense vs τ=0.9 e2e interleaved median-of-ratios —
      **1.166× @16K / 1.310× @32K, 0 argmax flips on the perf items, G2 PASS**; box
      state disclosed (load ~2.0, AC, within-round pairing). Same bench 1151 record.
- [x] **T2.6** Issue 1156 T2 record (verdict + numbers), commit + push riir-infer
      and riir-ai. (riir-infer `198d245`; riir-ai `f92f0d6ed`)

- [-] CUDA-lane sparse port (the 4090's `att_pf_fa` family) — T6 scope, only if the
      Metal GOAT passes; the 4090 rows are the healthy ones (1.19×/1.14×).
- [-] T3 ΩMSR sweep instrument — next plan (the binding per-family floors live there).
- [-] gv/captured-graph posture for the sparse arm — league harness is eager; file
      with T6 if promotion ever needs captured serving.
