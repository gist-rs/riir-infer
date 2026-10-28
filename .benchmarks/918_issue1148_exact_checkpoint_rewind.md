# Bench 918 — Issue 1148 rung 2+3: the exact checkpoint rewind (no serialize) + the pair re-cell

**Date:** 2026-10-28 · **Box:** shikuwa-4090 (Windows 11, i7-13700K 24 threads, DDR-to-be-disclosed) ·
**Build:** `cargo run --release` (`riir-infer-dist`, feature `bonsai2_hadamard`), isolated `CARGO_TARGET_DIR=C:/t1148` ·
**Posture:** in-process 2-stage CPU pair over loopback TCP, split 63, **ctx 4096** (the Bench 916/917 posture;
the artifact now declares a 256k context — `load_pair` gained the dist_stage-style clamp, the Issue-929
over-allocation class) · **Model:** `riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf` (BLAKE3-of-record in
riir-train; PQ2_0) · **Prompt:** `bonsai_mtp/tokens.read.tokens[0..768]`, gen 192, k 2.
GPU idle (5%, 412 MiB), no competing cargo/llama processes at run time; RAM free ≈ 19.2 GiB.

## The change under measurement (riir-infer, this landing)

The spec rewind (T5.1b) replaces the per-round `HybridStateSegment` snapshot/restore (wire-shaped bytes,
FULL buffers) with an **in-process exact checkpoint**: `SpecStage::checkpoint_state(cursor, ck)` /
`restore_checkpoint(ck)` over retained buffers — byte-exact copies, **live KV rows only** (bounded by the
runner's cursor: the ternary forward never advances `fill_pos`, so the cursor is the only live bound — the
cursor-overwrite law the NaN-poison pin proves), zero steady-state allocation. The wire codec stays the
cross-process handoff format (Issue 028) and the G1 exactness oracle.

**The GDN journal rank-1 unwind is NOT wired — deliberately.** The issue's ground rule is byte-exact
(the lane's lossless G1: the stream == sequential byte-for-byte, any drafter); the journal carries a
measured 6.3e-6 reconstruction drift (riir-infer Bench 885). An approximate rewind would give up the
defining property. The plan-623 T5.1b text naming the journal as the fast path is superseded by the
issue's ground rule — recorded in the plan row.

## Rung-2 mechanism table (example `spec_rewind_timing`, median of 20, head stage [0,63), cursor 600)

| mechanism | ms | vs step |
|---|---|---|
| codec snapshot (`encode_into`) | 34.49 | 0.24× |
| codec restore (`restore_into`) | 34.94 | 0.24× |
| ck snapshot (`checkpoint_state`) | **12.24** | **0.085×** |
| ck restore (`restore_checkpoint`) | **12.34** | **0.086×** |
| decode step (`step`) | 143.75 | 1× |

Volume: codec segment **631.5 MB** (15 attention layers × full 4096-row buffers + 48 GDN rec/conv) vs
ck live **221.8 MB** (600 live rows × kv_dim 1024 + GDN full state) — the stale-tail rows are neither
captured nor restored. ck is **2.8× faster than the codec**, proportional to volume (both memcpy-class
post rung 1). Synth control: ck 0.13/0.12 ms vs codec 0.28/0.31 ms — same shape at 5.6/2.8 MB.

## Rung-3 pair re-cell (`mtp_head_acceptance`, split 63, prompt 768, gen 192, k 2)

| lane | tok/s | E[m] | rounds |
|---|---|---|---|
| mtp_head (spec, ck-path) | **5.032** | **1.6754** | 114 |
| repeat_last (baseline) | 3.188 | 1.2484 | 153 |
| ordinary pair (incumbent) | 7.619 | — | 0 |

- **E[m] 1.6754 reproduces Bench 917's projected ≈1.68** (the same-token read; the drafter is untouched).
- End-to-end: the spec lane moves **0.86–0.95 → 5.03 tok/s ≈ 12×** at this posture; vs the ordinary pair
  it reads **0.66×** (was ≈0.06×). **The 70× snapshot tax is dead** (the rung-3 acceptance bar: O(1×) ✓).
- The remaining 0.66× is the honest structural overhead: per-round checkpoint + restore ≈ 24.5 ms ≈ 0.17
  steps, plus the rejected-draft re-forwards (the rewind semantics, not the checkpoint). On a CPU pair the
  wire RTT spec-fill hides is ~0.3 ms — near-zero headroom; on the LINKED posture (Tailscale/RDMA, the
  multi-host lane's target: one RTT ≈ 10s of ms, T2.6) K tokens per exchange is a real win again. The
  serving claim stays the drafter's to make (owner-gated promotion, unchanged).

## Gates

- **G1**: `tests/spec_fill_g1.rs` 12/12 — full-accept, partial (m=0 AND m≥1), worst-case, roll-drafter,
  MTP-drafter streams all == sequential oracle byte-for-byte **through the checkpoint path**. New pins:
  `spec_checkpoint_restore_matches_codec_restore_bit_exact` (ck == codec state, f32-bits, including the
  retained-buffer reuse arm) and `kv_cursor_overwrite_truncate_law_nan_poison` (NaN planted past the
  rewound cursor cannot leak — the cursor-overwrite truncate law, the plan's named T5.1b pin in its
  strongest form) + `spec_checkpoint_shape_refusals` (kind/shape/cursor refusals loud).
- **G3**: feature-off byte-identity untouched (the checkpoint lives behind the existing `multi_host`
  lane; no new feature — a straight upgrade of the rewind internals, G1-pinned both arms).
- **Alloc**: retained buffers, zero steady-state allocation in the rewind path (the ck buffers are built
  once and reused every round; restore is `copy_from_slice` only).
- Clippy: 0 warnings (`-p riir-infer-dist --all-targets`).

## Incidental findings (recorded, not fixed here)

- `mtp_head_acceptance` could not construct stages against the current artifact at all (the 256k-context
  unbounded allocation, >100 GiB commit) — `load_pair` now carries the dist_stage-style ctx clamp. Bench
  917's numbers predate the context bump (its harness would fail today); the E[m] row still reproduces.
- `multi_stream_g1::two_streams_overlap_serial_vs_interleaved` (the 0.85 overlap bar) read 1.012 under
  load AND failed in a pristine HEAD worktree the same way — a load-sensitive timing bar on this box,
  NOT this landing (passed on the immediate re-run; the known PASSED-ALONE class, recorded for the box).

Box-state disclosure: CPU-bound run on a desktop box under no concurrent compute load; single run per
cell (mechanism table: median of 20 in-process).
