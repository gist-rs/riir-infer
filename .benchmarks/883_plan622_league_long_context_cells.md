# Bench 883 — Plan 622 League row: the long-context capability cells (pp16384/pp32768 si-ON vs flag-off, chunked-4096 posture)

**Date:** 2026-10-13, ~18:20–19:10 +07 · 4090 (Windows, RTX 4090 24 GB, `E:\git\riir-ai` @ origin/develop + this lane's bench amendment, `E:\git\riir-infer` @ `e2d3465` clean).
**Unit:** riir-refine idle loop cycle 55 (4090 lane) — Plan 622 Phase 7 League row (`katgpt-rs/.plans/622_scale_invariant_attn.md` L82): "record pp16384/pp65536 cells + train-length provenance column (standing law from riir-train 449)".

## Verdict

**G2 PASSES at both feasible cells — the si arm stays perf-neutral within ±3% on the long-context chunked posture. The 65536 cell is REFUSED BY GATE (a capability boundary, not a measurement), and 49152 OOMs the card — the record carries the honest boundary.**

| Cell (chunked-4096) | flag-off (incumbent) | si-ON (calibrated, τ=10) | Δ | Gate ±3% | train_len provenance |
|---|---|---|---|---|---|
| pp16384 | **3393 tok/s** (rounds 3393/3393/3388; pins GREEN) | **3366 tok/s** (3366/3362/3368; ramp wire green) | **−0.80%** | PASS | 262144 (GGUF `qwen35.context_length`) |
| pp32768 | **2941 tok/s** (2946/2941/2938; pins GREEN) | **2907 tok/s** (2914/2907/2907; ramp wire green) | **−1.16%** | PASS | 262144 (GGUF `qwen35.context_length`) |
| pp49152 | — OOM (see boundary) | — | — | — | 262144 |
| pp65536 | — REFUSED by Issue-994 KV gate | — | — | — | 262144 |

Both cells are IN-WINDOW for this checkpoint (16384, 32768 ≪ 262144) — the si
paper's beyond-train-length extrapolation axis is NOT exercised here by
construction; that claim lives in the trained track (riir-train Plan 449), per
the standing law that made the provenance column mandatory.

## The posture amendment (disclosed)

`bench_772_t0_gemm_share` previously forced `set_prefill_chunk_max(p.max(4096))`
— single chunk. At p=16384 that posture OOMs the 24 GB card (the P-width
gate/up/hid mlp scratch alone ≈ 6.3 GB; measured: allocation of 6,328,680,448 B
refused with 2.1 GB attention KV + ~8 GB weight mirrors resident). The
amendment — this lane's commit — caps the chunk width at the SHIPPING default:

```rust
set_prefill_chunk_max(p.min(4096));
```

- **p ≤ 4096: bit-identical** to the historical league posture (2048/4096 stay
  single-chunk; every existing pin untouched — the pp2048/pp4096 league cells
  of Bench 982 are unaffected).
- **p > 4096: chunked at the shipping 4096 width** — the Bench-710 capability
  path (GDN state + KV carried across chunk boundaries; the cudarc lane is
  chunk-aware per `prefill_tokens_chunk`'s base_pos/is_final_chunk contract).

## New pins registered (folded lane, q8_v10t route)

Captured `B772_CAPTURE=1` × 2 processes × (warm + 3 rounds) each — argmax/FNV
identical across every round of both processes:

- `(16384, q8_v10t)`: argmax **95759**, FNV **a047b42825a1e21c**
- `(32768, q8_v10t)`: argmax **72**, FNV **7a88a6ae357c9350**

Scored flag-off runs assert these GREEN (warm + every round). The si-ON runs
are their own deterministic posture (armed outputs differ by design — the
bench-878 law; e.g. pp16384 si warm argmax 248046/FNV b074a34ff396ce7c,
pp32768 si warm argmax 1739/FNV cd166a6bc3884ac0).

## The capability boundary (both honest refusals on record)

- **pp65536 — REFUSED at construction by the Issue-994 single-buffer KV gate**:
  per-layer KV at block_size 65536 = 0.25 GiB > the bisected 0.20 GiB safe
  bound. The ≥52K-position class is KNOWN broken on the cubecl/wgpu stack
  (block 57,408 lost the device during construction; block 65,472+ produced
  DETERMINISTIC silent corruption, cos 0.419531 / top-1 0.0150, zero
  allocation failures — riir-ai Issue 994). Raising `RIIR_GPU_KV_LAYER_MAX_BYTES`
  without a fresh bisect + external-reference validation is exactly what the
  gate forbids; a perf number on a silently-corrupting path is worthless. The
  gate's own safe ceiling is block_size ≤ 53248.
- **pp49152 — passes the 994 gate (per-layer 0.19 GiB) but OOMs total VRAM**:
  attention KV 6.44 GB + ~8 GB weight mirrors + graph workspace + the
  1.58 GB chunk-width mlp scratch do not fit 24 GB (allocation failures at
  98 MB scale — the card was full). The top FEASIBLE capability cell on this
  card/stack is pp32768.

## Arm wiring + box state

- Same ONE-build contract as Bench 878: `SI620_G2=1` + `SI620_TAU=10` +
  `SI620_CAL=E:/tmp/si622-cal-4090.bin` (sidecar digest `4d8f95efd6b0ccb2…` —
  identical to Bench 878's cell; `PREFILL ARMED tau=10 calibrated=true
  block={16384,32768} layers=64 heads=24` on the log line).
- Feature set (all cells): `cuda_graphs_forward,bonsai2_hadamard,
  scale_invariant_attn,ternary_gemm_batched,cubecl_runtime,ternary_gemv,
  ternary_deltanet_chunked_prefill,prefill_q8_act,prefill_mmq_v2`.
- Model: `Ternary-Bonsai-2-27B-PQ2_0.gguf` (7,206,168,928 B, FOLDED
  prism.hadamard) — the league file. Train-length provenance read from the
  GGUF header directly: `qwen35.context_length = 262144`, rope freq_base 1e7,
  64 blocks / 24 heads / kv 4 / k+v length 256 / 16 attention + 48 GDN layers.
- Protocol: Bench-976 amended — `B772_ROUNDS=3`, `B772_WARM_SECONDS=8`
  clock-warm burst (1–2 unscored rounds), ramp tripwire armed (green every
  cell; worst last-two spread 0.24% capture / 0.15% scored), pins asserted
  warm + per round on every flag-off cell.
- GPU exclusivity: the in-harness Issue-834 gate armed all runs; nvidia-smi
  beside the cells: 4–5% / 427–461 MiB idle between (48 °C, 210 MHz idle
  clock); no cargo.exe at any cell window. One earlier build attempt died to
  the known flaky-Windows-rustc class (0xc0000005 on cubecl-opt; the
  bench-878 precedent) — the resume re-run built clean, no code involvement.

## Consequences

- **Plan 622's League row CLOSES**: the pp16384/pp32768 cells recorded with
  the mandatory train-length provenance column; pp65536 is refused-by-gate
  with the Issue-994 citation (the honest capability boundary — 64K prefill
  on this stack is a KV-gate fix away, tracked in riir-ai Issue 994, not a
  number away).
- The G2 axis is now complete across the length ladder: 2048 (−0.32%, Bench
  878) · 4096 (league re-pin, Bench 982 flag-off posture) · 16384 (−0.80%)
  · 32768 (−1.16%) — si armed never exceeds −1.2% at any measured length.
- The promotion decision's four axes are now ALL measured: G1 PASS (932 +
  Lean 53-theorem gate), G2 PASS (878 + this bench), G3 FAIL (936 CPU / 873
  GPU / 881 τ-sweep at EVERY swept τ, both arms), G4 PASS (932). The
  pre-registered conditional — default-on ONLY if G1–G4 all pass — resolves:
  **no promotion, `scale_invariant_attn` stays opt-in**; the remaining value
  is the track-B training question (riir-train Plan 449).
