# Bench 943 — Issue 1172: the GDN-bucket GPU-timestamp census — the non-GEMM share is real kernel wall, the GEMM share is bounded by its own launch floor

**Status:** LANDED (2026-11-04, zed-986a7538@amo). Instrument `4ea7adb` +
deep-queue marker-chain fix `95fea10`; cells below from the fixed build.
G1: `tests/qwen38_gdn_census_g1.rs` — sleep walls (NS_PER_TICK),
hazard-pinned sandwich vs a CPU-sync reference (ratio ~0.7–1.05, the
reference carries host submission+poll overhead the census lacks),
30-launch linearity, grouping + inter-chunk skip. 6/6 runs green.

## Cells (full promoted posture: DENSE_MM2D + Q6K + RELAXED + F16ACT + ATTN_TENSOR + GDN_CENSUS, gdn layer 0, event-chained markers)

Pass-level sanity: pp5120 **961/973 tok/s** vs bench 942's 971 — the
instrument is free-running (no distortion). Reconciliation is exact on
every cell: `chunks × section_sum + inter_chunk = gpu_span` (e.g. 10 × 8.8
+ 4400.4 = 4488.9).

| section (per GDN layer) | pp512 pass-2 ms | pp2560 steady ms | pp5120 steady ms | pp5120 share |
|---|---:|---:|---:|---:|
| input_norm | 0.16–0.19 | 0.75 | 0.97–0.98 | 11.1% |
| gemm_qkv_z_ab (4 GEMMs + cast) | 8.33–8.47 | 0.55 | 0.55–0.57 | 6.2% |
| beta_decay | 0.01 | 0.60 | 0.59 | 6.7% |
| conv1d (8 sub-launches) | 0.66–0.68 | 0.48–0.52 | 0.65–0.66 | 7.4% |
| expand_l2 | 0.73 | 0.68 | 0.69 | 7.8% |
| scan (multi-token recurrence) | 1.10–1.13 | 0.74–0.79 | 0.77 | 8.7% |
| norm_ssm | 0.07 | 0.53 | 0.56–0.61 | 6.4% |
| z_gate | 0.04 | 0.55 | 0.56 | 6.3% |
| ssm_out_gemm (+cast) | 0.45 | 0.48–0.50 | 0.50–0.51 | 5.6% |
| residual_gdn | 0.03 | 0.35 | 0.48–0.50 | 5.7% |
| mlp_norm_gateup (norm+cast+2 GEMMs) | 2.14–2.15 | 0.55–0.56 | 0.66–0.68 | 7.5% |
| mlp_act (SiLU gate) | 0.10–0.11 | 0.55–0.56 | 0.53–0.55 | 6.2% |
| mlp_down_gemm (+cast) | 1.25 | 0.80 | 0.68 | 7.7% |
| mlp_residual | 0.02 | 0.71–0.72 | 0.60 | 6.8% |
| **section sum / layer** | **15.1–15.3** | **9.2–10.0** | **8.8–8.9** | |

× 48 GDN layers: pp512 ≈ **730 ms/chunk**, pp5120 ≈ **425 ms/chunk** —
consistent with bench 942's ~400 ms/chunk bucket (StageTimers shares).

## The two headline findings

1. **The GEMM share is bounded by a per-dispatch floor, not arithmetic.**
   The 4 GEMMs read **0.55 ms TOTAL** at pp5120 — 3.4× less than bench
   941's standalone ~0.84 ms/EACH — while the pp512 (idle-queue) cell
   reads 8.33 ms for the same shapes (30× the pp5120 value). The GEMMs
   are NOT compute-bound at these shapes: pp5122's floor ≈ p·n·4 B /
   0.55 ms ≈ 1.25 TB/s — memory-bound at output-write bandwidth, and the
   pp512 cell pays dispatch/queue-idle, not FLOPs. **The "279 ms of pure
   GEMM wall" arithmetic is refuted**: in-pass GEMM wall ≈ 0.55 × 7 × 48
   ≈ **185 ms/chunk**, and even that is write-bandwidth-bound.
2. **The non-GEMM share is REAL, size-independent wall (~0.5–0.7 ms per
   small kernel, ~6–8% of the layer each).** The pp5120 steady sections
   sit at 0.48–0.77 ms across beta_decay / conv / expand / scan / norms /
   z_gate / residuals — while the SAME kernels at pp512 (idle queue) read
   0.01–0.11 ms. This is not kernel time scaling with size — it is the
   per-kernel **launch floor** (queue submit + hazard resolution + GEMM
   blit interference). 14 sections × ~0.6 ms ≈ 8.4 ms/layer ≈ 403
   ms/chunk ≈ the whole non-GEMM share bench 942 attributed.

**Verdict per class (the issue's plan):**
- scan: measured-adequate in isolation (0.77 ms = 8.7% of the layer wall;
  matches bench 938's ~6% of the pass). Priced-lever only if fused into
  the norm (see below), not on its own.
- conv, expand, beta_decay, z_gate, norms, residuals: each
  measured-adequate as kernels (µs-scale real work at pp512) — the cost
  is the launch floor they all share. The lever is **fewer launches**,
  not faster kernels: the fused norm+gating family (norm_ssm + z_gate;
  input_norm + cast; residual + cast) is the one pre-registerable shape
  (filed as issue 1173).
- GEMMs: measured-adequate (write-bandwidth-bound, floor-dominated). No
  kernel work is justified by this census; the half-B arithmetic ceiling
  is not the binding constraint in-pass.

## The instrument (what landed, and what it survived)

- `qwen38_gdn_census.rs`: `MTLCounterSampleBuffer` with the device's
  **"timestamp" counter set** (Apple Silicon validates a set-less buffer
  with "Invalid counter set size. Expecting 1 counter" — the set carries
  one counter, each resolved sample is ONE u64 of gpu ticks), marker
  passes via `compute_command_encoder_with_descriptor` +
  `sample_buffer_attachments` (start/end-of-encoder sample indices), 15
  marks per chunk closing 14 sections, reads once per ingest behind ONE
  `wait_until_completed`.
- Measured instrument laws (each pinned by the G1 or a bench cell):
  1. Start-of-encoder samples are GPU **schedule-time** readings; END
     samples are completions — boundaries are end-to-end.
  2. Unpinned (hazard-free) markers **overlap adjacent work** (a
     hazard-free marker slid ahead of queued rmsnorm work and read half
     the wall); markers carry ReadWrite hazards on the boundary buffers
     (prev section outputs + next inputs).
  3. The first dispatch of a fresh pipeline state pays ~1.5–3 ms of
     prepare: construction commits one throwaway warmup marker at
     reserved tail indices and blocks.
  4. **Deep-queue marker ordering needs a GPU-side chain**: under a
     multi-chunk queue the per-section attribution ROTATED one window per
     chunk (pp2560/pp5120, pre-`95fea10`) even though commit order is
     synchronous — GPU scheduling of independent micro-passes is not
     commit-serial. Every marker now encodes
     `wait_for_event(chain, k)` / `signal_event(chain, k+1)`: windows are
     completion-to-completion at any queue depth (pp2560 groups 2–5 and
     both pp5120 passes are rotation-free and run-to-run stable).
- CPU-side alternatives are refuted for this shape: `StageTimers`
  per-stage syncs serialize the queue (bench 940's stall), unsynced CPU
  deltas measure enqueue only.

## Provenance

- Box: amo (M5 Ultra, macOS 27.2), AC power, load ~1.1, no other compute
  consumer (`ps` checked), therm clean (`pmset -g therm`). Release build
  `95fea10`, `/tmp/i1172-target` isolated target dir.
- Posture: `RIIR_QWEN38_DENSE_MM2D=1 RIIR_QWEN38_LMHEAD_Q6K=1
  RIIR_QWEN38_DENSE_MM2D_RELAXED=1 RIIR_QWEN38_DENSE_MM2D_F16ACT=1
  RIIR_QWEN38_ATTN_TENSOR=1 RIIR_QWEN38_GDN_CENSUS=1` (all promote-only
  exact literals; census diagnostic-only, env-off = byte-identical).
- Pre-943 bench arc: 940 (dense relaxed 2×) → 941 (half-B) → 942
  (tensor-flash attention, past the fork) → **943 (this)**.
