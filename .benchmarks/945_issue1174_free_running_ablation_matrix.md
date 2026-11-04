# Bench 945 — issue 1174: the free-running ablation matrix — the wall is
# GEMM-owned (61%), the census non-GEMM share was marker-boundary cost

**Status:** RECORD — the 1174 instrument landed and ran (matrix + a
confirmation run); the 942/943 attribution question is closed. ⚠
Supersedes the per-kernel reading of the 943 census section table
(small-kernel sections are boundary artifacts; the caveat rows on
942/943 point here). Reviewed: Claude verdict round 1 REVISE (five
revisions, all applied — see the last section), round 2 pending.

## Posture + provenance

- Box: **amo** (M5 Ultra, 30 cores, 96 GB), macOS 27.2, **AC power**;
  GPU quiet at both launches (`ps` census: no cargo/riir compute
  consumers; only the idle `riir-rethink serve` + `fleet-web`
  listeners — recorded, not a guaranteed exclusive window).
- Build: matrix run at `e59d59d`; confirmation run at the cast-fix
  commit (see below), release, `--features qwen38_wgpu,metal_tensor_gemm`.
- Posture: the promoted league env set (942/943/944 baseline) —
  `RIIR_QWEN38_DENSE_MM2D=1 RIIR_QWEN38_LMHEAD_Q6K=1
  RIIR_QWEN38_DENSE_MM2D_RELAXED=1 RIIR_QWEN38_DENSE_MM2D_F16ACT=1
  RIIR_QWEN38_ATTN_TENSOR=1`; census/stage-timers/FUSE_*/GDN_CHUNKED
  all absent (the ablation refuses them at construction).
- Cells: pp5120 (10 chunks × 512), 3 interleaved (base, ablated) pass
  pairs per class, one model load per process. Matrix: 28 cells; the
  confirmation re-ran `scan` + `mlp_gateup_gemm` + the ALL cell
  (reproduction + the gateup cast fix). Logs: `/tmp/ablate_matrix_1174.log`,
  `/tmp/ablate_confirm_1174.log` (ephemeral; the tables are the record).

## The instrument

`qwen38_ablate.rs` + 26 gated launch sites (`e59d59d`): arm a class to
launch NOTHING, diff the pass wall — no markers, no syncs, no hazards
anywhere (the 942 StageTimers and the 943 census both serialize; 944
proved their walls are boundary artifacts). Skips are
geometry-preserving (a skipped f16 cast keeps the twin handle — and,
after the verdict fix, a skipped GEMM class no longer skips its input
cast); `attn_scores` skips BOTH arms; the refusals pin the incumbent
launch set. The bench drives interleaved base/ablated pairs —
sequential arms measure the box, never the kernel. 4 unit tests incl.
the census-mapping drift pin.

## The matrix table (ms per chunk, median of 3 pair deltas)

Base wall median **526.3 ms/chunk (973 tok/s)** — reproduces bench
944's ~971 baseline, in BOTH runs; the instrument is free-running.

| class | Δ ms/chunk | share of wall | per-layer instance |
|---|---:|---:|---:|
| mlp_gateup_gemm | 135.56 ᵈ | 25.8% | 2.157 ms |
| mlp_down_gemm | 76.81 | 14.6% | 1.200 ms |
| gemm_qkv_z_ab | 61.97 | 11.8% | 1.291 ms |
| scan | 54.41 ᵇ | 10.3% | 1.134 ms |
| expand_l2 | 37.68 | 7.2% | 0.785 ms |
| conv1d | 34.81 | 6.6% | 0.725 ms |
| attn_scores | 25.06 | 4.8% | 1.566 ms |
| ssm_out_gemm | 21.63 | 4.1% | 0.451 ms |
| attn_qkv_gemm | 16.74 | 3.2% | 1.046 ms |
| f16_cast | 13.20 | 2.5% | 0.206 ms |
| attn_out_gemm | 8.05 | 1.5% | 0.503 ms |
| beta_decay | 4.62 | 0.9% | 0.096 ms |
| norm_ssm | 3.60 | 0.7% | 0.075 ms |
| qk_norm | 3.50 | 0.7% | 0.055 ms |
| wte_upload | 3.53 | 0.7% | once per chunk |
| mlp_act | 3.38 | 0.6% | 0.053 ms |
| lm_head_gemm | 2.62 | 0.5% | once per chunk |
| kv_cache_fill | 1.28 | 0.2% | 0.020 ms |
| z_gate | 1.17 | 0.2% | 0.024 ms |
| mlp_residual | 0.78 | 0.1% | 0.012 ms |
| attn_residual | 0.48 | 0.1% | 0.030 ms |
| tail_norm | 0.48 | 0.1% | once per chunk |
| input_norm | −0.93 | ≈0 | — (noise) |
| split_qg | −3.44 | ≈0 | — (noise) |
| mlp_norm | −2.41 | ≈0 | — (noise) |
| rope | −1.05 | ≈0 | — (noise) |
| residual_gdn | −0.03 | ≈0 | — (noise) |
| **ALL (floor)** | see the floor note | — | **28 ms/chunk wall, measured directly** |

ᵈ = the corrected value (the matrix run read 138.45 with a cast
double-count; the confirmation re-read 135.56 after the fix — the
delta is the per-layer input cast, now counted only by `f16_cast`).
ᵇ = reproduced in the confirmation run: 55.66 vs 54.41 (within noise).

## The findings

1. **The per-class wall contributions are additive within noise — the
   correct check is Σ single-class vs JOINT removal.** Σ classes =
   503.5 ms (with the corrected gateup); joint removal = base median
   526.3 − floor 28 = 498.3 ms. Δ = +1.0% — inside the noise spread.
   One-at-a-time removal sums to what removing everything buys, which
   is the non-trivial result: no hidden overlap penalty, no
   double-counting between classes. (The matrix printout's "Σ =
   1033.9 ms = 196.5%" was a **reporting bug** — the Σ included the
   ALL row. Fixed in the bin the same day; read the classes-only Σ.)
2. **The wall is GEMM-owned.** The GEMM family (gateup + down + qkv +
   ssm_out + attn_qkv + attn_out) = **321 ms/chunk ≈ 61%**; the MLP
   pair alone (gateup + down) = 212 ms = 40%. The GDN dataflow trio
   (scan + conv1d + expand_l2) = 127 ms = 24% (GDN layers only).
   Pass-wide non-GEMM = wall − GEMMs ≈ 205 ms (39%), of which
   attn_scores 4.8% + f16_cast 2.5% + the small-kernel family (net
   ~18 ms, 3.4%; every individual small class ≤ 1%).
3. **The 943 census's non-GEMM readings are marker-boundary artifacts,
   now quantified per section.** The census read ~0.48-0.77 ms for
   every small section and concluded "the non-GEMM share is REAL
   (~0.6 ms × 14 sections × 48 layers ≈ 403 ms/chunk)". The ablation
   reads the SAME sections at 0.0-0.1 ms/layer free-running (β_decay
   0.096, norm_ssm 0.075, z_gate 0.024, residual ≈ 0, mlp_act 0.053,
   input_norm ≈ 0) — a 5-50× inflation, and the census's ~0.5 ms
   per-marker-pair overhead is exactly the gap. The census's own G1
   honesty (reconciliation exact; pass wall reproduced) remains true —
   the instrument reproduced the wall while misattributing its
   interior; boundary-heavy instruments do that.
4. **The launch floor is 28 ms/chunk (5.3%), measured directly.** The
   ALL cell's ablated pass ran a steady 0.27-0.28 s in both runs —
   that IS the floor (dispatch + the 10 per-chunk argmax sync/reads;
   near-zero kernels). ⚠ **The ALL cell's pair DELTAS are
   drift-inflated and must not be read as the floor**: in BOTH runs
   the cell's pair-0 base pass was normal (5.25-5.26 s) and pairs 1-2
   drifted to 5.52-5.57 s (+5.5%, identical pattern). Inference
   (labeled, not proven): the near-idle ablated pass lets the GPU
   clock drop, inflating the next base pass until sustained load ramps
   it back. The pair-0 delta (497.4 ms, normal-clock base) agrees with
   the Σ-vs-joint check above; the drifted pairs would over-read by
   ~30 ms. Confirms 944's 0.02-0.05 ms/launch estimate at pass scale
   and closes the launch-floor question with a number: even zero
   kernels cannot buy more than ~5% back.

## The census-vs-ablation section table (per GDN layer, pp5120)

| section | 943 census | 945 ablation | reading |
|---|---:|---:|---|
| input_norm | 0.97-0.98 | ≈0 | boundary artifact |
| gemm_qkv_z_ab | 0.55-0.57 | 1.291 | census LOW — see the mechanism note |
| beta_decay | 0.59 | 0.096 | boundary artifact |
| conv1d | 0.65-0.66 | 0.725 | agree |
| expand_l2 | 0.69 | 0.785 | agree |
| scan | 0.77 | 1.134 | ablation higher |
| norm_ssm | 0.56-0.61 | 0.075 | boundary artifact |
| z_gate | 0.56 | 0.024 | boundary artifact |
| ssm_out_gemm | 0.50-0.51 | 0.451 | agree |
| residual_gdn | 0.48-0.50 | ≈0 | boundary artifact |
| mlp_norm_gateup | 0.66-0.68 | −0.04 + 2.157 (norm + gateup GEMMs) | census LOW on the GEMM — see the mechanism note |
| mlp_act | 0.53-0.55 | 0.053 | boundary artifact |
| mlp_down_gemm | 0.68 | 1.200 | census LOW — see the mechanism note |
| mlp_residual | 0.60 | 0.012 | boundary artifact |
| **section sum** | **8.8-8.9** | **≈8.0 (chain)** | the sums agree — the interiors did not |

The layer-level sums agreeing while the interiors disagree is the
finding's shape: the census reproduced the wall (its G1), priced the
dataflow kernels roughly right, and burned ~2.5 ms/layer of
marker-boundary cost on the small sections — which it then read as
"the non-GEMM share is REAL". 944's suspicion, now measured.

## Why the GEMM sections disagree in BOTH directions (mechanism note)

The ablation's GEMM deltas exceed the census's GEMM sections by
0.6-1.5 ms/layer. **Verified in source**
(`gemm_dense_f16_metal_zero_copy.rs`): every dense-GEMM dispatch does
more than the matmul — a `client.flush()` (closes CubeCL's pending
encoder), then its OWN `MTLCommandBuffer`, the kernel writing a
DEDICATED output buffer, and a **GPU→GPU blit copy-back of the full
P×M f32 output** into the CubeCL pool, then commit. Skipping a GEMM
class therefore removes flush + command-buffer + kernel + blit +
commit per instance (gateup: 2 outputs × 512 × mlp_hidden × 4 B ≈
tens of MB per layer of blit alone). Byte-counted, the blits are
~0.05-0.2 ms/layer — a real component, not the whole gap. The census
side shows a **time-moved pattern**: wherever a GEMM section reads
low, the NEXT small section reads high by a matching amount (mlp_down
−0.52 ↔ mlp_residual +0.59; gemm_qkv −0.74 ↔ beta_decay +0.49;
gateup ~−1.5 ↔ mlp_act +0.48) — consistent with the GEMM's tail work
(flush/commit/blit completion) landing in the next census section.
The residue is honestly UNSETTLED by this instrument; candidates:
per-dispatch command-buffer submit overhead, queue overlap contention,
census tail-attribution shift. **Doctrine consequence:** the MLP 41%
share may include zero-copy-wrapper cost, not only matmul — the
discriminating next cell is a direct-store GEMM arm (the
`dispatch_gemv` pattern already writes straight to the pool buffer;
a GEMM variant doing the same would split kernel vs wrapper). Until
that cell runs, read the MLP share as "GEMM-class work", not
"GEMM-kernel time". (Adjacent to the 1167 prefill lane — not filed
here to avoid duplicating a foreign lane.)

## Instrument laws (measured)

- **Deltas are wall contributions** (one-class-at-a-time removals on a
  fixed dependency chain), validated additive against joint removal
  (Δ +1.0%). For the GEMM classes the contribution includes the
  removed wrapper work (flush/cmdbuf/blit) — the lever-relevant unit
  ("what does removing this class buy"), but not the kernel's
  execution time in isolation.
- **Noise: within-class spread reaches ~7 ms/chunk at 3 pairs**
  (observed: lm_head 2.6-9.2, mlp_act 2.0-7.5). The ±3-5 ms figure of
  bench 944 is the cross-cell spread; per-cell pair spreads are wider.
  Negative rows sit inside it and read ≈0.
- **The ALL row is a floor, never an additive class** — and its pair
  deltas are drift-inflated (see finding 4); read the floor as the
  ablated wall measured directly.
- **The refusals are load-bearing**: census/stage-timers/FUSE_*/GDN-
  CHUNKED each change the launch set or serialize; ablation with any
  of them armed refuses loud at construction.

## Verdict

**G1 (attribution): PASS** — base reproduces 944's baseline twice;
Σ-vs-joint +1.0%; the floor cell and the scan/gateup cells reproduce
across processes. No optimization lever is claimed by this record; it
only attributes. The pre-registered consequence stands with one
sharpening: **the only lever families above noise are GEMM-class
(~61%) and the GDN dataflow trio (24%, GDN layers)**; launch reduction
is closed as a lever family (measured twice: 944 + here; the entire
small-kernel family is 3.4% net). Before the MLP share is treated as
a GEMM-KERNEL lever, the direct-store cell must split kernel vs
wrapper (the mechanism note).

## The verdict-flow record (Claude, round 1 — REVISE, applied)

1. ALL-row drift: disclosed (finding 4); floor re-read as the direct
   ablated wall; the additivity check re-stated as Σ vs joint.
2. The "bandwidth contention" explanation: replaced by the verified
   wrapper mechanism (flush + dedicated cmdbuf + blit — checked in
   `gemm_dense_f16_metal_zero_copy.rs`, not taken on faith) + the
   census time-moved pattern; the residue marked unsettled.
3. Gateup-site double-count: the input cast moved outside the gateup
   gate (code fix, this commit); gateup re-read at 135.56 (the ~3 ms
   delta vs 138.45 is the removed cast — measured confirmation).
4. "≤1-2%" family share: corrected (net smalls 3.4%, every class ≤1%,
   plus attn_scores/f16_cast named); non-GEMM scopes labeled (GDN-only
   trio vs pass-wide).
5. Noise: the observed within-class spread (~7 ms) quoted instead of
   the cross-cell ±3-5 ms.

## Provenance

- Issue 1174 (closed; record in `HISTORY.md`), instrument commit
  `e59d59d`, claim `d6b4f85`, record `9a7476a`, revisions this commit.
- Bench 943 (the census being re-read), bench 944 (the refutation +
  the free-running launch-cost estimate), bench 942 (the StageTimers
  non-GEMM share being corrected).
