# Bench 945 — issue 1174: the free-running ablation matrix — the wall is
# GEMM-owned (61.5%), the census non-GEMM share was marker-boundary cost

**Status:** RECORD — the 1174 instrument landed and ran; the 942/943
attribution question is closed. ⚠ Supersedes the per-kernel reading of
the 943 census section table (small-kernel sections are boundary
artifacts; the caveat rows on 942/943 point here).

## Posture + provenance

- Box: **amo** (M5 Ultra, 30 cores, 96 GB), macOS 27.2, **AC power**,
  load 1.45-1.85 at record time; GPU quiet at launch (`ps` census: no
  cargo/riir compute consumers; only the idle `riir-rethink serve` +
  `fleet-web` listeners — recorded, not a guaranteed exclusive window).
- Build: `e59d59d` (the 1174 instrument), release,
  `--features qwen38_wgpu,metal_tensor_gemm`.
- Posture: the promoted league env set (942/943/944 baseline) —
  `RIIR_QWEN38_DENSE_MM2D=1 RIIR_QWEN38_LMHEAD_Q6K=1
  RIIR_QWEN38_DENSE_MM2D_RELAXED=1 RIIR_QWEN38_DENSE_MM2D_F16ACT=1
  RIIR_QWEN38_ATTN_TENSOR=1`; census/stage-timers/FUSE_*/GDN_CHUNKED
  all absent (the ablation refuses them at construction).
- Cell: pp5120 (10 chunks × 512), 28 cells × 3 interleaved (base,
  ablated) pass pairs, one model load (~16.8 GB GGUF). Log:
  `/tmp/ablate_matrix_1174.log` (full pair lines; the table below is
  the printed medians).

## The instrument

`qwen38_ablate.rs` + 26 gated launch sites
(`e59d59d`): arm a class to launch NOTHING, diff the pass wall — no
markers, no syncs, no hazards anywhere (the 942 StageTimers and the
943 census both serialize; 944 proved their walls are boundary
artifacts). Skips are geometry-preserving (a skipped f16 cast keeps the
twin handle; `attn_scores` skips BOTH arms; refusals pin the incumbent
launch set). The bench drives 3 interleaved base/ablated pairs per
class — sequential arms measure the box, never the kernel. 4 unit
tests incl. the census-mapping drift pin.

## The table (ms per chunk, median of 3 pair deltas)

Base wall median **526.3 ms/chunk (973 tok/s)** — reproduces bench
944's baseline (~971) exactly; the instrument is free-running.

| class | Δ ms/chunk | share of wall | per-layer instance |
|---|---:|---:|---:|
| mlp_gateup_gemm | 138.45 | 26.3% | 2.163 ms |
| mlp_down_gemm | 76.81 | 14.6% | 1.200 ms |
| gemm_qkv_z_ab | 61.97 | 11.8% | 1.291 ms |
| scan | 54.41 | 10.3% | 1.134 ms |
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
| mlp_act | 3.38 | 0.6% | 0.053 ms |
| wte_upload | 3.53 | 0.7% | once per chunk |
| z_gate | 1.17 | 0.2% | 0.024 ms |
| kv_cache_fill | 1.28 | 0.2% | 0.020 ms |
| mlp_residual | 0.78 | 0.1% | 0.012 ms |
| lm_head_gemm | 2.62 | 0.5% | once per chunk |
| attn_residual | 0.48 | 0.1% | 0.030 ms |
| tail_norm | 0.48 | 0.1% | once per chunk |
| input_norm | −0.93 | ≈0 | — (noise) |
| split_qg | −3.44 | ≈0 | — (noise) |
| mlp_norm | −2.41 | ≈0 | — (noise) |
| rope | −1.05 | ≈0 | — (noise) |
| residual_gdn | −0.03 | ≈0 | — (noise) |
| **ALL (floor)** | **527.50** | — | 27.5 ms/chunk wall |

## The four findings

1. **The per-class wall contributions are approximately ADDITIVE: Σ
   classes = 506.4 ms ≈ 96.2% of the 526.3 ms wall.** The free-running
   wall decomposes nearly cleanly into one-class-at-a-time removal
   deltas; the ~20 ms residue is dispatch/gaps + the per-chunk argmax
   syncs. (The first run's printout reported Σ = 1033.9 = 196.5% — that
   was a **reporting bug**: the Σ included the ALL floor row. Fixed in
   the bin same-day; read the classes-only Σ from the table.)
2. **The wall is GEMM-owned.** The GEMM family (mlp_gateup + mlp_down +
   gemm_qkv + ssm_out + attn_qkv + attn_out) = **323.7 ms/chunk =
   61.5%**; the MLP pair alone (gateup + down) = 215 ms = 41%. The GDN
   dataflow trio (scan + conv1d + expand_l2) = 127 ms = 24%.
   Everything else — the entire small-kernel family the census priced
   at ~0.5-0.7 ms/section — is ≤ ~1-2% of the wall.
3. **The 943 census's non-GEMM readings are marker-boundary artifacts,
   now quantified per section.** The census read ~0.48-0.77 ms for
   every small section and concluded "the non-GEMM share is REAL
   (~0.6 ms × 14 sections × 48 layers ≈ 403 ms/chunk)". The ablation
   reads the SAME sections at 0.0-0.1 ms/layer free-running (β_decay
   0.096, norm_ssm 0.075, z_gate 0.024, residual ≈ 0, mlp_act 0.053,
   input_norm ≈ 0) — a 5-50× inflation, and the census's per-marker
   pair overhead (~0.5 ms) is exactly the gap. The REAL non-GEMM wall
   is scan + conv + expand + β_decay ≈ 131.5 ms/chunk (25%), not 403.
   The census's own G1 honesty (reconciliation exact; pass wall
   reproduced) remains true — the instrument reproduced the wall while
   misattributing its interior; boundary-heavy instruments do that.
4. **The launch floor is 27.5 ms/chunk (5.2%)** — the ALL cell's
   remaining wall (dispatch + 10 argmax sync/reads). Confirms 944's
   0.02-0.05 ms/launch estimate at pass scale and closes the
   launch-floor question with a number: even zero kernels cannot buy
   more than ~5% back.

## The census-vs-ablation section table (per GDN layer, pp5120)

| section | 943 census | 945 ablation | reading |
|---|---:|---:|---|
| input_norm | 0.97-0.98 | ≈0 | boundary artifact |
| gemm_qkv_z_ab | 0.55-0.57 | 1.291 | census LOW (isolated) — ablation includes freed contention |
| beta_decay | 0.59 | 0.096 | boundary artifact |
| conv1d | 0.65-0.66 | 0.725 | agree |
| expand_l2 | 0.69 | 0.785 | agree |
| scan | 0.77 | 1.134 | ablation higher (contention + serial critical-path node) |
| norm_ssm | 0.56-0.61 | 0.075 | boundary artifact |
| z_gate | 0.56 | 0.024 | boundary artifact |
| ssm_out_gemm | 0.50-0.51 | 0.451 | agree |
| residual_gdn | 0.48-0.50 | ≈0 | boundary artifact |
| mlp_norm_gateup | 0.66-0.68 | −0.04 + 2.16 (norm + gateup GEMMs) | census LOW on the GEMM — the gateup GEMM's wall contribution includes the bandwidth contention it imposes on its overlap window |
| mlp_act | 0.53-0.55 | 0.053 | boundary artifact |
| mlp_down_gemm | 0.68 | 1.200 | census LOW (contention-inclusive) |
| mlp_residual | 0.60 | 0.012 | boundary artifact |
| **section sum** | **8.8-8.9** | **≈8.0 (chain)** | the sums agree — the interiors did not |

The layer-level sums agreeing while the interiors disagree is the
finding's shape: the census reproduced the wall (its G1), priced the
big kernels roughly right, and burned ~2.5 ms/layer of marker-boundary
cost on the small sections — which it then read as "the non-GEMM share
is REAL". 944's suspicion, now measured.

## Instrument laws (measured)

- **Deltas are wall contributions, not isolated execution times.** For
  big bandwidth classes the removal delta includes the freed contention
  (mlp_gateup reads 2.16 ms/layer where its isolated time is ~0.7) —
  the honest unit for "what does removing this class buy", which is
  the lever-relevant one.
- **Noise floor ±3-5 ms/chunk** (the negative rows; consistent with
  944's spread). Classes below it read ≈0 — the small-kernel family's
  total is genuinely ≤ ~1-2%.
- **The ALL row is a floor, not an additive class** — never Σ it
  against the wall (the reporting bug above).
- **The refusals are load-bearing**: census/stage-timers/FUSE_*/GDN-
  CHUNKED each change the launch set or serialize; ablation with any
  of them armed refuses loud at construction.

## Verdict

**GOAT G1 (attribution): PASS** — Σ classes 96.2% of wall, base
reproduces 944's baseline, layer sums reconcile with the census. No
optimization lever is claimed here (this record only attributes); the
pre-registered consequence stands: **the next lever is GEMM-class**
(mlp gateup+down = 41% of the wall, already half-B in this posture —
kernel geometry/weight-layout work), and the launch-reduction doctrine
is closed as a lever family (fusions ≤1% measured twice: 944 + here).

## Provenance

- Issue 1174 (closed; record in `HISTORY.md`), instrument commit
  `e59d59d`, claim `d6b4f85`.
- Bench 943 (the census being re-read), bench 944 (the refutation +
  the free-running launch-cost estimate), bench 942 (the StageTimers
  non-GEMM share being corrected).
- Log: `/tmp/ablate_matrix_1174.log` (ephemeral; the table above is
  the record).
