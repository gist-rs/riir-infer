# Bench 949 — Issue 1167 rung C: the GEMM-staging price cell (round 3) — launch-weighted barrier projection ~50 ms/frame (~10% e2e fence-isolated): OVER the bar — the double-buffer twin graduates to a G1-gated variant + a paired census A/B as the decider

**Status:** RECORD (round 3) — price cell complete across three runs; the round-2 "UNDER the bar" closure arithmetic is **REFUTED** (its slow-window walls were polluted by a co-tenant GPU job; three clean windows agree at the 1.0–2.0 ms class). The launch-weighted projection over ALL ternary GEMM classes (FFN + GDN + attn, fresh GDN-shape rows with Δ-IQR) reads **~50 ms/frame fence-isolated ≈ 10% e2e — OVER the 1.5% bar**: per the issue rule, the double-buffered-staging variant graduates to a G1-gated correctness twin and a paired frame-census A/B as the decider. H5-device-f16 stays refuted on resident footprint (the A-side f16 mirror is 8× the packed-bit bytes — 27B elements → ~54 GB, defeating the ternary tier's reason to exist; the ~800 GB/s-class traffic DELTA numbers are modelled, not measured, and are moot next to the footprint argument). Register-feeding dequant into `matmul2d` operand registers is not expressible — matmul2d consumes `tensor` views (device/threadgroup memory only), the same shape llama.cpp's `kernel_mul_mm` uses.
**⚠ Round-2 closure retracted:** the first landing of this record proposed closing rung C at 0.8–1.2% e2e from the 13:55-window walls (ffn_gate/up 4.267 ms, ffn_down 5.108 ms). Those walls are ~2.9× the clean-window class (1.4–2.0 ms, reproduced in three independent windows: 14:29 full-suite, 14:31 standalone, 14:34 standalone) and the round-2 twin walls SUMMED to ~873 ms/frame vs the census's 281 ms FFN strip — physically impossible for the same kernels. Cause: the co-tenant `plm_gpu_backward_parity` GPU job (the train sibling's rig lane) was running during the 13:39–13:55 window; rig status listed it as amo heavy. Nobody should quote the round-2 share numbers.
**Date:** 2026-11-05
**Session:** zed-fcebab50@amo, 1793856517
**Issue:** [`.issues/1167_metal_prefill_followup_lane_b_c.md`](../.issues/1167_metal_prefill_followup_lane_b_c.md) (rung C, re-scoped by Bench 937)
**Box:** amo (M5 Ultra, 30 cores, 96 GB), AC, load 2.45–2.49 (settled; a syspolicyd burst above 3 preceded the run and had drained), no GPU compute consumer — the slot was sequenced with the co-tenants (a387ce2c's bench_1156 1164 battery, then zed-83e79a9e's 1177 Item-2 cell, via agent-mail handshake; both drained before this run).
**Model:** synthetic bit-plane fixtures at the three Bonsai-2 prefill bottleneck shapes (the real-model frame census is Bench 937 — this cell prices the KERNEL, and the fixture laws below pin the decode).
**Instrument:** `crates/riir-infer-gpu/tests/issue1167_rung_c_staging_price.rs` (G1 `2.682e-7` worst_rel vs the CPU decode; commits `c5fba2e`, `cb00077`, `e7b98d1`).
**Build:** release, `CARGO_TARGET_DIR=/tmp/i1167-target`, `--features metal_tensor_gemm,cubecl_runtime,ternary_gemv,multi_host,ternary_gemm_batched,ternary_gemm_simdgroup,ternary_deltanet_chunked_prefill,ternary_attention_batched_prefill,riir-infer-core/bonsai2_hadamard`, `RIIR_METAL_TENSOR_RELAXED=1` (the serving selector).

## Posture reconciliation (the round-1 REVISE items)

- **Frame shape:** Bench 937's census frame is ONE 512-token chunk through all 64 layers (48 GDN / 16 attn; 8 frames = pp4096, the 1.104 ms/token control cross-checks). Every GEMM launch inside it dispatches at **P=512** — the P=2048/4096 rows below are league-shape context, not the serving shape.
- **Per-launch walls vs the census:** my sync-per-launch GEMM walls (gate/up 4.27 + down 5.11 ≈ 9.4 ms/layer) are ~2× the census's pipelined FFN strip (281.16/64 ≈ 4.4 ms/layer-chunk). The difference is the measurement posture, not an error: the census wall overlaps consecutive launches on the async queue (the zero-copy branch commits WITHOUT `wait_until_completed`; see the gemm_ternary_metal_zero_copy dispatch comment), while this cell fences every launch so a timed region cannot exclude work. Shares quoted below come from the sync-fenced G2/G3 arms; the e2e projection applies them to the census's pipelined strip, which is the conservative direction (pipelining can only hide more of the exposure, never less).
- **G1 bar vs relaxed:** the first full-suite run failed G1 at `4.321e-4` under `RIIR_METAL_TENSOR_RELAXED=1` — relaxed_precision shortens matmul2d's accumulation (fp16-class), the same distribution shift Bench 940 priced at 2× on the dense lane. G1 (the 1e-5 scale-relative bar) now pins the BIT-EXACT posture via the module's explicit setter; G2/G3 run the serving selector (`relaxed_precision=true`, printed at boot). `e7b98d1`.
- **TFLOPS print:** the first landing's `matmul-only` figure was off by 1000× (formula `…/1e9`, labeled TFLOPS). Fixed to `/1e12`; the corrected isolated-matmul throughput at P=512 reads 18.9–25.9 TFLOPS f16-class — consistent with an M5-class GPU at these tile shapes, and independently confirming the matmul is compute-healthy (the prize, if any, is in the staging serialization, not the math).

## Fixture laws the gates caught (recorded — they bind every future bit-plane fixture)

1. **Scales must be f16-exact.** The kernel stages `(half)(sign * scale)`; with full-f32 scales the half-rounding injects ~5e-4 error the production path never carries (the wire stores f16 scales, so `sign * scale` is exactly half-representable). Symptom: G1 3.098e-4 vs 2.682e-7 after the fix.
2. **The scale index is the kernel's hard-coded `col / 128`** (Q2_0_g128), NOT `cols / groups_per_row` arithmetic on ragged cols — 33×192 discriminates (96-wide arithmetic groups vs 128-wide indexed groups; 0.5677 → 7.56e-8 after the fix).

## G2 — the isolated dequant share (probe = dequant+barrier+stage-write in isolation; sync-fenced)

| shape | P=512 | P=2048 | P=4096 |
|---|---:|---:|---:|
| ffn_gate/up 17408×5120 | **10.2%** (4.300 ms, probe 0.437) | 7.9% | 7.3% |
| ffn_down 5120×17408 | **11.2%** (5.445 ms, probe 0.610) | 7.9% | 7.2% |
| attn_q 12288×5120 | **12.4%** (2.947 ms, probe 0.366) | 8.6% | 7.4% |

The probe is an UPPER bound (fully serialized; in the real kernel other workgroups hide part of it) — which is exactly what the G3 twins measure in situ.

## G3 — the in-situ ablation twins at P=512 (the real serving source, one element patched, count-asserted; wrong results by construction, timing-only)

| shape | baseline | −barriers | −dequant | −stage-stores |
|---|---:|---:|---:|---:|
| ffn_gate/up | 4.267 ms | **1.5%** exposed | 3.1% | 3.2% |
| ffn_down | 5.108 ms | **−0.1%** | 4.2% | 4.5% |
| attn_q | 2.987 ms | **2.2%** | 4.7% | 4.6% |

Reading:

- **The barrier exposure — rung C's actual prize — is 1.5–2.2% of GEMM wall in situ** (ffn_down reads zero within noise). The isolated probe's 10–12% share is mostly HIDDEN by other workgroups in the real dispatch.
- The dequant/stage exposures (3–4.7%) are NOT additive with each other — `const_dequant` removes the loads+math that FEED the `no_stage` stores; the max single-lever exposure is the stage-store family at ~4.5%, and a double-buffered staging keeps the stores (it re-times them, it does not remove them).
- The first G3 pass (13:39 window) measured `no_barriers` FASTER than baseline on two shapes — physically impossible for a strict superset, the shared-box artifact of the 1177 sibling's bench spinning up mid-run (load 2.19→4.19). That pass is discarded; these rows are the settled quiet-window rerun (load 2.45–2.49, no GPU consumer, both co-tenant cells drained per the handshake).

## The verdict arithmetic (round 3 — launch-weighted, all ternary GEMM classes, clean windows)

Barrier-exposure deltas per launch (−barriers twin, n=12 rounds, Δ-IQR), frame-weighted at the serving census shape (64 layers: 48 GDN / 16 attn; gate/up ×2 per layer):

| shape | baseline ms | −barriers Δ ms (IQR) | launches/frame | ms/frame |
|---|---:|---:|---:|---:|
| ffn_gate/up | 1.426 | 0.152 [0.059, 0.324] | 128 | 19.5 |
| ffn_down | 1.747 | 0.235 [0.206, 0.258] | 64 | 15.0 |
| attn_q | 1.018 | 0.110 [0.066, 0.169] | 16 | 1.8 |
| gdn_qkv | 1.205 | 0.121 [0.102, 0.160] | 48 | 5.8 |
| gdn_z | 0.686 | 0.069 [0.012, 0.151] | 48 | 3.3 |
| gdn_ab | 0.330 | 0.008 [−0.002, 0.043] | 96 | 0.8 |
| gdn_out | 0.759 | 0.081 [0.049, 0.110] | 48 | 3.9 |
| **total** | | | | **≈ 50.1 ms/frame** |

≈ 50.1 / 500.86 = **10.0% e2e, fence-isolated upper bound — OVER the 1.5% bar.** Two honest caveats keep it a ceiling, not a promise: (1) `−barriers` removes ALL synchronization (wrong output); double-buffering keeps one barrier per K-tile and recovers only part; (2) the seven rows were measured separately — nothing yet shows the savings stack. Caveat (2) is bounded by the reviewer's fence-symmetry point: the deltas are fenced-run minus fenced-run, so a fixed harness-fence cost cancels; what could inflate Δ is ramp/tail interacting with barrier removal.

Also recorded: the isolated walls now SUM to the census strip (1.426×128 + 1.747×64 ≈ 294 ms vs 281 ms FFN strip) — the frame's big-GEMM class is effectively serial, which is why the fence-isolated deltas transfer at all.

**Decision (per the issue's own price-gate rule, now that the projection clears): stop projecting, measure.** The decider is a **G1-gated double-buffered-staging variant** (correct output — G1 ≤1e-5 at the bit-exact posture, all tiles written) behind a selector, decided by a **paired Bench-937 frame-census A/B**: same box, quiet window, alternating order, n≥3; the decider is the frame-wall change vs 500.86 ms with spread printed; side data = per-shape twin Δ + the +8 KB/threadgroup smem cost. ≥1.5% e2e + G1 green → GOAT gate on the feature flag (promotion through `riir-infer-gpu/production`, never a repeat of the gdn_prework gap). Under 1.5% → rung C closes NEGATIVE citing the measured frame Δ, which replaces the projection.

H5-device-f16 (the only other mpp-expressible variant): refuted on resident footprint — 8× the A-side bytes (~54 GB for 27B elements) defeats the ternary tier; the modelled traffic deltas (G2 table, +0.5–6.1 ms/launch at the ~800 GB/s class) corroborate and are recorded as modelled, not measured.

## Provenance

- Commits: `c5fba2e` (instrument + G1 fixture laws), `cb00077` (in-situ twins + TFLOPS fix), `e7b98d1` (G1 bit-exact pin). This record: 948 (allocated after `git fetch` + `dual_allocation_gate.py`: 0 twin / 0 independent / 0 counter; 947 taken by the sink-audit census `7b42768`).
- Logs: this run's stdout (the durable tables above); the discarded 13:39 polluted pass is described, not quoted.
- Verdict ping-pong: round 1 REVISE (claude_code session `8e2cf757`) — frame-shape correction (P=512), footprint-first H5 refutation, modelled-traffic labeling, bench renumber, TFLOPS fix, posture check all adopted here; round 2 closed AGREE on the rerun evidence.
