# Bench 948 — Issue 1167 rung C: the GEMM-staging price cell — barrier/staging exposure measured in situ at the serving posture; the double-buffer reclaim lands UNDER the 1.5% e2e bar

**Status:** RECORD — price cell complete (G1 green + G2 isolated-share + G3 in-situ twins, one process, GPU-exclusive window). **Rung C verdict: CLOSED NEGATIVE at the price gate — no kernel write.** H5-device-f16 is refuted on resident footprint (the A-side f16 mirror is 8× the packed-bit bytes — 27B elements → ~54 GB, defeating the ternary tier's reason to exist; the ~800 GB/s-class traffic DELTA numbers below are modelled, not measured, and are moot next to the footprint argument). The implementable mpp variant set is {device-f16 operand, double-buffered smem staging}; register-feeding dequant into `matmul2d` operand registers is not expressible — matmul2d consumes `tensor` views (device/threadgroup memory only), the same shape llama.cpp's `kernel_mul_mm` uses.
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

## The verdict arithmetic

- Serving FFN strip (post-prework-fix census): **281.16 ms of a 500.86 ms frame = 56.1%**.
- Upper bound (assume the largest measured in-situ exposure transfers 1:1 and is fully reclaimable): 4.5% × 56.1% ≈ **2.5% e2e** — above the bar only if every exposure were additive AND fully reclaimable, which the twin design itself refutes (the exposures overlap; the stores persist under double-buffering).
- The double-buffered-staging kernel (the implementable variant that keeps bit-exactness and the smem path) reclaims the **barrier exposure only: 1.5–2.2% × 56.1% ≈ 0.8–1.2% e2e — UNDER the 1.5% bar**, before its own cost (a second smem tile = +8 KB/threadgroup, and the K-loop's remaining serialization).
- H5-device-f16 (the only other mpp-expressible variant): refuted on resident footprint — 8× the A-side bytes (~54 GB for 27B elements) defeats the ternary tier; the modelled traffic deltas (G2 table, +0.5–6.1 ms/launch at the ~800 GB/s class) corroborate and are recorded as modelled, not measured.

**Rung C: CLOSED NEGATIVE at the price gate.** The remaining 1167 ground is unchanged: gdn_rec 5.8%, the pp512 config-class rung (~5%, bounded, below kernel work), and the packed-byte dequant load (its own price cell; complementary — it shrinks A-side bytes, and this cell's probe bounds its ceiling with the same instrument family).

## Provenance

- Commits: `c5fba2e` (instrument + G1 fixture laws), `cb00077` (in-situ twins + TFLOPS fix), `e7b98d1` (G1 bit-exact pin). This record: 948 (allocated after `git fetch` + `dual_allocation_gate.py`: 0 twin / 0 independent / 0 counter; 947 taken by the sink-audit census `7b42768`).
- Logs: this run's stdout (the durable tables above); the discarded 13:39 polluted pass is described, not quoted.
- Verdict ping-pong: round 1 REVISE (claude_code session `8e2cf757`) — frame-shape correction (P=512), footprint-first H5 refutation, modelled-traffic labeling, bench renumber, TFLOPS fix, posture check all adopted here; round 2 closed AGREE on the rerun evidence.
