# Bench 914 — Issue 1146: the dist gpu feature-row A/B + the prefill frame census

**Status:** RECORD — feature-row fix CONFIRMED ENGAGED but only 1.09× (the dispatch-storm hypothesis REFUTED); the census attributes the real floor: the M=128 ternary GEMM projection classes (FFN 65% + DeltaNet 25%) + a live 1.55× wide-chunk lever; the W-frame rung + the width re-curve are the ladder
**Date:** 2026-10-28 (sessions 05:08–06:30 +07; A/B 05:08–05:13, census v1 05:36 JIT-polluted, v2 05:50 host-load-polluted on the coarse strips, v3 06:06–06:17 clean)
**Session:** zed-amo-1146-prefill, 1793133500
**Issue:** [`.issues/1146_prefill_throughput_gap_1box.md`](../.issues/1146_prefill_throughput_gap_1box.md) (root-cause chain + the instrument landing)
**Box:** amo (Mac17,15, M5 Ultra, 30 cores, 96 GiB, macOS 27.0.1); load 1.5–2.2 at the clean census; the 641-T5 league lane held the box's attestation attention all night (its attempts REFUSED — no GPU overlap with any census window; v2's coarse-strip pollution was its attempt-prep BUILD burst on the host, see the honesty notes)
**Model:** `Ternary-Bonsai-2-27B-PQ2_0.gguf` sha256 `3907dc16…62ec1` = the league pin EXACT

## The A/B (the feature-row fix, scripts/i1146_ab_session.sh)

Two `dist_stage` release builds, one box, the Bench-911 cell rig (loopback 127.0.0.1:17641, `--split 32`, `--prefill-chunk 128`, both stages `--ctx P+256`):

| cell | incumbent (`gpu` row as shipped; commit 95e5068) | armed (`gpu` + the two prefill features; commit 412c887) | ratio |
|---|---:|---:|---:|
| pp4096 prefill | 24.579 s (166.7 tok/s) | 22.504 s (182.0 tok/s) | **1.09×** |
| pp512 prefill | ~3–4 s (wall 11 s incl. boot) | ~3–4 s (wall 10 s) | ~1.0× |
| tg128 decode | token stream G1 | token stream G1 | **G1 CROSS-POSTURE PASS** (identical decode ids) |

- The incumbent pp4096 cell REPRODUCES Bench 911 (166.9 tok/s there, 166.7 here) — the rig is the same instrument.
- Engagement proven at the binary level: the armed binary is 18,923,952 B vs the incumbent's 17,209,136 B (+1.7 MB = the compiled batched-attention + chunked-conv/recurrence kernel set), and the G1 pass proves the two postures compute identical streams.
- **Verdict: the dispatch-storm arithmetic (~21.5K launches × ~35 µs ≈ 0.75 s/frame) is REFUTED as the floor mechanism** — the CubeCL/Metal submission path is µs-class at these shapes, not the 50–105 µs class Bench 1007 priced on the cudarc/4090 lane. The feature row was a REAL omission (the sequential paths did run) but a MINOR cost. The fix stays (correct posture, matches the engine default, G1-clean); its measured value is 1.09×.

## The census (bench_1146_prefill_frame_census, the strip-down Δwall instrument)

13 postures (3 coarse + 7 parent-gated deltanet sub-bits + embed/tail), 1-box `prefill_range_chunk(0..64)`, prompt 1024 = 8×128, 2 warmup + 5 timed passes each, per-frame wall = pass/8; ALL runs LAST (warmest); bit-identity anchors (never-touched == ALL-first == ALL-after) PASSED ×3 in every run. Clean run (v3, load 1.5–2.2):

| class | Δwall ms/frame | share |
|---|---:|---:|
| `-ffn` (gate/up/down ternary batch GEMMs ×64 layers + swiglu) | **569.67** | **65%** |
| `-deltanet` (4 in-proj + out-proj ternary batch GEMMs ×24 layers) | **220.79** | **25%** |
| `-attention` (flash + attn projections, 8 layers) | 80.04 | 9% |
| `-embed` (128 per-token dequant dispatches + inverse rotate) | 55.05 | 6% |
| `-gdn_prework` (chunked conv, C=64) | 11.31 | 1% |
| `-gdn_rec` (multi-token recurrence) | 11.37 | 1% |
| `-gdn_post` (rec norm + z-gate + permute + rotate) | 3.36 | <1% |
| `-gdn_norm` / `-gdn_rotin` / `-gdn_beta` / `-gdn_residual` / `-tail` | ≤1.3 each | ~0 |
| **ALL baseline** | **873.36** | |
| G-C1 coarse additivity | 925.7 attributed of 873.4 | +6% (closed, within the band) |

**The floor is the ternary batch GEMM at M=128 — hypothesis 1 (decode-shaped kernels) CONFIRMED, now with the dispatch-storm alternative eliminated by the A/B.** The GDN sequential-chain class the root-cause note priced at ~21.5K dispatches measures ~30 ms/frame TOTAL (3%) once the batched bodies are in.

**Wide-chunk control (same run, same kernels, zero new code):** width 512 reads **4.413 ms/token vs 6.823 at 128 = 1.55× per-token** — Bench 909's "width 128 optimal" was measured through the OLD sequential posture; the curve re-opens under the batched bodies. (The serving curve still needs the pp4096+ attention-growth cells before a width flip; this is the 1024-token read.)

## The ladder this census names (each its own GOAT)

1. **Chunk-width re-curve** (config-class, cheapest): re-measure 128/256/512/1024 through the batched posture at serving contexts; the width lever alone reads 1.55× at 512 on this run.
2. **W-frame ternary GEMM tiles** (the kernel rung — the issue's fix direction 1): the M=128 batch GEMMs hold 90% of the frame (FFN 65% + DeltaNet 25%); an M≥512-class tile path (or routing wide-M through the matmul2d tensor path at prefill shapes) is the real fix. Design constraint: the Bench-895 spill cliff (regs=128/local=0).
3. **Batched embed dequant** (minor, mechanical): 128 per-token dispatches → one batched launch (55 ms/frame).
4. The GDN chain (~30 ms, 3%): CLOSED as a floor class — no rung owed.

## Honesty notes

- Census v1 was JIT-polluted (ALL ran first; its timed passes carried one-time kernel compiles — 1711 ms/frame vs ~870 steady) → the harness now runs ALL LAST with 2 warmup passes (the WARMUP_PASSES doc records the lesson).
- Census v2's three coarse strips (first three postures, ~05:50 +07) were host-load-polluted (the 641-T5 league lane's attempt-prep build burst; load spiked ~7→22 during that window; its ATTEMPTS were attestation-refused — no GPU overlap) → v3 re-ran everything on a quiet box (load 1.5–2.2).
- The census rig is 1-box (LayerRange 0..64, no wire): 873 ms/frame vs the pipeline's ~700 ms/frame per stage-pair — same kernel set, the extra being the read_logits-per-frame sync the census pays deliberately (uniform across postures, cancels in Δ).
- The site-timing snapshot printed NO rows in the census rig: the Metal-tensor zerocopy dispatch flag (`PREFILL_USE_METAL_TENSOR_ZEROCOPY`) is not armed by this harness (the dist serving arm sets it via env) — so the census's projection Δs measure the simdgroup-GEMM batch body. The tensor-vs-simdgroup comparison at prefill shapes is a SEPARATE cell (the i1122 gate harness owns that axis; the promoted ktile64 posture was measured THERE).
- The A/B walls include the head's model boot (~6–7 s, mmap); the printed per-cell prefill lines are the compute walls (24.579/22.504 s) — boot-free.
- The 1147 lane's commit (203e922) discarded its own loopback smoke because these cells shared the GPU — cross-lane hygiene noted, no contamination either way (their discarded cell was a smoke, their scored cell was the 2-box TB5).

## Provenance

- A/B: `.benchmarks/i1146_ab/20261028T050830Z/` (per-cell head/tail logs, provenance.txt with both binary sha256s: incumbent `f17d28593d95…`, armed `d2a74b652e62…`); script `scripts/i1146_ab_session.sh` (committed 6a80b4f).
- Census: commits 95e5068 (instrument) + the harness fixes in this landing; runs in `/tmp/i1146_census_run{,2,3}.log` (v3 = the record), driver `crates/riir-infer-gpu/tests/bench_1146_prefill_frame_census.rs`.
- Fix commit: `412c887` (the dist gpu row). Issue: `.issues/1146_prefill_throughput_gap_1box.md`.
- Session: zed-amo-1146-prefill, 1793133500.
