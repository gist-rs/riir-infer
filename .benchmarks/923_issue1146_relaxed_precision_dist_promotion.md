# Bench 923 — Issue 1146 rung 2b: matmul2d relaxed_precision in the dist prefill — 1.96–2.19× on top of H4, per-family quality PASS, PROMOTED (dist-scoped)

**Status:** RECORD — PROMOTED in the **dist stage only** (`TernaryGpuStage::new` → `set_metal_tensor_relaxed_precision(true)`, `RIIR_DIST_METAL_TENSOR_RELAXED=0` demotes). The **substrate default stays OFF**: this is a lossy surface, and the riir-ai league harness plus every bit-exact tensor-vs-simdgroup gate keep the incumbent bytes. Stacked dist prefill at w512: **pp512 0.72 s (709 tok/s, 0.65× llama.cpp), pp4096 7.13 s (574 tok/s, 0.55×)**, up from 0.13–0.16× this morning.
**Date:** 2026-10-28 (quality 19:38–19:56, stacked A/B 19:58–20:02 +07)
**Session:** katgpt-rs-71, 1793178290 (measurement, promotion). Selector: katgpt-rs-31 `6125f9e`. The source read that ranked it one of two kernel differences from llama.cpp (upstream `19ab5a5` / prismml fork `73bfe6c` compile `relaxed_precision=true`): katgpt-rs-bd. KL instrument: katgpt-rs-bd `083b465`.
**Issue:** [`.issues/1146_prefill_throughput_gap_1box.md`](../.issues/1146_prefill_throughput_gap_1box.md)
**Box:** bmo (Mac17,15, M5 Ultra, 96 GiB), AC
**Model:** `Ternary-Bonsai-2-27B-PQ2_0.gguf` sha256 `3907dc16…62ec1` (league pin)

## What it is

The tensor GEMM's `matmul2d` descriptor takes a `relaxed_precision` flag. Ours was pinned `false` for bit-level G1, while llama.cpp (upstream and the prismml fork) compiles it `true`. With it on, the hardware may accumulate at reduced precision. The output is therefore **not** bit-identical, so this is a lossy surface and the AGENTS lossy-surface rule applies: judge it on deployed-path behaviour, per family, not on bit identity or aggregate perplexity.

## Quality (the gating read)

**Logit level** (`issue1146_tensor_range_g1` under `RIIR_METAL_TENSOR_RELAXED=1`, relaxed tensor vs the simdgroup incumbent, w512; the bit-exact assert fails by design):

| P | argmax off / on | max_abs (all 248,320 logits) |
|---|---|---|
| 512 | 13 / 13 | 4.03e-3 |
| 1024 | 261 / 261 | 6.00e-3 |
| 4096 | 248046 / 248046 | 5.57e-3 |

The three postures (whole / range-full / range-split) are identical in every row. The error does not grow with length.

**Per-family teacher-forced KL** (`issue1146_relaxed_kl`, P=256, K=16 positions per family, f64 log-softmax over f32 logits, ref = simdgroup). The STRICT control run (relaxed OFF) reads **KL exactly 0, top-1 16/16, ΔNLL 0 in every family**. The RELAXED run:

| family | n | KL mean | KL max (nats) | top-1 | ΔNLL mean | NLL(ref) |
|---|---:|---:|---:|---:|---:|---:|
| code | 16 | 2.97e-7 | 2.67e-6 | 16/16 | +1.75e-4 | 0.316 |
| math | 16 | 3.36e-8 | 3.04e-7 | 16/16 | +2.77e-4 | 0.546 |
| prose | 16 | 1.08e-6 | 2.73e-6 | 16/16 | −2.72e-4 | 2.407 |
| zh | 16 | 9.99e-7 | 3.83e-6 | 16/16 | +3.28e-4 | 2.489 |

ΔNLL is ≤ 0.06% of NLL(ref) in every family, with mixed sign (prose improves). That is noise, with no family-conditional shift.

**Behavioural** (dist, w512, greedy 256 tokens from a chat-templated `--text` prompt, relaxed off vs on): code / math / prose / zh all **first_divergence NONE**, 256/256 identical tokens. The on-arm text is coherent and on task in all four families (Fibonacci code with docstring, step-by-step unit conversion, Rayleigh-scattering prose, a Great Wall essay in Chinese). Logs: `.benchmarks/i1146_relaxed/quality/`.

**Regression bar (named re-gate):** any future change to the tensor GEMM kernel or its compile flags re-runs `issue1146_relaxed_kl` under `RIIR_METAL_TENSOR_RELAXED=1` with **`I1146K_MAX_KL=1e-4`** (~26× headroom over the measured 3.83e-6 max), plus the strict control.

## G2 — stacked on the post-H4 build (one binary at `40ac136`, env-only A/B, per-pair contention guard, zero CONTAM)

| cell | a (H4, relaxed off) | b (H4 + relaxed) | b vs a |
|---|---:|---:|---:|
| pp512 w512 r1 / r2 | 1.588 / 1.579 s | 0.720 / 0.725 s | **2.19×** |
| pp4096 w512 r1 / r2 | 13.968 / 13.984 s | 7.140 / 7.126 s | **1.96×** |

The a arm reproduces Bench 921's H4 values (1.630 / 13.959 s; pp512 now reads slightly faster), so the box was clean. The pre-H4 read (run `/tmp/i1146q/relaxed`, 3 guard-flagged pairs re-run) was 1.82× / 1.68×. Relaxed stacks **more** than multiplicatively with H4: H4 removed the dequant staging cost, and relaxed accelerates the matmul that is then the larger share.

## The day's 1146 ladder (dist prefill, w512 unless noted, bmo)

| posture | pp512 | pp4096 | vs llama.cpp |
|---|---:|---:|---:|
| this morning: w128, simdgroup GEMM (Bench 911/914) | 3.04 s | 25.6 s | 0.13–0.16× |
| + width 512 (Bench 919) | 2.19 s | 18.4 s | 0.21–0.23× |
| + tensor zero-copy GEMM (Bench 920) | 1.75 s | 15.24 s | 0.26–0.27× |
| + H4 dequant (Bench 921) | 1.63 s | 13.96 s | 0.28–0.29× |
| **+ relaxed_precision (this)** | **0.72 s** | **7.13 s** | **0.65× / 0.55×** |

The width-512 row needs `--prefill-chunk 512`. The `dist_stage` default stays 128 until the 2-box TB5 leg runs (amo is held by the riir-ai 641-T5 window).

## Why dist-scoped, not substrate

H4 flipped substrate-wide because it is bit-exact. Relaxed is not. A substrate flip would silently change the riir-ai league's prefill outputs, break every bit-exact tensor-vs-simdgroup gate (riir-ai `bench_1122` G1 `max_abs 0`, our `issue1146_tensor_range_g1`), and land inside the 641-T5 protected window. The lossy posture lives only where its quality read was taken. A league-side flip is riir-ai's own decision, made on this record.

## Provenance

- Stacked A/B: `.benchmarks/i1146_relaxed/relaxed_v5/` (binary sha `61527e80…`, `env_b: RIIR_METAL_TENSOR_RELAXED=1`). Quality: `.benchmarks/i1146_relaxed/quality/` + `kl_relaxed.log`.
- The promoted build sets the same selector through the setter (wins over the env) before the forwarder compiles its pipelines, so `compose_msl_source` emits the same `#define SWEEP_RELAXED true` as the measured arm.
