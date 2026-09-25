# Bench 001 — EXL3 residency + context ceiling on the 4090 (Issue 001 T5)

**Status:** COMPLETE (2026-09-24) · method: measured pack bytes + arithmetic
projection · evidence chain: Issue 001 §13 (`.docs/001_exl3_trellis_format_support.md`, relocated at close)

## What this measures

The axis Issue 001 §2 licenses for EXL3 (trellis-coded weights): **residency
and context ceiling** — NOT throughput (§2's own decode data was inside
sample spread; no serving path exists until T7).

- **Residency bytes**: measured from real safetensors shard headers
  (`Exl3Pack::residency`, landed `bd93a7b`) — exact, not derived from config
  geometry. The 4bpw row is the fully downloaded pack (sha256-verified vs the
  HF LFS hash); other rows are surgical HTTP-Range header fetches of the same
  repo's branches (byte counts are shape-determined → exact for any era).
- **Context ceiling**: arithmetic projection from measured bytes with STATED
  assumptions (below) — not a served measurement.

## Specimen

`turboderp/Qwen3.8-27B-exl3` @ `SC_4.00bpw_H5_V6` (`516bf129`) — the format
author's pack of the league model. Qwen3.8-27B: 64 layers, hidden 5120, vocab
248320; HYBRID attention (48/64 `linear_attention`, 16 full-attn every 4th,
4 KV heads × 256); MTP ×1; vision tower. exl3 1.4.2, mul1 codebook, pin-era
(2026-08-27 vs pin 2026-09-20). 16,349,968,660 B on disk.

Box (G2 box-state rule): i7-13700K, RTX 4090 24 GiB, CPU ~8%, GPU idle, AC.

## Residency (measured bytes)

| pack | total GiB | achieved bpw | per-layer K | quantized GiB | dense GiB | dequant-f16 GiB |
|---|---:|---:|---|---:|---:|---:|
| 2.00bpw | 10.039 | 2.232 | 2.0–6.0 | 6.763 | 3.275 | 51.75 |
| 3.00bpw | 12.871 | 3.167 | 3.0–6.0 | 9.595 | 3.275 | 51.75 |
| **SC_4.00_H5_V6** | **15.227** | **4.087** | **3.0–6.0** | **12.601** | **2.626** | **51.95** |
| 5.00bpw | 18.535 | 5.037 | 4.0–6.0 | 15.259 | 3.275 | 51.75 |
| 6.00bpw | 21.367 | 5.972 | 4.0–6.0 | 18.091 | 3.275 | 51.75 |
| (f16 baseline) | — | 16.0 | — | — | — | 51.95 |

Model ≈ 27.8–28.2 G params. The f16 model does not fit a 24 GiB box at any
context; every EXL3 variant does. 4bpw interior: 573 groups / 3080 tensors;
trellis 12.586 GiB + scales 14.8 MiB + markers 2.2 KiB + bias 0.5 MiB.

## Context ceiling on 24 GiB (projection; assumptions stated)

KV/token f16 = 16 full-attn × 2 × 4 heads × 256 × 2 B = **64 KiB/token**
(hybrid advantage: linear-attn state is FIXED ~151 MB f32, not per-token).
Assumed overhead: 1.5 GiB engine + 0.15 GiB state. Cap 262,144.

| pack | weights GiB | ceiling (tok) |
|---|---:|---:|
| 2.00bpw | 10.04 | ~201,600 |
| 3.00bpw | 12.87 | ~155,300 |
| SC_4.00 | 15.23 | ~116,700 |
| 5.00bpw | 18.54 | ~62,500 |
| 6.00bpw | 21.37 | ~16,100 |
| f16 | 51.95 | 0 (does not fit) |

§2's direction CONFIRMED on our silicon: 2.0 vs 4.0 bpw → −5.2 GiB resident
→ +73% ceiling (201k vs 117k). q8_0 KV would ~double the ceilings (engine
knob, not a format fact).

## CPU reference dequant (baseline for T7)

Release, single thread: **6.1–6.5 M weights/s** (o_proj 31.5M w / 5.01 s;
mlp 89.1M w / 14.5 s; lm_head 1.27B w / 197 s — scalar O(in·out·128)).
Full-model single-thread extrapolation ≈ 70 min. Zero NaN; sane stats.

**T7a CPU arm (Issue 001 §15, `725a8a5`):** codebook LUT + rayon over
disjoint row strips — **bit-identical** to the scalar reference — measures
**65–69 M weights/s (10.6–11.4×)** across the same classes (lm_head 197 s →
19.4 s). Full-model parallel extrapolation ≈ 6.7 min.

## Correctness (era-gate, native oracle on the same pack)

Rust `Exl3Pack::layer().dequantize_f32` vs exllamav3-native
`get_weight_tensor`, 5 classes: rel-Frobenius ≤ 1.26e-07 (lm_head and
self_attn.o_proj EXACTLY 0.0), max_abs ≤ 1.0e-03 on max|W| ≤ 1.35 — the
fp16-intermediate class T4b documented. Mixed-K recipe read end-to-end
(mlp K=3, attn K=4–5, lm_head K=5, shard-spanning lm_head group).

## NOT claimed

Decode/prefill tok/s (no serving path until T7); quality (T6/GOAT's
per-family retention walk is the instrument, not run here); MTP/acceptance
(closed negative); GGUF-vs-EXL3 same-model rows (no qwen3.8-27B GGUF on box
— open until one lands).
