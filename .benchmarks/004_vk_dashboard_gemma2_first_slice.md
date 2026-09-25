# Bench 004 — Issue 883 P0 R² dashboard: gemma-2-2b-it FIRST SLICE (the fitted-token-value-table go/no-go)

**Status:** COMPLETE — MEASUREMENT-ONLY (the P0 law; no quality claim). **The offline go/no-go reads GO for the P1/P2/P3 products**: mean ρ_l(V−K) = 0.49 (layer range 0.30–0.98, U-shaped through depth) — token identity carries ~half the V−K residual's variance on this artifact, so the fitted `E_l[s]` table has real signal to refund; NOT a null. Feature `vk_calibration` stays OPT-IN (an offline instrument, never a serving path).

**Box:** 4090 workstation (i7-13700K, Windows 11, AC power, CPU lane at 8 tok/s; the sibling Laya-CUDA agent's builds ran concurrently — both arms equally exposed). Pass: 120,000 tokens in 14,374 s. Tables: top_k=8192 tracked rows ×3 signals ×26 layers = 2.44 GiB (tracked mass 97.6% — the tail-lump lower bound is tight).

**Method:** one causal pass (seq chunks of 1024 « the 4096 SWA window) through the frozen checkpoint, V/K tapped at every layer between the QKV projections and RoPE (pre-RoPE K, post-W_V V — the tap-point law), feeding the shared `katgpt-core/fitted_anchor_tables` substrate (katgpt-rs Bench 886). Corpus: one `chat_probe` HF page (~641 KB natural chat text, 132k tokens, first 120k taken). Top n_s = 4,816 — no small-n ANOVA inflation (the 500-token smoke read ρ≈0.75–1.0 was singleton-inflated; this slice's common tokens carry n_s in the thousands).

## Headline numbers

- mean over 26 layers: **ρ(V)=0.4798 · ρ(K)=0.4966 · ρ(V−K)=0.4896**
- depth profile: layer 0 ≈ 0.98 (embedding-near determinism) → mid-stack trough 0.27–0.35 (layers 6–14, contextual mixing dominates) → late rise 0.45–0.67 (layers 18–24)
- per-head spread at layer 13: 0.26–0.39 (head heterogeneity real — P1's per-head read is the right grain)
- Zipf coverage: K=1024 → 73.6%, K=4096 → 90.9%, K=8192 → 97.6% (the P4 storage dial)

## Fixture caveats (issue 010 — recorded beside every figure)

1. 288-tensor conversion, NO q/k-norm tensors (llama.cpp standard = 340) — ρ(K)/ρ(V−K) describe THIS artifact; V is QK-norm-free even upstream. Trap 1 re-arms on a standard conversion.
2. First slice 120k tokens vs the issue's 10⁸–10⁹ full spec — a reduced-scale read with measured coverage; the full pass is a wall-clock matter (8 tok/s CPU), not a design question.

---

# Generated artifact (verbatim from `vk_calibration`)

# Issue 883 P0 — R² dashboard: gemma-2-2b-it-f16.gguf (288-tensor conversion, no q/k-norm — caveat 1)

slice: 120000 tokens | top_k=8192 | seq_len=1024 | tap: pre-RoPE K, post-W_V V

| layer | ρ_l(V) | ρ_l(K) | ρ_l(V−K) | V mass | K mass | V−K mass |
|---|---|---|---|---|---|---|
| 0 | 0.9842 | 0.9810 | 0.9823 | 0.976 | 0.976 | 0.976 |
| 1 | 0.8453 | 0.8784 | 0.8637 | 0.976 | 0.976 | 0.976 |
| 2 | 0.7920 | 0.8139 | 0.8043 | 0.976 | 0.976 | 0.976 |
| 3 | 0.6783 | 0.7186 | 0.7002 | 0.976 | 0.976 | 0.976 |
| 4 | 0.5145 | 0.5346 | 0.5275 | 0.976 | 0.976 | 0.976 |
| 5 | 0.4721 | 0.5756 | 0.5351 | 0.976 | 0.976 | 0.976 |
| 6 | 0.3476 | 0.3275 | 0.3349 | 0.976 | 0.976 | 0.976 |
| 7 | 0.3540 | 0.4281 | 0.4060 | 0.976 | 0.976 | 0.976 |
| 8 | 0.3713 | 0.4279 | 0.4078 | 0.976 | 0.976 | 0.976 |
| 9 | 0.3716 | 0.4511 | 0.4206 | 0.976 | 0.976 | 0.976 |
| 10 | 0.3628 | 0.4326 | 0.4072 | 0.976 | 0.976 | 0.976 |
| 11 | 0.3034 | 0.3887 | 0.3594 | 0.976 | 0.976 | 0.976 |
| 12 | 0.2657 | 0.3219 | 0.3018 | 0.976 | 0.976 | 0.976 |
| 13 | 0.2943 | 0.3556 | 0.3343 | 0.976 | 0.976 | 0.976 |
| 14 | 0.3152 | 0.3199 | 0.3180 | 0.976 | 0.976 | 0.976 |
| 15 | 0.3615 | 0.3562 | 0.3581 | 0.976 | 0.976 | 0.976 |
| 16 | 0.4094 | 0.3565 | 0.3728 | 0.976 | 0.976 | 0.976 |
| 17 | 0.3951 | 0.4158 | 0.4082 | 0.976 | 0.976 | 0.976 |
| 18 | 0.5244 | 0.4588 | 0.4779 | 0.976 | 0.976 | 0.976 |
| 19 | 0.5218 | 0.4517 | 0.4737 | 0.976 | 0.976 | 0.976 |
| 20 | 0.5685 | 0.5058 | 0.5253 | 0.976 | 0.976 | 0.976 |
| 21 | 0.4127 | 0.4689 | 0.4492 | 0.976 | 0.976 | 0.976 |
| 22 | 0.6682 | 0.4801 | 0.5291 | 0.976 | 0.976 | 0.976 |
| 23 | 0.5352 | 0.5007 | 0.5110 | 0.976 | 0.976 | 0.976 |
| 24 | 0.4585 | 0.4957 | 0.4873 | 0.976 | 0.976 | 0.976 |
| 25 | 0.3463 | 0.4671 | 0.4329 | 0.976 | 0.976 | 0.976 |

## per-head ρ_l(V) (layer 0 and 13)

| layer 0 | 0.9856 | 0.9863 | 0.9825 | 0.9830 |
| layer 13 | 0.3875 | 0.2783 | 0.2595 | 0.2616 |

## coverage(K) — layer-0 V table (storage dial P(K)=b_w·L·K·d_v)

- K=16: 0.3040
- K=64: 0.4482
- K=256: 0.5778
- K=1024: 0.7359
- K=4096: 0.9094
- K=8192: 0.9764

## top-10 n_s (layer 0)

- [4816, 4249, 4098, 3579, 3385, 2659, 2202, 2140, 1692, 1586]

mean over layers: ρ(V)=0.4798 ρ(K)=0.4966 ρ(V−K)=0.4896

MEASUREMENT-ONLY (P0 law): no quality claim. rho_l ≈ 0 on this GQA fixture would be a legitimate recorded negative (trap 4 exempts only fixture-class nulls — this IS the mechanism-bearing fixture).
