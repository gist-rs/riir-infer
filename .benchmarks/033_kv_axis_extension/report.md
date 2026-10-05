# DQ phase-matrix run (Plan 614 / Issue 026) — RAW DUMP

- model: `E:/git/riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf` blake3=`b4f6ab953ef6d1452682bd861b7a1711a4a9945a74ca292659803978fd2550f6`
- corpus blake3: `2dd4544f009a8a21108a2b5247d918ac6f95039084ae2a9fcfdfda5ead93e059`
- gpu: NVIDIA GeForce RTX 4090, 24564 MiB, 23301 MiB
- compute apps at start: none with dedicated memory
- grids: [A2, A4]
- kv grids (Issue 033): [A8, A4]
- arith_n: 48, arith_hard: true, ni_lengths: [4096, 8192, 16384], ni_per_len: 32, ni_needles: 10
- bootstrap: 10000
- attn_layers: 16

## G-i1/G-i4

- G-i1 (knob-off counters == 0): PASS
- G-i4 (base twice byte-stable): PASS

## Per-cell accuracy + counters

| cell | arith acc | nih acc per len | prefill launches | decode launches | kv launches |
|---|---|---|---|---|---|
| base | 0.7500 | 4096: 1.000, 8192: 1.000, 16384: 1.000 | 0 | 0 | 0 |
| both_aq.a2 | 0.0000 | 4096: 0.156, 8192: 0.188, 16384: 0.062 | 69632 | 3746304 | 0 |
| both_aq.a4 | 0.7917 | 4096: 0.969, 8192: 0.969, 16384: 1.000 | 69632 | 1555200 | 0 |
| dec_a8 | 0.7500 | 4096: 1.000, 8192: 1.000, 16384: 1.000 | 0 | 1496320 | 0 |
| dec_aq.a2 | 0.1250 | 4096: 0.969, 8192: 1.000, 16384: 0.969 | 0 | 1737472 | 0 |
| dec_aq.a4 | 0.7708 | 4096: 0.969, 8192: 1.000, 16384: 1.000 | 0 | 1520128 | 0 |
| kv_aq.a4 | 0.7708 | 4096: 1.000, 8192: 1.000, 16384: 1.000 | 0 | 0 | 195744 |
| kv_aq.a8 | 0.7500 | 4096: 1.000, 8192: 1.000, 16384: 1.000 | 0 | 0 | 195744 |
| pf_aq.a2 | 0.2083 | 4096: 0.250, 8192: 0.188, 16384: 0.125 | 69632 | 0 | 0 |
| pf_aq.a4 | 0.7917 | 4096: 1.000, 8192: 0.969, 16384: 1.000 | 69632 | 0 | 0 |
| pfkv_aq.a4 | 0.8125 | 4096: 1.000, 8192: 1.000, 16384: 1.000 | 69632 | 0 | 201472 |
| pfkv_aq.a8 | 0.7500 | 4096: 1.000, 8192: 1.000, 16384: 1.000 | 69632 | 0 | 195744 |

## Paired stats (Δdec − Δpf per item; bootstrap 95% CI)

- **a2 decode-heavy**: acc base=0.7500 pf=0.2083 dec=0.1250; Δpf=0.5417 Δdec=0.6250; R=1.154; CI(Δdec−Δpf)=[-0.2292,0.0417] → **NULL**
- **a2 prefill-heavy (pooled)**: acc base=1.0000 pf=0.1875 dec=0.9792; Δpf=0.8125 Δdec=0.0208; R=0.026; CI=[0.7083,0.8750] → **INADMISSIBLE**
- **a4 decode-heavy**: acc base=0.7500 pf=0.7917 dec=0.7708; Δpf=-0.0417 Δdec=-0.0208; R=NaN; CI(Δdec−Δpf)=[-0.1042,0.0625] → **NULL**
- **a4 prefill-heavy (pooled)**: acc base=1.0000 pf=0.9896 dec=0.9896; Δpf=0.0104 Δdec=0.0104; R=1.000; CI=[-0.0312,0.0312] → **INADMISSIBLE**

## Issue 033 — KV-store axis + interaction

- Spelling: KV rows rounded in place to the grid at WRITE (the q8kv store posture). For accuracy this equals a q8 read rounding (idempotence) — the axis is ONE accuracy experiment.
- **kv[a8] decode-heavy (arith)**: acc base=0.7500 kv=0.7500; Δkv=0.0000; CI(Δkv)=[0.0000,0.0000] → **NULL**
- **kv[a8] prefill-heavy (pooled)**: acc base=1.0000 kv=1.0000; Δkv=0.0000; CI=[0.0000,0.0000] → **INADMISSIBLE**
- interaction [a8]: SKIPPED — no same-grid pf_aq cell (DQ_GRIDS did not include a8)
- **kv[a4] decode-heavy (arith)**: acc base=0.7500 kv=0.7708; Δkv=-0.0208; CI(Δkv)=[0.0000,0.0625] → **NULL**
- **kv[a4] prefill-heavy (pooled)**: acc base=1.0000 kv=1.0000; Δkv=0.0000; CI=[0.0000,0.0000] → **INADMISSIBLE**
- **interaction pf→pfkv [a4] decode-heavy**: pf=0.7917 pfkv=0.8125; Δ(kv|pf armed)=-0.0208; CI=[-0.0417,0.1042] (marginal KV damage ON TOP of prefill quant)

### Axis dominance (decode-heavy, |Δacc| vs base)

- decode-act[a2]: +0.6250
- prefill-act[a2]: +0.5417
- prefill-act[a4]: -0.0417
- decode-act[a4]: -0.0208
- kv-store[a4]: -0.0208
- kv-store[a8]: +0.0000

## dec_a8 control (D1 fallback): |Δarith| = 0 items (gate ≤ 2) → PASS

## Verdicts

- a2.decode_heavy: **NULL**
- a2.prefill_heavy: **INADMISSIBLE**
- a4.decode_heavy: **NULL**
- a4.prefill_heavy: **INADMISSIBLE**
- kv[a4].decode_heavy: **NULL**
- kv[a4].prefill_heavy: **INADMISSIBLE**
- kv[a8].decode_heavy: **NULL**
- kv[a8].prefill_heavy: **INADMISSIBLE**
