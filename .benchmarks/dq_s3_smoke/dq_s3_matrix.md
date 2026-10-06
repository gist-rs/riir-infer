# dq_s3_matrix — the dual-PTQ measurement (plan 618 S3 / issue 028 T4)

- **Lane (frozen):** cudarc per-token GEMV, ALL arms — kernel policy constant; the lane's int8 activation quantization is COMMON to every arm, so deltas isolate the weight format. NOT the 614 GEMM-prefill lane (cross-lane comparability approximate).
- **Corpus blake3:** `27ccb5bae17b2f111b0734915a0b8e01c715f0cb48f9c1a79d076e3ff6d52552` — arith 4 items (the 614 v2 corpus at hard=false), niah 1/len × 8 needles, lengths [1024]
- **Gen caps:** arith ≤ 24, niah ≤ 8 · bootstrap 10000
- **GPU:** NVIDIA GeForce RTX 4090, 24564 MiB, 23579 MiB
- **Packs:** decode `../riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf` · pf `../riir-train/data/Ternary-Bonsai-2-27B-Q4_K.pf.gguf` · q6 `../riir-train/data/Ternary-Bonsai-2-27B-Q6_K.sg.gguf`
- **Storage pairing:** base 6.7 GB · dual 21.1 GB · matched ≈20.7 GB (the control ~2% CHEAPER — the conservative direction, disclosed) · q4single 14.4 GB (the decomposition control)

## Cells

| arm | family | len | n | acc | TTFT p50 (ms) | TTFT p99 (ms) | decode tok/s |
|---|---|---|---|---|---|---|---|
| dual | arith | 0 | 4 | 0.0000 | 6102.4 | 6477.2 | 73.90 |
| dual | niah | 1024 | 1 | 1.0000 | 22091.9 | 22091.9 | 57.70 |
| matched | arith | 0 | 4 | 0.0000 | 8354.4 | 8716.5 | 34.71 |
| matched | niah | 1024 | 1 | 1.0000 | 29416.7 | 29416.7 | 30.73 |
| q4single | arith | 0 | 4 | 0.0000 | 6099.2 | 6272.7 | 46.72 |
| q4single | niah | 1024 | 1 | 1.0000 | 21900.6 | 21900.6 | 39.69 |
| base | arith | 0 | 4 | 0.0000 | 3690.8 | 4218.8 | 74.06 |
| base | niah | 1024 | 1 | 1.0000 | 14066.4 | 14066.4 | 57.74 |

## Paired vs base (bootstrap 95% CI on the mean paired difference)

| arm | family | len | recovery | CI lo | CI hi | n | note |
|---|---|---|---|---|---|---|---|
| dual | arith | 0 | +0.0000 | +0.0000 | +0.0000 | 4 | INADMISSIBLE base |
| dual | niah | 1024 | +0.0000 | +0.0000 | +0.0000 | 1 | INADMISSIBLE base |
| matched | arith | 0 | +0.0000 | +0.0000 | +0.0000 | 4 | INADMISSIBLE base |
| matched | niah | 1024 | +0.0000 | +0.0000 | +0.0000 | 1 | INADMISSIBLE base |
| q4single | arith | 0 | +0.0000 | +0.0000 | +0.0000 | 4 | INADMISSIBLE base |
| q4single | niah | 1024 | +0.0000 | +0.0000 | +0.0000 | 1 | INADMISSIBLE base |

## Verdict (pre-registered, plan 618 §S3.3)

- niah 1024: dual rec +0.0000 (CI +0.0000) · matched rec +0.0000 (CI +0.0000) · dual≥matched true · dual-CI>0 false
- niah 1024 direct dual−matched: mean +0.0000 CI [+0.0000, +0.0000]

(chance ≈ 1/8 (8-needle first-match); arith chance ≈ 0)

**NULL** — the matched-storage single ≥ dual everywhere. Container SHELVED per the pre-registered gate; T4's decomposition number stands on its own.

## Disclosures

- TTFT here is per-token-GEMV prefill wall — the kernel-maturity posture: walls track weight bytes (≈2.15×/3.05×); the paper's TTFT thesis needs a q4-class GEMM prefill arm and stays a kernel-build question.
- p99 rows print beside n — at n=16 the p99 IS ~max (the percentile lesson; tail support 1).
- Cross-lane: the 614 activation-phase cells (GEMM prefill, A8/A4 fake-quant) remain the long-context reference; this lane's arm deltas are weight-format-only by construction.
- matched-storage delta: the Q6_K control rides ~2% LESS storage than the dual pair — the conservative direction for the container's verdict.
- Machine resume is not implemented: a dead run re-runs (the partial JSONL is the post-mortem record).
