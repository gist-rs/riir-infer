# dq_s3_matrix — the dual-PTQ measurement (plan 618 S3 / issue 028 T4)

- **Lane (frozen):** cudarc per-token GEMV, ALL arms — kernel policy constant; the lane's int8 activation quantization is COMMON to every arm, so deltas isolate the weight format. NOT the 614 GEMM-prefill lane (cross-lane comparability approximate).
- **Corpus blake3:** `33c8f4aefdd4fc631cdfab477f751d026b4aaa413d31f70bc394e3c7c5c09b35` — arith 48 items (the 614 v2 corpus, hard=true), niah 16/len × 10 needles, lengths [2048]
- **Gen caps:** arith ≤ 256, niah ≤ 32 · bootstrap 10000
- **GPU:** NVIDIA GeForce RTX 4090, 24564 MiB, 23521 MiB
- **Packs:** decode `../riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf` · pf `../riir-train/data/Ternary-Bonsai-2-27B-Q4_K.pf.gguf` · q6 `../riir-train/data/Ternary-Bonsai-2-27B-Q6_K.sg.gguf`
- **Storage pairing:** base 6.7 GB · dual 21.1 GB · matched ≈20.7 GB (the control ~2% CHEAPER — the conservative direction, disclosed) · q4single 14.4 GB (the decomposition control)

## Cells

| arm | family | len | n | acc | TTFT p50 (ms) | TTFT p99 (ms) | decode tok/s |
|---|---|---|---|---|---|---|---|
| dual | arith | 0 | 48 | 0.7500 | 6585.2 | 7917.6 | 68.57 |
| dual | niah | 2048 | 16 | 1.0000 | 53297.5 | 53847.6 | 40.70 |
| matched | arith | 0 | 48 | 0.7500 | 8897.2 | 9448.2 | 33.13 |
| matched | niah | 2048 | 16 | 1.0000 | 68625.5 | 69491.0 | 24.72 |
| q4single | arith | 0 | 48 | 0.7708 | 6552.9 | 7153.1 | 43.79 |
| q4single | niah | 2048 | 16 | 1.0000 | 53088.8 | 53557.6 | 30.46 |
| base | arith | 0 | 48 | 0.7500 | 3962.1 | 4647.9 | 68.92 |
| base | niah | 2048 | 16 | 1.0000 | 36209.4 | 36612.4 | 40.94 |

## Paired vs base (bootstrap 95% CI on the mean paired difference)

| arm | family | len | recovery | CI lo | CI hi | n | note |
|---|---|---|---|---|---|---|---|
| dual | arith | 0 | +0.0000 | +0.0000 | +0.0000 | 48 |  |
| dual | niah | 2048 | +0.0000 | +0.0000 | +0.0000 | 16 | INADMISSIBLE base (ceiling/floor) — excluded from the verdict |
| matched | arith | 0 | +0.0000 | +0.0000 | +0.0000 | 48 |  |
| matched | niah | 2048 | +0.0000 | +0.0000 | +0.0000 | 16 | INADMISSIBLE base (ceiling/floor) — excluded from the verdict |
| q4single | arith | 0 | +0.0208 | +0.0000 | +0.0625 | 48 |  |
| q4single | niah | 2048 | +0.0000 | +0.0000 | +0.0000 | 16 | INADMISSIBLE base (ceiling/floor) — excluded from the verdict |

## Verdict (pre-registered, plan 618 §S3.3)

- niah 2048: dual rec +0.0000 (CI +0.0000) · matched rec +0.0000 (CI +0.0000) · dual≥matched true · dual-CI>0 false
- niah 2048 direct dual−matched: mean +0.0000 CI [+0.0000, +0.0000]

(chance ≈ 1/8 (8-needle first-match); arith chance ≈ 0)

**NULL** — the matched-storage single ≥ dual everywhere. Container SHELVED per the pre-registered gate; T4's decomposition number stands on its own.

## Disclosures

- TTFT here is per-token-GEMV prefill wall — the kernel-maturity posture: walls track weight bytes (≈2.15×/3.05×); the paper's TTFT thesis needs a q4-class GEMM prefill arm and stays a kernel-build question.
- p99 rows print beside n — at n=16 the p99 IS ~max (the percentile lesson; tail support 1).
- Cross-lane: the 614 activation-phase cells (GEMM prefill, A8/A4 fake-quant) remain the long-context reference; this lane's arm deltas are weight-format-only by construction.
- matched-storage delta: the Q6_K control rides ~2% LESS storage than the dual pair — the conservative direction for the container's verdict.
- Machine resume is not implemented: a dead run re-runs (the partial JSONL is the post-mortem record).
