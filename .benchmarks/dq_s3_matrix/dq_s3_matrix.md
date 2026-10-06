# dq_s3_matrix — the dual-PTQ measurement (plan 618 S3 / issue 028 T4)

- **Lane (frozen):** cudarc per-token GEMV, ALL arms — kernel policy constant; the lane's int8 activation quantization is COMMON to every arm, so deltas isolate the weight format. NOT the 614 GEMM-prefill lane (cross-lane comparability approximate).
- **Corpus blake3:** `9367ebb9fe7a60df30f67f010d5052987fa87319e07efe611874d65ff87cb2ad` — arith 48 items (the 614 v2 corpus at hard=false), niah 16/len × 8 needles, lengths [1024, 2048, 4096]
- **Gen caps:** arith ≤ 256, niah ≤ 32 · bootstrap 10000
- **GPU:** NVIDIA GeForce RTX 4090, 24564 MiB, 23579 MiB
- **Packs:** decode `../riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf` · pf `../riir-train/data/Ternary-Bonsai-2-27B-Q4_K.pf.gguf` · q6 `../riir-train/data/Ternary-Bonsai-2-27B-Q6_K.sg.gguf`
- **Storage pairing:** base 6.7 GB · dual 21.1 GB · matched ≈20.7 GB (the control ~2% CHEAPER — the conservative direction, disclosed) · q4single 14.4 GB (the decomposition control)

## Cells

| arm | family | len | n | acc | TTFT p50 (ms) | TTFT p99 (ms) | decode tok/s |
|---|---|---|---|---|---|---|---|
| dual | arith | 0 | 48 | 0.9583 | 6022.0 | 6583.2 | 73.17 |
| dual | niah | 1024 | 16 | 1.0000 | 21959.7 | 22445.2 | 57.41 |
| dual | niah | 2048 | 16 | 1.0000 | 50552.9 | 51004.8 | 42.85 |
| dual | niah | 4096 | 16 | 1.0000 | 125810.2 | 126555.6 | 28.37 |
| matched | arith | 0 | 48 | 0.9583 | 8119.1 | 8595.5 | 34.96 |
| matched | niah | 1024 | 16 | 1.0000 | 29234.8 | 29820.6 | 30.65 |
| matched | niah | 2048 | 16 | 1.0000 | 65034.0 | 65869.4 | 26.06 |
| matched | niah | 4096 | 16 | 1.0000 | 158107.1 | 165519.7 | 19.29 |
| q4single | arith | 0 | 48 | 0.9792 | 6328.8 | 6850.5 | 44.12 |
| q4single | niah | 1024 | 16 | 1.0000 | 23007.7 | 23557.1 | 37.72 |
| q4single | niah | 2048 | 16 | 1.0000 | 53004.5 | 53500.3 | 30.48 |
| q4single | niah | 4096 | 16 | 1.0000 | 131769.4 | 132767.3 | 22.25 |
| base | arith | 0 | 48 | 0.9583 | 3788.0 | 4205.7 | 69.85 |
| base | niah | 1024 | 16 | 1.0000 | 15011.7 | 15587.7 | 54.08 |
| base | niah | 2048 | 16 | 1.0000 | 36145.8 | 37265.6 | 41.14 |
| base | niah | 4096 | 16 | 1.0000 | 98062.6 | 98976.8 | 27.22 |

## Paired vs base (bootstrap 95% CI on the mean paired difference)

| arm | family | len | recovery | CI lo | CI hi | n | note |
|---|---|---|---|---|---|---|---|
| dual | arith | 0 | +0.0000 | +0.0000 | +0.0000 | 48 | INADMISSIBLE base |
| dual | niah | 1024 | +0.0000 | +0.0000 | +0.0000 | 16 | INADMISSIBLE base |
| dual | niah | 2048 | +0.0000 | +0.0000 | +0.0000 | 16 | INADMISSIBLE base |
| dual | niah | 4096 | +0.0000 | +0.0000 | +0.0000 | 16 | INADMISSIBLE base |
| matched | arith | 0 | +0.0000 | +0.0000 | +0.0000 | 48 | INADMISSIBLE base |
| matched | niah | 1024 | +0.0000 | +0.0000 | +0.0000 | 16 | INADMISSIBLE base |
| matched | niah | 2048 | +0.0000 | +0.0000 | +0.0000 | 16 | INADMISSIBLE base |
| matched | niah | 4096 | +0.0000 | +0.0000 | +0.0000 | 16 | INADMISSIBLE base |
| q4single | arith | 0 | +0.0208 | +0.0000 | +0.0625 | 48 | INADMISSIBLE base |
| q4single | niah | 1024 | +0.0000 | +0.0000 | +0.0000 | 16 | INADMISSIBLE base |
| q4single | niah | 2048 | +0.0000 | +0.0000 | +0.0000 | 16 | INADMISSIBLE base |
| q4single | niah | 4096 | +0.0000 | +0.0000 | +0.0000 | 16 | INADMISSIBLE base |

## Verdict (pre-registered, plan 618 §S3.3)

- niah 1024: dual rec +0.0000 (CI +0.0000) · matched rec +0.0000 (CI +0.0000) · dual≥matched true · dual-CI>0 false
- niah 2048: dual rec +0.0000 (CI +0.0000) · matched rec +0.0000 (CI +0.0000) · dual≥matched true · dual-CI>0 false
- niah 4096: dual rec +0.0000 (CI +0.0000) · matched rec +0.0000 (CI +0.0000) · dual≥matched true · dual-CI>0 false
- niah 1024 direct dual−matched: mean +0.0000 CI [+0.0000, +0.0000]
- niah 2048 direct dual−matched: mean +0.0000 CI [+0.0000, +0.0000]
- niah 4096 direct dual−matched: mean +0.0000 CI [+0.0000, +0.0000]

(chance ≈ 1/8 (8-needle first-match); arith chance ≈ 0)

**NULL** — the matched-storage single ≥ dual everywhere. Container SHELVED per the pre-registered gate; T4's decomposition number stands on its own.

## Disclosures

- TTFT here is per-token-GEMV prefill wall — the kernel-maturity posture: walls track weight bytes (≈2.15×/3.05×); the paper's TTFT thesis needs a q4-class GEMM prefill arm and stays a kernel-build question.
- p99 rows print beside n — at n=16 the p99 IS ~max (the percentile lesson; tail support 1).
- Cross-lane: the 614 activation-phase cells (GEMM prefill, A8/A4 fake-quant) remain the long-context reference; this lane's arm deltas are weight-format-only by construction.
- matched-storage delta: the Q6_K control rides ~2% LESS storage than the dual pair — the conservative direction for the container's verdict.
- Machine resume is not implemented: a dead run re-runs (the partial JSONL is the post-mortem record).

## CORRECTED VERDICT (instrument-label fix, post-run)

The auto-generated verdict above printed **NULL**, but every cell is
INADMISSIBLE-at-ceiling (base arith 0.9583 > 0.95; base niah 1.0000) — the
verdict legs never had admissible ground to stand on. The bin's verdict gate
was repaired (admissibility-aware; commits land with S3b) and the honest
verdict per the S3.5 pre-registration is:

**RECORD-SATURATED** — zero admissible cells; the instrument cannot falsify
at this difficulty. THE LOAD-BEARING OBSERVATION IS EXACT-ZERO: dual vs base
paired diffs are +0.0000 on ALL 96 items (arith + niah × 3 lengths) — the
q4-prefilled KV produced IDENTICAL greedy decisions to the ternary prefill
on every item. matched ≡ base identically. q4single arith +0.0208 (1 item,
CI [0, 0.0625] — not significant). No arm loses a single decision to base;
no arm gains one (outside q4single's 1-item noise).

S3b (the pre-registered hard corpus) decides whether ANY headroom exists.

The perf axis (no admissibility needed): dual decode == base decode
(73.2 vs 69.9 tok/s — the shared ternary copy); dual prefill = 1.46× base
per token (22.0 vs 15.1 ms/tok at niah1024 — the q4 GEMV's byte cost, no
q4 GEMM exists); matched prefill 1.95× base (29.3 ms/tok, 20.7 GB/token).
Storage: dual 3.15× base. On parsimony the container buys nothing measured
and costs storage + prefill latency; the S3b accuracy read is the last open
question.
