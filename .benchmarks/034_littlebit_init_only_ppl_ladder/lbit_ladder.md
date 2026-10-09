# Issue 036 T1/T2 — svd_lbit init-only PPL ladder (gemma-2-2b-it f16)

model: ../riir-train/data/gemma-2-2b-it-f16.gguf | layers 26 | n_embd 2304 | q_dim 2048 | kv_dim 1024 | mlp_hidden 9216 | vocab 256000

corpus: .raw/wikitext2/wikitext2_test.txt | windows 48 × ≤1024 (cap 48, non-overlapping) | eval tokens 49152 | nll 49104 | seed 0x36

PROVENANCE (best-effort, std-only): start loadavg { 19.25 21.80 23.48 } | workers 10 | os macos — per-pass loadavg in the table

## PPL ladder

| arm | target | PPL | nll n | transform s | ppl s | loadavg |
|---|---|---|---|---|---|---|
| anchor | — | 18.6698 | 49104 | 9 | 6856 | { 14.09 21.11 23.32 } |
| lbit1 | 1 | 692310871.8024 | 49104 | 196 | 5896 | { 8.10 16.76 20.02 } |
| lbit1 | 0.55 | 2457940839.9765 | 49104 | 52 | 2685 | { 9.43 12.37 14.41 } |
| lbit1 | 0.3 | 7353469560.4803 | 49104 | 20 | 2692 | { 9.21 11.54 13.55 } |
| lbit1 | 0.1 | 313074062.3108 | 49104 | 6 | 3410 | { 17.65 19.64 19.55 } |
| lbit2 | 1 | 470759235.6663 | 49104 | 474 | 3424 | { 17.22 20.14 24.37 } |
| lbit2 | 0.55 | 1159402672.8585 | 49104 | 159 | 3371 | { 14.72 16.99 18.76 } |
| lbit2 | 0.3 | 829003364.0400 | 49104 | 59 | 3714 | { 35.32 32.20 29.28 } |
| lbit2 | 0.1 | 3808193279.9392 | 49104 | 21 | 3369 | { 18.57 24.10 29.21 } |
| svd | 1 | 347379214.7283 | 49104 | 5 | 3806 | { 68.60 60.08 53.04 } |
| svd | 0.55 | 1325874482.5759 | 49104 | 6 | 3563 | { 17.14 18.13 22.25 } |
| svd | 0.3 | 2549233907.7304 | 49104 | 2 | 4247 | { 18.31 20.73 25.76 } |
| svd | 0.1 | 13485447834.7762 | 49104 | 1 | 3650 | { 14.59 22.68 28.91 } |
| rtn | — | 34484807.5742 | 49104 | 1 | 3422 | { 16.25 30.65 43.79 } |

## Achieved bpw per tensor (by pass)

| pass | attn_wq | attn_wk | attn_wv | attn_wo | gate_proj | up_proj | down_proj |
|---|---|---|---|---|---|---|---|
| anchor | 16.000 | 16.000 | 16.000 | 16.000 | 16.000 | 16.000 | 16.000 |
| lbit1@1 | 0.999 | 0.999 | 0.999 | 0.999 | 1.000 | 1.000 | 1.000 |
| lbit1@0.55 | 0.550 | 0.549 | 0.549 | 0.550 | 0.550 | 0.550 | 0.550 |
| lbit1@0.3 | 0.300 | 0.299 | 0.299 | 0.300 | 0.300 | 0.300 | 0.300 |
| lbit1@0.1 | 0.099 | 0.099 | 0.099 | 0.099 | 0.100 | 0.100 | 0.100 |
| lbit2@1 | 1.000 | 0.998 | 0.998 | 1.000 | 1.000 | 1.000 | 1.000 |
| lbit2@0.55 | 0.550 | 0.550 | 0.550 | 0.550 | 0.550 | 0.550 | 0.550 |
| lbit2@0.3 | 0.300 | 0.297 | 0.297 | 0.300 | 0.300 | 0.300 | 0.300 |
| lbit2@0.1 | 0.100 | 0.099 | 0.099 | 0.100 | 0.100 | 0.100 | 0.100 |
| svd@1 | 0.989 | 0.993 | 0.993 | 0.989 | 0.998 | 0.998 | 0.998 |
| svd@0.55 | 0.546 | 0.542 | 0.542 | 0.546 | 0.547 | 0.547 | 0.547 |
| svd@0.3 | 0.295 | 0.293 | 0.293 | 0.295 | 0.295 | 0.295 | 0.295 |
| svd@0.1 | 0.089 | 0.090 | 0.090 | 0.089 | 0.095 | 0.095 | 0.095 |
| rtn | 1.014 | 1.014 | 1.014 | 1.016 | 1.014 | 1.014 | 1.003 |

Disclosures: ONE k_max Halko basis per tensor serves lbit1 + lbit2-path1 + svd at ALL targets (T0 — exact per-r SVD is a documented approximation class); lbit2 path 2 factorizes the per-target residual freshly; rtn arms ONLY at 1.0 (its floor is ~1.03 bpw, printed per shape above); eval is non-overlapping teacher-forced windows, f64 accumulation.

MEASUREMENT-ONLY (Issue 036 P0 law): the bin never writes the model; artifacts are this report + the JSON sidecar only.
