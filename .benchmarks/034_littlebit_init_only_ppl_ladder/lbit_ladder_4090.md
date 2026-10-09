# Issue 036 T1/T2 — svd_lbit init-only PPL ladder (gemma-2-2b-it f16)

model: E:\git\riir-train\data\gemma-2-2b-it-f16.gguf | layers 26 | n_embd 2304 | q_dim 2048 | kv_dim 1024 | mlp_hidden 9216 | vocab 256000

corpus: E:\git\riir-infer\.raw\wikitext-2-raw-test.txt | windows 48 × ≤1024 (cap 48, non-overlapping) | eval tokens 49152 | nll 49104 | seed 0x36

PROVENANCE (best-effort, std-only): start loadavg unavailable | workers 6 | os windows — per-pass loadavg in the table

## PPL ladder

| arm | target | PPL | nll n | transform s | ppl s | loadavg |
|---|---|---|---|---|---|---|
| anchor | — | 18.6579 | 49104 | 1 | 3484 | unavailable |
| lbit1 | 1 | 813952160.3724 | 49104 | 86 | 3490 | unavailable |
| lbit1 | 0.55 | 2245863219.3102 | 49104 | 46 | 3485 | unavailable |
| lbit1 | 0.3 | 7994979669.2925 | 49104 | 25 | 3529 | unavailable |
| lbit1 | 0.1 | 288913933.7228 | 49104 | 7 | 3500 | unavailable |
| lbit2 | 1 | 419533636.0744 | 49104 | 453 | 3504 | unavailable |
| lbit2 | 0.55 | 1530740211.2472 | 49104 | 145 | 3417 | unavailable |
| lbit2 | 0.3 | 815248184.0904 | 49104 | 57 | 3179 | unavailable |
| lbit2 | 0.1 | 2732079981.8015 | 49104 | 17 | 3244 | unavailable |
| svd | 1 | 335223960.5128 | 49104 | 5 | 3267 | unavailable |
| svd | 0.55 | 1356306848.4400 | 49104 | 3 | 3262 | unavailable |
| svd | 0.3 | 2631147287.7076 | 49104 | 2 | 3286 | unavailable |
| svd | 0.1 | 12845814460.8908 | 49104 | 18 | 3431 | unavailable |
| rtn | — | 32672131.5960 | 49104 | 2 | 3347 | unavailable |

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
