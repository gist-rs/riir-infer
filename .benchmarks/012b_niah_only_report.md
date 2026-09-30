# Issue 013 T2 — K=V+ λ ladder: gemma-2-2b decode (kv_plus_ladder)

seq_len 1024 | cal 61440 eval 12288 search 1024 tokens | top_k 8192 | table blake3-verified artifact, cal_tokens 61440

box: 4090 workstation i7-13700K, CPU lane, AC; sibling python trainer active (steady-state since 2026-09-29)

- G3 probe (hard): PASS | table LOADED (blake3-verified artifact, cal_tokens 61440)
- λ*: (none)
- grid: skipped (niah-only run — the ladder lives in the parent run's log)

**--niah-only run: the ladder/grid/validation sections below are ABSENT — those numbers live in the parent run's log + the Bench 012 doc.**

## NIAH (Bench-814 shape, 6 trials, direction-only)

| arm | median best rank | hits (rank 1) | mean answer NLL |
|---|---|---|---|
| f16 | 1 | 6/6 | 15.461 |
| k-0.00 | 37 | 0/6 | 44.285 |
| k-0.50 | 9 | 1/6 | 32.374 |
| k-1.00 | 5 | 0/6 | 26.392 |


---
*Measurement-only (issue 013 T2). The only claim under test is quality(K=V+) > quality(K=V); no parity claim vs full V. Promotion is katgpt-rs-side.*
