# Bench 013 — T3 P3 V-cache reconstruction gate (kv_reconstruct_gate)

**Status:** COMPLETE — G3 PASS, G1 PASS (G2/bytes recorded)

Box: 4090 workstation i7-13700K, CPU lane, AC, box exclusive (T2 exited)  
Fixture: gemma-2-2b-it-f16, chat_probe, eval tokens [61440..63488), seq 1024, 2 chunks, teacher-forced NLL.  
Table: .benchmarks\012_kv_table_residual.bin (7348 rows, cal 61440) — T2's artifact, BLAKE3-verified.  

## P1 — G3 seam probe

- plain vs `NoVQuant` (the selector's default = the cache slice): **PASS** (65-position decode, logits `to_bits`).  
- recon-λ0 vs store-λ0 max \|Δlogit\|: **1.254e-4** (bound 0.05 — the rotation-rounding class).  

## P2 — G1 paired store-vs-reconstruct

| λ | ppl store | ppl recon | mean ΔNLL | mean \|Δ\| | max \|Δ\| | flips | verdict |
|---|---|---|---|---|---|---|---|
| — | **4.5901** | — | — | — | — | — | context (base) |
| k-0.00 | 1931443.0536 | 1931443.9140 | +4.45e-7 | 8.32e-6 | 4.98e-5 | 0/2046 | PASS |
| k-0.50 | 10519.8704 | 10519.8685 | -1.84e-7 | 5.29e-6 | 4.42e-5 | 0/2046 | PASS |
| k-1.00 | 476.8323 | 476.8323 | +5.17e-8 | 2.43e-6 | 2.10e-5 | 0/2046 | PASS |

Tolerances (pre-registered): mean \|ΔNLL\| ≤ 0.002, max ≤ 0.05 — the rotation-rounding class.  

### Retention walk (recon − store, by target-token frequency band × tracked)

| band | tracked | n | mean ΔNLL | mean \|Δ\| | flips |
|---|---|---|---|---|---|
| freq≤q1 | miss | 155 | +2.92e-7 | 3.24e-6 | 0 |
| freq≤q1 | tracked | 324 | -2.41e-7 | 2.69e-6 | 0 |
| q1<freq≤q2 | miss | 38 | +1.74e-7 | 3.11e-6 | 0 |
| q1<freq≤q2 | tracked | 194 | +6.11e-7 | 2.57e-6 | 0 |
| freq>q2 | miss | 35 | -4.08e-7 | 3.10e-6 | 0 |
| freq>q2 | tracked | 1300 | +2.14e-8 | 2.21e-6 | 0 |

## P3 — G2 tg64 paired interleave

12 interleaved (full-cache, recon-λ0, recon-λ1) triples; 129-token prefill + 64 timed decode steps; median of per-pair medians; ratios are medians of per-pair ratios (the katgpt-rs `ab_timing` shape).  

| arm | median µs/step | tok/s | min µs | ratio vs full | prefill ms |
|---|---|---|---|---|---|
| full-cache | 265772 | 3.8 | 258177 | 1.000× | 34214 |
| recon-λ0 | 285703 | 3.5 | 276841 | 1.079× | 35830 |
| recon-λ1 | 288489 | 3.5 | 279162 | 1.088× | 36448 |

Box: 4090 workstation i7-13700K, CPU lane, AC, box exclusive (T2 exited). Recorded, not gated — the kernel levers (the deferred restore, block angle-addition) are T4's lane.  

## P4 — KV bytes/token record

- Full cache: 2 × 1024 × 4 B × 26 layers = **212992 B/token** (K + V, f32).  
- P3 (key-only): 1024 × 4 B × 26 = **106496 B/token** — the law's exact **50.0%** (`n_v/(n_kv+n_v) = 1/2`; gemma-2-2b is 8q:4kv at hd 256).  
- Sliding-window layers (window 4096): K and V are window-bounded equally, so the fraction holds at every context; below the window this lane's saving is uniform.  
- Instrument caveat: this lane still WRITES the raw V row at store (one memcpy/step, ~0.03% of step cost); the READ path is fully reconstructed — a production P3 cache drops the V allocation entirely, which is the recorded arithmetic.  

## Verdicts

- **G3 (seam identity): PASS**  
- **G1 (reconstruct == store within rotation rounding): PASS**  
- **G2 (tg64 read-path cost): recorded** (see P3)  
- **Bytes/token: 50.0%** (the law, recorded)  

A G1 FAIL means a convention/wiring bug (the wrong rotation subgroup, a stale token map), not a model effect — the reconstruction is deterministic algebra.  
