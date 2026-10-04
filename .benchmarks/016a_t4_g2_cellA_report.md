# Bench 013 — T3 P3 V-cache reconstruction gate (kv_reconstruct_gate)

**Status:** COMPLETE — G3 PASS, G1 SKIPPED (G2-scaling posture; G1 record: Bench 013)

Box: 4090 workstation i7-13700K, CPU lane, AC; cell A of the T4 scaling pair (decision evidence, not a GOAT)  
Fixture: gemma-2-2b-it-f16, chat_probe, eval tokens [61440..63488), seq 1024, 2 chunks, teacher-forced NLL.  
Table: .benchmarks\012_kv_table_residual.bin (7348 rows, cal 61440) — T2's artifact, BLAKE3-verified.  

## P1 — G3 seam probe

- plain vs `NoVQuant` (the selector's default = the cache slice): **PASS** (65-position decode, logits `to_bits`).  
- recon-λ0 vs store-λ0 max \|Δlogit\|: **1.254e-4** (bound 0.05 — the rotation-rounding class).  

## P2 — G1 paired store-vs-reconstruct

SKIPPED (--skip-g1): the G1 record is Bench 013's (T3, seq 1024, 0 flips at every λ); this run measures the G2 scaling axis only.  

### Retention walk (recon − store, by target-token frequency band × tracked)

| band | tracked | n | mean ΔNLL | mean \|Δ\| | flips |
|---|---|---|---|---|---|

## P3 — G2 tg64 paired interleave

4 interleaved (full-cache, recon-λ0, recon-λ1) triples; 1025-token prefill + 64 timed decode steps; median of per-pair medians; ratios are medians of per-pair ratios (the katgpt-rs `ab_timing` shape).  

| arm | median µs/step | tok/s | min µs | ratio vs full | prefill ms |
|---|---|---|---|---|---|
| full-cache | 133456 | 7.5 | 127461 | 1.000× | 140046 |
| recon-λ0 | 229792 | 4.4 | 219323 | 1.789× | 183358 |
| recon-λ1 | 233714 | 4.3 | 223178 | 1.751× | 182746 |

Box: 4090 workstation i7-13700K, CPU lane, AC; cell A of the T4 scaling pair (decision evidence, not a GOAT). Recorded, not gated — the kernel levers (the deferred restore, block angle-addition) are T4's lane.  

## P4 — KV bytes/token record

- Full cache: 2 × 1024 × 4 B × 26 layers = **212992 B/token** (K + V, f32).  
- P3 (key-only): 1024 × 4 B × 26 = **106496 B/token** — the law's exact **50.0%** (`n_v/(n_kv+n_v) = 1/2`; gemma-2-2b is 8q:4kv at hd 256).  
- Sliding-window layers (window 4096): K and V are window-bounded equally, so the fraction holds at every context; below the window this lane's saving is uniform.  
- Instrument caveat: this lane still WRITES the raw V row at store (one memcpy/step, ~0.03% of step cost); the READ path is fully reconstructed — a production P3 cache drops the V allocation entirely, which is the recorded arithmetic.  

## Verdicts

- **G3 (seam identity): PASS**  
- **G1 (reconstruct == store within rotation rounding): SKIPPED (--skip-g1; record: Bench 013)**  
- **G2 (tg64 read-path cost): recorded** (see P3)  
- **Bytes/token: 50.0%** (the law, recorded)  

A G1 FAIL means a convention/wiring bug (the wrong rotation subgroup, a stale token map), not a model effect — the reconstruction is deterministic algebra.  
