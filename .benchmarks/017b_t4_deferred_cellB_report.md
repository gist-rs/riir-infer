# Bench 013 — T3 P3 V-cache reconstruction gate (kv_reconstruct_gate)

**Status:** COMPLETE — G3 PASS, G1 SKIPPED (G2-scaling posture; G1 record: Bench 013)

Box: 4090 workstation i7-13700K 16-core, CPU lane, AC; box exclusive (schtasks); T4 4-arm deferred mode; exe from fe75497  
Fixture: gemma-2-2b-it-f16, chat_probe, eval tokens [61440..65536), seq 4096, 1 chunks, teacher-forced NLL.  
Table: .benchmarks\012_kv_table_residual.bin (7348 rows, cal 61440) — T2's artifact, BLAKE3-verified.  

## P1 — G3 seam probe

- plain vs `NoVQuant` (the selector's default = the cache slice): **PASS** (65-position decode, logits `to_bits`).  
- recon-λ0 vs store-λ0 max \|Δlogit\|: **1.254e-4** (bound 0.05 — the rotation-rounding class).  

## P1b — deferred-lane probe (--recon deferred)

65-position decode on the real model + table.  

- deferred-λ0 vs eager-λ0: **PASS** (to_bits — the zero-λ law).  
- deferred-λ1 all-miss vs eager-λ1: **PASS** (to_bits — the miss law).  
- tracked λ1, deferred vs eager max \|Δlogit\|: **8.011e-5** (the regrouped-association class; bound 0.05).  
- tracked λ1, deferred vs STORE max \|Δlogit\|: **7.915e-5** (the full P3 class: regrouping + rotation rounding).  

## P2 — G1 paired store-vs-reconstruct

SKIPPED (--skip-g1): the G1 record is Bench 013's (T3, seq 1024, 0 flips at every λ); this run measures the G2 scaling axis only.  

### Retention walk (recon − store, by target-token frequency band × tracked)

| band | tracked | n | mean ΔNLL | mean \|Δ\| | flips |
|---|---|---|---|---|---|

## P3 — G2 tg64 paired interleave

2 interleaved triples over 4 arms (full-cache, eager-λ1, def-λ0, def-λ1); 4097-token prefill + 64 timed decode steps; median of per-pair medians; ratios are medians of per-pair ratios (the katgpt-rs `ab_timing` shape). Mode: **deferred (the T4 fused lane)**.  

| arm | median µs/step | tok/s | min µs | ratio vs full | prefill ms |
|---|---|---|---|---|---|
| full-cache | 143862 | 7.0 | 143414 | 1.000× | 543681 |
| eager-λ1 | 557841 | 1.8 | 557473 | 3.887× | 1347558 |
| def-λ0 | 151072 | 6.6 | 150218 | 1.050× | 556007 |
| def-λ1 | 161051 | 6.2 | 160914 | 1.123× | 582295 |

Box: 4090 workstation i7-13700K 16-core, CPU lane, AC; box exclusive (schtasks); T4 4-arm deferred mode; exe from fe75497. Recorded, not gated — the decision rule reads the WINDOW-EDGE deferred ratio: ≤ 1.20 re-fires the katgpt-core promotion; > 1.20 holds (riir-infer Issue 013 T4 pre-registration).  

## P4 — KV bytes/token record

- Full cache: 2 × 1024 × 4 B × 26 layers = **212992 B/token** (K + V, f32).  
- P3 (key-only): 1024 × 4 B × 26 = **106496 B/token** — the law's exact **50.0%** (`n_v/(n_kv+n_v) = 1/2`; gemma-2-2b is 8q:4kv at hd 256).  
- Sliding-window layers (window 4096): K and V are window-bounded equally, so the fraction holds at every context; below the window this lane's saving is uniform.  
- Instrument caveat: this lane still WRITES the raw V row at store (one memcpy/step, ~0.03% of step cost); the READ path is fully reconstructed — a production P3 cache drops the V allocation entirely, which is the recorded arithmetic.  

## Verdicts

- Read posture: **deferred (the T4 fused lane)**  
- **G3 (seam identity): PASS**  
- **Deferred-lane probe (λ0/miss to_bits + regrouping bound): PASS**  
- **G1 (reconstruct == store within rotation rounding): SKIPPED (--skip-g1; record: Bench 013)**  
- **G2 (tg64 read-path cost): recorded** (see P3)  
- **Bytes/token: 50.0%** (the law, recorded)  

A G1 FAIL means a convention/wiring bug (the wrong rotation subgroup, a stale token map), not a model effect — the reconstruction is deterministic algebra.  
