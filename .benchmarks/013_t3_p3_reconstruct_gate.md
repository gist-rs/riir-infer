# Bench 013 — Issue 013 T3: the P3 V-cache reconstruction gate on gemma-2 (G1+G3 PASS at every λ; the tg128 KV bytes/token law holds at exactly 50.0%)

**Status:** COMPLETE — chained after T2's exit (the `riir_infer_t3_reconstruct`
scheduled task waited 399 min for the T2 process, verified T2's table artifact,
and ran the prebuilt exe — no second calibration). **G3 PASS, G1 PASS at all
three λ; G2 + bytes/token recorded.** The reconstruction-from-K lane is the
first P-product with a promotable quality surface on this model: its cost is
rotation rounding alone (mean |ΔNLL| ≤ 8.3e-6 against a 2e-3 bound, **0 flips
in 2046 scored positions at every λ**), not the V:=K destruction T2 measured.

**Box:** 4090 workstation (i7-13700K 16 cores, Windows 11, AC, CPU lane; the
sibling python trainer still held ~2.8 GB / part of the cores — the paired
interleave's per-pair ratios are the load-robust figure by design). Launch
06:31, exit 08:25 (rc 0).

**Method:** the pre-registered T3 protocol (issue 013, committed before the
run). `kv_reconstruct_gate` bin + `transformer::gemma2_vrecon`
(`HalfSplitRopeInverse` over THIS forward's own freq table — the NeoX
half-split convention, never katgpt-core's adjacent-pair `RopeAction`; ONE
one-directional inverse rotation per read, never a round-trip). Table = T2's
`012_kv_table_residual.bin` (BLAKE3-verified, 7348 rows). Eval slice [61440..
63488), 2 chunks, 2046 scored positions per arm — self-contained store-vs-
reconstruct pairing, valid whatever T2's verdict said.

**The gates (pre-registered):**

| gate | verdict | the numbers |
|---|---|---|
| G3 (wiring) | **PASS** | plain vs `NoVQuant` (default = cache slice): logits `to_bits`-identical, 65-position decode. recon-λ0 vs store-λ0 max \|Δlogit\| **1.254e-4** (bound 5e-2 — the rotation-rounding class, in family with the smoke's 1.25e-4) |
| G1 (the claim) | **PASS at every λ** | k-0: mean Δ +4.45e-7 / max 4.98e-5 / **0 flips**; k-0.5: −1.84e-7 / 4.42e-5 / **0 flips**; k-1: +5.17e-8 / 2.10e-5 / **0 flips** — all mean \|ΔNLL\| ~1000× inside the 2e-3 bound |
| G2 (read-path cost) | **recorded** | tg64 paired interleave (12 triples, median of per-pair medians): full-cache 265,772 µs/step · recon-λ0 **1.079×** · recon-λ1 **1.088×**. NOT the primitive-level 14–15× — the model-bound read is mild at seq 129 prefill + 64 decode |
| Bytes/token | **50.0% — the law exact** | full 212,992 B/token → P3 key-only 106,496 B/token (`n_v/(n_kv+n_v) = 1/2`; sliding-window layers keep the fraction) |

**Retention walk (recorded):** recon − store by target-frequency band ×
tracked/miss — every cell 0 flips, mean |ΔNLL| 2.2e-6…3.2e-6. The tracked-token
bands carry the table's E add-back at λ=1 and stay rounding-clean.

**Honest reading:**

1. **This is the promotable half of the P-family on gemma-2.** T1 (P1 mean
   removal) measured negative; T2 (P2 V:=K+) passed its formal gate against a
   catastrophic baseline and is not viable standalone; T3 (P3 reconstruct)
   passes with rounding-class cost ONLY — the served V is algebraically the
   stored V up to f32 rotation rounding, so the quality surface is the
   FULL-V surface, not the V:=K surface. What P3 buys: the V cache allocation
   drops (the recorded 50% bytes/token); what it costs today: +7.9–8.8% on
   the tg64 read path (G2, recorded) — that gap is T4's named target (the
   deferred restore / block angle-addition levers).
2. **The G2 baseline is mild but is a baseline, not a win.** The pre-
   registered expectation ("the naive read path is EXPECTED to be a
   regression at long context") was too pessimistic at this fixture's
   geometry — 8-9% at seq 129 + 64 decode steps. The long-context scaling
   (the inverse-rotation cost per READ is per-position-per-head) is the
   number T4 must re-measure at tg128/4k before anyone reads 8% as the
   production cost.
3. **Instrument caveats carried:** this lane still WRITES the raw V row at
   store (one memcpy/step); the READ path is fully reconstructed. A
   production P3 cache drops the V allocation entirely — the recorded
   arithmetic, not yet the measured allocation win.

**Files:** run report `.benchmarks/013_t3_reconstruct_report.md` (verbatim,
incrementally written — the T2 writer's Phase-D-only defect does not exist
here), run log `.benchmarks/013_t3_run.log`, table input T2's
`012_kv_table_residual.bin`. Instrument: `src/bin/kv_reconstruct_gate.rs` +
`transformer::gemma2_vrecon` (feature `vk_p3_tg`, measurement-only) at
`2ead866`. Protocol pre-registration: issue 013 §"T3 protocol".

**What remains in issue 013:** T4 — re-measure the P3 read path inside THIS
repo's real attention kernel at longer contexts (the G2 numbers here are the
baseline the kernel levers must beat), then the katgpt-rs-side promotion
decision with the loser demoted per the standing rule.
