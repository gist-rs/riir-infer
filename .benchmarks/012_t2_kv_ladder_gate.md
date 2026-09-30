# Bench 012 — Issue 013 T2: the K=V+ λ ladder on gemma-2-2b (G-A PASS against a catastrophic baseline; the table refunds a third of the destruction, not the serve; G-D records overfit; NIAH crashed at build)

**Status:** COMPLETE-EXCEPT-NIAH → **COMPLETE (G-E recorded via the 012b rerun)** — the run executed cal → 4 ladder arms → 54-pass
schedule grid → validation, then **crashed at NIAH trial 0** (fixture-construction
bug: filler token budget 972 < target 1023, `vk_harness.rs::build_niah_trial`;
fix + `--niah-only` landed at `d9acd5a`, rerun `.benchmarks/012b_niah_only_report.md`).
All pre-registered gates are adjudicated — G-A..G-D from the run log (the report
writer runs at Phase D, after NIAH, so the structured report was never written —
the log carries every number the gates read), G-E from the 012b rerun. **G-A PASS** (λ*=1.00, mean paired
ΔNLL vs k-0 = −8.11487 < 0) — but the honest reading is that the ladder's claim
holds against a **catastrophic baseline** and the refund lands nowhere near the
f16 base: k-1.00 ppl **563.7 vs f16 6.0907 (92.6×)**. K=V+ as a standalone
V-cache-elimination posture is **not viable on gemma-2**; the recorded negative
for that posture is the finding. P3/T3 (reconstruct-from-K + E, ran next in
the chain, PASS) is a different product — its G1 tolerance is rotation-rounding class
(2e-3 mean |ΔNLL| vs the STORE arm), not the V:=K destruction class.

**Box:** 4090 workstation (i7-13700K 16 cores, Windows 11, AC power, CPU lane).
Launch state 12:51: 20.4 GiB free RAM of 32.5. ⚠ Load caveat: a sibling python
trainer (PID 29012) held ~14 cores for most of the run — cal ran at **3 tok/s**
vs T1's uncontended 9 tok/s (wall 20,559 s), and the eval arms at 2–3 fwd/s vs
T1's 2.2–4.4. Quality gates are deterministic numerics (unaffected); every wall
figure here is a contended-box figure. T1's box-state disclosure applies with
the trainer added to the concurrent-process list.

**Method:** the pre-registered T2 protocol (issue 013, committed before the run).
One process: (A) calibrate `forward_gemma2_f16_tapped` over [0..61440) → (B)
freeze + dump `FittedTokenTable::from_calibration(Residual, λ_js=0)` top_k 8192 →
`.benchmarks/012_kv_table_residual.bin` (**783,556,688 B, sha `3a0f5333d6bafd63…`,
BLAKE3-pinned; T3 reuses it — no second calibration**) → (C1) ladder on held-out
eval [61440..73728), 12 chunks × 1023, 12,276 scored per arm → (C2) schedule grid
(54 measured passes on the held-out search chunk [73728..74752), interactions
never summed — the chosen schedule re-measured as one pass) → (C3) validation of
the chosen schedule on the full eval → (C4) NIAH — **crashed here**.

**The ladder (the numbers):**

| arm | ppl | vs f16 |
|---|---|---|
| f16 (base) | **6.0907** | — (== T1's 6.0907 exactly — cross-run determinism PASS) |
| k-0.00 (V:=K) | 1,884,959.35 | ×308,568 (the tax) |
| k-0.50 | 11,233.31 | ×1844 |
| k-1.00 (λ*) | **563.71** | ×92.6 |
| k-sched (grid) | 1238.00 | ×203 (validation) |

**The gates (pre-registered):**

| gate | verdict | the numbers |
|---|---|---|
| Tax cross-check | **recorded — catastrophic-class** | Δppl(k-0 − f16) = **+30,948,056%**. The issue cited 2.5–3.1% from its source context; this fixture measures its own (pre-registered) and the V:=K tax on gemma-2 is total value destruction, not a tax. G-A's bar is therefore the weakest bar in the issue — read every pass against the ×92.6 residual |
| Consistency | **PASS** | f16 ppl 6.0907 == T1's 6.0907 (Δ 0.000%, same fixture/slices/protocol, cross-run) |
| G-C (bit-identity) | **PASS** | in-run G3 probe: the λ=0 hook's logits `to_bits`-identical to the direct V:=K copy across the 64-position probe |
| G-A (the claim) | **PASS** | λ* = 1.00, mean paired ΔNLL vs k-0 = **−8.11487** < 0. The table refunds a real fraction of the destruction (3344× ppl recovery k-0 → k-1). NOT a serve claim — see the honest reading |
| G-D (schedule transfer) | **recorded OVERFIT** | chosen schedule won −0.01643 mean ΔNLL on the search chunk, then read **1238.0 vs uniform λ=1's 563.7 on the full eval (2.20× worse)**. The single-layer search-chunk wins (worst −0.215 at layer 14) did not compose — exactly the interaction class the "never summed" re-measure exists for, and the composed −0.016 still inverted. The ladder's λ* stands; per-layer schedules on this fixture are search-chunk noise |
| G-E (NIAH) | **RECORDED (012b rerun) — direction holds** | Original run: build crash at trial 0 (`token budget 972 < target 1023` — `body_chars = seq_len·42/10 + 64` assumes 4.2 chars/token; the pool tokenizes at ~4.49; the shrink path absorbs overshoot only). Fix landed (`d9acd5a`: ratio-adaptive grow-retry + `--niah-only`), rerun off the saved table `.benchmarks/012b_niah_only_report.md`: **f16 median rank 1 (6/6 hits) · k-0.00 rank 37 (0/6, answer NLL 44.3) · k-0.50 rank 9 (1/6, 32.4) · k-1.00 rank 5 (0/6, 26.4)** — K=V+ strictly improves retrieval over k-0.00 on every axis (rank 37→5, NLL 44.3→26.4): the pre-registered direction-only requirement (no degradation vs k-0.00) PASSES, and the ladder's destruction ordering reproduces on the retrieval axis. 6382 s under the sibling trainer's load; 6/6 trials built past the exact crash point |

**Honest reading:**

1. **G-A PASS is the weakest form of the claim.** The gate as pre-registered
   asks quality(K=V+) > quality(K=V) — against a baseline that destroys the
   value content entirely (ppl 1.88M). Any λ>0 wins that bar; λ=1 wins it by
   3344×. What the gate does NOT claim — and what the numbers now measure — is
   viability: 92.6× off f16 is nowhere near a serve posture. The refund law's
   prediction (mean ρ(V−K) 0.49 → λ=1 halves the residual MSE) is CONSISTENT
   with the outcome (halving a catastrophic residual is still catastrophic),
   and the per-layer ρ_l(V−K) dashboard was lost with the report writer — the
   artifact (table bin) retains the fitted rows if the dashboard is ever
   wanted back.
2. **The tax cross-check's cited range was from another posture.** 2.5–3.1%
   described V-quantization-class taxes, not V:=K substitution. On this
   fixture V:=K is not a degradation, it is a destruction; the issue's own
   "this fixture measures its own" clause carried the day.
3. **G-D overfit is the second real finding.** The grid found per-layer
   structure on the search chunk (12 of 26 layers diverging from λ*=1,
   middle-layer band preferring λ ∈ {0, 0.5}) worth −0.016 composed — and the
   composed schedule then read 2.20× WORSE on validation. On a 1024-token
   search chunk, per-layer λ interactions are noise-shaped; the uniform λ*
   arm is the only defensible posture.
4. **Instrument gap that cost the NIAH phase:** the report writer runs at
   Phase D only — a crash in any phase loses the structured report AND the
   in-memory per-arm NLL detail (win shares, flip distributions, the ρ
   dashboard). The log's per-arm summary lines carried every gate input this
   time; the fix (incremental report write after each arm) is filed with the
   NIAH bug.

**Files:** run log `.benchmarks/012_t2_run.log` (verbatim, all arms + grid +
the crash); table artifact `.benchmarks/012_kv_table_residual.bin` (sha
`3a0f5333d6bafd63…`); T3's consumer log `.benchmarks/013_t3_run.log`; NIAH
rerun `.benchmarks/012b_niah_only_report.md` + `012b_niah_run.log` (the G-E
half, off the saved table). No parent-run report md (Phase-D-only writer —
see finding 4). Instrument:
`src/bin/kv_plus_ladder.rs` + `transformer::gemma2_ktov::KToVState` (feature
`fitted_v_tables`, measurement-only). Protocol pre-registration: issue 013
§"T2 protocol" (committed before launch).

**What remains in issue 013:** T3 (P3 tg128, running in the chain at this
doc's writing — self-contained gates, verdict next), T4 (the P1/P3 G2
kernels — the P1 half already moot after Bench 011's negative; T4 re-measures
the P3 read path inside the real attention kernel), and the NIAH follow-up
(grow-retry fix + `--niah-only` rerun off the saved table).
