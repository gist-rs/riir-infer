# Bench 016 — Issue 013 T4 (scaling half): the P3 naive read-path cost explodes with context — 1.08× @129 → 1.79× @1025 → 3.82×/3.96× @4097; the pre-registered promotion rule fires HOLD (katgpt-core `fitted_v_reconstruct` stays opt-in; the deferred-restore lever is the remaining promotion gate)

**Status:** COMPLETE — both cells rc=0. **The decision rule fired:** window-edge
ratio 3.82/3.96 ≫ the ≤ 1.20 promotion bar. The katgpt-rs-side promotion of
`fitted_v_reconstruct` is **HELD** on this evidence; the lossy-free 50% KV-bytes
law stands, but the naive read path's cost is context-multiplied and the
window edge — where the memory win matters most — is where it is worst. The
named lever (deferred restore by linearity, then block angle-addition) is the
remaining lane: when it re-measures ≤ 1.20 at the window edge, promotion fires
per the same rule.

**Box:** 4090 workstation (i7-13700K 16 cores, Windows 11, AC, CPU lane). The
two cells ran back-to-back in one scheduled task (launch 21:32, cell A exit
22:27, cell B exit 00:28). Cross-cell ABSOLUTE figures are not comparable
(cell A's box was quieter than T3's trainer-loaded G2 — full-cache 133 ms/step
at 1025 vs 266 ms/step at 129 in T3); every decision figure is the
within-cell paired ratio, per the pre-registered discipline.

**Protocol:** the pre-registered T4 scaling protocol (issue 013, committed
before the run). `kv_reconstruct_gate --skip-g1` (G1's record is Bench 013's;
G3 still ran and still gated — PASS in both cells at 1.254e-4), T2's table
artifact, arms {full-cache, recon-λ0, recon-λ1}, the paired-interleave shape.
En-route instrument fix (`2300457`): the timing cache was sized
`seq_len + 64` while the G2 loop walks `prefill + BOS + decode` positions —
cell A overflowed by exactly ONE position and both cells segfaulted
(0xC0000005) on the first attempt; the block now sizes for the G2 geometry
explicitly.

**The curve (median µs/step, ratio vs the same-cell full-cache control):**

| context | full-cache | recon-λ0 | ratio λ0 | recon-λ1 | ratio λ1 |
|---|---|---|---|---|---|
| 129 (T3/Bench 013, 12 pairs) | 265,772 | 285,703 | 1.079× | 288,489 | 1.088× |
| 1025 (cell A, 4 pairs) | 133,456 | 229,792 | **1.789×** | 233,714 | **1.751×** |
| 4097 (cell B, 2 pairs) | 149,956 | 550,358 | **3.821×** | 560,263 | **3.957×** |

Read the pair counts honestly: cell B's 2 pairs give a median of two per-pair
ratios — enough for a 3.8× verdict against a 1.20 bar, not for fine
distinctions (the lever's own bench will re-run the full 12-pair protocol).

**Reading:**

1. **The cost is per-context-row and the value is per-context-row — they
   scale together, but not equally.** The full-cache read amortizes against
   the fixed per-step FFN compute; the reconstruction adds rotation work for
   EVERY cached row on EVERY decode step. The naive read path's ratio grows
   ~linearly in context; at the gemma-2-2b sliding-window edge it is ~4×.
2. **This does not touch the G1/G3 quality surface** (0 flips at every λ,
   Bench 013; G3 PASS in both cells here) nor the 50.0% bytes/token law
   (re-confirmed in both cells). The HOLD is a COST verdict on the naive
   read path, exactly the axis the katgpt-rs primitive G2 already failed
   (14–15× RopeAction / 1.6× table-driven) — the model-bound lane now
   confirms the shape at production geometry.
3. **The promotion gate's remaining lane is concrete:** the deferred restore
   by linearity (Σ_p w_p·v̂_p + Σ_s W_s·E[s] with the row index pre-resolved;
   katgpt-rs Bench 895 Addenda I/II measured its primitive-level cost at
   +1.5–3.6% test-local) regroups the sum so the per-read rotation
   amortizes; block angle-addition (exact re-anchoring every N positions) is
   the second lever. Either landing re-measures THIS curve; ≤ 1.20 at 4097
   fires the promotion.
4. **What was NOT measured:** context beyond the window (global layers
   reading > 4096 rows) — out of scope per the pre-registration (the naive
   CPU path is not the production shape; the levers change the curve). Cell
   B's prefill figures (587 s full-cache vs 1,335/1,599 s recon) record the
   same cost shape on the prefill side.

**Files:** run log `.benchmarks/016_t4_g2_run.log` (box state + both cells);
cell reports `016a_t4_g2_cellA_report.md` / `016b_t4_g2_cellB_report.md`.
Instrument: `src/bin/kv_reconstruct_gate.rs` at `2300457` (+ the `--skip-g1`
posture, this session). Protocol pre-registration: issue 013 §"T4 protocol".

**What remains in issue 013:** T4's kernel half — the deferred restore inside
THIS repo's attention loop (its last per-position scalar add belongs in the
softmax-weight loop), re-measuring this curve; then the katgpt-rs-side
promotion re-fire per the ≤ 1.20 window-edge rule.
