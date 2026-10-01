# Bench 023 — Plan 614 T5 / Issue 026: the DQ phase-sensitivity matrix on the 4090 (A2+A4 × prefill/decode activation fake-quant)

**Status:** RECORD — run LANDED (EXIT0, all gates PASS) and every axis reads **INADMISSIBLE** by the frozen base-only rule (base arith 0.9583 > 0.95; base NIAH pooled 0.9896 > 0.95): no directional gate from this instrument at these corpora. The REPORT-ONLY damage tables are the run's value (a2 prefill/both devastate NIAH ~4× below base; a2 decode halves arith; A4 near-clean) — and the instrument itself is what stands: 9/9 gates green across two independent full runs that agree byte-for-byte per item.

## Pre-registration (frozen before any accuracy cell)

- Plan 614 round-3 AGREE freeze (`d3300f2`); corpora deterministic (arith-CoT 48 with the 4-shot completion prefix + exact-answer parse — v2 gold = standard precedence, `7c578b7`; NIAH 8-needle/one-queried over the inline paragraph bank), BLAKE3-frozen before cell 1: **corpus `6f3c6f02f4dd0fa4b3e77f656795311624238126f9387b6aa8a3f41d7f04c53d` · model `b4f6ab953ef6d1452682bd861b7a1711a4a9945a74ca292659803978fd2550f6`**.
- Admissibility window 0.25 ≤ acc(base) ≤ 0.95 per axis, decided by base accuracy ALONE before any phase comparison; label precedence INSTRUMENT-FAIL > INADMISSIBLE > SATURATED > HIT/REVERSED/NULL; R reported, never tested.
- **LANE RECORD (frozen): the fallback is operative** — shipping A8 prefill in every cell + the dec_a8 control (D1's "non-int8 folded prefill lane" was found not to exist at T2; the cudarc lane always activation-quantizes). The lane-acceptance predicate ran first (base arith 0.9583 — above the window edge, recorded, run proceeded per the frozen classification ladder).
- Cells: base(1)/base(2) (G-i4 byte-stability) + {pf_aq, dec_aq, both_aq} × {a2, a4} + dec_a8 (control, |Δarith| ≤ 2 items).

## Box state

4090 workstation (i7-13700K, RTX 4090 24 GB), AC, GPU-EXCLUSIVE per the AGENTS rule (compute-apps probe: GUI-only at launch; the bin refuses co-resident compute). `RIIR_PREFILL_CUDA_GRAPHS=0`, `BONSAI_GGUF=E:/git/riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf`, detached via the guarded schtasks task `dq614_matrix` (log `F:/wt/dq614-matrix2.log`, report `F:/wt/dq614-matrix2/dq_phase_matrix.md`, done-marker `EXIT0`). Forward load 201 s; run wall ~4h40m (07:46–12:25 ICT). **Determinism is double-sourced**: the v1 run (01:44–06:02, killed at the final gate by the control defect below) and this v2 run measured base(1), base(2), pf_aq[a2], dec_aq[a2], both_aq[a2], pf_aq[a4], dec_aq[a4], both_aq[a4] — every arith item and every NIAH length cell REPRODUCED byte-identically across the two independent runs (different processes, same frozen corpus).

## Result — the instrument landed; every axis INADMISSIBLE at these corpora

| cell | arith acc | NIAH per len (4096 / 8192 / 16384) | prefill fq launches | decode fq launches |
|---|---|---|---|---|
| base | **0.9583** (46/48) | 1.000 / 0.969 / 1.000 | 0 | 0 |
| pf_aq[a2] | 0.3333 | **0.219 / 0.281 / 0.250** | 69 632 | 0 |
| dec_aq[a2] | **0.5000** | 1.000 / 0.875 / 0.906 | 0 | 919 552 |
| both_aq[a2] | 0.0417 | **0.281 / 0.250 / 0.156** | 69 632 | 3 474 688 |
| pf_aq[a4] | 0.8542 | 1.000 / 1.000 / 1.000 | 69 632 | 0 |
| dec_aq[a4] | 0.9583 | 1.000 / 1.000 / 1.000 | 0 | 922 112 |
| both_aq[a4] | 0.8750 | 1.000 / 1.000 / 1.000 | 69 632 | 848 128 |
| dec_a8 (control) | 0.9583 | 1.000 / 0.969 / 1.000 | 0 | 900 096 |

Verdicts (the runner's own, per the frozen D6 table): `a2.decode_heavy`, `a2.prefill_heavy`, `a4.decode_heavy`, `a4.prefill_heavy` — all **INADMISSIBLE** ("no gate from that axis; the window verdict is decided by base accuracy alone"). Paired stats as printed (REPORT-ONLY, never gates): a2 prefill-heavy Δpf=0.7396 vs Δdec=0.0625, R=0.085, CI [0.5729, 0.7708]; a2 decode-heavy R=0.733, CI [−0.0208, 0.3542]; a4 decode-heavy R=0.000, CI [0.0208, 0.1875]; a4 prefill-heavy Δpf=Δdec=−0.0104 (both armed cells ABOVE base), R=NaN.

Gates: G-i1 knob-off counters == 0 PASS · G-i2 exact per-phase counts from actual lengths PASS (0 mismatches across all 9 cells × 2 runs) + the PHASE-MATCHED positive control PASS (each armed cell's logits differ from base in its armed phase) · G-i4 base twice byte-stable PASS (incl. the decode-phase FNVs) · positive-control FNV arm PASS · **dec_a8 control |Δarith| = 0 items (gate ≤ 2) PASS** — the shipping-A8 prefill + A8-fq decode posture is behaviorally indistinguishable from base on both corpora at item granularity.

## Reading (honest, per the pre-registered outcome table)

- **The instrument stands; the corpora do not bind it.** The a07fffd `reset_state` fix lifted base arith from the leak-contaminated 12.5% to the model's genuine 95.8% — above the 0.95 window edge — and pooled NIAH sits at 0.9896. Both axes therefore read INADMISSIBLE and Issue 026 T3's directional assertions receive NO GATE from this run. The remedy for a gating run is a NEW pre-registration with harder corpora (more ops / longer chains on arith; more needles or adversarial haystacks on NIAH) — never a protocol edit of this freeze.
- **The report-only tables still carry signal** (recorded, no gate): at the 2-bit-class A2 grid, PREFILL-only injection collapses NIAH retrieval to ~0.25 pooled (base 0.99) while decode-only leaves it at 0.93; decode-only halves arith (0.50 vs 0.96) while prefill-only takes it to 0.33; combined injection is the worst on both axes. Direction on the prefill-heavy axis (Δpf ≫ Δdec, R=0.085) is the paper's reversal SIGN, at a larger magnitude than the source's 1.1–4.1× band — but with both axes inadmissible this stays an observation, not a confirmed transfer. At A4 (16-level affine) damage is nearly absent (arith −0.10 pf / 0.00 dec; NIAH flat): the phase-sensitivity phenomenon is a 2-bit-tier floor phenomenon on this artifact.
- **A4 decode-heavy CI [0.0208, 0.1875]** excludes 0 — but the axis is INADMISSIBLE by the frozen label precedence, so this reads as a report-only observation (Δdec − Δpf > 0 favors decode damage on arith even at A4), never a HIT.
- **dec_a8 PASS at |Δ|=0**: the fallback lane satisfies its pre-registered control exactly — shipping-A8 prefill + A8-fake-quant decode is item-indistinguishable from base on 48 arith + 96 NIAH items.

## The instrument defects this run paid for (all fixed at source, one commit each)

1. **`a07fffd` — the GDN recurrent state leaked across ALL items** (the M3 session's fix, upstream): every item after the first started from the previous item's final state; degradation ACCUMULATED across the run (arith fell 4/5 → 2/43 within base(1); base(2)'s NIAH@4096 collapsed to 12/32). All pre-fix accuracy readings are superseded.
2. **`23bbff5` — the G-i2 positive control was PHASE-BLIND**: the v1 run measured every cell cleanly then FATALED at the final gate because the control hashed the prefill's final logits for decode-only cells, which the D1 boundary keeps identical to base by construction — structurally unsatisfiable. The control is now phase-matched (prefill FNV for pf arms, first-decode-step FNV for decode arms). The v1 FATAL teardown also HANGS (RAM 4.7→8.8 GB over ~10 min, GPU released, process stuck unwinding; killed by hand) — the EXIT0 path exits cleanly; the error-exit path still needs its own fix (follow-up).
3. **`3cf9ae9` — clippy discharge** (the plan T4 4090-clippy leftover): 7 pre-existing mechanicals; the two seed regroups are VALUE-PRESERVING leading-zero pads — this run's corpus blake3 reproduces the frozen `6f3c6f02…` byte-exact.
4. **The collision class, twice**: 00:43 an unguarded second chain launch truncated the first lane check's log (~1.5 h destroyed); 07:39 the v2 re-run launched by THIS session died silently (EXIT-1, no message) under the twin session's concurrent worktree reset+rebuild — a shared-worktree-mutation variant. All detached stages now carry the ALIVE guard (`E:/git/_sync/dq614_{chain,matrix}.cmd`, machine-local, live-tested), but the residual lesson stands: **two sessions drove one run lane concurrently and the run paid for it** — a run lane needs ONE owner at a time, and the handoff summary's "active plan state" must name the owning session.

**Twin-duplicate record (Batch-169 class)**: two idle-loop sessions independently diagnosed + fixed the same phase-blind control ~20 min apart (`23bbff5` origin-canonical vs this session's `5977565`, died in the reset per the origin-canonical precedent; its non-overlapping value re-landed as `3cf9ae9`).

## Standing rule (Issue 026 T5, now attached)

Any future weight format publishes its per-phase R before default promotion — a format that damages decode ≫ prefill at equal bpw is misallocating bits where chat users feel it. This run ARMS nothing (no axis admissible); the rule rides the instrument, not this run's verdict. A gating re-run needs a NEW pre-registration with corpora inside the admissibility window.

Session: riir-infer-030-dq-matrix-rerun (+ the twin idle-loop session's 23bbff5/matrix2 run, which is the run recorded here)
