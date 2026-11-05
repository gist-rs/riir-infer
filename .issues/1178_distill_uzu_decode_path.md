# Issue 1178 — distill the uzu Metal decode path: close the 4.4× dense decode gap (Bench 1167 datum)

**Status:** OPEN — staged 2026-11-05; the first task is a riir-refine `distill` verdict (deep read), not a code edit.

## The fire condition (measured 2026-11-05, Bench 1167, bmo M5 Ultra)

Same box, same model family (qwen3.8-27B), same quant class (4-bit): uzu decodes **52.9 tok/s** serial (≈770 GB/s effective ≈ **94% of the serial roofline**) vs our qwen38 dense wgpu lane **12.05 tok/s** (≈175-197 GB/s ≈ 24%) — **0.23×, a 4.4× gap**. Prefill is nearly closed (993 vs 1156.9 = 0.86×, the 1176 promoted posture). The decode gap is the ground; owner directive: "i hate to lose".

This is NOT a mystery class — riir-infer 1168 already triaged the same class on the Bonsai lane ("the per-token host-sync serialization is the measured deficit", kernel-variant family measured-closed). What we lack is the OPPONENT'S mechanism: how uzu structures a decode step to extract 94% of roofline.

## Prior art (dedup entry — do NOT re-mine these)

- **uzu #1022 weaver** → `adjacent-step-kernels-fuse-mode-specialized` (B299, entries_gpu_134): mode-specialized merged decode kernels, −15 dispatches. THE closest law — our deficit is dispatch-count × per-dispatch cost.
- **uzu #1023** low-batch matmul → owned band (B107-class gpu_20 family + B307-era reads).
- **uzu #1030** → `contraction-partials-sum-once-at-drain` (B307 face).
- **uzu gpu_20** `packed-total-order-multi-field-argmax` (B107).
- The 11-04 watch intake carries uzu **Trellis-GEMM + 2-bit (Dash/MXU)** leads `#900/#1054/#1061/#1068/#1070/#1071` — GEMM/prefill/2-bit class, NOT the decode-step dispatch class; complementary, do not fold.

## Tasks

- [ ] **T0 — distill verdict (riir-refine, kernel_opt route)**: clone `trymirai/uzu` (MIT verified) into `.raw`, deep-read the decode-step path: (a) dispatches per decode step per layer (count vs our ~64×k), (b) command-buffer strategy (captured/compressed/re-encoded per step? — the cubecl indirect-command-buffer-replay law from the 10-30 watch is the known Apple hazard), (c) their GEMV/quant-kernel families for 4-bit weights (mirai-m format), (d) host-sync pattern (per-step submit vs batched), (e) their specdec draft interface (the 129-headline mechanism, even though the wheel refuses it). Verdict per the distill skill: GOAT candidates × dedup × the 1168-measured-class cross-check. Pin + .raw removed at close.
- [ ] **T1 — map laws to our decode loop**: the 116 ms/step (c5d35bb2 posture) decomposed the same way Bench 937 did for prefill — a decode frame-census (attn/GDN/FFN/embed/small-stages/dispatch-gap shares) on bmo, so the distilled laws land on measured shares, not vibes. Instrument: extend the Bench-937 census pattern to decode.
- [ ] **T2 — price cells** (bmo quiet windows; the i1167 rig-job lessons apply: PATH=/opt/homebrew/bin, no fetch in job context, worktree pre-created): each distilled mechanism priced against the 12.05 baseline with G1 token-identity pins (greedy walk digit-exact vs the c5d35bb2 posture).
- [ ] **T3 — GOAT gate + consume**: e2e decode A/B on bmo (≥1.5% bar, preregistered), winner joins `production` via the one-row act; loser demoted with the record.
- [ ] Cross-links: feeds **1168** (host-sync triage — same class, Bonsai lane; shared mechanisms consume once), **1177 item 2** (dense-f16 decode GEMV price cell, leased to zed-83e79a9e@amo to 2026-11-07 — coordinate, don't duplicate), and the league uzu block (a decode re-cell after any promoted gain re-reads the 0.23× row).

## Honest ceiling note

uzu at 94% roofline means the MAX prize is ~4.3× (12 → ~52 tok/s serial at this quant). The 129 headline is specdec-only (wheel-refused, Bench 1167); if our MTP/specdec lane (riir-ai Issue 755 drafter-block; Bench 924 direction) ever arms on this family, that's the multiplier above serial — separate lane, this issue is the serial close.
