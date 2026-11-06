# Issue 1179 — the qwen3.8 specdec verify lane: the ABOVE-SERIAL half of the uzu beat (owner directive: >121 tok/s on bmo)

**Status:** OPEN — filed 2026-11-06 (session zed-uzu-distill@amo, epoch 1793928926). The serial half lives in Issue 1178 (T2 claimed zed-8c98fa2a@m3 to 11-08); this issue owns the SPECDEC multiplier — the piece that turns a ~38-50 tok/s serial into the 121.1-class headline. No filed task owned it before this issue: 1178 names it ("ABOVE-SERIAL") and hands the distill feed (Research 378 T0(e)); riir-ai Issue 755 is the drafter-blocked chat lane this re-opens UNDER NEW REGIME. **Claude verdict round 1: REVISE, all revisions applied same session** (head-instance not family; the `suffix_repetition_length` confound named; G1 reference re-based to the batched kernel at k=1; G3 added; the STOP bar preconditioned on 1178's serial; the size-gap disclosure; Path B plan opens in parallel now). **Claude verdict round 2: AGREE** — the three non-blocking notes folded (the T2 k-invariance pre-check, the 2.5-continues vs ~3.2-beats bar clarification, the instance wording residue).
**Claimed:** zed-amo-specdec@amo, 1793929668, expires 2026-11-08T02:00:00+00:00
(T1 — the head-only acceptance price cell on amo Metal; T2-T5 stay unclaimed)

## The fire condition (Bench 1167 R2 decomposition, bmo M5 Ultra, 2026-11-06)

uzu 0.6.17 vendor-method reads **121.1 tok/s median** vs the published 129 (M5 Max) — REPRODUCED on our box. The number is NOT serial:

```
121.1 tok/s  =  3.18 tok/pass (shipped speculator, measured)  ×  ~38 serial-equiv passes/s
```

- uzu serial-equiv ≈ 38 tok/s ≈ 552 GB/s ≈ 67% of the serial roofline (the honest roofline the speculator multiplies; their R1 0.5.23 SERIAL cell read 52.9 = 94% — wheel-version-dependent).
- ours: decode 12.17 tok/s (Bench 950 re-anchor, 82.13 ms/step; pre-1176 was 8.61 — the league arms already bought ~29%).

**The 3.18 confound (verdict round 1):** the R2 vendor posture ran with `suffix_repetition_length=1` (0 is refused by the 0.6.17 API — the bench's one disclosed deviation). Whether that suffix arm contributes to the measured 3.18 is undisclosed by the wheel — and suffix/lookup is exactly the class refuted for chat text in-house (Bench 746/824). **This lane's reopen rests on the HEAD-ONLY number T1 measures**, never on 3.18 as quoted.

**Why this lane exists (the arithmetic gate):**
- Specdec alone on our serial: 12.17 × ~2.5-3 net ≈ **30-36 tok/s** — 3-4× under the bar.
- Serial alone (Issue 1178's full prize, ~38-55): still under the bar without the multiplier.
- **Only serial × specdec together reach >121.** Beat arithmetic: serial 45-55 + acceptance ≥3.2 + draft cost ≤6-8 ms/cycle → cycle ≈ 25-33 ms → 3.2-3.8/0.025-0.033 ≈ **100-135 tok/s**. The beat bar therefore DEPENDS on 1178 T2/T3 landing serial ≥ ~40 (soft dependency — T1 acceptance pricing below unblocks immediately; the e2e bar does not).

## Why the 4090 chat specdec closure does NOT block this (the reopen record)

The qwen3.8 chat specdec lane closed NEGATIVE on the 4090 (three-wall decomposition, riir-train Issue 493 + Benches 824/942/943): Wall 1 cycle cost (Q2_K drafter +48% slower draft), Wall 2 acceptance (z-lab head 2.648 max, training ladder monotone DOWN), Wall 3 compose (746/779 refuted); the ≥100 tok/s ask needed mean-len ≥4.0 = "a head 1.5× better than z-lab". The issue's own reopen conditions: **"a new head family or a target swap"**.

**How the conditions read today (verdict round 1 rewording):**
1. **Target swap — cleanly met.** The bmo dense regime makes draft cost nearly free: a ~1.1 GB draft at M5-Ultra bandwidth ≈ 1.3-1.5 ms/token → k=4-6 draft ≈ 5-9 ms against an 82 ms serial step (6-11% overhead), vs the 4090 chat regime where draft cost was ~50%+ of the cycle. The closure's Wall 1 does not transfer.
2. **Head — a new head INSTANCE above the named ceiling, not a new family.** z-lab are the DFlash authors (the league's DFlash2 port lineage), and uzu consumes the same DFlash/weaver draft tree (Research 378 T0(e)) — so uzu's speculator is the SAME family with a better-trained instance. The honest reopen form: the closure named acceptance (2.648) as the wall; an instance measuring above it satisfies the condition's intent. ⚠ The "1.2-1.4× the z-lab ceiling" comparison is **CROSS-FIXTURE** (3.18 = uzu's head, uzu's 4-bit target, T=0.95, 4096-token outputs — long outputs favour acceptance; 2.648 = a different fixture and setting) — never printed as a measurement. **T1 decides.**

Also in-house precedent: the 4090 DFlash2 loop banked **1.242×** (49.9 vs 39 spec-less, league chat row) — the verify-loop machinery class is proven in our stack; what is missing is the METAL DENSE wiring + a head with above-ceiling acceptance on OUR target.

**Modelless note (do not re-litigate):** n-gram/lookup specdec stays REFUTED for chat-class text — Bench 746 grid ("lookup empty on novel text") + Bench 824 Wall 3. Not re-opened. (Exact greedy specdec would be a pure speed lever with output-identity, but measured headroom on chat text is empty; the head path is the one with measured headroom.)

## The two paths to an above-ceiling head

**Path A — consume the vendor artifact (fastest unblock; owner-gated posture):**
trymirai ships the speculator WITH the checkpoint: `trymirai/Qwen3.8-27B-M` on HF, **license apache-2.0**, speculator artifact **1,106,748,808 B, blake3 `f6c67be2…9638748`** (Bench 1167 R2 FairnessManifest). The MIT uzu engine source (Research 378, pin `1866d189`) documents the consuming format (`encodable_block/weaver.rs` + `dflash.rs` + the speculator-stats surface). Work = a riir-infer loader arm for the speculator format + the verify loop. **A1/A8 lineage holds:** weights-from-disk, never committed to any repo, blake3-pinned at load, the pin recorded beside every quoted cell. A loader reading a frozen vendor artifact serves the modelless mandate — no training happens in this repo.
⚠ **The unmeasured premise (T1):** their head was distilled against THEIR mirai-m 4-bit target; acceptance against OUR Q4_K_M GGUF target is UNMEASURED. Same-family, close-quant (their own card: "comparable to Unsloth UD-Q4_K_M") — plausible transfer, never assumed.

**Path B — train our own head (the durable answer; plan opens NOW, in parallel):**
An EAGLE3/MTP-class draft head for Qwen3.8-27B, distilled in **riir-train on the 4090** (the standing GPU lane; riir-train 493 v1-v4 is the old family's training ladder — the new-instance rung is the same unit with the new target class). **Per the standing no-training-deferral rule, T5's PLAN opens now in parallel — only the 4090 RUN waits on the T1 numbers.** Path B is also the durable answer even if Path A passes: the vendor head is tuned to THEIR quant. Minted as an artifact; weights never enter repos; consumed by this lane's loader identically to Path A.

## Machinery inventory (consume, do not rebuild)

| piece | where | posture |
|---|---|---|
| `SPEC_MAX_K = 8` vocab-handle seam | `riir-infer-gpu/src/ternary_deltanet_gpu_forward.rs:1694` behind the `speculative_decode` feature | exists (GDN lane) |
| p-ROW batched verify family | `riir-infer-gpu/src/cudarc_kernels/attention.rs` (Issue 742 T9.9, "the Q4_K speculative verify port") | CUDA posture |
| multi-query verify attention | `attention_cubecl/mod.rs` (`attention_decode_folded`, seq_len_q > 1) | CubeCL |
| qwen3.8 DFlash2 + regime gate + accept calibration | riir-ai i1156 wave (`dspark_drafter.rs`, `dflash_config.rs`, `regime_gate.rs`, riir-gpu `relaxed_accept_calibration`) | CUDA/4090 lane, 1.242× banked |
| speculative fill (2-box form) | `riir-infer-dist/src/spec_fill.rs` (Plan 623 T5.1) | dist lane |
| **MISSING on Metal dense** | the verify pass on the qwen38_wgpu dense path (a k-position batched forward — can ride the existing prefill/tensor mm2d machinery + per-position logits), KV rollback, accept/reject sampling (T=0.95 top-k=20), the draft loader | **this issue** |

## Preregistered gates (verdict round 1 form)

- **G1 correctness — the reference is the SAME batched verify kernel at k=1.** The accepted greedy stream is **byte-identical to a serial run through the same batched kernel with k=1** — isolating the specdec loop as the only difference. The incumbent serial-matvec lane is a DIFFERENT kernel (Metal reduction-order near-tie class): any batched-k vs serial-matvec walk delta is a kernel-order fact, not a specdec bug — flips are COUNTED and DISCLOSED with their top-2 logit margins (the Issue-940 first-index/order law), never silently absorbed, and this definition is fixed BEFORE T2 (never loosened after a failure). The sampled arm (T=0.95 top-k=20) uses **standard lossless rejection sampling — required**; a relaxed/typical-acceptance rule would inflate tok/pass and quietly break distribution preservation, so it is refused by construction. T1 additionally records whether uzu's own acceptance rule is lossless (read from the MIT source at pin `1866d189`).
- **G2 perf:** the vendor-method re-cell on bmo (the Bench-1167 harness twin: 1342-in prompt, T=0.95 top-k=20, 4096 out cap ×3, exclusive GPU via the queue, PROVENANCE + box-state lines mandatory, FairnessManifest rows). **FairnessManifest carries the SIZE-GAP disclosure:** ours 16.81 GB vs their 14.53 GB ≈ 16% more bytes per pass under the same method — a real handicap, disclosed, bar unchanged (the owner directive is the raw number). Intermediate bar **≥100 tok/s**; **BEAT bar > 121.1 median, same methodology**. **Precondition (verdict round 1): the intermediate bar and its loud STOP fire only once Issue 1178 has landed serial ≥ ~40 tok/s** — before that, the arithmetic guarantees ~30-36 and a below-bar read is a DEPENDENCY result, not a lane failure.
- **G3 no-regression (added, verdict round 1):** feature OFF — the serial lane stays byte-identical (league walk pins) and its decode tok/s stays within noise of Bench 950. The incumbent never pays for the opt-in.
- **G4:** no new steady-state hot-path allocations.
- **Feature posture:** `qwen38_specdec` opt-in, PROMOTE-ONLY env per the env-switch law; incumbent byte-identical when off. Promotion = GOAT + owner verdict.

## Tasks

- [ ] **T1 — acceptance price cell (Path A gate, unblocks NOW — speed-independent, HEAD-ONLY):** load the vendor speculator (blake3 `f6c67be2…9638748`) into a head-only harness — our loader has no suffix tree, so the measurement is naturally head-only — and run draft+verify acceptance against OUR Q4_K_M target on the league fixture (`uzu_t2_prompt.txt`) + the vendor prompt, greedy AND T=0.95 reported separately (greedy is the fairer cross-engine comparison). Record uzu's own acceptance rule (lossless or not) from the MIT source at pin `1866d189`. **Decision gate: ≥2.5 tok/pass head-only on our target continues Path A; below → Path A closes with the record, Path B (already planned) takes the 4090 run.** ⚠ **Verdict round 2: 2.5 is the Path-A CONTINUATION bar, not the beat bar** — the beat arithmetic needs ~3.2 at serial 45-55 to clear 121; at 2.5, Path A clears only if serial passes ~55, so a passing T1 makes Path A an interim lever while Path B's run decides the beat.
  **— PROGRESS 2026-11-06 (Bench 952, d3c5da8 + post-fix): harness + loader + probes LANDED; uzu format fully decoded (uint4-affine + incoherence fold verified 8e-6; acceptance rule confirmed LOSSLESS; `suffix_repetition_length` confound NEUTRALIZED — repetition-penalty ring at penalty 1.0 = no-op). FIVE bugs fixed (double offset, fold columns, KV norm+rope, per-head rope, **qkvg section-major slicing — the verdict reviewer's root cause**). Pieces VERIFIED against their CPU engine at the pin: premix 11.000-vs-10.98, conv_pre 69.066-vs-69.05, L0-with-fix shortcut cos 0.9992 / branch cos 0.9891, feature capture 7-decimal logits. **First acceptance: mean j = 1.083** (24 cycles, fixture, greedy, argmax chain-15; histogram 22×1 + 2×2; G1-T1 PASS) — below the 2.5 gate. **Claude verdict round 2: AGREE** — with four next-session conditions (Bench 952 has the ordered checklist): (1) q/k/v dump-vs-prepare parity FIRST on the vendor target; (2) per-layer branch-cos check (L0 branch cos 0.9891 is 10× off the shortcut — hunt norm eps/rope base/GQA/gate if it decays); (3) rank-0 hit rate per draft position + more cycles; (4) the vendor chain-only cell decides confound-vs-bug (their j≈1 ⇒ confound; their ≥2.5 ⇒ residual bug, gate unchanged). Lease held (zed-amo-specdec).**
- [ ] **T2 — the verify pass on Metal dense:** k-position batched forward reusing the prefill/tensor machinery + per-position logits readback; G1 greedy-identity harness vs the batched-k=1 reference (definition fixed above, before any run). **Pre-check (verdict round 2, guards the G1 definition): before the harness runs, verify the batched kernel is k-INVARIANT per row — row r's logits at k=1 must match k=k_max EXACTLY. The mm2d path may pick a different tile shape or split-K at k=1 vs k=4-8, which would change each row's reduction order with batch size and break the reference's isolation; if they diverge, pin ONE tile config for verify.**
- [ ] **T3 — draft loop + accept/rollback + the sampled arm:** KV discipline at the accept boundary, T=0.95 top-k=20 **lossless rejection sampling**, the speculator-stats readout (tok/pass, passes/s) matching the vendor harness's own instrumentation.
- [ ] **T4 — e2e vendor-method re-cell on bmo + the record:** G2 bars + precondition above; row lands as WATCH/owner class in the league uzu block (the SCORED opponents remain llama.cpp + vLLM — this lane's bar is the owner directive, not a scoreboard gate; the Bench-831 format-cross law applies until a same-format cell exists).
- [ ] **T5 — Path B head-distillation PLAN (opens NOW, in parallel per the no-training-deferral rule):** plan-first in riir-train, 4090 GPU, the new-instance rung; **the 4090 RUN waits on the T1 numbers** (a passing T1 deprioritizes the run, never the plan); cross-link the artifact pin here.
- [ ] Cross-links at every landing: 1178 (serial — the beat arithmetic needs both), 1168 (host sync — the resident-greedy loop is the natural verify-loop host), riir-ai 755 (the drafter-blocked lane this re-opens), Bench 824/942/943/950/951/1167, Research 378 (T0(e) trie-tree feed).

## Boundary

riir-infer owns forward/verify/loader (this issue). riir-train owns any trained head (T5). riir-gpu/riir-ai own acceptance-calibration consumption. The league scoreboard stays llama.cpp-anchored. Weights never committed (A1/A8). Qwen3.8 base + the vendor checkpoint + speculator are Apache-2.0; the uzu loader reference is MIT — attribution notes in the loader arm, fixtures original.
