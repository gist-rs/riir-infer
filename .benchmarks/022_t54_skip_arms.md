# Bench 022-t54 — the equal-FLOP skip-class arms (Issue 022 T5.4)

**Status:** RECORD — the T5.4 gate (the lane's one honest comparison debt, katgpt-rs Research 594 §7).

Date: 2026-10-05 21:38 → 2026-10-06 (overnight run, detached) · Box: Windows workstation, i7-13700K (16 threads), CPU lane ~1.8–2.1 tok/s, AC power, sleep disabled · Model: `Ternary-Bonsai-2-27B-PQ2_0.gguf` (the league model; 64 layers = 48 GDN + 16 attention, Hadamard-folded) · Corpus: `../riir-train/data/chat_probe` (13,121,614 chars → 4096 tokens → 8 chunks × ≤512; 4088 scored positions) · Commit: instrument `a7b080c` (src/bin/twt_skip_amputation.rs, required-features `twt_bonsai`) · The run's exact command + the M3 cross-reference: `.issues/022` T5.4 block.

## The gate

T5.4: the published zero-training skip/merge class must be IN the comparison or the claim cannot be said to beat it. Arms at EQUAL FLOP reduction (3 layers dropped each — the TWT PASS artifact's drop count {4, 12, 15}):

- **twt** — this lane's auditioned zero-training collapse: the ε=0.01 DP partition's dropped set, medoid byte-copied passthrough.
- **shortgpt** — ShortGPT (Block Influence + greedy elimination): BI through the shared capture pass, greedy removal to the same count.
- **hydra** — the SHIPPED katgpt-rs `hydra_budget` criterion (`calibrate_profiles` transcribed verbatim, pinned by a known-answer unit test against katgpt-rs's own module test): logit-lens DE profiles, equal-FLOP translation = |mean_de| ranking over non-backup layers. Native threshold rule disclosed (skips 0 at 0.01 — the rule's native skip set is empty here; the arm drops the 3 lowest |mean_de| non-backup layers instead).
- **random** — seeded floor control (seed 20261002).
- (LaCo-RDSC merge arm: already answered per-block by the T5.0e audition — mean/RDSC lost at EVERY mergeable block, so it cannot win end-task; recorded there, not re-run.)

## Instrument gates (all passed before any arm)

- **Parent cache**: self-generated over the REAL forward (one 4096-position pass; the M3's cache was not portable — cross-hardware argmax replay was rejected as unsound, NEON vs AVX2), 4088 positions, hit **0.4936**.
- **Glue parity**: empty skip set reproduced the parent's cached argmax **511/511 BYTE-IDENTICAL** — the amputation forward is the parent forward when nothing is dropped.
- Dropped-set disclosure: `[4, 12, 15] = [4=gdn, 12=gdn, 15=attn]` (2 GDN + 1 attention).

## Results (4088 scored positions each, teacher-forced top-1 agreement vs the parent's own argmax)

| arm | drop set | agreement | hit rate | first div | tok/s |
|---|---|---|---|---|---|
| **twt (this lane)** | [4, 12, 15] | **0.8919** (3646/4088) | 0.4861 | 2 | 1.88 |
| hydra (lens DE, equal cut) | [19, 20, 21] | 0.8642 (3533/4088) | 0.4883 | 0 | 1.90 |
| **random (seed 20261002, floor)** | [7, 5, 57] | **0.8437** (3449/4088) | 0.4758 | 2 | 1.91 |
| shortgpt (BI greedy, equal cut) | [0, 1, 29] | **0.0320** (131/4088) | 0.0296 | 0 | 20.99 |

`same_set_as_twt`: false for both criteria arms. Parent hit 0.4936; the twt/hydra hit rates sit at parent parity (0.4861 / 0.4883) — both survivors keep functional quality while their trajectories diverge (the two-column law: trajectory divergence overstates functional damage).

## Reading

1. **TWT wins the gate on this corpus** — +2.77 pt over the best published criterion (hydra 0.8642), and the selection DISAGREEMENT is total (no shared layer with either arm; shortgpt's drop set {0, 1, 29} vs twt's {4, 12, 15} vs hydra's {19, 20, 21}).
2. **⚠ THE FLOOR IS HIGH — the 4.7%-cut task is nearly floor-degenerate on this model class.** Random removal of 3 layers reads 0.8437 (hit-parity 0.4758) — the 64-layer quantized hybrid is that redundant at this depth. The honest decomposition: TWT beats the floor by +4.8 pt; hydra beats it by only +2.1 pt (the lens criterion adds little over random here); the ordering twt > hydra > random is REAL but the margins are single-digit points, not cliffs.
3. **ShortGPT's Block-Influence greedy is catastrophically anti-correlated with safety on this class**: its BI ranking picked layers {0, 1} (the first two) plus 29 — dropping the first two layers collapses the model to babbling (0.0320, first divergence at position 0, hit 0.0296) — **81 points BELOW the random floor**. A dense-model removal criterion does not transfer to the ternary GDN/attention hybrid; on this class its greedy actively seeks the worst layers (the BI meter is near-FLAT across layers — block_influence in the JSON ranges ~3863–4064, <5% spread — so the greedy's early picks ride noise, and position noise at the front of the network is fatal).
4. **Hydra's logit-lens DE is a real criterion** (0.8642, hit-parity 0.4883, the strongest published-class arm) but still loses to the auditioned surrogate by ~2.8 pt, and its first divergence at position 0 (vs twt's position 2) shows the gap opens immediately.
5. **The absolute-0.9 bar caveat (corpus, not construction)**: the M3's certified read was 0.9486 on ITS calibration corpus; this box reads 0.8919 on the chat_probe corpus — just under the pre-registered absolute bar ON THIS CORPUS, with the hit-rate column at parent parity. The gate's verdict is the WITHIN-RUN comparison (all arms on the same corpus/cache/protocol), and the cross-box corroboration (construction robustness) is recorded above.
6. **XMerge-class scope** (decided 2026-10-05, pre-adjudication): the gate compares selection criteria; XMerge's delta over ShortGPT removal is post-removal boundary reconstruction — a repair axis, not a selection criterion. Recorded as the explicitly scoped complementary axis in Research 594 §7's close-out line. Conditional: had shortgpt landed within a few points of twt, the refit delta would be material and earn its own arm; at 81 points BELOW random it is moot.

## Verdict

**T5.4 CLOSED — the lane wins the equal-FLOP comparison against the published zero-training class (twt 0.8919 > hydra 0.8642 > random 0.8437 >> shortgpt 0.0320).** Two honest negatives ride the win: (a) the 4.7%-cut floor is 0.84 — most of the agreement mass is model redundancy, not selection quality, so the lane's edge is the +4.8 pt over random, single-digit and real; (b) ShortGPT's criterion is worse-than-random on this class — the published class does not transfer unmodified to quantized ternary hybrids. The novelty wording in katgpt-rs Research 594 (decoder-LLM × quantized/ternary-GGUF × auditioned zero-training surrogate × DP partition × throughput gate) drops the T5.4 caveat; the XMerge-class scope note above replaces it. The pre-registered TWT absolute-0.9 agreement bar passed on the M3's corpus (0.9486) and misses by 0.008 on this box's chat_probe corpus — the miss is corpus-attributed (hit-rate parity both boxes) and disclosed, not silently absorbed.
