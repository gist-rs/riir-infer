# Bench 889 — Plan 622 T3.4: the four-arm acc-vs-K harness + the GOAT gate (ceiling / random-K floor / derived / FastKV-class)

**Status:** RECORD — T3.4 DONE; **the pre-registered GOAT gate (derived ≥ random-K on every cell, both axes, both postures) FAILED — the full-overlay promotion is REFUSED.** The failure is structured, not noisy: derived dominates the retrieval axis (9/10 cells) but loses the continuation-ppl axis at K ∈ {0.50, 0.75} in BOTH postures, plus one best-of-3 lucky-seed floor cell at K=0.0625. FastKV-class: ppl-competitive selection-only, structurally blind to the deepest needle at every K, and CATASTROPHIC under the derived-magnitude bias overlay (retrieval 0, ppl ~7). Verdict nuance is T4.2's to adjudicate.

**Date:** 2026-10-25 (zed-m3-continue)
**Plan:** [.plans/622_tokengate_keybias_pruned_prefill_lane.md](../.plans/622_tokengate_keybias_pruned_prefill_lane.md) T3.4
**Instrument:** `src/bin/tokengate_fourarm.rs` (new; `required-features = ["tokengate", "deltanet_inference"]`) + the T3.4 kernel half: the layer-L tap widened with `tap_window: Option<usize>` (`None` = the T3.3 mass sum; `Some(w)` = the FastKV-class max-pool over the last-w queries' per-query head-sum rows — ONE softmax-row helper, `attention_softmax_row_into`, now carries the kernel-exact form for both taps).
**Model:** `../riir-train/data/Qwen3.5-0.8B-Base-Q8_0.gguf` — qwen35-hybrid, 24 layers (GDN×2 / full-attn×2), n_head 8 / n_kv 2, head_dim 256, vocab 248320. The bench-888 restored copy.
**Corpus:** `../riir-train/data/chat_probe`.
**Suite:** the bench-888 NIAH rig (4 needles at depths {377, 746, 1115, 1484}, ctx 2048, EPS 1.0 nat/token) PLUS a 128-token corpus continuation teacher-forced after the answer — ONE decode pass carries both instruments (needle NLLs + continuation mean nats/token, ppl = exp). L = 15 (the bench-888 pin: mid full-attention). Scaffold = 4 sinks + last; prunable 2043.
**Postures:** `sel` = selection-only diagnostic (no bias, no scale anywhere — what the SELECTION alone buys); `gate` = the full T3 overlay (pruned-prefill steering through `KeyBiasLayout` + the T2.3 decode gate via `build_decode_gate_into`, magnitude × sequence-RMSNorm(score) rows, scale ≡ 1.0). The steering table must exist before layer L+1 runs, so the CPU harness builds it from a first (selection) pass and re-runs, ASSERTING the second pass selects the identical kept set (the tap is bias-blind by construction; the assert holds in all 20 gate cells). Bucket = None: no pads, the PAD_KEY_BIAS extension inert in this harness. `head_stride = prompt + decode steps` covers both consumers' reads (layers ≤ L read rows up to t_n = prompt_len — zero rows there, inert adds).
**Arms:** ceiling (no prune), random (seeded SplitMix64 Fisher–Yates through the SAME scorer seam via `random_scores_into`, best-of-3 seeds, **per-axis best floor** — the GOAT reads random's best retrieval AND best ppl over the reps: the maximally defensible floor), derived (the T3.3 mass tap), fastkv (the same tap at window 32; REPORTED, non-gating).
**Box:** M3 Max (m3), loadavg 2.43, mem free 93%, AC; release build; CPU forward. Main run total wall 4557.6 s, seed 622. Two supplementary single-posture re-runs (per-needle NLL detail for the K∈{0.0625, 0.75} sel cells, ~5 min) AFTER the per-needle progress lines were added to the bin — the main run's verdict numbers predate that edit and are unaffected (the print is the only change).

## The main grid (ctx 2048; ceiling needle NLL [0.790, 0.143, 0.009, 0.004], ceiling ppl 3.216)

### posture = sel

| cell | kept | retrieval | meanΔ nats | ppl | Δppl vs ceiling |
|---|---|---|---|---|---|
| random K=0.0625 (best-ppl rep) | 133 | 0.750 | 0.690 | 4.092 | +0.876 |
| derived K=0.0625 | 133 | 0.750 | 0.694 | 3.998 | +0.782 |
| fastkv K=0.0625 | 133 | 0.750 | 0.732 | 3.733 | +0.517 |
| random K=0.1250 | 260 | 0.750 | 0.686 | 3.768 | +0.553 |
| **derived K=0.1250** | **260** | **1.000** | **0.260** | **3.615** | +0.399 |
| fastkv K=0.1250 | 260 | 0.750 | 0.751 | 3.757 | +0.541 |
| random K=0.2500 | 516 | 1.000 | 0.222 | 3.591 | +0.375 |
| **derived K=0.2500** | **516** | **1.000** | **0.151** | **3.478** | +0.262 |
| fastkv K=0.2500 | 516 | 0.750 | 0.748 | 3.407 | +0.191 |
| random K=0.5000 | 1027 | 1.000 | 0.166 | 3.244 | +0.029 |
| derived K=0.5000 | 1027 | 1.000 | 0.048 | 3.342 | +0.127 |
| fastkv K=0.5000 | 1027 | 0.750 | 0.714 | 3.224 | +0.009 |
| random K=0.7500 | 1537 | 1.000 | 0.072 | 3.126 | −0.089 |
| derived K=0.7500 | 1537 | 1.000 | −0.003 | 3.306 | +0.091 |
| fastkv K=0.7500 | 1537 | 0.750 | 0.704 | 3.223 | +0.007 |

### posture = gate

| cell | kept | retrieval | meanΔ nats | ppl | Δppl vs ceiling |
|---|---|---|---|---|---|
| random K=0.0625 | 133 | 0.250 | 1.240 | 4.720 | +1.504 |
| derived K=0.0625 | 133 | 0.750 | 0.660 | 4.206 | +0.991 |
| fastkv K=0.0625 | 133 | **0.000** | 1.853 | **7.465** | +4.249 |
| random K=0.1250 | 260 | 0.500 | 1.056 | 4.227 | +1.011 |
| **derived K=0.1250** | **260** | **1.000** | **0.302** | **3.869** | +0.654 |
| fastkv K=0.1250 | 260 | **0.000** | 1.873 | **7.229** | +4.014 |
| random K=0.2500 | 516 | 0.750 | 0.665 | 3.742 | +0.526 |
| **derived K=0.2500** | **516** | **1.000** | **0.179** | **3.738** | +0.522 |
| fastkv K=0.2500 | 516 | **0.000** | 1.878 | **7.174** | +3.959 |
| random K=0.5000 | 1027 | 1.000 | 0.151 | 3.318 | +0.102 |
| derived K=0.5000 | 1027 | 1.000 | 0.091 | 3.614 | +0.399 |
| fastkv K=0.5000 | 1027 | **0.000** | 1.806 | **7.021** | +3.806 |
| random K=0.7500 | 1537 | 1.000 | 0.046 | 3.172 | −0.044 |
| derived K=0.7500 | 1537 | 1.000 | 0.053 | 3.573 | +0.357 |
| fastkv K=0.7500 | 1537 | **0.000** | 1.838 | **7.046** | +3.830 |

## The GOAT verdict (derived ≥ random, per axis, floor = per-axis best over 3 reps)

| posture | K | retrieval derived vs floor | ppl derived vs floor | cell |
|---|---|---|---|---|
| sel | 0.0625 | 0.750 vs **1.000** ✗ | 3.998 vs 4.092 ✓ | **FAIL** |
| sel | 0.1250 | 1.000 vs 1.000 ✓ | 3.615 vs 3.768 ✓ | pass |
| sel | 0.2500 | 1.000 vs 1.000 ✓ | 3.478 vs 3.591 ✓ | pass |
| sel | 0.5000 | 1.000 vs 1.000 ✓ | 3.342 vs **3.244** ✗ | **FAIL** |
| sel | 0.7500 | 1.000 vs 1.000 ✓ | 3.306 vs **3.126** ✗ | **FAIL** |
| gate | 0.0625 | 0.750 vs 0.250 ✓ | 4.206 vs 4.720 ✓ | pass |
| gate | 0.1250 | 1.000 vs 0.500 ✓ | 3.869 vs 4.227 ✓ | pass |
| gate | 0.2500 | 1.000 vs 0.750 ✓ | 3.738 vs 3.742 ✓ (0.004 margin) | pass |
| gate | 0.5000 | 1.000 vs 1.000 ✓ | 3.614 vs **3.318** ✗ | **FAIL** |
| gate | 0.7500 | 1.000 vs 1.000 ✓ | 3.573 vs **3.172** ✗ | **FAIL** |

**GOAT VERDICT: FAIL — the full-overlay promotion is refused by the pre-registered gate.** Per the Feature-Flag-Discipline law: a perf/quality gain on a biased answer is not a modelless gain, and a gate that fails is recorded, not argued away. The lane stays opt-in behind `tokengate`; T4.2 owns the verdict write-up with the axis split below.

## Findings

1. **The retrieval axis: derived dominates (9/10 cells), and the one loss is the floor's luckiest seed at the harshest K.** At K=0.0625 (keep 128 of 2043) EVERY arm drops needle 0 (the deepest, depth 377): derived Δ = 2.23 nats, fastkv Δ = 2.13, random's best-PPL rep Δ = 1.77 — but ONE of random's 3 reps retained all four needles (P ≈ 10%/rep — the floor caught its tail draw). At K ≥ 0.125 derived keeps all needles (retrieval 1.000) while random's floor needs K ≥ 0.25 to reach 1.000 reliably (gate posture: random 0.25→0.50→0.75→1.00 climbs with K; derived is 1.000 from K=0.125).
2. **The ppl axis: derived wins K ≤ 0.25 and LOSES K ∈ {0.50, 0.75} to random in BOTH postures — and random at K=0.75 sits BELOW the ceiling** (sel 3.126 vs 3.216; gate 3.172). Dropping 25% of mid-prompt rows denoises the continuation for ANY uniform half (the bench-888 −0.003 needle meanΔ at K=0.75 was the same effect on the needle axis); the mass-top selection instead concentrates kept rows in needle/sink/recent regions, and non-uniform coverage costs the filler continuation (+0.09..+0.36 vs random). This is a coverage-vs-salience tradeoff, measured cleanly by the two-posture design: sel shows it is the SELECTION alone (no bias involved).
3. **FastKV-class, the honest column:** (a) selection-only it is ppl-COMPETITIVE — best or second-best at 4/5 sel Ks (3.733/3.407/3.224/3.223), with Δppl-vs-ceiling as low as +0.007 at K=0.75; (b) it is STRUCTURALLY blind to the deepest needle — retrieval pinned at exactly 0.750 at EVERY K (needle-0 Δ ≈ 2.1–2.2 nats even keeping 1537/2043 rows): the last-32-query max-pool ranks the earliest needle below hundreds of later tokens, the window heuristic's recency bias; (c) under the derived-magnitude gate overlay it COLLAPSES — retrieval 0.000, ppl 7.0–7.5 at every K. Mechanism: the max-pooled score profile is extremely peaked (a few recent keys carry the RMS), so `magnitude × RMSNorm(score)` × 2.75 turns the additive bias into a hard mask over everything non-recent. **FastKV-class selection cannot ride the derived-magnitude bias law** — the T3.2 magnitude derivation (calibrated on upstream's mass-like statistics) transfers only to mass-shaped scores.
4. **The gate posture HELPS retrieval at small K and costs ppl at large K:** derived gate-vs-sel retrieval 0.75→0.75 (K=0.0625, no change), but random gate-vs-sel 0.25 vs 0.75 (K=0.0625) — wait, inverted: random's gate retrieval is WORSE (0.25/0.50/0.75/1.00 vs sel 0.75/0.75/1.00/1.00) — the random-score steering bias actively hurts the random arm's needle retrieval, while the derived arm's bias rows are self-consistent (selection order ≡ bias order) and never hurt retrieval. The overlay is only safe when the bias encodes the SAME ranking that selected the kept set — upstream's co-designed pair assumption, now measured.
5. **Two-pass gate wiring held:** all 20 gate cells selected identically across the steering-armed second pass (the assert never fired) — the tap is bias-blind as designed, and the prefill steering never moves selection.
6. **Compute note:** sel cell ≈ 56–71 s, gate cell ≈ 102–135 s (the steering re-run doubles the prefill; the CPU forward is GDN-recurrence-dominated — the bench-888 note). The K=0.0625 cells are the lane's actual target regime (4× row reduction above L) and run the FASTEST per cell in gate posture relative to full prefill.

## En-route

- The kernel half landed as the minimal seam widening: `attention_mass_received_into` gained the `window: Option<usize>` pooling arm + a shared `attention_softmax_row_into` helper (ONE home for the kernel-exact softmax row — the mass tap and the fastkv tap re-derive rows through it, so a kernel-form change cannot drift between the serving pass and a tap); `prefill_qwen_deltanet_pruned_into` + `prefill_attention_branch` gained `tap_window: Option<usize>`. All 10 prior call sites updated (None = byte-identical incumbent posture, pinned by the existing G1 batteries — plan622 pruned-prefill 6/6, bucketed 3/3, decode-fold 4/4, keybias-walk 6/6, tokengate lib 14/14).
- New G1 battery `tests/plan622_fastkv_g1.rs` (4 tests): determinism (run-to-run bits), window monotonicity (fastkv(w=1) ≤ fastkv(w=T) pointwise, EXACT — the per-query head-sums are computed identically), max ≤ sum (fastkv(w=T) ≤ mass pointwise), last-key window invariance (key T−1 is covered by exactly one query — its fastkv score equals the mass score bitwise at every window), w=1 conservation (Σ_k = n_head), clamps (0→1, >T→T, bitwise), GDN-layer refusal stays loud.
- `random_scores_into` (tokengate.rs): the seeded floor rides the SAME selection machinery (scaffold + top-K); unit test pins determinism, distinctness, scaffold immunity, K=full ≡ all-ones, K=0 ≡ zeros.
- This session's write_file tooling twice landed NEW files in sibling repos (riir-refine/src/bin/, katgpt-rs/tests/) before the path convention was fixed to root-prefixed paths — both strays MOVED to this repo the same session, both siblings verified clean (`git status`), no sibling WIP touched.

## The bench-888 Finding-4 ablation: hiddennorm at L=15 (the arm-isolation row)

The confound bench 888 named (mass ran at L=15 while hiddennorm ran only at the GDN layers 0/1) is now isolated — the mixer-agnostic arm at the ATTENTION layer L, sel posture (no tap needed; `--arms hiddennorm`):

| cell (sel) | kept | retrieval | meanΔ nats | ppl | Δppl vs ceiling |
|---|---|---|---|---|---|
| hiddennorm K=0.0625 | 133 | **1.000** | 0.338 | 4.002 | +0.786 |
| hiddennorm K=0.1250 | 260 | **1.000** | 0.332 | 3.518 | +0.303 |
| hiddennorm K=0.2500 | 516 | **1.000** | 0.229 | 3.442 | +0.227 |
| hiddennorm K=0.5000 | 1027 | **1.000** | 0.087 | 3.243 | +0.027 |
| hiddennorm K=0.7500 | 1537 | **1.000** | 0.078 | 3.195 | −0.021 |

**The ablation INVERTS one cell of the bench-888 story and strengthens the refusal:** hiddennorm at L=15 retrieves 1.000 at EVERY K — including K=0.0625, where the mass arm dropped needle 0 (its needle-0 NLL 1.40 vs the mass arm's 3.02, Δ 0.61 vs 2.23 — kept) — and its ppl beats derived at every K (3.195 vs 3.306 at K=0.75; 4.002 vs 3.998 at K=0.0625 ≈ tie). Reading: **the retrieval dominance is the MID-LAYER PLACEMENT, not the mass arm** — any reasonable scorer at L=15 ranks the needles above the filler; mass is not uniquely required, and the hidden-state norm is both more robust at the harshest K and cheaper (no tap pass). The GOAT verdict is UNCHANGED (it failed on the ppl axis and the floor's lucky cell, not on arm choice), but the owner's re-weigh should read the arm story as "place the gate at the mid full-attention layer; the scorer arm is a free parameter" — which also means the bench-888 pin "arm (a) mass" is the DEFAULT choice, not a measured necessity (the T2.5 evidence was confounded exactly as its Finding 4 disclosed; this row un-confounds it).

Run: `/tmp/p622t/t34_hiddennorm_ablation.{md,log}` — same bin at `--posture sel --arms hiddennorm --ppl-tokens 128`; box as the main run (loadavg 3.07, mem 93% free, AC, release).

## T4.2 disposition (the verdict write-up, folded here — one record, not two)

**The lane stays OPT-IN forever; promotion to default is REFUSED by policy** (the lossy-surface law — a lossy gate on the deployed serving path gates on deployed-path behavior, and the pre-registered four-arm GOAT failed on its own terms). Disposition components:

1. **The full overlay (selection + steering bias + decode gate) is refused as a default-on posture at ANY workload** — derived ≥ random-K failed on every-cell-both-axes in this harness, and the lossy-surface law requires per-family conditional retention evidence a single NIAH+ppl suite cannot supply.
2. **The retrieval-axis dominance (9/10 cells; 1.000 from K=0.125) is the measured basis for the lane's scoped value**: retrieval-shaped serving (needle/RAG-style prompts) at aggressive K. That is an OPT-IN deployment shape, not a default — the same table shows the ppl axis paying for it at K ∈ {0.5, 0.75}.
3. **The bias overlay is safe only when the bias ranking ≡ the selection ranking** (Finding 4: random-score steering hurt random's own retrieval; the derived pair is self-consistent). Any future arm must ship the pair co-designed — a selector cannot ride the T3.2 magnitude law unless its score profile is mass-shaped (FastKV-class collapse is the counterexample).
4. **The only re-open is the owner re-weighing the lossy law with this table in hand** (the Issue-971 absorption-counter posture). T4.3 (a trained gate via Plan 469) would re-run this harness as a fifth arm under the same gate — the trained-vs-modelless comparison the lane exists to adjudicate; until then the trained-gate accuracy claim stays unmeasured and unclaimed.
5. **The kernel capability (per-key bias) and the pruned-prefill quartet stay as landed** — the G1/identity batteries pin them byte-identical at the inert posture; the FAIL binds the GATE OVERLAY's promotion, not the substrate (T4.1's break-even row remains open for the quartet's systems value independent of the overlay).

## Provenance

- Run: `/tmp/p622t/t34_run.md` (stdout) + `t34_run.log` (stderr progress) — reproduced by `cargo run --release --features tokengate,deltanet_inference --bin tokengate_fourarm -- --context 2048 --needles 4` (defaults: k-fracs 0.0625..0.75, fastkv-window 32, ppl-tokens 128, rand-reps 3, posture both) from the riir-infer root with the sibling layout.
- Supplementary per-needle reads: the same bin at `--posture sel --k-fracs 0.0625 --rand-reps 3` and `--posture sel --k-fracs 0.75 --rand-reps 1` (the NLL vectors quoted in Findings 1/3).
- The bin + kernel half land in the same commit as this record; the plan tick carries the commit hash.
