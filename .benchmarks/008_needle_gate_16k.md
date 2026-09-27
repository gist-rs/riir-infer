# Bench 008 — multi-needle KV-eviction gate @16K, qwen35-hybrid CPU lane (Issue 012 T2)

**Status:** RECORD — 16K primary matrix COMPLETE (deferred protocol, budget 25%); 50% cell + 64K are recorded follow-ups (compute arithmetic at the bottom).

**Pre-registered before any budget arm ran** (riir-infer Issue 012 T2):

- Model / lane: `Qwen3.5-0.8B-Base-Q8_0.gguf` (qwen35 hybrid, 24 layers = 18 DeltaNet + 6 full-attention, n_kv_head=2, head_dim=256, ctx 262144, rope θ=10M) on the CPU hybrid lane, `--release`, batched prefill.
- Why this model and not Ternary-Bonsai-2-27B-PQ2 (the issue's preference): the 27B ternary lane has no batched prefill (its projections are per-token matvecs) and no f32 matmat path — a 64K prefill is hours per arm on this box. The 0.8B qwen35 checkpoint is the SAME arch family (the wiring target), 64K-in-distribution (ctx 262K), and its prefill fits the box. The arithmetic is recorded in the issue.
- λ* = 1.0, β = 0.1, W = 256 — frozen from the 8K pilot grid (λ ∈ {0.5, 1.0, 1.5}: retrieval 0.625 / 0.625 / 0.625, meanΔ +0.760 / +0.751 / +0.751 — flat; λ=1.0 is the middle of the plateau and the Bench-894 grid shape).
- EPS_NATS = 1.0 nats/token (retained ⟺ per-needle NLL delta vs full ≤ 1.0). G1 bar: retrieval ≥ 7/8 = 0.875 (Bench 894's "≥ full − 1/16" analog).
- Protocol: DEFERRED eviction (`defer_until = prompt_len`) — the prompt (haystack + question) prefills into the full cache; the first compression fires at decode start, on evidence that includes the question's queries. Rationale: the streaming protocol was measured first and is a recorded NEGATIVE — no query attends a mid-haystack needle while its evidence window is live, so every policy (differential, max-recent, usage-rate) evicts the needles before the question arrives: needle-row survival 0.000 at 2K and 8K pilots. The deferred protocol is the regime Bench 894's synthetic win describes (needle evidence must be able to compete at eviction time) and the SnapKV/H2O evaluation shape. Streaming stays available (`--defer 0`).
- Arms: full reference (armed, budget = usize::MAX — T3-bit-identical to the unarmed path, pinned by test) × differential (λ*=1.0) / max-recent (λ=0, bit-identical reduction) / usage-rate / seeded random (seed 1337), at 25% and 50% of context. n_sink = 4 (the kv_sink_window default). Cadence 512.

## Box state

- shikuwa: i7-13700K (8P+8E, 16 threads), 32 GiB RAM, Windows 11 / MSYS, AC power, no GPU compute used (CPU-only lane).
- Concurrent load during the run: sibling agent sessions active (compile traffic only, no sustained compute).
- Per-arm wall at 16K: ~1300-1355 s (prefill ~1300 s of it; the O(N²) attention Phase B at 16K × 6 layers dominates).

## Results (16K context, 8 needles, budget 25% = 4096 slots, deferred protocol)

| arm | retrieval | mean Δ nats | needle-row survival | generic NLL | evicted rows |
|---|---|---|---|---|---|
| full | 1.000 | +0.000 | 1.000 | 3.001 | 0 |
| diff λ=1.0 @25% | **0.750** (6/8) | **+0.715** | 1.000 | 3.012 | 73,896 |
| maxrecent @25% | 0.750 (6/8) | +0.714 | 1.000 | 3.011 | 73,896 |
| usage @25% | 0.500 (4/8) | +1.083 | 1.000 | 3.129 | 73,896 |
| random @25% | 0.750 (6/8) | +0.843 | 1.000 | 3.054 | 73,896 |

Per-needle Δ (differential): `[0.007, 0.004, 0.508, 0.899, 1.011, 1.550, 0.812, 0.931]` for depths 1.8K → 14.3K. Wall 6802 s total (5 arms; full 1325 s, budget arms 1345-1416 s). Run log: this bench's companion `needle_16k_v2.log` (machine-local); JSON report committed at `needle_gate_16k/needle_gate_report.json`.

## Verdict (Issue 012 T2 — the model-bound G1)

1. **G1 bar (retrieval ≥ 0.875) FAILS for every policy at 25% budget.** No policy keeps 7/8 needles within 1.0 nat at 4× compression. The primitive's synthetic numbers (0.945 at 25%) do NOT transfer to this model/regime at the pre-registered bar.
2. **differential ≡ max-recent here** (meanΔ +0.715 vs +0.714; per-needle deltas agree to ~3 decimals): the λ common-mode correction does not change the selection when the observation window's evidence is dominated by the SAME question-attended rows. Bench 894's separation (0.945 vs 0.680) required hub-heavy filler where a hub's window-max crowds out needle evidence; natural-text filler at W=256 does not produce that regime. The λ-insensitivity was already visible in the pilot grid (0.5/1.0/1.5 → identical retrieval).
3. **usage-rate is the measured loser** (4/8, +1.083) with a clean monotone depth pattern: `cum_mass/age` scores young rows high, so OLD needle rows evict first (Δ 3.90 at depth 1.8K → 0.01 at 14.3K). The shipped H2O-class score is anti-recency in exactly the wrong direction for old-cold needles.
4. **`beats_random_prompt_pin`: a MEAN win, not a count win** — differential's meanΔ (+0.715) is better than random's (+0.843) and its worst needle is less bad (1.55 vs 1.53... comparable), but both retain 6/8. On the scored bar (strictly beat random at matched budget) this is NOT a clean pass; record as TIED on retrieval, WON on mean damage.
5. **Trap-4 (T4) does not fire**: generic-continuation NLL ratio diff/maxrecent = 1.000 — the differential policy's damage is needle-specific, not generic-text degradation, at this budget.
6. **Runaway (T4)**: every arm capped at gen=32 including full — no relative runaway signal at this cap (a longer cap is a follow-up; the promotion-grade runaway gate on a sealed eval remains katgpt-rs-side and UNRUN).

**Issue 012's promotion posture (katgpt-rs side): the model-bound gate returns a split verdict — the wiring is sound (T3 bit-identity pinned), needle retention works (survival 1.000), usage-rate loses, trap-4 is quiet, but the G1 bar fails at 25% and differential does not separate from λ=0 on this corpus. NOT a promotion result.** The recorded regime gap: the synthetic gate's hub-heavy fixture is where the primitive wins; natural text needs the regime re-created (hub sentences planted as distractors) before a differential-vs-max-recent separation can even be measured — that is the follow-up, with the 50% + 64K cells.

## What the pilots already established (measured, pre-16K)

1. **Streaming eviction is a recorded NEGATIVE for mid-context needles** (2K + 8K pilots): with eviction live during the haystack, needle-row survival reads 0.000-0.036 under every scored policy — a needle's specificity ages to −∞ after 2W queries (W=256) and no query attends the needle mid-haystack, so it evicts like any other low-evidence row. Retrieval then runs only through the DeltaNet layers' recurrent state (which never evicts): 0.25-0.5. This is the primitive's design boundary, now measured on a real model — the differential table is a RECENCY instrument; anything needing older evidence must use the deferred protocol.
2. **The deferred protocol retains the needle rows** (survival 1.000 at 25% budget, 8K pilot): the question's queries re-arm needle specificity and the compression keeps them. The open question at that point is the MODEL's tolerance: at 8K/25%, meanΔ = +0.75-0.76 nats (retrieval 0.625 at EPS 1.0) — the 0.8B base model's copy behavior degrades even with the needle rows resident, because 75% of the supporting context is gone.
3. **The random null is competitive at small scale** (8K: random meanΔ +1.066 vs diff +0.751): with no hub structure in natural-text filler, uniform random retention is a strong baseline. The Bench-894 regime (hub-heavy, needle-vs-hub-window) is what separates the scored policies; whether real text at 16K+ reproduces it is exactly what the 16K table measures.

## T3 (bit-identity) — PINNED, with the bug the gate caught

- Synthetic (lib test, both postures): armed-with-headroom ≡ unarmed `to_bits`, decode and chunked prefill; chunking bit-transparent vs the legacy whole-prompt path. Hardened after the incident below: EVERY dense projection is perturbed (fill_seeded), not just the embedding.
- Real model (bisect bin, pre-removal record): legacy whole-prompt vs chunked ≡ 0/248320 logit bits, chunks=1 and chunks=4, on Qwen3.5-0.8B-Base-Q8_0.
- The bisect caught a real wiring bug the zeros-weighted synthetic test had hidden: the batched-deltanet Phase 4 must OVERWRITE the normed hidden with out_proj and add the PRE-NORM residual — the first version added onto the normed base (layer 0's recurrence state stayed clean while the hidden stream diverged; layer-1+ states diverged; 248320/248320 logits differed). Fixed; both gates green.

## Perf (the prefill work this issue forced)

- The qwen35-hybrid prefill's DeltaNet layers ran their four projections + out_proj as PER-TOKEN matvecs (~39 MB weight traffic / token / layer) — 60% of prefill wall. Chunk-batched via `matmat` + the shared `deltanet_layer_recurrent_body`: arm wall 105 s → 62 s at 2K context (prefill 47 → 25 ms/tok), 2.2× on the bisect harness (25.8 s → 13.5 s at 512 tokens ×4 chunks... legacy 23.4-25.8 s vs chunked 10.7-13.5 s).
- Remaining prefill wall at 16K: the O(N²) attention Phase B (serial per position, rayon over 8 heads) + the serial `simd_matmul_rows_batched` GEMM (~56 GFLOPS measured single-stream). A parallel-batched GEMM is the next kernel lever (upstream katgpt-types; GOAT-gated there, out of this issue's scope).
- The 64K envelope: 16K full arm ≈ 22 min → 64K ≈ 4× O(N²) attention + 4× GEMM ≈ 70-90 min/arm → 5-arm matrix ≈ 6-7 h DETACHED. Deferred to its own run; this bench's 16K matrix is the in-session gate.

## Trap-4 / runaway (T4)

- Generic-continuation NLL and the greedy free-decode cap probe run per arm (numbers in the table; the trap-4 ratio vs max-recent prints in the log). All arms CAPPED at gen=32 at 16K — the base model's greedy continuation runs to the cap with or without eviction, so the runaway axis is uninformative at this gen cap (recorded; a longer cap is a follow-up, not a blocker — the full-cache arm caps identically, so there is no RELATIVE runaway signal).
