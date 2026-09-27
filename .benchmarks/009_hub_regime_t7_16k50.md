# Bench 009 — Issue 012 T7 (hub-distractor regime, λ separation scan) + T6 (the 16K/50% cell) + the survival-instrument vacuity fix

**Status:** RECORD — T7 scan COMPLETE: **λ separation NEGATIVE, with the mechanism-level account** (the EMA horizon makes `d ≡ a` at any deferred eviction snapshot); T6 16K/50% cell COMPLETE: **usage-rate meets the G1 bar at 2× compression** (7/8) while differential/max-recent take 6/8 with the best mean damage; the survival readouts were measured PRE-eviction (vacuously 1.000) — instrument fixed (`6ca4a30`), Bench 008's survival row corrected here.

**Pre-registered before any budget arm ran** (in `.issues/012` at `2e07e5f`, before the scan):

- Fixture: 8K context, 8 QWARF needles (asked) + 8 QZARF hub codes (sealed-archive notices, code-shaped, semantically excluded from the question) × 4 MID blocks each (108 hub code rows) **+ a WALL**: 4 more hub blocks ending 23 tokens before the question (12 wall code rows), so the wall's own queries sit inside the W=256 observation window and its rows carry sustained attention mass at eviction time — the regime the mechanism needs.
- Budgets {256, 512} (3.1%, 6.2%); λ grid {1.0, 1.5, 2.0} + maxrecent (λ=0 bit-identical baseline) + random (seed 1337); deferred protocol, cadence 512, W=256, β=0.1, n_sink=4, EPS_NATS=1.0, G1 bar ≥ 7/8.
- **Pre-registered separation readout:** wall-row survival must FALL with λ while needle-row survival does not. No λ moving wall survival at any budget ⇒ T7 closes as a recorded NEGATIVE.
- Model / lane: `Qwen3.5-0.8B-Base-Q8_0.gguf` (qwen35 hybrid: 24 layers = 18 DeltaNet + 6 full-attention, n_kv_head=2, head_dim=256, ctx 262144), CPU hybrid lane, `--release`, batched prefill. Same model/lane as Bench 008.

## Box state

- shikuwa: i7-13700K (8P+8E, 24 logical), 32 GiB RAM (16.9 free at launch), commit 45.7/73.8 GB, AC power, CPU-only lane.
- Concurrent load: sibling agent sessions active (compile traffic only).
- Per-arm wall at 8K: ~630 s (prefill-dominated); at 16K: ~1335 s. A first detached-launch attempt died to process-group cleanup (the agent tool kills background children on launcher exit — `nohup` does not survive it); all runs moved to foreground invocations.

## T7 matrix (8K, hubs=8, deferred protocol; survival = post-first-eviction, averaged over the 6 attention layers)

| arm @3pct (256) | retrieval | meanΔ nats | needle_surv | hub_surv | wall_surv | genNLL |
|---|---|---|---|---|---|---|
| full | 1.000 | +0.000 | 1.000 | 1.000 | 1.000 | 3.302 |
| diff λ=1.0 | 6/8 | +0.656 | **0.012** | 0.086 | **0.500** | 3.736 |
| diff λ=1.5 | 6/8 | **+0.583** | 0.012 | 0.086 | 0.500 | 3.736 |
| diff λ=2.0 | 6/8 | **+0.577** | 0.012 | 0.085 | 0.505 | 3.730 |
| maxrecent | 6/8 | +0.651 | 0.012 | 0.087 | 0.500 | 3.722 |
| random | 5/8 | +1.053 | 0.018 | 0.025 | **0.000** | 3.569 |

| arm @6pct (512) | retrieval | meanΔ nats | needle_surv | hub_surv | wall_surv | genNLL |
|---|---|---|---|---|---|---|
| diff λ=1.0 | 5/8 | +0.999 | 0.077 | 0.169 | 0.716 | 3.634 |
| diff λ=1.5 | 4/8 | +1.002 | 0.077 | 0.169 | 0.715 | 3.628 |
| diff λ=2.0 | 4/8 | +1.016 | 0.071 | 0.170 | 0.724 | 3.644 |
| maxrecent | 5/8 | +1.007 | 0.080 | 0.169 | 0.710 | 3.649 |
| random | 5/8 | +0.800 | 0.071 | 0.067 | 0.037 | 3.479 |

Per-needle Δ (diff λ=1.0 @3pct): `[1.476, 0.842, 1.358, 0.458, 0.728, 0.201, 0.152, 0.038]` (depths 608 → 5233). Determinism: the whole matrix was run twice (`run_t7.log` pre-fix, `run_t7_surv.log` post-fix) — every NLL figure byte-identical across runs; only the survival columns changed (the fix moved their measurement point, not the forward).

## T7 verdict — the pre-registered separation readout FAILS, and the mechanism is now provable

1. **Wall survival is λ-identical**: 0.500 / 0.500 / 0.505 at λ = 1.0 / 1.5 / 2.0 vs max-recent's 0.500. The μ-correction does not demote the wall rows even in the planted regime. Hub survival likewise (0.086 ≡ 0.087), needle survival likewise (0.012).
2. **Why — the EMA horizon.** `μ ← μ + β(a − μ)` at β=0.1 averages over ~10 queries. The deferred protocol's eviction snapshot sits at decode start, after a prompt-sized observation gap (8K queries of haystack, then a 27-token question burst): every row's μ except the last ~dozen question rows has decayed to ~0. So `d = a − λμ ≈ a` for every row that matters, and differential ≡ max-recent **at any deferred eviction on any real sequence** — the class-level selections cannot differ. The synthetic (katgpt-rs Bench 894) worked because its whole world was 192 queries with hubs attended by EVERY query — μ stayed live because observation ended AT the eviction. Real attention is episodic; continuous per-row attention right up to the snapshot exists only for the final ~10-20 positions (which recency protects anyway).
3. **The streaming protocol doesn't rescue it either**: μ is live only for continuously-attended (local-recency) rows; needles and hubs both have dead μ mid-haystack — which is exactly why streaming evicts them (the Bench 008 recorded negative, survival 0.000).
4. **λ is not literally a no-op** — it perturbs near-tie filler selections with model-visible consequences (needle 4: Δ 0.728 at λ=1.0 → 0.248 at λ≥1.5 @3pct, a −0.48 nat change from kept-set differences in filler rows), but the effect is sign-inconsistent across budgets (@3pct: λ≥1.5 better on meanΔ; @6pct: λ≥1.5 loses a needle, 4/8 vs 5/8). Not a policy-level win.
5. **Needle rows are evicted under the deferred protocol at tight budgets** (0.012 @3pct ≈ the 3.1% tie base rate; 0.077-0.080 @6pct ≈ the 6.2% rate) — their question-time mass is real but too small to reach the keep line. Retrieval nevertheless reads 6/8: **the hybrid's DeltaNet layers carry retrieval** (their recurrent state is never evicted; only the 6 full-attention layers compact). The needle NLL damage (+0.58-1.05) is the loss of the attention layers' supporting context, not the needle rows themselves.
6. Trap-4 quiet everywhere: genNLL ratio diff/maxrecent = 1.004 @3pct, 0.996 @6pct. Runaway uninformative (every arm caps at gen=32, including full — no relative signal).

## T6 — the 16K/50% cell (Bench 008 fixture, no hubs, λ*=1.0, deferred protocol)

| arm @50% (8192) | retrieval | meanΔ nats | needle_surv | genNLL |
|---|---|---|---|---|
| full | 1.000 | +0.000 | 1.000 | 3.199 |
| diff λ=1.0 | 6/8 | **+0.489** | 0.812 | 3.185 |
| maxrecent | 6/8 | **+0.489** | 0.815 | 3.179 |
| usage | **7/8 = bar met** | +0.584 | 0.354 | 3.182 |
| random | 6/8 | +0.589 | 0.393 | 3.167 |

Per-needle Δ (diff): `[0.040, 0.001, 0.013, 0.002, 1.077, 1.416, 0.640, 0.726]` — failures are mid-depth needles 4/5; the shallow needles are at full-cache parity.

- **The G1 bar (≥ 7/8) is MET at 2× compression — by usage-rate**, the policy that LOST at 25% (4/8, monotone depth). Its retention spreads across the whole context (needle survival 0.354 yet its kept filler spans all depths), so damage is moderate everywhere (max Δ 1.224) and only one needle crosses the bar. diff/max-recent concentrate retention at the recent tail + question-mass rows, so their failures are exactly the mid-depth needles 4/5 — but their MEAN damage is best (+0.489).
- **The count bar and the magnitude metric disagree about the winner.** Single seed, one cell: the bar-met claim carries that caveat. But the shape (spread-retention wins count, recent-retention wins mean) is the same measured pattern as the usage-vs-maxrecent 25% row, in mirror image.
- diff ≡ maxrecent again (+0.489 vs +0.489, per-needle agreement to ~3 decimals) — consistent with the T7 mechanism verdict.

## The survival-instrument vacuity (found by this bench, fixed at `6ca4a30`)

The survival readouts sat between prefill and the answer decode. Under the deferred protocol NO eviction fires during the prefill (`defer_until = prompt_len`; every prefill position is below it), so the readout was **vacuously 1.000 for every arm — including random**, which is what exposed it (random @3pct read 1.000/1.000/1.000 while evicting 47,784 rows). The readouts now run after the answer decode: the first decode step is the first `maybe_evict` past the deferral, and cadence 512 guarantees no second selection inside the ~100-token answer — post-answer is exactly post-first-eviction.

**Bench 008 correction:** its "needle-row survival 1.000" row and the "deferred protocol retains the needle rows" inference carried the same vacuity — a pre-eviction readout, not a retention measurement. The true deferred-protocol needle survival at 16K/25% is *unknown* (not re-measured; at 8K it reads 0.012-0.080 and the 16K/50% cell reads 0.81-0.815). **Bench 008's NLL verdicts are unaffected** (the NLL metric runs through the decode, which does evict) — its G1-bar and policy-ranking conclusions stand.

## Issue 012 posture after this bench

- **T7: RESOLVED NEGATIVE with mechanism** — the λ axis is synthetic-regime-bound: the differential-vs-max-recent separation requires continuous per-row attention up to the eviction snapshot, which real deferred/streaming attention never provides. Not a promotion result; the max-recent reduction (λ=0) is the effective real-text policy.
- **T5 (sink A/B at λ>1): MOOT** — its precondition was "λ > 1 changes the ranking"; the T7 scan shows the ranking is λ-invariant at the class level under every protocol. The `--sinks` lever ships in the bin for any future re-open.
- **T2 verdict update:** at 2× compression the G1 bar is met by usage-rate (single-seed caveat); at 4× it fails for every policy; differential never separates from its own λ=0 reduction on real text.
- **T6: the 50% cell measured above; the 64K matrix remains the only open cell** (~6-7 h detached for 5 arms; the rig is validated; its marginal value after T7 is the compression-scaling of the DeltaNet-carried retrieval, not the λ axis).

Run artifacts: `needle_gate_8k_hub/l200_surv/needle_gate_report.json` (scan 2, post-fix — authoritative), `needle_gate_16k_50pct/needle_gate_report.json`, logs `run_t7.log` (scan 1 — hit the 7000 s foreground cap before its JSON write; the incremental per-arm lines carry its full NLL matrix) / `run_t7_surv.log` / `run_16k50.log` beside them. Scan 1 vs scan 2 NLL figures byte-identical (determinism cross-check, two independent invocations).
