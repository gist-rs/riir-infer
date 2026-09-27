# Issue 012 — consume katgpt-core `differential_kv_eviction` on a real KV cache: the multi-needle@64K half of katgpt-rs Issue 882 P3's G1

**Status:** OPEN → T1/T3 RESOLVED, T2/T4 MEASURED (split verdict at 16K/25% — Bench 008); T5 + the 50%/64K cells recorded follow-ups. Filed 2026-09-25 from katgpt-rs Issue 882 P3. The primitive landed there at katgpt-rs `f4926c44a` (katgpt-rs Bench 894, all synthetic gates PASS), and this issue is its model-bound quality gate.

## Landed (T1/T3 + the gate rig)

- **T1 RESOLVED** — `src/deltanet/kv_evict.rs` wires `DifferentialEvictTable` (+ max-recent / usage-rate / seeded-random arms) onto the qwen35-hybrid attention KV path via `Option<&mut EvictorState>`: per-(layer, head) tables observing the post-softmax rows the forward already computes; per-layer budget with MEAN aggregation across heads (KV rows are shared under GQA); sink-exempt selection through the shipped `select_evict_sink_exempt` (n_sink=4); in-place K/V + table + slot→logical compaction (the `gather_rows` twin, katgpt-rs `c1c63435e`); cadence-gated selection; RoPE stays on the LOGICAL position. Forward seam: `forward_attention_layer_evictable`, `forward_qwen_deltanet_evictable`, `prefill_qwen_deltanet_chunk_into` (pos0 offset + staged per-position writes when armed). Feature `kv_eviction` (default-off), forwards `katgpt-core/differential_kv_eviction`.
- **T3 RESOLVED, twice-pinned** — armed-with-headroom ≡ unarmed `to_bits` on the synthetic hybrid (lib tests, every dense projection perturbed) AND on the real Qwen3.5-0.8B weights (legacy whole-prompt vs chunked: 0/248320 logit bits, chunks=1 and 4). The real-model bisect caught a wiring bug the zeros-weighted synthetic had hidden (Phase-4 residual-add base); both gates now pin the class.
- **The gate rig** — `needle_eviction_gate` bin: K needle sentences at fixed depths + question + teacher-forced answer; per-needle NLL-delta-vs-full metric (EPS_NATS=1.0, pre-registered); needle-row survival; trap-4 generic NLL; greedy free-decap runaway probe; deferred or streaming protocol.
- **The prefill unblock** — the hybrid prefill's DeltaNet projections were per-token matvecs (~39 MB weight traffic/token/layer, 60% of prefill wall); now chunk-batched through a shared `deltanet_layer_recurrent_body` (2× arm wall, 47→25 ms/tok). Bit-identity pinned as above.

## T2/T4 verdict at 16K/25% (Bench 008, `.benchmarks/008_needle_gate_16k.md`)

Pre-registered: λ*=1.0 (8K pilot grid flat: 0.5/1.0/1.5 → identical), EPS_NATS=1.0, bar ≥ 7/8, deferred protocol (streaming is the recorded NEGATIVE: needle-row survival 0.000 — no query attends a mid-haystack needle while its evidence window is live).

| arm | retrieval | meanΔ nats | survival |
|---|---|---|---|
| full | 1.000 | 0 | 1.000 |
| diff λ=1.0 @25% | 0.750 | +0.715 | 1.000 |
| maxrecent @25% | 0.750 | +0.714 | 1.000 |
| usage @25% | 0.500 | +1.083 | 1.000 |
| random @25% | 0.750 | +0.843 | 1.000 |

1. **G1 bar FAILS for every policy** at 4× compression — the synthetic 0.945 does not transfer to the model/regime at the pre-registered bar.
2. **differential ≡ max-recent** in this regime (no hub structure to reject): the λ separation needs Bench-894's hub-heavy fixture re-created on real text.
3. **usage-rate loses** (4/8, monotone depth pattern — old needles evict first).
4. **vs random: mean win (+0.715 vs +0.843), count tie (6/8)** — not a clean `beats_random_prompt_pin` pass.
5. **Trap-4 quiet** (generic NLL ratio 1.000). Runaway uninformative at gen=32 (all arms cap incl. full).
6. **NOT a promotion result.** Follow-ups: the 50% cell, 64K (rig validated; ~6-7 h detached for 5 arms), and the hub-distractor regime.

## Model note (the rig decision, recorded)

The issue preferred the Ternary-Bonsai-2-27B-PQ2 lane; that lane has no batched prefill and its 64K prefill is hours per arm on this box (arithmetic in Bench 008). The gate ran on `Qwen3.5-0.8B-Base-Q8_0.gguf` — the SAME qwen35-hybrid arch the wiring targets, 64K-in-distribution (ctx 262144), prefill-feasible. The wiring itself is family-generic (same forward paths the bonsai lane uses); a bonsai-family gate waits on either a small qwen35 ternary checkpoint or a GPU lane.

## What exists (katgpt-rs `f4926c44a`, feature `differential_kv_eviction`, opt-in, implies `kv_sink_window`)

- `kv_eviction::differential::DifferentialEvictTable` is one head's side table, SoA, 3 `f32`/key.
  - `observe_query(masses)` takes one query's attention row over the live cache slots and performs `d = a − λ·μ(t−1)`, the EMA update `μ ← μ + β(a − μ)`, and a max over a two-bucket window of `W` queries. It costs ~1 ns/key/step on the M3 CPU.
  - `specificity_into` / `select_evict_into` / `select_evict_sink_exempt` go through the shipped `select_evict_into` and `kv_sink_window::sink_pin_mask_into`.
  - `reset_row(idx)` on slot reuse; `admit_prefix(n)` for a prefilled context.
- `DiffEvictConfig::new(λ, β, W)`. `DiffEvictConfig::max_recent(W)` is the λ = 0 baseline, the plain max-recent-attention (TOVA/H2O-class) score, bit-identical.
- `evictions_for_budget(live, budget)` returns 0 at budget ≥ live, which is the no-op.
- **The masses are CALLER-SUPPLIED.** The consumer must surface the per-head softmax row its attention kernel already computes. No kernel change is needed on the katgpt-core side.

Synthetic reading (katgpt-rs Bench 894, 55% hub keys, 16 needles, N = 1024): retrieval **0.945 / 0.977 at 25% / 50% cache**, against full 1.000.

- The λ = 0 baseline reads 0.680 / 0.758 and the pinned-random null 0.289 / 0.531.
- The shipped usage-rate (cumulative mass / age) score reads **0.000**.
- The win is regime-conditional: it appears where a needle's recent evidence is below a hub's max-over-window mass.

## What this issue owns (the gate katgpt-rs cannot run)

- [x] **T1 — wire the table into one real decode KV path.** DONE — `deltanet::kv_evict` on the qwen35-hybrid attention path (feature `kv_eviction`, default-off, forwards `katgpt-core/differential_kv_eviction`). The Bonsai-27B lane itself is compute-gated on this box (no batched prefill; hours/arm at long context) — the gate ran on the same-arch qwen35 0.8B checkpoint; the wiring is family-generic. Mean aggregation across heads (pre-registered; KV rows shared under GQA); cadence 512; n_sink=4.
- [x] **T2 — G1: multi-needle @64K at 25% / 50% cache ≥ full-cache − ε.** MEASURED at 16K/25% (the 64K/50% cells are compute follow-ups): **bar FAILED for every policy** (best 6/8 = 0.750 vs the 0.875 bar), differential ≡ max-recent (no hub regime on natural text), usage-rate loses (4/8), vs random = mean win / count tie. Full table + per-needle deltas in Bench 008.
- [x] **T3 — G3: no-eviction is bit-identical.** DONE — pinned on the synthetic hybrid (both postures, hardened to perturb every projection) AND on the real Qwen3.5-0.8B weights (0/248320 logit bits, chunks=1 and 4). The real-model bisect caught the Phase-4 residual-base bug the zeros-weighted synthetic had hidden.
- [x] **T4 — trap 4 on real text.** MEASURED quiet at 16K/25%: generic-continuation NLL ratio diff/maxrecent = 1.000. The runaway probe is uninformative at gen=32 (all arms cap incl. full); the promotion-grade runaway gate on a sealed long-context eval remains UNRUN (katgpt-rs side).
- [ ] **T5 — the sink exemption at λ > 1.** The pilot λ grid (0.5/1.0/1.5 at 8K) read IDENTICAL retrieval with n_sink=4 pinned throughout — no λ-sensitivity, so the A/B has no signal yet; rerun the grid when the hub-distractor regime exists (that is where λ > 1 changes the ranking).
- [ ] **T6 (follow-up) — the 50% + 64K cells** (rig validated; 64K ≈ 6-7 h detached for 5 arms) and **T7 — the hub-distractor regime** (plant hub sentences as distractors so differential-vs-max-recent separation becomes measurable on real text; without it λ is provably inert here).

## Traps

1. **"Generically attended" ≈ "consistently relevant"** (katgpt-rs Issue 882 trap 4). A query-distribution shift after eviction breaks the score, and it does so silently. T4 exists for this.
2. **Sinks are legitimate outliers.** Keep the pin, and do not trust λ ≤ 1 to protect them on a model whose sink mass is less dominant than the fixture's.
3. **λ is a prior, not a law** (trap 5). Pick λ on this issue's own needles, never by importing Bench 894's grid.
4. **The admission bucket.** A fresh key starts at `μ = 0`, so for its first `W`–`2W` queries its score is its raw mass (a built-in recency prior). Size `W` against the decode horizon.
5. **Measure with `--release`**, paired interleave (the katgpt-rs `tests/common/ab_timing.rs` protocol), and record box state with every figure (free RAM, swap, load, power source).

Promotion (katgpt-rs side, default-on) waits on T2 + T3 + T4 passing here, plus the `runaway_gate` on a sealed long-context eval.
