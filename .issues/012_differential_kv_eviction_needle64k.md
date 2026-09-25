# Issue 012 — consume katgpt-core `differential_kv_eviction` on a real KV cache: the multi-needle@64K half of katgpt-rs Issue 882 P3's G1

**Status:** OPEN — filed 2026-09-25 from katgpt-rs Issue 882 P3. The primitive landed there at katgpt-rs `f4926c44a` (katgpt-rs Bench 894, all synthetic gates PASS), and this issue is its model-bound quality gate.

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

- [ ] **T1 — wire the table into one real decode KV path.** Prefer the Ternary-Bonsai GGUF lane (`riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf`). Qwen is an acceptable second family. The work is to surface each head's post-softmax row to `observe_query` and evict to a budget with `select_evict_sink_exempt` (`n_sink` from the model's sink convention; the `kv_sink_window` default is 4). The feature is gated default-off and forwards `katgpt-core/differential_kv_eviction`. The cadence (select every k decode steps, not every step: the N log n sort is ~120 µs at N = 4096 on the M3) is part of the wiring.
- [ ] **T2 — G1: multi-needle @64K at 25% / 50% cache ≥ full-cache − ε.** Plant several needles in a 64K haystack and compare retrieval at a 25% and a 50% budget against full cache.
  - Arms: differential (λ from a grid, pre-registered before the run), the λ = 0 max-recent baseline, katgpt-core's shipped usage-rate score, and prompt-pinned random (the `kv_eviction::beats_random_prompt_pin` bar — a scored policy must strictly beat it at matched budget).
  - Report per-needle-position retrieval, not only the mean.
- [ ] **T3 — G3: no-eviction is bit-identical.** Budget ≥ context must leave the logits `to_bits`-identical to the path without the table wired in, across a full decode.
- [ ] **T4 — trap 4 on real text (the measured negative from Bench 894).** On a long context whose observation window holds one-off spikes the continuation does not need, then a generic continuation, the differential policy was 1.35–1.47× the baseline's output error synthetically. Measure whether real text hits this. Candidates are ppl on the continuation, or a runaway check via katgpt-core `runaway_gate`, which is mandatory for any lossy KV policy's promotion.
- [ ] **T5 — the sink exemption at λ > 1.** Bench 894's λ grid peaked at λ = 1.5 on the fixture, and that is exactly where unpinned sinks are evicted first (32/32). If the real-model λ* lands above 1, A/B the pin mask on and off on real sinks.

## Traps

1. **"Generically attended" ≈ "consistently relevant"** (katgpt-rs Issue 882 trap 4). A query-distribution shift after eviction breaks the score, and it does so silently. T4 exists for this.
2. **Sinks are legitimate outliers.** Keep the pin, and do not trust λ ≤ 1 to protect them on a model whose sink mass is less dominant than the fixture's.
3. **λ is a prior, not a law** (trap 5). Pick λ on this issue's own needles, never by importing Bench 894's grid.
4. **The admission bucket.** A fresh key starts at `μ = 0`, so for its first `W`–`2W` queries its score is its raw mass (a built-in recency prior). Size `W` against the decode horizon.
5. **Measure with `--release`**, paired interleave (the katgpt-rs `tests/common/ab_timing.rs` protocol), and record box state with every figure (free RAM, swap, load, power source).

Promotion (katgpt-rs side, default-on) waits on T2 + T3 + T4 passing here, plus the `runaway_gate` on a sealed long-context eval.
