# Issue 013 — consume katgpt-core `fitted_value_tables` / `fitted_v_reconstruct` on gemma-2-2b: the model-bound G1 of katgpt-rs Issue 883 P1–P3

**Status:** OPEN — filed 2026-09-25 from katgpt-rs Issue 883 (closed there 2026-09-25; record in katgpt-rs HISTORY.md § Issue 883). The primitives landed there at katgpt-rs `0b768e95d` (katgpt-rs Bench 895). This issue is their model-bound quality gate and the tg128 cell.

## What exists (katgpt-rs `0b768e95d`, opt-in)

- `FittedTokenTable::from_calibration(&LayeredVkCalibration, VkSignal::{ValueMean, Residual}, λ_js)` freezes the P0 tables this repo's `vk_calibration` bin already accumulates (Bench 004: 26 layers × top_k=8192 × kv_dim 1024).
  - `row(layer, token) -> Option<&[f32]>`. An untracked token returns `None`, and the consumer then takes the plain path.
  - `LayeredVkCalibration` is the same builder this repo re-exports as `CalibrationTables`, so no conversion step is needed.
- **P1** (`fitted_value_tables`): `MeanRemovedValueCache<C: QuantizedKVCache>` decorates any KV backend. It stores `q(V − E^V_l[s])` and reads `dequant + E^V_l[s]`, with `set_token(pos, tok)` before each store.
  - The fused decode step is `accumulate_value(layer, pos, w, acc)`.
  - Synthetic (KVarN): the dashboard's `1 − ρ_V` predicts the quant-MSE ratio within ±5% at 2/4/8 bits, and sinks are absorbed.
  - **Caveat, measured:** at 2 bits the off-mean rows of a bimodal token get 3.8× WORSE.
  - **G2 FAILED:** the fused restore costs +4.8–5.2% on the V-aggregation kernel, against a ≤ +1% bar.
- **P2** (`fitted_value_tables`): `v_from_k_plus(k, row, λ, out)` = `K + λ·E_l[s]`. λ = 0 or a missing row is the `V := K` copy, bitwise.
  - Synthetic refund law `MSE(λ)/MSE(0) = 1 − (2λ − λ²)·ρ(V−K)`, exact to 0.11%.
  - At this repo's measured mean ρ(V−K) = 0.49 (Bench 004), λ = 1 would refund ~49% of the V:=K residual energy. Whether that shows up in PPL is T2's question.
- **P3** (`fitted_v_reconstruct`): `reconstruct_v_from_rope_k(rope: &impl PositionGroupAction, pos, k_cached, row, λ, out)` = `G(−θp)·K̂ + λE`, with one inverse rotation per head. `read_v(VReadPath::{FullCache, Reconstruct{λ}}, …)` is the kill switch.
  - Exact to 1.83ε up to position 131071.
  - **G2 FAILED:** 14–15× slower with katgpt-core's `RopeAction`, which computes sin/cos on every read.

## ⛔ Tap-point / convention trap (found while filing — trap 1 of katgpt-rs 883)

katgpt-core's `RopeAction` rotates **interleaved** pairs `(2i, 2i+1)`. This repo's gemma-2 RoPE (`src/rope.rs::apply_rope_heads_precomputed`) rotates **half-split** pairs `(i, i + head_dim/2)`, the NeoX convention.

Passing `RopeAction` to `reconstruct_v_from_rope_k` against this cache is **silent corruption**: plausible-looking V with no NaN. P3 must pass a half-split `PositionGroupAction` built on the forward's own cos/sin tables. That also fixes the transcendental-bound G2 cost. The primitive is generic over the action by design, so no katgpt-core change is needed.

## Tasks (the gates katgpt-rs cannot run)

- [ ] **T1 — P1 G1: gemma-2-2b PPL at matched bits, with and without mean removal.**
  - ⛔ **Blocked on katgpt-rs Issue 896** (found 2026-09-25). KVarN's raw tile buffer was shared across layers, so decode-order stores at `n_layers ≥ 2` read back the LAST layer's data. The in-progress tile also dequantized to zeros, and 2-bit keys panicked there. Consume KVarN only at or after the 896 fix commit, or every T1 number carries the corruption.
  - Setup: V-cache quantizer from the in-tree backends (KVarN 2/3/4-bit), the table from `vk_calibration`'s `E^V` signal at top_k ∈ {1024, 8192}. The coverage dial is Bench 004's 73.6% / 97.6%.
  - Gate 1, prediction vs measurement: per layer, the measured V quant-MSE drop must agree with the dashboard's `1 − ρ_l(V)` (per-head aggregates, Bench 004) within a pre-registered tolerance. Where it does not, record the direction; the absmax caveat is the arbiter.
  - Gate 2: PPL at matched bits is ≤ the plain-quant PPL.
  - Gate 3: a **per-family conditional retention walk**, because this is a lossy surface and aggregate PPL alone is disqualified (the riir-ai Bench 948 pattern, the Orthrus law). The caveat concentrates on off-mean occurrences of frequent tokens, so the walk must slice by token frequency band, not only by task family.
  - Gate 4: sinks do not regress. The table absorbs the sink means; check `kv_sink_window` behaviour on the BOS/sink positions.
- [ ] **T2 — P2 G1: the K=V+ λ ladder.** Serve `V = K + λ·E_l[s]` with W_V deleted, at λ ∈ {0, 0.5, 1} plus a per-layer schedule chosen by direct grid evaluation on held-out fixtures (never GD).
  - Measure held-out PPL and NIAH (the katgpt-rs Bench 814 harness shape) against (a) full V and (b) `V := K`.
  - The only claim under test is `quality(K=V+) > quality(K=V)`, i.e. that the table refunds part of the 2.5–3.1% tax. **§3.6 discipline: no parity claim vs full V is made or expected.** Record the ladder honestly.
  - G3: λ = 0 must be `to_bits`-identical to the `V := K` path across a full decode.
- [ ] **T3 — P3: tg128 KV bytes/token −50%.** Drop the persistent V cache and reconstruct V from the cached post-RoPE K plus `E_l[s]` using a **half-split** `PositionGroupAction` that reads the forward's own cos/sin tables (see the trap above).
  - G1: PPL within the T2-measured tolerance at the same λ, plus the per-family retention walk.
  - G2: tg128 tok/s against the full-cache control, using the paired interleave (the katgpt-rs `tests/common/ab_timing.rs` protocol) with box state recorded.
  - G3: `VReadPath::FullCache` is bitwise identical to today's path.
  - Record KV bytes/token. The law predicts exactly 50%, because gemma-2-2b is 8q:4kv at hd 256 and `n_v/(n_kv + n_v) = 1/2`. Sliding-window layers scale with the window and keep the same fraction.
- [ ] **T4 — the P1/P3 G2 kernels on the real decode path.** The katgpt-rs primitive-level G2 bars failed: +5% for P1, 14–15× for P3.
  - **Update (katgpt-rs 2026-09-25):** folding the add-back into KVarN's dequant was a LOSER (+13.9–14.2%, reverted). After katgpt-rs Issue 894 made the plain pass 2.46× faster, P1 fused/plain reads **+12.9–13.3%**: the absolute cost is unchanged but the denominator is smaller. The floor is the per-position token→row lookup (+2.1–2.8%). The named lever is the **deferred restore** by linearity, `Σ_p w_p·v̂_p + Σ_s W_s·E[s]` with the row index pre-resolved per position (+1.5–3.6% test-local). It regroups the sum, so it needs a stated error bound, and the miss and zero-table paths must stay bitwise. Its last per-position scalar add belongs in THIS repo's softmax-weight loop. Record: katgpt-rs Bench 895 Addenda I/II, HISTORY.md § Issue 883.
  - Re-measure inside this repo's actual attention kernel (the V-aggregation epilogue and the RoPE tables that already exist there) before anyone reads the primitive numbers as the model-level cost.
  - If P3 is still transcendental- or table-bandwidth-bound, a block angle-addition rotation (exact re-anchoring every N positions) is the named kernel to try.

## Traps

1. **Tap point:** the fit tapped pre-RoPE K / post-W_V V (Bench 004). The P2/P3 serve must consume K at the same point, and P3 must invert exactly the rotation the cache applied, half-split for gemma-2. On a gemma-3/4-class model, K is post-QK-norm.
2. **Never round-trip rotations.** Use one inverse rotation per read. katgpt-rs Bench 895 pinned the round-trip drift: 8.3e-6 after 64 round-trips.
3. **The Bench 004 fixture caveats carry over:** the artifact is the 288-tensor conversion with no q/k-norm, and the dashboard is a 120k-token slice.
4. **λ by direct evaluation, never GD. λ = 0 must be bit-identical.**
5. **Box state on every perf figure:** free RAM, commit-vs-limit, concurrent jobs, power state.

Promotion (katgpt-rs side, default-on) waits on T1/T2/T3 passing here plus the loser demoted, per the standing rule. A measured null (the table refunds nothing on gemma-2) is a legitimate recorded negative.
