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
  - ✅ **Blocker CLEARED 2026-09-28 (was found 2026-09-25): katgpt-rs Issue 896 is FIXED and CLOSED at `62fd22b5f`** (layer-major raw tile buffers, one per layer; layer-major pin 10,427,200 elements 0 differing bits; 240 revert-probed cases; no perf regression). Consume KVarN at katgpt-rs ≥ `62fd22b5f`. T1 is unblocked — but re-verify the fix commit is in the checkout actually linked before any number is recorded (`git -C ../katgpt-rs merge-base --is-ancestor 62fd22b5f HEAD`).
  - Setup: V-cache quantizer from the in-tree backends (KVarN 2/3/4-bit), the table from `vk_calibration`'s `E^V` signal at top_k ∈ {1024, 8192}. The coverage dial is Bench 004's 73.6% / 97.6%.
  - Gate 1, prediction vs measurement: per layer, the measured V quant-MSE drop must agree with the dashboard's `1 − ρ_l(V)` (per-head aggregates, Bench 004) within a pre-registered tolerance. Where it does not, record the direction; the absmax caveat is the arbiter.
  - Gate 2: PPL at matched bits is ≤ the plain-quant PPL.
  - Gate 3: a **per-family conditional retention walk**, because this is a lossy surface and aggregate PPL alone is disqualified (the riir-ai Bench 948 pattern, the Orthrus law). The caveat concentrates on off-mean occurrences of frequent tokens, so the walk must slice by token frequency band, not only by task family.
  - Gate 4: sinks do not regress. The table absorbs the sink means; check `kv_sink_window` behaviour on the BOS/sink positions.

### T1 protocol — PRE-REGISTERED 2026-09-28 (before any arm ran; the 012 a57145c precedent)

**Instrument:** `fitted_v_gate` bin + `transformer::gemma2_vquant` (feature `fitted_v_tables`, opt-in, measurement-only). The seam is `ValueStoreHook` on the f16 decode loop (step f2: after the V store, before the read; `NoVQuant` monomorphizes away — the delegation is pinned bit-identical per run by an in-bin probe).

**The simulation law (the honest-surface decision, made before measuring):** KVarN quantizes a V tile only when it FILLS (128 rows); until then `dequantize_value_into` serves the RAW row (Issue-896 semantics). The hook therefore feeds the backend the raw row the forward just stored, and — at a tile's closing store, detected via `value_row_view(..).is_some()` — rewrites every row of the tile in the plain cache with its dequantized (mean-restored) form, BEFORE attention reads. A row is read lossy iff a KVarN-backed cache would serve it lossy; within an open tile nothing diverges. Storage is virtual (the KVarN buffers are the accounting; the plain cache is scratch) — this is a QUALITY gate, not a G2 gate.

**Fixture:** `../riir-train/data/gemma-2-2b-it-f16.gguf` (the Bench-004 artifact — caveats carry), corpus `../riir-train/data/chat_probe` natural-chat pages. Calibration = tokens [0..61 440) (60 chunks × 1024, cache reset per chunk — the Bench-004 protocol); eval = tokens [61 440..73 728) — DISJOINT slices, same distribution. seq = BOS + 1023 corpus tokens; 12 chunks; 12 276 scored positions per arm; teacher-forced NLL, no sampling (deterministic).

**Freeze:** `FittedTokenTable::from_calibration(ValueMean, λ_js = 0)` at top_k = 8192, plus the dial table = rows 0..1024 (rank-ordered remap — no second pass). Untracked tokens take the plain path (never an error, never a zero-row guess).

**Arms (8):** `f16` (base, no V quant) · `p-b2` `p-b3` `p-b4` (plain KVarN V quant, the matched-bits controls) · `mr-b2` `mr-b3` `mr-b4` (the P1 `MeanRemovedValueCache` decorator) · `mr-b4-k1024` (the coverage dial). K stays plain f32 in every arm (V-cache quantizer only); the ONLY arm delta at matched bits is mean removal. KVarN instantiation recorded: tile 128, hadamard OFF, var-norm ON at b > 2, skip-varn + grouped-4 RTN at b2 (KVarN's `with_config` derivation — the same shape as the kv_cache_flatten bench row).

**Pre-registered tolerances (Gate 1):** the prediction is `ratio_l = MSE_mr,l / MSE_plain,l ≈ 1 − ρ_l(V)`, with ρ_l(V) from THIS run's calibration (cross-read against Bench 004's per-layer table; the Bench-895 synthetic read was ±5%, exact under stationary ranges). Real-text rows are absmax-quantized, so range effects add error the synthetic didn't carry: PASS at 3 and 4 bits = within ±0.20 on ≥ 20/26 layers AND |mean(ratio) − mean(1−ρ)| ≤ 0.10. At 2 bits: direction-only (ratio < 1 on most layers) — the Bench-895 G1c 3.8× off-mean worsening lives there; the arbiter is the per-layer `max |V − E|` vs `max |V|` telemetry (an encode-range GROWTH is the recorded direction the prediction degrades in).

**Gate 2:** mean paired ΔNLL(mr − plain) < 0 at every bit width (primary); chunk-paired win share > 0.5 (secondary, 12 paired chunks). Gate 3: flips/ΔNLL by TARGET-token frequency band — top-64 / 64–1024 / tracked / tail (the caveat's off-mean-frequent-token concentration must be VISIBLE, and is recorded, not gated). Gate 4: (a) `‖E^V[BOS]‖/√d` vs the median tracked-row norm on layers 0/13/25 (the table absorbs the sink mean — the mechanism evidence); (b) the sink-window bin (scored positions < 32, the `kv_sink_window` n_sink convention's neighbourhood) must not flip worse under mr than plain at any bits.

**Box state:** 4090 workstation, i7-13700K 16 cores, CPU lane, AC power; launch-time free RAM + commit-vs-limit recorded in the bench doc; runs > 30 min use the scheduled-task recipe (`run_*.cmd`, Issue 012). Expected wall ≈ 6 h (cal ≈ 2 h at the tapped ~8 tok/s + 8 eval arms ≈ 4 h).

**Promotion statement (unchanged):** a measured null is a legitimate recorded negative; promotion is katgpt-rs-side and waits on T1/T2/T3 + the loser demoted.
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
