# Issue 013 — consume katgpt-core `fitted_value_tables` / `fitted_v_reconstruct` on gemma-2-2b: the model-bound G1 of katgpt-rs Issue 883 P1–P3 [L1-198]

**Status:** T1 EXECUTED — measured negative for P1 (recorded). T2 EXECUTED 2026-09-30 — G-A PASS vs a catastrophic baseline, K=V+ not viable standalone, G-D overfit, NIAH build crash (follow-up filed). T3 EXECUTED 2026-09-30 — G1+G3 PASS at every λ, 50.0% bytes/token confirmed, G2 +7.9–8.8% recorded (Bench 013). T4 pending.

**Status:** OPEN — T1 PAUSED 2026-09-28 by owner re-aim ("we don't care about gemma, pivot bonsai/qwen"): the harness (`vk_p1_g1` bin + `transformer::gemma2_quantized`, clippy-clean, smoke-run at 512 cal / 1k eval) is LANDED as WIP for whoever picks the lane up; the full-grid run was NOT taken. The pre-registered protocol below stands. 896 blocker cleared (katgpt-rs `62fd22b5f`) — T1 is unblocked when prioritized.
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

## T1 pre-registered protocol (added 2026-09-28, BEFORE any measurement)

Written before the harness ran — the gates and tolerances below do not move
after numbers exist (the absmax caveat is the only arbiter, per Bench 895
G1c).

**Fixture:** gemma-2-2b-it-f16.gguf (the Bench 004 artifact, 288-tensor
no-qk-norm conversion — caveat 1 carries). Corpus: natural chat text, the
`chat_probe` loader shape — HF datasets-server `HuggingFaceH4/ultrachat_200k`
`train_sft` pages fetched 2026-09-28 (`riir-infer/.raw/chat_probe/page_{0,100,200}.json`,
BLAKE3 recorded in the bench note). **Held-out split:** the first `--cal-tokens`
(default 60k) = the calibration slice; the NEXT `--eval-tokens` (default
12k = 12 × `[BOS]+1024` chunks) = the eval slice. Gate 1's ρ_l(V) comes from
the run's OWN calibration slice — a held-out prediction on unseen text, not
a re-read of the Bench 004 numbers. Chunks `[BOS] + 1024`, KV reset per
chunk, PPL over next-token NLL (the `row_logit_floor_ppl` T2 convention).
Backend: katgpt-kv `KVarNKVCache` (tile 128) at the requested bits; both
arms quantize K identically (`MeanRemovedValueCache` passes keys through).
λ_js = 0 (the Bench 895 convention). Eval token ids come from the SAME
token stream — a chunk position's V tap is keyed by the token fed at that
position.

- **Gate 1 (prediction, pre-registered):** the measured per-layer V
  quant-MSE drop `1 − MSE_mean/MSE_plain` — both quantizations replayed from
  the SAME f16-trajectory K/V captures (captured from the full-precision
  cache during the f16 control pass, so no cross-arm trajectory divergence
  pollutes the comparison) — must agree with the prediction `1 − m·ρ_l(V)`
  (m = the calibration slice's tracked mass at the run's top_k) with **mean
  absolute error ≤ 0.10 over layers**. Per-layer misses are listed with
  direction; the absmax caveat arbitrates direction, not the gate.
- **Gate 2 (quality, hard):** PPL(mean-removed) ≤ PPL(plain) at each bits.
- **Gate 3 (pilot slice):** NLL share by calibration-frequency band
  (rank 0–1k / 1k–8k / tail) — the mean arm must not worsen any band by
  more than its own aggregate PPL delta; the full per-family conditional
  retention walk stays DEFERRED (needs task datasets; this pilot is
  single-corpus).
- **Gate 4 (pilot slice):** mean NLL over chunk positions 0–7 (the sink
  proxy) recorded per arm; the `kv_sink_window` check stays DEFERRED.
- A 2-bit PPL regression combined with a Gate-1 pass is the recorded
  absmax-caveat confirmation (the off-mean-rows class), not a contradiction
  — the issue's own caveat predicts it.
- **Promotion reading:** this run is T1's pilot cell (60k cal / 12k eval).
  The full T1 grid (bits 2/3/4 × top_k 1024/8192, larger eval) re-runs the
  same bin; only the full grid passes promote.

## Tasks (the gates katgpt-rs cannot run)

- [x] **T1 — P1 G1: gemma-2-2b PPL at matched bits, with and without mean removal.**
  - ✅ **EXECUTED 2026-09-29 — MEASURED NEGATIVE for P1 promotion (record: `.benchmarks/011_t1_fitted_v_p1_gate.md`, run `011_run_report.md`/`011_run.log`).** Gate 2: PASS at 2 bits only (−0.022% Δppl, 7/12 win share — within noise), FAIL at 3 (+1.717%) and 4 (+0.489%). Gate 1: the `1 − ρ` prediction overshoots on real rows (within ±0.20 on 15/26 layers at b3/b4; ratio above pred nearly everywhere; encode range grew on 11/26 layers — the absmax direction). Gate 4 PASS at all bits (sink-bin flips 0.00%; the 4a BOS-norm evidence unmeasurable on this fixture — BOS untracked in chat text). Gate 3: the predicted b2 off-mean-frequent damage did NOT materialize. **No promotion — the legitimate recorded negative.** Second, unpooled finding: KVarN's V-row bit arms are not a pure bit ladder on this fixture (plain flips b2 1.04% < b4 2.63% < b3 5.72%; p-b3 PPL −0.24% BELOW f16) — grouped-4 b2 vs per-row varn b3/b4 are different quantizers; recorded for the katgpt-rs mining intake. Instrument + protocol landed at `bfc923d` (protocol PRE-REGISTERED before any arm ran).
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
- [x] **T2 — P2 G1: the K=V+ λ ladder.** Serve `V = K + λ·E_l[s]` with W_V deleted, at λ ∈ {0, 0.5, 1} plus a per-layer schedule chosen by direct grid evaluation on held-out fixtures (never GD).
  - ✅ **EXECUTED 2026-09-30 — G-A PASS against a catastrophic baseline; K=V+ not viable as a standalone posture (record: `.benchmarks/012_t2_kv_ladder_gate.md`, run `012_t2_run.log`; launched via task `riir_infer_t2_ladder` 2026-09-29 12:51, exited 2026-09-30 06:30).** Ladder: f16 6.0907 (== T1 exactly, cross-run determinism PASS) · k-0.00 ppl 1,884,959 (**tax +30,948,056% — catastrophic-class, not the cited 2.5–3.1%**) · k-0.50 11,233 (×1844) · **k-1.00 563.7 (×92.6; λ* = 1, mean paired ΔNLL vs k-0 −8.11487 < 0 → G-A PASS; 3344× recovery of the destruction)**. G-C PASS (in-run to_bits probe). **G-D OVERFIT**: the 54-pass grid's chosen schedule (12/26 layers diverging, composed −0.01643 on the search chunk) read 1238.0 vs uniform λ=1's 563.7 on validation (2.20× worse) — per-layer λ interactions on a 1024-token search chunk are noise; the uniform λ* stands. **G-E MISSING**: NIAH crashed at trial 0 (`token budget 972 < target 1023 (grow the filler)` — `build_niah_trial`'s 4.2 chars/token estimate undershoots this pool at ~4.49; the shrink path absorbs overshoot only). Pre-registered direction-only, flips no gate. **The verdict substance: the table refunds a real fraction of the V:=K destruction, and the destruction is total — 92.6× off f16 is nowhere near a serve posture. The recorded negative for K=V+ as V-cache elimination on gemma-2 is the finding; P3 (T3) is a different product (rotation-rounding-class tolerance vs the STORE arm, not the V:=K destruction class).** Instrument gap filed with the NIAH bug: the report writer runs at Phase D only — a crash loses the structured report + in-memory per-arm NLL detail (win shares, flips, the ρ dashboard survived only as log lines). Follow-up: grow-retry fix + `--niah-only` rerun off the saved table (`012_kv_table_residual.bin`, sha `3a0f5333d6bafd63…`) + incremental report write. Protocol below was pre-registered before any arm ran.
  - Measure held-out PPL and NIAH (the katgpt-rs Bench 814 harness shape) against (a) full V and (b) `V := K`.
  - The only claim under test is `quality(K=V+) > quality(K=V)`, i.e. that the table refunds part of the 2.5–3.1% tax. **§3.6 discipline: no parity claim vs full V is made or expected.** Record the ladder honestly.
  - G3: λ = 0 must be `to_bits`-identical to the `V := K` path across a full decode.

### T2 protocol — PRE-REGISTERED 2026-09-29 (before any arm ran; the T1/012 precedent)

**Instrument:** `kv_plus_ladder` bin + `transformer::gemma2_ktov::KToVState`
(the K=V+ serve hook, feature `fitted_v_tables`, measurement-only). The
seam widens [`ValueStoreHook`] with `keys_pre_rope(layer, pos, k)` — called
between the QKV projections and the in-place RoPE, exactly the Bench-004
tap point — and the existing `value_stored` then rewrites the just-stored
V row to `v_from_k_plus(k_pre, row, λ_l)` (katgpt-core P2 verbatim;
untracked token or λ=0 ⇒ the bitwise `V := K` path — the primitive's early
return means the multiply is never executed). Attention scores are
untouched (the K path is unchanged); the served V surface is the only
delta. W_V keeps running in the forward — the FLOP/byte claim is T3's G2
lane; this is the quality ladder, and T3's read-path reconstruction serves
bitwise this value up to rotation rounding (1.83ε).

**Fixture:** identical to T1 (gemma-2-2b-it-f16.gguf, chat_probe corpus,
cal [0..61 440), eval [61 440..73 728) = 12 chunks × [BOS]+1023, seq 1024,
teacher-forced NLL, cache reset per chunk). NEW: schedule-search chunk =
tokens [73 728..74 752) — held out from calibration (table fit), the eval
(validation), and the ladder. Table = `FittedTokenTable::from_calibration
(Residual, λ_js = 0)` at top_k = 8192 (the Bench-004 coverage; the storage
dial stays T1's question), dumped to `.benchmarks/012_kv_table_residual.bin`
(BLAKE3-pinned, in-process round-trip verified) so T3's reconstruction lane
reuses it without a second 2 h calibration.

**Arms (ladder, full eval):** `f16` (base, full V) · `k-0.00` (V:=K) ·
`k-0.50` · `k-1.00` — the issue's λ set. Paired per-position ΔNLL + top-1
flips vs f16 AND vs k-0.00; per-chunk win shares both ways.

**Schedule grid (only if some λ > 0 beats k-0.00; skipped in smoke):**
per layer λ_l ∈ {0, 0.5, 1} \ {λ*}, all other layers at λ*; ONE sweep over
the 26 layers on the search chunk (52 passes + the all-λ* incumbent + the
final chosen-schedule pass — interactions are never summed through
layers); argmax per layer by mean paired ΔNLL vs the incumbent, ties
(|Δ| < 1e-4) resolve to λ*. Direct grid evaluation, never GD. The chosen
schedule then validates on the FULL eval chunks (the search→validation
transfer is the recorded overfit check, G-D).

**Gates (pre-registered):**
- **G-A (the claim):** mean paired ΔNLL(λ − k-0) < 0 for at least one
  λ ∈ {0.5, 1.0} on the held-out eval chunks. `quality(K=V+) >
  quality(K=V)` is the ONLY claim; no parity claim vs full V (§3.6).
- **G-C (hard):** the λ=0 serve hook's logits `to_bits`-identical to a
direct `V := K` copy hook across a 64-position decode probe (tracked AND
  untracked tokens) + the module unit tests (`v_from_k_plus` λ=0
  early-return ⇒ bitwise copy, −0.0 entries included) + the T1 NoVQuant
  delegation probe.
- **G-D (recorded):** search-chunk ΔNLL vs held-out validation ΔNLL for
  the chosen schedule; a schedule that wins on search and loses on
  validation is recorded overfit, and the ladder's λ* stands.
- **G-E (direction-only, n too small to gate):** NIAH in the Bench-814
  shape — the 10-sentence filler pool (verbatim), one needle ("The magic
  password is sunset{1000+137t}. Remember it for later. ") at depths
  {0.25, 0.5, 0.75} × 6 trials, continuation tail (" The magic password
  is"); password-token best rank + hits (rank 1) + answer NLL,
  teacher-forced; arms {f16, k-0.00, k-0.50, k-1.00} (+ k-sched when the
  grid ran). The recorded direction must not show K=V+ DEGRADING retrieval
  vs k-0.00.
- **Tax cross-check (recorded):** Δppl(k-0 − f16) — the V:=K tax the
  refund is measured against (the issue cited 2.5–3.1%; this fixture
  measures its own).
- **Consistency (recorded):** the f16 arm must reproduce T1's 6.0907 PPL
  (same fixture/slices/protocol — the free cross-run determinism check).

**Box state:** 4090 workstation, i7-13700K 16 cores, CPU lane, AC power;
launch-time free RAM + commit-vs-limit recorded in the bench doc;
scheduled-task launch (`run_kv_plus_ladder.cmd`, the Issue-012 recipe).
Expected wall ≈ 13 h (cal ≈ 2.1 h at ~8 tok/s + 4 ladder arms ≈ 3.1 h at
~4.4 tok/s + grid 54 × 3.9 min ≈ 3.5 h + validation ≈ 0.8 h + NIAH 5 arms
× 6 trials ≈ 1.7 h).
- [x] **T3 — P3: tg128 KV bytes/token −50%.** Drop the persistent V cache and reconstruct V from the cached post-RoPE K plus `E_l[s]` using a **half-split** `PositionGroupAction` that reads the forward's own cos/sin tables (see the trap above).
  - ✅ **EXECUTED 2026-09-30 — G1+G3 PASS at every λ (record: `.benchmarks/013_t3_p3_reconstruct_gate.md`, run `013_t3_reconstruct_report.md` + `013_t3_run.log`; chained task `riir_infer_t3_reconstruct`, waited 399 min for T2's exit, verified the table, ran the prebuilt exe; 06:31→08:25 rc 0).** G3: seam identity to_bits PASS; recon-vs-store max |Δlogit| 1.254e-4 (rotation-rounding class). G1: **0 flips in 2046 scored positions at EVERY λ** — k-0 mean Δ +4.45e-7 / k-0.5 −1.84e-7 / k-1 +5.17e-8, all max |Δ| ≤ 5.0e-5 against the 5e-2 bound (~1000× inside); retention walk clean in every band. **Bytes/token: 212,992 → 106,496 = exactly 50.0%** (the law). G2 recorded (NOT gated): tg64 paired interleave — full-cache 265,772 µs/step, recon-λ0 1.079×, recon-λ1 1.088× — mild at this geometry, and the BASELINE T4's kernel levers (deferred restore, block angle-addition) must beat; long-context re-measure before reading 8% as the production cost. **This is the promotable P-half on gemma-2: the served V is algebraically the stored V up to f32 rotation rounding — the quality surface is the FULL-V surface, unlike T2's V:=K destruction.** Instrument caveat carried: the lane still writes the raw V row at store (read path fully reconstructed); a production P3 cache drops the V allocation.
  - **Instrument landed 2026-09-29** (`2ead866`): `transformer::gemma2_vrecon` (`HalfSplitRopeInverse` — the rotate-half `PositionGroupAction` over THIS forward's own freq table, pos-0 early return matching `apply_rope_with_freq`; `VReconState` — the scratch-serving hook state) + the `values_for_attention` seam on `ValueStoreHook` (`None` default = the cache slice, bitwise today's path; `Some(slice)` = the P3 scratch) + the `kv_reconstruct_gate` bin (`vk_p3_tg` feature, implied `fitted_v_tables` + katgpt-core `fitted_v_reconstruct`; launcher `run_kv_reconstruct_gate.cmd`, task `riir_infer_t3_reconstruct`). SMOKE PASS (synthetic zero table, wiring-validated: G3 seam identity to_bits, recon-vs-store max |Δlogit| 1.25e-4, bytes/token 50.0%). Run chained after T2 — protocol below pre-registered before the run.
  - G1: PPL within the T2-measured tolerance at the same λ, plus the per-family retention walk.
  - G2: tg128 tok/s against the full-cache control, using the paired interleave (the katgpt-rs `tests/common/ab_timing.rs` protocol) with box state recorded.
  - G3: `VReadPath::FullCache` is bitwise identical to today's path.
  - Record KV bytes/token. The law predicts exactly 50%, because gemma-2-2b is 8q:4kv at hd 256 and `n_v/(n_kv + n_v) = 1/2`. Sliding-window layers scale with the window and keep the same fraction.

### T3 protocol — PRE-REGISTERED 2026-09-29 (before the run; the T1/T2 precedent)

**Instrument:** `kv_reconstruct_gate` bin + `transformer::gemma2_vrecon`
(feature `vk_p3_tg`, measurement-only). The seam is a defaulted
[`ValueStoreHook::values_for_attention`] on the f16 decode loop: `None`
(default) reads the layer cache — bitwise today's path; `Some(slice)` reads
the lane's scratch. `VReconState` serves rows `0..t_n` of
`V = G(−θp)·K̂ + λ_l·E_l[s]` via katgpt-core's `reconstruct_v_from_rope_k`
(P3 verbatim), with `HalfSplitRopeInverse` — the rotate-half action over
THIS forward's own `RopeFreqTable` (katgpt-core's `RopeAction` is the
ADJACENT-pair subgroup — the wrong convention here, the rope.rs RoVE note
and Issue 013 trap 1; the action matches `apply_rope_with_freq` exactly:
same `angle = pos·freq` expression, same `f32::sin_cos`, and the same pos-0
early return so the pos-0 rows recover BITWISE, −0.0 included). ONE
one-directional inverse rotation per read — never a round-trip (trap 2).
Module tests pin: action round-trip vs the repo's own rope (pos-0 bitwise,
else ≤1e-3), reconstruct == `v_from_k_plus(k_pre)` on rotated keys (the
P3 == P2 law), `read_v` FullCache bitwise copy, and the state end-to-end.

**Fixture:** identical to T1/T2 (gemma-2-2b-it-f16.gguf, chat_probe, seq
1024, teacher-forced, cache reset per chunk). Table = T2's dumped artifact
`.benchmarks/012_kv_table_residual.bin` (BLAKE3-verified; NO second
calibration). Eval slice = T2's eval start, tokens [61440..63488) — the
FIRST 2 eval chunks (2046 scored positions per arm; the G1 pairing is
self-contained store-vs-reconstruct at the same λ, so 2 chunks suffice to
resolve the rotation-rounding class vs a wiring bug, which moves PPL by
orders of magnitude).

**Arms:** f16 base (context) + for λ ∈ {0, 0.5, 1}: store arm
(`KToVState` λ, T2's serve — the pairing base) and reconstruct arm
(`VReconState` λ). The retention walk rides the largest-λ pair.

**Gates (pre-registered):**
- **G3 (hard, wiring):** plain `forward_gemma2_f16` vs
  `forward_gemma2_f16_hk(NoVQuant)` logits `to_bits`-identical across a
  65-position decode — pins the seam insertion post-edit (the default
  returns the cache slice). Plus (recorded, bound 5e-2): recon-λ0 vs
  store-λ0 max |Δlogit| — the rotation-rounding class.
- **G1 (the claim):** for every λ, paired per-position ΔNLL(recon − store):
  mean |Δ| ≤ 2e-3 AND max |Δ| ≤ 5e-2 (pre-registered tolerances; the
  rotation-rounding class measured ~4e-5 in smoke — a convention/wiring bug
  — wrong rotation subgroup, stale token map — blows past 0.05 by orders
  of magnitude; the reconstruction is deterministic algebra, so a FAIL is a
  bug, never a model effect). PPL per arm recorded; flips recorded; the
  retention walk (ΔNLL/flips by target-token count tertile × tracked/miss)
  is recorded, not gated.
- **G2 (recorded, not gated):** tg64 decode after a 128-token prefill —
  12 interleaved (full-cache, recon-λ0, recon-λ1) triples, median of
  per-pair medians, ratios as medians of per-pair ratios (the katgpt-rs
  `tests/common/ab_timing.rs` paired-interleave shape). The naive read
  path's cost is EXPECTED to be a regression at long context — the issue
  already names the kernel levers (deferred restore, block
  angle-addition) as T4's lane; this number is the baseline those levers
  must beat. Box state headed into the log; absolute µs on a loaded box
  are noise, the per-pair ratios are the figure.
- **Bytes/token (the law, recorded):** full = 2·1024·4 B·26 = 212,992
  B/token; P3 key-only = 106,496 B/token = exactly 50.0%
  (`n_v/(n_kv+n_v) = 1/2`; sliding-window layers scale with the window and
  keep the fraction). Instrument caveat recorded: this lane still WRITES
  the raw V row at store (one memcpy/step); the READ path is fully
  reconstructed — a production P3 cache drops the V allocation.

**Dependency on T2:** the artifact (else the run refuses). λ* and the T2
verdict are CONTEXT, not inputs — T3's gates read only its own
store-vs-reconstruct pairing, so the run is valid whatever T2's G-A says
(λ*=0 makes every arm's claim the V:=K̂ reconstruction at the same
tolerance).

**Box state:** 4090 workstation, i7-13700K 16 cores, CPU lane, AC; launch-
time free RAM + commit-vs-limit + the concurrent-process list recorded in
the log; chained scheduled task (`run_kv_reconstruct_gate.cmd`, the
Issue-012 recipe) that WAITS for the T2 process to exit (the box is
exclusive), verifies the table artifact exists, rebuilds in `target-rel`
(T2's warm cache), then runs. Expected wall ≈ 1.2 h (7 G1 passes ≈ 45 min
+ G2 ≈ 25 min + probes).
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
