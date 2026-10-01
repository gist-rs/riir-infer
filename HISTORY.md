# HISTORY.md — riir-infer

Durable records for resolved questions and closed lanes (the noise-reduction
convention: the record lands here, hash-pinned; open work lives in `.issues/`
and `.plans/`). Created 2026-09-23 at the first record.

## 2026-10-01 — Issue 032 DONE: Metal CubeCL prefill computes Hadamard-folded (Bonsai-2) models (`5d0592f`)

- **What was blocked:** `prefill_tokens_chunk` panicked on folded files whenever the cudarc whole-prefill lane was unavailable — always on macOS — so Metal could never batch-prefill the production Bonsai-2 PQ2_0 file (riir-ai Plan 602 Phase C / Plan 607 A4 named it the hard predecessor of every M3 prefill cell; neither owned the task).
- **What landed:** batched twins of all 8 decode eager rotation sites (Plan 602 B3) in the CubeCL prefill body; dense a/b via the new bit-identical token-grid GEMV (`GemvBatchedCubeCL::launch_token_grid`); the non-macOS cudarc FFN block skipped for folded files (no rotation wiring); the refusal replaced by a marker/tables agreement assert.
- **Gates (M3 Max, AC):** G1 vs folded decode-eager at P=64/100/128/512 — top-1 equal, top-5/top-20 1.00, worst rel err ≤ 1.5e-5 at the last position and one decode step later; revert probe (site 7 dropped) reds it (top-5 0.00); G3 pre-rotation `Q2_0` pin `99a0733c45a0e663` @2048 exact.
- **First Bonsai-2 M3 prefill reading (unscored):** 100.9 tok/s @2048 batched vs 15.7 tok/s token-by-token = 6.44× time-to-first-token. The league cell stays riir-ai Plan 602 C4 (Ultra-gated).
- **Test:** `crates/riir-infer-gpu/tests/g1_032_folded_prefill_metal.rs` (three `#[ignore]` arms: G1, pin, throughput).
- ⛔ **Correction (same day): the record above was measured only to P=2048, and production chunks reach 4096.** Past ~2730 tokens the folded prefill CRASHED — wgpu `dispatch group size dimension ([98304, 1, 1]) must be <= 65535` — because the rotation launchers and `CopyCubeCL` (the gdn_v staging copy) put every workgroup on the x axis: fine at decode's P=1, over Metal's cap at 4096 × 6144 / 256. Found by the Issue 1004 Bonsai-2 e2e at 4096, not by any 032 gate. Fixed in `5f075ff` (rotation launchers, 2-D grid) + `7e99f34` (`CopyCubeCL` 2-D path with an explicit u32 bound; old callers byte-identical). Both carry unit tests past 65535 workgroups that red on the old code with the identical error. G1 re-run at **P=4096** with both: top-1 equal, top-5/top-20 1.00, worst rel err 1.23e-5 (2.80e-6 one decode step later). Lesson: a G1 that stops below the production chunk size certifies a size nobody ships — run `I032_P=4096`.

## 2026-09-30 — Issue 054 Part 2 (reflex) REFUTED for this lane: the prefix-state handoff lead — laya is ModernBERT (bidirectional), not GDN; the coupling gate landed

The reflex issue's Part 2 asked the riir-infer-laya owning session to evaluate
open-jev-fast's `fla_mode="state"` prefix-state handoff (encode a case's shared
state once, hand its final GDN recurrent state to each question's suffix) for
`riir-infer-laya` serving. **Verdict: does not transfer — the premise named the
wrong architecture.** The laya checkpoints are ModernBERT-large / mmBERT-base
(`encoder_config.json`), a bidirectional encoder: no causal mask anywhere in
the op stream (the only mask is the symmetric sliding-window band), while GDN's
causal delta-rule recurrence is exactly what makes the upstream handoff
lossless. The `deltanet` substrate serves the ternary Bonsai/GDN lane,
unrelated to laya. Two further structural grounds: the state sits AFTER the
per-question head span in `build_sequence`'s render (different RoPE offset per
question — measured 34 vs 35), and the state truncation room is per-question
(the "shared prefix" is not guaranteed identical tokens).

**The gate** (`crates/riir-infer-laya/tests/prefix_state_coupling.rs` + its
Cargo `[[test]]` row, feature `laya-riir`): TWO-SIDED — the shared span must
drift STRICTLY ABOVE 1e-4 (a reading at or below the 1e-5 packed-equivalence
GEMM budget FAILS the test, so an architecture change that ever makes the span
independent re-opens the record mechanically). Measured (CPU posture, release
+ debug agree): real typed checkpoint, two choice questions against one shared
state — the shared span (283 of 317/318 tokens: the state is ~89% of each
sequence, so the as-filed ~5× prize was real) drifts **5.1e2**; synthetic
4-layer geometry arm **2.7e0** over a 16-row shared prefix with a bit-identical
determinism control. Consequence: the packed per-question pass (reflex issue
020 T5) stays the exact floor for laya case serving; the lead's shape stays
valid only for causal serving models (GDN/KV decoders). Full verdict: reflex
`.issues/054_openjev_lane_and_prefix_state_handoff.md` Part 2 + its HISTORY.

## 2026-09-30 — Issue 013 T2 EXECUTED: the K=V+ λ ladder on gemma-2 — G-A PASS vs a catastrophic baseline; K=V+ not viable standalone; G-D overfit; NIAH build crash (029)

Run 2026-09-29 12:51 → 2026-09-30 06:30 (task `riir_infer_t2_ladder`, ~17.7 h
on a contended box: a sibling python trainer held ~14 cores, cal 3 tok/s vs
T1's uncontended 9). Record: `.benchmarks/012_t2_kv_ladder_gate.md`; run log
`012_t2_run.log`; table artifact `012_kv_table_residual.bin` (783,556,688 B,
sha `3a0f5333d6bafd63…`, reused by T3 — no second calibration).

- **Ladder (held-out eval, 12,276 scored/arm):** f16 **6.0907** (== T1 exactly,
  cross-run determinism PASS) · k-0.00 1,884,959 (**tax +30,948,056% —
  catastrophic-class; the issue's cited 2.5–3.1% was another posture, this
  fixture measures its own**) · k-0.50 11,233 (×1844) · **k-1.00 563.7 (×92.6;
  λ\* = 1, mean paired ΔNLL vs k-0 −8.11487 < 0 → G-A PASS; 3344× recovery of
  the destruction)** · k-sched 1238.0 (validation).
- **G-C PASS** (in-run to_bits λ=0 probe). **G-D OVERFIT**: the 54-pass grid
  (12/26 layers diverging on the search chunk, composed −0.01643) read 2.20×
  WORSE than uniform λ=1 on validation — per-layer λ interactions on a
  1024-token search chunk are noise; the uniform λ\* stands.
- **G-E MISSING**: NIAH crashed at trial 0 (`token budget 972 < target 1023
  (grow the filler)` — `build_niah_trial`'s 4.2 chars/token estimate
  undershoots the pool at ~4.49; shrink path absorbs overshoot only).
  Pre-registered direction-only, flips no gate. Fix + `--niah-only` rerun:
  `.issues/029_niah_builder_undershoot_and_phase_d_report.md`.
- **The verdict substance:** the table refunds a real fraction of the V:=K
  destruction and the destruction is total — 92.6× off f16 is nowhere near a
  serve posture. The recorded negative for K=V+ as standalone V-cache
  elimination on gemma-2 is the finding. P3 (T3, chained, started 06:31 with
  the table BLAKE3-verified) is a different product: its G1 tolerance is
  rotation-rounding class vs the STORE arm (1.254e-4 max |Δlogit| at the G3
  probe — in family with the smoke), not the V:=K destruction class.
- **Instrument gap:** the report writer runs at Phase D only — a crash loses
  the structured report + in-memory per-arm detail (win shares, flips, the ρ
  dashboard); the gates here were adjudicated from log lines. Incremental
  per-arm report write filed with 029.

## 2026-09-30 — Issue 013 T3 EXECUTED: P3 V-cache reconstruction — G1+G3 PASS at every λ; the 50.0% bytes/token law holds; G2 +7.9–8.8% recorded (the promotable P-half)

Chained run 06:31 → 08:25 (rc 0): the `riir_infer_t3_reconstruct` task waited
399 min for the T2 process, BLAKE3-verified T2's table artifact, and ran the
prebuilt exe — the chain design held even through T2's crash. Record:
`.benchmarks/013_t3_p3_reconstruct_gate.md`; run report `013_t3_reconstruct_report.md`
(incrementally written — the Phase-D defect does not exist here) + `013_t3_run.log`.

- **G3 PASS** (seam identity to_bits; recon-vs-store max |Δlogit| 1.254e-4,
  the rotation-rounding class). **G1 PASS at every λ: 0 flips in 2046 scored
  positions** — mean ΔNLL ≤ 8.3e-6, max ≤ 5.0e-5 against the 2e-3/5e-2
  pre-registered bounds; retention walk clean in every band.
- **Bytes/token 212,992 → 106,496 = exactly 50.0%** (n_v/(n_kv+n_v) = 1/2;
  sliding-window layers keep the fraction).
- **G2 recorded (not gated):** tg64 paired interleave — full-cache 265,772
  µs/step, recon-λ0 1.079×, recon-λ1 1.088×. Mild at this geometry (seq 129 +
  64 decode), NOT the primitive-level 14–15×; the long-context re-measure +
  the kernel levers (deferred restore, block angle-addition) are T4's lane,
  and these numbers are the baseline they must beat.
- **The verdict substance:** P3 is the promotable P-half on gemma-2 — the
  served V is algebraically the stored V up to f32 rotation rounding (the
  FULL-V quality surface), unlike T2's V:=K destruction. What it buys: the V
  cache allocation (50%); what it costs today: the +8–9% read-path baseline.
  T1 (P1) negative, T2 (P2) not viable standalone, T3 (P3) clean — the
  katgpt-rs-side promotion decision now has its full model-bound input set.

## 2026-09-27 — Issue 014 RESOLVED: activation-aware ternary scale fit — SPLIT BY LANE (born-ternary clean negative; dense-parent mechanism transfer)

Filed 2026-09-25 from katgpt-rs Issue 886 (the per-family conditional
retention walk's model-bound G1; the substrate landed there at `0fb2254d9`,
katgpt-rs Bench 896). Issue file removed 2026-09-28 (noise-reduction; this
is the durable record). Resolution commit `3c2b569`; lane commits `03681e9`
(T2 prep: `act_taps` to the lib), `fc31777` (T2/T3: the
`act_retention_walk` bin — born-ternary scale-refit arms + the per-family
conditional retention walk, 6 arms incl. the ZeroQAT-class comparator at
default knobs), `28b1566` (T2(b): the dense-parent PTQ lane on gemma-2 f16
— linear-input tap forward + held-out capture + the Bench-896
reconstruction metric over the five fits). Benches: `005_act_diagonal_bonsai_first_slice.md`
(T1) + `007_act_scale_refit_walk_and_ptq.md` (T2/T3/T4).

- **T1 (Bench 005, `act_diagonal_calibration` bin, feature
  `act_diagonal_calibration`, rides the forward's own `TernaryMatvecHook`
  seam — post-rotation inputs observed side-band, forward bit-identical):
  the rotation did NOT flatten the diagonal globally** — layer_out heaviest
  tap (max/med up to 90.9, top-1% share up to 11.6%), attn_in 6.7/56.9
  median/worst, final_in 9.3; `swiglu` near-uniform (1.6× / 1.4%) — the
  null prediction live for down_proj specifically. Artifact digest
  `5b6aead0901e3133cd7ee28512e75122c788ae9d720dec78d099e6b390d9a9d2`.
- **Born-ternary (Ternary-Bonsai-2-PQ2): CLEAN NEGATIVE** — the shipped
  amax payload is the fit family's fixed point: requant controls collapse
  it (scale ×0.668, 14.3% codes, NLL 2.68→8.78, 96.7% flips uniform across
  all 6 families) while both search arms recover it (ws_ex2 0.001% delta /
  31 coin-flip flips / 100% top-8 retention; ws_uniform 0.000% / 2 / 100%)
  — diagonal-independently, because there is no quantization error to
  redistribute. ZeroQAT-class at default knobs: structurally stationary on
  shipped codes; at the requant insertion point its ≈1% GD drift made the
  model WORSE than its own starting point (NLL 9.10 vs 8.78) — the
  layer-local surrogate misaligns with model quality on damaged payloads.
- **Dense-parent (gemma-2-2b f16, layer-level Bench-896 metric on real
  parents + 48 real activations/tap): the mechanism TRANSFERS** — ws_ex2
  −34.6% vs mean_abs, −20.1% vs the blind-search control (0.4176 vs
  0.6383 / 0.5230), ordering exactly as Bench 896. katgpt-rs 886 P1 closed
  on this evidence.

## 2026-09-27 — Issue 020 RESOLVED: the len_derived CAPACITY finding was the classifier's, not ours — and both fixed temp paths are gone

Filed 2026-09-26 from the katgpt-rs drift sweeps (the standing red on this
repo since the carve). Both findings closed; the issue file removed.

- **Finding 1 — CAPACITY at `encoder_lane_cubecl.rs` was a classifier false
  positive, closed in the instrument (katgpt-rs `43c14d754`).** The bind
  `BufferArg::from_raw_parts(rows_handle, rows_host.len())` is exact by
  construction: `rows_host` is the HOST slice param whose `.len()` IS the
  live row count (the launcher scans the same slice for per-row bounds and
  asserts the binding floor from it), and every caller creates the handle
  exactly-sized (`create_u32` → `create_from_slice`, plus the test). The
  depth-1 classifier flagged ANY `.len(` in a bind-length position as
  allocation-as-length; the upstream `classify_pair` already required the
  receiver to BE the handle. katgpt-rs now shares ONE rule between both
  classifiers (`length_from_handle_size_method`) and the bind capture is
  paren-balanced (it truncated `kv_handle.len()` at the first `)`, which the
  old broad regex matched by accident). The bind now reads UNRESOLVED
  (reported-and-unpinned by design, Issue 785) and the riir-infer sweep row
  is green at its `max_findings 0` wall. No riir-infer code changed for this
  finding — the contract was always right.
- **Finding 2, site 1 — the ANE compile-cache root (riir-infer `20ea589`).**
  Default root moved `temp_dir().join("riir-laya-ane-cache")` →
  `~/Library/Caches/riir-infer/laya-ane` (macOS cache home; this lane is
  ANE-only): persists across /tmp cleanup, keeps the compile-once-per-
  process-tree contract across processes, and leaves the
  `env::temp_dir().join("literal")` shape. `LAYA_ANE_CACHE` override
  unchanged; pid-suffixed temp only when HOME is unset. Trade-off recorded
  in the module doc: the persistent root grows one bundle per digest,
  unbounded — clear it or set the env to reclaim. The cross-process race was
  already mechanized (pid-unique staging + atomic rename claim) — the pid
  recipe applied to the ROOT would have defeated the cache for zero safety
  gain.
- **Finding 2, site 2 — the exl3 oracle fixture (same commit).**
  `real_pack_oracle_k_proj`'s fixture moved `/tmp/exl3-pack` →
  `<manifest>/.raw/exl3-pack` (repo-local, gitignored,
  `CARGO_MANIFEST_DIR`-rooted). The test never wrote the path and the
  referenced fetch script (`.raw/exl3_layer_oracle.py`) is gone — both stale
  references corrected, the skip-loud contract unchanged, the `.raw`-cleanup
  risk named in the doc. `.docs/001`'s gate parenthetical updated.
- **Rider — the riir-instinct walk-floor DELEGATION break (katgpt-rs, same
  commit as the classifier).** Both sweeps red on "no non-zero min_rs_files
  row for riir-instinct (16 tracked .rs)" — the floors said md-only from
  birth on 2026-09-26 and the P1–P5 code landed the same day. Re-pinned at
  measured (clean HEAD `337da41`: 16 rs / 33 cfg sites / 178 cand / 0
  findings) in orphaned_attr, platform_dead_code and percentile.
- **Validation:** audit selftest clean; len_derived `--canary` 13/13;
  len_derived sweep PASSED (riir-infer EXACT-UPSTREAM=3 GUARD-ONLY=4
  GUARDED=25 PERSISTENT-UPSTREAM=8 UNRESOLVED=129, no CAPACITY);
  shared_temp_path riir-infer `fixed=0` (was 2). Clippy `-D` clean at
  `laya-riir-ane` (the module compiles to nothing at default features —
  feature-aware check done) and `--features exl3 --all-targets`; exl3 lib
  tests 214 passed / 0 failed. Remaining sweep reds (mmorpg-editor 3>1,
  riir-ai 5>4 on shared_temp_path) are those repos' own pre-pinned backlogs,
  each owner's adjudication.

## 2026-09-26 — Issue 019 RESOLVED: the c0dfa06 Metal barrier REVERTED — A alone was the wobble, and serial dispatches never needed barriers

The 019 verdict (T3), landed by the c0dfa06 author after conceding the
contested-evidence point (`4205c12`):

- **Necessity (T1) conceded on the timing re-check:** the deep-probe
  triplet that read "clean ×3 with barriers only" was confounded — runs
  4/5 carried `2a34bd3` in the working tree (log mtimes 16:07:52 /
  16:09:22 vs the sibling's 16:07:41 edit); run 3 (mine-only, 7 clean
  passes) is ~4% luck at the measured 36% base fire rate, and
  probe_deep_2's single `L3.scores` fire is exactly A's signature
  (post-softmax corruption, clean q/k), not a between-dispatch race.
- **Premise refuted on the code:** the fork's `begin_compute_pass`
  never sets `MTLDispatchType` — Metal defaults to SERIAL, where the
  GPU does not begin a dispatch until prior dispatches COMPLETE, so
  memory visibility is implied and the empty transition bodies are
  upstream-correct behavior, not a defect. The durable artifact is that
  premise (if a concurrent-dispatch-type encoder ever lands, 019's
  analysis is the map).
- **The revert:** the barrier emissions are out of the fork; the probe
  instruments (repeat loop, deep mode, tracked indices — zero prod
  cost) STAY. Post-revert validation: smoke 9/9, G5 cubecl ×3
  bit-identical at the floor (3.092e-6 / 2.233e-6 / 5.187e-6), clippy
  `-D` clean; G5 wall 22-27 s without the barriers vs 24-31 s with
  (directional; T2 formally MOOT — nothing ships).
- 019 marked RESOLVED in-file (kept; it carries the premise record and
  the re-open trigger).

## 2026-09-26 — Plan 611 DONE: the T7 op-layer unification verdict (Bench 006) — riir-reflex Issue 008 closed

The last open task of the riir-infer consolidation campaign (riir-reflex
Issue 008 T7, mirrored as 998/610 S8). One portable CubeCL implementation
of the laya `Backend` trait was built over this repo's op layer (S1–S4) and
A/B'd against the hand lanes (S5). The verdict rules were pre-registered at
`fac0dfd`, before any run.

- **The hand Metal lane stays the macOS default.** CubeCL ran 5.1–8.4×
  slower than Metal, 0/12 wins in every cell of three permutation-balanced
  runs (Bench 006, results `ac85c8e`; loaded box, preflight refused on load
  only, GPU canary OK).
- **The CubeCL arm is KEPT opt-in** (`laya-riir-cubecl`; not default, not
  in any release set). It beats the CPU lane on the short ladder (seq
  16/54/128: 0.54–0.72, 12/12) and on the 1-question case (0.75), so the
  pre-registered delete-if-slower-than-CPU-everywhere rule fails. Its G5
  gate is armed (riir-reflex `ccb5bd0`). **Nothing was deleted or
  promoted.**
- **Engine-side payoff, kept regardless:** the two-pass mean-centered
  LayerNorm (`LayerNormMeanBatchedCubeCL` — the GAP kernel; the engine's
  norms were RMS-only), the batched in-place row-softmax, the offset +
  head-batched tiled matmuls (z-dispatched heads), and the
  rope/split/merge/gather permutation kernels.
- **Defects the arm surfaced and fixed on the way:** the `gather_rows`
  residency class (016); the one-pass LN cancellation at deep layers
  (two-pass now); the softmax max-read race (018, `2a34bd3`), which is live
  in the engine's single-row `softmax_f32` too; and the chunked-softmax row
  offset. 019's contested barrier was reverted (`4205c12`).
- Harness: `crates/riir-infer-laya/tests/backend_ab.rs` (measurement-only).
  Re-read on a quiet box with one command (AGENTS.md).

## 2026-09-26 — Issue 018 CLOSED: the Cubecl-posture drift wobble was a softmax write-after-read race (fix `2a34bd3`)

Plan 611 S4's residual: the laya G5 gate at the cubecl posture held top-1
1.000000 in every run, but the prob-drift magnitude sporadically read
10–1000× its per-checkpoint floor, the outlier moving between checkpoints
and runs (9/20 G5 runs failing). The gate was `#[ignore]`d and every Cubecl
number held PROVISIONAL.

- **Root cause:** both CubeCL softmax kernels (`softmax_f32`,
  `softmax_rows_inplace_f32`) read the reduced max from shared slot 0 and
  then, with **no barrier**, let the sum phase write `smem[tid] =
  local_sum`, so thread 0 could overwrite slot 0 while a slower SIMD group
  was still reading the max. Scheduling-dependent, so worse under load.
  One `sync_cube()` after the read. The same commit fixes a latent chunked
  launch defect: rows past `MAX_WG_X` re-normalized rows 0.. (the chunk's
  first row now rides `params[1]`; heads·seq > 32768 only, i.e. long
  context).
- **How it was localized:** the issue's own lever 1 (the encoder probe's
  repeat loop, per-op drains). Every fire FIRST diverged at an `L*.attn`
  output, and the row-softmax is the only reduction inside the composed
  attention. A scan of every `let … = smem…[0usize]` read followed by a
  shared write before the next barrier found exactly the two softmax
  sites. The deltanet and two-pass-LN hits were false positives: only
  thread 0 consumes the deltanet value, and LN keeps separate arrays.
- **Refuted first, measured:** hypothesis 3 / lever 2 (a stream drain per
  `begin_pass`): 10 interleaved pairs, excursions to 5e-2 in both arms.
  Zero-filling every write-first slot (`client.empty` recycles pool bytes):
  excursions to 2.4e-2. Neither the drain nor the pool was the class.
- **Measured fix (M3, release, AC, load 12.8–16.6):** repeat-loop probe
  **43/120 fired passes at HEAD vs 0/120** with the fix (same tree,
  interleaved); G5 cubecl **10/10 PASS, every run bit-identical** at the
  floor (english 3.092e-6 · typed 2.233e-6 · multilingual 5.187e-6), then
  3/3 through the re-armed gate (riir-reflex `ccb5bd0`). Regression arms in
  `elementwise_cubecl::tests` (chunked-offset parity + a 200-rep
  bit-identical repeat at the 16×400 score geometry) both fail 5/5 with the
  fix reverted. gpu lib 215/0; clippy `-D` workspace `--all-features
  --all-targets` green.
- **The second class is contested (Issue 019).** A concurrent session
  (`riir-infer-m3-t7c`) landed `c0dfa06`, which emits Metal memory barriers
  from the vendored `wgpu-hal`, and recorded 018 as two classes (`270740c`).
  The measurement above was taken in a clean worktree at HEAD plus ONLY
  `2a34bd3`, with no barrier code present, so **A alone was sufficient for
  the observed wobble**. Class B's localizing evidence (`scores` diverging
  with q/k/v clean) is read after the in-place softmax, which is A's site.
  Its premise also does not match the fork's encoders, which never set a
  dispatch type and so are serial. Necessity and per-rebind cost are
  tracked in Issue 019. Nothing reverted.
- Cubecl numbers are no longer provisional; plan 611 S5 may publish.
- The issue file is removed; its full text (the pre-resolution trail + the
  two-class record as written by `riir-infer-m3-t7c`) is at `270740c`.

## 2026-09-26 — the riir-ai carve docs adopted into this repo (Issues 998 + 1003, Plan 610, Proposal 041, Benches 870 + 871, doc 002)

The riir-infer formation records moved home to the subject repo (owner noise-reduction pass on riir-ai), **numbers kept verbatim** so every existing "riir-ai Issue 998 / Plan 610 / Proposal 041" citation resolves to the same document:

- `.issues/998_riir_infer_repo_promotion.md` + `.issues/1003_riir_infer_carve_remains_4090.md` — both OPEN (998: S8 remains, gated on riir-reflex 008 P2/T4; 1003: 4090-executable content done, remains = the owner-gated S6b edge-drop franchise + S8). riir-ai's issue ledger keeps 998/1003 allocated; its `.highwater` is unchanged.
- `.plans/610_riir_infer_gpu_carve_slice1.md` — IN PROGRESS (S8 remains).
- `.proposals/041_riir_infer_model_based_inference_split.md` — the founding split proposal (Phase 1 landed; Phase 2 executing via riir-reflex Issue 008). First file in this repo's new `.proposals/` ledger.
- `.benchmarks/870_ternary_recipe_audit.md` + `.benchmarks/871_issue879_t1_gdn_quant_certification.md` — the Issue 879 GDN quant-survival records (871's harness is `tests/issue879_gdn_quant_certification.rs` here since the carve; Issue 879 itself was resolved in the riir-ai ledger, record in riir-ai HISTORY.md).
- `.docs/002_riir_infer_core.md` — the crate doc from riir-ai's docs book (renumbered 002 in this repo's `.docs/` ledger).

Cross-repo relative links inside the moved files were re-pointed; prose paths inside them remain written from the riir-ai side at writing time (each file carries a Moved header note saying so). riir-ai side: index rows struck + BOUNDARY.md/AGENTS.md citations re-qualified + HISTORY.md tombstone, same window.

## 2026-09-25 — Issue 001 CLOSED: EXL3 (trellis-coded) weight format — reader complete, fused-GEMV lane closed at this tier

The full record (§1–§17, section numbers unchanged, so every `Issue 001 §N`
citation in `src/quant/exl3*.rs`, `riir-infer-gpu`, the benches and plans
still resolves) moved to
[`.docs/001_exl3_trellis_format_support.md`](.docs/001_exl3_trellis_format_support.md).

- **Banked (opt-in `exl3`, not promoted):** the safetensors-side grouped
  reader + `quant/exl3.rs` (T2 Option A, no `GgmlType` coupling), the CPU
  reference and LUT+rayon fast arm (`725a8a5`, 10.6–11.4×), the pack loader
  (`bd93a7b`), the CubeCL GPU decode v1/v2 (Bench 002; 71–87× wall on the
  4090), the sound bench harness (`9989e9f`), the whole-pack bit-exact
  gate (`425e0c5`: 573/573 layers, 26.48 G weights, 0 mismatches on CUDA
  AND Metal), and the fail-closed era gate at open (`55c154f`, known-good
  `{"1.4.2"}`, `open_unverified_era` escape).
- **The axis that is real is residency** (Bench 001: 3.4× vs f16; context
  0 → ~117k on 24 GiB), not decode throughput. Decode runs 21–29 Gw/s and is
  NOT bandwidth-bound; the binding mechanism is UNMEASURED.
- **Closed at this tier:** T7c-2a/2b/3 (fused GEMV). By composition,
  26.5 Gw/step ÷ ~28 Gw/s ≈ 0.95 s/step against a 10–20 ms incumbent, so
  §14's delivered-gain trigger cannot be met. Reopen triggers r1–r4 are in
  §17.6. r1 (a serving consumer of the residency axis) is the owner of the
  "engine serving integration" line, and it needs a NEW issue when it fires.

## 2026-09-25 — T10 rung 2 (Metal): the attn_rope hoist pre-pass — default-off, promotion probe pending

`attn_rope` now derives the Q/K rope ONCE per layer into a packed
`[2, seq, d]` device scratch (`5ef7442`), and `flash_attn`'s staging copies
the rotated K instead of re-rotating it per (query block × head × key
tile). Opt-in `LAYA_METAL_ROPE_HOIST=1`; the in-kernel rope arm stays the
shipped default and is bit-identical by construction (same expressions,
same order) — promotion follows the quiet-box position-balanced A/B
(instrument archived: riir-reflex `.benchmarks/033_rope_hoist_ab/`), not a
flag flip.

The landing's own defect is the reusable lesson: widening the buffer list
moved the flash staging to `[[threadgroup(11)]]` while the host kept
binding its length at index 9 — the unbound-pointer class. Small smoke
shapes passed on allocation luck; full-model G5 failed DEGENERATE (uniform
probs, agreement 12/26), which is what surfaced it. Fixed (9 → 11) in the
same commit: a shape-widening edit must move the host bind and the kernel
binding together.

Gates (both postures, hoist on/off): `metal_ops_smoke` 7/7 ×2,
`packed_forward_equiv` 4/4 ×2, `packed_same_shape_gate` 1/1 raw-bit,
lib 41/41; consumer G5 metal top-1 1.000000 ×3 checkpoints, drift ≤ 6.1e-6
×2; `laya_batch_parity` ×2; clippy `-D` ×3 feature postures.

Session: issue020-metal-lane, 1790285773

## 2026-09-25 — two silent MiniCPM5 (llama-arch GGUF) defects, found bringing it up as Issue 011's long-context fixture

Both defects produced finite, plausible output that was wrong, and both were settled against a reference implementation (HF `tokenizers` / `transformers` fp32) rather than by reasoning.

- **RoPE pairing — `0b26b9a`.** llama.cpp's converter (`LlamaModel.permute`) stores llama Q/K rows in the interleaved order (GGUF row `2j+s` = HF row `s·hd/2 + j`). This crate's RoPE, CPU and `riir-infer-gpu` alike, is rotate-half, and `load_llama_weights_gguf` never un-permuted. On MiniCPM5-1B, tinyshakespeare BOS+256, riir ppl went **259.93 → 69.0547**, against **69.0547** from HF fp32 on the same ids. Fix: `rope::unpermute_interleaved_rows`, llama loader only (gemma2/qwen2 GGUFs are NEOX and unpermuted).
- **BPE digit rule — `175f67f`.** `BpeTokenizer` hard-coded Qwen2's one-digit-per-pre-token rule for every GGUF. llama-bpe groups `\p{N}{1,3}`. Text without numbers tokenizes identically, which is why a 1000/1000 tinyshakespeare diff passed. After the fix, 54 271 of 54 271 ids match on number-dense text. How it was isolated: the passkey needle failed 0/2 at base, and HF `transformers` scoring riir's EXACT prompt ids failed on the same digits, so the fault was in the ids and not the forward. Now 2/2, answer ppl 2.81 → 1.21.
- **Downstream:** every consumer that re-exports this loader or tokenizer inherits both fixes. The riir-train canon cross-arch benches (422/423/424/426/427/605) paired Gemma-2 with this MiniCPM5 forward, so their verdicts are unverified: riir-train Issue 571.
- **Instrument kept:** `row_logit_floor_ppl --dump-tokens N`. In ppl mode it prints the corpus ids; in needle mode it prints each prompt as `score_from|ids`. It is the diff against a reference that found both defects.

## 2026-09-25 — Issue 010 CLOSED (REFUTED): Gemma-2 has no QK-norm; the 288-tensor fixture is upstream-faithful

Issue 010 (`95f52fb`) claimed `riir-train/data/gemma-2-2b-it-f16.gguf` was a
mutated conversion: 288 tensors with no `attn_q_norm`/`attn_k_norm`, against a
supposed llama.cpp standard of 340 (13/layer), and that upstream Gemma-2 applies
per-head q/k RMSNorm. **That premise is false.** Checked against upstream
HuggingFace `models/gemma2/modular_gemma2.py`: `Gemma2Attention` defines no
`q_norm`/`k_norm`, and bounds scores with `attn_logit_softcapping` (tanh, cap 50),
which this repo's forward already applies (`attention_head_softcap`). Per-head
QK-norm arrived in Gemma-3, where it replaced softcapping. The 11 tensors/layer
(attn_norm, q, k, v, o, post_attention_norm, ffn_norm, gate, up, down,
post_ffw_norm) × 26 + 2 globals = 288 is the standard shape.

- The fixture and forward are upstream-faithful. Option (a), re-converting, was
  never needed. Nothing pinned to the old artifact needs a re-run.
- `transformer/gemma4.rs`'s "q/k norm … NEW vs Gemma 2" doc was **correct** and
  stays.
- Corrected in the same commit: the `vk_calibration` bin's caveat 1 (doc and the
  two printed lines) and the `gemma2_calibration.rs` tap-law paragraph. For
  gemma-2 the pre-RoPE K tap is exactly the cache's K. Issue 883 trap 1's
  post-QK-norm wrinkle applies to gemma-3/4-class stacks only.
- Why it happened: a tensor count was compared against a remembered "standard"
  rather than against the upstream model definition. Check the reference source
  before filing a fixture-integrity finding.

## 2026-09-25 — Issue 009 CLOSED: software-pipelined narrow staging — NEGATIVE at the gate, and the probe's tiny-op floor exposed

The `.issues/008` follow-up ("double-buffered staging / cp.async — latency
hidden, not bandwidth saved") was built as the REGISTER form and answered
**NO on the pre-registered gate** — with two findings that outlive the rung.

**The rung** (built, measured, reverted): `sgemm_narrow_pipe` — the same
32×64×BK64 tile/threads/fragment/grid as `sgemm_narrow`, k-loop software-
pipelined: tile t+64's guarded loads issue into 12 registers at loop top, a
compiler fence (`asm volatile("" ::: "memory")` — measured load-bearing,
see below) pins their issue order, tile t's compute (~1300 cy) hides their
latency, stores land after the post-compute barrier. Double-buffering both
operands needs 51 456 B > the 48 KB static cap, so B ping-pongs (`tb[2]`)
while A register-defers into the single `ta` — 43 136 B total. Bit-identical
on every probe shape both postures, both runs; all gates green (smoke with
8 pipe-posture arms incl. the ragged/epilogue paths, packed_equiv 4/4, lib
41/41).

**Finding 1 — the fence iteration**: the first build measured FLAT (median
−1.8 % on the narrow rows) because nvcc SINKS the register loads to just
before their consumers (a register-pressure choice), collapsing the window
— the fence pins program order and is mandatory for this form. With it the
probe still read −1.5 % (−1.2..−2.3 on the m=106/45 rows).

**Finding 2 — the probe's tiny-op floor (the real yield)**: across the
narrow-zone rows, FLOPs vary 400× (1×1028×256 = 0.5 MFLOP at 43.5 µs →
106×1024×1024 = 222 MFLOP at 45.5 µs) while time varies 1.06× — every
row is `≈ 40-41 µs launch/WDDM floor + flops/50 TFLOP/s`. The
`sgemm_shape_timing` tiny-op rows are ~90 % submission floor and CANNOT
resolve kernel-level rungs in the narrow zone; `.issues/006`'s "head tails
−13..−17 %" rows were the same floor class (the float4 win was real but the
row magnitudes were floor-shifted), and 008's reg4 reading only registered
because a 4× warp cut punched the kernel ~75 µs PAST the floor. Future
narrow-zone rungs must gate at the FORWARD level (launches amortize in the
deep queue) or add a same-kernel floor-subtraction arm to the probe.

**The honest instrument**: forward-level paired env-flip (`laya_fixture_timing`,
english — the narrow-heavy checkpoint, 3 alternating pairs, quiet box):
off 12.6/12.6/12.6 ms → on 12.4/12.4/12.5 ms row p50 — **−0.8..−1.6 %,
reproducible 3/3** (p90 and ms/question improve too; the alternation kills
drift). The pipelined kernel IS genuinely faster — by ~1.6 %, not the
predicted ~25 %: TLP across 16 warps already hides most of the staging
convoy at the phase boundary, and the residual (2 barriers + the store
phase ≈ 1.5-2 % of the loop) is what the pipeline actually recovered.

**Verdict** (the pre-registered gate binds): both instruments read under
the ≥3 % bar → NO-GO. `cuda.rs` + the smoke arms + the reflex probe env
dance reverted byte-identical to the pre-rung state; the kernel is not kept
behind a switch because no posture clears the bar. The cp.async form
(1 sync/iter, no register round-trip) remains the recorded next lever,
priced by this record at ≤ ~4 % forward (the barrier + store residual) —
not worth a rung unless the forward-level instrument shows the narrow share
growing.

## 2026-09-25 — Issue 008 CLOSED: the narrow reg4 rung — NEGATIVE, the narrow zone is TLP-bound, not bandwidth-bound

The `.issues/007` open question ("the NARROW instance (1×4 fragment, ~20 %
ceiling, m<256 single-wave zone) is the next relative laggard — a narrow
reg4 arm is the recorded follow-up rung") is answered **NO on measurement**,
and the measurement's real yield is the mechanism: **the narrow zone's
binding constraint is thread-level parallelism, not shared-memory
bandwidth** — register blocking, which trades warps for bytes-per-FMA, is
the wrong currency exactly where the ladder runs 1 block per SM.

**The rung** (built, measured, reverted): `sgemm_narrow_reg4` — the 007
family 4×4 fragment (16 accumulators, 4 LDS.32 + 1 LDS.128 per 16 FMAs =
2 B/FMA vs the 2-acc narrow's 5, ceiling 20 %→ 50 % on the 007 roofline
arithmetic). The fragment-area law forces a 32×16 warp tile; on the narrow
32×64×BK64 tile that is a 1×4 warp grid = **128 threads** (vs 512). Same
tiles, same staging footprints, same grid arithmetic — every block-fit
property carries over; per-output accumulation k-ascending in one thread
(bit-identical held on EVERY probe shape, both postures, both runs).

**Measured** (`sgemm_shape_timing`, the `LAYA_CUDA_REG4` A/B axis — the
narrow rows went live for the first time; two full runs, quiet box, GUI-class
load only): every true narrow-served row regressed **+41..+177 %** —
106×1024×1024 50→125 µs (+150/+139 %), 106×2624×1024 119→332 µs
(+178/+171 %), the n=1536/2048 crossover rows +139/+152 %, short-seq
45×1024×1024 +151/+157 %, the m=4/1 head tails +118/+41 %, act a0 +113 %.
Ten times outside the ±8-15 % instrument band — no `--control` arm needed to
adjudicate. The wide/xwide-class rows (n ≥ 2560 at m=106, banking77, the
packed zone) re-measured the 007 rung at −2.6..−20 % on both runs — the
consistency check; the one historically noisy row (packed O 424) flipped
+12 %→−12 % between runs exactly as the recorded band predicts.

**The mechanism, read off the kernel's own structure**: the narrow zone's
grids are ≤ 128 blocks BY CONSTRUCTION (the block-fit pick) — 1 block per SM,
so the SM's latency hiding is the block's warps and nothing else. The
k-loop is sync→stage→sync→compute: staging is 48 global loads per thread
per k-tile (A 16 + B 32 at 128 threads), and with 4 warps the DRAM/L2
latency of a stage is exposed almost fully — 16 warps (512 threads) at
least keep 4× the loads in flight and 4× the barrier-arrival slack. The
2 B/FMA bandwidth win is real but irrelevant: the measured narrow ceiling
was never the 20 % roofline — at m=106 the 2-acc narrow computes at ~5 %
of fp32 peak (50 µs for 222 MFLOP), i.e. the zone sits ~4× under its own
bandwidth ceiling already. Cutting warps scaled the wall time almost
exactly as TLP arithmetic predicts (512→128 threads ≈ 4× offered parallelism
→ 2.3-2.8× wall on the load-dominated loop).

**Verdict** (the pre-registered gate's NO-GO path): no partial win, no
carve-out — every true narrow row lost. `cuda.rs` reverted byte-identical
to the pre-rung state (the BK48 precedent — only the docs carry the
negative); the kernel is not kept behind a switch because there is no
posture in which it wins.

**The follow-up rung this record opens** (replacing the 007 pointer):
the narrow zone needs LATENCY hidden, not bandwidth saved —
**double-buffered staging / `cp.async`** (prefetch k-tile t+1's A/B while
computing t, sm_89's async copy path bypassing registers) is the recorded
next lever. Note the symmetry with the 007 record: double buffering was
correctly dismissed for the BANDWIDTH-bound wide zone and is exactly the
right repair for the LATENCY-bound narrow zone — the two zones' rooflines
diverge, so their rungs must too.

## 2026-09-25 — Issue 007 CLOSED: the register-blocking sgemm rung — 4×4 fragments, −9..−21 % kernel on every wide/xwide shape

The `.issues/006` follow-up rung ("register blocking / double-buffered
staging — the instances sit at 25-30 % of fp32 peak") landed as REGISTER
BLOCKING, and the choice between the two candidates was settled by the
arithmetic, not taste: after the float4 rung the inner loop is
**2×LDS.32 (A) + 1×LDS.128 (B) ≈ 6 SM-cycles of shared-memory bandwidth per
8 FMAs ≈ 2 SM-cycles of FFMA capacity** — a ~33 % shared-bandwidth roofline
at full occupancy, matching the measured 25-30 %. The lane is BANDWIDTH-
bound, not latency-bound, so double buffering (a latency repair) buys
nothing; register blocking raises the ratio.

**The rung**: `sgemm_wide_reg4` + `sgemm_xwide_reg4` — the SAME tiles and
grids as wide/xwide at HALF the threads (256 / 512), warp tile 32×16 (warp
grid 2×4 / 2×8), thread fragment **4 rows × 4 cols = 16 accumulators**:
4 LDS.32 + 1 LDS.128 per 16 FMAs = **2 B of smem reads per FMA (the 2-acc
instances' 3)** → ceiling 50 %. Same grids mean every measured ladder
property (block-fit cliffs, straggler tails) carries over untouched; xwide
additionally rises to 3 blocks/SM = 48 warps (the 1 024-thread instance
capped at 32). Per-output accumulation stays k-ascending in ONE thread —
the result-identity law is untouched. Kill-switch `LAYA_CUDA_REG4=0` holds
the 2-acc posture (the `LAYA_CUDA_LADDER` contract).

**Measured** (`sgemm_shape_timing`, the A/B axis moved to
`LAYA_CUDA_REG4`; two full runs + `--control`): every wide/xwide-served
row negative in run 2 — banking77 zone −9.4..−16.1 %, the packed
multi-wave zone −9.4..−21.4 % (O m=1268 175→151 µs, down 436→345, QKV
477→387, gate/up 735→637), the m=106/45 n≥2560 rows −10..−22 %. Run-1's
lone positive (424 QKV +3.5 %) flipped to −9.4 % in run 2 — inside the
instrument band, which the `--control` arm (SAME-kernel pairs) measured at
−14.9..+10.1 % on this box this hour (the sibling session's builds — wider
than the recorded ±8 %, why every row was read against it). Narrow-served
rows flat on both runs (both postures route narrow — the design's
consistency check). Live-row median ≈ −12..−13 %, gate was ≥3 %.

Forward level (paired same-binary env-flip, 3 alternating pairs):
english 13.1→12.4 ms row p50 (−5.3 %), typed 12.9→12.4 (−3.9 %),
multilingual flat — the dilution is structural: at fixture seqs only the
n≥2560 projections route wide-class (the rest narrow, unchanged; attention
and the row kernels untouched).

**Published row refresh** (reflex `.benchmarks/031`, 15 suites PASSED,
host 4090-windows): every suite ≤ 030's p50 — median −6.2 %, best −9.0 %
(the packed typed_decisions trio), the short fixed-overhead suites flat.
Accuracy 14/17 rows BIT-IDENTICAL; the three typed_decisions rows wobble
3-7 cases in 2000 — that lane's pre-existing `determinism_ok: false`
variance (false in 026/028/029/030 AND 031, on record since v1).

Gates at the landing: `cuda_ops_smoke` 4/4 (the ladder-boundary + ragged
arms now route through the reg4 kernels at the default posture) ·
`packed_forward_equiv` 4/4 · lib 41/41 · consumer G5 at the cuda posture
2/2 (top-1 1.000000 ×3, prob drift ≤ 3.3e-6 — the lane's own class) ·
`laya_batch_parity` 1/1 · clippy clean both repos. Remaining upside on
record: the reg4 instances should now sit near ~40 % of fp32 peak (the
50 %-ceiling minus overheads), and the NARROW instance (1×4 fragment,
~20 % ceiling, m<256 single-wave zone) is the next relative laggard — a
narrow reg4 arm is the recorded follow-up rung, measured before built
(the `.issues/007` §Open question).

## 2026-09-25 — Issue 006 CLOSED: the float4 sgemm rung — every instance's B loads collapsed, −7..−17 % on every suite

The `.issues/004` open question ("the packed multi-wave zone has no measured
win — split-K or an occupancy-tuned instance") is closed with the question
REFRAMED and answered: the zone's problem was never the straggler tail —
it is LOAD-ISSUE THROUGHPUT. The probe's appended packed-zone population
(m=424 ≈ 4×106, m=1268 ≈ 4×317) measured the wide instance at **12-16.5
TFLOP/s = 15-20 % of the 4090's 82.6 fp32 peak** (gate/up at m=1268:
13.65 GFLOP in 829 µs), 2.5-3× under the cuBLAS-class roofline — and the
inner loop's own shape explains it: **6 smem loads per 8 FMAs**, with the
four B-fragment loads CONTIGUOUS in the staging tile.

**The rung**: every instance's B staging row pads to a 16 B multiple
(wide 65→68, narrow 65→68, xwide 129→132 — col0 is a multiple of 4 at
every call site, so `&tb[kk*pad + col0]` is float4-aligned BY CONSTRUCTION)
and the four B loads collapse to ONE `reinterpret_cast<float4>` — 6 loads
per 8 FMAs becomes 3 (narrow's 1×4 fragment: 5 → 2). float4 loads change
no arithmetic order, so the instances stay result-identical by construction
(verified BIT-IDENTICAL on every probe shape, both arms) and every ragged
edge is staging-side-safe (out-of-range elements stage as 0.0f; stores stay
guarded per element).

**Measured** (`sgemm_shape_timing`, in-process base/ladder A/B with the
experiment arm on the would-be-wide slots, two full runs): −10.3..−25.2 %
on the multi-wave zone rows, −5..−14 % on the single-question wide rows,
the m=4/m=1 head tails −13..−17 % (those are narrow-served — narrow's own
float4 win). Landed (both arms new kernels), the absolutes vs the morning
baseline: gate/up m=1268 839→727 µs (−13 %), QKV m=1268 536→445 (−17 %),
down m=1268 470→406 (−14 %), m=317 QKV 158→125 (−21 %). Forward level
(fixture probe, cross-binary same-box same-session): english 15.1→12.9 ms
row p50 (−14.6 %), multilingual 7.2→6.3 (−12.5 %), typed 15.1→12.8
(−15.2 %).

**The published row refresh** (reflex `.benchmarks/030`, 15 suites
PASSED, host 4090-windows): every suite −6.8..−16.7 % p50 — the packed
suites TOO this time (typed_decisions 109→100/59→55/109→99, banking77
28→25, code_fixtures 31→28) — the multi-wave zone's first measured win,
which was the `.issues/004` open question. Accuracy: 13/16 rows
BIT-IDENTICAL; the three typed_decisions rows wobble 1-4 cases in 2000
within that lane's pre-existing `determinism_ok: false` variance (false
in 026, 028, 029 AND 030 — on record since v1, independent of every
kernel rung).

Gates at the landing: `cuda_ops_smoke` 4/4 (tile-edge + ragged arms on the
new kernels) · `packed_forward_equiv` 3/3 · lib 41/41 · consumer G5 at the
cuda posture 2/2 (top-1 1.000000 ×3, drift ≤ 5.1e-5 class) ·
`laya_batch_parity` 1/1 · clippy clean both repos. No kill-switch added —
the float4 form IS the wide/narrow/xwide kernels now (same tile geometry,
same pick, same result chain; `LAYA_CUDA_LADDER=0` still holds the ladder
A/B posture). Remaining upside on record: the instances still sit at
~20-24 TFLOP/s ≈ 25-30 % of peak — the next rung (register blocking /
double-buffered staging) is a bigger redesign, not landed today.

## 2026-09-25 — Issue 005 CLOSED: CUDA graphs — NEGATIVE, the lane is GPU-bound (submit fully hidden)

The standing follow-up rung (recorded at the close of 002/003/004: "CUDA
graphs for per-op dispatch overhead") is closed on measurement, with the
pre-registered go/no-go gate of `.issues/005` §2 answered NO by a DIRECT
device-timeline reading rather than by the submit/wall ratio alone.

**The instrument (ships, `LAYA_CUDA_STATS=1`)**: `begin_pass` stamps t0 and
records a timing event at the head of the now-idle stream; the pass's FIRST
`download_into` (the pipeline-draining sync) records the end event, syncs,
and prints `submit` (pre-sync − t0: the whole CPU path — host embedding
gather, H2D uploads, slot `cuMemAlloc`s, every launch call), `wall`
(post-sync − t0), `gpu` (the device-timeline elapsed between the two events
— the GPU critical path of the prefix, inter-kernel gaps and alloc-induced
stalls INCLUDED), plus the submit-path decomposition (upload count/time/
bytes, alloc count/time, accumulated at the call sites). Zero cost when the
env is unset — the events are only created under the flag, and the stats-off
posture re-measured byte-stable (english 15.1 ms / multilingual 7.2 ms row
p50, identical to the pre-instrumentation readings).

**The measurement** (fixture probe, all three checkpoints, reps=3, GPU
exclusive — GUI apps only, the owner-call exemption):

| row class | submit | wall | gpu | uploads | allocs |
|---|---|---|---|---|---|
| english p50 | 4.6-5.2 ms | 12-15 ms | **= wall ±0.3 %** | 6x ~0.07 ms | 18-25x ~0.2 ms |
| multilingual p50 | 1.7-2.8 ms | 6.1-7.2 ms | **= wall ±0.1 %** | 6x ~0.04 ms | 19-24x ~0.13 ms |
| typed p50 | 4.5-6.5 ms | 12.8-15.7 ms | **= wall ±0.3 %** | 6x ~0.08 ms | 19-24x ~0.2 ms |
| long rows (63-133 ms) | 16-32 ms | 63-133 ms | **= wall** | ≤0.8 ms | ≤1.9 ms (one 6.8 ms malloc hiccup) |

**`gpu == wall` on every row of every checkpoint.** The device timeline
fills the entire wall: the CPU submit path — all of it, launches, uploads,
allocs — executes entirely INSIDE the GPU's execution window. The
pre-registered GO premise ("the CPU path co-determines the wall") is false
here, and the ratio arm of the rule (multilingual max 0.46 < 0.5, median
~0.33) concurs. A graph replay would remove CPU work that is already free;
the remaining win is bounded at GPU-side inter-kernel gap reduction
(~300 launches × ~0.5-1 µs ≈ 0.15-0.3 ms ≈ 1-2 % of wall) — under the
lane's measured noise band (the `sgemm_shape_timing` control's ±8-10 %
two-context artifact band; the ±6 % idle per-round spread) — while the
signature-keyed arena + pinned-slot + staging-refresh machinery would add
exactly the stale-replay correctness surface `.issues/003` exists to
prevent, and per-signature capture costs more than it saves on novel
shapes (the serving stream's one-shot signatures). The pass TAIL (act head:
host math between three data-dependent downloads) is unreachable by graphs
by construction.

Two observations recorded for the future (the reopen triggers, both on the
instrument, one command away):
1. **The launch path costs ~15 µs/call** (submit minus gather/upload/alloc
   over ~300 launches) — expensive per call but FULLY HIDDEN at every
   current geometry. If the kernels ever get much faster (the ladder rungs
   compound, or a bigger GPU), the submit path becomes the wall and the
   ratio flips — re-run the probe before reopening, the number decides.
2. **Per-pass allocs are ~0.2 ms and uploads ~0.05-0.1 ms** — both already
   hidden; no slot-pooling or pinned-staging rung is warranted either (the
   cheaper remedies the issue §3 pre-identified die by the same evidence).

Gates at the close: `cuda_ops_smoke` 4/4 · `packed_forward_equiv` 3/3 · lib
41/41 · consumer G5 at the cuda posture 2/2 (top-1 1.0, drift ≤ gate) ·
`laya_batch_parity` 1/1 · clippy clean both repos · stats-off parity rows
byte-stable. No published number changes (nothing shipped that moves
them); the instrument is debug-only.

## 2026-09-25 — Issue 004 CLOSED: the sgemm tile ladder — block-fit floors, the narrow single-question win

The `.issues/002` v1 backend ran ONE sgemm instance (64×64×32, 512
threads) for every GEMM shape. The lane now ships THREE instances —
`sgemm_narrow` (32×64×64, 512 thr, staging A[32][65]+B[64][65] = 24 960 B),
`sgemm_wide` (the v1 kernel, renamed) and `sgemm_xwide` (64×128×32,
1 024 thr, staging A[64][33]+B[32][129] = 24 960 B) — picked per call by
**BLOCK-FIT on the SM count**, a floor MEASURED on this box, not ported:
the M3 Metal lane's `m < 256` threshold does NOT transfer to the 128-SM
4090 (it is subsumed by the block-fit arithmetic on the real population).

**The cliff that sets the floor** (the probe's CUDA arm,
`sgemm_shape_timing`): at m=106 the narrow instance wins −15.7 % at
n=2048 — exactly 128 blocks, one per SM — and LOSES +46 % at n=2560 —
160 blocks: static block scheduling strands 32 SMs at 2× work while 96
idle after one. Every instance whose grid exceeds the SM count pays that
straggler tail, so the pick is: narrow iff its grid (× batch) fits one
wave; xwide iff m≥256 ∧ n≥2048 ∧ its grid fits (the multi-wave zone —
gate/up at n=5248, 205 blocks — reverts to the proven wide instance:
readings sat inside the instrument's measured ±8-10 % two-context
artifact band). Per-output accumulation stays k-ascending in ONE thread
on every instance, so the ladder is result-identical by construction —
and measured: 13/16 bench suite-lane rows bit-identical vs the 028 run
(every single-question suite; the three typed_decisions rows wobble 1
case in 2000 within that lane's pre-existing `determinism_ok: false`
variance, on record since the 026 v1 run).

Measured wins: per-shape narrow −13.7..−19.6 % (n=1024/1536/2048, the
m=4/m=1 head tails −12..−17.5 %); xwide QKV (120 blocks) −6.8..−8.5 %
across three runs; forward-level A/B on the fixture rows (ABAB,
single-backend-per-process): english −10.4 % · multilingual −14.1 % ·
typed −8.4 %. Published row refresh (reflex `.benchmarks/029`): the
single-question suites −6..−14 % p50, packed suites flat by the
conservative floor. Kill-switch `LAYA_CUDA_LADDER=0` (wide everywhere —
the A/B posture, never a silent default).

A launch defect fixed in passing: the v1 form passed the staging
footprint as DYNAMIC shared memory on top of the kernels' STATIC
`__shared__` arrays — harmless at wide's 2×16 768 B, but the new
instances' 2×24 960 B crosses the 48 KB static default and the launch
dies `CUDA_ERROR_INVALID_VALUE` (caught by the first smoke arm — the
static-smem constant never reached the launch). All instances now
launch with dynamic smem 0; the footprint constants live on as
compile-time bounds.

Gates at the final floors: `cuda_ops_smoke` (with the new boundary arms
— m=33/255/256, n=65/2047/2048/2080, k=33/63/65, m=321 — every tile
edge on every instance) · consumer G5 at the cuda posture ·
`laya_batch_parity` · `packed_forward_equiv` · clippy −D warnings both
repos. Follow-up rungs stay open: CUDA graphs for per-op dispatch
overhead; the packed multi-wave zone has NO measured win yet — split-K
or an occupancy-tuned instance is the open question, not another tile
size.

## 2026-09-25 — Issue 003 CLOSED: CUDA flash attention — the packed-path zeros defect fixed + the fused rung

The `.issues/002` v1 posture ran `attention_forward` through the TRAIT
DEFAULT op sequence on CUDA. That default SLICES host memory
(`&qkv[qkv_off..]`) — correct at offset zero (the slice IS the parent
the device op wrote → same `(ptr,len)` chain key → hit → device-current)
and SILENTLY WRONG at non-zero offsets: the packed multi-question
forward's slice is a NEW key → chain MISS → uploads the host bytes,
which under the write-first discipline are STALE (the parent was written
device-side only; the host vec holds its `resize(.., 0.0)` zeros).
**Every multi-question case's attention ran on zeros at v1.**

The evidence was the published bench, not the code: reflex
`.benchmarks/026_4090windows_cuda` vs `018_4090windows_run` (CPU, same
box) — typed_decisions english 0.3575→0.2690, multilingual 0.3490→0.2690,
typed **0.7445→0.2690** (−47.5 pt), code_fixtures 0.5417→0.2917 (2
q/case), while every 1-question-per-case suite was byte-identical
(ag_news 0.9500, banking77 0.4980). The consumer-side G5 passed green
because its fixture rows are single-question (offset zero — the correct
path); `laya_batch_parity` (the multi-question gate) had not been run at
the cuda posture. The 026 close-out's "accuracy byte-identical on every
lane" claim was wrong for the multi-question suites — corrected in the
consumer repo's bench doc the same day.

The fix IS the rung: the Metal lane's one-pass online-softmax flash
kernel (MSL_FLASH, the reflex Issue 020 T10 rung-3 form) ported to CUDA C
at plain fp32 FMA — ONE dispatch per layer over the packed qkv (split,
rope, q-scale, scores, sliding window, softmax, value mix, head merge
in-kernel; the seq² scores parent never exists; **offsets bind at
dispatch**, so the packed forward is the unbatched kernel's exact math —
Metal's design, which is why the Metal lane was immune). 256 threads,
one block per (32-row query block, head); shared staging
tq[32][65]/tk[64][33]/tv[32][65]/ts[32][33]/tacc[32][65]+mrow/lrow/arow
= 38 016 B dynamic smem; the online rescale α = expf(m_old − m_new) is
exactly 1.0f when the max does not move. Kill-switch `LAYA_CUDA_FLASH=0`
→ the reference sequence, which now PANICS on non-zero offsets (the
Metal guard — the silent zeros are now a loud breach, and
`supports_packed_attention` answers false there so the agent takes the
per-question loop). `needs_window_mask` mirrors the armed path (the
encoder stops building `[seq,seq]` masks on the fused lane).

Gates green on this box, CUDA 13.3 / driver 610.62, GPU clear of compute
consumers (GUI apps only — the exempt class):
- `cuda_ops_smoke` + 3 new arms: fused full (seq 1/9/37/64/129 — every
tile edge) drift 1.2–1.8e-7; sliding (w8@64/w4@37/w16@130) 1.2–1.8e-7;
  the PACKED-offsets arm (two sequences at non-zero qkv/rope/out offsets,
  the encoder's whole-parent call shape) 1.2e-7 — GREEN FIRST RUN.
- `packed_forward_equiv` gained a CUDA arm (non-macOS,
  `laya-riir-cuda`-gated) — the gate class that catches the zeros defect
  at the substrate level; verified it FAILS LOUD under
  `LAYA_CUDA_FLASH=0` (the fallback's offset guard panics).
- Consumer-side at `LAYA_DEVICE=cuda`: `laya_batch_parity` — 26+26+36
  batched multi-question forwards, top-1 1.000000, drift ≤ 5.1e-5
  (~20× under the 1e-3 gate) — THE gate that would have caught v1; G5
  parity english 3.3e-6 · typed 1.3e-6 · multilingual 4.7e-6 (the
  online-softmax restructure's expected class, ~200× under the gate).
- Latency (fixture rows, short seqs — flash vs `LAYA_CUDA_FLASH=0`):
  english 17.5→16.2 ms (−7.4%), multilingual 9.0→8.5 (−5.6%), typed
  17.6→16.6 (−5.7%). The long-seq suites gain far more (the seq² scores
  traffic ~6×`heads·seq²·4B` per layer never exists, and the windowed
  key walk cuts attention FLOPs ~2.4× at window 64) — measured in the
  consumer repo's refreshed bench.

The bench refresh + the published-numbers correction live in the consumer
repo (reflex `.issues/028` — renumbered from 027 after a same-window
dual-allocation; the record is reflex HISTORY §2026-09-25 bench 028).
Remaining follow-up rungs (open, each G5-gated at the cuda posture): the
sgemm tile ladder (closed same day as `.issues/004`), CUDA graphs for
per-op dispatch overhead.

Session: 4090-cuda-flash, 2026-09-25

## 2026-09-25 — CalibrationTables promoted substrate-side (883 P0 Kimi fixture rider)

The gemma-2 harness's layered table builder moved upstream:
`katgpt_core::fitted_anchor_table::LayeredVkCalibration` (with
`VkLayerTables`) is now the ONE builder for both 883 P0 fixtures — this
repo's gemma-2 dashboard and katgpt-rs's new Kimi-K3 dashboard (katgpt-rs
Bench 889, real weights: ρ(V)≈ρ(K)≈ρ(V−K) per MLA layer — the "coupled
through one latent" signature; KDA layers = fixture-class null, no KV
cache). `gemma2_calibration.rs` re-exports it under the historical name
`CalibrationTables` — zero behavior change, the `vk_calibration` bin
compiles + clippy-clean unchanged. The tap forward, corpus loader, and
dashboard format stay here (gemma-specific); the row-map/triplet/scratch
plumbing is substrate-owned (DRY: one builder, two fixtures).

## 2026-09-24 — Issue 002 CLOSED: the CUDA backend for the laya lane (the 4090 bench row, 17–60× the CPU posture)

Landed `9b52cb1`/`99f156e`→rebased `e99d767`: the lane's third compute
backend, `laya-riir-cuda` — cudarc 0.19 (`driver`+`nvrtc`, `cuda-13030`+
`fallback-dynamic-loading`, target-scoped `not(macos)`, one workspace version
with `riir-infer-gpu`'s raw-CUDA lane), CUDA C compiled to PTX at construction
(NVRTC, arch `sm_89`). The Metal backend's architecture ported verbatim:
permanent `(ptr,len)` weight cache, epoch-keyed chain slots with `begin_pass`
invalidation, `download_into` prefix-read barrier (a `CudaView` slice —
`memcpy_dtoh` asserts on the longer parent slot), lazy async submission on one
stream. ONE strided batched sgemm (64×64×32 tile, 512 threads, fp32 FMA;
the weight binds row-major `[n,k]` directly — the B-tile loader maps lanes
along whichever stride is 1, so NO device transpose cache) + the 14 tail
kernels at the CPU lane's exact semantics. Attention v1 = the trait DEFAULT
op sequence, fully device-side (attention <6% of forward FLOPs at the pinned
geometries — a fused flash kernel is a follow-up rung, tracked below).
`CudaSlice::clone()` is a device-to-device COPY in cudarc (not a refcount
bump like Metal's `Buffer`) — the caches hold `Arc<CudaSlice<f32>>`.

Two kernel defects found by the op gate before any G5 run: the staging loops
loaded 1024 of 2048 tile elements at 512 threads (the Metal kernel's 1024-
thread q<2 shape copied straight over), and the `b_cs==1` staging branch
dropped the `n0` column offset (columns ≥ 64 served tile 0's B — the tiny
shapes passed, n=128 failed from column 64 on; a ones/identity ladder
localized it in two runs).

Gates (all green on the 4090 box, CUDA 13.3 / driver 610.62):
- `cuda_ops_smoke`: every backend op vs the CPU free fns — bit-exact data
  movement, 1e-7…1e-4 reductions;
- consumer-side G5 at `LAYA_DEVICE=cuda`: english 26/26 top-1 drift 1.863e-6,
  typed 26/26 2.471e-6, multilingual 36/36 2.894e-6 — the Metal drift class,
  ~500× under the 1e-3 gate, GREEN FIRST RUN;
- `packed_forward_equiv` at the cuda posture (the same-day concurrent
  packed-forward landing composed cleanly — `copy_at` added at the rebase).

Measured (fixture rows, same-session A/B on this box): english 210.4→17.5 ms
(12.0×), multilingual 95.0→9.0 ms (10.6×), typed 208.4→17.4 ms (12.0×) —
every checkpoint BELOW the M3 Metal row (28.3/12.2/28.3 ms) at v1. The full
15-suite bench refresh + the site publish record lives in the consumer repo
(riir-reflex `.issues/026`, `.benchmarks/026_4090windows_cuda/`). Follow-up
rungs (open, unordered): flash-attention port (the MSL two-pass online-softmax
form), tile ladders for the m<64/n≤1024 shapes, CUDA graphs for the
per-op dispatch overhead — each G5-gated at the cuda posture before any
number replaces a published one.

## 2026-09-23 — crates.io publication: keep `publish = false` until the vendor patches upstream (owner-gates menu v2 row 2)

Owner verdict: the crate and `crates/riir-infer-gpu` stay **closed to
crates.io** — this is a HARD blocker, not a preference. Both vendor forks under
`vendor/` are load-bearing (`[patch.crates-io]` in the root manifest):

- `cubecl-runtime` — carries the #1359 drop-queue fix.
- `wgpu-hal` — carries the VRAM accessors the GPU code probes.

A `[patch.crates-io]` section does not survive publication: a crates.io
consumer would build against the UNPATCHED upstream crates — the drop-queue bug
and the missing accessors included. Publication becomes available the day the
vendor deltas land upstream (the forks shrink to zero and the patches drop out
of the manifest).

Boundary note: this repo stays upstream of the engine regardless — the public
funnel for the stack's primitives remains `katgpt-rs` (its katgpt-core family
publishes), not this repo.

Session: owner-gates-m2, 1790121600

## 2026-09-25 — Issue 020 T6 CLOSED NEGATIVE: the encoder's host side is 1–1.6% of forward wall (measured before building)

T6 proposed pooling the per-forward allocation churn — ~60 MB of host `Vec`s
rebuilt every forward (`encoder.rs` `Scratch::new()` + `gathered`/`h`/`out`/
rope tables) plus every activation device buffer freed by the per-pass chain
clear (`metal.rs` `begin_pass_impl`). The 09-25 head scratch-pool rung (built,
measured, moved nothing, reverted — the consumer repo's issue-020 follow-up)
already said the head is dispatch+GPU bound; this measurement settles the
encoder's half the same way, and it was taken BEFORE building the pool (the
discipline the head rung paid for).

Instrument: `crates/riir-infer-laya/tests/metal_host_gpu_split.rs` — a
`#[ignore]`d, `required-features = ["laya-riir-metal"]`-gated, measurement-only
probe (`--ignored --nocapture`). It splits one `Encoder::forward_packed` at the
real english geometry (d 1024, 28 layers, intermediate 2624) into `enq` (the
wall of the forward alone — the body contains no sync, so that is the whole
host side: MSL dispatch encoding, the chain uploads + destination slots, the
host `Vec` churn) and `sync` (`download_into`'s commit + wait — the GPU side).
The decision rule was recorded in the probe doc before measuring: pooling can
shrink only part of `enq`; if `enq` is a small fraction of the wall, T6 closes
NEGATIVE; a host-bound reading under load defers to a quiet box.

Measured 2026-09-25, M3, AC, load 11–12 (falling), 88% RAM free, one sibling
CPU bench (~6 cores, memory-bandwidth pressure — which inflates BOTH the host
reading and the GPU's unified-memory reads), no GPU consumers, 9 rounds/shape,
2 warmups:

- seq188: enq p50 **0.80 ms** (min 0.64) · sync p50 **48.7 ms** (min 36.1) — host share **1.6%**
- seq512: enq p50 **1.45 ms** (min 1.29) · sync p50 **141.6 ms** (min 132.6) — host share **1.0%**
- packed2x256: enq p50 **1.37 ms** (min 1.22) · sync p50 **135.2 ms** (min 130.7) — host share **1.0%**

The host reading is load-INFLATED (CPU contention inflates the host, never the
GPU), so the verdict is robust in the recorded direction: a quiet box only
shrinks the 1.0–1.6%. Allocation pooling can shrink only part of that share —
dispatch encoding stays — and cannot move case wall. The per-pass chain clear
STAYS (its staleness guard is load-bearing: host-authored buffers are rebuilt
per forward at recycled heap addresses, the Issue-015 class). The probe stays
as the standing instrument for any future host-side rung claim.

Also landed in the same commit: the `[[test]]` required-features row for the
new target (the T1.1e repo-birth law — the row keeps a feature-less selection
skipping loudly instead of printing a green zero).

Session: issue020-t6, 1790323200

## 2026-09-25 — the narrow sgemm's shape is the measured local optimum (T7 occupancy axis refuted, both arms)

The question a future kernel reader will ask: narrow stages A[32][65] +
B[64][65] = 24 960 B — one threadgroup per core — why not shrink the staging
so 2–3 co-reside and hide the staging-load latency? Measured from the
consumer's `sgemm_shape_timing` probe (reflex `ebe667e`, the record lives in
its issue-020 T7 section): two challengers behind a temporary
`LAYA_METAL_SGEMM_VAR` flag — **bk32** (BK 32, same 32×64 tile, 12 544 B → 2
TGs/core) and **bn32** (32×32 tile, 8 448 B → 3 TGs/core), both keeping the
k-ascending per-element chain (every row bit-identical to `sgemm`).

bk32 LOST 17–33% on every resolvable cell, growing with k exactly as the
barrier model predicts (BK 32 doubles the k-loop's two barriers); bn32 lost
harder (15–45% — the same doubling plus halved B reuse). The 2–3× co-residency
gain is strictly smaller than the barrier cost. At BK 64 a 2-TG fit would
break the bank-conflict padding (stride 65) or the 8×8 block structure (BN ≤
24 idles 4 of 16 simdgroups); wide (BM 64) already lost zero-for-zero in the
band rung. The variant code never landed — this record and the reflex issue
carry the negative, the BK=48 precedent.

Session: issue020-occ, 1790323200

## 2026-09-25 — the sgemm MMA-roofline probe: the narrow instance is staging-bandwidth-bound (reflex Issue 020 T7 follow-up)

`crates/riir-infer-laya/examples/sgemm_roofline.rs` (new, measurement-only,
`[[example]]` required-features row per the T1.1e law): the f16-axis
discriminator. A verbatim copy of the shipped narrow kernel (CPU-drift-checked
on the ragged cell) against an MMA-only twin — same inner k-chunk, staged
ONCE, `reps` iterations of the 8-wide chunk with no re-staging and no
barriers, FLOPs matched by reps = k/64, position-balanced rounds in one
process. Measured (AC, load 2.4–3.1): 317×3072×1024 — narrow 3.13 TF/s vs
roofline 5.11 (+63%); 1024×3072×1024 — 4.88 vs 9.21 (+89%); 1024×8192×1024 —
4.37 vs 10.24 (+134%). The traffic math closes: per k-tile each threadgroup
stages 24 KB (A 8 + B 16), so cell 3's B re-reads alone are ≈1.6 GB ≈ the
measured 3.93 ms wall at ~400 GB/s. Verdict: NOT MMA-bound — the f16 lever
is the B-OPERAND BYTES (halving staging + device reads; predicted ~1.4–1.7×
on the hot cells), not the f16 MMA; double-buffering is dead with it
(bandwidth-bound, not latency-bound). Numerics note for the rung: f16 weight
rounding is ~4.9e-4 relative per term (< the 1e-3 G5 gate) but promotion is
an Issue-750-T3 lossy-surface call — per-family retention, never the
aggregate.

## 2026-09-25 — f16-B staging REFUTED at kernel level (the roofline probe's own follow-up arm)

The roofline verdict (staging-bound, +63..+134% MMA-only headroom) predicted
the f16-B lever: halve the B-operand bytes (16 of 24 KB staged per k-tile),
halve the dominant traffic, ~1.4-1.7x on the hot cells. Measured (the probe
extended with a `sgemm_hb` arm — `device const half*` B, converted to f32 at
staging, everything after the staging bit-identical; f16 seed via a
round-to-nearest-even `f32_to_f16`; plumbing verified against the f32 arm
within the 2e-2 rounding gate): **flat within ±2% on every cell** — 317x3072x1024
−1.1%, 1024x3072x1024 +1.9%, 1024x8192x1024 −0.8% (the deep-k cell where the
model predicted the most). Load 8.9-9.5 (sibling resumed) — irrelevant: both
arms inflate together and the ratios are the decision axis.

Mechanism, now measured rather than modeled: B (1024x3072 f32 = 12.6 MB, f16
= 6.3 MB) FITS IN L2 on this GPU (~32 MB), so B's per-m-tile re-reads were
never DRAM traffic — the binding cost is the L1/threadgroup-issue path
(TG writes + simdgroup loads + the barriers ordering them), which halving
bytes does not relieve. Same mechanism as the 09-24 coalescing negative
("sector waste already absorbed by L2/MLP"), one level deeper.

Consequences: the f16-B BACKEND rung (f16 transposed weight cache + dispatch
+ G5 re-gate + the Issue-750-T3 per-family retention walk) is dead before
being built — days of lossy-surface work for a measured ±0%. With this, five
axes are refuted at kernel level (occupancy bk32/bn32, BK=48 barriers,
coalescing x2, f16-B) and the roofline gap (+63..+134%) is the staging-issue
path itself, which none of the tried geometries reaches. Narrow's shape is
the measured local optimum on this hardware/toolchain. Reopen triggers: a
Metal/toolchain change exposing direct-to-MMA staged layouts, or an L2-
oversized working set (n > ~4096 changes B's residency class — the packed
path's n grows with question count, worth re-probing there first).

## 2026-09-27 — Issue 021 CLOSED: the cuda CLS-row corruption — a chain-cache prefix-match
## alias, not a kernel race (the fused-GLU temp's slot vs the hidden that reused its address)

The harness banking77 cuda repeat check (`determinism_ok = false`, Bench 052 §4090
re-run) is fixed at the root: `chain_buf`/`chain_slot_for` now EVICT same-pointer
different-length entries on bind. Plus a second, independent defect the hunt
surfaced: `download_into`'s `memcpy_dtoh` is `cuMemcpyDtoHAsync` (cudarc 0.19) —
stream-ordered but ASYNC — so the pre-copy `synchronize` never guarded the host
read; a trailing sync now does.

The mechanism, isolated by four probes (all committed under `tests/`):

- `cuda_repeat_probe` (op level, banking77's real shapes ×200–2000 incl. the
  act-head GEMMs `[1,1028]×[256,1028]` / `[1,256]×[2,256]`): every kernel
  bit-stable — kernels exonerated.
- `cuda_packed_repeat_probe` (real english checkpoint, packed forward ×30):
  bit-stable — the encoder exonerated (and explains why the harness flag
  carried "picks still match metal": the marker gather never reads row 0).
- The harness bisect (flash=0 / ladder=0 / reg4=0 / head-defer=0): fires under
  EVERY posture — posture exonerations; `LAYA_HEAD_DEFER` mechanically cannot
  reach banking77 (1-question cases never take the packed path; the one clean
  run was luck — never trust a single clean cell).
- `cuda_agent_repeat_probe` (full `system_one`, 12 real cases, 30 rounds):
  RED at round 1 on 6/12 cases — only `act_probability` moves (saturating to
  1.0 on the bad read), probs/confidence bit-identical.

The `LAYA_DEBUG_ACT_ECHO` instrument (head.rs, env-gated) then split the
inputs: logits identical, feats identical, **the CLS row's raw-bit checksum
different**; and the `LAYA_CUDA_TRACE` download-candidate log named the
collision: the CLS prefix read matched TWO same-epoch slots at the hidden's
base pointer — the hidden itself (320512 floats) and a 1642624-float slot =
exactly `total·2·i_sz`, the fused GLU temp inside `matmul_w_glu`'s default
composition. The temp is allocated+dropped per layer; its DEVICE slot entry
survives in the epoch map (the map owns the Arc); the hidden Vec then
allocated at the freed address and bound a second entry; `download_into`'s
prefix match ties on epoch and `max_by_key` falls through to HashMap
iteration order — ~50/50 per call, both directions, stable within a process
for fixed key sets (why every earlier repeat probe was green). Metal is
clean because it OVERRIDES `matmul_w_glu`/`matmul_w_accum` (the T11 fold
rungs) — no fused temp, no address churn.

The fix (cuda.rs, both bind sites): `map.retain(|k,_| k.0 != ptr || k.1 ==
len)` on miss — a different-length slot at a recycled address is provably a
dead buffer (two LIVE host allocations cannot share an address), so eviction
is always sound. Acceptance: agent probe 30×12 GREEN (was 6/12 red at round
1); harness banking77 cuda `determinism_ok = true` ×4/4 (was firing every
run); accuracy unchanged (0.4220); `cuda_ops_smoke` 4/4; G5 cuda parity
GREEN; clippy clean at cuda + all-features; the M3 lib tests 41/41.

Rustc side-note (the box, not the code): two reproducible
STATUS_ACCESS_VIOLATION rustc crashes on the 4090 (the cubecl test closure's
katgpt-speculative/katgpt-forward, then riir-reflex lib) — both clear at
`-j 4`; the first blocked the per-op encoder probe (`forward_probe` rides
the cubecl gate), which is why the isolation went through the agent level.

Instruments kept, env-gated, zero cost when unset: `LAYA_DEBUG_ACT_ECHO`
(head.rs — CLS bits-sum + act_logits per call), the download-candidate trace
under the existing `LAYA_CUDA_TRACE`, and the four probe tests (the [[test]]
rows keep the green-zero rule honest).

Landed at `c64d0b1` (fix + probes + instruments; the issue file is removed with this record — the noise-reduction rule; the docs commit is `9de953c`).

Session: issue021-cuda-determinism

## 2026-09-27 — Issue 011 CLOSED: `row_logit_floor` model-bound G1 complete — needle@64K PASS at every arm

The gate katgpt-rs could not run for itself finished measuring and sat
unread for ~30 h: T3c (MiniCPM5-1B at 64K real dilution) completed
2026-09-26 03:16 +0700 after a 6.8 h shared-dense-prefix run (196 593
rows at 8.05 tok/s), and the issue still read "measuring". Harvested
2026-09-27 (this session), log `/tmp/ri011run/t6_minicpm64k.log`.

**Result: PASS at every arm, b4 included.** 3/3 passkey prompts
seq-exact, 0.00% top-1 flips over the 12 scored tokens at b8/b6/b6s0/b4;
m_Y preserved within 0.0007 with the same top head (L15H7); base ppl
1.0362, all Δppl inside noise. Three readings worth keeping:

- **The tv-budget width holds at 64K.** The floored fraction saturated
  at 8.4% (16K read 8.9%) — `ln(n/ε)` width growth (16.61 → 18.00 nats)
  compensates the 4× dilution exactly as `A ≤ n·e^{−w}` predicts. The
  width-sanity check is load-bearing: mean w = 18.00 = ln(65536/1e-3)
  confirms the run was genuinely 64K.
- **The closed-form envelope is vacuous at 4 bits / 64K** (mean env TV
  1.31 > 1), so b4's 64K row is measured-retrieval evidence, not a
  bound check. Teeth survive at 6 bits (0.169) and 8 (0.037).
- **b4's 16K m_Y "switch" was a tie broken, not a perturbation** — at
  64K the top head is already L15H7 and b4 holds it.

⚠ n = 12 over 3 prompts (same as T3b): Δppl signs are noise (three of
four arms read negative — sign cancellation, T2's shape). The row
proves retrieval did not break at 64K; it cannot rank arms. T4's
exemption verdict rests on T2's 4096-token table.

**Final gate standing:** T2 PASS 8/6-bit (gemma-2, 4096 tok: b8 +0.033%
ppl / 0.17% flips; b6 +0.066% / 0.90%) · T3 PASS 64K (T3a proxy + T3b
16K + T3c 64K, 0 flips everywhere) · T4 sink exemption load-bearing
(s0: floored 1.58×, |ΔNLL| 1.35×, flips 1.49×, aggregate ppl looks
*better* — the lossy-surface failure shape on real rows). 6-bit is the
admissibility floor.

**Promotion handoff:** issue 011's last condition ("promotion waits on
T2 + T3 passing here") is met; the promotion lane itself is owned by
katgpt-rs and filed there as **Issue 903** (per-family retention walk
per the lossy-surface rule + full-forward G2 + the 8-vs-6-bit width
decision). The primitive stays opt-in; `ForwardContext.logit_floor:
None` is the plain path, bit-identical.

Bench record: `.benchmarks/003_row_logit_floor_ppl_needle.md` (T3c
section + Verdict added this session). The issue file is removed with
this record — the noise-reduction rule.

Session: issue011-t3c-harvest

## 2026-09-28 — Issue 017 (gpu_transpose dead module) + Issue 023 (fence F2 self-alias FP): both closed

**Issue 017 / owner-gate D7 — deleted.** `gpu_transpose.rs` (286 L) +
`src/kernels/transpose.wgsl` (57 L) removed; zero callers anywhere
(grep incl. riir-ai's `riir_gpu::gpu_transpose` re-export consumers —
none exist). `transpose_cubecl.rs` (Issue 572) covers the reachable use
with owned `cubecl::server::Handle` in/out. lib.rs mod line removed;
`transpose_cubecl.rs` relationship note rewritten as provenance;
BOUNDARY.md riir-infer-gpu line repointed at `transpose_cubecl.rs` in
the same commit (the D7 recipe). riir-ai's dead re-export dropped in
the paired commit there (`6c6bf169b`). Clippy clean at
cubecl_runtime + no-default postures; bench_663 target compiles.

**Issue 023 — fence green.** The third file renamed
(`cubecl_encoder_probe.rs`, `riir_weights` → `lane_weights`, the
capture driver's convention; the two siblings landed earlier the same
day after their fmt edits landed). `fence_gate.py`:
`✓ PASSED — 0 undefended, 0 pinned`. Compile-checked at the file's
required-features (`laya-riir-cubecl`).

**Session note (the staged-set hazard, twice in one session):** the
gpu_transpose deletions were swept by a sibling's index-commit into
`3bed93f` 27s before this session's own commit (repaired by the
companion commit `734ef14` — 3bed93f alone does not compile); in
riir-ai the sibling's staged ambient files were swept into this
session's first commit (`8330e196b`, repaired by reset + pathspec
recommit `6c6bf169b`). Standing remedy: in shared checkouts, commit
via pathspec (`git commit -- <paths>`), never the bare index.

Issue files removed with this record — the noise-reduction rule.

Session: owner-gate-pickup-d7-fence

## 2026-09-28 — Issue 025 (owner-gate pickup) closed: D7/D8 executed, D9/D10 recorded

D7: gpu_transpose deleted + BOUNDARY.md repointed (record above).
D8: the audio-lane BOUNDARY widening landed — an AUDIO Owns row
(loader/serving-scoped, published CoreML bundles on `laya-riir-ane`)
+ the `objc2-core-ml` allowlist row's condition column names the
reuse (no new dep; the fence did not move). 015's T1–T5 remain the
PoC's own tasks.
D9: research 327–332 routing stays deferred with Plan 611 T7/S8
(tracked in 1004).
D10: the S6b training-families disposition is ratified into 1003's
status — riir-gpu-side by design (public repo vs training surface);
closed absent a real consumer pull.

Issue 025 removed with this record — the noise-reduction rule.

Session: owner-gate-pickup-d7-d8-d10

## 2026-09-29 — Issue 022 Phase 3 complete (audition + zero-training surrogate); Issue 024 closed measured-N/A both mechanisms

**022 Phase 3 (T3.1–T3.3) landed** at `27c9f86`: `src/twt/audition.rs`
(pure merge operators mean + LaCo RDSC — clone fixed-points gate-tested —
the per-channel α/β branch-correction fit + the through-origin
`fit_alpha_only`, and the BLAKE3-pinned selection with its argmin
cross-check) + `examples/twt_laya_audition.rs` (the driver) + the gated
`Encoder::layer_weights` borrow accessor. The apply path is a verbatim
mirror of the forward's per-layer body, proven BIT-IDENTICAL against the
parent forward EVERY run (the parity arm: layers 5 sliding + 6 full, both
checkpoints). 15-gate battery (`twt_audition_gates`); the whole
`twt_profile` surface clippy-clean; pre-existing gated-posture warnings in
the Phase-1/2 files (unused import, OR-pattern ranges, unused mut)
repaired in the same landing.

Measured both checkpoints × the full pre-registered grid (corpus:
typed_decisions states + banking77 texts, 500 prompts → 1484 rows,
stride-2 fit/held-out pinned in the driver doc before the first run):
**every block's winner is a member passthrough — the mean/RDSC merges
never won a single block.** Structural: laya's G-S-S pattern caps
homogeneous blocks at k=2 (RDSC(k=2) = the last member exactly); the
merges' room is the Bonsai GDN lane (48-run runs), not laya. The Phase-2
DP carries no type constraint, so real blocks ARE type-mixed — a merge
across RoPE thetas/mask types is incoherent (the tier-(i) rule), and
mixed blocks ran passthrough-only pools, recorded per block. The α/β
branch correction recovers 13–84% of held-out boundary mapping error on
small blocks (k ≤ 6) and 6–24% on whole-model blocks (ε=1.2) — the
recorded No-GD boundary datum for the Phase-5 track split. Two measured
refinements recorded in-issue: the +α decomposition rung must be its OWN
through-origin fit (the joint α with β=0 mis-prices it — +12.7% on block
14..18 typed vs improvement from its own fit), and the +5% pathology
guard skips zero-raw singleton blocks (a relative guard refuses an exact
block on its own f64 rounding). Artifacts:
`.raw/twt/{typed,english}_audition.json` (gitignored, BLAKE3-pinned).
Phase 4 (ternarize arms + collapsed-GGUF writer) is next.

**Issue 024 CLOSED — both NaiveRT mechanisms measured N/A** (removed
with this record; the full verdict tables live in git history):
024a (norm-share upper bound): `LAYA_METAL_PROFILE=1` × 2 checkpoints,
preflight PASSED (PROVENANCE: power=AC, powermode=2, load 4.25, canary
118.3 µs) — RMSNorm (`ln_rows_wide`) is 2.1–2.4% of its stage ≈
1.8–1.9% of end-to-end GPU, an order of magnitude under the ±6% floor
(the lane's own run-to-run spread measured ~1% — the borrowed floor was
conservative); a perfect weight-fetch overlap cannot save more than the
norm's own share → build nothing. 024b (small-m histogram, DERIVED):
encoder packed sgemm m ∈ [773, 2716]; per-question head ops m ∈ [124,
576] over 80 real questions — min m = 124 ≈ 4× the narrow tile M=32; the
lane never runs m < tile M in anger, so the heads-in-n remap has no
target shape and no kernel_opt rule was filed (a rule needs a measured
win on our shapes). The generalizing lesson recorded as a distill note,
not a rule: NaiveRT's intra-kernel wins are tied to TP8-decode shapes
(8 heads/rank; TMA weight streaming) a single-GPU MSL classification
encoder never produces — shape-transfer requires the shape, not the
paper.

Session: owner-gate-pickup-022-p3-024-na

## 2026-09-29 — Issue 022 Phase 4 LANDED (re-ternarization arms + κ budget + collapsed-GGUF writer)

**T4.1–T4.3 landed** at `baeb686` behind the new `twt_collapse` feature
(`twt_profile` + `deltanet_ternary_inference`). `src/twt/ternarize.rs`:
the three deterministic re-ternarization arms + the PRE-REGISTERED
budget (κ = 2.0, τ_code = 1 — the file's git history is the
pre-registration). Two pinned conventions beyond the issue text: arm B
is the integer CODE vote (scale-free — the issue's literal scale-weighted
`sign(Σwᵢ)` was rejected at pre-registration: wildly-different member
scales let one big-scale member dominate for scale reasons, not
agreement reasons), and arm C divides by the f16-ROUNDED scale so the
codes are self-consistent with the emitted wire. Arm C on ternary input
is BIT-EXACT (gate). `src/twt/collapse_writer.rs`: GGUF v3 collapsed
emission streamed from the parent mmap — member passthroughs are
BYTE-COPIES renamed to the new index, merged blocks carry per-suffix
payloads with completeness enforced against the block's first member
(a missing suffix refuses loud), the parent's metadata mirrors IN FILE
ORDER (the reader gained `metadata_order` + `GgmlType::id()`, Q2_0
emitting the fork-tip relabel 142), the `{arch}.block_count` override is
REQUIRED to equal the reduced count, and the `twt.*` provenance keys
(`block_table`, `arm_codes` + legend, `parent_weights_blake3`) are
standard metadata the train-side probe reads unchanged.

Two real defects the gate batteries caught at landing, both fixed in
the same commit: the Q2_0 wire pack (`pack_ternary_group_to_q2_0`, the
repack's new inverse) initially skipped zero weights — a skipped nibble
IS code 0, which decodes as −1; the bit-exact round-trip gate held it
(zeros must emit code 1). And the writer's offset plan desynced from
its own write loop on the first misaligned tensor (the debug assert
fired before the alignment pad) — caught by the synthetic-parent
battery.

**First real Phase-4 measurement** (`examples/twt_ternarize_probe`, the
league model `Ternary-Bonsai-2-27B-PQ2_0.gguf`, GDN triples
[0,3)/[32,35)/[60,63), `ffn_down` 5120×17408, deterministic LCG
inputs): **damage_A(f16) ≈ 3.7–3.9e-8; damage_B(majority) ≈ 20 — arm B
is DESTROYED on cross-scale merges** (the supported-amax scale
overshoots the typical |f̄| ~3× and the vote destroys the magnitude
structure; arm B is dead for merged blocks — a same-scale-only arm at
best); **damage_C(source-quant) ≈ 0.31** — the 5→3 level reduction's
price, so the κ=2 budget admits arm C only where the audition's own
surrogate error ≥ ~0.31; on strong merges **arm A (dense f16, one GEMM
per block) is the only budget-viable arm** — the issue's own arm-A
framing. Baseline clarification pinned the hard way in the module doc:
the T4.2 denominator is the SURROGATE's end-to-end error vs the parent
(the audition's E_dense), never arm A's f16 rounding floor — κ·1e-8
would be unfailingly tight and every arm would die mechanically.

Gates: `twt_ternarize_gates` 15, `twt_ternarize_g4` 1 (alloc-free
rel-err loop, own binary per the counting-allocator isolation rule),
`twt_collapse_writer_gates` 4 (re-open round-trip, byte-identity,
refusals, determinism over a synthetic ternary parent); lib 291 green
at the feature; clippy clean at default, twt_collapse, and
all-targets-at-the-feature. The train-side probe verify waits for a
REAL collapsed file — which waits on the Bonsai audition (the apply
path needs GDN cache snapshot/restore threading the laya mirror lacks;
the Phase-5 prerequisite and the lane's next work).

**Second real measurement (same session, `--compare-members`, the
operator-level surrogate-pool pre-read):** on the same three GDN
triples, the mean merge f̄ sits **0.65–0.69 rel-op-dist from EVERY
member**, and the members sit **1.9–2.07 from each other** —
consecutive GDN layers inside an S-close triple are near-ORTHOGONAL as
operators (rel-dist ≈ 2 ⇔ ‖A−B‖² ≈ 2‖B‖², no shared structure), and
the merge is a genuinely different operator from all of them. The
Phase-1 S-matrix's STATE similarity does not transfer to OPERATOR
similarity — the laya audition's S-vs-function gap, now visible
operator-level without any forward. Pre-read verdict: the merge arms
will likely lose the Bonsai audition too, so **the collapse lane's
Bonsai value is (a) passthrough/pruning collapse and (b) arm-A dense
blocks — reorder Phase 5 to a passthrough-collapsed checkpoint first**
(the writer is already its deliverable), with the expensive GDN
apply-path audition gated behind that GOAT result. Caveat kept honest:
rel-op-dist is not mapping error; the audition remains the decision
instrument — this pre-read only reorders what to build first.

Session: riir-infer-022-phase4-arms-writer

## 2026-09-28 — Issue 015 audio PoC DEFERRED (owner: until M5 Ultra) + 023's missed hunk landed

Owner directive same evening: defer the audio lane until an M5 Ultra is
available — the PoC targets ANE latency/working-set posture and M5-class
silicon is the intended measurement box, so a pre-M5 measurement would
not be the record that matters. Issue 015 stays OPEN with T1–T6
deferred (`- [-]`) and a turnkey recon recorded IN the issue (exact
bundle `silero-vad-unified-256ms-v6.2.1.mlmodelc`, the full wire
contract from FluidAudio's `VadManager` @ `20d4f0bd`, and the T2 shape
— the generic `ane.rs` load path takes any bundle URL; the audio lane
bypasses the laya digest/manifest coupling and should REPORT the plan
verdict rather than reuse the strict laya gate). T0's boundary rows
stay landed (D8) — the widening is timing-independent.

Follow-through: `e8e17b8` landed the `cubecl_encoder_probe.rs` alias
rename that Issue 023's closing record (`61ebf9c`, HISTORY row 1047)
claimed landed but the sweep repair left in the worktree — the
record-vs-tree divergence class again; caught because HEAD's test file
still carried the pre-rename spelling.

Session: owner-gate-pickup-audio-defer

## 2026-09-29 — Issue 022 T5.0 LANDED: the passthrough-collapsed checkpoint + the first real GOAT numbers (coarse grid FAIL; fine end measured)

**The loader prerequisite (T5.0a `318b9fa` + T5.0b `ceb96c3`).** The stock
qwen35 loader types layers by `full_attention_interval` INDEX arithmetic —
a collapsed+renumbered file would type every winner WRONG. Fixed at both
seams, fail-closed: the writer REFUSES a qwen35 collapse without an
explicit `twt.layer_types` array (U8 `DeltaNetLayerType` discriminants —
no second vocabulary; legend in-file) and refuses a stale
`qwen35.nextn_predict_layers` (MTP blocks are not main-stack layers; the
override-to-0 heals); the loader (`qwen35_deltanet_config_from_gguf_metadata`,
now `pub`) REPLACES the derived types when the key is present, refusing
loud on length/vocabulary mismatch — legacy files keep the interval
derivation. The second defect surfaced while wiring the real lane: the
league checkpoint IS Hadamard-folded, so `prism.hadamard.weight_names`
must renumber with the tensors it names (stale names refuse at load;
renamed names at the WRONG block would be worse). The writer transforms
the key in place: kept members rename, dropped members' entries drop, a
MERGED block keeps its first member's entries (the fold is linear — a
mean of folded weights IS the folded mean), `sign_widths`/`sign_values`
are width-keyed and survive untouched. Interlock: every SURVIVING folded
name must pass `is_known_folded_name` against the collapsed types — a
GDN-typed attention winner refuses (its rotation would silently skip).

**The emit lane (T5.0c `9905c7c`).** `examples/twt_collapse_emit`: the
profile artifact's real 64×64 S matrix (`.raw/twt/bonsai_ultrachat_profile.json`,
51 ultrachat sequences, corpus BLAKE3-pinned) → the crate's own
`minmax_partition` at a chosen ε → per-block winner = minimax medoid
member → writer emission, all-member passthrough. Extra provenance keys:
`twt.partition_eps`, `twt.partition_worst`, `twt.profile_corpus_blake3`;
`twt.parent_weights_blake3` = BLAKE3 over the parent's tensor payloads in
file order (the key name says exactly what it commits).

**The artifacts + wire verification.** SEVEN real collapsed checkpoints of
`Ternary-Bonsai-2-27B-PQ2_0.gguf` in /tmp (ε ∈ {0.015, 0.02, 0.03} fine +
{0.05, 0.1, 0.2, 0.3} coarse). The ε=0.05 file hand-verified on the wire:
`twt.layer_types` = the winner table exactly (18 DeltaNet + 7 Attention),
`prism.hadamard.weight_names` = 158 entries exactly (18×6 GDN + 7×7
attention + `output.weight`), block_count=25 — and riir-train's
`plan402_gguf_probe.py` reads every `twt.*` key cross-repo AS-IS
(T4.3's owed verify DONE, no train-side edit).

**The GOAT (`twt_goat_agreement` bin).** Teacher-forced greedy argmax
agreement parent-vs-collapsed over frozen corpus tokens (the T5.1
pre-registered ≥0.9 ABSOLUTE bar); the parent arm runs once and is
cacheable (`--cache`, params-keyed, loud on replay) — one ~55-min pass
serves every future sweep point.

**The coarse-sweep negative (4096 positions, 8 × 512-token chunks, M3
CPU AC): ALL FOUR pre-registered grid points FAIL.** ε=0.05 (25 blocks,
39% depth): agreement 0.1945; ε=0.1 (14, 22%): 0.0051; ε=0.2 (8,
12.5%): 0.0029 (collapsed hit rate 0.0022); ε=0.3 (5, 7.8%): 0.0000.
Parent hit rate 0.7478 — the harness measures a real signal. Reading:
the grid's finest point already cuts 61% of depth, the known-fatal
regime in the layer-pruning literature — cosine redundancy (the S
matrix) is NOT a sufficient license for depth cuts on this model class
(activation-space redundancy ≠ functional redundancy). This is the
honest negative that JUSTIFIES the merge/audition question per T5.0's
own gate. The fine end (ε=0.015 → 11% cut, 0.02 → 23%, 0.03 → 45%) is
the bar-or-bounded question, measured in-flight (log:
`.raw/twt/goat_agreement_sweep.log`; the fine results + verdict land in
the issue row).

**FINDING (kill-rule provenance):** the bonsai profile artifact's
recorded `SURVIVES` is inconsistent with the current kill rule over its
own stored S — re-derived: `minmax_partition(S, 1.2)` = ONE block
(global worst 0.801 ≤ 1.2) against bar = ceil(0.8 × 32) = 26 →
`KillBlockCount`. The laya SURVIVES verdicts (the Phase-2 record) are
unaffected; the bonsai capture's `forced` was most plausibly degenerate
(forced=1 → Survives trivially). Re-derive before any ε-sweep Pareto
claim cites the artifact (T5.5).

Session: riir-infer-022-phase5-t50

## 2026-09-29 — Issue 022 T5.0 COMPLETE: the agreement cliff mapped — 4.7% depth cut PASSES the GOAT bar, quality parity holds to 11%

Bench record: `.benchmarks/022_t5_collapsed_goat_agreement.md` (full
table + box state). The complete ε→agreement curve at 4,088 frozen
positions (8 × 512-token teacher-forced chunks): **ε=0.01 (61/64
blocks, 4.7% cut) agreement 0.9486 — the FIRST PASS of the
pre-registered ≥0.9 absolute bar**; ε=0.015 (10.9% cut) 0.8955 (0.0045
short); ε=0.02 (23.4%) 0.5247; ε=0.03 (45.3%) 0.0301; the four
pre-registered grid points (0.05/0.1/0.2/0.3 → 61-92% cuts) all FAIL
(0.19 → 0.00). Parent hit rate 0.7478.

**The two-metrics finding (both recorded):** the collapsed model's own
top-1 hit rate holds PARITY with the parent to 11% cut (0.7505/0.7495
vs 0.7478 — the 11%-cut model is marginally BETTER at next-token
prediction than its parent) while agreeing with the parent on only
89.6% of argmaxes — trajectory divergence (chaotic stream sensitivity)
overstates functional damage by one full grid notch. An
agreement-controlled claim needs the hit-rate column beside it (T5.2's
separation, one notch finer). Past 11% the hit rate falls off the same
cliff (0.5076 at 23%, 0.024 at 45%).

Consequences recorded in the issue: the merge/audition question is
justified ONLY for quality at REAL depth cuts; riir-train 423's
distillation owns the regime beyond ~11%. The parent arm is cached
(params-keyed, loud replay) — every future sweep point pays only its
collapsed arm.

Kill-rule provenance finding (from the same session): the bonsai
profile artifact's recorded `SURVIVES` is inconsistent with the current
kill rule over its own stored S (re-derived KillBlockCount; the laya
SURVIVES verdicts unaffected) — re-derive before any T5.5 Pareto claim
cites it.

Session: riir-infer-022-phase5-t50

## 2026-09-29 — Issue 022 T5.1 lane (1) COMPLETE: gemma-2 f16 control — clean negative at every real depth cut; the fine-end bracket is structurally empty

Bench record: `.benchmarks/015_t51_gemma2_control_goat.md` (full table,
box state, pre-registration). Instrument landed at `0e0436c`
(`PostLayerHook` capture seam on `forward_gemma2_f16_tapped` — NoHook
monomorphizes to the unchanged forward; profile driver
`examples/twt_gemma2_profile` behind feature `twt_gemma2`; gemma2 arch
arms in `twt_collapse_emit` + `twt_goat_agreement`; the pre-registered
protocol in the issue in the SAME commit), verdict at `55fa831`.

**The verdict: ε=0.05 identity row 1.0000 (4092/4092 — the pipeline is
bit-faithful), then EVERY real depth cut FAILS** — 65.4% depth →
0.2571 agreement, 38.5% → 0.0831, 23.1% → 0.0132, down to 0.0015 at
3.8% depth with hit rates dying to 0.0002. Parent hit rate 0.4746
(healthy signal on a no-BOS raw-text stream). **The fine-end bracket
is structurally EMPTY**: at ε≤0.03 the partition is m=26 with
worst-block 0.0000 — ZERO merges, no S rung between identity and the
65.4% cut (unlike bonsai's 4.7% pass). The four bracket arms were
emitted and their agreements not run — each is a byte-identical
re-emit of the ε=0.05 identity row, which already measures that point.

**The zero-training passthrough question is closed on TWO
architectures** (hybrid ternary bonsai + dense f16 gemma-2, same
meter/DP/writer): cosine redundancy ≠ functional redundancy, and
gemma-2's phase structure (m=10 at ε=0.2) transfers to merge safety no
better than bonsai's did. The rescue, if one exists, lives in the
apply-path (auditioned merges / distillation, riir-train 423) — the
control lane hands riir-train 423 its baseline artifact + negative
control. Kill-rule footnote carried in the bench: the single-operator
stack makes `KillBlockCount` fire on a technicality (bar trivially 1);
the depth-reduction license on this lane is carried by the S sweep,
adjudicated by the bench — the T5.5 Pareto report cites the sweep,
never the kill verdict.

En-route: a live Bench-number collision with the M3 sibling's
in-flight typed-partition/GDN-audition lane (their Bench 014,
committed + referenced first) — my bench renumbered 014→015 per the
collision rule (theirs kept it), highwater 15, ff-merged onto theirs;
the merged tree verified compile-clean. Collapsed GGUFs (~21 GB)
deleted post-measurement — regenerable from the kept profile artifact
(`.raw/twt/gemma2_profile.json`, corpus BLAKE3'd) + the cached parent
arm in ~20–90 s per point.

Session: riir-infer-022-t51-gemma2-control

## 2026-09-29 — two Windows batch traps hit by the T2/T3/T5.1 schtask runners (recorded; the deleted wait-loop script's durable note)

Both diagnosed live while landing the Issue-013 T2/T3 chain and the
T5.1 GOAT pipeline; recorded because the deleted script's header note
died with it:

1. **`Start-Process`-launched `.cmd` inherits the SPAWNING MSYS session's
PATH.** Inside such a script, `tasklist | find` resolves `find` to
`/usr/bin/find` (MSYS), which does not read stdin the way cmd's
`find.exe` does — a `tasklist /FI ... | find /I "name"` wait-loop then
wedges forever (the wait loop never observes the process exit). The
T2/T3 schtask launchers are IMMUNE: Task Scheduler runs with a clean
system environment, so `find` resolves to `C:\Windows\System32\find.exe`.
   *Rule: never hand-launch a wait-loop `.cmd` via `Start-Process` from
an MSYS shell; use `schtasks /Run`, or fully-qualify
`%SystemRoot%\System32\find.exe` inside the script.*
2. **A hand-rolled no-wait batch rewrite died silently while identical
constructs passed in isolation.** Root cause never fully isolated
(multiple rewrites, all plausible, all dead); bypassed by invoking the
exes directly from the agent shell. *Rule: for chained measurement
runs, prefer `schtasks` wrappers (the Issue-012 recipe — clean env,
survives agent teardown) or PowerShell; never a hand-rolled wait-loop
batch whose failure mode is silence.*
3. **A literal `)` in an echo INSIDE a parenthesized block silently turns
the block's follow-up `exit /b 1` unconditional** (found 2026-09-29
23:15, the T3 launcher `run_kv_reconstruct_gate.cmd`). cmd's block
parser treats the echo text's `)` as the block terminator: the guard
echo `echo ... (%DATE% %TIME%) === >> log` inside the wait-loop's
`if %ERRORLEVEL% EQU 0 ( ... )` closed the block early, so the
`exit /b 1` after it executed the MOMENT T2 was found running — the
chained T3 task fired at 22:30, wrote its header, and died rc=1 in
seconds, three times (22:30 scheduled, 23:15 manual re-fire, clean-env
repro), silently. Found by bisection only because Task Scheduler
history is disabled on this box and the failure mode was silence; the
`noguard` variant (nested block deleted) was the flip. SAME latent bug
in `run_twt_gemma2_goat.cmd` (3 in-block echoes) — fixed in the same
pass. *Rule: inside a `( ... )` block, an echo line must carry NO
literal `)` — reword to `at %DATE% %TIME%` or escape `^)`.*

Session: riir-infer-022-t51-gemma2-control

## 2026-09-29 — Research 004: Disaggregated Quantization distilled (arXiv:26.26333) — three issues filed, riir-train Plan 430 routed

Full-read distill of arXiv:2609.26333 (DQ/QADD, NVIDIA+ISTA), deepening the riir-clippy arxiv-walk-217 row. Track (a) Gain: Issues 026 (phase-isolated quant sensitivity bench — the falsifiable damage-ratio instrument), 027 (T0 encoder-only asymmetric Q2_0 — code 3 already decodes as +2d but is encoder-unreachable and bridge-rejected; then offline Lloyd-Max grids + non-uniform Q2_0A which must beat T0; dense-GGUF gain site only), 028 (dual-PTQ resident disaggregated serving — our hardware inverts the paper's ODP premise; the PTQ-vs-trained recovery measurement is the science). Track (c) Gain → riir-train Plan 430 (QADD prefiller pre-registration; secondary by serving-envelope fit; 27B convergence explicitly not priceable). Track (b) Pass. Signal-diffs on record: the GDN escape set (Issue 980 `gate_projections()` + issue879 f32 recipe) is corroborated from the training side; `dl_qat.rs` verified to carry no teacher/phase-mask (grep); landscape prior art named (OverFill 2508.08446, Decode-Branch 2608.12385; no scooper). League: no upstream absorption, no re-arm trigger; ODP fork honesty ≈1.3× released vs 1.78× claimed, Blackwell-only. Verdict gate: claude ping-pong AGREE round 2 (session `c68b3113-bb7f-49d1-853b-ac6e215e46be`; round 1 REVISE caught the reserved-vs-decodable misread + the two-level-scale conflation — both fixed before commit). Session: riir-infer-004-dq-distill, 2026-09-29T10:41Z (unix 1790678474)

## 2026-10-01 — Plan 614 LANDED: the DQ phase matrix ran to EXIT0; every axis INADMISSIBLE at the frozen corpora; the instrument (and its defect chain) is the deliverable

The Issue-026 instrument (`dq_phase_matrix`, Plan 614) completed its 4090 run — v2, `23bbff5`+`3cf9ae9`, EXIT0 12:25 ICT after ~4h40m (log `F:/wt/dq614-matrix2.log`, report `F:/wt/dq614-matrix2/dq_phase_matrix.md`, record `.benchmarks/023_dq_phase_matrix.md`, `.highwater` 15→23). Verdict per the frozen D6 table: **all four axes INADMISSIBLE** (base arith 0.9583 > 0.95 — the a07fffd `reset_state` fix lifted the model's genuine rate from the leak-contaminated 12.5%; NIAH pooled 0.9896 > 0.95) — Issue 026 T3's directional assertions receive NO GATE. The report-only tables carry the signal (a2 prefill-only collapses NIAH to 0.25 pooled vs base 0.99 while decode-only holds 0.93; a2 decode-only halves arith; A4 near-clean — the phenomenon is a 2-bit-tier floor phenomenon); the standing promotion rule (any future weight format publishes its per-phase R before default promotion) is ATTACHED to the instrument. Gates all green: G-i1, G-i2 counts (0 mismatches) + the PHASE-MATCHED positive control, G-i4 byte-stability (incl. decode-phase FNVs), dec_a8 control |Δarith|=0. Determinism double-sourced: the v1 run reproduced v2's every arith item + NIAH cell byte-identically before dying at the final gate.

**The defect chain the lane paid for (all fixed at source):** `a07fffd` the GDN recurrent state leaked across ALL items (M3 session — accumulation, not model behavior); `23bbff5` the G-i2 positive control was PHASE-BLIND (a decode-only arm's prefill is clean by the D1 boundary — structurally unsatisfiable; the v1 run measured every cell then FATALed at the final gate; the control is phase-matched now); `3cf9ae9` the plan-T4 4090-clippy discharge (7 mechanicals; the corpus/bootstrap seed regroups are value-preserving leading-zero pads — the frozen corpus blake3 `6f3c6f02…` reproduces byte-exact). Open follow-up: the FATAL teardown path HANGS (RAM climb, GPU released, process stuck unwinding; the EXIT0 path exits cleanly) — the error-exit needs its own fix.

**The collision class, twice, and the twin-duplicate:** 00:43 an unguarded second chain launch truncated the first lane check's log (~1.5 h destroyed); 07:39 this session's v2 launch died silently (EXIT-1) under the twin session's concurrent worktree reset+rebuild — the shared-worktree-mutation variant. The detached stages carry the ALIVE guard now (`E:/git/_sync/dq614_{chain,matrix}.cmd`, machine-local, live-tested), and the durable lesson is recorded in bench 023: a run lane needs ONE owner at a time — the handoff summary's active-plan section must name the owning session. The twin-duplicate (two idle-loop sessions fixed the same control ~20 min apart; origin `23bbff5` canonical, `5977565` died in the reset per the Batch-169 precedent, its non-overlapping value re-landed as `3cf9ae9`) is the Issue-825 class resolving CORRECTLY: the duplicate was rebase-not-fight, and the merged tip is what ran. Issue 030 (the run handoff) closed + removed per the noise-reduction rule — its evidence trail lives in this row + bench 023. Issue 026 stays open on T4 (the KV-axis arm — deliberately outside the Plan-614 freeze). Session: riir-infer-030-dq-matrix-rerun (+ the twin idle-loop session)

## 2026-10-01 — Issue 027 CLOSED measured-negative + removed: the offline LUT grid lane (Plan 615) — weight-space wins were real, the 2-bit class is function-destroying on dense artifacts

**Verdict (Plan 615 T6, commit `2cb807d`):** T0's asymmetric encoder (`quantize_row_q2_0_asymmetric`, signed `d = ±amax/2` steering the largest-magnitude element onto code 3 = +2d) measured **+4.9 dB/family** over the symmetric reference on both dense artifacts, and Q2_0A's Lloyd-Max grids added **+0.5–1.2 dB/family** over T0 — but the model-level gate killed the whole class: MiniCPM5-1B ppl **6.8e5–7.0e6 vs base 49.8**, EVERY family (chat + docs), EVERY reachable granularity (per-128/64/32 — the T5 sweep's per-64/per-32 weight-space gains do not transfer). The lane's own pre-registered gate ("a per-family loss at matched bpw = recorded negative, lane closes") fires on the class, not one arm. The paper's 27B dense regime is UNMEASURED here (infeasible with the fakequant harness on both boxes) — the honest limit, recorded in the plan.

**What stays in-tree (tested, ungated tools — Issue 028 may cite them if a dense serving regime re-opens):** `src/quant/q2_0.rs` (`quantize_row_q2_0_asymmetric` + the grid-aware decode `dequantize_row_q2_0_grid`), the Lloyd-Max solver `src/quant/lut_grid.rs` (d²-weighted histogram — the objective must match the metric; unweighted pooling won the histogram and LOST 0.7 dB SNR — + EM from the T0 incumbent + BLAKE3 grid commitment, cross-box determinism proven gemma `b28b3661…` / MiniCPM `642e2d0b…`), bins `lut_grid_solve`/`lut_grid_ppl`. **T2 (the Q2_0A wire format) is MOOT** — no dense-arm operating point to serve, so no grid-aware decode kernel gets built or priced (T4). The **Bonsai ternary lane is untouched by construction** (trained-ternary weights, zero code-3 occurrences; the byte-identity control is now a test: `symmetric_reencode_of_ternary_blocks_is_byte_identical`). Master: `.research/004_DQ_Disaggregated_Quantization.md` (arXiv:2609.26333 §2.3 + A.3). File removed per the noise-reduction rule — full task narrative: `git log --follow -- .issues/027_lut_grid_optimization_lane.md`. Hygiene session: riir-clippy idle loop (2026-10-01).
