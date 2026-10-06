# HISTORY.md — riir-infer

Durable records for resolved questions and closed lanes (the noise-reduction
convention: the record lands here hash-pinned; open work lives in `.issues/`
and `.plans/`). Created 2026-09-23 at the first record. Headings are kept
verbatim — issue/plan/bench numbers + dates are read by the workspace
numbering + citation gates; each record below is one compact entry (full
narratives: `git log --follow -- .issues/<file>` or this file's git history).

## 2026-10-06 — Issue 028 T4 S2: the q4 weight arm — the container loads and runs the dual-format pair — (this commit)

`ProjWeights { Ternary(TernaryGroupWeights), Q4K(Vec<BlockQ4K>, rows, cols) }` at the 10 per-layer projection fields; `bitlinear` dispatches the Q4_K arm to fused `gemv_q4_k_row` (rayon across 16-row chunks, bit-stable), GPU hooks refuse q4 LOUD, loader `load_proj` storage-type dispatch. Real pair (PQ2_0 + Q4_K.pf) loads at production scale in 630 s / 702 s (~20.5 GB resident); lib 318 green at `deltanet_ternary_inference`, issue-028 battery 9/9 incl. always-on `dual_format_q4_prefill_pair_loads_and_runs`; the real-pair PREFILL smoke did not complete (30-min tool ceiling, plan 618 — the cudarc q4 GEMV is the scoped next item).

## 2026-10-06 — Issue 028 T4 S1: the Q4_K prefill pack exists — plus the science premise settled — `94a4e7c`

`Ternary-Bonsai-2-27B-Q4_K.pf.gguf`, 14.43 GB, blake3 `0a7a5605fadc37b482c0fddf6c28334244eea818b1716461bddb412f1a28309c`; the science premise: the bonsai is TRAINED TERNARY, so the PQ2_0 wire is exact up to f16 group scales — requant carries only ε(Q4_K) (`UnsupportedFourthState` is the structural gate). Geometry fix: mirror the parent's own metadata value verbatim (`geometry_fingerprint` is discriminant-tagged; the first run re-widened block count to U64 and would have REFUSED at T4 load). Verifies exit 0; the container could not LOAD until S2. Plan: `.plans/618_dq_t4_q4_prefill_pack.md`.

## 2026-10-06 — the GPU crate is clippy `-D` clean at every posture — `e993eff`

Pre-existing warnings gone; the rotation quantize pair (`quantize_rotate_q8`/`_permute_rotate_q8` + `*_warp` fields) is test-consumed only — allow gate `not(all(test, feature = "prefill_q8_act"))` (the first `not(any(test, …))` was wrong in exactly the firing posture); `DQ_FQ_CUDA_SRC` is dead only without `dq_phase_bench`. Clean at `--workspace --all-targets` default, `-p riir-infer-gpu --all-targets` at `ternary_gemv_cuda_raw` AND `--all-features`; lib tests 248 + 56 green unchanged.

## 2026-10-06 — Issue 035 P2.6 LANDED: real-vocabulary compile + the row-fold — the honest G2 regime — `ddcaeb0`

Trie-stepped subset construction in `fa_schema.rs` (256k vocab compiles the fn-call grammar in 60s; toy 0.13s, object 1.8s) + one vocab-wide exp row per position in `fa_posterior.rs` (fn-call draw 11.79s → 1.98s at L=256). Bench gained `--real-vocab <path.gguf>`; gemma-2-2b 256k on the 4090 box: toy-enum 0.93×, object-2p ~1.8×, fn-call-5p ~6.0× — the REAL regime table the P3 owner verdict reads. P3 stays owner-gated.

## 2026-10-06 — Issue 035 P2.5 LANDED: DFA minimization — the G2 ratio drops ~4.4× — `9e94684`+`672e2b0`

`src/fa_minimize.rs` (UNGATED, 7 tests): trim + block-granularity partition refinement applied ALWAYS inside `fa_schema::compile` (language-preserving by construction); empty-language schemas now fail at COMPILE time (`SchemaError::InvalidSchema`). Ratios: toy 0.50-0.64× (faster than unconstrained), object ~6.2-6.6×, fn-call ~96-111×; exactness pinned distributionally (2×20k seeds) + exhaustive language enumeration. P3 owner-gated; the GOAT verdict consumes the minimized table.

## 2026-10-06 — Issue 035 P2 LANDED: the schema front end + the honest overhead table — `1254264`+`5b5c462`

`src/fa_schema.rs` (UNGATED, 12 tests): JSON-schema subset (six types · properties/required · items · enum · anyOf) → Thompson NFA → on-demand subset graph → token `Automaton`; three grammar-soundness laws pinned (fresh ws wrappers, per-(mask,member) member chains, acceptance = walk ∧ final). Bench (`examples/fa_schema_overhead_bench.rs`, CPU best-of-5, PROVENANCE line): toy 1.9-2.2×, object ~22×, fn-call ~440× — the CPU sequential lane meets single-digit-% ONLY on small automata; promote/demote owner-gated on this table.

## 2026-10-06 — Issue 022 T5.4 CLOSED: the equal-FLOP skip-class gate — the lane wins, honestly — `8bd5c2d`+this commit

Record `.benchmarks/022_t54_skip_arms.md` (instrument `src/bin/twt_skip_amputation.rs` at `a7b080c`): twt 0.8919 > hydra 0.8642 > random floor 0.8437 >> ShortGPT 0.0320 at the equal 3-layer cut — selection disagreement total, ShortGPT's BI greedy 81 points below random (a dense-model removal criterion does not transfer to the quantized ternary GDN/attention hybrid). katgpt-rs Research 594 §7 updated — the public-novelty caveat discharged; T1.4's real parent-vs-quantized ΔS pair run stays deferred (`src/twt/delta.rs` landed). Same wave: Issue 035 P0+P0.5 `src/fa_posterior.rs` — the FA-constrained exact joint sampler (Mosaic arXiv:2607.07026 distill; f32 sequential + O(log L) f64 segment-tree parallel, 1e-9-verified marginals, `tests/fa_g4_alloc.rs`).

## 2026-10-06 — Issue 035 P1 LANDED: the `fa_constraint` decode wiring — exact joint x→0 per denoising step — `304876f`+`f4ffa93`

P0's sampler wired into the `gemma2_d2f` denoising loop behind opt-in `fa_constraint` (implies `gemma2_d2f` ⇒ `dllm`): ONE exact joint draw per step; commit axes `RawTop1` / `ConstrainedMarginal` (marginal log rides `SamplerFeatures::marginal_log`, the 7-param `ConstrainedSampler` — the 6-slot `to_array` untouched); block carry-over (`sample_joint_from`/`build_tree_from`/`walk_from` + `ParallelTree.start`); mask-free output guarantee (`WalkBrokeContract` guard); `FaDecodeError` fails loud at decode entry; separate loop `d2f_decode_gemma2_constrained`, feature-off byte-identical. 6 CPU + 4 GPU tests; clippy `-D` clean; P2 next, P3 GOAT owner-gated.

## 2026-10-01 — riir-ai Issue 1004 R2 LANDED: the fused GDN prework (conv1d + SiLU + q/k L2-norm + head expansion, ONE dispatch) — `929d31e`

Feature `deltanet_prework_fused` (toggle `set_prefill_prework_fused`): 2 dispatches/layer vs the shipping chain's ~69; stage-isolated 7.29× @2048 / 9.13× @4096; G1 bit-identical (logits FNV `0ec2396fd4627f29` @2048 / `f35280cb95f5d306` @4096). e2e medians inside the round spread → NOT promoted (the R1 standard), opt-in stands; record `.benchmarks/024_deltanet_prework_fused_ab.md`; carry-update dispatch extracted to `launch_conv1d_carry_update` (DRY); the league re-pin bout stays the league loop's job (riir-ai Issue 1004's own law).

## 2026-10-01 — Issue 026 CLOSED (hygiene): the DQ phase-matrix instrument landed and ran — every axis INADMISSIBLE at the frozen corpora

Plan 614's instrument (`dq_fakequant` + the `dq_phase_matrix` runner; record `.benchmarks/023_dq_phase_matrix.md`) ran to EXIT0: base arith 0.9583, NIAH pooled 0.9896 — both OUTSIDE the pre-registered 0.25–0.95 window, so T3's damage-ratio assertions receive NO GATE; T5's promotion rule (any future weight format publishes its per-phase R before default promotion) rides the INSTRUMENT. T4 split to `.issues/033_kv_axis_extension_arm.md`. `.issues/.highwater_local` repaired 031→033 (Issue 032's bump was missed — it landed `5d0592f`, file removed `4f68e5d`; the katgpt-rs `.issues/121` recycling-bug class).

## 2026-10-01 — Issue 032 DONE: Metal CubeCL prefill computes Hadamard-folded (Bonsai-2) models (`5d0592f`)

Batched twins of all 8 decode rotation sites in the CubeCL prefill body + `GemvBatchedCubeCL::launch_token_grid`; G1 vs folded decode-eager top-1 equal, worst rel err ≤ 1.5e-5; pre-rotation Q2_0 pin `99a0733c45a0e663` @2048 exact; first Bonsai-2 M3 prefill 6.44× TTFT (100.9 tok/s @2048); test `crates/riir-infer-gpu/tests/g1_032_folded_prefill_metal.rs`. ⛔ Correction: production 4096 chunks CRASHED (>65535 workgroups on Metal's x-axis cap) — fixed `5f075ff` + `7e99f34`, G1 re-run green at P=4096 (`I032_P=4096`): a G1 that stops below the production chunk size certifies a size nobody ships.

## 2026-09-30 — Issue 054 Part 2 (reflex) REFUTED for this lane: the prefix-state handoff lead — laya is ModernBERT (bidirectional), not GDN; the coupling gate landed

The premise named the wrong architecture: laya checkpoints are ModernBERT-large / mmBERT-base (no causal mask; the state also sits after the per-question head span — different RoPE offset per question). Gate `crates/riir-infer-laya/tests/prefix_state_coupling.rs` (`[[test]]` row, feature `laya-riir`) is TWO-SIDED at 1e-4 — the shared span measured 5.1e2 (synthetic arm 2.7e0, determinism control bit-identical). The packed per-question pass (reflex issue 020 T5) stays the exact floor; the lead's shape stays valid only for causal serving models. Full verdict: reflex `.issues/054_openjev_lane_and_prefix_state_handoff.md` Part 2.

## 2026-09-30 — Issue 013 T2 EXECUTED: the K=V+ λ ladder on gemma-2 — G-A PASS vs a catastrophic baseline; K=V+ not viable standalone; G-D overfit; NIAH build crash (029)

Record `.benchmarks/012_t2_kv_ladder_gate.md` (+ `012_t2_run.log`, table `012_kv_table_residual.bin` sha `3a0f5333d6bafd63…`, reused by T3): f16 6.0907 (== T1, cross-run determinism PASS) vs k-0.00 catastrophic (tax +30,948,056%); k-1.00 λ*=1 G-A PASS (3344× recovery) but still ×92.6 off f16 → K=V+ NOT viable standalone; G-C PASS; G-D OVERFIT (uniform λ stands); G-E missing (NIAH crashed at trial 0 — builder undershoot → `.issues/029_niah_builder_undershoot_and_phase_d_report.md`).

## 2026-09-30 — Issue 013 T3 EXECUTED: P3 V-cache reconstruction — G1+G3 PASS at every λ; the 50.0% bytes/token law holds; G2 +7.9–8.8% recorded (the promotable P-half)

Chained run rc 0 (waited 399 min for T2, BLAKE3-verified its table artifact). Record `.benchmarks/013_t3_p3_reconstruct_gate.md` (+ `013_t3_reconstruct_report.md` written incrementally, `013_t3_run.log`): G3 max |Δlogit| 1.254e-4 (the rotation-rounding class); G1 PASS 0 flips in 2046 scored positions; bytes/token 212,992 → 106,496 = exactly 50.0%; G2 recorded 1.079×/1.088× (the long-context re-measure is T4's lane). P3 is the promotable P-half — the katgpt-rs-side promotion decision has its full input set.

## 2026-10-05 — Issue 013 T4 scaling half EXECUTED: the naive read-path cost explodes with context (1.08→1.79→3.82×); the pre-registered rule fired HOLD — katgpt-core `fitted_v_reconstruct` stays opt-in

Bench 016 (`.benchmarks/016_t4_g2_scaling_hold.md`, cells `016a`/`016b`, log `016_t4_g2_run.log`; cache-sizing fix `2300457`): 1.079× @129 → 1.789× @1025 → 3.821× @4097 window edge ≫ the ≤1.20 promotion bar → katgpt-rs promotion HELD (`682936a1c` there; HISTORY § Issue 883 amended); quality untouched (G3 PASS both cells at 1.254e-4, the 50% law re-confirmed). Promotion re-fires when the deferred-restore lever (Bench 895 Addenda I/II) or block angle-addition re-measures ≤1.20 at the window edge — T4's kernel half is that lane.

## 2026-10-05 — Issue 013 COMPLETE (T4 kernel half + the promotion EXECUTED): both levers landed as one read path, the window-edge bar met — `fitted_v_reconstruct` PROMOTED to katgpt-core default

Bench 017 (`.benchmarks/017_t4_deferred_lever.md`, cells `017a`/`017b`, log `017_t4_deferred_run.log`): the fused deferred-restore attention kernel + the bitwise per-position (sin,cos) table landed as ONE read path (`fe75497`, extended by katgpt-rs `618d7ec71`'s `FittedTokenTable::row_index_of` + `layer_rows_slab`; dispatch via `ValueStoreHook::fused_recon`). Curve 1.020×/1.053× @1025 and 1.050×/1.123× @4097 — both under the ≤1.20 bar. Promotion EXECUTED (katgpt-rs `34145bfd5`): `fitted_v_reconstruct` → katgpt-core default (compile availability; bench_895's primitive-level ≤1.00× gate flipped to RECORDED).

## 2026-10-05 — Issue 033 COMPLETE: the KV-axis extension arm landed and ran — the KV-store axis is NULL (a4-class KV storage accuracy-free), the activation axis dominates; instrument extended with the KV arm + hardened corpora

Bench 033 (`.benchmarks/033_kv_axis_extension/`, exe at `ad683a6`): the runner gained the KV-STORE axis (`DQ_KV_GRIDS` — every stored K/V row rounded at the CUDA `kv_fill` prefill + CubeCL `forward_attention_layer_gpu` decode sites) + `DQ_ARITH_HARD=1` / `DQ_NI_NEEDLES` corpus hardening. Verdict: KV stored/read precision NULL at a8 AND a4 (Δ = 0.0000 CI [0,0] at a8); the activation axis dominates catastrophically at a2 (pf 0.2083 / dec 0.1250 arith) and is free at a4 — the hybrid's KV cache tolerates a4-class storage. All gates PASS across 16 cells; issue file removed.

## 2026-09-27 — Issue 014 RESOLVED: activation-aware ternary scale fit — SPLIT BY LANE (born-ternary clean negative; dense-parent mechanism transfer)

From katgpt-rs Issue 886 (substrate `0fb2254d9`, katgpt-rs Bench 896). Resolution `3c2b569`; lanes `03681e9` (`act_taps`), `fc31777` (`act_retention_walk` bin), `28b1566` (dense-parent PTQ); benches `005_act_diagonal_bonsai_first_slice.md` (T1, feature `act_diagonal_calibration` over the `TernaryMatvecHook` seam; artifact digest `5b6aead0901e3133cd7ee28512e75122c788ae9d720dec78d099e6b390d9a9d2`) + `007_act_scale_refit_walk_and_ptq.md`. Born-ternary CLEAN NEGATIVE (the shipped amax payload IS the fit family's fixed point; ZeroQAT-class misaligns on damaged payloads); dense-parent gemma-2 f16 the mechanism TRANSFERS (ws_ex2 −34.6% vs mean_abs) — katgpt-rs 886 P1 closed on this evidence.

## 2026-09-27 — Issue 020 RESOLVED: the len_derived CAPACITY finding was the classifier's, not ours — and both fixed temp paths are gone

Finding 1: CAPACITY at `encoder_lane_cubecl.rs` was a classifier false positive, closed in the katgpt-rs instrument (`43c14d754`, the shared `length_from_handle_size_method` rule) — no riir-infer code changed. Finding 2 (riir-infer `20ea589`): the ANE compile-cache root → `~/Library/Caches/riir-infer/laya-ane` (`LAYA_ANE_CACHE` override), the exl3 oracle fixture → `<manifest>/.raw/exl3-pack`; `.docs/001` parenthetical updated. Rider: riir-instinct walk-floor rows re-pinned in katgpt-rs (clean HEAD `337da41`). shared_temp_path riir-infer `fixed=0`; clippy `-D` clean at `laya-riir-ane` and `--features exl3`.

## 2026-09-26 — Issue 019 RESOLVED: the c0dfa06 Metal barrier REVERTED — A alone was the wobble, and serial dispatches never needed barriers

Necessity conceded on the timing re-check (the "clean ×3 with barriers" triplet was confounded by `2a34bd3` in the working tree); premise refuted on the code: the fork's `begin_compute_pass` never sets `MTLDispatchType` — Metal defaults to SERIAL, memory visibility implied (the durable map if a concurrent-dispatch encoder ever lands). Barriers out (`4205c12`), probe instruments stay (zero prod cost); smoke 9/9, G5 cubecl ×3 bit-identical (3.092e-6 / 2.233e-6 / 5.187e-6), G5 wall 22-27 s without vs 24-31 s with.

## 2026-09-26 — Plan 611 DONE: the T7 op-layer unification verdict (Bench 006) — riir-reflex Issue 008 closed

One portable CubeCL `Backend` over this repo's op layer (S1–S4), A/B'd against the hand lanes (S5; rules pre-registered at `fac0dfd`): CubeCL 5.1–8.4× slower than Metal, 0/12 wins (Bench 006, results `ac85c8e`) — the hand Metal lane stays the macOS default; the CubeCL arm KEPT opt-in (`laya-riir-cubecl`; beats CPU on the short ladder). Engine-side payoffs kept regardless: `LayerNormMeanBatchedCubeCL` (two-pass mean-centered LN), batched in-place row-softmax, tiled matmuls, permutation kernels. Defects fixed en route: `gather_rows` residency (016), one-pass LN cancellation, the softmax race (`2a34bd3`, Issue 018), chunked-softmax row offset; 019's barrier reverted (`4205c12`). Harness: `crates/riir-infer-laya/tests/backend_ab.rs`.

## 2026-09-26 — Issue 018 CLOSED: the Cubecl-posture drift wobble was a softmax write-after-read race (fix `2a34bd3`)

Both CubeCL softmax kernels (`softmax_f32`, `softmax_rows_inplace_f32`) read the max from smem slot 0 then wrote the sum over it with NO barrier; one `sync_cube()` after the read (+ chunked-launch fix: rows past `MAX_WG_X` re-normalized row 0 — the chunk's first row now rides `params[1]`). Localized by lever 1 (per-op drains; every fire diverged at `L*.attn`). Measured: 43/120 fired passes at HEAD → 0/120 with the fix; G5 cubecl 10/10 bit-identical, gate re-armed (riir-reflex `ccb5bd0`); regression arms in `elementwise_cubecl::tests` fail 5/5 with the fix reverted. The second class was contested (Issue 019 — resolved: A alone sufficed). Full issue text at `270740c`.

## 2026-09-26 — the riir-ai carve docs adopted into this repo (Issues 998 + 1003, Plan 610, Proposal 041, Benches 870 + 871, doc 002)

Moved home with numbers verbatim (every "riir-ai Issue 998 / Plan 610 / Proposal 041" citation resolves to the same document): `.issues/998_riir_infer_repo_promotion.md` + `.issues/1003_riir_infer_carve_remains_4090.md` (OPEN), `.plans/610_riir_infer_gpu_carve_slice1.md` (IN PROGRESS, S8), `.proposals/041_riir_infer_model_based_inference_split.md`, `.benchmarks/870_ternary_recipe_audit.md` + `.benchmarks/871_issue879_t1_gdn_quant_certification.md` (the Issue 879 GDN quant records; harness `tests/issue879_gdn_quant_certification.rs` here since the carve), `.docs/002_riir_infer_core.md`. Cross-repo links re-pointed; each file carries a Moved header.

## 2026-09-25 — Issue 001 CLOSED: EXL3 (trellis-coded) weight format — reader complete, fused-GEMV lane closed at this tier

Full record (§1–§17 numbers unchanged) in `.docs/001_exl3_trellis_format_support.md`. Banked (opt-in `exl3`): CPU ref + LUT+rayon fast arm (`725a8a5`, 10.6–11.4×), pack loader (`bd93a7b`), CubeCL GPU decode v1/v2 (Bench 002, 71–87× wall on the 4090), sound bench harness (`9989e9f`), whole-pack bit-exact gate (`425e0c5`: 573/573 layers, 26.48 G weights, CUDA AND Metal), era gate (`55c154f`, known-good `{"1.4.2"}`, `open_unverified_era` escape). The real axis is residency (Bench 001: 3.4× vs f16, context 0 → ~117k on 24 GiB), not decode; fused GEMV closed by composition (0.95 s/step vs the 10–20 ms incumbent); reopen triggers §17.6.

## 2026-09-25 — T10 rung 2 (Metal): the attn_rope hoist pre-pass — default-off, promotion probe pending

`attn_rope` derives Q/K rope ONCE per layer into packed `[2, seq, d]` device scratch (`5ef7442`); opt-in `LAYA_METAL_ROPE_HOIST=1`, the in-kernel rope arm stays default and bit-identical; promotion via quiet-box position-balanced A/B (instrument archived: riir-reflex `.benchmarks/033_rope_hoist_ab/`). Lesson: widening the buffer list moved flash staging to `[[threadgroup(11)]]` while the host bound its length at index 9 (the unbound-pointer class; full-model G5 DEGENERATE surfaced it) — a shape-widening edit must move the host bind and the kernel binding together. Gates: `metal_ops_smoke` 7/7 ×2, `packed_forward_equiv` 4/4 ×2, `packed_same_shape_gate` 1/1, lib 41/41, G5 metal top-1 1.000000 ×3.

## 2026-09-25 — two silent MiniCPM5 (llama-arch GGUF) defects, found bringing it up as Issue 011's long-context fixture

RoPE pairing `0b26b9a`: llama.cpp stores llama Q/K rows interleaved; this crate's rotate-half RoPE never un-permuted — `rope::unpermute_interleaved_rows` in `load_llama_weights_gguf` (ppl 259.93 → 69.0547 == HF fp32; gemma2/qwen2 are NEOX, untouched). BPE digit rule `175f67f`: `BpeTokenizer` hard-coded Qwen2's one-digit-per-pre-token rule for every GGUF; llama-bpe groups `\p{N}{1,3}` (54,271/54,271 ids match after; found because passkey needles failed on number-dense text only). Instrument kept: `row_logit_floor_ppl --dump-tokens N`. Downstream consumers inherit; the riir-train canon benches (422/423/424/426/427/605) verdicts are unverified → riir-train Issue 571.

## 2026-09-25 — Issue 010 CLOSED (REFUTED): Gemma-2 has no QK-norm; the 288-tensor fixture is upstream-faithful

The premise was false: upstream `Gemma2Attention` defines no `q_norm`/`k_norm` — it bounds scores with `attn_logit_softcapping` (tanh, cap 50), already applied here (`attention_head_softcap`); per-head QK-norm arrived in Gemma-3. 288 = 11 tensors/layer × 26 + 2 globals is standard; `riir-train/data/gemma-2-2b-it-f16.gguf` is upstream-faithful, nothing re-run (Issue 010 filed `95f52fb`). Corrected in-commit: the `vk_calibration` bin's caveat 1 + `gemma2_calibration.rs`'s tap-law paragraph (for gemma-2 the pre-RoPE K tap IS the cache's K; katgpt-rs Issue 883 trap 1 applies to gemma-3/4-class stacks only). Lesson: check the reference source before filing a fixture-integrity finding.

## 2026-09-25 — Issue 009 CLOSED: software-pipelined narrow staging — NEGATIVE at the gate, and the probe's tiny-op floor exposed

`sgemm_narrow_pipe` built, measured, reverted: gate NO-GO (probe −1.5%; forward-level paired env-flip −0.8..−1.6%, reproducible 3/3, under the ≥3% bar; nvcc SINKS register loads — the `asm volatile("" ::: "memory")` fence is load-bearing). The real yield: `sgemm_shape_timing` tiny-op rows are ~90% launch/WDDM floor (FLOPs vary 400×, time 1.06×) — narrow-zone rungs must gate at FORWARD level or add a floor-subtraction arm; `.issues/006`'s head-tail rows were the same floor class. cp.async remains the recorded next lever (priced ≤ ~4%).

## 2026-09-25 — Issue 008 CLOSED: the narrow reg4 rung — NEGATIVE, the narrow zone is TLP-bound, not bandwidth-bound

`sgemm_narrow_reg4` (4×4 fragments, 32×16 warp tile → 128 threads): every true narrow-served row regressed +41..+177% (`sgemm_shape_timing`, the `LAYA_CUDA_REG4` axis; the wide/xwide rows re-measured the 007 rung at −2.6..−20% both runs — the consistency check). Mechanism: the narrow zone's grids are ≤128 blocks by construction — 1 block/SM, staging latency exposed; the measured ceiling was never the 20% roofline (~5% of fp32 peak at m=106). Reverted byte-identical (the BK48 precedent). Follow-up on record: double-buffered staging / `cp.async` — the right repair for a LATENCY-bound zone.

## 2026-09-25 — Issue 007 CLOSED: the register-blocking sgemm rung — 4×4 fragments, −9..−21 % kernel on every wide/xwide shape

`sgemm_wide_reg4` + `sgemm_xwide_reg4` (16 accumulators, 2 B smem/FMA vs 3, ceiling 50%; same grids so block-fit cliffs carry over; k-ascending per-output accumulation). Measured: banking77 zone −9.4..−16.1%, packed multi-wave −9.4..−21.4%, m=106/45 n≥2560 −10..−22%; forward english −5.3% / typed −3.9%; the reflex `.benchmarks/031` refresh (15 suites, host 4090-windows) median −6.2% (14/17 rows BIT-IDENTICAL; the typed_decisions wobble is that lane's pre-existing `determinism_ok: false`). Kill-switch `LAYA_CUDA_REG4=0`; gates `cuda_ops_smoke` 4/4, `packed_forward_equiv` 4/4, lib 41/41, G5 cuda 2/2, `laya_batch_parity` 1/1.

## 2026-09-25 — Issue 006 CLOSED: the float4 sgemm rung — every instance's B loads collapsed, −7..−17 % on every suite

The packed zone's problem was LOAD-ISSUE THROUGHPUT (12-16.5 TF/s = 15-20% of peak; 6 smem loads per 8 FMAs with the four B loads contiguous). B staging rows pad to a 16 B multiple (65→68, 129→132) and the four B loads → one `reinterpret_cast<float4>` (result-identical by construction, bit-verified). −10.3..−25.2% multi-wave, forward −12.5..−15.2%; the reflex `.benchmarks/030` refresh −6.8..−16.7% p50 incl. the packed suites, 13/16 rows bit-identical. No kill-switch — the float4 form IS the kernels now; `LAYA_CUDA_LADDER=0` still holds the A/B posture.

## 2026-09-25 — Issue 005 CLOSED: CUDA graphs — NEGATIVE, the lane is GPU-bound (submit fully hidden)

Instrument ships (`LAYA_CUDA_STATS=1`, zero cost unset): `gpu == wall ±0.3%` on every row of every checkpoint — the whole CPU submit path (launches ~15 µs/call, uploads, ~0.2 ms allocs) executes inside the GPU window; a graph replay removes work that is already free (~1-2% bounded, under the noise band) while adding exactly the stale-replay correctness surface. Reopen triggers (both on the instrument): kernels get much faster (submit becomes the wall) or allocs/uploads stop being hidden. Gates: `cuda_ops_smoke` 4/4, `packed_forward_equiv` 3/3, lib 41/41, G5 2/2, `laya_batch_parity` 1/1; stats-off parity byte-stable.

## 2026-09-25 — Issue 004 CLOSED: the sgemm tile ladder — block-fit floors, the narrow single-question win

Three instances (`sgemm_narrow` 32×64×64, `sgemm_wide`, `sgemm_xwide` 64×128×32) picked by BLOCK-FIT on the SM count — the M3 `m < 256` rule does NOT transfer to the 128-SM 4090 (measured cliff: narrow −15.7% at n=2048 exactly 128 blocks, +46% at n=2560/160 blocks). Narrow −13.7..−19.6%, xwide QKV −6.8..−8.5%, forward −8.4..−14.1%; the reflex `.benchmarks/029` single-question suites −6..−14%. Result-identical by construction (k-ascending per-output); 13/16 rows bit-identical. Kill-switch `LAYA_CUDA_LADDER=0`. Launch defect fixed in passing (dynamic smem over the 48 KB static default → `CUDA_ERROR_INVALID_VALUE`; all instances launch dynamic smem 0).

## 2026-09-25 — Issue 003 CLOSED: CUDA flash attention — the packed-path zeros defect fixed + the fused rung

v1's trait-default attention sliced host memory at non-zero offsets → `(ptr,len)` chain MISS → stale-zeros upload (typed_decisions 0.7445→0.2690; the evidence was the published bench, reflex `.benchmarks/026_4090windows_cuda` vs `018_4090windows_run`). Fix = the rung: the Metal one-pass online-softmax flash kernel ported to CUDA at fp32 FMA, offsets bind at dispatch; `LAYA_CUDA_FLASH=0` now PANICS on non-zero offsets (`supports_packed_attention` false; `needs_window_mask` mirrors the armed path). Gates: `cuda_ops_smoke` + the packed-offsets arm 1.2e-7, `packed_forward_equiv` CUDA arm (fails loud under the kill-switch), `laya_batch_parity` top-1 1.000000 drift ≤5.1e-5, G5 ~3e-6 class. Latency −5.6..−7.4% short seqs. Bench refresh + the published-numbers correction live in reflex (`.issues/028`, renumbered from 027 after a dual-allocation; reflex HISTORY §2026-09-25 bench 028).

## 2026-09-25 — CalibrationTables promoted substrate-side (883 P0 Kimi fixture rider)

`katgpt_core::fitted_anchor_table::LayeredVkCalibration` (+ `VkLayerTables`) is now the ONE builder for both 883 P0 fixtures — this repo's gemma-2 dashboard and katgpt-rs's new Kimi-K3 dashboard (katgpt-rs Bench 889: ρ(V)≈ρ(K)≈ρ(V−K) per MLA layer, the "coupled through one latent" signature; KDA layers = fixture-class null). `gemma2_calibration.rs` re-exports under the historical name `CalibrationTables` — zero behavior change; the tap forward, corpus loader, dashboard format stay here.

## 2026-09-24 — Issue 002 CLOSED: the CUDA backend for the laya lane (the 4090 bench row, 17–60× the CPU posture)

Landed `9b52cb1`/`99f156e`→`e99d767`: `laya-riir-cuda` — cudarc 0.19 (`driver`+`nvrtc`, `cuda-13030`+`fallback-dynamic-loading`, target-scoped `not(macos)`), CUDA C → PTX at construction (NVRTC, sm_89); the Metal architecture ported verbatim (weight cache, epoch-keyed chain slots, `download_into` prefix-read barrier via `CudaView`, lazy async one stream; `CudaSlice::clone()` is a d2d COPY → caches hold `Arc<CudaSlice<f32>>`). Two op-gate kernel defects fixed pre-G5 (staging loaded half the tile; the `b_cs==1` branch dropped the `n0` offset). G5 at `LAYA_DEVICE=cuda`: drift 1.863e-6 / 2.471e-6 / 2.894e-6 GREEN FIRST RUN; english/typed 12.0×, multilingual 10.6×, all below the M3 Metal row. Full bench refresh in riir-reflex `.issues/026`, `.benchmarks/026_4090windows_cuda/`.

## 2026-09-23 — crates.io publication: keep `publish = false` until the vendor patches upstream (owner-gates menu v2 row 2)

HARD blocker: both `vendor/` forks are load-bearing via `[patch.crates-io]` (`cubecl-runtime` — the #1359 drop-queue fix; `wgpu-hal` — the VRAM accessors) and a `[patch.crates-io]` section does not survive publication. Publication opens the day the vendor deltas land upstream. Boundary note: the public funnel for the stack's primitives remains `katgpt-rs`, not this repo.

## 2026-09-25 — Issue 020 T6 CLOSED NEGATIVE: the encoder's host side is 1–1.6% of forward wall (measured before building)

Instrument `crates/riir-infer-laya/tests/metal_host_gpu_split.rs` (`#[ignore]`, `required-features = ["laya-riir-metal"]`, measurement-only) splits `Encoder::forward_packed` into `enq` (the whole host side) vs `sync` (`download_into`): seq188 0.80/48.7 ms, seq512 1.45/141.6, packed2x256 1.37/135.2 — host share 1.0–1.6%, load-INFLATED (verdict robust). Allocation pooling cannot move case wall; the per-pass chain clear STAYS (its staleness guard is load-bearing, the Issue-015 class). The `[[test]]` required-features row landed per the T1.1e law.

## 2026-09-25 — the narrow sgemm's shape is the measured local optimum (T7 occupancy axis refuted, both arms)

Two challengers behind a temporary `LAYA_METAL_SGEMM_VAR` flag (from the consumer's `sgemm_shape_timing` probe, reflex `ebe667e`): bk32 lost 17–33%, bn32 15–45% — the barrier cost beats the 2–3× co-residency gain; at BK 64 a 2-TG fit breaks the bank-conflict padding or the 8×8 block structure. The variant code never landed — this record and the reflex issue carry the negative (the BK=48 precedent).

## 2026-09-25 — the sgemm MMA-roofline probe: the narrow instance is staging-bandwidth-bound (reflex Issue 020 T7 follow-up)

`crates/riir-infer-laya/examples/sgemm_roofline.rs` (measurement-only, `[[example]]` required-features row per the T1.1e law): the shipped narrow kernel verbatim vs an MMA-only twin (staged once, no re-staging, position-balanced). Narrow 3.13–4.88 TF/s vs roofline 5.11–10.24 (+63..+134%); the traffic math closes (B re-reads ≈1.6 GB ≈ the measured wall at ~400 GB/s). Verdict: NOT MMA-bound — the lever is B-OPERAND BYTES (f16 staging, predicted ~1.4–1.7×), not the f16 MMA; f16 weight rounding ~4.9e-4 is under the 1e-3 G5 gate but promotion is an Issue-750-T3 lossy-surface call (per-family retention, never the aggregate).

## 2026-09-25 — f16-B staging REFUTED at kernel level (the roofline probe's own follow-up arm)

The `sgemm_hb` arm (`device const half*` B, RNE `f32_to_f16` seed): flat ±2% on every cell — B (12.6 MB f32) FITS L2 (~32 MB), so the binding cost is the L1/threadgroup-issue path, which halving bytes does not relieve; the f16-B backend rung is dead before being built. Five axes now refuted at kernel level (occupancy bk32/bn32, BK=48, coalescing ×2, f16-B); narrow's shape is the measured local optimum. Reopen triggers: a Metal/toolchain change exposing direct-to-MMA staged layouts, or an L2-oversized working set (n > ~4096 — re-probe the packed path first).

## 2026-09-27 — Issue 021 CLOSED: the cuda CLS-row corruption — a chain-cache prefix-match
## alias, not a kernel race (the fused-GLU temp's slot vs the hidden that reused its address)

Fixed at the root: `chain_buf`/`chain_slot_for` now EVICT same-pointer different-length entries on bind; plus `download_into`'s `memcpy_dtoh` is `cuMemcpyDtoHAsync` (stream-ordered but ASYNC) — a trailing sync now guards the host read. Isolated by four committed probes (`cuda_repeat_probe`, `cuda_packed_repeat_probe`, the posture bisect incl. `LAYA_HEAD_DEFER`, `cuda_agent_repeat_probe`): the CLS prefix read tied two same-epoch slots at the recycled address of `matmul_w_glu`'s fused-GLU temp and `max_by_key` fell to HashMap order (~50/50, stable per process — why earlier repeat probes were green). Fix `c64d0b1`: agent probe 30×12 GREEN, harness `determinism_ok = true` ×4/4, accuracy unchanged. Instruments kept env-gated: `LAYA_DEBUG_ACT_ECHO`, `LAYA_CUDA_TRACE`; docs commit `9de953c`.

## 2026-09-27 — Issue 011 CLOSED: `row_logit_floor` model-bound G1 complete — needle@64K PASS at every arm

T3c (MiniCPM5-1B at 64K real dilution, log `/tmp/ri011run/t6_minicpm64k.log`): PASS at every arm b8/b6/b6s0/b4 — 3/3 seq-exact, 0.00% top-1 flips over 12 scored tokens, m_Y preserved within 0.0007 (same top head L15H7), base ppl 1.0362. Standing: T2 PASS 8/6-bit (gemma-2 4096), T3 PASS 64K, the T4 sink exemption load-bearing (the lossy-surface failure shape on real rows); 6-bit is the admissibility floor; n = 12 over 3 prompts — the row proves retrieval did not break, it cannot rank arms. Promotion owned by katgpt-rs Issue 903; the primitive stays opt-in (`ForwardContext.logit_floor: None` bit-identical). Bench `.benchmarks/003_row_logit_floor_ppl_needle.md`.

## 2026-09-28 — Issue 017 (gpu_transpose dead module) + Issue 023 (fence F2 self-alias FP): both closed

017: `gpu_transpose.rs` + `src/kernels/transpose.wgsl` deleted (zero callers then known), `transpose_cubecl.rs` (Issue 572) covers the reachable use, BOUNDARY.md repointed, the riir-ai re-export dropped (`6c6bf169b`). ⛔ Correction 2026-10-03: riir-train's `riir-train-engine` used `riir_gpu::gpu_transpose` behind default-off `kimi_k3_gpu_backward` — the module + WGSL moved beside that consumer (riir-train `04bd961b`); a dead-module grep must cover every path-dep repo at `--all-features`. 023: the third file renamed (`cubecl_encoder_probe.rs`), `fence_gate.py` 0 undefended, 0 pinned. Session hazard (twice): sibling sweeps staged these edits into their commits (`3bed93f` repaired by `734ef14`; riir-ai `8330e196b` repaired by `6c6bf169b`) — commit via pathspec in shared checkouts.

## 2026-09-28 — Issue 025 (owner-gate pickup) closed: D7/D8 executed, D9/D10 recorded

D7: gpu_transpose deleted + BOUNDARY.md repointed (record above). D8: the audio-lane BOUNDARY widening landed — an AUDIO Owns row (loader/serving-scoped, published CoreML bundles on `laya-riir-ane`) + the `objc2-core-ml` allowlist condition named (no new dep). D9: research 327–332 routing stays deferred with Plan 611 T7/S8 (tracked in 1004). D10: the S6b training-families disposition ratified into 1003's status — riir-gpu-side by design, closed absent a real consumer pull.

## 2026-09-29 — Issue 022 Phase 3 complete (audition + zero-training surrogate); Issue 024 closed measured-N/A both mechanisms

022 P3 (`27c9f86`): `src/twt/audition.rs` (mean + LaCo RDSC merge operators, the per-channel α/β branch-correction fit, BLAKE3-pinned selection) + `examples/twt_laya_audition.rs`; the apply path proven BIT-IDENTICAL against the parent forward every run. Every block's winner is a member passthrough (G-S-S caps homogeneous blocks at k=2); the α/β correction recovers 13–84% of boundary error on small blocks; artifacts `.raw/twt/{typed,english}_audition.json` (BLAKE3-pinned). 024 closed measured-N/A: 024a norm-share 2.1–2.4% of stage ≈ 1.8–1.9% of GPU (`LAYA_METAL_PROFILE=1`, preflight PASSED) — build nothing; 024b min m = 124 ≈ 4× the narrow tile M — no target shape, no kernel_opt rule filed (shape-transfer requires the shape, not the paper).

## 2026-09-29 — Issue 022 Phase 4 LANDED (re-ternarization arms + κ budget + collapsed-GGUF writer)

`baeb686` behind `twt_collapse`: `src/twt/ternarize.rs` (three deterministic arms; PRE-REGISTERED budget κ = 2.0, τ_code = 1; arm B = integer CODE vote, arm C divides by the f16-ROUNDED scale and is BIT-EXACT on ternary input) + `src/twt/collapse_writer.rs` (GGUF v3 streamed from the parent mmap — passthroughs BYTE-COPIES, metadata mirrored IN FILE ORDER (`metadata_order` + `GgmlType::id()`), the `{arch}.block_count` override required, `twt.*` provenance keys). The gate batteries caught two real defects at landing (the Q2_0 pack skipped zero weights — a skipped nibble IS code 0; the writer's offset plan desynced). Measured on `Ternary-Bonsai-2-27B-PQ2_0.gguf`: arm B DESTROYED on cross-scale merges (damage ≈20), arm C ≈0.31 (κ admits it only where surrogate error ≥ ~0.31), arm A the only budget-viable arm on strong merges; the operator pre-read shows GDN members near-orthogonal (rel-dist ≈2) — Phase 5 reordered to passthrough-collapse first. Gates: `twt_ternarize_gates` 15, `twt_ternarize_g4` 1, `twt_collapse_writer_gates` 4.

## 2026-09-28 — Issue 015 audio PoC DEFERRED (owner: until M5 Ultra) + 023's missed hunk landed

Issue 015 stays OPEN, T1–T6 deferred (`- [-]`), the turnkey recon recorded in-issue (exact bundle `silero-vad-unified-256ms-v6.2.1.mlmodelc`, the wire contract from FluidAudio's `VadManager` @ `20d4f0bd`; the audio lane bypasses the laya digest/manifest coupling and REPORTS the plan verdict). T0's boundary rows (D8) stay landed. `e8e17b8` landed the `cubecl_encoder_probe.rs` alias rename that 023's closing record (`61ebf9c`) claimed but the sweep repair left in the worktree — the record-vs-tree divergence class.

## 2026-09-29 — Issue 022 T5.0 LANDED: the passthrough-collapsed checkpoint + the first real GOAT numbers (coarse grid FAIL; fine end measured)

Loader prerequisites fail-closed (`318b9fa`/`ceb96c3`): the writer REFUSES a qwen35 collapse without `twt.layer_types` (U8 `DeltaNetLayerType`) and refuses a stale `qwen35.nextn_predict_layers`; the loader (`qwen35_deltanet_config_from_gguf_metadata`, now `pub`) replaces derived types on the key; `prism.hadamard.weight_names` renumbers in place (interlock via `is_known_folded_name`). Emit lane `9905c7c` (`examples/twt_collapse_emit` over `.raw/twt/bonsai_ultrachat_profile.json`, `minmax_partition`, `twt.parent_weights_blake3`). SEVEN real collapsed checkpoints of `Ternary-Bonsai-2-27B-PQ2_0.gguf`; riir-train's `plan402_gguf_probe.py` reads every `twt.*` key cross-repo AS-IS. `twt_goat_agreement` (`--cache`, params-keyed): the coarse sweep ALL FOUR grid points FAIL (ε=0.05 → 0.1945 … ε=0.3 → 0.0000; parent hit rate 0.7478) — cosine redundancy is NOT a license for depth cuts. Kill-rule finding: re-derive the bonsai profile's `SURVIVES` before any T5.5 Pareto claim cites it.

## 2026-09-29 — Issue 022 T5.0 COMPLETE: the agreement cliff mapped — 4.7% depth cut PASSES the GOAT bar, quality parity holds to 11%

Bench `.benchmarks/022_t5_collapsed_goat_agreement.md`: ε=0.01 (61/64 blocks, 4.7% cut) agreement 0.9486 — the FIRST pass of the pre-registered ≥0.9 bar; ε=0.015 → 0.8955; ε=0.02 → 0.5247; ε=0.03 → 0.0301; the grid points all FAIL. The two-metrics finding: the collapsed model's top-1 hit rate holds PARITY to 11% cut (0.7505 vs 0.7478) while agreeing on only 89.6% of argmaxes — trajectory divergence overstates functional damage one notch; carry the hit-rate column beside any agreement claim. Past 11% the hit rate falls off the same cliff; riir-train 423's distillation owns the regime beyond. The parent arm is cached (params-keyed, loud replay).

## 2026-09-29 — Issue 022 T5.1 lane (1) COMPLETE: gemma-2 f16 control — clean negative at every real depth cut; the fine-end bracket is structurally empty

Instrument `0e0436c` (`PostLayerHook` capture seam on `forward_gemma2_f16_tapped`; `examples/twt_gemma2_profile` behind `twt_gemma2`; arch arms in `twt_collapse_emit` + `twt_goat_agreement`), verdict `55fa831`; bench `.benchmarks/015_t51_gemma2_control_goat.md`. ε=0.05 identity 1.0000 (4092/4092 — bit-faithful); EVERY real cut FAILS (65.4% → 0.2571 … 3.8% → 0.0015); the fine-end bracket is structurally EMPTY (ε≤0.03 → zero merges). Zero-training passthrough closed on TWO architectures; the rescue lives in the apply-path (auditioned merges / distillation, riir-train 423 — the control hands it the baseline + negative control). En-route: a live Bench-number collision with the M3 sibling — renumbered 014→015 per the collision rule; the collapsed GGUFs (~21 GB) were deleted post-measurement (regenerable from `.raw/twt/gemma2_profile.json` + the cached parent arm).

## 2026-09-29 — two Windows batch traps hit by the T2/T3/T5.1 schtask runners (recorded; the deleted wait-loop script's durable note)

(1) A `Start-Process`-launched `.cmd` inherits the MSYS PATH — `tasklist | find` resolves `/usr/bin/find` and the wait-loop wedges forever; the schtask launchers are immune (clean system env → `C:\Windows\System32\find.exe`). Rule: never hand-launch a wait-loop `.cmd` from MSYS; use `schtasks /Run` or fully-qualify `%SystemRoot%\System32\find.exe`. (2) A hand-rolled no-wait batch rewrite died silently — for chained measurement runs prefer `schtasks` wrappers (the Issue-012 recipe) or PowerShell. (3) A literal `)` in an echo INSIDE a parenthesized block silently makes the follow-up `exit /b 1` unconditional (found in `run_kv_reconstruct_gate.cmd`; the same latent bug fixed in `run_twt_gemma2_goat.cmd`) — in-block echoes must carry NO literal `)`.

## 2026-09-29 — Research 004: Disaggregated Quantization distilled (arXiv:26.26333) — three issues filed, riir-train Plan 430 routed

Distill of arXiv:2609.26333 (DQ/QADD), deepening the riir-clippy arxiv-walk-217 row: Issues 026 (phase-isolated quant sensitivity bench), 027 (T0 encoder-only asymmetric Q2_0 + Lloyd-Max grids), 028 (dual-PTQ resident disaggregated serving) filed; track (c) → riir-train Plan 430 (QADD prefiller pre-registration). Signal-diffs: the GDN escape set (Issue 980 `gate_projections()` + the issue879 f32 recipe) corroborated; `dl_qat.rs` carries no teacher/phase-mask; prior art named (OverFill 2508.08446, Decode-Branch 2608.12385). Verdict gate: claude ping-pong AGREE round 2 (session `c68b3113-bb7f-49d1-853b-ac6e215e46be`). Master: `.research/004_DQ_Disaggregated_Quantization.md`.

## 2026-10-01 — Plan 614 LANDED: the DQ phase matrix ran to EXIT0; every axis INADMISSIBLE at the frozen corpora; the instrument (and its defect chain) is the deliverable

v2 `23bbff5`+`3cf9ae9`, EXIT0 (log `F:/wt/dq614-matrix2.log`, report `F:/wt/dq614-matrix2/dq_phase_matrix.md`, record `.benchmarks/023_dq_phase_matrix.md`, `.highwater` 15→23): all four axes INADMISSIBLE (base arith 0.9583 — the `a07fffd` `reset_state` fix lifted the genuine rate; NIAH pooled 0.9896) — Issue 026 T3's assertions receive NO GATE; the report-only tables carry the signal (a2 prefill-only collapses NIAH to 0.25 while decode-only holds 0.93). Defect chain fixed at source: `a07fffd` GDN state leak, `23bbff5` phase-blind G-i2 control, `3cf9ae9` the 4090-clippy discharge (the frozen corpus blake3 `6f3c6f02…` reproduces byte-exact). Open follow-up: the FATAL teardown path HANGS (→ Issue 031). Durable lesson (bench 023): a run lane needs ONE owner at a time (the twin-session collision; the twin-duplicate resolved per the Issue-825 class — `5977565` died in the reset, its value re-landed as `3cf9ae9`); ALIVE guards on `E:/git/_sync/dq614_{chain,matrix}.cmd`.

## 2026-10-01 — Issue 027 CLOSED measured-negative + removed: the offline LUT grid lane (Plan 615) — weight-space wins were real, the 2-bit class is function-destroying on dense artifacts

`2cb807d`: T0's asymmetric encoder (`quantize_row_q2_0_asymmetric`) +4.9 dB/family and Q2_0A's Lloyd-Max grids +0.5–1.2 dB/family — but MiniCPM5-1B ppl 6.8e5–7.0e6 vs base 49.8 at EVERY reachable granularity; the pre-registered class gate fired. What stays in-tree (tested, ungated): `src/quant/q2_0.rs` (+ `dequantize_row_q2_0_grid`), `src/quant/lut_grid.rs` (d²-weighted histogram — the objective must match the metric; EM + BLAKE3 grid commitment, cross-box determinism gemma `b28b3661…` / MiniCPM `642e2d0b…`), bins `lut_grid_solve`/`lut_grid_ppl`. T2 (the Q2_0A wire) MOOT; the Bonsai lane untouched (`symmetric_reencode_of_ternary_blocks_is_byte_identical`). Master `.research/004_DQ_Disaggregated_Quantization.md`.

## 2026-10-01 — Issues 998 + 1004 CLOSED (hygiene): the D4 re-narrowing dissolved under the ratified no-migration architecture; the corpus-follows triggers resolved negative

998: the D4 re-narrowing is resolved-dissolved — P3/T7 landed as the ENCODER-lane unification under its pre-registered NO-DELETION verdict and D10 (owner-gate, riir-ai 1016) ratified the training families stay riir-gpu-side BY DESIGN; every widening's consumption edge is live, reversal would break the ratified architecture (the riir-ai BOUNDARY.md D4 row carries this resolution). 1004: both triggers fired WITHOUT the migrations — zero corpus movement (`crates/riir-gpu/tests/bench_874_issue879_t3_kv_weight_quant_nll.rs`, the kernel tree, `scripts/perf_rematch.sh` + `.docs/09_performance/` all still riir-ai-side; the 870/871 rule holds). Task 3 (research 327–332 routing; katgpt-rs the candidate destination) stands on its own. Issue 1003 stays OPEN deliberately. Narratives: `git log --follow -- .issues/998_riir_infer_repo_promotion.md` / `.issues/1004_corpus_follows_op_layer_migration.md`.

## 2026-10-05 — Issue 031 CLOSED measured-non-repro + removed: the dq614 FATAL teardown hang did not reproduce across 6 arms; the A/B instrument is landed as durable

Instrument: `crates/riir-infer-gpu/src/bin/dq614_teardown_repro.rs` (`required-features = ["ternary_gemv_cuda_raw"]`, whole-file `#![cfg]`, dev-profile refusal) with arms `--exit plain|graceful|hard` × `--sticky` × `--alloc-mb` × `--iters` × `--no-cuda`; no-op repro knobs `DQ614_FORCE_FATAL=1` / `DQ614_EXIT_PLAIN=1` in `dq_phase_matrix.rs` (the default stays `hard_exit`). All 6 arms exited clean incl. the full 7.2 GB Bonsai engine-state shape (clean exit at 215 s). Remaining hypotheses recorded, not chased (driver/OS drift, or ~6-hour context wear). Defense double-covered: `hard_exit` + `dq614_watchdog.ps1` (machine-local at `E:/git/_sync/dq614_watchdog.ps1`; narrative `git log --follow -- .issues/031_dq614_fatal_teardown_hang.md`).

## Lessons

- Check the reference implementation, not a remembered standard, before filing a fixture-integrity finding (Issue 010).
- A G1 that stops below the production chunk size certifies a size nobody ships (Issue 032).
- A dead-module grep must cover every repo that path-depends on the crate, at `--all-features` (Issue 017).
- In shared checkouts, commit via pathspec (`git commit -- <paths>`), never the bare index.
- A run lane needs ONE owner at a time; the handoff summary's active-plan section must name the owning session (bench 023).
- A shape-widening edit must move the host bind and the kernel binding together (T10 rung 2).
- Tiny-op kernel probes read a ~90% submission floor — narrow-zone rungs must gate at FORWARD level (Issue 009).
- Bandwidth-bound zones want register blocking; latency-bound zones want cp.async/double-buffering — the rooflines diverge, so the rungs must too (Issues 007/008).
- The sgemm block-fit floor is MEASURED per GPU (SM count), never ported from another box (Issue 004).
- Trajectory (argmax-agreement) divergence overstates functional damage — carry the hit-rate column beside any agreement claim (Issue 022 T5.0).
- Cosine/state redundancy ≠ functional redundancy, on ternary and dense alike (Issue 022 T5.0/T5.1).
- Weight-space dB wins mean nothing without the model-level gate — the class verdict can kill every arm (Issue 027).
- Windows batch: no literal `)` in echoes inside `( ... )` blocks; never hand-launch a wait-loop `.cmd` via `Start-Process` from MSYS; prefer `schtasks` wrappers.
- Feature-gated targets carry their `required-features` row at birth — a whole-file `#![cfg]` target without one prints a green zero forever (the T1.1e law).
- `[patch.crates-io]` does not survive crates.io publication — `publish = false` until the vendor forks land upstream.
