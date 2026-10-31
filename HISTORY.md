# HISTORY.md — riir-infer

Compacted 2026-10-30: every `##` heading kept verbatim, one compact entry per record (≤2 short lines); the full pre-compaction text is in git history. Operational rules live in `AGENTS.md`; removed issue files: git history.

## 2026-10-30 — Issue 1157 CLOSED + removed: regime_gate inline sigmoid delegated to the substrate — bit-preserving, pinned

- `crates/riir-infer-gpu/src/regime_gate.rs` `fn sigmoid` is now a thin delegation to `katgpt_core::exact_sigmoid` (the bit-stable two-branch reference form, body-identical — NOT the Cephes `sigmoid`, whose ±40 clamp / ~1.2e-7 drift a gate decision must not inherit); delegation pinned bit-for-bit against the frozen legacy body over a fixed grid (`sigmoid_delegation_matches_frozen_legacy_body`, this commit). Copy-gate class finding from the substrate-first audit; 20/20 regime_gate tests green, clippy clean.

## 2026-10-30 — AGENTS.md compaction: the Role section's lane-landing narratives moved here (switches + postures stay in AGENTS.md's lane table)

- GDN L8 eager prefill (plan 621): PROMOTED DEFAULT-ON 2026-10-24 — riir-ai Bench 993 +3.22% / Bench 997 +2.76% pp2048, pp4096 +2.62% post-promotion, G1 bytewise; the ±8.4% band clause superseded by measurement.
- GEMV WG occupancy (riir-ai Issue 1114): default-on Metal 2026-10-24, CUDA 2026-10-25 (Bench 887, Issue 1115); +4.88/+5.52% e2e on the m5ultra. Fused fold-tail/SwiGLU-rotate kernels (Issue 1113): −160 launches/token, +1.40% e2e below bar → opt-in (Bench 995).
- GDN delta-roll (Issue 049, 2026-10-20): rank-1 unwind recovery ~64× vs snapshots, G1 drift 6.3e-6 (Bench 885). Stability-depth readout (Issue 1106, 2026-10-22). TOKENGATE T3.4 four-arm GOAT FAIL (bench 889); break-even ceiling ~1.5–1.8× at K≤0.25 (bench 891).
- Plan 623 multi-host: Phase 1 2026-10-26 (G1 63/63 qwen splits); verbs FFI `226cace`+`16fb64a` — RTT p50 7.7 µs, 9.368 GiB/s (Bench 927); Bench 928 `570ee79`+`f9c4803`; T3.4 async-finish `9f3c0f7` demoted promote-only (`6a1c524`, −48.5% on wire); T3.5 decode 52.3–52.6 tok/s vs llama.cpp-RPC 48.6 (Bench 929).
- Issue 1124 GLU-fused prefill FFN `1320dda`; T1.5 folded-lane wiring `c60b34d`; promoted `57fa7a7` (verdict `9ca951de`, +6.31/+6.72% pp2048); T3 mined via refine `d871a0fc`.
- Decode-hold rungs: core-fused `4132d94` (G2 negative), beta-expand `8aabe55` (+0.6–1.0%), quantize census falsifier `a2ebb37`. Dense-pair (issue 1129) `c400dbc` +9.88/+9.07% decode, CUDA re-cell +1.54% (Bench 1014, test head `0e509e4`). Prework-fused `e2edf61`+`2581fdd` (riir-ai `d25c7c460`), CUDA N/A proof `69e7d8e`. Qwen3.8 dense mm2d 50.22× kernel-level (Issue 1150).

## 2026-10-29 — Issue 654 promotion e2e EXECUTED + the gate REFUSED: the CUDA-placed head measured, the roll-chain loss priced structural (Bench 931)

The owner-GO'd promotion rung ("654 lgtm", claimed zed-m3-i654-promo) executed on the 4090 primary platform per the pinned posture-B gate — promotion REFUSED, the arm stays opt-in default-off: every replicated cell loses to plain decode (k=2: 0.846×/0.946× paired ratios; k=4: 0.745× — riir-train Bench 682's …)
Commits: `06c184a`, `4c77521`, `7e20540`, `b769ccc`.

## 2026-10-29 — Bench 929 follow-up: the plan-623 launch-context discriminator RE-TESTED — class refuted, harness spawn-retry landed

The Bench-929 PROVENANCE discriminator ("detached-matrix spawns EHOSTUNREACH 6/6 while foreground connects, root cause not found") was re-tested on demand (zed-m3-idle1029b): the launch-context class is REFUTED as a deterministic posture property — the exact matrix launch shape …

## 2026-10-29 — Issue 1155 CLOSED + removed: the exact slot-addressable GDN state PoC — hypothesis CONFIRMED (claim `89b4880`, tests `36c71f6`, addendum `5552964`, hash re-pin `074fc8f`)

The Qwenputer identity PROVED on our kernels (the divagr18/qwenputer distill, Research 018): one-hot keys address one state column through the REAL `gated_deltanet_step_inplace` bit-exactly when payloads are bounded integers in the ±2^24 grid + gate=1 + β=1 (G1a PASS incl.)

## 2026-10-29 — Issue 1150 follow-up LANDED: the dense-f16 matmul2d prefill arm — the Metal scalar-GEMM wall broken on the dense lane (A/B 50.22× kernel-level, claim `2c6dfaa`+`049d3ce`, this commit)

The Issue-1150 session record's named follow-up (all in-crate CubeCL GEMMs at ~110 GFLOP/s on Metal — the wall the Bonsai league's 0.249× prefill row shares): `gemm_dense_f16_metal_tensor.metal` + `gemm_dense_f16_metal_zero_copy.rs` — the dense twin of the ternary zero-copy lane (riir-ai Issue 663 T5 …)

## 2026-10-29 — Plan 623 Phase 3 COMPLETE: T3.5 scoreboard LANDED + the T3.4 arm wire-refuted and demoted (Bench 929, issue 1153 `ed10f14`+record, issue 1154 `6a1c524`)

The Phase-3 scoreboard on the amo⇄bmo TB wire (zed-m3-i971-housekeeping, foreground-chunked cells, bmo carrying the disclosed mammoth-capture co-tenant): decode row — ours 2-box s1 52.28–52.59 tok/s BEATS llama.cpp-RPC 2-box 48.6 (+7.6%; the Issue-1141..1146 production rungs flipped T2.7's −20% …)
Commits: `6a1c524`.

## 2026-10-29 — Plan 623 Phase 3 T3.3 CLOSED + T3.2's rkey half CLOSED: the MTU ladder is neutral, RDMA WRITE refused-by-silence (Bench 928, T3.2/T3.3 `570ee79`+`5304716`, bench `f9c4803`)

`verbs_write` + `verbs_metal` (feature `rdma_metal`, metal 0.31 — one workspace version, the BOUNDARY.md metal row) + the verbs-module surface they needed: `create_qp_typed`/`create_rc_qp`, `post_rdma_write`/`post_rdma_read` (the WR opcode + `IbvRdmaPart` path), `wr/wc_opcode::RDMA_READ` pins, and …

## 2026-10-29 — Plan 623 Phase 3 T3.4 LANDED: the decode-feed async-finish arm (issue 1151, T3.4 `9f3c0f7`)

> ⚠ **1151 is DOUBLY HELD** — this is the SECOND holder (`1151_plan623_t34_gpu_comm_sync`, filed `148f22d` reading the counter's value 1151 instead of value+1 — the counter already stood at 1151 after `559be00` — removed `726cec7`). The first holder is *Issue 1151 — the production posture single home* below. Both files are deleted, so neither can be renumbered; cite this one as **Plan 623 T3.4 (the issue-1151 GPU↔comm sync unit)**, never bare "Issue 1151".

`riir-infer-gpu` re-exports `{block_on, DynFut}` + `read_hidden_state_fut()` — the staged-copy read future ('static via a client clone, resolving to the f32 vector), created at ENQUEUE time so the copy encodes FIFO-after this stage's chain and its resolution IS the chain-done signal value …

## 2026-10-29 — Plan 623 Phase 3 T3.1 LANDED: the verbs FFI lane — first RDMA traffic on the amo⇄bmo TB bridge (Bench 927, T3.1 `226cace`+`16fb64a`, bench `322e843`)

`crates/riir-infer-dist/src/verbs/{ffi,mod}.rs` behind the opt-in `rdma` feature (macOS-only by contract — the `ibv_context_ops` vtable layout is Apple's librdma.dylib header's; rdma-core's evolved vtable is not claimed): the SDK header mirrored repr(C) with layout pins held to the C compiler's …

## 2026-10-28 — Issue 1151 CLOSED + removed: the promoted serving posture has a single home — the `production` aggregate + `PRODUCTION_RUNGS` + the dist posture floor (fix `80ff665`, close `b5acf92`)

> ⚠ **1151 is DOUBLY HELD in this repo** — this is the FIRST holder (`1151_production_posture_aggregate_single_home`, filed `559be00`, removed `f6b3b6a`), and it KEEPS the number (first allocation; katgpt-rs `citation_weight.py` awards it 11–5). The second is *Plan 623 T3.4 — the GPU↔comm sync unit* above (`1151_plan623_t34_gpu_comm_sync`, filed `148f22d`, removed `726cec7`). Cite this one as **Issue 1151 (production posture)**.

The miss class fired THREE times in nine days — each a consumer's hand-typed serving-feature list omitting a rung the league/engine posture carried: 1141 (dist ran the bare decode kernels, 37.24 → 21.4x ms/token after repair, 1.74×), 1146 (chunked prefill + batched attention omitted from dist's …)
Commits: `b5acf92`.

## 2026-10-28 — Issue 1129 CLOSED + removed: the cuda_backend CubeCL decode A/B validates the dense-pair default (riir-ai Bench 1014, the 4090 lane)

The remaining scope (the CudaRuntime arm of the DEFAULT-ON dense-pair fusion — the unmeasured-default class; the scored 4090 decode lane is cudarc, whose K2 fuses the pair) measured + closed from the amo idle lane over SSH while amo held zero-cargo under the Plan-337 T5 posture.
Commits: `cbcdb5a`, `0e509e4`.

## 2026-10-28 — Plan 623 T2.5: the boot-probe planner is fully wire-automatic (the probe EXCHANGE + its residue)

The EXCHANGE (zed-amo-i623-exchange, `22eb298`): the head fetches the tail's boot-probe record over the wire and composes `plan_split` live — no operator hand-carries JSON between hosts.
Commits: `a260b62`.

## 2026-10-28 — the grid-reselection drafter arm LANDED (katgpt-rs 966/plan-632; the arc closed NEGATIVE on the lane read, the arm stays opt-in)

`lattice_walk_reselect` (feature `grid_reselect`, `c04e3fa`): the keyed-walk body with the pick replaced by the prefix-first scored law `argmax_b [ln softmax(s/T) + λ·ln P(b|a)]` over each slot's own selector grid — re-scored along the re-selected prefix (`chain_step_scores` conditions on …)

## 2026-10-27 — Issues 1145 + 1139 CLOSED + removed: the M5 Ultra decode-gap campaign closes at 1.04× vs llama.cpp (Bench 907: multi-cube norm+rotate PROMOTED DEFAULT-ON, +2.83%)

Issue 1145 (the campaign's last routed rung, filed from the 1139 T4 census): one launch of `dim/hblock` cubes (5 at Bonsai-2's 5120/1024) replaces the mlp4-norm + r4-staging PAIR at each decode norm site — every cube redundantly runs the incumbent's Phase-1 mlp4 hoist + tree over the WHOLE vector …
Commits: `d1b6156`.

## 2026-10-27 — Plan 623 Phase 4 (TP): T4.1/T4.2 cell built + correct, gate REFUTED by measurement (Bench 906); Issue 1109 CLOSED + removed

First real TP cell over jaccl (`scripts/plan623_mlx_tp2_bench.py`): column/row sharding of the Bonsai-2 pack runtime, both ranks run all 64 layers, `all_sum` after row mats + vocab-split lm_head `all_gather`, greedy lockstep with NO token exchange.

## 2026-10-27 — Issue 1127 CLOSED + removed: the decode-hold escalation ladder — T1–T4 all executed with records; no owner gate remains (verdict, this session)

All four tasks closed with measured records: T1 attention-chain rung opt-in +0.77% (Bench 900); T2 census-negative (Bench 902); T3 measured-negative at the bit-identical boundary (Bench 903); T4 league re-cell EXECUTED (Bench 1008, riir-ai `e3ffd3004`) and adjudicated (`efd9332`, reviewer …)
Commits: `9ca951de`, `57fa7a7`, `c7ebe93`, `df9c464`.

## 2026-10-27 — Issue 1133: GDN conv+beta/decay+expand merge rung MEASURED-NULL (Bench 905: G1 exact everywhere, G2 2/4 pairs, pooled ≈0% ± 0.5%) — shipped opt-in, NOT promoted (`a4da7be` + the rung commit)

The last unmerged launch-bound boundary in the armed GDN decode chain: `conv_beta_decay_expand_fused_f32` folds conv1d + the Bench-899 beta/decay+expand pair into ONE launch (48 layers × 1 node ≈ 62 µs/token dispatch predicted ≈ +0.57%).
Commits: `fe06c7e`.

## 2026-10-26 — Issue 1122: history-fed n-gram drafter probe CLOSED-NEGATIVE (Bench 895: re-roll α 0.063 vs breakeven 0.58 — T1 wiring refused) — file removed

The NgramRL class (draft decode call N from the model's own same-prompt outputs) probed BEFORE wiring per the Issue-665 honest-assessment pattern.

## 2026-10-26 — Issue 1121: StabilityTracker first-ordinal pin CLOSED — landed `c34b954` — file removed

Hardening pin (Issue 1106's `stability.rs`): the first observed stage is now `debug_assert_eq!(ordinal, 0)` — a partial/offset capture feed (e.g. starting at `AfterLayer(30)`) passed both shipped guards (width + strictly-increasing) and reported INDEX-space depth, silently wrong by the offset.
Commits: `22afc87`.

## 2026-10-25 — Issue 1118 Finding 1: BPE merge ranks re-keyed to packed token-id pairs — LANDED (Bench 894: G1 bit-identical, G2 2.03× encode, G4 zero-alloc probes) — `ad7b5a0`+`54a6616`+`ddc7105`+`c5032f6`

The merge loop allocated two heap Strings per adjacent pair per merge iteration (its own comment admitted it, deferring "a future optimization").
Commits: `7b2fed9`, `addd0b4`, `0e9cc4ce`.

## 2026-10-24 — Issue 1112: arm8 whole-prefill CUDA crash on unfolded (pre-rotation) synth shapes CLOSED — root cause + fix + regression GREEN (`0522312` + `441b78f` + `63ab2d2`) — file removed

The crasher was never a kernel: the synth Bonsai-2-style file is UNFOLDED with DENSE a/b, `upload_layer_weights` stores the dense f32 mirrors beside EMPTY dummy `TernaryGroupWeights::new(0, 0)` handles, and the whole-prefill PLAIN lane (`rotation.is_none()`) mirrored the dummies into 0-word CUDA …
Commits: `1b5b623`, `d53c8f3`, `63ab2d2`, `e90d775`, `441b78f`.

## 2026-10-13 — Issue 042: cubecl #1806 idle-server park fix vendored CLOSED — landed `fa86871` (mirrors: riir-ai `4b86d2003`, riir-refine `f0718269`) — file removed

The idle-queue's "cubecl pin-bump re-gate" lane (queued cycles 38–41, blocked on the 038 sibling's WIP — cleared when their lane landed in full at `4db9e15`).
Commits: `6bba5fdf0`, `58e7689`, `0c5ed0fe7`.

## 2026-10-10 — Issue 037: wgpu full-suite device-loss cascade CLOSED — root cause + VRAM admission gate landed — `3564a72` (+ upstream draft `dad3e7c`) — file removed

The "which test kills the device" question was a category error: no single killer test exists — the full `cubecl_runtime` suite needs ~16 GiB contiguous commit and the loss fired at the box's GLOBAL VRAM commit position (trainer 7.6 GiB resident; VRAM trajectory 9.0 → 23.8 GiB at the loss second) …
Commits: `02e7a65`.

## 2026-10-09 — the elementwise params[0]=n root fix + the vendored metal adapter E0308 — `02e7a65`

Retired katgpt-rs Issue 928's two len_derived stability rows: sigmoid_f32 / silu_f32 bound by an EXPLICIT params[0]=n (the copy_f32_grid2d idiom), never input.len(); the new oversized-binding sentinel test proves [n, n+pad) untouched.]
Commits: `3564a72`.

## 2026-10-09 — Issue 036: LittleBit init-only sub-1-bit PTQ CLOSED — NO-GO for the format+kernel lane (Bench 034; T4 closed-by-verdict) — file removed

The paper's unmeasured point (Dual-SVID init-only, no QAT) measured end-to-end on gemma-2-2b-it (the T0 host decision `8d3fd7e`: local PPL bin over existing substrate — `forward_gemma2_f16` + `nll` + SentencePiece; wikitext-2-raw test sha256 `aca2f467…`, 49,152 tokens; anchor 18.67 validates the …)
Commits: `b235f4e`.

## 2026-10-07 — Issue 035: FA-constrained dLLM decoding POC CLOSED — NOT PROMOTED (G2 FAIL, sequential lane) — `7d90dfd` (+ file removed this commit)

P3 executed on the 4090 box (quiet: desktop-only GPU consumers, 26.6/33.3 GB RAM free; real-vocab arm = gemma-2-2b 256k, `../riir-train/data/gemma-2-2b-it-f16.gguf`).
Commits: `f4ffa93`, `7bda765a`.

## 2026-10-06 — Issue 028 T4 S2: the q4 weight arm — the container loads and runs the dual-format pair — (this commit)

`ProjWeights { Ternary(TernaryGroupWeights), Q4K(Vec<BlockQ4K>, rows, cols) }` at the 10 per-layer projection fields; `bitlinear` dispatches the Q4_K arm to fused `gemv_q4_k_row` (rayon across 16-row chunks, bit-stable), GPU hooks refuse q4 LOUD, loader `load_proj` storage-type dispatch.

## 2026-10-06 — Issue 028 T4 S1: the Q4_K prefill pack exists — plus the science premise settled — `94a4e7c`

`Ternary-Bonsai-2-27B-Q4_K.pf.gguf`, 14.43 GB, blake3 `0a7a5605fadc37b482c0fddf6c28334244eea818b1716461bddb412f1a28309c`; the science premise: the bonsai is TRAINED TERNARY, so the PQ2_0 wire is exact up to f16 group scales — requant carries only ε(Q4_K) (`UnsupportedFourthState` is the structural …)

## 2026-10-06 — the GPU crate is clippy `-D` clean at every posture — `e993eff`

Pre-existing warnings gone; the rotation quantize pair (`quantize_rotate_q8`/`_permute_rotate_q8` + `*_warp` fields) is test-consumed only — allow gate `not(all(test, feature = "prefill_q8_act"))` (the first `not(any(test, …))` was wrong in exactly the firing posture); `DQ_FQ_CUDA_SRC` is dead only …

## 2026-10-06 — Issue 035 P2.6 LANDED: real-vocabulary compile + the row-fold — the honest G2 regime — `ddcaeb0`

Trie-stepped subset construction in `fa_schema.rs` (256k vocab compiles the fn-call grammar in 60s; toy 0.13s, object 1.8s) + one vocab-wide exp row per position in `fa_posterior.rs` (fn-call draw 11.79s → 1.98s at L=256).

## 2026-10-06 — Issue 035 P2.5 LANDED: DFA minimization — the G2 ratio drops ~4.4× — `9e94684`+`672e2b0`

`src/fa_minimize.rs` (UNGATED, 7 tests): trim + block-granularity partition refinement applied ALWAYS inside `fa_schema::compile` (language-preserving by construction); empty-language schemas now fail at COMPILE time (`SchemaError::InvalidSchema`).

## 2026-10-06 — Issue 035 P2 LANDED: the schema front end + the honest overhead table — `1254264`+`5b5c462`

`src/fa_schema.rs` (UNGATED, 12 tests): JSON-schema subset (six types · properties/required · items · enum · anyOf) → Thompson NFA → on-demand subset graph → token `Automaton`; three grammar-soundness laws pinned (fresh ws wrappers, per-(mask,member) member chains, acceptance = walk ∧ final).

## 2026-10-06 — Issue 022 T5.4 CLOSED: the equal-FLOP skip-class gate — the lane wins, honestly — `8bd5c2d`+this commit

Record `.benchmarks/022_t54_skip_arms.md` (instrument `src/bin/twt_skip_amputation.rs` at `a7b080c`): twt 0.8919 > hydra 0.8642 > random floor 0.8437 >> ShortGPT 0.0320 at the equal 3-layer cut — selection disagreement total, ShortGPT's BI greedy 81 points below random (a dense-model removal …)

## 2026-10-06 — Issue 035 P1 LANDED: the `fa_constraint` decode wiring — exact joint x→0 per denoising step — `304876f`+`f4ffa93`

P0's sampler wired into the `gemma2_d2f` denoising loop behind opt-in `fa_constraint` (implies `gemma2_d2f` ⇒ `dllm`): ONE exact joint draw per step; commit axes `RawTop1` / `ConstrainedMarginal` (marginal log rides `SamplerFeatures::marginal_log`, the 7-param `ConstrainedSampler` — the 6-slot …)

## 2026-10-01 — riir-ai Issue 1004 R2 LANDED: the fused GDN prework (conv1d + SiLU + q/k L2-norm + head expansion, ONE dispatch) — `929d31e`

Feature `deltanet_prework_fused` (toggle `set_prefill_prework_fused`): 2 dispatches/layer vs the shipping chain's ~69; stage-isolated 7.29× @2048 / 9.13× @4096; G1 bit-identical (logits FNV `0ec2396fd4627f29` @2048 / `f35280cb95f5d306` @4096).

## 2026-10-01 — Issue 026 CLOSED (hygiene): the DQ phase-matrix instrument landed and ran — every axis INADMISSIBLE at the frozen corpora

Plan 614's instrument (`dq_fakequant` + the `dq_phase_matrix` runner; record `.benchmarks/023_dq_phase_matrix.md`) ran to EXIT0: base arith 0.9583, NIAH pooled 0.9896 — both OUTSIDE the pre-registered 0.25–0.95 window, so T3's damage-ratio assertions receive NO GATE; T5's promotion rule (any future …)
Commits: `5d0592f`, `4f68e5d`.

## 2026-10-01 — Issue 032 DONE: Metal CubeCL prefill computes Hadamard-folded (Bonsai-2) models (`5d0592f`)

Batched twins of all 8 decode rotation sites in the CubeCL prefill body + `GemvBatchedCubeCL::launch_token_grid`; G1 vs folded decode-eager top-1 equal, worst rel err ≤ 1.5e-5; pre-rotation Q2_0 pin `99a0733c45a0e663` @2048 exact; first Bonsai-2 M3 prefill 6.44× TTFT (100.9 tok/s @2048); test …
Commits: `5f075ff`, `7e99f34`.

## 2026-09-30 — reflex Issue 054 Part 2 REFUTED for this lane: the prefix-state handoff lead — laya is ModernBERT (bidirectional), not GDN; the coupling gate landed

The premise named the wrong architecture: laya checkpoints are ModernBERT-large / mmBERT-base (no causal mask; the state also sits after the per-question head span — different RoPE offset per question).

## 2026-09-30 — Issue 013 T2 EXECUTED: the K=V+ λ ladder on gemma-2 — G-A PASS vs a catastrophic baseline; K=V+ not viable standalone; G-D overfit; NIAH build crash (029)

Record `.benchmarks/012_t2_kv_ladder_gate.md` (+ `012_t2_run.log`, table `012_kv_table_residual.bin` sha `3a0f5333d6bafd63…`, reused by T3): f16 6.0907 (== T1, cross-run determinism PASS) vs k-0.00 catastrophic (tax +30,948,056%); k-1.00 λ*=1 G-A PASS (3344× recovery) but still ×92.6 off f16 → K=V+ …

## 2026-09-30 — Issue 013 T3 EXECUTED: P3 V-cache reconstruction — G1+G3 PASS at every λ; the 50.0% bytes/token law holds; G2 +7.9–8.8% recorded (the promotable P-half)

Chained run rc 0 (waited 399 min for T2, BLAKE3-verified its table artifact). Record `.benchmarks/013_t3_p3_reconstruct_gate.md` (+ `013_t3_reconstruct_report.md` written incrementally, `013_t3_run.log`): G3 max |Δlogit| 1.254e-4 (the rotation-rounding class); G1 PASS 0 flips in 2046 scored …

## 2026-10-05 — Issue 013 T4 scaling half EXECUTED: the naive read-path cost explodes with context (1.08→1.79→3.82×); the pre-registered rule fired HOLD — katgpt-core `fitted_v_reconstruct` stays opt-in

Bench 016 (`.benchmarks/016_t4_g2_scaling_hold.md`, cells `016a`/`016b`, log `016_t4_g2_run.log`; cache-sizing fix `2300457`): 1.079× @129 → 1.789× @1025 → 3.821× @4097 window edge ≫ the ≤1.20 promotion bar → katgpt-rs promotion HELD (`682936a1c` there; HISTORY § Issue 883 amended); quality …

## 2026-10-05 — Issue 013 COMPLETE (T4 kernel half + the promotion EXECUTED): both levers landed as one read path, the window-edge bar met — `fitted_v_reconstruct` PROMOTED to katgpt-core default

Bench 017 (`.benchmarks/017_t4_deferred_lever.md`, cells `017a`/`017b`, log `017_t4_deferred_run.log`): the fused deferred-restore attention kernel + the bitwise per-position (sin,cos) table landed as ONE read path (`fe75497`, extended by katgpt-rs `618d7ec71`'s `FittedTokenTable::row_index_of` + …)
Commits: `34145bfd5`.

## 2026-10-05 — Issue 033 COMPLETE: the KV-axis extension arm landed and ran — the KV-store axis is NULL (a4-class KV storage accuracy-free), the activation axis dominates; instrument extended with the KV arm + hardened corpora

Bench 033 (`.benchmarks/033_kv_axis_extension/`, exe at `ad683a6`): the runner gained the KV-STORE axis (`DQ_KV_GRIDS` — every stored K/V row rounded at the CUDA `kv_fill` prefill + CubeCL `forward_attention_layer_gpu` decode sites) + `DQ_ARITH_HARD=1` / `DQ_NI_NEEDLES` corpus hardening.

## 2026-09-27 — Issue 014 RESOLVED: activation-aware ternary scale fit — SPLIT BY LANE (born-ternary clean negative; dense-parent mechanism transfer)

From katgpt-rs Issue 886 (substrate `0fb2254d9`, katgpt-rs Bench 896). Resolution `3c2b569`; lanes `03681e9` (`act_taps`), `fc31777` (`act_retention_walk` bin), `28b1566` (dense-parent PTQ); benches `005_act_diagonal_bonsai_first_slice.md` (T1, feature `act_diagonal_calibration` over the …)

## 2026-09-27 — Issue 020 RESOLVED: the len_derived CAPACITY finding was the classifier's, not ours — and both fixed temp paths are gone

Finding 1: CAPACITY at `encoder_lane_cubecl.rs` was a classifier false positive, closed in the katgpt-rs instrument (`43c14d754`, the shared `length_from_handle_size_method` rule) — no riir-infer code changed.
Commits: `20ea589`, `337da41`.

## 2026-09-26 — Issue 019 RESOLVED: the c0dfa06 Metal barrier REVERTED — A alone was the wobble, and serial dispatches never needed barriers

Necessity conceded on the timing re-check (the "clean ×3 with barriers" triplet was confounded by `2a34bd3` in the working tree); premise refuted on the code: the fork's `begin_compute_pass` never sets `MTLDispatchType` — Metal defaults to SERIAL, memory visibility implied (the durable map if a …)
Commits: `4205c12`.

## 2026-09-26 — Plan 611 DONE: the T7 op-layer unification verdict (Bench 006) — riir-reflex Issue 008 closed

One portable CubeCL `Backend` over this repo's op layer (S1–S4), A/B'd against the hand lanes (S5; rules pre-registered at `fac0dfd`): CubeCL 5.1–8.4× slower than Metal, 0/12 wins (Bench 006, results `ac85c8e`) — the hand Metal lane stays the macOS default; the CubeCL arm KEPT opt-in …
Commits: `2a34bd3`, `4205c12`.

## 2026-09-26 — Issue 018 CLOSED: the Cubecl-posture drift wobble was a softmax write-after-read race (fix `2a34bd3`)

Both CubeCL softmax kernels (`softmax_f32`, `softmax_rows_inplace_f32`) read the max from smem slot 0 then wrote the sum over it with NO barrier; one `sync_cube()` after the read (+ chunked-launch fix: rows past `MAX_WG_X` re-normalized row 0 — the chunk's first row now rides `params[1]`).
Commits: `ccb5bd0`, `270740c`.

## 2026-09-26 — the riir-ai carve docs adopted into this repo (Issues 998 + 1003, Plan 610, Proposal 041, Benches 870 + 871, doc 002)

Moved home with numbers verbatim (every "riir-ai Issue 998 / Plan 610 / Proposal 041" citation resolves to the same document): `.issues/998_riir_infer_repo_promotion.md` + `.issues/1003_riir_infer_carve_remains_4090.md` (OPEN), `.plans/610_riir_infer_gpu_carve_slice1.md` (IN PROGRESS, S8) …

## 2026-09-25 — Issue 001 CLOSED: EXL3 (trellis-coded) weight format — reader complete, fused-GEMV lane closed at this tier

Full record (§1–§17 numbers unchanged) in `.docs/001_exl3_trellis_format_support.md`. Banked (opt-in `exl3`): CPU ref + LUT+rayon fast arm (`725a8a5`, 10.6–11.4×), pack loader (`bd93a7b`), CubeCL GPU decode v1/v2 (Bench 002, 71–87× wall on the 4090), sound bench harness (`9989e9f`), whole-pack …
Commits: `425e0c5`, `55c154f`.

## 2026-09-25 — T10 rung 2 (Metal): the attn_rope hoist pre-pass — default-off, promotion probe pending

`attn_rope` derives Q/K rope ONCE per layer into packed `[2, seq, d]` device scratch (`5ef7442`); opt-in `LAYA_METAL_ROPE_HOIST=1`, the in-kernel rope arm stays default and bit-identical; promotion via quiet-box position-balanced A/B (instrument archived: riir-reflex …)

## 2026-09-25 — two silent MiniCPM5 (llama-arch GGUF) defects, found bringing it up as Issue 011's long-context fixture

RoPE pairing `0b26b9a`: llama.cpp stores llama Q/K rows interleaved; this crate's rotate-half RoPE never un-permuted — `rope::unpermute_interleaved_rows` in `load_llama_weights_gguf` (ppl 259.93 → 69.0547 == HF fp32; gemma2/qwen2 are NEOX, untouched).
Commits: `175f67f`.

## 2026-09-25 — Issue 010 CLOSED (REFUTED): Gemma-2 has no QK-norm; the 288-tensor fixture is upstream-faithful

The premise was false: upstream `Gemma2Attention` defines no `q_norm`/`k_norm` — it bounds scores with `attn_logit_softcapping` (tanh, cap 50), already applied here (`attention_head_softcap`); per-head QK-norm arrived in Gemma-3. 288 = 11 tensors/layer × 26 + 2 globals is standard …
Commits: `95f52fb`.

## 2026-09-25 — Issue 009: software-pipelined narrow staging CLOSED — NEGATIVE at the gate, and the probe's tiny-op floor exposed

`sgemm_narrow_pipe` built, measured, reverted: gate NO-GO (probe −1.5%; forward-level paired env-flip −0.8..−1.6%, reproducible 3/3, under the ≥3% bar; nvcc SINKS register loads — the `asm volatile("" ::: "memory")` fence is load-bearing).

## 2026-09-25 — Issue 008: the narrow reg4 rung CLOSED — NEGATIVE, the narrow zone is TLP-bound, not bandwidth-bound

`sgemm_narrow_reg4` (4×4 fragments, 32×16 warp tile → 128 threads): every true narrow-served row regressed +41..+177% (`sgemm_shape_timing`, the `LAYA_CUDA_REG4` axis; the wide/xwide rows re-measured the 007 rung at −2.6..−20% both runs — the consistency check).

## 2026-09-25 — Issue 007: the register-blocking sgemm rung CLOSED — 4×4 fragments, −9..−21 % kernel on every wide/xwide shape

`sgemm_wide_reg4` + `sgemm_xwide_reg4` (16 accumulators, 2 B smem/FMA vs 3, ceiling 50%; same grids so block-fit cliffs carry over; k-ascending per-output accumulation).

## 2026-09-25 — Issue 006: the float4 sgemm rung CLOSED — every instance's B loads collapsed, −7..−17 % on every suite

The packed zone's problem was LOAD-ISSUE THROUGHPUT (12-16.5 TF/s = 15-20% of peak; 6 smem loads per 8 FMAs with the four B loads contiguous).

## 2026-09-25 — Issue 005: CUDA graphs CLOSED — NEGATIVE, the lane is GPU-bound (submit fully hidden)

Instrument ships (`LAYA_CUDA_STATS=1`, zero cost unset): `gpu == wall ±0.3%` on every row of every checkpoint — the whole CPU submit path (launches ~15 µs/call, uploads, ~0.2 ms allocs) executes inside the GPU window; a graph replay removes work that is already free (~1-2% bounded, under the noise …)

## 2026-09-25 — Issue 004: the sgemm tile ladder CLOSED — block-fit floors, the narrow single-question win

Three instances (`sgemm_narrow` 32×64×64, `sgemm_wide`, `sgemm_xwide` 64×128×32) picked by BLOCK-FIT on the SM count — the M3 `m < 256` rule does NOT transfer to the 128-SM 4090 (measured cliff: narrow −15.7% at n=2048 exactly 128 blocks, +46% at n=2560/160 blocks).

## 2026-09-25 — Issue 003 CLOSED: CUDA flash attention — the packed-path zeros defect fixed + the fused rung

v1's trait-default attention sliced host memory at non-zero offsets → `(ptr,len)` chain MISS → stale-zeros upload (typed_decisions 0.7445→0.2690; the evidence was the published bench, reflex `.benchmarks/026_4090windows_cuda` vs `018_4090windows_run`).

## 2026-09-25 — CalibrationTables promoted substrate-side (883 P0 Kimi fixture rider)

`katgpt_core::fitted_anchor_table::LayeredVkCalibration` (+ `VkLayerTables`) is now the ONE builder for both 883 P0 fixtures — this repo's gemma-2 dashboard and katgpt-rs's new Kimi-K3 dashboard (katgpt-rs Bench 889: ρ(V)≈ρ(K)≈ρ(V−K) per MLA layer, the "coupled through one latent" signature; KDA …)

## 2026-09-24 — Issue 002 CLOSED: the CUDA backend for the laya lane (the 4090 bench row, 17–60× the CPU posture)

Landed `9b52cb1`/`99f156e`→`e99d767`: `laya-riir-cuda` — cudarc 0.19 (`driver`+`nvrtc`, `cuda-13030`+`fallback-dynamic-loading`, target-scoped `not(macos)`), CUDA C → PTX at construction (NVRTC, sm_89); the Metal architecture ported verbatim (weight cache, epoch-keyed chain slots, `download_into` …)

## 2026-09-23 — crates.io publication: keep `publish = false` until the vendor patches upstream (owner-gates menu v2 row 2)

HARD blocker: both `vendor/` forks are load-bearing via `[patch.crates-io]` (`cubecl-runtime` — the #1359 drop-queue fix; `wgpu-hal` — the VRAM accessors) and a `[patch.crates-io]` section does not survive publication.

## 2026-09-25 — Issue 020 T6 CLOSED NEGATIVE: the encoder's host side is 1–1.6% of forward wall (measured before building)

Instrument `crates/riir-infer-laya/tests/metal_host_gpu_split.rs` (`#[ignore]`, `required-features = ["laya-riir-metal"]`, measurement-only) splits `Encoder::forward_packed` into `enq` (the whole host side) vs `sync` (`download_into`): seq188 0.80/48.7 ms, seq512 1.45/141.6, packed2x256 1.37/135.2 …

## 2026-09-25 — the narrow sgemm's shape is the measured local optimum (T7 occupancy axis refuted, both arms)

Two challengers behind a temporary `LAYA_METAL_SGEMM_VAR` flag (from the consumer's `sgemm_shape_timing` probe, reflex `ebe667e`): bk32 lost 17–33%, bn32 15–45% — the barrier cost beats the 2–3× co-residency gain; at BK 64 a 2-TG fit breaks the bank-conflict padding or the 8×8 block structure.

## 2026-09-25 — the sgemm MMA-roofline probe: the narrow instance is staging-bandwidth-bound (reflex Issue 020 T7 follow-up)

`crates/riir-infer-laya/examples/sgemm_roofline.rs` (measurement-only, `[[example]]` required-features row per the T1.1e law): the shipped narrow kernel verbatim vs an MMA-only twin (staged once, no re-staging, position-balanced).

## 2026-09-25 — f16-B staging REFUTED at kernel level (the roofline probe's own follow-up arm)

The `sgemm_hb` arm (`device const half*` B, RNE `f32_to_f16` seed): flat ±2% on every cell — B (12.6 MB f32) FITS L2 (~32 MB), so the binding cost is the L1/threadgroup-issue path, which halving bytes does not relieve; the f16-B backend rung is dead before being built.

## 2026-09-27 — Issue 021 CLOSED: the cuda CLS-row corruption — a chain-cache prefix-match

## alias, not a kernel race (the fused-GLU temp's slot vs the hidden that reused its address)

Fixed at the root: `chain_buf`/`chain_slot_for` now EVICT same-pointer different-length entries on bind; plus `download_into`'s `memcpy_dtoh` is `cuMemcpyDtoHAsync` (stream-ordered but ASYNC) — a trailing sync now guards the host read.
Commits: `c64d0b1`, `9de953c`.

## 2026-09-27 — Issue 011 CLOSED: `row_logit_floor` model-bound G1 complete — needle@64K PASS at every arm

T3c (MiniCPM5-1B at 64K real dilution, log `/tmp/ri011run/t6_minicpm64k.log`): PASS at every arm b8/b6/b6s0/b4 — 3/3 seq-exact, 0.00% top-1 flips over 12 scored tokens, m_Y preserved within 0.0007 (same top head L15H7), base ppl 1.0362. Standing: T2 PASS 8/6-bit (gemma-2 4096), T3 PASS 64K, the T4 …

## 2026-09-28 — Issue 017 (gpu_transpose dead module) + Issue 023 (fence F2 self-alias FP): both closed

017: `gpu_transpose.rs` + `src/kernels/transpose.wgsl` deleted (zero callers then known), `transpose_cubecl.rs` (Issue 572) covers the reachable use, BOUNDARY.md repointed, the riir-ai re-export dropped (`6c6bf169b`).
Commits: `04bd961b`, `3bed93f`, `734ef14`, `8330e196b`.

## 2026-09-28 — Issue 025 (owner-gate pickup) closed: D7/D8 executed, D9/D10 recorded

D7: gpu_transpose deleted + BOUNDARY.md repointed (record above). D8: the audio-lane BOUNDARY widening landed — an AUDIO Owns row (loader/serving-scoped, published CoreML bundles on `laya-riir-ane`) + the `objc2-core-ml` allowlist condition named (no new dep).

## 2026-09-29 — Issue 022 Phase 3 complete (audition + zero-training surrogate); Issue 024 closed measured-N/A both mechanisms

022 P3 (`27c9f86`): `src/twt/audition.rs` (mean + LaCo RDSC merge operators, the per-channel α/β branch-correction fit, BLAKE3-pinned selection) + `examples/twt_laya_audition.rs`; the apply path proven BIT-IDENTICAL against the parent forward every run.

## 2026-09-29 — Issue 022 Phase 4 LANDED (re-ternarization arms + κ budget + collapsed-GGUF writer)

`baeb686` behind `twt_collapse`: `src/twt/ternarize.rs` (three deterministic arms; PRE-REGISTERED budget κ = 2.0, τ_code = 1; arm B = integer CODE vote, arm C divides by the f16-ROUNDED scale and is BIT-EXACT on ternary input) + `src/twt/collapse_writer.rs` (GGUF v3 streamed from the parent mmap …)

## 2026-09-28 — Issue 015 audio PoC DEFERRED (owner: until M5 Ultra) + 023's missed hunk landed

Issue 015 stays OPEN, T1–T6 deferred (`- [-]`), the turnkey recon recorded in-issue (exact bundle `silero-vad-unified-256ms-v6.2.1.mlmodelc`, the wire contract from FluidAudio's `VadManager` @ `20d4f0bd`; the audio lane bypasses the laya digest/manifest coupling and REPORTS the plan verdict).
Commits: `e8e17b8`, `61ebf9c`.

## 2026-09-29 — Issue 022 T5.0 LANDED: the passthrough-collapsed checkpoint + the first real GOAT numbers (coarse grid FAIL; fine end measured)

Loader prerequisites fail-closed (`318b9fa`/`ceb96c3`): the writer REFUSES a qwen35 collapse without `twt.layer_types` (U8 `DeltaNetLayerType`) and refuses a stale `qwen35.nextn_predict_layers`; the loader (`qwen35_deltanet_config_from_gguf_metadata`, now `pub`) replaces derived types on the key …
Commits: `9905c7c`.

## 2026-09-29 — Issue 022 T5.0 COMPLETE: the agreement cliff mapped — 4.7% depth cut PASSES the GOAT bar, quality parity holds to 11%

Bench `.benchmarks/022_t5_collapsed_goat_agreement.md`: ε=0.01 (61/64 blocks, 4.7% cut) agreement 0.9486 — the FIRST pass of the pre-registered ≥0.9 bar; ε=0.015 → 0.8955; ε=0.02 → 0.5247; ε=0.03 → 0.0301; the grid points all FAIL.

## 2026-09-29 — Issue 022 T5.1 lane (1) COMPLETE: gemma-2 f16 control — clean negative at every real depth cut; the fine-end bracket is structurally empty

Instrument `0e0436c` (`PostLayerHook` capture seam on `forward_gemma2_f16_tapped`; `examples/twt_gemma2_profile` behind `twt_gemma2`; arch arms in `twt_collapse_emit` + `twt_goat_agreement`), verdict `55fa831`; bench `.benchmarks/015_t51_gemma2_control_goat.md`.

## 2026-09-29 — two Windows batch traps hit by the T2/T3/T5.1 schtask runners (recorded; the deleted wait-loop script's durable note)

(1) A `Start-Process`-launched `.cmd` inherits the MSYS PATH — `tasklist | find` resolves `/usr/bin/find` and the wait-loop wedges forever; the schtask launchers are immune (clean system env → `C:\Windows\System32\find.exe`).

## 2026-09-29 — Research 004: Disaggregated Quantization distilled (arXiv:26.26333) — three issues filed, riir-train Plan 430 routed

Distill of arXiv:2609.26333 (DQ/QADD), deepening the riir-clippy arxiv-walk-217 row: Issues 026 (phase-isolated quant sensitivity bench), 027 (T0 encoder-only asymmetric Q2_0 + Lloyd-Max grids), 028 (dual-PTQ resident disaggregated serving) filed; track (c) → riir-train Plan 430 (QADD prefiller …)

## 2026-10-01 — Plan 614 LANDED: the DQ phase matrix ran to EXIT0; every axis INADMISSIBLE at the frozen corpora; the instrument (and its defect chain) is the deliverable

v2 `23bbff5`+`3cf9ae9`, EXIT0 (log `F:/wt/dq614-matrix2.log`, report `F:/wt/dq614-matrix2/dq_phase_matrix.md`, record `.benchmarks/023_dq_phase_matrix.md`, `.highwater` 15→23): all four axes INADMISSIBLE (base arith 0.9583 — the `a07fffd` `reset_state` fix lifted the genuine rate; NIAH pooled …)

## 2026-10-01 — Issue 027 CLOSED measured-negative + removed: the offline LUT grid lane (Plan 615) — weight-space wins were real, the 2-bit class is function-destroying on dense artifacts

`2cb807d`: T0's asymmetric encoder (`quantize_row_q2_0_asymmetric`) +4.9 dB/family and Q2_0A's Lloyd-Max grids +0.5–1.2 dB/family — but MiniCPM5-1B ppl 6.8e5–7.0e6 vs base 49.8 at EVERY reachable granularity; the pre-registered class gate fired.

## 2026-10-01 — Issues 998 + 1004 CLOSED (hygiene): the D4 re-narrowing dissolved under the ratified no-migration architecture; the corpus-follows triggers resolved negative

998: the D4 re-narrowing is resolved-dissolved — P3/T7 landed as the ENCODER-lane unification under its pre-registered NO-DELETION verdict and D10 (owner-gate, riir-ai 1016) ratified the training families stay riir-gpu-side BY DESIGN; every widening's consumption edge is live, reversal would break …

## 2026-10-05 — Issue 031 CLOSED measured-non-repro + removed: the dq614 FATAL teardown hang did not reproduce across 6 arms; the A/B instrument is landed as durable

Instrument: `crates/riir-infer-gpu/src/bin/dq614_teardown_repro.rs` (`required-features = ["ternary_gemv_cuda_raw"]`, whole-file `#![cfg]`, dev-profile refusal) with arms `--exit plain|graceful|hard` × `--sticky` × `--alloc-mb` × `--iters` × `--no-cuda`; no-op repro knobs `DQ614_FORCE_FATAL=1` / …

## 2026-10-10 — Issue 040 CLOSED landed + removed: the weight-histogram MDL description-length audit (GGUF lane); the EXL3 rows ride the loader lane

`483f2ec`: `quant::desc_len` (feature `desc_len`) + the `desc_len_audit` report bin — per-tensor entropy / floor / slack sorted by HONEST slack, with the two-part honesty law (`N·H(p)` prices the weight VALUES only; the codebook + block-format side-info is priced separately, advisory rows sink).

## 2026-10-11 — Issue 1003 CLOSED (owner-gate record superseded): the D10 disposition stands — the 69 training families stay riir-gpu-side BY DESIGN; the live homes are the riir-ai BOUNDARY.md D4 row + riir-ai HISTORY:360

This closure SUPERSEDES the 2026-10-01 sentence in the Issue-998 closure row ("Issue 1003 stays OPEN deliberately") — that sentence predates the owner's 2026-10-05 rule ("rm owner-gate issues; ask Claude to judge; defer→HISTORY+close", riir-ai HISTORY:360) and the Claude-adjudicated verdict of …
Commits: `65a7268779`, `7e0e7d74`, `498b1732b9`.

## 2026-10-11 — Issue 028 CLOSED (PoC gate fired NULL, container shelved): the dual-PTQ disaggregated serving record; T5 stays deferred with its trigger

T1–T3 landed `09d0dd5`: the `DisaggregatedTernaryWeights` container (`load_pair` / `.pf` one-file / `from_single`), the ownership-preserving in-process `PhaseHandoff`, the escape-set law (`ssm_a`/`ssm_dt.bias`/`ssm_alpha`/`ssm_beta` bit-shared between copies, loud refusal naming divergence), the …

## 2026-10-11 — katgpt-rs Plan 622 Phase 7 engine lane LANDED (doc-sync row): the scale-invariant attention vehicle on the packed Bonsai ternary decode path; GPU legs open, GPU-window-gated

Landed `d1b9bd5` (session katopz-m3-622-continue) + follow-ups `71a0845` (loop_probe stale layer_idx repair) / `47f8f6f` (quiet the posture-dependent unused layer_idx warning) / `9f3b27b` (the harness compiles at its OWN required-features posture — the P43 class: single-feature clippy runs skipped …)

## 2026-10-13 — Issue 041 CLOSED (T1–T5 all landed): the prefix-cache snapshot submit burst — 96 submits → 2 launches per direction, the serving-lane TTFT lane

Landed `971b842` (T1+T2: the pointer-table copy path — module-local `TableCopyEngine` in `qwen38_prefix_cache.rs`, dual device pointer tables (live-state freshness-checked via a host address compare + per-entry tables), ONE `prefix_table_copy` launch per class per direction, opt-in …)
Commits: `48c2948`, `bb2a4b5`.

## 2026-10-13 — Issue 1005 CLOSED (T1–T8 all landed; defers carry their triggers): the eDLM/Drex inference lane — GGUF arch, block-causal mask, pointer head, marker encode, real-artifact parity, GPU forward, KV reuse, CUDA graphs

Phase 1–3 complete across four sessions: T1–T3 (the `edlm` GGUF arch + segment block-causal mask + row-form fallback), T4–T6 (marker encode/pack + the pointer head + T6 parity PASSED on the real `drex-dlm-Q8_0.gguf` — 87 tokens EXACT vs their published count, worst drift 0.0103, all three argmaxes …)

## 2026-10-13 — Issue 039 CLOSED measured-non-repro + removed: the wgpu `--lib` device-loss tail did NOT reproduce at HEAD on a clean exclusive box (green ×2 parallel + ×1 serial); the CUDA-posture leg surfaced and fixed a posture-biased TEST instead

The free window opened (verified idle: 5% util, 376 MiB, no compute apps) and the full recipe ran from a fresh isolated worktree at the then-tip `fbd6e6c` (`E:\git\wt039infer`, isolated `CARGO_TARGET_DIR`): wgpu posture (parallel) 258/0 in 520.27s; wgpu posture again (the issue's own twice-in-a-row …)
Commits: `450cdfc`.

## 2026-10-22 — Issue 1107 CLOSED negative-audit + removed: NO riir-infer path right-pads rows into a shared recurrent (GDN) prefill — the TokenGate pads-to-front law has nothing to repair here; the one multi-sequence surface (laya `forward_packed`) is varlen-packed by construction, and the riir-ai HLA note does not fire either

Audit per the issue's task 3 (a negative result is a valid close; not left open as a maybe). Paths checked, each at its own entry point's signature and launch geometry: (1) CPU `prefill_qwen_deltanet` / `_into` / `_chunk_into` (`src/deltanet/forward.rs`) — `prompt_tokens: &[usize]` single slice …

## 2026-10-23 — Issue 1111 CLOSED + removed: `forward_qwen_deltanet_ternary_hidden` returns the head basis (`R·h` on a Hadamard-folded file) — KEPT and documented, width fixed, pinned on the synthetic folded fixture

Filed from riir-train Plan 461 T2.0 (P3 measured the returned vector = the FORWARD-ROTATED final-norm state on `Ternary-Bonsai-2-27B-PQ2_0.gguf`; the doc claimed the primal state and D = 12288).
Commits: `4ede8e7b4`.

## 2026-10-25 — Issue 1113 CLOSED + removed: the mma `kv_f16_to_f32` +112 re-bias — fix + both pins EXECUTED on the 4090 (the compile-in-context window the M3 could not reach)

The one-line re-bias (`(exp + 112u) << 23`) and the two device-free pins landed 2026-10-24 (zed-m3-docsync-956-close) from the M3, where the module compiles to nothing (`#[cfg(all(feature = "ternary_gemv_cuda_raw", not(target_os = "macos")))]`); the M3's scratch `rustc` harness validated the pin …
Commits: `f72eff6`.

## 2026-10-25 — Issue 1114 CLOSED + removed: the Issue-994 KV guard default is backend-keyed (Metal measured clean to 128K)

The single-buffer KV guard's 208 MiB default is the 4090/`wgpu<spirv>` bisect (riir-ai Issue 994). On the M5 Ultra under `wgpu<msl>` the 64K and 128K rungs both passed their anchor-ref gates against an independent 32K artifact (riir-train Bench 657), so the guard refused working configurations …
Commits: `cd748a6`, `38612c5`.

## 2026-10-25 — Issue 1110 CLOSED + removed: the β-reopen probe — ROW DIES AT PROBE COST (Bench 890, renumbered from a colliding 888 — see the record's numbering note): the only real gemma2 checkpoint is a plain IT model, the d2f decode is a structural 1-step `<eos>` collapse, the drift census has zero substrate

T2 executed on the 4090 (CUDA posture `gemma2_d2f,cuda_backend`, detached runs, GPU 44 °C / 5 % util, no compute consumers).
Commits: `b2fe307`.

## 2026-10-26 — Issue 1119 CLOSED + removed: the zero-copy cache-split handle gap — fix `7b2fed9`, acceptance = the i1122 gate1 cell re-run

The riir-ai Issue 1122 gate1 cell (`gate1_site_isolated_tax`) ran bit-identical simdgroup code with 0 site-timing rows and n=0 site dispatches — the G1-instrument illusion class.
Commits: `7b2fed9`.

## 2026-10-25 — Issue 1115 CLOSED + removed: the GEMV occupancy wg rule default-on EVERYWHERE (the 4090 CudaRuntime A/B measured the platform split away)

Bench 887: prefill-tail lm_head A/B on the 4090 CudaRuntime — G1 FNV-identical ×6+6 rounds (pp512 + pp4096, Bonsai-2 vocab 248320 → the 128-wg bucket), reach exact (+1/±0 across 12/12 rounds via the counter), G3 1.0013×/1.0000× sub-noise → per the issue's own done-when …
Commits: `f4e6369c`.

## 2026-10-26 — Issue 015 CLOSED + removed: the silero-vad audio lane PoC — T6 verdict finalized (quiet-ish-box latency re-run, replicated ×2)

The PoC itself LANDED 2026-10-23 (T0–T3: `vad-ane` feature on `riir-infer-audio`, digest-walled loads both postures via `riir-infer-ane`'s `load_model_precompiled` + `predict_f32_multi`, 5-test gate suite green, the ANE finding — all-CPU placement on M5 Ultra/macOS 27, strict ANE gate unusable as a …)

## 2026-10-26 — Issue 1123 CLOSED + removed: the issue028 dual-format q4-prefill gate red on develop — the TEST was stale, not the loader (the TernaryQ2 promotion outran the battery)

Filed 2026-10-26 from the plan-623 Phase 1 validation sweep: `dual_format_q4_prefill_pair_loads_and_runs` tripped `matches!(dec.layers[0].in_proj_qkv, ProjWeights::Ternary(_))` at pristine `7a01f05` (filer's detached-worktree run verified it was NOT a 623 regression).
Commits: `5ad32ba`, `c787460`, `11a30c6`.

## 2026-10-26 — Issue 1125 CLOSED negative-already-shipped + removed: the Bench-1004 fused-quantize rung — the #395 mechanism IS the shipped K1–K5 posture; the census falsifier discharges it dynamically (0 standalone quantize launches on the 4090)

Filed from riir-ai Bench 1004 (the 0.9730× decode RED, first sub-parity decode cell of the scored era; the opponent's +3.84% jump rode prism b10826's #395/#401/#402 FWHT/q8_1 fusion quartet).
Commits: `a2ebb37`, `2e3889e`, `1320dda`.

## 2026-10-26 — Issue 1126 CLOSED measured-negative + removed: the GDN core fusion rung (Bench 898 −3.5%/−2.1%, G1 exact everywhere, opt-in `RIIR_GDN_CORE_FUSED`) and its named 4→3 follow-up (Bench 899 direction-positive 3/3 pairs, opt-in `RIIR_GDN_BETA_EXPAND_FUSED` — NOT promoted, league re-cell owner-gated)

The #398 conv-chain twin implemented as `recurrence_f32_fused_expand_hd{64,128,256}` — ONE launch replacing beta_decay + expand_and_l2_normalize + recurrence_fused, the Issue-742 body shared verbatim via `recurrence_fused_body_impl<HD, EXPAND_INLINE>` (`4132d94`).
Commits: `8aabe55`.

## 2026-10-26 — Issue 1127 T1 LANDED (Bench 900): the attention-chain fusion rung — 5→1 launches, G1 bit-identical incl. the graph lane, G2 4/5 paired windows positive (armed median +0.77%), opt-in `RIIR_ATTN_CHAIN_FUSED` — + Issue 1131 filed: the GDN rotation FWHT geometry rung (the census rows outside T3's family)

The T1 rung (`attn_chain_fused_f32` + the Issue-618 devpos twin for the graph-capture lane, `8d51f21`+`1d0291d`) merges split_qg + q/k-norm + RoPE + kv-append into ONE launch per attention layer — every body verbatim, ordering confined to block-local barriers, read-barrier-write rope staging …
Commits: `687f5669`, `63617c9`, `86e419d`.

## 2026-10-28 — Issue 1147 CLOSED + removed: the T2.4 kernel micro-batching rung — fused-batch v0 LANDED (`03f0fe6`, Bench 912) then DEMOTED same-day on the 2-box wire refutation (Bench 913: armed 46.9–52.6 vs incumbent 99.2–108.5 aggregate decode window, −52% — the BatchN wave forfeits the head⇄tail GPU overlap that IS the 2.37× bubble fill)

`pair_decode` ships default-OFF (`RIIR_DIST_PAIR=1` arms it for loopback rigs only, where the +39% was real and stands).

## 2026-10-28 — Issue 1130 CLOSED G2-FAIL + removed: the Metal concat-epilogue split rung (folded GDN qkv|z Split2 into the ternary concat GEMV epilogue)

Rung `b635a43` (`gemv_ternary_plane_rowtiled8_f16scale_split` + `GemvTernaryCubeCL::launch_split`); G1 to_bits EXACT 13/13 on amo + m3 (`9f1b2ba`).
Commits: `445cc4e5a`, `760265d`.

## 2026-10-28 — Issue 1131 CLOSED + removed: the T6 GDN rotation-FWHT geometry rung (CUDA) re-priced sub-bar, re-pin adjudicated NO-GO

Filed `86e419d` (Bench 902 rows `gdn_input_rotabq` + `gdn_out_norm_gate_permute_rotq`, 564 µs/token).
Commits: `8d204ba`, `194b952`.

## 2026-10-28 — Issue 1132 CENSUS COMPLETE + removed: the Metal per-dispatch floor census (whole-graph launch attribution on the folded decode)

Instrument `3d95ff9`: `DECODE_STAGE_MASK` 11→22 bits (per-launch-site refinement classes + the previously-ungated decode tail; mask=ALL bit-identical) + riir-ai `b864dc936` Phase E — the census rides the stage-instrument plumbing, no xctrace.
Commits: `772c8bf`, `1a28511`, `800d7b4`, `c38d28f`.

## 2026-10-28 — Issue 1133 CLOSED measured-null + removed: GDN conv+beta/decay+expand merge rung (row above, 2026-10-27)

Rung `0eaebad`, record Bench 905 (`.benchmarks/905_i1133_conv_merge_ab.md`): G1 exact everywhere, G2 2/4 pairs, pooled ≈0% ± 0.5%; ships opt-in `RIIR_GDN_CONV_MERGE=1`, default OFF.
Commits: `537ca18`.

## 2026-10-28 — Issue 1134 CLOSED G2-FAIL + removed: the norm-pair epilogue fold rung (input_norm + mid_norm)

Donated kernel `ResidualAddRmsNormRotateCubeCL` (`cbb5bf1`; G1 to_bits 3/3 + REACH PASS) is an e2e LOSS in 7/7 A/B runs (riir-ai `bench_1137_normrot_e2e_ab`): −2.96/−2.23 … −8.90/−9.48%.
Commits: `b02f379`, `4aed461`, `f44c740`, `ffb07bf`.

## 2026-10-28 — Issue 1135 CLOSED ceiling-gate NULL + removed: the attention proj+decode launch-cluster rung

Three verified-idle interleaved windows on amo (riir-ai `bench_1138_attn_proj_decode_strip_ceiling` + raw log `.benchmarks/1138_issue1135_attn_cluster_strip_ceiling_amo.raw.log`): proj 32 la/tok × 13.4-20.5 µs (pool 0.43-0.66 ms/tok); the only legal shape (648-F9 wq+wkv row-concat, −16 la/tok) …
Commits: `eafd97d`.

## 2026-10-28 — Issue 1136 CLOSED + removed: the gdn_out_gemv rung — ceiling real, G1 structurally dead; GDN chain launch budget CLOSED BY MEASUREMENT

Ceiling (riir-ai `bench_1139_gdn_outproj_strip_ceiling`, raw log `.benchmarks/1139_issue1136_gdn_outproj_strip_ceiling_amo.raw.log`): +0.524/+0.592 ms/tok (+2.16/+2.46%) @ 48 la/tok. Commits: `afc1f43`.

## 2026-10-28 — Issue 1137 CLOSED NULL + removed: the FFN decode-GEMV ALU-chain rung (and its hand-MSL lane i1138, Bench 1149)

Fork mapping study (`0cc47fa`: `kernel_mul_mv_tq2_0_f32`, PR-26980 algebra) found its in-tree twin `gemv_ternary_fma_cubecl` (Plan 563 / Bench 847, `RIIR_GEMV_FMA` default-OFF); riir-ai `bench_1141_fma_aspect_probe` on M5 Ultra: FMA loses ~3× on every aspect (gate/up 156.8 vs 47.9 µs, down 178.3 vs …)
Commits: `6277cd2`, `194b952`; lane i1138 closed MEASURED-NEGATIVE in riir-ai `.benchmarks/1149_i1138_field_e2e_ab_amo.md`.

## 2026-10-31 — Issue 1158 CLOSED + removed: mask-input drafter refinement audit (JetSpec++ pass-1 collapse class)

Inventory of every `mask_token_id` reader, two classes, never pooled. **HAZARD (mask-input block drafters, single-pass J0 by contract):** riir-infer `qwen38_dflash2.rs` (`draft_block_hidden` input = anchor + all-mask; `lattice_walk_reselect` re-scores the J0 grid, no re-forward) + `speculative_decode/dspark_drafter.rs`; riir-ai `riir-gpu/src/dflash_config.rs` + DFlash2 consumers (benches 746/747/759/779/989, plan614 lanes/spec/floorgap/premise_966, plan632, issue1142, riir-engine bench_694); riir-train `dflash_trainer.rs`, `dataloader.rs::batches_dflash` (THE fully-noised recipe), `dflash2_chat_head_train.rs`, `chat_dflash2_probe`. No refinement-pass knob exists anywhere in this class today. **NOT the class (dLLM/D2F multi-step denoisers, trained on partial ratios — J≥1 inputs in distribution):** katgpt-forward `d2f`, riir-infer `gemma2_d2f*` + `issue1110_reopen_probe`, riir-train `dllm/*`, `distill_tsd/gdsd/harness`, `opsd_loop` (the filing's trainer list misfiled these three — `noise_min/max_ratio` corruption), plan357/389/409/410/433/437 rigs, `bench_dllm_kernels`, riir-ai `goat_250`; katgpt-rs `funcattn_g6_*` are MLM tests. Guards (doc-only, name this issue): `draft_block_hidden`, DSpark noise-block fill, `DFlashConfig.mask_token_id`, `batches_dflash`; cliff detector pinned at the verify seam (`speculative_decode/mod.rs` — J1 τ below J0 ⇒ refuse the arm). Fix vehicle unchanged: riir-train Plan 376 T3. Commits: riir-infer `1f85c99`, riir-ai `e6a2cd1e0`, riir-train `bcd9e30e`.

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
- A tail-read of never-written buffer regions is only a valid bounds check if the buffer STARTS zero on EVERY runtime — `client.empty()` promises nothing on the CUDA runtime (Issue 039).
- Cosine/state redundancy ≠ functional redundancy, on ternary and dense alike (Issue 022 T5.0/T5.1).
- Weight-space dB wins mean nothing without the model-level gate — the class verdict can kill every arm (Issue 027).
- Windows batch: no literal `)` in echoes inside `( ... )` blocks; never hand-launch a wait-loop `.cmd` via `Start-Process` from MSYS; prefer `schtasks` wrappers.
- Feature-gated targets carry their `required-features` row at birth — a whole-file `#![cfg]` target without one prints a green zero forever (the T1.1e law).
- `[patch.crates-io]` does not survive crates.io publication — `publish = false` until the vendor forks land upstream.
