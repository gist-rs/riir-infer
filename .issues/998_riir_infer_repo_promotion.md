# Issue 998 — riir-infer repo promotion: the riir-ai mirror (Proposal 041 T-B fired by riir-reflex Issue 008)

> **Moved 2026-09-26:** this file lives in `riir-infer/.issues/` now (moved verbatim from riir-ai, same number — riir-ai's ledger keeps 998 allocated). Prose paths below are written from the riir-ai side at writing time; **`../riir-infer` means THIS repo**.

**Status:** OPEN — P0 contract landed with this issue; T2 audit recorded;
below; T3 carve (P1) executes in the same window. **P4/T6 EXECUTED
2026-09-23 — both repos PUBLIC** (riir-infer + reflex; owner green-light;
record at the end of this file). The 16:4x HEAD↔HEAD default-lane break
(the squash dropped the slice-5 wiring block) was **RESOLVED same day** —
fix landed as riir-infer `d058f48`, and the consumer lane is verified
green at HEAD↔HEAD (`riir-ai cargo check --workspace` rc=0, 2026-09-23
post-landing, M3). Owner directive
2026-09-22 (recorded verbatim in riir-reflex `.issues/008`): new repo
**`riir-infer`** consolidates the LLM-inference substrate — riir-infer-core
plus the CLEAN GPU kernel layer plus (P2) reflex's laya/MSL encoder lane —
public, consumed by BOTH riir-ai and riir-reflex.

This is the riir-ai-side half. The campaign plan, verdict, and phase
ordering live in **riir-reflex `.issues/008_riir_infer_consolidation.md`**
(the executing plan); this mirror owns the riir-ai contract rows, the T2
audit table, and the riir-ai-side gates. Proposal 041's §Session 5
**T-B trigger fired 2026-09-22** (a repo outside riir-ai needs
riir-infer-core without riir-engine): Phase 2's deferral is discharged, the
promotion runbook is 041's T2.1–T2.4 extended by 008's P2/P3.

## P0 — the contract rows landing with this issue

1. **BOUNDARY.md §Owns** — the `riir-infer-core` bullet now records the
   carve-out: the crate leaves this workspace for `../riir-infer`
   (name-unchanged per 041 §Session 5 Q2 — no shim, no rename); this
   contract's ownership of it transfers to riir-infer's own BOUNDARY.md.
2. **BOUNDARY.md §Does not own** — `../riir-infer` row added.
3. **BOUNDARY.md §May depend on** — `riir-infer-core | ../riir-infer |
   riir-engine non-optional (the T1.2 same-path re-exports keep resolving;
   feature forwards unchanged — they are package-name-keyed, not
   path-keyed)`.
4. **BOUNDARY.md §CANONICAL matrix** — `riir-infer` row: **none** INTO
   riir-ai (upstream-clean: katgpt-rs path deps only); riir-ai's
   riir-engine consumes it — the first workspace repo that is UPSTREAM of
   the engine by design. Post-P3 shape: riir-gpu's residue also depends on
   it (`riir-gpu ─► riir-infer`, 041's own dep-direction row).
5. **BOUNDARY.md D4 drift row** — annotated: T-B fired; the widening
   re-narrowing rides P3/T7 (reflex Issue 008), never before.
6. **Research 003 amendment** — dated note in `.research/003` (below) +
   reflex BOUNDARY's "Private forever" amended in the same stroke.
7. **`repo_set.txt` + the pin ripple** — lands in the same window the repo
   appears on disk (katgpt-rs side; see T3).

## The Research 003 amendment (recorded, dated, owner authority)

The 2026-09-22 owner directive is the authority (record it, don't re-ask):
`riir-infer` and `riir-reflex` are the **first sanctioned exceptions** to
Research 003's "anything riir-* is internal, no exceptions" rule. The
opening itself (P4/T6) remains owner-gated go, behind T5's fence gate +
sanitized-docs checklist; until T6 executes, both repos stay private with
`publish = false` intact. Net-new IP exposure was adjudicated LOW in
riir-reflex Issue 008 §verdict (the ternary/modelless approach is already
public via katgpt-rs; quant formats are GGUF-standard; architectures are
public models; the moat — product/chain/tokenomics/cognition — is not in
the move-set). Weights stay private (riir-train checkpoints never move;
riir-infer ships loaders, not weights).

## T2 — the riir-gpu CLEAN / SEAM / STAYS audit (measured 2026-09-22)

Method: comment-stripped grep + per-line verification of every
fully-qualified ref (raw `grep riir_engine` overcounts — 3 modules carry
comment-only engine refs). Population = the **171 modules** declared in
`riir-gpu/src/lib.rs` (152 files + 19 dirs). lib.rs is wiring, covered in
the residue note.

| class | count |
|---|---|
| **CLEAN** (moves verbatim) | **98** |
| **SEAM** (moves with `riir_engine::`→`riir_infer_core::` rewrite) | **34** |
| **STAYS** (residue) | **39** |
| UNRESOLVED | 0 (⚠ flags name human-pass questions, not unknowns) |

Never-move list fully fenced: `game`, `game_mux`, `weaver_gpu`,
`weaver_gpu_corrector`, `weaver_gpu_dflash`, `gpu_thoughtfold`,
`memory_soup_gpu`, `rosetta_gpu`, `moa`, `spec_marketplace` — all STAYS.
**Measured dead deps from riir-gpu's side:** `riir-router` (optional, zero
src references) and `riir-gpu-async` (non-optional, zero references in
src/tests/benches/examples — its real consumers are
`riir-engine/analytic_lattice` and itself).

### Verified T1.2 re-export list (engine `src/lib.rs` — 14 named full-module re-exports, no globs)

`dflash, gemma_layer, gguf_loader, llama_layer, ternary_layer*,
quant, rope, safetensors_loader, simd, spec_types, transformer, types,
wall, deltanet*` (* = feature-gated: `ternary_inference`,
`deltanet_inference`). The engine declares no shadowing local module for
any of these names, so every `riir_engine::<X>` path that compiles today
physically resolves to `riir_infer_core` — the SEAM class is exactly the
modules whose engine imports all sit in this set. Caveat: the
`deltanet`→`riir_infer_core::deltanet` rewrite must enable infer-core's
`deltanet_inference` feature.

### CLEAN (98)

| module | note |
|---|---|
| buffer, pool_poison, context, weight_buffer_cache, gpu_transpose, cubecl_runtime | core buffer/runtime helpers |
| ane_prefill, browser_gpu_init, cpu_reference, gemv_batched, mla_cubecl, moe_cubecl, kda_cubecl, kimi_k3_gpu_forward, forward_sp_kv, domino_gpu, forward_flashprefill, forward_turboquant, spectralquant, maxsim, dflash_config, lora_fused_cubecl, gemv_autotune, gemv_cubecl, weight_readback, test_gpu_support, t916_compile_smoke, qwen38_kv_f16_gates | katgpt + crates.io only |
| matmul_cubecl, matmul_swap_ab_cubecl, attention_cubecl, gemma2_d2f_sc, attention_causal_fused_cubecl | attention/matmul core |
| gemv_ternary_cubecl, gemv_ternary_scale_ab_cubecl, gemv_ternary_fma_cubecl, gemv_ternary_residual_cubecl, gemv_ternary_block_contiguous_cubecl, gemv_ternary_cuda_raw | ternary gemv family |
| gemm_ternary_batched_cubecl, gemm_ternary_tiled_cubecl, gemm_ternary_cmma16_cubecl, gemm_ternary_cmma_i8_cubecl, gemm_ternary_cmma_i8_direct_cubecl, gemm_ternary_cmma_i8_t64_cubecl, gemm_ternary_cmma_i8_psplit_cubecl, gemm_ternary_simdgroup_cubecl, gemm_ternary_block_contiguous_cubecl | ternary cubecl gemm family |
| gemm_ternary_metal_tensor, gemm_ternary_metal_wgpu, gemm_ternary_metal_zero_copy | macOS-gated |
| qwen38_verify_mma, qwen38_prefix_cache, qwen38_dflash2 | qwen38 family |
| gemm_ternary_i8_mma_cuda_raw, gemm_ternary_i8_mma_v6_src, prefill_cuda_mma, prefill_cuda_ffn, prefill_cuda_deltanet, prefill_cuda_attention, prefill_cuda_attention_vec, prefill_cuda_attention_gang, prefill_cuda_attention_fa, prefill_cuda_gdn_chunked, ternary_ffn_fused | CUDA raw family |
| gemv_f16_cubecl, norms_cubecl (the RMSNorm family — no mean-centered variant, confirmed), sampling_cubecl, elementwise_cubecl, gemv_geglu_cubecl, gemv_geglu_f16_cubecl, gemv_qkv_f16_cubecl, epilogue | elementwise/norm family |
| deltanet_chunked_cubecl, deltanet_delta_rule_chunked, deltanet_tree_verify_cubecl, deltanet_input_proj_fused, deltanet_pre_rec_fused_cubecl | deltanet kernels |
| qwen_attention_cubecl, qwen_attention_prefill_m16/m32/m64/m32_pipe/m32_kvf16/cmma/cmma_pv_cubecl, qwen_prefill_q8kv_cubecl | qwen attention family |
| wall_cubecl, wall_lora | wall kernels |
| kernels ⚠ | WGSL zoo shared by inference AND training/game consumers (`loss`, `gpu_thoughtfold`, `memory_soup_gpu`, `oscpart`, `lora/tests`) — share or split at P3 |
| wall ⚠ | one cfg(`mux_training`) edge → `mux_wall_weights` (STAYS) — strip on move |
| segment_cache ⚠ | zero deps, zero in-tree consumers found; Plan 254 (training) lineage — verify before moving |
| spec_adapter, spec_data_gen, spec_quant, spec_hybrid_router ⚠ | same feature family as the fenced `spec_marketplace` — adjudicate at P3 |
| issue832_float_order_tests | `#[cfg(test)]` |

### SEAM (34) — engine imports all resolve through the T1.2 re-exports

| module | engine imports (abbrev) | note |
|---|---|---|
| speculative_decode | `gguf_loader::GgufFile` | ngram_drafter's refs are `#[cfg(test)]`-only; tokenizer is engine-local → re-point tests on move |
| forward | `deltanet`, `transformer`, `types`, `llama_layer` | ⚠ cfg-gated adapter slots → STAYS modules `moa`/`oft`/`oscpart`/`speft` — strip or feature-forward |
| set_diffusion_decoder | `transformer`, `types` | inference |
| gemma2_forward | `gemma_layer`, `gguf_loader`, `transformer`, `types` | ⚠ imports `crate::gemma2_cubecl` (STAYS-cand) — the gemma-cluster decision |
| gemma2_q4k_weights | `gemma_layer`, `gguf_loader`, `quant`, `types` | |
| lora | `types` (mod), `transformer` (tests) | dual-use: LoRA load/export (inference + rtg training) |
| forward_prefill, dflash_context | `types` | |
| forward_target_extract, forward_dflash, forward_dflare | `transformer`, `types` | ⚠ `dflash_training`/`dflare_training` features (draft-training forward) |
| vram_budget | `deltanet`, `types` | |
| state_readback | `types` | |
| gemma2_d2f | `gemma_layer`, `types` | ⚠ imports `crate::gemma2_cubecl` |
| gemma4_cubecl | `transformer`, `types` | |
| llama_cubecl | `llama_layer`, `transformer`, `types` | ⚠ real `crate::gemma2_cubecl::{CpuKVCache, GpuKVCache, apply_rope, rmsnorm_gamma, KvStoreCubeCL}` imports |
| gemv_q4k_cubecl, gemv_q4k_batched_cubecl, gemv_q4k_batched_rmsnorm_cubecl, gemv_qkv_q4k_cubecl, gemv_geglu_q4k_cubecl, attention_q8kv_cubecl | `quant` | |
| qwen38_dense_cudarc, qwen38_dflash2_gpu | `gguf_loader` | |
| prefill_cuda_full | `types` | |
| cudarc_kernels | `deltanet::qv_lora::QvLora`, `types::Rng` | ⚠ `backward.rs` serves rtg training |
| ternary_deltanet_gpu_forward | `deltanet`, `types` | ⚠ un-cfg'd `use crate::training_activation_cache` (L179, STAYS) — P3 must handle |
| deltanet_rotation_cubecl, deltanet_rotation_cudarc | `deltanet` | |
| ternary_deltanet_gpu_forward_cudarc | `deltanet`, `types` | |
| deltanet_cubecl | `deltanet`, `types` | types-only-by-design gate documented in its lib.rs row |
| ternary_tree_verify_driver, hybrid_dispatch | `deltanet`/`types` | |
| rope_geglu_cubecl | `rope` (test-block) | |

### STAYS (39)

| module | reason |
|---|---|
| weaver_gpu, weaver_gpu_corrector, weaver_gpu_dflash, gpu_thoughtfold, memory_soup_gpu, rosetta_gpu, moa, game, game_mux, spec_marketplace, thoughtfold_types | never-move list (a) / tied to it |
| linoss_cubecl, linoss_dispatch | (b) `riir_engine::linoss` is engine-local, not re-exported |
| lora_still_forward | (b) `lora_still` engine-local |
| wall_mla, wall_decode | (b) `wall_config::WallConfig` engine-local — re-homing WallConfig unlocks the gemma cluster |
| gemma2_cubecl | (b) ONE engine-local import (`wall_config`, L87) forces STAYS — **nearly-SEAM**; the gemma-cluster hinge |
| memory_soup_gpu | (a)+(b) `adapters` engine-local |
| hypernet, speft, depth_tier, oscpart, oft, replaid, elf, domain_latent, config, training_activation_cache, ropd, rim, rim_routing, loss, loss_mux, loss_mux_wgsl, mux_config, mux_adaptive_span, mux_wall_weights, sdpg, kvarn, spectral_adaptive | (d) training/game/MUX purpose — stays as residue |

### Residue wiring (P3 grounds)

- lib.rs carries the 171 declarations + ~150 root `pub use` re-exports and
  exactly ONE real engine import (`deltanet::minimal_activation_cache` —
  itself re-export-set). At P3 the moved modules' root re-exports become
  `pub use riir_infer::<mod>::…` (the transition re-export).
- **STAYS→moving edges** (residue depends on riir-infer + re-exports):
  `weaver_gpu`→`gemv_cubecl`; `gpu_thoughtfold`/`memory_soup_gpu`/`hypernet`/`oscpart`/`loss`→`buffer`/`context`/`kernels`; `oft`→`lora`.
- **Moving→STAYS edges** (strip/cfg-forward/adjudicate at P3): `forward`'s
  adapter slots (`moa`/`oft`/`oscpart`/`speft`); `lora/ia3.rs`→`moa`;
  `wall`→`mux_wall_weights`; the gemma cluster (`llama_cubecl`,
  `gemma2_d2f`, `gemma2_forward`)→`gemma2_cubecl`;
  `ternary_deltanet_gpu_forward`→`training_activation_cache`.
- **riir-train-gpu back-reach** (pub-mod paths documented in lib.rs):
  moving — `forward`, `kernels`, `buffer`, `lora`, `mla/moe/kda`,
  `kimi_k3_gpu_forward`, `forward_flashprefill`; staying — `loss`,
  `speft`, `config`, `replaid`, `elf`, `domain_latent`, `depth_tier`,
  `thoughtfold_types`, `mux_config`, `loss_mux`. The riir-gpu re-export
  layer preserves all of these paths — no rtg edit required at P3.

## Tasks (riir-ai side)

- [x] T1 P0 contract rows (this issue + BOUNDARY amendments + the 003
      amendment note) — landed with the carve window.
- [x] T2 audit recorded above (subagent-measured, comment-stripped grep,
      171/171 modules classified, 0 unresolved).
- [x] T3 carve verification (P1): repo `../riir-infer` exists with the
      moved crate (gist-rs/riir-infer @ 86a5986, fresh history); this
      workspace green (`cargo check --workspace` FINISHED post-carve);
      `cargo tree` in the new repo proves zero `riir-*` deps; engine T1.2
      re-exports compile clean (no flip needed at P1 — `cargo check -p
      riir-engine --lib` green in 10.9s); Cargo.lock byte-identical (path
      deps record no source in the lock); the two guard surfaces that
      named the crate by `-p` re-pointed cross-repo with loud sibling-absent
      skips (ci_feature_guard Layers 1.6/1.7 — the flashmemory surfaces
      measured 5 passed from the new home; ci_modelless_residue scans
      `../riir-infer` — VERDICT clean); boundary contract clean at 23
      repos; registered as the 23rd contract repo (katgpt-rs bd2ce3cca);
      4090 synced (E:/git/riir-infer via bundle — GitHub clone auth hangs
      from that box).
- [ ] T7/P3 riders (deferred by design, execute with P3 — never here):
      T2.3 consumer retarget (riir-train + riir-clippy path deps →
      `../riir-infer`); D4 widening re-narrowing (per 041's D4 discharge,
      now riding reflex Issue 008 P3/T7).
      ⚑ T2.3 MEASURED DISCHARGED 2026-09-23: neither consumer carries a
      direct `riir-infer*` path dep (comment-stripped manifest grep — both
      consume transitively through riir-engine/riir-gpu, retargeted at T3),
      so the wording reduces to CI sibling provisioning, and BOTH halves
      are landed: riir-clippy `b69a47e4` there (rust.yml loop + MSG;
      release.yml's loops also gained the kat-plane siblings they had been
      missing since the 09-10 issue-088 riir-kat root dep — pre-existing,
      would have died at workspace load on the next v* tag) and riir-train
      `3ce79e23` there (both loops, carve slice 1). What remains of this
      row: D4 re-narrowing only.
- [x] T8 P3 slice 1 (the base GPU runtime cluster) — LANDED 2026-09-23
      (Plan 610, riir-ai): `buffer`/`context`/`pool_poison`/
      `weight_buffer_cache`/`gpu_transpose`/`cubecl_runtime` + the
      adapter VRAM probe moved to `crates/riir-infer-gpu` in
      `../riir-infer`; riir-gpu re-exports (cfg-matched) — zero
      consumer edits, 426/0 lib suite, clippy clean both postures;
      vendor forks (`cubecl-runtime` #1359 fix + `wgpu-hal` accessors)
      copied byte-identical with `[patch.crates-io]` in the new
      workspace root. T5 (fence gate + CI + opening checklist) landed
      in the same window — see reflex Issue 008 T5.
- [x] T8 P3 slice 2 (the elementwise/norm/matmul/attention family) —
      LANDED 2026-09-23 (Plan 610 S2, riir-ai): 28 files moved
      (cpu_reference, the GEMV family incl. the q4k/qkv quant kernels,
      matmul + swap_ab, attention flash/causal-fused/q8kv + tests,
      gemma2_d2f_sc, norms/sampling/elementwise, epilogue,
      params_cache); the five quant SEAM imports became
      `riir_infer_core::quant`; six features mirrored + forwarded;
      riir-infer-gpu gained the `riir-infer-core` path dep +
      half/papaya/blake3; the fence gate's F1 path-dep check fixed to
      real containment (the `../..` member dep would have red the
      slice by construction) + selftest arms. Zero consumer edits;
      riir-infer-gpu 206/0 at the full combo; riir-gpu 283/0 (gemma2
      tests exercise the moved kernels through the re-exports on
      Metal). Found pre-existing bench_603 all-features rot (S3
      territory, verified by stash test); one forced divergence
      recorded (ArgmaxCubeCL pub(crate)->pub in the new copy).
- [x] T8 P3 slice 3 (the ternary gemv/gemm + metal + CUDA-raw
      families) — LANDED 2026-09-23 (Plan 610 S3, riir-ai): 33 files
      moved (31 .rs + 2 .metal) + the bench_663_t5_single_gemm_isolation
      test with its kernel; 9 pub(crate)->pub widenings for the
      remaining SEAM consumers (recorded in the plan);
      canonical_expand_forms moved with its kernel family; 10 features
      mirrored + forwarded (ternary_gemv drops the riir-engine leg
      infer-side — the fence); katgpt-core joins riir-infer-gpu via a
      new [workspace.dependencies] table (member manifests never spell
      the deeper sibling path — fence raw-prefix check stays green);
      the #[path] simdgroup_smem child moved with its parent (audit-gap
      note in the plan). bench_603 rot FIXED (Issue-980
      GateProjWeights ripple; pre-existing at HEAD, stash-verified) +
      the same rot class fixed in riir-train-engine's bonsai-go lane
      (f5bfe3f9, 1691/0). riir-infer-gpu 225/0 all-features (Metal);
      riir-gpu 267/0 default; the all-features lib suite is
      jetsam-killed on the loaded M3 AT HEAD TOO (stash-verified,
      box-state not slice) — the gemma2_d2f re-export-path tests pass
      individually (4/4). Commits: riir-infer `0c2f3b4`, riir-ai
      `027aef2d6`, riir-train `f5bfe3f9`. S4 ready-notes
      pre-adjudicated in plan 610 (the qwen38/cudarc SEAM cluster + the
      ternary_deltanet_forward cluster + the deltanet CLEAN kernels).

- [x] T8 P3 slice 4a (the S4 dependency closure: qwen attention family +
      deltanet CLEAN kernels) — LANDED 2026-09-23 (Plan 610 S4a, riir-ai,
      executed on the 4090 box — the only box that compiles the
      not(macos) CUDA arms): 12 files / ~15.4k LOC moved
      (qwen_attention_cubecl + 8 prefill arms + qwen_prefill_q8kv +
      deltanet_chunked/delta_rule_chunked/tree_verify); 1 visibility
      widening (Q8PrefillScratch fields+buffer_bytes — STAYS consumer
      constructs cross-crate); 4 feature mirrors + forwards (+ the
      katgpt-core/gdn_tree_verify leg for the moved tree-verify oracle
      tests). Riders: S3's dead canonical_expand_forms wrapper +
      canonical_recurrence_forms test paths fixed (the not-macos blind
      spot — first-ever CUDA-family lib-test compile); S1's
      cubecl_runtime doc-lint fixed; fence_gate.py Windows selftest
      path-separator bug fixed (the gate now runs on the CUDA box);
      S1-latent weight_buffer_cache cudarc-posture test failure RECORDED
      (needs the cudarc extract path investigated — separate issue).
      Validation: fence gate PASSED (284 .rs); infer-gpu 206/0 at the S4a
      combo (moved kernels GPU-validated from the new home); riir-gpu
      325/0 default; engine + train-gpu + clippy checks green.
- [x] T8 P3 slice 4b (the S4 SEAM core: qwen38/cudarc cluster + the
      deltanet-forward family) — LANDED 2026-09-23 (Plan 610 S4b, the 4090
      box): 31 files / ~45k LOC moved, + the ane_prefill rider (7 files,
      CLEAN) + vram_budget/state_readback (audit-table gaps coupled to the
      forward). The two S4 adjudications RESOLVED: backward.rs →
      riir-train-gpu (training surface, the C6 remedy; the forward's
      backward field stripped, ctx widened pub, train-gpu holds
      TrainingForwardCudarc with Deref); the L179 training_activation_cache
      pair stripped (zero external callers). The T2.3 riir-clippy ternary
      bench retarget is UNBLOCKED. Riders: 3 gate tightenings (latent
      couplings), the E0004 unification trap routed through infer-core's
      features, 7 feature mirrors, 2 visibility widenings,
      qwen38_kv_f16_gates stayed home (test-only file can't live in a dep),
      Issue-980 rot fixed at 2 never-compiled lanes, BOUNDARY gpu-dep row
      widened. Validation: infer-gpu clippy at 5 postures + lib tests
      379/1 serialized at all-features (the 1 = Issue 999); fence PASSED
      (314 .rs); riir-gpu 236/0 default; train-gpu 407/0 at the backward
      posture; boundary 0 violations.
- [x] T8 P3 slice 5 (the gemma kernel cluster) — LANDED 2026-09-23 (Plan
      610 S5, the 4090 box): 17 files / ~13.6k LOC moved (gemma2_cubecl/,
      llama, gemma2_d2f/, gemma2_q4k_weights, gemma4_cubecl/ rider,
      rope_geglu_cubecl + wall_cubecl riders, test_gpu_support COPIED) +
      the wall_config hinge re-homed to riir-infer-core (a de-fork re-export
      module — engine keeps the same-path re-export, zero consumer edits).
      gemma2_forward + forward_prefill + forward_flashprefill HELD (the WGSL
      zoo split is its own slice — training WGSL can never move to the
      public inference repo). gpu_training_resident MIRRORED (generic
      launches; test-only consumer). Riders: 9 feature mirrors (gemma2_d2f
      widened to the whole module), log+fastrand deps, the S4b-latent
      deltanet_pre_rec test-arm gate rot fixed, the delta_routing tests
      fixture gated, the S2/S3/S4b orphan residue cleaned (26 dead source
      copies), Issue 1001 filed (pre-existing d2f convergence failure,
      stash-verified at the old home). Validation: infer-gpu clippy at 4
      postures + tests-compile at 3; lib tests 269/271 at the gemma combo
      (the 1 = Issue 1001) + all-features 449/451 post-merge (the 2 = Issues
      999 + 1001, targeted-verified — no new failures); riir-gpu clippy
      default+all-features all-targets + 199/0/1 default lib; engine checks;
      train-gpu check. The repo-level twin landing reconciled at riir-infer
      fa69d01 + the Plan-611 split2 test compile repaired (6811b3f).

## Fence conditions (from reflex Issue 008 — each load-bearing)

Fresh git history at the carve commit (no workspace narrative in the new
repo's messages/docs); sanitized public docs + fence gate (T5) before any
opening (T6, owner-gated go); `repo_set.txt` + pin ripple lands with the
carve; weights never move; reflex→riir-infer allowed (reflex BOUNDARY
amended — riir-infer ONLY, never riir-ai: the layering
reflex → riir-infer → katgpt-core stays all-public).

## 2026-09-23 16:4x +07 — ⛔ cross-repo default-lane break at HEAD↔HEAD (the P4-opening squash dropped the slice-5 wiring; fix is riir-infer-side)

Found by the idle Protocol B sweep (4090, 16:14–16:27, both repos clean +
synced at origin/develop: riir-ai `175f245802` ↔ riir-infer `eb3438f`).
UNRESOLVED as of this note — the riir-infer session's P4 opening checklist
closed green on validations that predate the squash.

**Symptom (riir-ai side):** `cargo clippy --workspace --all-targets` at
default features fails with 4×E0432 in `crates/riir-gpu/src/lib.rs`:

```
crates\riir-gpu\src\lib.rs:185:9: error[E0432]: unresolved import `riir_infer_gpu::gemma2_q4k_weights`
crates\riir-gpu\src\lib.rs:629:9: error[E0432]: unresolved import `riir_infer_gpu::gemma2_cubecl`
crates\riir-gpu\src\lib.rs:657:9: error[E0432]: unresolved import `riir_infer_gpu::llama_cubecl`
crates\riir-gpu\src\lib.rs:1369:9: error[E0432]: unresolved import `riir_infer_gpu::rope_geglu_cubecl`
```

Line 185 is UNCONDITIONAL (the gemma2_forward `crate::` re-export hinge);
the other three are `cubecl_runtime`-gated but that feature is unified ON
in the default workspace graph — so the AGENTS.md baseline
`cargo check --workspace` is broken at HEAD↔HEAD. Blast radius beyond
riir-ai: every consumer that compiles riir-gpu with the graph unified —
riir-train's engine/gpu lanes and riir-clippy's `ternary_inference`
(feature-gated) arm both path-escape into `../riir-infer` post-carve.

**Root cause (riir-infer side):** the P4 public-opening history squash
(`eea8092`, "history squashed to four sanitized commits (force-pushed with
lease)") mis-resolved `crates/riir-infer-gpu/src/lib.rs` and dropped the
46-line P3-slice-5 gemma-cluster wiring block that `6b6de2b` had added at
old lines 578–623: `pub mod gemma2_cubecl / gemma2_q4k_weights /
llama_cubecl / gemma2_d2f / gemma4_cubecl / rope_geglu_cubecl /
wall_cubecl` + the `#[cfg(test)]` test_gpu_support module. In the squashed
chain the deletion rides the `9bce010` diff (`@@ -578,46 +581,3 @@`), whose
message is entirely the deltanet pre_rec gate fix — nothing about gemma:
a rebase casualty, not a decision. Inside riir-infer the eight modules are
now silent dead files (undeclared `.rs` files compile to nothing and warn
nowhere), so its own postures stay green — `eb3438f`'s "validated on the
4090: all 5 gemma2_d2f::tests PASS" necessarily ran on the PRE-squash
tree. The consumer is the only place the break surfaces — exactly the
cross-repo seam this mirror exists to gate, and the exact shape the slice-5
landing's "riir-gpu clippy default+all-features all-targets" green row was
measuring before the force-push invalidated it.

**Fix (riir-infer, one commit):** re-add the wiring block verbatim from the
removed side of `git -C ../riir-infer show 9bce010 --
crates/riir-infer-gpu/src/lib.rs` (equivalently the added side of
`6b6de2b`), then re-run the slice-5 postures INCLUDING the consumer lane:
riir-ai `cargo clippy --workspace --all-targets` at default features —
plus `cargo test -p riir-infer-gpu --features gemma2_d2f` verifying the 5
d2f tests actually RUN again (post-squash they compile to zero). Both
boxes pull the repair (all repos must stay synced m3↔4090).

**Completeness audit (same session, reflog forensics):** the pre-loss
local slice-5 commit `0e3556c` (13:33, reflog) diffs against HEAD
`eb3438f` with **zero dropped files** — all 17 slice-5 sources are present
at HEAD; the lib.rs wiring block is the SOLE casualty (the loss rode the
15:26 cherry-pick-onto-`origin/develop` sequence after the P4 squash
force-push; `a488066` — the 4090-validated d2f fix — was re-landed as
`eb3438f` on a base that no longer carried the wiring). All four features
the block gates on (`cubecl_runtime`, `gemma2_d2f`, `wall_attention`,
`gemma4_gpu`) exist in HEAD's Cargo.toml — the lib.rs block re-add is the
COMPLETE repair, nothing else is missing.

## 2026-09-23 17:5x +07 — fix VERIFIED end-to-end in a /tmp mirror (zero sibling writes); landing remains one riir-infer commit

The 4090 session (which cannot write `../riir-infer` — outside its
workspace, read-only per the global rule) reproduced the full
break-and-fix cycle in a scratch mirror at `E:/tmp-infer-verify` (REMOVED
after — worktrees pruned, clones rm'd, both real repos left clean): riir-infer
CLONED at HEAD `eb3438f`; riir-ai worktree at `46b743ba33`; katgpt-rs
worktree at `ae584dbf`; the manifest-closure siblings (neuron-db, chain,
dapps, train, kat, auth) cloned at their origin-develop HEADs — the same
commits this box's synced checkouts hold, so the mirror is
resolution-faithful to the real layout at every path depth.

**Method:** the wiring block was re-added by reverse-applying ONLY the final
hunk of `git show 9bce010 -- crates/riir-infer-gpu/src/lib.rs`
(`@@ -578,46 +581,3 @@`, the deletion hunk; equivalently the added side of
`6b6de2b`). `git apply -R --check` PASSES at `eb3438f` — the insertion
point (the lib.rs tail after `pub mod qwen38_dflash2_gpu;` + its cfg) is
unchanged, so the 46 lines re-add verbatim with zero contextual drift.

**Verification matrix — all four cells MEASURED:**

| check | pre-patch (`eb3438f`) | post-patch |
|---|---|---|
| riir-ai `cargo check -p riir-gpu --all-targets` (default features) | **exactly the 4×E0432** (gemma2_q4k_weights / gemma2_cubecl / llama_cubecl / rope_geglu_cubecl) | **0 errors — Finished** |
| riir-ai `cargo check --workspace` (the AGENTS.md default baseline) | red (the 16:14 sweep's finding) | **rc=0, 4m03s** |
| riir-infer `cargo test -p riir-infer-gpu --features gemma2_d2f --lib` | **0 `gemma2_d2f::tests` exist** — `-- --list | grep gemma2_d2f` returns ONLY the 23 `gemma2_d2f_sc::tests::*` (the declared single-file sibling); the dir-module's 5 compile to nothing = the green-zero, live-proven | **5/5 PASS** (`with_sampler` / `converges` / `valid_tokens` / `sc_identity_matches_disabled` / `sc_enabled_runs`); lib **262 passed / 0 failed / 1 ignored** |
| same command, totals | 200 passed / 1 ignored | **+62 tests surfaced, all green** (the other re-declared modules' tests) |

Cost note for the landing re-run: the d2f test binary alone runs **~21 min
(1244s)** — the convergence tests are slow, not hung (GPU-idle box, single
run). Do not mistake it for a wedge.

**The complete repair text** (verbatim from the removed side of `9bce010`;
append at the lib.rs tail, after the `qwen38_dflash2_gpu` cfg line — or
just extract + reverse the hunk as above):

```rust
// ---------------------------------------------------------------------------
// P3 slice 5 (the gemma kernel cluster — riir-ai Plan 610 S5). The gemma2
// CubeCL decode stack + its cluster dependents (llama, d2f, q4k weights,
// gemma4) + the RoPE/GeGLU + Wall kernel riders. The wall_config hinge:
// `WallConfig` re-homes to riir-infer-core (this crate consumes it as
// `riir_infer_core::wall_config::WallConfig` — the one engine-local import
// that forced gemma2_cubecl STAYS in the T2 audit). gemma2_forward +
// forward_prefill stay engine-side: their WGSL zoo (`kernels::GpuPipelines`)
// is shared with training/game consumers and cannot move to this
// inference-only repo (the audit's "share or split at P3" adjudication —
// its own slice).
// ---------------------------------------------------------------------------

// Gemma 2 CubeCL decode stack: GEMV/flash-attention dispatch, KV cache,
// weight buffers (Plan 106 T2.6 + 087 Phase 4.4-4.7).
#[cfg(feature = "cubecl_runtime")]
pub mod gemma2_cubecl;
// Gemma 2 Q4_K quantized weight upload (Plan 087 Phase 4.8) — wgpu-only,
// ungated like its engine home.
pub mod gemma2_q4k_weights;
// Llama CubeCL decode stack (consumes the gemma2 dispatch machinery).
#[cfg(feature = "cubecl_runtime")]
pub mod llama_cubecl;
// Gemma2 D2F block-causal decode (dllm lane; the feature also turns on
// fastrand + infer-core's dllm Config fields).
#[cfg(feature = "gemma2_d2f")]
pub mod gemma2_d2f;
// Gemma4 CubeCL stack (partial RoPE / QK-Norm / GeGLU / layer output scale).
#[cfg(feature = "gemma4_gpu")]
pub mod gemma4_cubecl;
// GPU-side RoPE + GeGLU kernels (Plan 106 T2.13) — the gemma decode
// cluster's positional/activation path.
#[cfg(feature = "cubecl_runtime")]
pub mod rope_geglu_cubecl;
// Wall Attention CubeCL kernels (Plan 173; gated like its engine home).
#[cfg(all(feature = "wall_attention", feature = "cubecl_runtime"))]
pub mod wall_cubecl;
// Shared GPU test helpers (Issue 712 heavy-model gate + page release).
// COPIED, not moved — the engine gpu crate keeps its own for the staying
// gemma2_forward tests (test-only infra, the cross-repo duplication class).
#[cfg(test)]
mod test_gpu_support;
```

**Blast-radius precision (measured vs constructed):** the default-lane 4×E0432
are the measured set. TWO more import sites exist — `pub use
riir_infer_gpu::gemma4_cubecl` (lib.rs:650) and `pub use
riir_infer_gpu::wall_cubecl` (lib.rs:1815) — gated behind `gemma4_gpu` /
`wall_attention`+`cubecl_runtime` (not unified on at default, so silent in
the default-lane check). Pre-patch they are E0432s at those postures;
post-patch they resolve BY CONSTRUCTION (the block declares both modules
with exactly those gates; both module sources verified present at HEAD;
the pre-squash all-features green row covered them). The landing session's
all-features re-run is the executor for that half.

**Landing checklist** (the carve session or owner — one commit):
1. In `../riir-infer` at `eb3438f` (or later): apply the block above to
   `crates/riir-infer-gpu/src/lib.rs` tail; commit; push.
2. Both boxes pull (m3 + 4090 sync rule).
3. Re-run at the REAL layout: riir-ai `cargo clippy --workspace --all-targets`
   at default features (expect green — `cargo check --workspace` already
   proven green in the mirror) + `cargo test -p riir-infer-gpu --features
   gemma2_d2f` (expect the 5; ~21 min).
4. The in-repo clippy postures + fence ride the landing session's own gate
   (the modules' code is unchanged from the pre-squash slice-5 state that
   passed them).

## 2026-09-23 (evening) — P4/T6 EXECUTED: both repos PUBLIC (the opening record)

Owner green-light landed in-session ("green light, do it"); both sanctioned
exceptions flipped the same day:

- **gist-rs/riir-infer PUBLIC** — history sanitized per the fresh-history
  law: the carve root + a single squashed GPU-layer commit + the opening
  fixes + the opening posture (MIT LICENSE, cargo-about
  THIRD_PARTY_LICENSES.md, BOUNDARY private-sibling paths rewritten), then
  the S5 gemma cluster + gemv split2 rung + the reconciliation repairs as
  one sanitized commit. The opening's full-gate execution caught two real
  defects no prior lane had executed: a GPU-crate doctest importing the
  pre-carve crate name (doc comments are masked to the fence by design;
  `cargo test --doc` is the lane that compiles them) and a whole-file-cfg
  test target reporting a green zero without its `required-features` row —
  both fixed at the opening. The private-repo Actions lane had NEVER
  started (spending limit); the first green CI run is the public repo's.
  The mid-opening twin landing (S5 + the merge that re-introduced the old
  history) was re-squashed once with lease; a follow-on squash window then
  dropped the slice-5 lib.rs wiring block — the 16:4x break documented
  above, fixed at `d058f48` and verified green at the consumer (the
  `cargo check --workspace` row above). STANDING RULE for every future
  pusher to the public repo: REBASE onto the squashed line, never merge;
  commit messages stay free of internal numbers, sibling names, and box
  references. CI trigger now main-only (`74850b5`, the spending-limit
  posture); develop pushes run no lane.
- **gist-rs/riir-reflex PUBLIC** — history decision: FRESH two-commit
  history (the engine root + the opening posture); the old campaign
  narrative history is gone from the public line; the source-side v0.2.0
  tag deleted (the dist repo's releases are the install record — v0.2.2
  has since been cut from the new line). README/AGENTS/BOUNDARY
  private-forever statements rewritten; LICENSE added; minimal ci.yml
  (check at the SHIPPED release set — not --all-features:
  laya-riir-metal is macOS-only by construction — clippy -D, and
  test --release because the G2 harness latency bar measures an
  unoptimised binary in debug); a machine-local path scrubbed from the
  tracked benchmark results. Verified green in an isolated worktree
  before the flip; CI green on every push since.
`publish = false` stays both sides — crates.io publication remains
  owner-gated. Remaining in this campaign: P2/T4 (the encoder move —
  LANDED 2026-09-24) + P5/T7 (op-layer unification), tracked in reflex
  Issue 008. **T7's execution plan LANDED 2026-09-26:
  `.plans/611_t7_op_layer_unification.md`** (S1–S6; S1a's
  mean-centered-LN GAP kernel is the engine-side payoff that survives
  every A/B outcome; the deletion criterion is pre-registered in the
  plan before any number is read). **S1 COMPLETE 2026-09-26:** S1a
  `bc93e70` + S1b `a5688a9` — the `CubeclBackend` skeleton (feature
  `laya-riir-cubecl`, the Metal lane's residency model on CubeCL's
  server, `wgpu<msl>` pinned at construction) with the trivial op
  family + residency parity green (gpu lib 208/0; op-by-op vs Cpu +
  the begin_pass/download_into/copy_at semantics arms). **S2
  COMPLETE 2026-09-26 (`a3f8928`):** the full matmul family —
  `matmul_w` over the shipped derived-dims transB kernel, the four
  offset/batched ops over two new z-dispatched tiled kernels (the
  head batch on CUBE_POS_Z, one launch per batch), the trait-default
  folds proven behaviorally; gpu lib 210/0, smoke 7/7, max drift
  1.07e-4 vs the 1e-3 budget. S3 (head ops + attention) next.
