# Plan 610 — the riir-infer-gpu carve: P3 slice 1 of the riir-infer consolidation

> **Moved 2026-09-26:** this plan lives in `riir-infer/.plans/` now (moved verbatim from riir-ai, same number — riir-ai's ledger keeps 610 allocated). Prose paths are written from the riir-ai workspace root at writing time; **`../riir-infer` means THIS repo**, `../riir-ai/*` is the riir-ai checkout.

**Status:** IN PROGRESS — slice 1 LANDED 2026-09-23 (T1–T7 green); S2 + S3 + S4a + S4b + S5 LANDED 2026-09-23. S4 (SEAM adjudication core) RESOLVED by S4a+S4b; **S6a LANDED 2026-09-24** (the T2.3 clippy retarget + its two carve riders, commits below). **S6b LANDED 2026-09-24 at honest scope** (patch re-points + adjudications; the full edge-drop is owner-blocked on the training-families home — measured 12/81 forwards mirrored, see the S6b row). **S7 LANDED 2026-09-24** (dead riir-gpu-async row removed; riir-router measured NOT dead — default-on feature; flip residue measured complete by compilation; riir-ai `498b1732b9`). **S8 remains** (gated on P2/T4).

Executes **P3** of riir-reflex `.issues/008_riir_infer_consolidation.md`
(the campaign plan; riir-ai mirror `.issues/998`). T2's module audit
recorded there classifies riir-gpu's 171 modules CLEAN 98 / SEAM 34 /
STAYS 39. P3 moves the CLEAN (+ adjudicated SEAM) GPU kernel layer into
`../riir-infer` and re-points the riders (T2.3 retarget + D4
re-narrowing). P2 (reflex laya lane) is separately sequenced and waits
for that repo's sibling WIP; P5 (op-layer unification) needs both.

## The measured blocker that shaped slice 1 — the cubecl vendor patches

`[patch.crates-io]` rows do NOT cross workspace boundaries. riir-ai
vendors three patched crates.io forks in its workspace root; private
consumers (riir-clippy) redeclare them pointing at
`../riir-ai/vendor/*`. **riir-infer is public with a zero-`riir-*`
fence** — it can never patch from a private sibling, so it must carry
its own copies:

| vendor fork | why | slice-1 posture |
|---|---|---|
| `cubecl-runtime-0.11.0-pre.2` | drop-queue policy fix (upstream tracel-ai/cubecl#1359) — behavioral | **vendored** (byte-identical copy; correctness) |
| `cubecl-wgpu-0.11.0-pre.2` | adds `as_raw_metal_*` accessors — additive API only | not needed (serves `metal_tensor_gemm`, not slice 1) |
| `wgpu-hal-30.0.0` | adds `raw_handle()` — additive API only | not needed (same) |

Death condition for the vendored copy is upstream's own: remove when
tracel-ai/cubecl#1359 lands in a release both workspaces can adopt.

## Slice 1 — the base GPU runtime cluster (this plan's executing task)

The six modules every kernel builds on; measured zero engine refs
(CLEAN) and zero moving→STAYS edges (only intra-cluster + std/wgpu/
cubecl imports):

| module | LOC | gate in riir-gpu |
|---|---|---|
| `buffer` | 969 | ungated |
| `context` | 431 | ungated (inner `cubecl_runtime` cfgs) |
| `pool_poison` | 180 | ungated (inner backend cfgs) |
| `weight_buffer_cache` | 527 | `#[cfg(feature = "cubecl_runtime")]` |
| `gpu_transpose` | 290 | `#[cfg(feature = "cubecl_runtime")]` |
| `cubecl_runtime` | 633 | `#[cfg(feature = "cubecl_runtime")]` |

- [x] T1 New workspace member `crates/riir-infer-gpu` in `../riir-infer`
      (single crate at root becomes a root-package workspace; the new
      crate mirrors riir-gpu's exact dep rows for what the cluster
      needs: wgpu 30 + cubecl `=0.11.0-pre.2` wgpu-msl on macOS /
      wgpu-spirv elsewhere, optional behind `cubecl_runtime`; features
      `cubecl_runtime`, `cuda_backend = ["cubecl_runtime",
      "cubecl/cuda"]`; no half/log/papaya — measured unused by the six).
- [x] T2 Vendor `cubecl-runtime-0.11.0-pre.2` (byte-identical copy from
      the riir-ai vendor dir) + `[patch.crates-io]` in the riir-infer
      workspace root with the upstream-issue provenance note.
- [x] T3 Move the six files (git-tracked copies; verbatim contents) +
      `riir-gpu` re-export layer: `pub use riir_infer_gpu::{...}` for
      the modules (cfg-matched), item re-exports at the gpu root
      unchanged (`CubeCLContext`, `ActiveComputeClient/Device/Runtime`,
      `GpuContext/GpuError`, the buffer fns) — zero consumer edits
      anywhere (engine, agents, games, games-quest, examples, poc,
      riir-clippy all keep resolving).
- [x] T4 riir-gpu manifest: non-optional `riir-infer-gpu` path dep
      (default-features = false); `cubecl_runtime` feature forwards
      `riir-infer-gpu/cubecl_runtime`; keeps its own `dep:cubecl` for
      the remaining kernels.
- [x] T5 BOUNDARY rows both sides: riir-ai May-depend-on gains
      `riir-infer-gpu | ../riir-infer | riir-gpu non-optional`; riir-infer
      BOUNDARY Owns + May-depend-on gain the gpu crate + wgpu/cubecl
      allowlist rows (crates.io only; still zero `riir-*` deps).
- [x] T6 Validate: `cargo check --workspace` in riir-infer (both
      postures ±cubecl_runtime); `cargo check -p riir-gpu` +
      `cargo clippy -p riir-gpu` in riir-ai; the fence gate green; lib
      tests of the moved cluster from the new home.
- [x] T7 Commit + push both repos; tick T5 (fence gate) + P3 slice-1
      progress in reflex `.issues/008` + riir-ai `.issues/998`. — LANDED
      2026-09-23: riir-infer `c5d901a`, riir-ai `d50254e91`+`36cc59cda`+
      `d8ed85f7c`, reflex `63b1552`; 4090 synced (riir-infer `c5d901a`,
      riir-ai `d8ed85f7c`, riir-train `8ef6560d` via bundle — GitHub auth
      hangs from that box).

## Later slices (each independently landable, each needs its own edge
adjudication from the T2 table)

- [x] S2 — LANDED 2026-09-23. The elementwise/norm/matmul/attention
      family moved: `cpu_reference`, `gemv_batched`, `gemv_autotune`,
      `gemv_cubecl`, `gemv_f16_cubecl`, `gemv_geglu_cubecl`,
      `gemv_geglu_f16_cubecl`, `gemv_qkv_f16_cubecl`, `gemv_q4k_cubecl`,
      `gemv_q4k_batched_cubecl`, `gemv_q4k_batched_rmsnorm_cubecl`,
      `gemv_qkv_q4k_cubecl`, `gemv_geglu_q4k_cubecl`, `matmul_cubecl`,
      `matmul_swap_ab_cubecl`, `attention_cubecl` (+tests),
      `attention_causal_fused_cubecl`, `attention_q8kv_cubecl`,
      `gemma2_d2f_sc`, `norms_cubecl`, `sampling_cubecl`,
      `elementwise_cubecl`, `epilogue` (4 files) — 28 files. The
      quant-typed SEAM members of these families rode the slice (the
      quant-dep note below made them in-scope): the five
      `riir_engine::quant` imports became `riir_infer_core::quant`
      (q4k BlockQ4K/QK_K/gemv_q4_k + q8kv BlockQ8_0; the T1.2
      re-export source — same items, one hop shorter).
      riir-gpu re-exports all modules (cfg-matched `pub use
      riir_infer_gpu::<mod>`) — item re-exports unchanged; zero
      consumer edits (riir-engine, riir-train-engine
      `riir_gpu::gemv_batched::launch_gemv`, riir-train-gpu
      `gemma2_cubecl_train`/`gemma4_*_train`, riir-poc all verified
      green).

      Slice-2 riders the plan did not predict (the slice-1 class):
      - **fence-gate F1 was under-implemented** — the path-dep check
        used a leading-`..` heuristic, which would have red the new
        in-repo `riir-infer-gpu -> riir-infer-core` dep (`path =
        "../.."`) BY CONSTRUCTION. Fixed to real containment (resolve
        the path against the manifest dir, require it stays inside the
        repo root) + two selftest arms (member-dep `../..` legal,
        `../../../X` fires). The gate's own docstring already stated
        the right rule; the code now matches it.
      - **`params_cache` moved with the family** (the slice-1
        vram-probe class): 175 LOC, the blake3-keyed params-handle
        cache consumed by norms/elementwise NOW and the deltanet/qwen
        families (S3/S4) later; the audit table never named it.
        `params_handle_cache` feature mirrored + forwarded (pure
        forward — enabling it without cubecl_runtime stays inert at
        riir-gpu's surface exactly as before).
      - **Six riir-gpu features mirrored into riir-infer-gpu** because
        moved modules carry cfgs on them (`unexpected_cfgs` would
        red clippy -D warnings): `swap_ab_gemm`, `gemma2_d2f`,
        `q4k_rowtiled_gemv`, `gemv_fma_contract`, `fold_dispatch`,
        `q8kv_sink_guard` — all `= ["cubecl_runtime"]` mirrors +
        riir-gpu forwards.
      - **deps added to riir-infer-gpu**: `half` + `papaya` + `blake3`
        (all cubecl_runtime-gated; slice 1 measured them unused by
        the SIX, and S2's modules do use them) + `riir-infer-core`
        (non-optional path dep — the quant block layouts ARE the
        kernels' wire formats).
      - **one forced divergence**: `sampling_cubecl::ArgmaxCubeCL`
        widened `pub(crate)` -> `pub` in the riir-infer-gpu copy —
        the REMAINING `gemma2_cubecl` (S5 hinge) consumes it
        cross-crate through the re-export; riir-gpu's source copy
        keeps `pub(crate)` until S5.
      - **found pre-existing rot (NOT this slice's)**:
        `tests/bench_603_gpu_integrated_forward.rs` fails E0308 at
        `--all-features --tests` — `GpuTernaryMatvec`/`GpuTernaryInputProj`
        hooks expect `TernaryGroupWeights`, the loader's layers now
        carry `deltanet::ternary_weights::GateProjWeights` (enum
        Dense/Ternary). Verified pre-existing at HEAD (stash test):
        identical failures with this slice's changes stashed; the
        lane (`--all-features --all-targets`) is one slice 1 never
        ran. S3 territory (the ternary family owns the hooks); every
        other test target compiles at all-features.

      Validation: riir-infer fence gate green (241 .rs, 0 pins);
      workspace check + clippy -D warnings at default / no-default /
      full-combo postures; lib tests 158 passed (cubecl posture) and
      **206 passed / 0 failed** (all-features combo). riir-ai: riir-gpu
      check + clippy -D warnings default & all-features (0 errors);
      **lib suite 283 passed / 0 failed** at the cubecl posture
      (426 - 143 = the moved tests now in riir-infer-gpu; the 28
      gemma2 tests exercise the moved kernels through the re-exports
      on Metal — GPU-matches-CPU parity incl.); riir-engine,
      riir-train-engine, riir-train-gpu checks green.
- [x] S3 — LANDED 2026-09-23. The ternary gemv/gemm + metal + CUDA-raw
      families moved (33 files: 31 .rs + 2 .metal): ternary gemv family
      (gemv_ternary_cubecl hub, scale_ab, fma, residual,
      block_contiguous, cuda_raw), ternary gemm family (batched, tiled
      +8x8/Xfix, cmma16, cmma_i8 +direct/t64/psplit, simdgroup + its
      #[path] smem child, block_contiguous), the macOS metal-tensor
      family (metal_tensor, metal_wgpu, metal_zero_copy — include_str!
      .metal paths resolve verbatim, same-dir), the CUDA raw prefill
      family (gemm_ternary_i8_mma_cuda_raw, i8_mma_v6_src,
      prefill_cuda_{mma,ffn,deltanet,attention,attention_vec,
      attention_gang,attention_fa,gdn_chunked}, ternary_ffn_fused), and
      deltanet_input_proj_fused (rides its CLEAN T2 row, pulled forward
      by the bench_603 rot — GpuTernaryInputProj is one of its hooks).
      qwen38_* / prefill_cuda_full / cudarc_kernels stay for the S4 SEAM
      window as pre-adjudicated.

      Slice-3 riders (the S2 class, all recorded):
      - **Nine visibility widenings** (pub(crate) -> pub in the
        infer-gpu copies), each consumed by a REMAINING riir-gpu SEAM
        module through the re-export: gemv_ternary_cuda_raw::
        {convert_bitplane_to_packed_codes, GEMV_CUDA_SRC}
        (ternary_deltanet_gpu_forward_cudarc); prefill_cuda_mma::
        {launch_prefill_gemm_cached, launch_prefill_gemm_pair_cached,
        build_weight_cache, prefill_use_cuda_mma, dispatch}
        (prefill_cuda_full + ternary_deltanet_gpu_forward);
        prefill_cuda_ffn::{dispatch_ffn_block, prefill_use_cuda_ffn}
        (ternary_deltanet_gpu_forward).
      - **canonical_expand_forms moved** from prefill_cuda_full (STAYS,
        SEAM) into prefill_cuda_deltanet, whose L2Accum/L2Inv types it
        names; the moved file's #[ignore]d timing-probe test reaches it
        in-module now (was super::super::prefill_cuda_full), and
        prefill_cuda_full consumes it through the module re-export.
      - **10 features mirrored + forwarded**: ternary_gemv (the
        infer-gpu mirror DROPS the riir-engine/deltanet_ternary_inference
        forward leg — the fence; riir-gpu's own feature keeps it),
        ternary_gemv_residual, ternary_gemm_batched,
        ternary_gemm_simdgroup(+_f16), ternary_lut_gemv,
        ternary_gemv_cuda_raw (+dep:cudarc, not-macos target),
        prefill_q8_act, prefill_mmq_v2, metal_tensor_gemm (+dep:metal,
        macOS target).
      - **katgpt-core joins riir-infer-gpu via a new
        [workspace.dependencies] table** (S3 modules consume
        katgpt_core::TernaryGroupWeights directly; the member manifest
        inherits with workspace = true so it never spells the deeper
        ../../../katgpt-rs path — the fence's raw-prefix check stays
        green). The ready-notes' feature-mirror checklist is fully
        consumed; cuda_graphs_forward / ternary_attention_batched_prefill
        / ternary_deltanet_chunked_prefill measured NO cfg in the moved
        set (they gate staying qwen38/deltanet modules).
      - **bench_663_t5_single_gemm_isolation moved with its kernel**
        (include_str!("../src/gemm_ternary_metal_tensor.metal") + the
        riir_gpu:: imports are all infer-gpu natives; [[test]] row
        required-features mirrored).
      - **S2 rider found + fixed**: the ArgmaxCubeCL divergence note sat
        BETWEEN #[cfg] and the struct, splitting the doc block —
        clippy::doc_lazy_continuation at --all-features; merged into the
        main doc block.
      - **bench_603 rot FIXED** (pre-existing at HEAD, stash-verified in
        S2 and re-verified here): the Issue-980 loader change turned
        in_proj_a/in_proj_b into GateProjWeights; the bench preuploads
        the Ternary arm via gate_projections()/as_ternary() and skips
        Dense. The same rot class hit riir-train-engine's bonsai-go lane
        (layer_backward/synthetic/model_backward_recompute — fixed there
        at f5bfe3f9, 1691/0).

      Validation: riir-infer fence gate green (272 .rs); infer-gpu
      clippy -D warnings at 3 postures; infer-gpu lib **225 passed / 0
      failed** at the all-features combo (Metal; the ternary kernels
      GPU-validated from the new home). riir-ai: riir-gpu clippy -D
      warnings at default + all-features, all-targets (0 errors); lib
      tests **267/0** at default (949s, loaded box); the all-features
      lib suite is jetsam-killed on this box AT HEAD TOO (pre-existing,
      stash-verified — S2's 283/0 was a different box state), so the
      gemma2_d2f re-export-path tests ran individually — 4/4 pass (80s
      each); all-features --tests compile lane clean; riir-train-engine
      + riir-train-gpu checks green.
- [x] S4a — LANDED 2026-09-23 (the 4090 box; the S4 dependency closure).
      The qwen attention family (qwen_attention_cubecl + the 8 prefill
      arms m16/m32/m64/m32_pipe/m32_kvf16/cmma/cmma_pv +
      qwen_prefill_q8kv_cubecl) + the deltanet CLEAN kernels
      (deltanet_chunked_cubecl, deltanet_delta_rule_chunked,
      deltanet_tree_verify_cubecl) — 12 files, ~15.4k LOC. Every external
      dep was already moved (cubecl_runtime S1, params_cache S2); zero
      real SEAM rewrites (the 2 engine refs in tree_verify were comments).
      Held OUT of S4a: deltanet_pre_rec_fused (deps deltanet_cubecl — S4),
      weight_readback (deps ternary_deltanet_gpu_forward — S4),
      forward_flashprefill (deps the `kernels` WGSL zoo — own
      adjudication). The qwen family's `ternary_deltanet_gpu_forward`
      refs are doc-comment links (the apparent S4a/S4 cycle was an
      illusion) — they re-resolve when S4 lands.

      S4a riders (the S2/S3 class, all recorded):
      - **One visibility widening**: Q8PrefillScratch's 5 fields +
        buffer_bytes pub(crate)->pub — the STAYS consumer
        (ternary_deltanet_gpu_forward:5319-5333) CONSTRUCTS the struct
        cross-crate through the re-export (the S3 class).
      - **S3 residue fixed (the not-macos blind spot)**: the dead local
        `canonical_expand_forms` wrapper in prefill_cuda_full (S3 re-pointed
        the call site but left the wrapper; the M3 never compiled the
        not(macos)-gated module) + its orphaned L2Accum/L2Inv imports.
      - **S3 residue fixed (the same class, test-target)**:
        `canonical_recurrence_forms` moved into prefill_cuda_deltanet (the
        S3 canonical_expand_forms precedent) — S3 fixed expand's
        super::super::prefill_cuda_full test paths but missed recurrence's
        (L1784/L1981); first-ever compile of the CUDA-family lib tests on
        non-macOS caught it. Root re-export + the prefill_cuda_full prod
        call site re-pointed; bench_734_arm12's `riir_gpu::
        canonical_recurrence_forms` path unchanged.
      - **S1 residue fixed**: cubecl_runtime.rs doc-list indentation
        (doc_lazy_continuation at cuda_backend-on-Windows — a posture/box
        combo S1's M3 validation never ran).
      - **fence_gate.py selftest Windows bug fixed**: the selftest compared
        `rel == "src/lib.rs"` against backslash relatives — every Windows
        run red with the scanner working; normalized. The fence gate now
        PASSES on the CUDA box (284 tracked .rs — grew by the 12).
      - **S1 latent RECORDED (not fixed)**: weight_buffer_cache
        `test_slot_from_data_and_write_in_place` FAILS at --all-features on
        non-macOS (ActiveRuntime=CudaRuntime — extract_buffer's cudarc
        path returns None for slot.buffer); passes at the WgpuRuntime
        posture. First-ever all-features-on-CUDA-box run surfaced it; the
        M3's all-features had cuda_backend inert. Needs the cudarc
        buffer-extraction path investigated — separate issue, before any
        cuda_backend-dependent default promotion.
      - **Pre-existing RECORDED**: ane_prefill test `ctx` unused-var on
        non-macOS (the assert is macOS/aarch64-gated) — riir-ai lane,
        untouched by this slice.

      Feature mirrors added to riir-infer-gpu (all cfg keys found in the
      moved code): ternary_deltanet_chunked_prefill,
      deltanet_recurrence_rowpar, speculative_tree_verify (+ the
      katgpt-core/gdn_tree_verify leg — the moved tree-verify tests assert
      against that CPU oracle), ternary_attention_batched_prefill; riir-gpu
      forwards added to the four engine-side features.

      Validation (the 4090 box — the only box that compiles the CUDA arms):
      infer-gpu check+clippy at default/cubecl/all-features (clean);
      workspace check both postures; fence gate PASSED; lib tests 206/0 at
      the S4a combo (the moved kernels GPU-validated from the new home:
      deltanet_chunked 7/7, delta_rule_chunked 3/3 incl. production shape,
      tree_verify 3/3 incl. CPU-oracle match, qwen_attention decode/tile/
      rope/tree/kv-cache G1s), 166/0 at cubecl_runtime, 266/267 at
      all-features (the 1 = the recorded S1 cudarc-posture failure);
      riir-gpu check+clippy default/all-features (clean modulo the
      recorded pre-existing ane warning), lib tests 325/0 at default;
      riir-engine + riir-train-gpu + riir-clippy(latent_retrieval) checks
      green; boundary contract: 0 violations (the 2 findings are box-state
      partial-clone rows — ../riir-esp32 drift-ledgered, ../riir-reflex
      absent on this box).

- [x] S4b — LANDED 2026-09-23 (the 4090 box; the S4 SEAM core). The
      qwen38/cudarc cluster + the deltanet-forward family + the ane_prefill
      rider moved (31 files, ~45k LOC): qwen38_{dense_cudarc, verify_mma,
      prefix_cache, dflash2, dflash2_gpu}, cudarc_kernels (6 files minus
      backward.rs), prefill_cuda_full, ternary_deltanet_gpu_forward{,_cudarc},
      deltanet_rotation_{cubecl,cudarc}, deltanet_cubecl,
      deltanet_pre_rec_fused_cubecl, ternary_tree_verify_driver,
      hybrid_dispatch, weight_readback, state_readback (audit-table gap —
      SEAM L125, coupled to the forward), vram_budget (audit-table gap —
      mutually coupled to the forward), ane_prefill/ (7 files — the rider
      resolving the forward's 125 ane cfg sites; CLEAN per audit, the
      params_cache precedent).

      The two S4 adjudications, RESOLVED:
      - **`cudarc_kernels/backward.rs` (training surface)**: moved to
        **riir-train-gpu** as `cudarc_backward_kernels` (the C6 residue
        gate's own remedy — self-contained std+cudarc, consumed ONLY by
        train-gpu's backward section). The moved forward_cudarc STRIPPED
        its `pub backward: BackwardKernels` field + construction;
        `ForwardInfraCudarc.ctx` widened to `pub` (the D4 class) so the
        training side constructs its own kernels. train-gpu gains
        `TrainingForwardCudarc { fwd, backward }` (Deref/DerefMut to the
        inference forward — every existing call site unchanged) holding
        the impl of `TernaryDeltanetBackwardTraining`.
      - **`ternary_deltanet_gpu_forward` L179 (training_activation_cache)**:
        the `gpu_training_resident`-gated legacy pair
        (`forward_token_training` + `forward_from_x_training`, the
        4-field `TrainingActivationCollector` path) STRIPPED from the moved
        copy — zero external callers (the live training forwards are the
        UNGATED `forward_token_training_minimal` + the cudarc variant,
        both infer-core-typed). `training_activation_cache.rs` stays
        riir-gpu-side as audit-designated residue.

      S4b riders (the S2/S3/S4a class, all recorded):
      - **Gate tightenings at the new home** (latent couplings the home's
        default-feature unification never separated):
        `prefill_cuda_full` bare-cubecl → the S3 sibling set
        (cuda_raw+gemm_batched+cubecl+not-macos; it launches cudarc kernels
        + reads deltanet_rotation_cudarc ungated); `vram_budget` bare-cubecl
        → +ternary_gemv (reads ternary_weights + the forward's SPEC_MAX_K);
        `deltan_cubecl` bare-cubecl → any(ternary_gemv, deltanet_inference)
        (DeltaNetStateBuffers::new takes &[DeltaNetLayerType]).
      - **The E0004 trap, measured**: a DIRECT `katgpt-core/deltanet_inference`
        leg on infer-gpu's cubecl_runtime compiles katgpt-types'
        `ModelArchitecture::QwenDeltaNet` variant WITHOUT infer-core's
        cfg-gated match arm (transformer/mod.rs forward_with_lora) —
        non-exhaustive match in the UNIFIED build. Every katgpt deltanet
        leg is therefore routed THROUGH infer-core's own features
        (ternary_gemv → riir-infer-core/deltanet_ternary_inference).
      - **Feature mirrors added** (7): ane_prefill (dep:metal),
        attn_mass_tap, speculative_decode, cuda_graphs_forward,
        deltanet_recurrence_parallel, deltanet_inference (engine leg →
        direct riir-infer-core leg), + ternary_gemv gained the
        riir-infer-core/deltanet_ternary_inference leg. riir-gpu forwards
        on all six features.
      - **Two visibility widenings** (the S3 class):
        cudarc_kernels/attention::ATTENTION_CUDA_SRC +
        qwen38_dense_cudarc::QWEN38_DENSE_CUDA_SRC pub(crate)→pub
        (t916_compile_smoke consumes them cross-crate).
      - **`qwen38_kv_f16_gates` STAYED HOME**: a `#![cfg(all(test, ..))]`
        file cannot live in a dependency crate (deps never compile under
        cfg(test)) — test-only module, test-only home; its imports resolve
        through the re-exports.
      - **Posture-split imports**: DeltanetZGatingCubeCL +
        RmsNormBatchedCubeCL split into `#[cfg(feature =
        "ternary_gemm_batched")]` use lines (used only by the batched
        arms; the stripped training pair was their other consumer).
      - **Two honest allows** for postures the home never ran: the stacked
        cfg-guard fn `prefill_cuda_gate_ok` (unreachable_code when a later
        guard's feature is on while an earlier one is off) + the six
        CUDA-graph-capture fields on ActivationsCudarc
        (cfg_attr(not(cuda_graphs_forward), allow(dead_code))).
      - **Issue-980 GateProjWeights rot fixed at two never-compiled-here
        lanes** (the f5bfe3f9 recipe — local `gate_matvec` arm-dispatch):
        train-gpu's beta_decay_backward_cpu + the arm_c example's CPU
        reference; train-engine's own gate_matvec test-gated with its only
        callers.
      - **arm_c example**: `bonsai-lora-accuracy-parity-arm-c-cudarc` gained
        the `riir-train-gpu/ternary_backward_cudarc` leg (the wrapper
        constructs the kernels the forward used to build internally —
        identical cost); the cudarc alias is now TrainingForwardCudarc.
      - **ane_prefill test ctx** cfg-gated with its macOS-only use (the
        S4a-recorded warning class, fixed at the new home).
      - **serde_json** joined infer-gpu (dflash2 header parsing, ungated
        like its home); fastrand as dev-dep (dflash2_gpu test fixtures).
        BOUNDARY.md's gpu-dep row widened to the actual set.

      Validation (the 4090 box): infer-gpu check+clippy at 5 postures
      (default / cubecl / ternary / cuda+gemm / all-features — clean);
      workspace check default + all-features; fence gate PASSED (314 .rs);
      **lib tests 379/1 at all-features --test-threads=1** (the 1 = the
      recorded Issue-999 weight_buffer_cache cudarc-posture failure;
      3 further full-suite failures are intra-process CUDA test concurrency
      — all pass alone, the first time this many CUDA tests share one
      process); riir-gpu check+clippy default + all-features (clean),
      all-features --tests compile, **lib tests 236/0 at default**; riir-engine
      check default + all-features; riir-train-gpu check+clippy at
      ternary_backward_cudarc + **lib tests 407/0** (the backward kernels
      GPU-validated at their new home); the arm_c example compiles at the
      full cudarc lane; riir-clippy (latent_retrieval) check green;
      boundary contract 0 violations (C6 green after the backward
      relocation; the 1 contract-rot = the pre-existing box-state
      ../riir-reflex absence).

      **The twin-landing reconciliation (the Issue-825 class, live):** a
      concurrent M3 session landed its OWN S4b on riir-infer (`9f224cc`,
      10:27 — mid-flight for the 4090 session) with divergent adjudications
      (backward.rs kept in infer-gpu; training_activation_cache MOVED;
      kv_f16_gates moved; dep-row deltanet hardcode). Reconciled at
      riir-infer `4f354f5`: OURS (the 4090, CUDA-validated versions) for
      every overlapping source; THEIRS absorbed where it was more
      complete (the ane_prefill build.rs + objc/ bridge the 4090 move
      missed, the two .metal sources, the bench_663 test); THEIRS dropped
      where superseded (backward/training_activation_cache/kv_f16_gates
      in infer-gpu, the dep-row hardcode). riir-ai `892dec017d` removed
      the now-orphaned riir-gpu build.rs + objc/. Lesson re-learned:
      fetch origin immediately before EVERY slice commit — the
      pre-flight sync at session start does not cover a 100-minute
      execution window.

- [ ] S4 — SEAM adjudications: `forward`'s adapter slots
      (moa/oft/oscpart/speft — strip or feature-forward), `lora/ia3` →
      moa edge, `ternary_deltanet_gpu_forward` → `training_activation_cache`
      un-cfg'd use (L179) — this one gates the T2.3 retarget of
      riir-clippy's ternary bench, which consumes
      `TernaryDeltanetGpuForward`.
      → RESOLVED by S4a+S4b 2026-09-23: the L179 edge is stripped (the
      gated pair had zero external callers); the T2.3 retarget is UNBLOCKED
      (`TernaryDeltanetGpuForward` now lives in riir-infer-gpu, re-exported
      at riir_gpu::). The engine-side `forward` adapter slots remain S6
      rider territory if the gemma cluster (S5) needs them.
- [x] S5 — LANDED 2026-09-23 (the 4090 box; the gemma cluster unlock).
      `wall_config` re-homed to riir-infer-core (the module is the
      Issue-019-C.1 de-fork re-export of `katgpt_core::types::WallConfig`;
      engine becomes `pub use riir_infer_core::wall_config;` — the types/wall
      pattern, zero consumer edits) → `gemma2_cubecl` SEAM-unlocked. MOVED
      (17 files, ~13.6k LOC): gemma2_cubecl/ (6), gemma2_d2f/ (2),
      gemma4_cubecl/ (4 — the SEAM rider the audit's gemma-cluster note
      implies; its only blocker was crate::gemma2_cubecl), gemma2_q4k_weights,
      llama_cubecl, rope_geglu_cubecl (SEAM, rope test-block only),
      wall_cubecl (CLEAN rider), test_gpu_support (COPIED, not moved — the
      staying gemma2_forward/tests.rs needs it home; test-only infra).

      The gemma2_forward adjudication (HELD, own slice): gemma2_forward +
      forward_prefill + forward_flashprefill STAY — gemma2_forward holds
      `Arc<GpuPipelines>` UNGATED (new_internal constructs it), and the zoo
      carries training WGSL (loss/backward/optimizer) that can never move to
      the public inference-only repo (Research 003 charter + the audit's
      "share or split at P3"). The split (infra + inference WGSL vs training
      WGSL) is its own slice; forward_prefill deps gemma2_forward + the zoo
      so it rides that adjudication.

      The gpu_training_resident adjudication (MIRRORED, not stripped): the 8
      gated methods on the gemma dispatch are generic kernel launches (gemv/
      add/rmsnorm/attention — all moved kernels); the only consumer is
      riir-gpu's test_issue430_attention_gpu (test-only home, the
      kv_f16_gates precedent). Mirror + forward; the methods stay compiled
      at their historical opt-in posture.

      S5 riders (the S2/S3/S4 class, all recorded):
      - **Feature mirrors (9)**: gemma_lora, wall_attention, delta_routing,
        q8_kv_cache, gpu_decode_fusion, lm_head_cpu, gpu_training_resident,
        gemma4_gpu + gemma2_d2f WIDENED (the S2 SC-only mirror gained
        dep:fastrand + riir-infer-core/dllm — the whole module's sampler RNG
        + dllm-gated Config fields). riir-gpu forwards on all 8 (gemma2_d2f
        already forwarded).
      - **Two deps joined infer-gpu**: `log` (gemma2 KV-cache alloc reports,
        ungated like its home — the serde_json precedent) + `fastrand`
        (optional regular, activated by gemma2_d2f; the S4b dev-dep stays).
        BOUNDARY.md gpu-dep row widened.
      - **S4b-latent gate rot FIXED** (pre-existing at HEAD, stash-verified):
        deltanet_pre_rec_fused's `any(test, ternary_gemv)` decl arm broke the
        bare-cubecl `--tests` compile (its own tests import crate::
        deltanet_cubecl, tightened to ternary_gemv in S4b — the arm never
        compiled post-S4b). Dropped the test arm on decl + paired re-export;
        the tests run at every ternary posture — no coverage loss.
      - **Never-compiled-lane rot FIXED**: gemma2_cubecl/tests.rs constructed
        GemmaTransformerWeights with delta_routing fields UNGATED (riir-gpu's
        default always carried delta_routing) — cfg'd like the struct fields.
      - **The S2/S3/S4b ORPHAN RESIDUE cleaned** (26 entries): the moved
        sources those slices left in-tree with their mod decls replaced by
        re-exports — uncompiled dead copies (19 S2-era files + attention_
        cubecl/ + epilogue/ + cudarc_kernels/'s non-backward files). Every
        tracked-orphan scan + `#[path]` grep clean; the S4b git-rm pattern
        is now uniform across all slices.
      - **Issue 1001 FILED** (pre-existing, stash-verified at the old home):
        `test_d2f_decode_converges` fails on this box (steps_used=20,
        SemiActivated, confidence 0.0) — deterministic, minimal-posture,
      repro + M3-split hypotheses recorded; the last green record was the
        S2-era Metal run of a smaller subset.

      Validation (the 4090 box): infer-gpu check+clippy -D warnings at 4
      postures (default / cubecl / the gemma combo / all-features) + tests-
      compile at 3; workspace check default + all-features; **lib tests
      269/271 at the gemma combo --test-threads=1** (31 min; the 1 = Issue
      1001, the 1 ignored pre-exists); **all-features lib tests 449/451
      serialized** post-twin-merge (the 2 = the recorded Issues 999 + 1001,
      both targeted-verified; the sibling's Plan-611 split2 CUDA test + the
      gemma cluster at the CudaRuntime posture pass); riir-gpu check+clippy
      default + all-features all-targets (clean), **lib tests 199/0/1
      default** (down from 236 — the ~37 moved gemma tests now run from
      infer-gpu); riir-engine check default + all-features; riir-train-gpu
      check at the backward posture; fence PASSED (334 .rs); boundary
      contract 0 violations (18 repos / 312 edges).

- [x] **S6a — LANDED 2026-09-24 (the M3 box; the T2.3 clippy retarget + the two carve riders it needed).**
      The retarget: riir-clippy's `ternary_inference` lane consumes
      `riir-infer-core` + `riir-infer-gpu` directly from `../riir-infer` — its
      two riir-ai edges (riir-engine + riir-gpu) are GONE; the first consumer
      fully off riir-ai for inference (the 041 T2.3 named candidate).
      Two riders the retarget surfaced (both the audit-gap class, the
      params_cache precedent — engine-side modules the T1.2 re-export list
      never classified because the T2 audit covered only riir-gpu):
      - **`tokenizer.rs` moved riir-engine → riir-infer-core** (1450 lines:
        SentencePiece + Bpe + SentencePieceGguf tokenizers): its only in-crate
        dep (`crate::gguf_loader::GgufFile`) was already infer-core-resident;
        the `sentencepiece` native-only target dep row follows the module;
        engine keeps the same-path re-export (`pub use
        riir_infer_core::tokenizer;`) — zero consumer edits (cached_kv,
        causal_validation, go_bonsai_data, go_gemma_data, swir_validation,
        riir-agents all compile unchanged).
      - **`speculative_decode/` moved riir-gpu → riir-infer-gpu** (4 files +
        testdata, ~1950 lines: ngram + dspark drafters + regime router — the
        modelless CPU-side family). dspark's `riir_engine::gguf_loader`
        import re-pointed to infer-core; the verify/checkpoint dispatch
        methods stay on TernaryDeltanetGpuForward. riir-gpu re-exports the
        module at the same path — `riir_gpu::speculative_decode::*` paths
        (the train-engine maglev lanes) resolve unchanged. Its one lib-test
        GGUF fixture arm now exercises infer-core's tokenizer end-to-end
        (34/0 from the new home).
      - **infer-gpu root item re-exports added** (the S1 pattern):
        `TernaryDeltanetGpuForward`, `set_prefill_use_simdgroup`,
        `set_prefill_use_metal_tensor_zerocopy` — making the consumer swap
        purely textual (`riir_gpu::` → `riir_infer_gpu::`).
      - **The arc-swap genlock-load patch row STAYS in riir-clippy, and the
        reason is measured, not assumed**: riir-rag (kept for
        `latent_retrieval`) carries an OPTIONAL riir-engine dep, and cargo
        resolves optional deps' features BEFORE feature-pruning — deleting
        the row fails EVERY posture's resolution (not just latent_retrieval).
        The row leaves when riir-rag's bonsai_embedder edge retargets to
        riir-infer (its natural next step — the embedder is inference-only).
      - **cubecl-wgpu fork row DROPPED** from riir-clippy: nothing in the
        retargeted graph calls the fork's as_raw_metal_* accessors (grep
        measured over riir-infer; the accessor consumers live in riir-gpu's
        residual riir-ai-side modules, which riir-clippy no longer builds).
      - Patches re-pointed at `../riir-infer/vendor` (cubecl-runtime,
        wgpu-hal — the same byte-identical copies the substrate's CI builds).
      - Bookkeeping: clippy's standalone-dep pin 12 → 11 entries; BOUNDARY
        + AGENTS sibling tables both repos; riir-ai BOUNDARY CANONICAL
        riir-clippy row re-narrowed to riir-rag only (the D4 discharge for
        this consumer — re-narrowing what the new structure made
        unnecessary, per 041's T2.x bundling).

      Validation: workspace boundary contract CLEAN (23 repos, 321 edges, 0
      violations, 0 rot); riir-clippy clippy 0 errors at default + ternary +
      full release-set postures, lib 2257/0 release, standalone-dep gate
      PASSED; **the retargeted bench runs end-to-end on Metal** (Bonsai GGUF
      load → prefill 29.2 tok/s → decode → JSONL; the historical band);
      riir-infer: fence gate PASSED (339 .rs), infer-core lib 186/0,
      spec-decode tests 34/0, wasm32 clean, clippy 0 both crates; riir-ai:
      riir-gpu lib 188/0, engine lib 2943/0, wasm32 clean, featured lanes
      (causal_validation + go_bonsai_data) compile, riir-agents + riir-router
      green. Commits: riir-infer `8ecbc7f`, riir-ai `50f644f3d`,
      riir-clippy `cda28b7e`.
- [x] **S6b — LANDED 2026-09-24 at honest scope (the 4090 box, in a
      `riir-train.s6b` worktree — a sibling session was live in the main
      checkout; zero file overlap measured). riir-train `7e0e7d74`.**
      - Landed: the two patch rows (cubecl-runtime + wgpu-hal) re-pointed at
        `../riir-infer/vendor/*` — the S6a move, both forks measured
        byte-identical (`diff -r` rc=0, provably a content no-op).
      - The slice premise measured WRONG on this box: only **12 of riir-train's
        81** `riir-gpu/<feature>` forwards are mirrored in riir-infer-gpu
        (which carries 37 features); the other **69 are training families**
        (optimizers, losses, lora variants) that stay riir-ai-side BY DESIGN —
        riir-infer is PUBLIC (Research 003) and training IP must not move
        there. The full edge-drop (`riir-gpu` gone from riir-train) is
        therefore blocked on the owner decision of where the 69 training
        families live long-term: keep riir-gpu forever (facade stays the
        single correct surface) or migrate them into riir-train-gpu (then the
        inference imports swap wholesale). A PARTIAL swap now — inference
        imports direct + training via facade — was evaluated and REJECTED:
        two vocabularies for the same crate family in the same files is
        churn, not progress (~320 inference import sites vs 99 `lora::`
        training sites interleaved).
      - TrainingProvider adjudicated: **stays engine-side** (the acceptable
        residue). Measured: ZERO impls in riir-train, ZERO engine-side
        consumers — a design seam, not live surface. The two stale
        "Implements TrainingProvider" crate descriptions + lib.rs doc blocks
        corrected (riir-train `7e0e7d74`); the trait's own doc claims fixed
        engine-side (this repo).
      - The arc-swap genlock row STAYS: riir-train's riir-engine edge remains
        (356 real usages of the LoRA/training families in rt-gpu src alone).
      - Validation (4090, isolated target dir): workspace check rc=0;
        `ternary_backward_cudarc` CUDA posture rc=0; clippy `--workspace -D
        warnings` rc=0; boundary contract CLEAN (23 repos / 321 edges / 0
        violations / 0 rot). All-features skipped: no feature-surface change
        (byte-identical patches + docs only).
- [x] **S7 — LANDED 2026-09-24 (the 4090 box; riir-ai `498b1732b9`).**
      - Dead-dep cleanup: riir-gpu's REQUIRED `riir-gpu-async` row removed —
        measured zero-reference from riir-gpu (code AND features; live
        consumers dep it directly: riir-engine's analytic_lattice_runtime,
        riir-train-gpu). The T2 audit's other candidate **riir-router is NOT
        dead** — `dynamic_pair_routing` is DEFAULT-on and forwards
        `dep:riir-router` — the row stays (verdict recorded at the manifest
        site).
      - The re-export flip measured COMPLETE by compilation (no residue):
        every live consumer path resolves — riir-train proven same-day at
        default + CUDA postures; riir-gpu at default + all-features green.
        ~~The one file-level gap (`gemm_ternary_simdgroup_smem_cubecl.rs`) is an
        undeclared ORPHAN inside riir-infer itself~~ **REFUTED 2026-09-24
        (the 4090 box, canary-proven)**: the file is NOT an orphan — it is
        declared by its PARENT `gemm_ternary_simdgroup_cubecl.rs` via
        `#[path = "gemm_ternary_simdgroup_smem_cubecl.rs"] pub mod smem;`
        (the deliberate zero-lib.rs-surface pattern), the production dispatch
        (`ternary_deltanet_gpu_forward.rs` prefill_project) calls
        `GemmTernarySimdgroupCubeCL::launch_smem` live, and an E0425 canary
        planted in the file REDS `cargo check -p riir-infer-gpu --features
        ternary_gemm_simdgroup` (the compile-to-nothing disproval — a bare
        `Finished` line never shows `-p riir-infer-gpu` for this unit, so
        only the canary distinguishes). The S7 audit's file-vs-lib.rs
        comparison was blind to the `#[path]` indirection — its "orphan"
        verdict was an instrument artifact, not a fact about the tree.
        `kernels` (the WGSL shader dir) + `test_gpu_support` (private mod)
        are STAYS, not residue.
      - CI ripple: none — riir-train's CI already clones riir-infer; the S6b
        patch re-point changed nothing there.
      - Riding fix: the S6b trait-doc edit (75b7630b38) had grown
        training_provider.rs 235→239 and red the C6 modelless-residue
        ratchet — compacted to net-zero (235); `ci_modelless_residue.sh`
        clean. Boundary contract rerun CLEAN (23 repos / 321 edges / 0
        violations / 0 rot).
- [ ] S8 — P5 (op-layer unification) AFTER P2 lands the laya lane: one
      Backend trait over the hand-MSL and CubeCL op layers, A/B vs the
      chart numbers, delete the loser. **SEQUENCING CALL 2026-09-24 (reflex
      session): deferred behind reflex issue 020 T5** — the T5 batched
      forward (riir-infer `da30007`) landed first so S8's A/B measures
      settled numbers; re-open after 020's publishable batched-vs-loop A/B
      (Bench 006 Addendum 7). Nothing of S8 landed; mirrors the note in
      reflex `.issues/008` T7.

## S3 ready-notes (pre-adjudicated 2026-09-23, from the S2 edge audit)

- The ternary gemv/gemm families are SELF-CONTAINED around the hub
  `gemv_ternary_cubecl` (TernaryHandle, TERNARY_* consts,
  prepare_block_contiguous_u32, U32_PER_BLOCK_GROUP): every gemm_* and
  prefill_cuda_* member only imports intra-family paths — no
  `riir_engine` refs in prod code across the whole S3 set (grep
  verified).
- **Do NOT pull the qwen38 prefix/kv gates into S3**:
  `qwen38_prefix_cache` → `crate::qwen38_dense_cudarc` and
  `qwen38_kv_f16_gates` → `crate::cudarc_kernels` +
  `qwen38_dense_cudarc` — both SEAM modules (gguf_loader edge).
  They ride the cudarc/qwen38 SEAM adjudication (S4 window), not the
  CLEAN family.
- `prefill_cuda_attention` → `prefill_cuda_deltanet::{LogForm,
  SigExp}` — intra-family, both move together.
- Manifest deltas S3 will need (the S2 pattern): the `cudarc` dep for
  the CUDA raw family + the macOS metal-tensor family's `.metal` files
  (`gemm_ternary_metal_tensor.metal`, `gemm_ternary_metal_wgpu.metal`)
  move WITH their modules — check their `include_str!`/`include_bytes!`
  paths on copy. ~~`gemm_ternary_simdgroup_smem_cubecl` (present in src,
  absent from the T2 table rows) belongs to the gemm family — move with it
  and note the audit gap.~~ **Corrected 2026-09-24**: the "absent from the
  T2 table rows" reading was the same file-vs-lib.rs instrument artifact —
  the module is declared via `#[path]` from the parent (`pub mod smem`),
  so it moves with the parent and no table row was ever missing.
- Feature mirror list to re-check on move: ternary_gemv,
  ternary_gemm_batched, ternary_gemm_simdgroup,
  ternary_attention_batched_prefill, ternary_deltanet_chunked_prefill,
  cuda_graphs_forward, q4k_rowtiled_gemv (already mirrored),
  metal_tensor paths — each cfg found in the moved code needs its
  riir-infer-gpu mirror + riir-gpu forward, or clippy -D warnings reds
  on unexpected_cfgs.

## S4 ready-notes (pre-adjudicated 2026-09-23, from the S3 edge audit)

- The SEAM window is the qwen38/cudarc cluster: `qwen38_dense_cudarc`,
  `qwen38_verify_mma`, `qwen38_prefix_cache`, `qwen38_dflash2_gpu`,
  `cudarc_kernels`, `prefill_cuda_full` — every one gated
  `ternary_gemv_cuda_raw + not(macos)` (dflash2 is ungated CPU), every
  one importing `riir_engine::{gguf_loader, types, deltanet::*}` which
  already re-resolves to riir_infer_core via the T1.2 re-exports. The
  SEAM rewrite is mechanical (`riir_engine::` -> `riir_infer_core::`
  + enable infer-core's deltanet_inference/q2_0_ternary_bridge
  features); the real adjudication is `cudarc_kernels`' `backward.rs`
  (rtg training surface — riir-train-gpu consumes `BackwardKernels`
  through the root re-export) and `prefill_cuda_full`'s ~5.6k lines.
- `ternary_deltanet_gpu_forward{,_cudarc}` + `deltanet_rotation_*` +
  `ternary_tree_verify_driver` + `training_activation_cache` form the
  second S4 cluster; `training_activation_cache` re-exports
  `riir_engine::deltanet::minimal_activation_cache` at its root —
  that re-export line is the S4/T2.3 hinge (riir-clippy's ternary
  bench consumes TernaryDeltanetGpuForward).
- The deltanet CLEAN kernels (`deltanet_chunked_cubecl`,
  `deltanet_delta_rule_chunked`, `deltanet_tree_verify_cubecl`,
  `deltanet_pre_rec_fused_cubecl`, + `forward_prefill`,
  `forward_flashprefill`, `weight_readback` from the row-95 list)
  can ride S4 or their own slice; their riir_engine refs (if any)
  need the same per-file grep S3 ran before the move.
- Box-state caveat measured 2026-09-23: the riir-gpu all-features lib
  suite is jetsam-killed on the M3 under load (reproduces at HEAD);
  validate moved-kernel behavior through infer-gpu's own lib suite
  (all-features combo) and per-test riir-gpu runs.

## The slice-1 rider the plan did not predict — the consumer-CI ripple

A path dep in a workspace MEMBER manifest is read by every consumer's
cargo. riir-gpu now path-deps `../../../riir-infer`, so every repo whose
CI clones riir-ai (and whose graph loads engine/gpu manifests) must
provision riir-infer too — the carve (engine → riir-infer-core) had
already created this latent gap and no lane fired it (main-only
triggers). Fixed in the same window: riir-ai, riir-clippy, riir-train,
riir-chain (5 job lists), riir-mmorpg-examples, seal-remake. Verified
NOT needed: riir-dapps (its tested feature sets never resolve engine's
manifest), riir-game-sdk (the skip-loud preflight pattern by design),
riir-neuron-db (its CI graph does not reach riir-ai crates),
riir-viewbridge/riir-auth/riir-shader (no engine reach or no clone
list).

## Non-goals

- No P2 work here (reflex sibling WIP owns that sequencing).
- No public opening (T6/P4 — owner-gated; the checklist doc ships in
  this slice so the gate is ready).
- No weights move; riir-infer ships loaders, not checkpoints.
