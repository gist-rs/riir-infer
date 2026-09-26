> **Moved 2026-09-26:** this proposal lives in `riir-infer/.proposals/` now (moved verbatim from riir-ai, same number — riir-ai's ledger keeps 041 allocated). Cross-repo links re-pointed to `../../riir-ai/...`.

# Proposal 041 — `riir-infer`: split the model-based inference layer out of riir-ai

> **Status:** Phase 1 LANDED + Phase 2 DECIDED (2026-08-27, owner call under the perf/sec mandate: **DEFER-to-trigger** — repo promotion waits for a measured consumer-pull, not a calendar; the 4 open questions answered in §Session 5). Phase 1 close-out complete: consumer spot-run green (rte 1448/1448), BOUNDARY.md Owns row, `.docs/02_crates/riir_infer_core.md`, D4 discharged.

Status: **Phase 0 COMPLETE · Phase 1 LANDED (T1.0–T1.2 + T1.5 + close-out) · Phase 2 EXECUTING — T-B FIRED 2026-09-22, P0/P1 (the carve) LANDED same day** (riir-reflex Issue 008: the owner pull for a public `riir-infer` consolidation — reflex consumes the substrate engine-free; promotion runbook now governed there as P0–P5, extending T2.1–T2.4 with the encoder-lane move and the riir-gpu CLEAN/SEAM/STAYS audit that supersedes this proposal's "moved verbatim" row; first-act-of-campaign rule honored — the carve was this campaign's first code act. Carve record: mirror Issue 998; repo gist-rs/riir-infer @ 86a5986; crate moved name-unchanged per §Session 5 Q2, zero consumer edits, Cargo.lock byte-identical; registered as the 23rd contract repo)
(updated 2026-08-27 — §Session 4 execution record + §Session 5 decision record; 2026-09-22 — T-B fired, see riir-reflex Issue 008)
Branch: `develop`
Owner: unassigned (Phase 2 decision taken by delegate under the "best perf/sec prod grade" mandate, 2026-08-27)
Verdict: **GO on Phase 0+1 (landed); Phase 2 = GO-behind-a-pull-trigger** (see §Session 5 + §Verdict)
Unblocked: ~~Issue 741~~ (RESOLVED 2026-08-27 — training residue fully evicted, 0 rows) × ~~Issue 744~~ (RESOLVED + removed)
Precedent followed: [Proposal 025](../../riir-ai/.proposals/025_riir_games_repo_promotion_verdict.md) (measure coupling *first*, DEFER if the seam is too wide — riir-ai `BOUNDARY.md` D3) × [Issue 739 / Bench 723](../../riir-ai/.benchmarks/723_engram_seam_relocation_goat.md) (the reusable seam-relocation shape, 89 → 61 packages) × [Plan 005](../../riir-ai/.plans/005_phase3_code_move.md) (the training eviction this completes)
Related: `BOUNDARY.md` (Owns / May depend on / CANONICAL direction matrix — all three change), `Issue 740` (D2 remainder)

## TL;DR

`riir-gpu` is **313,435 LOC — 25.0% of riir-ai's 1,254,628** — and it serves a
consumer that lives in another repo. Its only **non-optional** dependent in the
entire 15-repo workspace is `riir-train`; its second real dependent is
`riir-clippy` (dev tooling). **Zero** game-product repos depend on it:
`riir-game-sdk`, `riir-mmorpg-examples`, `riir-armageddon`, `riir-unity`,
`riir-dapps`, `riir-chain` have no edge, optional or otherwise. It fails
riir-ai's own `BOUNDARY.md` domain test.

But `riir-gpu` **cannot be split at the crate line**: it reaches back into
`riir-engine` across **95 unique paths in 86 files**, concentrated in
architecture and weight-loading (`types` 83 sites, `transformer` 62,
`deltanet` 40, `quant`+`gguf_loader` 24, `rope`+`gemma_layer`+`llama_layer` 19).
Splitting the crate alone creates a `riir-infer → riir-ai` back-edge, violating
the `BOUNDARY.md` rule that deps flow DOWN into leaves.

**The cut line is one layer lower and it measures clean.** The model layer
inside `riir-engine` (**47,635 LOC**) imports **zero** cognition
(`hla|mag|karc|clr|cgsp|latent_functor|cwm_runtime|entity_cognition` = 0 hits
across all six probed modules), and the reverse seam — cognition reaching down
into it — is only **56 unique symbols across 48 files**. That is *one third* of
the 170-symbol civ surface that made Proposal 025 an honest DEFER.

**Proposed: `riir-infer` = engine model layer (47,635) + `riir-gpu` (313,435) +
`riir-gpu-async` (741) ≈ 362k LOC**, with `riir-ai` becoming a normal downstream
consumer of it.

## The problem this solves

### 1. Two growth axes are welded into one repo

The model layer already carries **7 architecture families**, counted by
identifier census across `riir-gpu/src` + `riir-train-gpu/src`:

| family | refs |
|---|---:|
| gemma (incl. gemma2 / gemma4) | 347 |
| kimi_k3 | 92 |
| bonsai | 81 |
| llama | 64 |
| qwen / qwen3 | 31 |
| minicpm | 4 |
| deepseek | 2 |

Every new model POC lands **entirely** in this layer and touches **zero**
cognition code — that is the same measurement as the clean-cut finding, read
forward. Meanwhile riir-ai's other axis (game mechanics, cognition runtimes,
the commercial moat per Research 003) grows on a completely different clock.
Welded together, every model POC re-resolves the game/cognition graph and every
game change re-resolves 313k LOC of kernels.

The `riir-gpu` file listing is the symptom made visible: ~25 `gemm_ternary_*`
variants, ~26 `gemv_*` variants, `gemma2_*`/`gemma4_*`/`kimi_k3_*`/`llama_*`/
`qwen_*` forward+backward paths, `cudarc_kernels/` (9,224), `ane_prefill/`
(1,539). None of that is a game runtime concern.

### 2. It violates riir-ai's own stated domain test

`BOUNDARY.md` §Owns:

> **Domain test:** does this serve a game runtime concern (NPC cognition, game
> state sync, perception, emotion, spatial queries, engine inference used BY
> games)? NO → it belongs in another repo; file there.

Measured consumers of `riir-gpu`:

| consumer | edge | kind |
|---|---|---|
| `riir-train/Cargo.toml:20` | **non-optional** | training (other repo) |
| `riir-clippy/Cargo.toml:52` | optional, `default-features = false` | dev tooling (other repo) |
| `riir-ai/riir-games:1302`, `riir-agents:34`, `riir-examples:26`, `riir-games-quest:129` | all `optional = true` | in-repo, opt-in |
| `riir-game-sdk`, `riir-mmorpg-examples`, `riir-armageddon`, `riir-unity`, `riir-dapps`, `riir-chain` | **none** | — |

The clause "engine inference used BY games" is the only one that could admit it,
and no shipped game uses it. The in-repo edges are all opt-in, which is
precisely the shape of a dep that *could* live elsewhere.

### 3. `riir-gpu` is not what riir-ai's contract says it is

Verdict v2 (`.plans/005_phase3_code_move.md`) specifies `riir-gpu` = *"Pure
inference + GPU runtime… **ZERO training code**"* and marks the work complete.
Measured: **35,636 LOC** of optimizers, backprop, and `*_train` modules remain
in `riir-gpu` (first hand-audited at 26,963; the Issue 741 T6 gate then found
8,673 LOC the audit had missed), plus **15,022** more inside `riir-engine`'s
model layer. Filed as **Issue 741**, a hard prerequisite — see §Sequencing.

## The proposed design

### `riir-infer` contents (measured LOC)

| crate | source | LOC | contents |
|---|---|---:|---|
| `riir-infer-core` | extracted from `riir-engine` | **47,635** | `transformer/` 19,907 · `deltanet/` 17,500 · `gguf_loader` 2,699 · `linoss` 2,542 · `quant/` 2,066 · `dflash` 932 · `rope` 753 · `spec_types` 566 · `wall` 565 · `gemma_layer` 76 · `llama_layer` 29 |
| `riir-gpu` | moved verbatim | **313,435** | kernels, forward/prefill/decode paths, per-architecture GPU paths — *after* Issue 741 evicts the 26,963 |
| `riir-gpu-async` | moved verbatim | **741** | async composition layer (already `riir-gpu`-internal) |
| **total** | | **~362k** | |

### What does NOT move (measured, not assumed)

- **`fourier/` (16,879 LOC) — game domain, not model.** Files are `blast.rs`,
  `bomber_periodic.rs`, `dungeon.rs`, `economy.rs`, `formation.rs`,
  `blocker.rs`, `events.rs`. It imports **zero** cognition **and zero** model
  layer. It is Fourier feature-encoding of *game events*, and the name is a
  vocabulary trap — grepping "fourier" for model math finds a games module.
  Stays in riir-ai (arguably a future `riir-games-*` candidate; out of scope).
- **The cognition moat** — `latent_functor/` 44,956, `cgsp_runtime/` 21,675,
  `cognitive_branches_runtime/` 13,371, `cwm_runtime/` 11,047,
  `entity_cognition/` 10,632, `karc_bridge/` 8,926, `hla/` 4,094. Research 003
  names this the commercial moat; it stays.
- **`riir-games*`, `riir-net`, `riir-simloop`, `riir-wasm`, the SDKs** — untouched.

### Dep direction after the split

```
riir-games* ─► riir-engine (cognition moat) ─► riir-infer-core ─► [external]
                                    │                  ▲
riir-train ────────────────────────────────────────────┤
riir-clippy ───────────────────────────────────────────┘
                              riir-gpu ─► riir-infer-core
```

One-way, no back-edge. `riir-ai`'s `BOUNDARY.md` matrix gains a row
*"may depend on `riir-infer` (leaf)"*; `riir-train` and `riir-clippy` retarget
their `riir-gpu` path deps and **stop depending on riir-ai for inference at
all** — `riir-train`'s only remaining riir-ai edges become the genuinely
game-shaped ones (`riir-games`, `riir-games-civ`, `riir-data`, `riir-router`).

### Why the name is `riir-infer`, not `riir-gpu`

The layer is not GPU. It is GGUF weight loading, quantization, architecture
definitions, CPU reference paths (`cpu_reference.rs`), and ANE prefill.
`riir-gpu` stays a **crate inside** `riir-infer`. Naming the repo after one
backend would repeat the `fourier` mistake: a name that misroutes greps.

## The measurements this proposal rests on

All taken 2026-08-21 against `develop` at `e1b2b704d`.

**Baseline churn note (same day, before filing).** `develop` advanced to
`4ccd3db6f` while this proposal was being written (Issue 734 Arm 10/11 mma GEMM
work, +2,879 lines into `crates/riir-gpu/src/gemm_ternary_i8_mma_cuda_raw.rs`).
Re-measured at `4ccd3db6f`: riir-ai **1,257,289** / `riir-gpu` **315,506** =
**25.1%** (M1), and **M9 Tier A is bit-for-bit unchanged at 26,963** — the churn
landed entirely in a kernel file, touching neither the model layer nor the
training residue. The M1 figures below are the `e1b2b704d` snapshot; the +2,661
LOC delta moves no conclusion, and the fact that a single sibling session added
2,879 LOC to one `riir-gpu` kernel in an afternoon is itself evidence for §1
(the two-growth-axes problem this proposal exists to fix).

| # | measurement | value | method |
|---|---|---|---|
| M1 | riir-ai total / `riir-gpu` share | 1,254,628 / 313,435 = **25.0%** | `wc -l` over `crates/**/*.rs` |
| M2 | `riir-gpu` → `riir_engine` coupling | **95** unique paths, **86** files | `grep -rhoE 'riir_engine::[A-Za-z0-9_:]+' \| sort -u` |
| M3 | model layer → cognition | **0** across `transformer`, `deltanet`, `quant`, `gguf_loader`, `rope`, `dflash` | grep for `crate::(hla\|mag\|karc\|clr\|cgsp\|latent_functor\|cwm_runtime\|entity_cognition)` |
| M4 | cognition → model layer (the seam to build) | **56** unique symbols, **48** files | grep `crate::(transformer\|deltanet\|quant\|gguf_loader\|rope\|gemma_layer\|llama_layer\|dflash)::X` |
| M5 | model layer size | **47,635** LOC | per-module `wc -l` |
| M6 | architecture families | **7** | identifier census over `riir-gpu` + `riir-train-gpu` |
| M7 | non-optional `riir-gpu` consumers | **1** (`riir-train`) | `grep 'riir-gpu *= *{'` across all Cargo.toml |
| M8 | game-product repos depending on `riir-gpu` | **0** | same, over 6 product repos |
| M9 | training residue in `riir-gpu` | **26,963** Tier A / 45,668 training-named | Issue 741 |

**M4 is the decision number — and Phase 1 has now settled it at 44, not 56.**
Proposal 025 DEFERred the games split at civ's **170** unique symbols. The
Phase 1 re-measure (T1.3, §Phase 1 execution record) came in at **44** — 26% of
the gate, and *lower* than the 56 grep estimate because the measured dependency
closure absorbs `types.rs`/`simd/` traffic that previously crossed the seam. The
DEFER trigger this row was written to catch did not fire.

Additional Phase 1 measurements, recorded so they are not re-derived:

| # | measurement | value |
|---|---|---|
| M10 | dependency closure that must move together | **61** files / **45,572** LOC |
| M11 | glob imports from moved modules (the grep blind spot) | **0** — so M4 is trustworthy, not a floor |
| M12 | `pub(crate)` items inside the closure needing promotion | **12** distinct (47 occurrences) |
| M13 | features / cfg sites the closure carries | **33** features, **330** cfg sites; modules declared **ungated** in `lib.rs` |
| M14 | evictable training code inside riir-engine's model layer (T1.0) | **15,022** LOC = 33% of the closure |
| M15 | `cargo check -p riir-engine --lib` baseline | **45.8s** — iteration cost for T1.1 |
| M16 | total training-shaped LOC in riir-gpu + riir-engine (T6 gate) | **50,658** residue + 1,670 ruled keepers |

Session 3 measurements (2026-08-22, `f42bebec2`) — the T1.0 blocker census:

| # | measurement | value |
|---|---|---|
| M17 | T6-gate ledger, ratcheted | **31,358** residue + 1,957 keepers (from 50,658) |
| M18 | Tier C blockers **inside** `riir-gpu/src` | **7** files (was recorded as 3 — `mod.rs`-only grep) |
| M19 | Tier C blockers **outside** any `src/` (T6-gate blind spot) | **15** files / **14,591** LOC — `riir-examples/examples` ×9, `riir-gpu/tests` ×5, `riir-engine/tests` ×1 |
| M20 | total Tier C blocker surface | **22** files |
| M21 | of the 9 blocking `riir-examples` examples, how many are feature-gated | **8 of 9** (the 9th fixed — Issue 744 T4) |
| M22 | Tier C LOC in a **default** build (`cargo metadata`, 2026-08-22) | **1,507** before Issue 744 T2, **0** after — `riir-engine` has **59** default features (closure 76) **including `deltanet_inference` and `lora_still`**. An earlier `0 — riir-engine has no default features` reading was a regex artifact on a 14 kB single-line array; retracted. `gemma_lora`/`gemma4_train` are genuinely not default. |
| M23 | Tier C LOC gated only by an *inference* feature (fixed, Issue 744 T2) | **1,507** — `deltanet/backward` 921 + `deltanet/lm_head_lora_train` 586, both were riding in on `deltanet_inference` |

## Sequencing — and why the order is load-bearing

- [x] **Phase 0 — clear the training residue. COMPLETE 2026-08-27.** Issue 741
      closed at 0 tracked-residue rows / 20,120 LOC ruled keepers (each keeper
      with a measured reason); Tier A 35,636 LOC + Tier B 11,762 LOC relocated
      riir-gpu → riir-train-gpu; Tier C 15,022 LOC evicted from the engine
      model layer → riir-train-engine; TAIL P1–P6 swept the stragglers;
      enforcement shipped as `ci_modelless_residue.sh` wired into
      `ci_boundary_contract.sh` as **C6**. Issue 744 resolved + removed.
- [~] **Phase 1 — extract `riir-infer-core` as a crate, still inside riir-ai.**
      The Proposal-025-Phase-0 move: pay the cheap, reversible cost first and
      let the compiler audit the seam. **Both gate tasks (T1.3, T1.4) are DONE
      and both PASS** — see §Phase 1 execution record. T1.1/T1.2 are RE-SCOPED
      by a Phase 1 finding and now depend on a new prerequisite (T1.0).
  - [x] **T1.0 (NEW, blocking T1.1) — VERIFIED PASS 2026-08-27 (Session 4): all
        blockers cleared on the clean tree.** The 7 riir-gpu/src blocker files:
        6 moved wholesale by Issue 741 Tier A (`gemma2_cubecl_train/mod.rs`,
        `gemma4_cubecl_train/{mod,backward_gpu}.rs`,
        `gemma4_q4k_train/{mod,forward_batched,backward}.rs`); the 7th
        (`ternary_deltanet_gpu_forward_cudarc.rs`) got its **B-split** — now
        **zero `use riir_engine::` imports** remain in the file, and riir-gpu's
        Cargo.toml carries no riir-train dep (only comments documenting moved
        tests). The 15 non-`src/` blockers (riir-examples ×9, riir-gpu/tests ×5,
        riir-engine/tests ×1): swept by Issue 741's TAIL P1–P6 ("the 9
        test/example targets"). Original finding kept for the record: Tier C
        exports 102 pub items, of which 16 were imported from `riir_engine`
        by riir-gpu (bare word-grep said 58 — 3.6× over-count; only qualified
        `riir_engine::` imports count); corrected 2026-08-22 to 22 files
        (7 src + 15 non-src). Executable order was
        **741 T4 → B-split → T1.0 → T1.1 → T1.2 → T1.5** — the first two DONE.
  - [x] T1.1 — create `crates/riir-infer-core`, move the measured closure.
        **LANDED 2026-08-27** in two commits (Chunk A leaves `942f9b700`;
        Chunk B blob `6ba023c99` — atomic per the 3-cycle). 44 files /
        30,373 LOC. Compiler-audit findings + fixes recorded in §Session 4
        execution record (5 qualified-path deps, the transformer_still
        bridge re-declaration, 22 visibility promotions, 2 safety docs).
        **RE-SCOPED:** not "the 11 modules (M5)". The measured dependency
        closure is the 11 modules **minus** `transformer_still.rs` (a
        `lora_still` consumer — Verdict v2 keeps `lora_still` in riir-engine)
        and **minus** `linoss/` (its `mcts_adapter.rs` needs
        `crate::game_state`), **plus** four leaves the model layer cannot
        compile without: `types.rs` (150 LOC — pure leaf), `simd/` (20),
        `ternary_layer.rs` (130), `safetensors_loader.rs` (583).
        **RE-MEASURED 2026-08-27 (Session 4, clean tree @ `9a98a5efd`):
        44 files / 30,373 LOC** (matches the ~30,550 prediction).
        **267 cfg(feature) sites / 32 distinct features** must be declared and
        forwarded; module gates in `lib.rs`: `deltanet` ←
        `deltanet_inference`, `ternary_layer` ← `ternary_inference`, all
        others move ungated. **Move order is constraint-shaped by a 3-cycle**
        (`transformer → ternary_layer → deltanet → transformer`):
        Chunk A = leaves (`types`, `simd`, `rope`, `gemma_layer`,
        `llama_layer`, `quant`, `wall`, `safetensors_loader`; 13 files /
        ~4,225 LOC) moves first; Chunk B = the interdependent blob
        (`transformer` + `deltanet` + `ternary_layer` + `spec_types` +
        `dflash` + `gguf_loader`; 31 files / ~26,148 LOC) moves atomically.
        External deps (measured, any-position refs): katgpt-core 50,
        bytemuck 45, half 29, anyhow 23, katgpt-speculative 20, rayon 17,
        katgpt-forward 7, memmap2 4, katgpt-attn 4 (flashmemory_gqa only,
        optional), katgpt-transformer 3, katgpt-quant (turboquant re-export,
        optional).
  - [x] T1.2 — re-export from `riir_engine` at the **same paths** so no consumer
        import changes in the same commit (the Issue 739 / Bench 723 shape).
        **LANDED** alongside each chunk: `pub use riir_infer_core::X;` in
        engine lib.rs (gates preserved for `deltanet`/`ternary_layer`/
        `lora_still`-adjacent decls). Zero consumer edits required — riir-gpu
        (52 files importing `riir_engine::`), engine-internal cognition, and
        sibling consumers all compile unchanged.
  - [x] **T1.3 — re-measure M4 against compiler reality. DONE: 44 distinct seam
        symbols. PASS** (gate ≤ ~170). Measured with brace-import expansion
        (`use crate::types::{A, B}` — the naive regex misses these; 14 such refs
        exist) across the 443 remaining engine files, 63 of which touch the
        moved modules. **Zero glob imports** (`use crate::<mod>::*`) — the main
        grep blind spot is confirmed absent, so 44 is trustworthy rather than a
        floor. Only 12 distinct `pub(crate)` items exist inside the whole move
        set, bounding the visibility-promotion work.
  - [x] **T1.4 — assert one-way. DONE: ZERO real back-edges. PASS.** The only
        non-move-set path the closure references is `crate::turboquant`, which
        is `pub use katgpt_quant::turboquant` (`lib.rs:256`) — an **upstream
        re-export**, so in `riir-infer-core` it becomes a direct `katgpt-quant`
        dep: correct direction, not a back-edge. The 4 `riir_gpu::` and 2
        `riir_engine::` hits inside the closure are **doc comments only**
        (`transformer/gemma2_train/optimizer.rs` ×4,
        `deltanet/minimal_activation_cache.rs`, `transformer/mod.rs`) — zero
        code deps. M3's "0 cognition imports" holds under the expanded closure.
  - [x] T1.5 — measure package-count delta for `riir-clippy` and `riir-train`
        (the Bench 723 metric, which made 739 a provable win rather than a
        tidy-up). **Measured 2026-08-27**: riir-clippy default 68 packages /
        zero riir-* edges (untouched); riir-train 241 packages with +1
        (infer-core) — the rlib-boundary cost. The Bench-723-style shrink is
        a Phase 2 event (consumer graphs change only at repo promotion);
        cold build-time A/B honestly not re-measured in Phase 1. Full table
        in §Session 4 execution record.
- [ ] **Phase 2 — promote to a `riir-infer` repo.** Gated on T1.3 ≤ ~170 and
      T1.4 clean. Workspace **15 → 16 repos**.
  - [ ] T2.1 — new repo + root `BOUNDARY.md` (own/not-own/allowlist/direction
        matrix/drift ledger), per the C0b family-naming rule so
        `ci_boundary_contract.sh` discovers it automatically.
  - [ ] T2.2 — riir-ai `BOUNDARY.md`: remove `riir-gpu`/`riir-gpu-async` from
        §Owns, add `riir-infer` to §May depend on, add the direction-matrix row.
  - [ ] T2.3 — retarget `riir-train` + `riir-clippy` path deps.
  - [ ] T2.4 — `./scripts/ci_boundary_contract.sh` exit 0 across 16 repos.
- [ ] **Phase 3 — record.** `.benchmarks/NNN_riir_infer_split.md`: before/after
      package counts, build-time delta, the true M4. If Phase 2 is declined,
      the decision record + trigger conditions land here (the Proposal 025
      pattern — a DEFER with recorded triggers is a result, not a failure).

## Phase 1 execution record (2026-08-21, develop `a9a286b2e`)

**Both Phase 2 gate conditions are now measured and both PASS.**

| gate | threshold | measured | verdict |
|---|---|---|---|
| **T1.3** seam under compiler visibility | ≤ ~170 symbols (civ parity) | **44** distinct symbols, 63 files, 14 brace-form refs expanded, **0** glob imports, 12 `pub(crate)` items to promote | ✅ **PASS** — 26% of the gate |
| **T1.4** zero back-edge | 0 | **0** real; only `crate::turboquant` = upstream `katgpt_quant` re-export; 6 `riir_gpu::`/`riir_engine::` hits are doc comments | ✅ **PASS** |

### The dependency closure (what actually has to move)

M5's 11 modules do not compile alone. The measured closure:

| change | modules | LOC |
|---|---|---:|
| M5 baseline | 11 modules | 47,635 |
| **− exclude** | `transformer_still.rs` (consumes `crate::lora_still`, which Verdict v2 keeps in riir-engine) | ~~−1,169~~ **−356** |
| **− exclude** | `linoss/` (its `mcts_adapter.rs` needs `crate::game_state` — game domain) | −2,542 |
| **+ add** | `types.rs` 102 · `simd/` 20 · `ternary_layer.rs` 130 · `safetensors_loader.rs` 583 | +835 |
| **= closure** | **61 files** | ~~**45,572**~~ **46,385** |

**Closure figure CORRECTED 2026-08-22 (Issue 744 T10).** The
`transformer_still.rs` exclusion was pinned at **1,169 LOC**; the file measures
**356**, and `git show a9a286b2e:…` confirms it was 356 at *the exact commit
this table was measured against* — a 3.3× overstatement, not a since-shrunk
file. The closure is therefore **46,385**, not 45,572, and the post-T1.0
inference-only figure moves ~30,550 → **~31,363**. Neither shifts a verdict
(T1.3 = 44 and T1.4 = 0 are unaffected), which is exactly why an
un-recomputed row like this survives review. Found only because T10 went looking
for `transformer_still`'s gate and could not find it in `lib.rs` — it is
declared `#[path = "../transformer_still.rs"] mod transformer_still;` inside
`transformer/mod.rs:6`, invisible to a `lib.rs` grep.
| **− T1.0** | training code found inside the model layer (below) | −15,022 |
| **= inference-only** | | **~30,550** |

The `types.rs` result is the load-bearing one: **253 reference sites** made it
look like the split's biggest obstacle, but it is **102 LOC in 1 file with zero
`crate::` imports** — a pure leaf. Pulling it into the closure converts 253
would-be back-edges into intra-crate references. `simd/` (20 LOC) is the same
shape. This is why the grep-56 estimate and the measured-44 differ: the
expanded closure absorbs traffic that previously crossed.

### T1.0 — the finding that re-scopes T1.1

**15,022 LOC of evictable training code lives inside riir-engine's model layer**
— 33% of the closure (plus `training_provider.rs`, a ruled keeper). Issue 741 audited `riir-gpu` only; this is a second site:

| path | LOC | note |
|---|---:|---|
| `transformer/gemma2_train/` | 5,021 | ~~**UNGATED**~~ → **gated `#[cfg(feature = "gemma_lora")]`** (corrected 2026-08-22, Issue 744 §4 — mis-read, never a regression) |
| `transformer/gemma4_train/` | 2,937 | |
| `transformer/train_shared.rs` | 773 | |
| `deltanet/{model_backward_recompute,layer_backward,attention_backward,backward,model_backward,full_backward}.rs` | 4,142 | backprop |
| `deltanet/{lm_head_lora_train,qv_lora_train}.rs` | 1,149 | LoRA training |
| `training_provider.rs` | 235 | **KEEPER** — Verdict v2 names this the correct engine-side seam |

Extracting `riir-infer-core` before evicting these would carry gradient descent
into a modelless-inference crate — the Phase 0 error, one layer up. Hence T1.0.

### Session 2 (2026-08-21) — what changed

**Phase 1 remains `[~]`; T1.3/T1.4 still the only completed gate tasks, and they
still PASS.** What session 2 added:

1. **The requested execution order was proven non-executable.** T1.0 cannot go
   first — see the T1.0 entry above for the 16-symbol / 3-file / cycle proof.
   Executable order: **741 T4 → B-split → T1.0 → T1.1 → T1.2 → T1.5.**
2. **Issue 741 T1 LANDED instead** (it was the unblocked head of that chain):
   10,586 LOC / 6 modules + 16 test files moved riir-gpu → riir-train-gpu, both
   crates clean across every gating feature, T6 ledger ratcheted
   **50,658 → 43,138**. The "cycle" that blocked it was a false premise — all 3
   `GpuBackwardPass` consumers are training, so they moved too.
3. **A third training site is now on record:** `riir-engine`'s Tier C is not the
   end. `ternary_deltanet_gpu_forward_cudarc.rs` (6,043 LOC, riir-gpu) is
   `*_forward_*`-named and carries 8 backward fns — invisible to a name-based
   gate. Treat the T6 ledger as a floor, not a census.

### Bench 672 / 682 / 686 — NOT re-measured, and why (with the stronger argument)

Issue 741's acceptance criteria require these three to re-measure unchanged.
**They were not run.** All three are 4090/CUDA-only — Bench 672 pins
*"RTX 4090 (Ada, sm_89), CUDA 13.3, driver 610.62, WDDM"* and the MSVC
toolchain `1.95.0-x86_64-pc-windows-msvc`; 682 and 686 pin
*"4090 / cudarc (CUDA 13.3), MSVC"*. This session ran on m3 Metal, where they
cannot execute. Running them on the 4090 was also inadvisable: that box had
**uncommitted Issue 734 Arm 12 work in `crates/riir-gpu/src/{lib.rs,
prefill_cuda_full.rs}`**, so a build there would measure a mixed tree, not this
change.

**The available argument is stronger than a noisy re-measurement anyway.** The
entire non-deletion diff to `riir-gpu/src` is **6 `pub(crate)` → `pub`
visibility widenings** (`GpuForwardPass.ia3_inputs`, `.scratch`, and 4
`DFlashForward` accessors) plus comments and the removed `mod` declarations.
Visibility is resolved at compile time and cannot alter codegen, so it cannot
move a tok/s number. Nothing on the decode path was touched — everything deleted
was backprop or a trainer. `cargo clippy -p riir-gpu --lib --tests` is
error-free, and all 22 warnings sit in files this change never opened.

**Still owed:** an actual 4090 run once that box's Arm 12 work is committed.
Tracked as Issue 741 T8's remainder; do not close 741 without it.

### Not done, and why

T1.1/T1.2 (the actual extraction) and T1.5 (package delta) were **not
attempted**. Rationale: measurement re-scoped the move set mid-task, and 33% of
it turned out to be code that should go to `riir-train`, not `riir-infer`.
Building the crate on the pre-T1.0 scope would bake in the defect. The
extraction is mechanical once T1.0 lands — T1.2's re-export trick is confirmed
viable, so it needs zero consumer edits. **The Phase 2 decision does not wait
on T1.1**: T1.3 and T1.4 are the gate, and both are now answered.

### Session 3 (2026-08-22, develop `f42bebec2`) — what changed

**Phase 1's two gate tasks (T1.3, T1.4) still PASS and are untouched. Phase 2
stays HOLD.** What this session changed is Phase 0 throughput plus two
corrections to numbers this proposal was relying on.

**0. The tree this was measured on had to be repaired first.** riir-ai's m3
worktree was carrying an **older whole-repo snapshot**: 12 tracked files reverted
to already-committed revisions (dates spread 08-14 … 08-21) and 12 files deleted
that HEAD still ships — including `crates/riir-gpu/src/qwen38_dense_cudarc.rs`.
It was internally consistent (the stale `lib.rs` did not declare the deleted
module), which is exactly why it looked like WIP rather than drift. Proven
lossless before touching anything: every reverted file's worktree blob was
bit-identical to an ancestor commit's blob for that path, so nothing
uncommitted existed to lose. `HEAD` was 1 behind `origin/develop` and the
fast-forward was blocked by a single file whose worktree content equalled its
**first** commit — six commits of a sibling's Issue 742 work had been clobbered
in the worktree. Restored, fast-forwarded, 0/0 with origin. Only 5 genuinely
dirty files remained (`riir-mcp-client`, another session's work — left alone).
**The lesson is load-bearing for this proposal:** every LOC figure here is a
`wc -l` over a worktree, and this one silently disagreed with `HEAD`. Re-measure
after `git status` reads clean, or the split gets planned against a tree nobody
has.

**1. Issue 741 T9 LANDED (`f42bebec2`) — the residue gate is now a CI gate.**
`ci_boundary_contract.sh` gained **C6**, delegating to
`ci_modelless_residue.sh` and folding its verdict into the contract gate's own
0/1/2 exit contract. C1–C5 audit the dep *graph* and structurally cannot see
training code *inside* an inference crate; C6 closes that half, so a residue
regression now fails the same CI step as a boundary violation. All five
behaviours were **observed** firing (residue-FAIL → violation + exit 1;
residue-hard → exit 2; gate script missing → exit 2, never fail open;
`--list-deps` and `--repo <other>` both skip C6) via a `RIIR_RESIDUE_SH` test
seam added for the purpose — the real gate cannot be made to fail on demand
without editing a tracked file, and an unexercised failure branch is not a gate.
Real run: exit 0, 15 repos, 177 edges, 31,358 LOC tracked residue.

**2. Issue 741 T12 was already done.** The dangling-`#[cfg]` scanner ships as
**R4** inside `ci_modelless_residue.sh`. The task row was stale, not open. Its
residual limit is now recorded rather than assumed away: R4 walks only `lib.rs`
+ `mod.rs`, so a dangling attribute in an ordinary `.rs` file is still invisible.

**3. Two numbers in this proposal were wrong. Both corrected in place above,
both filed as `Issue 744`.**

| claim as written | measured 2026-08-22 |
|---|---|
| T1.0's blockers are "exactly **3** riir-gpu files" | **7** in `riir-gpu/src` — the original grep matched only `mod.rs`, missing `gemma4_q4k_train/{mod,forward_batched,backward}.rs` + `gemma4_cubecl_train/backward_gpu.rs`. A 2.3× under-count, and it routes the chain through the **cycle-blocked** T4c2 seed, so T1.0 is blocked *harder*, not softer. |
| `transformer/gemma2_train` is "declared **UNGATED**" | **Gated `#[cfg(feature = "gemma_lora")]`** — and it already was at `a9a286b2e`, the exact commit this proposal measured. Line 41 is the `pub mod`; line **40** carries the attribute. A mis-read, not a since-fixed regression. |

The second one had teeth: Issue 741 T10 used "largest AND ungated" to pick
`gemma2_train` as the first thing to evict. **And the first replacement for it
was also wrong, which is worth recording rather than quietly fixing.** The
follow-up claim — that `deltanet/lm_head_lora_train` is the genuinely ungated
module, "compiled into every build, default included" — does not survive one
more step of the same check — **but that correction was itself wrong, and is
retracted here.** `cargo metadata` (2026-08-22) reports **59 default features**
for `riir-engine`, closure **76**, including **`deltanet_inference`** and
**`lora_still`**. The earlier "no default features" reading came from a Python
regex that silently failed on a **14,288-character single-line** array. So
`pub mod deltanet` is default-ON and **1,507 LOC of BPTT + AdamW were in every
default build** until Issue 744 T2 gated them; the `lora_still` training half
likewise until T10 stage 2. What survives of the original correction is narrower
and still true: `transformer/gemma2_train` carries its own
`#[cfg(feature = "gemma_lora")]` and `gemma_lora` is not default.

What *is* real, and is now fixed (Issue 744 T2): `deltanet/lm_head_lora_train`
(586) and `deltanet/backward` (921) carried no `#[cfg]` of their own, so their
only gate was `deltanet_inference` — an **inference** feature dragging in
**1,507 LOC of BPTT + AdamW**. That is the Issue 741 T4a finding inverted, and
it is exactly the class of defect this proposal exists to stop shipping into a
repo named for modelless inference. Both now carry
`#[cfg(feature = "deltanet_ternary_inference")]`, matching their 6 sibling
backward modules; verified rc=0 / 0 errors across five feature arms including
the load-bearing `deltanet_inference`-alone case, plus clippy clean with and
without the feature. T10's "do the ungated one first" rationale is dropped
entirely — sequence it by the blocker graph.

**Methodology rule adopted, in its corrected form:** a gating claim sourced from
a bare `grep 'pub mod X'` is inadmissible, and `grep -B2` is **necessary but not
sufficient** — it catches the attribute missing from the matched line (the R4
bug inverted: there an attribute lost its item, here an item was read without
its attribute) and still misses the *parent* module's gate and the crate's
`default` list. Both errors above were the same mistake at different depths.

**4. A fourth training site is now on record — and it is what actually blocks
T1.0.** Issue 741's gate resolves every ledger row as `crates/$crate/src/$path`,
so it is blind by construction to `tests/`, to `examples/`, and to every crate
beyond `riir-gpu`/`riir-engine`. Measured in that blind spot: **15 files /
14,591 LOC** importing Tier C training paths — `riir-examples/examples` ×9,
`riir-gpu/tests` ×5, `riir-engine/tests` ×1. Honest scoping: 8 of the 9 examples
*are* feature-gated, so this is mostly a **relocation blocker**, not a
default-build mandate violation. But it is a hard one: 9 of the 15 are
`riir-examples` examples, and riir-ai must not gain a `riir-train` dep to keep
them compiling once Tier C moves. **Total Tier C blocker surface: 22 files**
(7 in `src` + 15 outside). Their destination is an owner call — Issue 744 T5 —
and Issue 741 T10 cannot start before it. Treat the 31,358-LOC ledger as a floor
on **two** axes now (name-based pattern *and* `src/`-only path scope), not one.

**5. One concrete defect found and fixed (Issue 744 T4).**
`riir-examples/examples/bonsai_lora_accuracy_parity_arm_c.rs` (4,208 LOC)
declared no `required-features` while being wrapped in a crate-level
`#![cfg(any(...))]`, so on default features it compiled to nothing and
`cargo check -p riir-examples --examples` failed **`E0601: main function not
found`**. All 8 sibling examples declare theirs; the needed feature already
existed. Fixed with one line, verified in **both** directions (default-features
`--examples` rc=0 / 0 errors, previously E0601; with-feature build rc=0 / 0
errors, proving the required-feature named is the right one).

**6. Two measurement traps, recorded because both nearly shipped as facts here.**
An `awk` block-parser over `Cargo.toml` reported all 9 examples as ungated; a
TOML-aware parse showed **8 of 9 declare `required-features`**. And a
feature-closure walk of the `[features]` table "proved" the failure would be
unresolved imports; the actual failure was E0601, a different mechanism. A
`zsh` `for m in $MODS` loop also silently produced one concatenated iteration
instead of ten (no word-splitting), and a nested `while read` + `$(git …)` loop
emitted `command not found: git` and mislabelled every file as a real edit.
**A feature or gating claim must be settled by a build; a measurement loop must
be shown to have iterated.**

**Not done, and why unchanged:** T1.0/T1.1/T1.2/T1.5 remain blocked. Session 2's
executable order (**741 T4 → B-split → T1.0 → T1.1 → T1.2 → T1.5**) now needs
one more node in front of T1.0: **Issue 744 T5** (where the 15 non-`src`
blockers go). Bench 672/682/686 are still owed an actual 4090 run — unchanged
from Session 2, and this session ran on m3 Metal.

## Session 4 (2026-08-27, 4090 box, develop `9a98a5efd`) — blockers cleared, T1.0 verified, Phase 1 executing

**The tree was clean and synced** (`HEAD == origin/develop`, zero dirty
files) and **no cargo/rustc/trainer processes were running** — the
`bonsai_clippy_l4_sft_train` window (Issue 766) has ended, so the CPU-only
Phase 1 work runs alongside nothing. Verified before any edit.

**T1.0 verification — every blocker cleared:**

| Session 3 blocker | State on clean tree |
|---|---|
| 7 riir-gpu/src Tier-C importer files | 6 moved wholesale by 741 Tier A; `ternary_deltanet_gpu_forward_cudarc.rs` got its B-split (**0** `use riir_engine::` imports remain) |
| 15 non-`src/` files (14,591 LOC) | swept by 741 TAIL P1–P6 ("the 9 test/example targets" et al.) |
| gemma4_q4k_train cycle-blocked seed | gone from riir-gpu/src entirely |
| riir-gpu → riir-train dep risk | none — Cargo.toml carries only comments documenting moved tests |
| engine model layer Tier C (15,022 LOC) | **zero** `*train*`/`*backward*` files remain under `transformer/` + `deltanet/` (only runtime `gemma2_lora.rs`/`gemma4_lora.rs` stay — inference primitives per the LoRA verdict) |

**Re-measurement on the clean tree (Session 3's own rule):**

- Move set: **44 files / 30,373 LOC** (linoss/ wholly excluded per re-scope;
  the proposal's ~30,550 prediction holds)
- M3 cognition imports: **0** (re-verified)
- T1.4 back-edges: **0 real** (only `crate::game_state` ×1 in the excluded
  `linoss/mcts_adapter.rs` + `crate::turboquant` ×4 = upstream
  katgpt-quant re-export)
- Inter-module graph: **3-cycle** `transformer → ternary_layer → deltanet →
  transformer` → two-chunk move order (leaves first, blob atomic)
- Feature surface: **267 cfg sites / 32 features** (6 pure-local: `hla`,
  `raven`, `gemma_lora`, `linoss_game`→n/a now, `lora_still`,
  `rotary_value_embedding`; the rest forward to katgpt crates and must be
  mirrored on infer-core's copies)
- `use super::` audit: all 48 sites intra-module (move with their parents) —
  no depth-1 file reaches engine root via `super`
- `katgpt_types::` ref ×1: doc comment only, no dep needed

**Also current:** `.issues/` is empty (741/744/753/754/764 all closed);
Plan 548 (llama.cpp+vLLM closure) is PLANNING-ONLY with GPU items gated on a
window that has ended — its file references will be updated in the same
commit series as the move if any path it names changes.

### Session 4 execution record — T1.1 + T1.2 LANDED (same day, 4090 box)

Four commits, each pushed immediately (rebased over 2 sibling pushes en
route): `464b86ec2` (Session 4 docs) → `d07459d6d` (scaffold: crate +
workspace member + engine dep + 31 feature forwards, both crates check
clean) → `942f9b700` (Chunk A leaves, 13 files / 4,225 LOC) → `6ba023c99`
(Chunk B blob, 31 files / 26,148 LOC, atomic — the 3-cycle). The full
move set is **44 files / 30,373 LOC**, exactly as re-measured.

**What the compiler audit caught (the T1.1 philosophy working):**

| finding | fix |
|---|---|
| `serde`/`serde_json`/`thiserror`/`blake3`/`log` referenced via derive attributes + qualified paths only (invisible to a `use`-statement census) | declared in infer-core's Cargo.toml (thiserror surfaced ONLY under `--features q2_0_ternary_bridge` — the feature-on gate is load-bearing, not ceremony) |
| `transformer/mod.rs` carried `#[cfg(feature="lora_still")] #[path="../transformer_still.rs"] mod transformer_still;` — a child-module decl for a file that must STAY engine-side (it imports `crate::lora_still`) | decl removed from the moved mod.rs; `transformer_still` re-declared top-level in engine lib.rs under the same gate; zero code callers existed (dead Plan-267 wiring preserved as-is) |
| 22 × E0603/E0616: engine cognition (`hla/forward.rs`, `causal_validation/gemma2.rs`, `latent_steering_bridge.rs`, `transformer_still.rs`) consuming `pub(crate)` items of the moved layer | visibility widened in infer-core (the T1.3-anticipated promotion): `attention_head{,_softcap,_set_causal}` + their mod.rs re-exports, `NoLora`, `forward_gemma2_layers`, `forward_base`, `clustered/standard_lm_head`, and the 14 `ForwardContext` scratch fields — the `raven.rs` "widened from pub(crate) to pub" precedent, applied to the coherent scratch-buffer group |
| clippy `missing_safety_doc` fired on the newly-pub `unsafe fn`s | `# Safety` doc sections added (content from the existing informal SAFETY notes) |

**Validation (every gate green before each push):**

- `cargo check`: infer-core default + `--all-features`; engine default +
  `--all-features` (54.8s) + `--tests` (3m35s).
- **Test conservation, stash-proven**: pre-move engine 3,047 + infer-core
  63 = **3,110**; post-move engine 2,881 + infer-core 229 = **3,110** —
  EXACT. (The naive bare-default comparison loses 56 feature-gated tests —
  infer-core has no `[default]` list; the honest count matches infer-core's
  features to engine's default resolution, extracted via
  `cargo tree -e features`.)
- Test RUNS: infer-core 228p/0f/1i (serial — the first parallel run hit a
  `STATUS_STACK_BUFFER_OVERRUN` contention artifact, the known box class;
  serial green in 138.6s); engine 2,879p/0f/2i in 0.53s (the slow
  transformer/deltanet tests moved with their modules — the engine lib's
  remainder is fast unit tests).
- clippy: 0 both crates (one exFAT stale-cache re-report trap en route,
  re-run clean).

**T1.5 (measured, the Phase 2 evidence):**

- riir-clippy DEFAULT: **68 packages, zero riir-* edges** — the default
  build is untouched by Phase 1 and needs nothing from Phase 2 (only its
  opt-in `latent_retrieval`/`ternary_inference` arms retarget, T2.3).
- riir-train workspace: **241 packages** including engine + infer-core +
  gpu via path deps. Phase 1 cost: **+1 package** (infer-core) for every
  engine consumer — the rlib-boundary cost, negligible at check time
  (engine check 5.3s incremental, all-features 54.8s).
- Phase 2's projected win remains as filed: inference-only consumers dep
  `riir-infer` directly and DROP the cognition tier from their graphs —
  the Bench-723 shrink happens at repo promotion, when consumer graphs
  actually change; a cold build-time A/B was NOT re-measured in Phase 1
  (same code volume, one extra rlib boundary — nothing to shrink yet).

**Standing follow-ups filed by this session:** AGENTS.md crate table row +
`.docs/02_crates/` entry for riir-infer-core (doc-sync); consumer spot-run
of `riir-train-engine`'s own tests against the split (its imports go
through `riir_engine::` re-exports — path-preserved, but a green run is
the honest close); boundary-guard pass after Phase 2 (new dep edges).

## Session 5 (2026-08-27, 4090 box) — Phase 1 close-out + the Phase 2 decision

**Consumer spot-run (the Session-4 honest close, discharged):**
`cargo test -p riir-train-engine --lib` on the post-split tree — **1447 passed /
1 failed / 1 ignored in 46.77s**, and the 1 failure is
`lattice_operad::transfer::tests::bench_transfer_overhead` (116.5× vs 100×
threshold), the **known load-flaky timing gate** (documented firing at 111.9×
under sibling load in the riir-clippy slice records; passes isolated). Isolated
re-run: **1 passed in 0.26s**. Verdict: **rte 1448/1448 green against the
split** — its imports ride the `riir_engine::` same-path re-exports exactly as
designed. No Cargo.lock committed (riir-train tree carries a sibling's lock
edit; left untouched).

### The Phase 2 decision — DEFER-to-trigger (owner-delegate call, "best perf/sec prod grade")

All filed gates measure PASS (T1.3 = 44 ≤ 170; T1.4 = 0; T1.5 in hand), so
promotion is *permissible*. The decision is that it is not yet *the best
perf/sec move*:

1. **Runtime perf: zero.** Repo org does not move tok/s. The perf campaign
   that does (Plan 548 — beat llama.cpp/vLLM closure; prefill 1.93× behind,
   decode ahead +15.8%) works in exactly these files, and the user rule
   "finish 4090/CUDA plans first" ranks it ahead of repo surgery.
2. **The projected win requires a second campaign anyway.** T1.5 measured the
   current structure's cost at **+1 package** for engine consumers — negligible.
   The Bench-723-class shrink only realizes when consumers retarget to
   `riir-infer` direct deps (T2.1–T2.4); promoting without retargeting buys an
   org chart, not a graph.
3. **No consumer has pulled.** riir-clippy default: 68 pkgs, zero riir-* edges.
   riir-train: +1 pkg. The pull trigger below is the promotion condition —
   demand-first, the Issue-528 no-premature-abstraction precedent.
4. **Mid-campaign surgery is the worst outcome**; deferring without a trigger
   invites it. So the deferral carries explicit fires.

**Promotion fires when ANY of:**

- **T-A (graph pain measured):** an inference-only consumer's cold build or
  dep graph measurably suffers the cognition tier (the Bench-723 metric class
  — e.g. Plan 548's serving harness, or a riir-train inference-only path
  going cold-build-bound).
- **T-B (second consumer pulls):** a repo outside riir-ai needs
  `riir-infer-core`/`riir-gpu` WITHOUT `riir-engine` (riir-clippy's opt-in
  arms retargeting = the first candidate, T2.3).
- **T-C (contention):** riir-ai repo contention becomes top pain (the D3
  reopen-trigger class).

When a trigger fires, the runbook is already written: T2.1 (repo bootstrap:
`riir-infer` = riir-infer-core + riir-gpu + riir-gpu-async, BOUNDARY.md,
AGENTS.md, `.docs/`, numbering) → T2.2 (riir-ai becomes downstream consumer,
path deps `../riir-infer/crates/*`) → T2.3 (consumer retarget, starting with
riir-clippy's opt-in arms) → T2.4 (`ci_boundary_contract.sh` 15→16 repos +
m3 clone sync). Do it as the FIRST act of whatever campaign pulled it, never
mid-campaign.

### The four open questions — answered

1. **Phase 2 threshold:** the bar stays **T1.3 ≤ 170** (parity with the civ
   surface that deferred Proposal 025). Measured 44 = 3.9× headroom, and the
   load-bearing gate graduated from grep to **compiler fact** at Phase 1
   (T1.4 back-edges = 0 proven by build; test conservation exact 3,110==3,110).
   A stricter symbol bar adds nothing — the binding constraint is the
   consumer-pull trigger, not symbol count.
2. **Naming: keep `riir-infer-core`; reject the fold-into-`riir-gpu` option.**
   Folding re-welds backend-neutral model definitions (transformer/deltanet
   CPU reference included) to a backend-named crate — the exact confusion this
   proposal exists to fix ("riir-gpu is not what riir-ai's contract says it
   is"). The `-core` suffix parallels `katgpt-core` (substrate-leaf naming),
   and a future `riir-infer` repo containing `riir-infer-core` + `riir-gpu` +
   `riir-gpu-async` is already coherent.
3. **`riir-train-engine` on the same axis: NO — out of scope, pull-gated.** rte
   is the *training* engine; folding it toward the inference repo would
   re-weld the train/infer coupling Issue 741 spent a campaign severing. The
   DRY exception is per-kernel, not per-crate: when a concrete >200-LOC kernel
   duplication between rte and infer-core appears (the "pattern borrowing ≠
   dependency" bar), file an issue for THAT kernel. rte's own size (338,467
   LOC) is a riir-train-internal concern.
4. **`fourier/` (16,879 LOC): LEAVE IT.** Confirmed game-domain by measurement,
   and it already sits in a legal home — `riir-engine` passes the domain test
   for cognition-serving code (KARC consumes it). It stayed engine-side at
   Phase 1 (zero references from the moved set). Relocating to
   `riir-games-shared` is a games-split concern (D3 axis), zero runtime perf
   effect, pure churn — revisit only under D3's reopen triggers, never as its
   own campaign.

### D4 discharge (the widening revisit this decision owed)

BOUNDARY.md D4 anchors the T4c/T4c2/T10 widening reversal to this proposal's
Phase 2 record. **Decision: the widenings STAY** — reversal is pure churn
while the edges they serve (riir-train consumers → riir-gpu/engine;
riir-engine cognition → riir-infer-core, the Phase-1 direction flip that
added ~22 more promotions: `attention_head{,_softcap,_set_causal}`, `NoLora`,
`forward_gemma2_layers`, `forward_base`, `clustered/standard_lm_head`, 14
`ForwardContext` scratch fields) are load-bearing. The reversal is bundled
into the promotion runbook: when T-A/B/C fires, re-narrow what the new
structure makes unnecessary as part of T2.x, not before. D4's Actual column
is extended with the Phase-1 line in the same commit as this record.

**Close-out artifacts landed with this session:** BOUNDARY.md `Owns` row for
`riir-infer-core` + D4 extension; `.docs/02_crates/riir_infer_core.md`;
AGENTS.md crate-table row updated to the decided state; this record.

## Verdict

**GO on Phase 0 and Phase 1. Phase 2's gate conditions now MEASURE PASS
(T1.3 = 44 ≤ 170, T1.4 = 0) — but Phase 2 stays HOLD pending T1.0 + T1.1,
because the extraction has not been built yet and T1.5's build-cost evidence
(the metric that made Issue 739 provable) does not exist.**

**Updated 2026-08-22 (Session 3):** the HOLD is now *further* from lifting than
it looked, and that is a measurement result, not a delay. T1.0's blocker set is
**22 files, not 3** (7 in `riir-gpu/src`, 15 outside any `src/`), the chain runs
through the cycle-blocked `gemma4_q4k_train` seed, and a new owner call
(Issue 744 T5) sits in front of it. Nothing here weakens the *case* for the
split — M3/M4/M6/M7/M8 are untouched and T1.3/T1.4 still PASS. What moved is the
cost of Phase 0, upward, twice, both times because a measurement got more honest
rather than because anything regressed.

Phase 0 is unconditionally correct — it fixes a contract that measures false
regardless of whether the split ever happens. Phase 1 is cheap, reversible, and
converts the load-bearing number (M4) from a grep into a compiler fact.

Phase 2 is *probably* right — M7/M8 say riir-gpu's consumers are not in this
repo, M6 says its growth axis is not this repo's, and M3/M4 say the seam is
narrower than the one that DEFERred Proposal 025. But 313k LOC and a 16th repo
is not a decision to take on a grep, and the honest precedent in this repo
(D3, Proposal 025) is to measure at the crate line before paying at the repo
line.

**Decided 2026-08-27 (§Session 5, owner-delegate under the perf/sec mandate):**
Phase 2 = **GO-behind-a-pull-trigger**. The gates all pass (T1.3 = 44, T1.4 =
0 compiler-proven, T1.5 = +1 pkg negligible; rte consumer spot-run 1448/1448
green) — permission is not the constraint; sequencing is. Promotion buys an
org chart until consumers retarget (T2.1–T2.4), runtime perf is unaffected,
and the 4090/CUDA campaign (Plan 548) outranks repo surgery on the owner's
own priority rule. Fires on T-A (measured graph pain), T-B (a second consumer
pulls), or T-C (riir-ai contention top pain) — then executed as the FIRST act
of the pulling campaign, never mid-campaign.

**Where this proposal pushes back on the motivating ask:** the intuition named
`riir-gpu` as the thing to split. Measured, `riir-gpu` alone is the *wrong*
boundary — M2's 95 paths would make `riir-infer` depend back into `riir-ai`,
which is the anti-pattern the split is meant to fix. The cut must include the
engine model layer or not happen.

## Risks

| risk | severity | mitigation |
|---|---|---|
| M4 balloons under real crate visibility | **high** — kills Phase 2 | That is exactly what T1.3 measures, before any repo cost is paid. |
| `riir-engine` build time worsens (extra crate boundary, no incremental win) | medium | T1.5 measures package counts + build time; Bench 723 already proved this metric discriminates. Note the riir-engine crate split was **CLOSED on build-cost grounds** (Issue 740: intersection = 0, no definable thin core) — *this* split is different because the model layer IS a definable core (M3 = 0), but the negative precedent is real and T1.5 is how we avoid repeating it. |
| 16th repo raises cross-repo sync cost | medium | Already the accepted cost of 15; `ci_boundary_contract.sh` C0b auto-discovers `riir-*`. Note the global rule: all repos must stay synced across m3/4090. |
| Model layer turns out to need cognition after all | low | M3 = 0 across 6 modules; T1.4 makes the compiler prove it. |
| Stale riir-ai clones (`../riir-ai-c`, `../riir-ai-t2`) diverge or poison greps | low but live | These are full clones beside riir-ai; they polluted the first audit grep of this very proposal. Resolve or document before Phase 2 mass-edits paths. |

## LoRA — the adjacent verdict, recorded so it is not relitigated

The split question arrived paired with *"verdict about lora, not sure it
works?"*. Measured, **two different things wear the name** and they get opposite
verdicts. Recording both here because the split routes them to different repos.

**(a) LoRA as a quality bet (train an adapter, get better behavior): LOSER.
Verdict already rendered 2026-06-15** by owner direction in
`.plans/005_phase3_code_move.md` and executed by Plan 005. Corroborating:

- `.docs/03_pillars/README.md:4` — the pillar architecture is *designed around*
  "LoRA bet fails"; 4 pillars work modelless. `:573` carries "LoRA never
  converges" as a live risk row.
- `.docs/03_pillars/wasm_validators.md:145-146` — LoRA+WASM > LoRA alone:
  target **+31**, measured **−71**. LoRA+WASM > WASM alone: target **+271**,
  measured **−69**. Both ❌ FAIL (Issue 018).
- `riir-train/.benchmarks/466` — lora arm **0/60 (0.0%)**,
  `FAIL (0 hits — modelless floor not beaten)`.

Honest counter-evidence, recorded so this reads as *unproven at product scale*
rather than *mathematically dead*: `riir-train` Plan 285 MSA-LoRA **7/7 GOAT
PASS, promoted default**; Plan 334 single-layer LoRA parity is a **Super-GOAT**;
`riir-train/.benchmarks/237` `cm_iso_lora` GOAT pass; `riir-train/.benchmarks/206` DA-LoRA GOAT proved
(gate later reverted by F6, verdict standing). Under the demote-loser rule the
burden is on LoRA and it has not paid — but the burden, not the mechanism, is
what failed.

**(b) LoRA as a runtime mechanism (hot-swap an overlay at inference for ~free):
WORKS, and it is measured.**

| bench | result |
|---|---|
| `.benchmarks/672` | 65.70 tok/s; overhead **0.72%**; G2/G3/G4 **PASS** |
| `.benchmarks/682` | graph decode with LoRA **93.86 tok/s, +21.4%, bit-identical quality** (eager 69.91 → 85.55, +22.4%) |
| `.benchmarks/686` | G3: LoRA within **1.16%** of frozen against a 5% gate |

This half is load-bearing for the modelless story, not incidental to it:
`katgpt-rs/CLAUDE.md` lists deterministically-constructed
`LoraPair { reader, writer }` hot-swap as **one of the three allowed modelless
weight-mutation paths**, and the AC-Prefix Path 2 result (Plan 313,
`.benchmarks/313_ac_prefix_modelless.md`) used exactly that to correct a
systematic bias **bit-identically**, which is what re-promoted `ac_prefix` to
default-on. Deleting the mechanism would remove a sanctioned modelless
correction path.

**Consequence for this proposal:** LoRA leaves riir-ai either way, to **two
different destinations by role** — the trained/quality half to `riir-train`
(Issue 741 Phase 0, mostly already done by Plan 005), the runtime half
(`GpuLoraBuffers`, `fused_lora`, `lora_still` forward, the reader overlay) to
`riir-infer` as an inference primitive. Do not kill it; route it.

**The "prod games won't use it" claim is confirmed measured:** `npc-lora`,
`plasma_lora`, and `da_lora_integration` are all **opt-in** in `riir-games`
(verified absent from its `default = [...]`), and M8 shows no game-product repo
depends on `riir-gpu` at all. Prod games are LoRA-free today.

## Open questions for the owner

> **ANSWERED 2026-08-27 (§Session 5, owner-delegate under the "best perf/sec
> prod grade" mandate):** (1) bar stays ≤170, the binding constraint is the
> pull trigger; (2) keep `riir-infer-core`, no fold; (3) rte = NO, per-kernel
> DRY issues only; (4) fourier = LEAVE. Original questions preserved below.

1. **Phase 2 threshold** — is "T1.3 ≤ 170 symbols" (parity with the civ surface
   that DEFERred Proposal 025) the right bar, or should a 313k-LOC repo move
   demand a stricter one?
2. **`riir-infer-core` naming** — or fold the model layer into `riir-gpu` itself
   as a `model` module and ship a 2-crate repo? Cheaper, but re-welds
   architecture definitions to a backend-named crate.
3. **Does `riir-train-engine` (338,467 LOC) belong on the same axis?** It is
   already the largest crate in the workspace and is model-shaped, not
   game-shaped. Out of scope here; flagging because the same argument applies.
4. **`fourier/` (16,879)** — confirmed game domain by measurement. Relocate to
   `riir-games-shared` in a follow-up, or leave it?
