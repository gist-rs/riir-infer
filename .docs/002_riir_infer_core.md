# riir-infer-core — the model-based inference layer (extracted)

> **Moved 2026-09-26** from riir-ai `.docs/02_crates/riir_infer_core.md` (renumbered `002` in this repo's `.docs/` ledger). The riir-ai docs book lives at `../../riir-ai/.docs/`; sibling links re-pointed.

[← riir-infer README](../README.md) · **The carved crate (from riir-ai's docs book, Part II · The Crates)**

The model-based inference substrate extracted from `riir-engine` by
[Proposal 041](../.proposals/041_riir_infer_model_based_inference_split.md)
Phase 1 (2026-08-27): everything a transformer/DeltaNet forward pass needs at
the model-definitions level — architectures, weight formats, SIMD kernels,
RoPE, quantization, loaders — with **zero cognition imports and zero training
code** (Issue 741 evicted Tier C first; T1.4 back-edges = 0 proven by the
compiler, not grep).

## What lives here

44 files / 30,373 LOC, moved in two chunks (leaves first, then the 3-cycle
`transformer → ternary_layer → deltanet → transformer` blob atomically):

| Module | Role |
|---|---|
| `transformer/` | The transformer forward stack (attention, `ForwardContext` scratch, clustered/standard lm_heads, gemma2 layer loop) |
| `deltanet/` | DeltaNet recurrence (chunked + decode paths) |
| `ternary_layer/` | Ternary-weight layer (the Bonsai class) |
| `quant/` | Quantization format types |
| `gguf_loader` / `safetensors_loader` | Weight file loading |
| `rope` | Rotary position embedding |
| `gemma_layer/` / `llama_layer/` | Architecture-specific layer defs |
| `spec_types` | Speculative-decode type surface |
| `dflash` | DFlash forward (Issue 708 made functional) |
| `types` / `simd` / `wall` | Shared types, SIMD kernels, the wall abstraction |

`transformer_still` stays engine-side (the `lora_still` bridge imports
`crate::lora_still` — a deliberate Plan-267 holdover, re-declared top-level in
engine `lib.rs` under the same feature gate).

## The consumption contract (why nothing broke)

`riir-engine` **re-exports everything at the SAME paths** (`riir_engine::transformer::*`
etc. now route through to `riir_infer_core`), so every consumer — riir-gpu's 52
importing files, `riir-train-engine`, riir-clippy's opt-in arms — compiles
unchanged. 31 engine features forward to it (`X = [..., "riir-infer-core/X"]`),
so feature unification is preserved bit-for-bit.

Downstream consumers continue to import via `riir_engine::`; direct
`riir_infer_core::` imports are legal but only interesting post-Phase-2 (see
below).

## Validation evidence (Phase 1 landing)

- Test conservation **exact**: pre-move engine 3,047 + infer-core 63 = 3,110 ==
  post-move engine 2,881 + infer-core 229 (stash-proven; the naive bare-default
  comparison loses 56 feature-gated tests — matched features, not defaults).
- `cargo check` green: both crates, default + `--all-features` + `--tests`.
- Consumer spot-run (the honest close): `riir-train-engine` **1448/1448** green
  against the split (the one full-run failure was the known load-flaky
  `bench_transfer_overhead` timing gate — passes isolated).
- clippy 0 both crates.

## Phase 2 state — DEFER-to-trigger (decided 2026-08-27)

The crate is a workspace member of riir-ai. Promotion to its own repo
(`riir-infer` = riir-infer-core + riir-gpu + riir-gpu-async ≈ 362k LOC, the
16th repo) is **decided GO-behind-a-pull-trigger**, not scheduled: it fires on
measured consumer-pull (T-A graph pain / T-B a second consumer wants the
inference-only graph / T-C riir-ai contention top pain) and is then executed as
the first act of the pulling campaign. The runbook is
[Proposal 041 §Session 5](../.proposals/041_riir_infer_model_based_inference_split.md) — Phase-2 deferral record.

## See also

- [Proposal 041](../.proposals/041_riir_infer_model_based_inference_split.md) — the split decision record (sessions 1–5)
- [riir-ai `BOUNDARY.md`](../../riir-ai/BOUNDARY.md) — Owns row + D4 widening disposition
- [riir_gpu.md](../../riir-ai/.docs/02_crates/riir_gpu.md) — the GPU kernel layer that consumes this crate
