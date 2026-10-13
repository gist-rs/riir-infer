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
| `transformer/edlm.rs` | eDLM (Drex DLM) segment-structured scoring lane (Issue 1005, opt-in `edlm`): `edlm` GGUF arch, segment block-causal mask, packed/row forwards, pointer head, state-prefix KV reuse (T7) — the GPU twin lives in `riir-infer-gpu` (`edlm_gpu`, f16-resident + a cooperative-matrix GEMM arm at m≥16) |
| `spec_types` | Speculative-decode type surface |
| `dflash` | DFlash forward (Issue 708 made functional) |
| `types` / `simd` / `wall` | Shared types, SIMD kernels, the wall abstraction |

`transformer_still` stays engine-side (the `lora_still` bridge imports
`crate::lora_still` — a deliberate Plan-267 holdover, re-declared top-level in
engine `lib.rs` under the same feature gate).

## The consumption contract (why nothing broke)

`riir-engine` **re-exports everything at the SAME paths** (`riir_engine::transformer::*`
etc. now route through to `riir_infer_core`), so every consumer — riir-gpu's 52
importing files, `riir-train-engine`, riir-refine's opt-in arms — compiles
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

## The disaggregated container (Issue 028, 2026-10-06)

`disaggregated.rs` — the dual-PTQ disaggregated serving container (Research
004, arXiv:2609.26333): one checkpoint quantized twice (compact decode copy +
higher-precision prefill copy), the prompt run through the prefill copy, the
resulting `HybridCache` handed off IN-PROCESS, generation continuing on the
decode copy. Container forms: two GGUFs (`load_pair` — compat checked on the
`qwen35.*` metadata plane BEFORE either copy dequantizes) or one file whose
prefill tensors carry the `.pf` suffix (`load_single_file`, read through the
suffix-parameterized `load_qwen_deltanet_ternary_from_gguf`); `from_single`
is the degenerate same-weights container. The T2 escape law refuses loudly
when `ssm_a`/`ssm_dt.bias`/`ssm_alpha`/`ssm_beta` are not bit-shared between
the copies (the paper's Appendix A.2 + our Issue-980 set). G1 gate: the
same-weights phase split is byte-identical to the single-checkpoint loop —
pinned on the synthetic contract AND on the real 6.7 GB
`Ternary-Bonsai-2-27B-PQ2_0.gguf` pack (`tests/issue028_disaggregated_handoff.rs`,
the real arm `#[ignore]`d, `BONSAI_PQ2_0_GGUF` env). T4 (the dual-PTQ
accuracy measurement — needs a q4-class prefill pack to exist) and T5 (ODP,
deferred by default) remain open in `.issues/028`.

## The GDN spectral-class spec test (Issue 046, 2026-10-13)

Closed **spec-test-only** (`.issues/046` removed, landing hash in git). The
issue's original premise — heads with g→1 accumulate `‖S‖` like `1/(1−g)` —
was corrected at its own verdict round: that bound is the ADDITIVE-write row
(arXiv:2610.13341 §C.3); the house recurrence is the delta rule with
L2-normalized keys, whose write is a keyed erase — bounded by construction.
What landed instead is the primary deliverable,
`tests/issue046_gdn_spectral_class_spec.rs` (feature `deltanet_inference`,
`[[test]]` row in Cargo.toml):

- **Recovery, not construction**: the homogeneous transition `T =
  g·(I − β·k̃k̃ᵀ)` is recovered from the REAL decode kernel
  (`gated_deltanet_step_inplace`) by basis probing (v = 0, one key-basis
  column at a time), so a transcription change in the kernel's operation
  order (kv pre-decay, or the rank-1 write transposed — Issue 594's class)
  moves the recovered spectrum and reds the file.
- **Spectral class**: eig(T) = `{g(1−β) along k̃, g ×(d−1)}` (the
  Gated-DeltaNet row, arXiv:2610.13341 App. A.4 — NOT DeltaNet's `{1−β, 0}`),
  across a (g, β) grid incl. β = 1 and the 128×128 Bonsai geometry, via a
  self-contained f64 Jacobi eigensolver.
- **Generator identity**: `T ≡ expm(−γ·I − b·k̃k̃ᵀ)` entrywise (γ = −ln g,
  b = −ln(1−β)) — the frozen continuous-time pair {−(γ+b), −γ} integrates to
  exactly the discrete class, to f32 tolerance.
- **Boundedness pin** (the corrected premise, through the real kernel):
  aligned steady-state gain `β/(1−g+gβ) ≤ 1` for EVERY g ≤ 1, β ∈ (0,1] —
  measured ≤ 0.999896 at the motivating corner (g = 0.9999, β = 0.5, d = 128).

The FV (Lean) half was judged overweight per the issue's own escape hatch (no
`.proofs/` instance exists in this repo; a numeric spec test carries the
assertion). The report-only fleet-replay probe (deliverable 2) closed under
the issue's negative-control clause: the aligned steady state IS the negative
control and earns the probe nothing — the Non-goals forbid a growth-alarm
threshold, and the growth class is refuted by execution.

Rider: gating `on_decode_pool` with its consumers (`deltanet_ternary_inference`)
fixed a dead-code warning the new posture exposed. Side finding (filed):
`.issues/047` — the plain `deltanet_inference` lib posture crashes at
parallel-harness teardown (pre-existing, single-thread clean).

## 2026-10-14 — Issue 047: lib test crash at `deltanet_inference` parallel posture CLOSED — OOM abort, not a teardown race; unit-test configs de-bloated

The `0xc0000409` ("STATUS_STACK_BUFFER_OVERRUN") exit was never a stack
overrun and never a teardown race. Direct-binary runs captured the real
death message: `memory allocation of 67108864 bytes failed` — Rust's
`handle_alloc_error` abort, which on Windows funnels through `__fastfail`
and therefore EXITS as 0xc0000409. The "teardown" attribution was an
artifact of stdout buffering; direct runs died mid-suite (3–16 s into a
57 s suite).

**Root cause**: ~20 unit tests built configs from the production preset
`Config::qwen_deltanet` (vocab 151936, block_size 32768, mlp 8192) and then
allocated `QwenDeltaNetWeights::zeros` — which holds the embedding table
TWICE (`wte` + the tied `lm_head` clone = ~2.3 GiB) — plus eager
`KVCache::new` (512 MiB per attention layer at block_size 32768) plus
~285 MiB/layer of projection weights. Per-test footprint 2.8–9.1 GiB (the
weights.rs shape tests used 24 layers → ~9 GiB each). At the default
test-thread count (`available_parallelism()` = 24 on this box) the
concurrent set exhausted the Windows commit limit; the observed 64 MiB
failure (exactly one 2048×8192×4 MLP weight) was the last straw, not the
culprit. Clean at every fixed thread count ≤16 because fewer giants ran
concurrently; flaky because overlap is scheduler-dependent.

**Fix** (`#[cfg(test)] crate::deltanet::unit_test_config`): the preset with
only the multi-GiB surfaces shrunk — vocab 151936→512, bos 151643→1 (bos
must stay a valid id: prompts embed `wte[bos*n_embd]`), block_size
32768→256, mlp 8192→512; `n_embd`/head geometry stay real. 21 call sites
converted (17 in `forward.rs`, 4 in `weights.rs`); kept on the full preset:
the `#[ignore]`d `bench_batched_prefill_vs_decode` (the real-dims lane),
the already-small issue-594 test, and the two validate-only tests that never
allocate.

**Evidence**: 5/10 crash → **0/15** at default threads; suite 57 s → 8.65 s
(the box was paging); 312 passed (`deltanet_inference`), 342 passed
(`deltanet_ternary_inference`), clippy clean, default posture unaffected.

## See also

- [Proposal 041](../.proposals/041_riir_infer_model_based_inference_split.md) — the split decision record (sessions 1–5)
- [riir-ai `BOUNDARY.md`](../../riir-ai/BOUNDARY.md) — Owns row + D4 widening disposition
- [riir_gpu.md](../../riir-ai/.docs/02_crates/riir_gpu.md) — the GPU kernel layer that consumes this crate
