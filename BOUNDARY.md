# riir-infer — boundary contract

> The single source of truth for what may live in and depend on this repo.
> Audited by the workspace boundary tooling (it reads this file; findings
> are contract violations or contract rot). Cross-repo rules LINK to their
> one canonical home — never copied.
>
> Drift ledger — Disposition: `fixable` | `owner-call` | `by-design`.
> `fixable`/`owner-call` rows REQUIRE an open issue (row ⟺ open issue); a
> `by-design` row cites the decision record instead. Issue closes → row removed
> in the same commit.

## Owns

**The LLM inference substrate** — everything a model-based inference engine
needs BELOW the engine:

- `riir-infer-core` (crate, repo root): GGUF + safetensors weight loading,
  the quantization zoo (q2k → q8kv, q2_0 ternary, PTQ/TurboQuant, EXL3
  trellis — multi-shard zero-copy pack reader + CPU reference + bit-identical
  fast arm, opt-in `exl3`, Issue 001), model
  architectures (gemma / llama / ternary / wall layers, deltanet,
  transformer, rope, dflash speculative types), SIMD helpers, CPU reference
  paths. NO cognition, NO training, NO game/chain semantics.
- `riir-infer-gpu` (crate, `crates/riir-infer-gpu`): the GPU runtime +
  kernel layer — device context, buffer helpers, the pool-poison
  detector, the CubeCL runtime, the persistent weight-buffer cache, the
  GPU transpose kernel (+ its WGSL), the adapter VRAM probe, and the
  EXL3 trellis GPU dequant kernels (feature `exl3_gpu`; decode bit-exact
  vs the CPU reference, block-Hadamard gated in the FMA-contraction
  class — Issue 001 T7b). The
  kernel-set migration from the engine's GPU layer is ongoing (tracked
  in the private workspace campaign that carved this repo).
- `riir-infer-laya` (crate, `crates/riir-infer-laya`): the pinned-checkpoint
  encoder lane — the Python-JSON byte writer (ungated, one DRY home), the
  tokenizer / config / weights (locate → verify → download) substrate, the
  answer envelopes, the temperature law + script detector, and the
  flat-`Vec<f32>` forward with its two backends: CPU (`gemm`) and the macOS
  Metal MSL family (target-scoped, feature `laya-riir-metal`). Everything
  except the writer sits behind `laya-riir`. The lane is a faithful port of
  a pinned reference; the consumer-side parity gate is its correctness
  authority.
- Planned (owner-directed, tracked in the private workspace): the
  remaining SEAM residues of the GPU kernel migration re-homing with their
  consumers.

**Domain test:** is this **model-based inference substrate** (weights,
quant, architectures, kernels, loaders — upstream of every engine and game
concern)? NO → it belongs in another repo; file there.

**Hard fences (never move here):** cognition/emotion/perception code,
game logic or vocabulary, chain/consensus, routing policy, trained weights
or checkpoints (this repo ships LOADERS, not weights).

## Does not own

| Concern | Correct home |
|---|---|
| Cognition runtimes / engine orchestration | the engine layer (private workspace repo) |
| Training (LoRA SFT, GRPO, optimizers, backprop) | the training repo (private workspace repo) |
| Engine-consumed game substrate (KV cache policy beyond loader types, routing) | the engine layer (private workspace repo) |
| Modelless decision engine / arena | the decision-engine repo (private workspace repo) |
| Trained checkpoints (.gguf artifacts) | the training repo's data tree (private) |

## May depend on

| Crate | Location | Condition |
|---|---|---|
| katgpt-core | `../katgpt-rs/crates/katgpt-core` | non-optional (public upstream) |
| katgpt-transformer / katgpt-speculative / katgpt-forward | `../katgpt-rs/crates/*` | non-optional (public upstream; transformer `default-features = false`) |
| katgpt-quant / katgpt-attn | `../katgpt-rs/crates/*` | optional (`turboquant` / `flashmemory_gqa`) |
| crates.io: half, rayon, anyhow, memmap2, fastrand, bytemuck, serde, serde_json, thiserror, log, blake3 | crates.io | as pinned in `Cargo.toml` |
| riir-infer-gpu deps: wgpu, cubecl (`=0.11.0-pre.2`), bytemuck, pollster, rayon, half, papaya, blake3, serde_json (S4b: dflash2 header parsing), log + fastrand (S5: gemma2 KV-cache alloc reports + the d2f sampler RNG), metal (macOS, opt-in), cudarc (non-macOS, opt-in) | crates.io | GPU crate only (`crates/riir-infer-gpu/Cargo.toml`); wgpu-msl shader backend on macOS, wgpu-spirv elsewhere; zero `riir-*` deps (fence-gated) |
| vendored crates.io forks: `vendor/cubecl-runtime-0.11.0-pre.2` (drop-queue policy fix — upstream tracel-ai/cubecl#1359), `vendor/wgpu-hal-30.0.0` (`total_video_memory_bytes()`/`raw_handle()` adapter accessors) | in-repo `vendor/` | `[patch.crates-io]` in the workspace root; byte-identical copies; remove when upstream lands |
| lane crate deps: serde (+derive), serde_json (`preserve_order` — JSON object insertion order IS the label order), tokenizers (**0.22, pinned on a measured negative** — the 1.0.0-rc line refuses the pinned BPE files; reopen at 1.0.0 stable), sha2 (the weight pins are SHA-256, an external fact), blake3 (small-file pins), gemm (0.18; any bump re-runs the parity gate), libm (0.2, **numerics pin** — bit-identical erf so the drift budget is spent on op order) | crates.io | `riir-infer-laya` only; serde/tokenizers/sha2/blake3/gemm/libm optional behind `laya-riir` |
| macOS target-scoped: metal (**0.31 — one workspace version**, shared with `riir-infer-gpu`; the parity gate is the acceptance for any bump), objc2 (0.6) | crates.io | `riir-infer-laya` `laya-riir-metal` feature, `cfg(target_os = "macos")` only — enabling it on Linux/Windows is inert, never a dep-tree failure |
| macOS target-scoped: objc2-core-ml (0.3, `block2` feature on), objc2-foundation (0.3), objc2 (0.6, shared with the metal lane), block2 (0.6 — the ObjC block runtime the no-copy input arrays, the output reader and the async `MLComputePlan` loader need) | crates.io | `riir-infer-laya` `laya-riir-ane` feature, `cfg(target_os = "macos")` only — inert on every other host; never wasm32, never default. Any bump re-runs the consumer-side G5-ANE gate (the ANE lane's parity authority) |

Explicitly NOT allowed from any feature combination: any `riir-*` crate
outside this repo (this repo is upstream of the engine by design —
`cargo tree` must show zero foreign `riir-*` packages), any game/chain/
cognition crate, any trained-weight artifact, any Python dependency.

## Consumed by

| Repo | Edge | Condition |
|---|---|---|
| the public decision-engine serving repo (`gist-rs/riir-reflex`) | path-deps `riir-infer-laya` behind its own `laya-riir` / `laya-riir-metal` feature names (a `pub use` shim keeps its public API); the parity fixtures + consumer-side gate stay THERE | ONE-WAY — this repo never depends on it, from any feature combination (zero back-edge, fence-gated) |

## Inherited boundaries (links)

- Dep-direction matrix + CANONICAL rows: the engine repo's BOUNDARY.md
  (private workspace sibling)
- Modelless-first mandate: the `katgpt-rs` AGENTS.md (public upstream)

## Drift ledger (target vs actual)

None at birth — seeded empty (the ledger fills only with a row + its open
issue, never with silence).
