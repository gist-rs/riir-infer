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
  the quantization zoo (q2k → q8kv, q2_0 ternary, PTQ/TurboQuant), model
  architectures (gemma / llama / ternary / wall layers, deltanet,
  transformer, rope, dflash speculative types), SIMD helpers, CPU reference
  paths. NO cognition, NO training, NO game/chain semantics.
- `riir-infer-gpu` (crate, `crates/riir-infer-gpu`): the GPU runtime +
  kernel layer — device context, buffer helpers, the pool-poison
  detector, the CubeCL runtime, the persistent weight-buffer cache, the
  GPU transpose kernel (+ its WGSL), and the adapter VRAM probe. The
  kernel-set migration from the engine's GPU layer is ongoing (tracked
  in the private workspace campaign that carved this repo).
- Planned (owner-directed, tracked in the private workspace): the
  laya/ModernBERT encoder lane, and the remaining SEAM residues of the
  GPU kernel migration re-homing with their consumers.

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
| riir-infer-gpu deps: wgpu, cubecl (`=0.11.0-pre.2`), bytemuck, pollster, rayon, half, papaya, blake3, serde_json (S4b: dflash2 header parsing), metal (macOS, opt-in), cudarc (non-macOS, opt-in) | crates.io | GPU crate only (`crates/riir-infer-gpu/Cargo.toml`); wgpu-msl shader backend on macOS, wgpu-spirv elsewhere; zero `riir-*` deps (fence-gated) |
| vendored crates.io forks: `vendor/cubecl-runtime-0.11.0-pre.2` (drop-queue policy fix — upstream tracel-ai/cubecl#1359), `vendor/wgpu-hal-30.0.0` (`total_video_memory_bytes()`/`raw_handle()` adapter accessors) | in-repo `vendor/` | `[patch.crates-io]` in the workspace root; byte-identical copies; remove when upstream lands |

Explicitly NOT allowed from any feature combination: any `riir-*` crate
(this repo is upstream of the engine by design — `cargo tree` must show
zero `riir-*` packages), any game/chain/cognition crate, any trained-weight
artifact, any Python dependency.

## Inherited boundaries (links)

- Dep-direction matrix + CANONICAL rows: the engine repo's BOUNDARY.md
  (private workspace sibling)
- Modelless-first mandate: the `katgpt-rs` AGENTS.md (public upstream)

## Drift ledger (target vs actual)

None at birth — seeded empty (the ledger fills only with a row + its open
issue, never with silence).
