# riir-infer — boundary contract

> The single source of truth for what may live in and depend on this repo.
> Audited by the `boundary-guard` skill +
> `../riir-ai/scripts/ci_boundary_contract.sh`
> (they read this file; findings are contract violations or contract rot).
> Cross-repo rules LINK to their one canonical home — never copied.
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
- Planned (owner-directed, tracked upstream): the laya/ModernBERT encoder
  lane (Backend trait, flat-Vec ops, MSL kernels, tokenizers substrate) and
  the CLEAN GPU kernel layer (CubeCL/CUDA/Metal op kernels).

**Domain test:** is this **model-based inference substrate** (weights,
quant, architectures, kernels, loaders — upstream of every engine and game
concern)? NO → it belongs in another repo; file there.

**Hard fences (never move here):** cognition/emotion/perception code,
game logic or vocabulary, chain/consensus, routing policy, trained weights
or checkpoints (this repo ships LOADERS, not weights).

## Does not own

| Concern | Correct home |
|---|---|
| Cognition runtimes / engine orchestration | `../riir-ai` (`riir-engine`) |
| Training (LoRA SFT, GRPO, optimizers, backprop) | `../riir-train` |
| Engine-consumed game substrate (KV cache policy beyond loader types, routing) | `../riir-ai` |
| Modelless decision engine / arena | `../riir-reflex` |
| Trained checkpoints (.gguf artifacts) | `../riir-train/data` |

## May depend on

| Crate | Location | Condition |
|---|---|---|
| katgpt-core | `../katgpt-rs/crates/katgpt-core` | non-optional (public upstream) |
| katgpt-transformer / katgpt-speculative / katgpt-forward | `../katgpt-rs/crates/*` | non-optional (public upstream; transformer `default-features = false`) |
| katgpt-quant / katgpt-attn | `../katgpt-rs/crates/*` | optional (`turboquant` / `flashmemory_gqa`) |
| crates.io: half, rayon, anyhow, memmap2, fastrand, bytemuck, serde, serde_json, thiserror, log, blake3 | crates.io | as pinned in `Cargo.toml` |

Explicitly NOT allowed from any feature combination: any `riir-*` crate
(this repo is upstream of the engine by design — `cargo tree` must show
zero `riir-*` packages), any game/chain/cognition crate, any trained-weight
artifact, any Python dependency.

## Inherited boundaries (links)

- Dep-direction matrix + CANONICAL rows: `../riir-ai/BOUNDARY.md`
- Modelless-first mandate: `../katgpt-rs/AGENTS.md`

## Drift ledger (target vs actual)

None at birth — seeded empty (the ledger fills only with a row + its open
issue, never with silence).
