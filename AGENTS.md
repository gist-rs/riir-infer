# AGENTS.md — riir-infer

The global `~/.agents/` rules apply; this file documents repo-local context.

## Boundary contract — read `BOUNDARY.md` first

[`BOUNDARY.md`](BOUNDARY.md) is the authoritative per-repo contract: what
this repo owns, what it does not own, the crate-granular allowlist, and the
drift ledger. On any conflict with prose in this file, BOUNDARY.md wins.

- **Domain test:** is this **LLM inference substrate** (weights, quant,
  architectures, kernels, loaders)? NO → it belongs in another repo; file
  there.
- **Zero `riir-*` deps** — upstream of the engine by design; `cargo tree`
  gates it. Enforcement: `../riir-ai/scripts/ci_boundary_contract.sh`
  (run VIA the `boundary-guard` skill, never ad-hoc greps).
- **Found a violation?** File the issue FIRST (`.issues/NNN_boundary_*.md`),
  add the drift row, then fix. Closing the issue removes the row in the
  same commit.

## Role

The LLM-inference substrate repo. Carved from the riir-ai workspace
2026-09-22 (the crate moved out name-unchanged — engines re-export it at
the same paths). Internal record + phase plan: **Issues 998 + 1003 (local
`.issues/`, moved from riir-ai 2026-09-26)** / riir-reflex Issue 008;
extraction history: **Proposal 041 (local `.proposals/`, moved from
riir-ai)**. The
laya/MSL encoder lane LANDED as `crates/riir-infer-laya` (the
pinned-checkpoint encoder lane, consumed by the public decision-engine
serving repo); remaining planned work: the CLEAN GPU kernel layer
migration.

## Build Commands

```bash
cargo check
cargo clippy --all-targets -- -D warnings
cargo test --lib
# The GDN quant-certification + bonsai2 rotation integration tests carry
# required-features (see Cargo.toml [[test]] rows):
cargo test --test issue879_gdn_quant_certification --features deltanet_ternary_inference
cargo test --test bonsai2_rotation_load --features bonsai2_hadamard
# BONSAI_GGUF env (defaults to a riir-train data path) names the real
# checkpoint for the certification test's full-file arm.
# The EXL3 trellis lane (opt-in; issue 001, closed — record in .docs/001; CPU reference + fast arm here,
# GPU kernels in riir-infer-gpu; oracle = the CPU reference):
cargo test --lib --features exl3
cargo test -p riir-infer-gpu --features exl3_gpu,cuda_backend --lib  # native CUDA arm (release recommended)
# The encoder-lane crate (features mirror the names consumers forward):
cargo check -p riir-infer-laya --features laya-riir
cargo clippy -p riir-infer-laya --all-targets --features laya-riir-metal -- -D warnings
cargo test -p riir-infer-laya --features laya-riir-metal --test metal_ops_smoke  # macOS only
# The CUDA backend (non-macOS; inert on a Mac — no dep pulled). The op-level
# gate + the consumer-side G5 (at LAYA_DEVICE=cuda) are the parity pair:
cargo check -p riir-infer-laya --features laya-riir-cuda
cargo clippy -p riir-infer-laya --all-targets --features laya-riir-cuda -- -D warnings
cargo test --release -p riir-infer-laya --features laya-riir-cuda --test cuda_ops_smoke
```

Sibling layout: `../katgpt-rs` must exist for every cargo command (path
deps). `../riir-ai` consumes this repo (its riir-engine re-exports the
crate's modules at the same paths), and the public decision-engine
serving repo (`../riir-reflex`) path-deps `crates/riir-infer-laya`.

Workspace layout: a root-package workspace — `riir-infer-core` at the
repo root (CPU substrate) + `crates/riir-infer-gpu` (the wgpu/CubeCL GPU
layer, its own crate so CPU-only consumers never resolve the GPU dep
tree) + `crates/riir-infer-laya` (the encoder lane, its own crate so
lane-free consumers never resolve the tokenizers/gemm tree — CPU + the
macOS Metal + the non-macOS CUDA backends, `laya-riir-cuda` since
riir-infer Issue 002). Vendored crates.io forks under `vendor/`
(`[patch.crates-io]` in the root manifest).

## Numbering Discipline

Issue, plan, doc, benchmark, and research numbers are **monotonic and never
reused**. Read the target dir's `.highwater`, use `value + 1`, write the
new value back.

## Branch

`develop` is the working branch. Don't create feature branches; commit
directly on `develop` per the global rule.
