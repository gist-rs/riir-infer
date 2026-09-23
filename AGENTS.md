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
the same paths). Internal record + phase plan: riir-ai Issue 998 /
riir-reflex Issue 008; extraction history: riir-ai Proposal 041. Planned
(owner-directed, tracked in the issues above): the laya/MSL encoder lane
and the CLEAN GPU kernel layer join here.

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
```

Sibling layout: `../katgpt-rs` must exist for every cargo command (path

Workspace layout: a root-package workspace — `riir-infer-core` at the repo root (CPU substrate) + `crates/riir-infer-gpu` (the wgpu/CubeCL GPU layer, its own crate so CPU-only consumers never resolve the GPU dep tree). Vendored crates.io forks under `vendor/` (`[patch.crates-io]` in the root manifest).
deps). `../riir-ai` consumes this repo (its riir-engine re-exports the
crate's modules at the same paths).

## Numbering Discipline

Issue, plan, doc, benchmark, and research numbers are **monotonic and never
reused**. Read the target dir's `.highwater`, use `value + 1`, write the
new value back.

## Branch

`develop` is the working branch. Don't create feature branches; commit
directly on `develop` per the global rule.
