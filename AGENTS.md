# AGENTS.md — riir-infer

The global `~/.agents/` rules apply; this file documents repo-local context.

## Boundary contract — read `BOUNDARY.md` first

[`BOUNDARY.md`](BOUNDARY.md) is the authoritative per-repo contract (owns /
does-not-own / crate allowlist / drift ledger); BOUNDARY.md wins on any
conflict with prose in this file.

- **Domain test:** is this **LLM inference substrate** (weights, quant,
  architectures, kernels, loaders)? NO → another repo; file there.
- **Zero `riir-*` deps** — upstream of the engine by design; `cargo tree`
  gates it. Enforcement: `../riir-ai/scripts/ci_boundary_contract.sh`
  (run VIA the `boundary-guard` skill, never ad-hoc greps).
- **Found a violation?** File the issue FIRST (`.issues/NNN_boundary_*.md`),
  add the drift row, then fix; closing the issue removes the row in the same commit.

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
# GDN quant-certification + bonsai2 rotation integration tests carry required-features (Cargo.toml [[test]] rows):
cargo test --test issue879_gdn_quant_certification --features deltanet_ternary_inference
cargo test --test bonsai2_rotation_load --features bonsai2_hadamard
# Issue 028 disaggregated handoff battery (the 6.7 GB PQ2_0 arm is #[ignore]d — BONSAI_PQ2_0_GGUF env, release run):
cargo test --test issue028_disaggregated_handoff --features deltanet_ternary_inference
# BONSAI_GGUF env (defaults to a riir-train data path) names the real checkpoint for the certification test's full-file arm.
# The EXL3 trellis lane (opt-in; issue 001, closed — record in .docs/001; CPU reference + fast arm here, GPU kernels in riir-infer-gpu; oracle = the CPU reference):
cargo test --lib --features exl3
cargo test -p riir-infer-gpu --features exl3_gpu,cuda_backend --lib  # native CUDA arm (release recommended)
# The encoder-lane crate (features mirror the names consumers forward):
cargo check -p riir-infer-laya --features laya-riir
cargo clippy -p riir-infer-laya --all-targets --features laya-riir-metal -- -D warnings
cargo test -p riir-infer-laya --features laya-riir-metal --test metal_ops_smoke  # macOS only
# The ANE whole-graph backend (macOS-only; opt-in; artifact tree built by the reflex converter;
# Plan 612's e8 int8-table stack rides the same feature — LAYA_ANE_TABLE=e8 selects it, fp16 default):
cargo clippy -p riir-infer-laya --all-targets --features laya-riir-ane -- -D warnings
cargo test -p riir-infer-laya --features laya-riir-ane --lib
# G1 parity consumer-side (the env needs no reflex change):
#   cd ../riir-reflex && LAYA_ANE_TABLE=e8 cargo test --release --features laya-riir-ane --test laya_ane_parity
# The CubeCL portability arm (opt-in; plan 611 — KEPT, never default; the consumer-side G5 at the cubecl posture is the acceptance gate):
cargo clippy -p riir-infer-laya --all-targets --features laya-riir-cubecl -- -D warnings
cargo test --release -p riir-infer-laya --features laya-riir-cubecl --test cubecl_ops_smoke
# The three-backend A/B (measurement-only; Bench 006 — quote box state):
cargo test --release -p riir-infer-laya --features laya-riir-metal,laya-riir-cubecl --test backend_ab -- --ignored --nocapture
# The CUDA backend (non-macOS; inert on a Mac — no dep pulled). The op-level gate + the consumer-side G5 (at LAYA_DEVICE=cuda) are the parity pair:
cargo check -p riir-infer-laya --features laya-riir-cuda
cargo clippy -p riir-infer-laya --all-targets --features laya-riir-cuda -- -D warnings
cargo test --release -p riir-infer-laya --features laya-riir-cuda --test cuda_ops_smoke
```

Sibling layout: `../katgpt-rs` must exist for every cargo command (path
deps). `../riir-ai` consumes this repo (riir-engine re-exports at the
same paths); the public decision-engine serving repo (`../riir-reflex`)
path-deps `crates/riir-infer-laya`.

Workspace layout: root-package workspace — `riir-infer-core` at the repo
root (CPU substrate) + `crates/riir-infer-gpu` (the wgpu/CubeCL GPU layer,
own crate so CPU-only consumers never resolve the GPU dep tree) +
`crates/riir-infer-laya` (the encoder lane, own crate so lane-free
consumers never resolve the tokenizers/gemm tree — CPU + macOS Metal +
non-macOS CUDA backends, `laya-riir-cuda` since riir-infer Issue 002).
Vendored crates.io forks under `vendor/` (`[patch.crates-io]` in the root
manifest).

## Numbering Discipline

Issue, plan, doc, benchmark, and research numbers are **monotonic and never
reused**: read the target dir's `.highwater`, use `value + 1`, write back.

⛔ **`.issues/` carries TWO lanes (carve-era, 2026-09-26) — read before
allocating.** The files `998/1003/1004` are MOVED documents from the riir-ai
carve (riir-ai's numbers, not allocations made here); `.highwater` reads 1004
for that reason only and must not be treated as the local counter. LOCAL
allocations run the small lane and own a dedicated counter:
**`.issues/.highwater_local`** — read it, allocate value + 1, write the new
value back in the same commit (a plain max-of-disk rule would REUSE a number
once a closed issue file is removed, which is the katgpt-rs `.issues/121`
recycling bug). Lane history: `011` → `012` → `013` → `014` → `015`. And
NEVER allocate `1005–1008` here — riir-ai's own counter has already consumed
that range (it read 1008 on 2026-09-26; re-check
`../riir-ai/.issues/.highwater` before touching the inherited range at all).

Self-allocation headings follow the grammar the citation oracle reads —
`## <date> — Issue NNN: <title> CLOSED — <verdict>` — the number sits
IMMEDIATELY before its title delimiter; never write a word (CLOSED, resolved,
follow-up) between the number and the delimiter — that shape is unread to the
sweep and re-reds the repo (katgpt-rs Issue 921).

## Branch

`develop` is the working branch. Don't create feature branches; commit
directly on `develop` per the global rule.
