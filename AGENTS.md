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

The LLM-inference substrate repo, carved from the riir-ai workspace
2026-09-22 (crate moved name-unchanged — engines re-export it at the same
paths). Phase plan: Issues 998 + 1003 (local `.issues/`, moved from riir-ai)
/ riir-reflex Issue 008; extraction history: Proposal 041 (local
`.proposals/`). Landing narratives for every lane below: `HISTORY.md`.

**Model scope (owner law, 2026-10-26):** the production surface is
Bonsai/ternary (+ qwen3.8) ONLY. gemma-2-2b is RETIRED (riir-train Issue
617) — no new gemma-2-2b execution cells anywhere; the gemma2 seams stay
compile-only. The gemma2 ARCHITECTURE lanes serving live products (laya
checkpoints, clef-ANE 0.6b) are not this retirement and keep their gates.

**Env-switch law:** a promoted arm's env is DEMOTE-ONLY (unset = ON; the
exact demote literal restores the incumbent byte-identically); an opt-in
arm's env is PROMOTE-ONLY (arms only on exact `1`/`true`/`on`). Promotion
needs the GOAT bars + an owner/reviewer verdict; read the record before
flipping.

| Lane / switch | Posture | Record |
|---|---|---|
| GDN L8 eager prefill (plan 621, `prefill_cuda_deltanet_l8.rs`; non-macOS) — `RIIR_PREFILL_GDN_L8=0` demotes | DEFAULT-ON (2026-10-24) | riir-ai Bench 993/997; `.benchmarks/926_plan621_gdn_l8_eager_prefill.md` |
| GEMV workgroup-occupancy rule — `RIIR_GEMV_WG_OCCUPANCY=0` demotes | DEFAULT-ON, every platform | riir-ai Bench 996, Bench 887 (Issue 1115) |
| GLU-fused ternary prefill FFN (Issue 1124) — `RIIR_PREFILL_FFN_GLU` demote-only; proof lines `[721-trace]` (!folded) / `[1127-glu] REACH` (folded lane) | DEFAULT-ON (2026-10-27) | riir-ai Bench 1018; `.benchmarks/897_issue1124_v10tglu_kernel_probe.md` |
| Metal/CubeCL decode dense-pair GEMV (issue 1129) — `RIIR_GDN_DENSE_PAIR_FUSED` demote-only; `RIIR_GDN_DENSE_PAIR_WG` unpromoted | DEFAULT-ON | riir-ai Bench 1007/1014 |
| GDN prework-fused prefill (riir-ai Issue 1004 R2) — `RIIR_PREFILL_PREWORK_FUSED` demote-only on Metal, promote-only elsewhere (macOS scoping PERMANENT — cudarc never consults it) | DEFAULT-ON, Metal/wgpu | riir-ai Bench 024; `.benchmarks/i1149_prework_cuda_routing` |
| Fused GDN fold-tail / SwiGLU-rotate decode — `RIIR_GDN_PERMUTE_ROTATE_FUSED` / `RIIR_SWIGLU_ROTATE_FUSED` | opt-in (e2e below bar) | riir-ai Bench 995 |
| Decode-hold rungs — `RIIR_GDN_CORE_FUSED=1` (G2 negative), `RIIR_GDN_BETA_EXPAND_FUSED=1` (re-cell owner-gated); tripwire `RIIR_QUANTIZE_CENSUS=1` (expect 0) | opt-in | Bench 898/899; issue 1127 |
| Dense qwen3.8 Metal prefill (Issue 1150) — `RIIR_QWEN38_DENSE_MM2D=1`; `client.sync()` after uploads (racy negatives once minted fake shape laws) | opt-in | `.benchmarks/i1150_qwen38_wgpu_session.md` |
| GDN delta-roll state journal (Issue 049) — feature `gdn_delta_roll` | opt-in; promotion rides a consumer | Bench 885 |
| Headless stability-depth readout (Issue 1106) — rides `twt_profile` | calibration-only, never serving | reflex Issue 094 |
| TOKENGATE (Plan 622) — feature `tokengate` | opt-in; full-overlay promotion REFUSED (T3.4) | bench 888/889/891 |
| Scale-invariant attention — feature `scale_invariant_attn` | opt-in | katgpt-rs Plan 622 |
| Multi-host pipeline (plan 623) — feature `multi_host`; RDMA verbs `rdma`, `rdma_metal`; `RIIR_DIST_ASYNC_FINISH` promote-only; `RIIR_DIST_STEP_TIMING` | opt-in | Bench 927/928/929; issue 1152 |

**Plan-623 wire laws (measured):** recv rings ≤ 3 MB or runtime-probed,
never hard-coded (the cliff is box-load-dependent, Bench 928); UC is the
wire shape (RC refused, RDMA WRITE never completes — SEND/RECV only); a
dead peer surfaces as a poll TIMEOUT, never a WC error; GIDs are not
IPv4-mapped on TB (index-0 fallback works). Wire cells run
FOREGROUND-chunked; detached runs need a SIGNED `dist_stage` (issue 1152's
owner gate) before they are trusted. Prefill chunk default 128 stands.

**Engagement proof first:** before any A/B, prove the arm is reached in the
posture being measured (the Issue-1124 T1.5 lesson — the folded league lane
had no GLU path and the A/B would have been pair-vs-pair).

## Build Commands

```bash
cargo check
cargo clippy --all-targets -- -D warnings
cargo test --lib
# GDN quant-certification + bonsai2 rotation integration tests carry required-features (Cargo.toml [[test]] rows):
cargo test --test issue879_gdn_quant_certification --features deltanet_ternary_inference
cargo test --test bonsai2_rotation_load --features bonsai2_hadamard
# Issue 046 GDN spectral-class spec (kernel-recovered transition vs the closed form; plain deltanet, no model):
cargo test --test issue046_gdn_spectral_class_spec --features deltanet_inference
# Issue 028 disaggregated handoff battery (the 6.7 GB PQ2_0 arm is #[ignore]d — BONSAI_PQ2_0_GGUF env, release run):
cargo test --test issue028_disaggregated_handoff --features deltanet_ternary_inference
# BONSAI_GGUF env (defaults to a riir-train data path) names the real checkpoint for the certification test's full-file arm.
# The EXL3 trellis lane (opt-in; issue 001, closed — record in .docs/001; CPU reference + fast arm here, GPU kernels in riir-infer-gpu; oracle = the CPU reference):
cargo test --lib --features exl3
# The EB-Sampler commit-policy lane (opt-in `entropy_bounded_commit` on riir-infer-gpu; katgpt-rs 917 T2' — armed per-config by `Gemma2D2fConfig::eb_gamma >= 0`, disarmed default −1.0 keeps every config on the incumbent τ seams; A/B blocked on riir-train Plan 437 Phase 0):
cargo test -p riir-infer-gpu --lib --features entropy_bounded_commit
cargo test -p riir-infer-gpu --features exl3_gpu,cuda_backend --lib  # native CUDA arm (release recommended)
# The scale-invariant attention lane (opt-in `scale_invariant_attn`, implies
# `deltanet_ternary_inference`; katgpt-rs Plan 622 Phase 7 engine lane — the G3 zero-shot
# vehicle on the packed Bonsai ternary decode path; canonical status lives in
# ../katgpt-rs/.plans/622_scale_invariant_attn.md; GPU legs open, GPU-window-gated):
cargo test --lib --features scale_invariant_attn   # 8 attention_si tests incl. 3 wiring e2e
cargo build --release --features scale_invariant_attn,bonsai2_hadamard --bin si_zero_shot_ppl  # the G3 harness (its OWN required-features posture — the P43 law)
# katgpt-rs Issue 959 T6 — the f64 golden-NLL lossy-quant oracle (opt-in `golden_nll`,
# forwards katgpt-core/fixed_log_softmax; `riir_infer_core::golden_nll` is
# the reusable oracle, the bin's `--golden-nll` the reference-mode arm:
# base-only, pooled incumbent-vs-golden Δ + the fp32/f64 confidence-band
# split + the disagreement census; a build without the feature REFUSES the
# flag loud). Instrument-grade — never a serving path:
#   cargo run --release --features scale_invariant_attn,bonsai2_hadamard,golden_nll \
#     --bin si_zero_shot_ppl -- --gguf <model> --corpus <text> --arms base --golden-nll
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
non-macOS CUDA backends, `laya-riir-cuda` since riir-infer Issue 002) +
`crates/riir-infer-ane` (the SHARED Core ML/ANE host — plan 619 T1.0:
digest/verify/stage, the compile cache, `MLModel` load with compute
units + `functionName`, the multi-array predict; macOS-only by contract,
a zero-dep stub elsewhere; the objc2 family is ITS dep set) +
`crates/riir-infer-clef` (the clef-text decision lane, plan 619 —
`clef-ane` opt-in, compiles empty by default, consumes the shared host).
Vendored crates.io forks under `vendor/` (`[patch.crates-io]` in the root
manifest).

## Numbering Discipline

Issue, plan, doc, benchmark, and research numbers are **monotonic and never
reused**: read the target dir's `.highwater`, use `value + 1`, write back.

⛔ **One lane only (the carve-era two-lane counter was RETIRED 2026-10-21).**
`998/1003/1004` are MOVED riir-ai documents, not allocations made here;
`.highwater` is the ONLY counter (it sits above every number the retired
local lane, top `049`, ever used). Never derive the next number from the
directory — a closed issue file is removed and its number would be REUSED
(the katgpt-rs `.issues/121` recycling bug). Cite cross-repo numbers with
the repo name (the workspace `issue_citation_gate`).

Self-allocation headings follow the grammar the citation oracle reads —
`## <date> — Issue NNN: <title> CLOSED — <verdict>` — the number sits
IMMEDIATELY before its title delimiter; never write a word (CLOSED, resolved,
follow-up) between the number and the delimiter — that shape is unread to the
sweep and re-reds the repo (katgpt-rs Issue 921).

## Branch

`develop` is the working branch. Don't create feature branches; commit
directly on `develop` per the global rule.
