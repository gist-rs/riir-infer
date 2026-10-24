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
serving repo); the scale-invariant attention lane LANDED 2026-10-11
(opt-in `scale_invariant_attn` — `transformer/attention_si.rs` + the
`si_zero_shot_ppl` G3 harness, katgpt-rs Plan 622's engine vehicle);
the GDN L8 eager-prefill arm LANDED 2026-10-19 (plan 621, opt-in env
`RIIR_PREFILL_GDN_L8=1` — `prefill_cuda_deltanet_l8.rs`, gated on
ternary_gemv_cuda_raw + ternary_gemm_batched + cubecl_runtime,
non-macOS; NOT-PROMOTED: kernel-only k4w4 1.26–1.51× but e2e FLAT at
the DQ-lane posture, bit-identity bytewise vs rowpar — see
`.benchmarks/884_plan621_gdn_l8_eager_prefill.md`; the league-posture
e2e re-gate EXECUTED 2026-10-21 (riir-ai Bench 993): G1 bytewise ×4 at
the league posture and +3.2% pp2048, band-clause gap → opt-in pending
the owner gate; second window REPLICATED 2026-10-24 (riir-ai Bench 997:
+2.76% pp2048, G1 bytewise ×8, between-window deltas inside the band;
pp4096 NO-READ — the OFF baseline's own halves moved −8.9%; promotion
STAYS OWNER-GATED));
the fused GDN fold-tail + SwiGLU-rotate decode kernels LANDED 2026-10-23
(riir-ai Issue 1113 lever 1 — `deltanet_rotation_cubecl.rs`, opt-in env
`RIIR_GDN_PERMUTE_ROTATE_FUSED` / `RIIR_SWIGLU_ROTATE_FUSED`, macOS
cubecl lane; bit-identical to the split chains BY CONSTRUCTION — the
full-K3 zgate-folded variant measured 1-ULP-divergent, the incumbent
zgate's compiled fma contraction does not reproduce across kernel
compilations, so the zgate stays its own launch; −160 launches/token
EXACT but e2e +1.40% BELOW the promote bar → opt-in kept — riir-ai
Bench 995; the kernel-EXEC GEMV campaign RESOLVED 2026-10-23 (riir-ai
Issue 1114, closed — the calibration re-priced the 767/995 per-shape µs
as 2–5× overhead-polluted; the class is WORKGROUP-COUNT-limited, not
shape-cursed; the dispatch-only `launch_rowtiled8_wg`/`launch_f16scale_wg`
shape rule (m≥8192→128, m≥4096→64, else 256) is byte-identical and won
+4.88/+5.52% e2e GOAT ×2 on the m5ultra — shipped here as `RIIR_GEMV_WG_OCCUPANCY`
(default OFF, `gemv_ternary_cubecl.rs` + `launch_f16scale_wg`); promotion
OWNER-GATED; measured basis riir-ai Bench 996, record in riir-ai HISTORY.md);
the GDN delta-roll state journal LANDED 2026-10-20 (Issue 049, opt-in
`gdn_delta_roll` — `deltanet/delta_roll.rs`: post-commit state recovery by
rank-1 unwind from the final state + per-token (u, k, β, g) contexts, ~64×
vs materialized per-token snapshots, under the per-head decay-product
conditioning gate; G1 drift 6.3e-6 worst admitted cell at floor 0.25,
G4 zero-alloc, drift-vs-replay ~6× cheaper than re-forward — Bench 885;
consumer seams named: spec-decode retry, kv_evict recovery, disaggregated
handoff — promotion rides a consumer);
the HEADLESS stability-depth readout LANDED 2026-10-22 (Issue 1106, rides
`twt_profile` — `riir/stability.rs` in the laya crate: per-stage
last-position residual cosine + L2 over the `forward_capture` seam →
earliest stabilized-suffix stage; sequential-fold deterministic,
zero-alloc track (`stability_g4_alloc`), brute-force bit-parity gates +
the real-checkpoint e2e smoke (`issue1106_stability_capture_smoke`, skips
loud without weights); calibration-class, never a serving path; the
substrate half of reflex Issue 094 — bands are the consumer's to pin);
remaining planned work: the CLEAN GPU kernel layer migration.

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
# Issue 959 T6 — the f64 golden-NLL lossy-quant oracle (opt-in `golden_nll`,
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
The files `998/1003/1004` are MOVED documents from the riir-ai carve
(riir-ai's numbers, not allocations made here); `.highwater` reads 1005 and
is the ONLY counter — new allocations continue from it, which sit above
every number the old local lane (top `049`, retired with its
`.highwater_local` file) ever used, so no fold collision is possible. The
older "reserve 1005–1008" text was STALE and self-contradictory (riir-ai's
climbing counter cannot be outrun by skipping a range — it had already
consumed the whole reserve by 2026-10; Issue 1106 landed under that stale
reading, kept per pushed-numbers-are-never-reused, with the real protection
being repo-named citations, enforced by the workspace `issue_citation_gate`).
A plain max-of-disk rule is still wrong — a closed issue file is removed and
its number would be REUSED (the katgpt-rs `.issues/121` recycling bug):
always read the counter, never the directory.

Self-allocation headings follow the grammar the citation oracle reads —
`## <date> — Issue NNN: <title> CLOSED — <verdict>` — the number sits
IMMEDIATELY before its title delimiter; never write a word (CLOSED, resolved,
follow-up) between the number and the delimiter — that shape is unread to the
sweep and re-reds the repo (katgpt-rs Issue 921).

## Branch

`develop` is the working branch. Don't create feature branches; commit
directly on `develop` per the global rule.
