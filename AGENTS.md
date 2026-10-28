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
the GDN L8 eager-prefill arm LANDED 2026-10-19 (plan 621,
`prefill_cuda_deltanet_l8.rs`, gated on ternary_gemv_cuda_raw +
ternary_gemm_batched + cubecl_runtime, non-macOS; **PROMOTED
DEFAULT-ON 2026-10-24 (owner call via reviewer verdict)** —
beyond-window replication ×2: riir-ai Bench 993 +3.22% (10-21) /
Bench 997 +2.76% pooled (10-24) pp2048, G1 bytewise ×8 (both postures,
both days, same FNV pins); the pre-registered ±8.4% band clause was
SUPERSEDED BY MEASUREMENT, never satisfied — it was calibrated as a
BETWEEN-window band, the paired 2×2 interleave reads ±0.3–0.8%
within-window and ∓1.3–1.7% between-window common-mode; kill-switch
`RIIR_PREFILL_GDN_L8=0` (any non-1/true/on value; env wins) restores
the incumbent; posture gates `recl8_posture_gates` pin both ways
(unset = ON, `=0` = incumbent); pp4096 e2e re-read EXECUTED 2026-10-24
post-promotion (window 3: +2.62% pooled, G1 bytewise ×4, clean
interleave — the owed cell CLOSED, every measured posture carries a
positive replicated read); prior
record: kernel-only k4w4 1.26–1.51× at the DQ-lane posture
(`.benchmarks/884_plan621_gdn_l8_eager_prefill.md`), the league
re-gate opened opt-in 2026-10-21 (riir-ai Bench 993));
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
(**PROMOTED DEFAULT-ON UNCONDITIONALLY** — 2026-10-24 on the wgpu/Metal runtime
(owner call via reviewer verdict) and 2026-10-25 on the `cuda_backend`
CudaRuntime (Issue 1115 / Bench 887: the 4090 prefill-tail lm_head A/B, G1
FNV ×6+6, G3 1.0013×/1.0000× — the platform split deleted); env
`RIIR_GEMV_WG_OCCUPANCY=0` demotes byte-identically on every platform;
measured basis riir-ai Bench 996, record in riir-ai HISTORY.md);
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
the TOKENGATE lane (opt-in `tokengate`, Plan 622) through T3.4: Phase 1
f16-bias wire + Phase 2 pruned prefill/pad buckets/decode fold + the
L=mid-attention pin (bench 888) + Phase 3 overlay fns (rank cooling,
magnitude table, scorer family) + the T3.4 four-arm GOAT gate
(`tokengate_fourarm` bin, bench 889 — VERDICT FAIL: derived dominates
needle retrieval 9/10 cells but loses the continuation-ppl axis at
K∈{0.5,0.75} in both postures and one lucky-seed floor cell at K=0.0625;
FastKV-class is ppl-competitive selection-only, needle-0-blind at every
K, and collapses under the derived-magnitude bias overlay — the
full-overlay promotion REFUSED, opt-in only, T4.2 owns the verdict
write-up — the hiddennorm ablation at L=15 retrieves 1.000 at every K
including the mass arm's one loss (K=0.0625): the retrieval dominance
is the mid-layer placement, not the mass arm); remaining planned work: T3.5 (league random-K census), T4.1 (measured TTFT-vs-K cells — the break-even half is DONE, bench 891: ceiling ~1.5–1.8× at K≤0.25, attention ~linear in p so the prunable share never grows with context; the T2.4 vehicle priced and owner-gated), T2.4 (GPU graphs — priced, owner call), T1.6 (Metal), and the CLEAN GPU kernel layer migration.
The multi-host pipeline lane LANDED Phase 1 2026-10-26 (plan 623: `LayerRange` +
`ActivationFrame` wire + `HybridStateSegment` + stage handshake in core; GPU range seams
`forward_range_dispatch`/`set_hidden_state`/`read_hidden_state`/`set_pos` + gemma2
`forward_gpu_range` in riir-infer-gpu — feature `multi_host`, opt-in): G1 bit-identity
every split EXECUTED on Bonsai-2 27B (63/63 qwen splits ×4 tokens, Metal). **MODEL SCOPE
(owner law, 2026-10-26): the production surface is Bonsai/ternary (+ qwen3.8) ONLY —
gemma-2-2b is RETIRED (riir-train Issue 617): its executed G1 cell was RESCINDED same-day,
the gemma2 seam stays compile-only, and NO new gemma-2-2b execution cells anywhere (T0.3's
1-box cells, Phase 0/2+ all qwen/bonsai-scoped). The gemma2 ARCHITECTURE lanes that serve
LIVE products (laya checkpoints, clef-ANE 0.6b) are NOT this retirement — they keep their
own gates.** Phase 0 UNBLOCKED 2026-10-26 (1109 T0–T3 DONE: bridge live at
192.168.10.x, bmo workspace + model synced, RPC smoke green; T4 RDMA stays
owner-physical); Phase 2+ gated on Proposal 042 D1.
The GLU-fused ternary prefill FFN arm LANDED 2026-10-26 (Issue 1124 T1, the llama.cpp
#29948 distill — `1320dda`): gate/up weight rows INTERLEAVED in 8-row chunks at
mirror-build time (`pack_bitplanes_to_q2_interleaved_gate_up` +
`interleave_gate_up_group_scales`, `prefill_cuda_mma.rs`) so the v10t MMA body runs
VERBATIM (single accumulator — each m16n8 fragment's row halves are the gate/up dots of
the SAME hid row; regs=128 local=0, the Bench-895 spill cliff dodged) and SwiGLU is
applied IN the transposed staging pass from registers (`GEMM_BODY_V10T_GLU_Q8`,
`gemm_ternary_i8_mma_v6_src.rs`) with the Bench-721 canonical forms in the standalone
kernel's op order — bit-identical to (v10t pair + standalone SwiGLU) by construction,
G1 gate 0 bit-diffs ×2 shapes vs the real standalone kernel. Env
`RIIR_PREFILL_FFN_GLU` — **PROMOTED DEFAULT-ON 2026-10-27** (riir-ai Bench 1008 / verdict
`9ca951de`: paired league windows +6.31/+6.72% pp2048, +7.27/+6.47% pp4096, 4/4
positive, G1 pins EXACT 4/4; riir-infer `57fa7a7`) — the env is DEMOTE-ONLY
(exact-literal `0`/`false`/`off` restores the incumbent pair path byte-identically;
unset = armed; fmt-route packed mirrors + `mlp % 128 == 0` posture; failure
degradation byte-identical). **T1.5 (2026-10-27, `c60b34d`): the arm was wired ONLY
into the !folded `dispatch_ffn_block` — the FOLDED league lane
(`prefill_cuda_full::ffn_block`) had NO v10tglu path; the staged T2 A/B would have
been pair-vs-pair (caught by the engagement-proof-first protocol — the `[721-trace]`
line is unreachable at the folded posture; the folded lane's proof is the one-shot
`[1127-glu] REACH` line). Kernel probe Bench 897
(`.benchmarks/897_issue1124_v10tglu_kernel_probe.md`): **1.258× @p2048 / 1.227×
@p4096** at the league FFN shapes (m=17408, n=5120). T2 DONE — riir-ai Bench 1008
(`../riir-ai/.benchmarks/1008_issue1127_t4_recell_bundle.md`); follow-ups: the
mirror-halving commit, the next-bout default-posture confirm (< +3% pooled or any
negative window → the demote hatch reverts). T3 mined via refine Batch 287
(`d871a0fc`, riir-refine Plan 292 / Research 325).
The decode-hold escalation rungs (the Bench-1004 RED response, 2026-10-26) LANDED as
eval-gated opt-ins with records, zero production change: `RIIR_GDN_CORE_FUSED=1`
(one-launch GDN core — G1 exact everywhere, G2 −3.5%/−2.1% NEGATIVE, stays opt-in
default-OFF so the shape is not re-tried blind, Bench 898 `4132d94`) and
`RIIR_GDN_BETA_EXPAND_FUSED=1` (the 4→3 beta/decay+expand rung — G1 exact ×6 windows,
G2 direction-positive 3/3 paired windows ~+0.6–1.0% e2e, −1.5% GPU busy, Bench 899
`8aabe55`; NOT promoted — the league re-cell is owner-gated, agenda in issue 1127).
`RIIR_QUANTIZE_CENSUS=1` prints the standalone-quantize launch census per capture
(expected 0 — the K1–K5 fused rotation kernels already eliminate it; the standing
decode tripwire, falsifier `a2ebb37`). Remaining rungs + the re-cell bundle: issue 1127.
The Metal/CubeCL decode dense-pair rung LANDED 2026-10-26 and is **DEFAULT-ON**
(riir-infer `c400dbc`, issue 1129 / riir-ai Bench 1007): `gemv_plane_f32_pair` — the
folded GDN a/b f32 escape-set GEMVs (48×5120 each, 64 launches/token) fused into ONE
launch, per-row math verbatim → bit-identical by construction (G1 to_bits EXACT ×6;
e2e +9.88/+9.07% decode ABBA); the dense-leg tax halved 4.73→2.56 ms/token. Env
`RIIR_GDN_DENSE_PAIR_FUSED` is DEMOTE-ONLY (unset = ON; any non-1 set value restores
the incumbent two-launch pair); `RIIR_GDN_DENSE_PAIR_WG` ships unpromoted (the geometry
rung measured DEAD — the pair cost is a per-dispatch floor, flat across wg sizes).
Remaining scope in issue 1129: the `cuda_backend` CubeCL arm (the scored 4090 lane is
cudarc, whose K2 already fuses this pair — untouched).
The GDN prework-fused rung (riir-ai Issue 1004 R2, Bench 024) is **PROMOTED DEFAULT-ON,
Metal/wgpu-scoped 2026-10-28** (riir-infer `e2edf61`+`2581fdd`, riir-ai `d25c7c460`;
delegated owner verdict): conv1d+SiLU+q/k L2-norm+head-expand in ONE dispatch per
layer (~67 fewer dispatches, ~160 MB/layer less traffic), bit-identical by
construction — G2 e2e pp2048 1.043× / pp4096 1.045× on the m5ultra re-cell,
post-promotion G1 anchors EXACT at the league posture both lengths. The initial
toggle value derives from `cfg!(target_os = "macos")` — the `cuda_backend` posture
keeps the incumbent pair until its own 4090 cell — **CLOSED N/A 2026-10-28** (issue 1149,
`.benchmarks/i1149_prework_cuda_routing`: the toggle is fn-private to the CubeCL forward while
the cudarc whole-prefill does its GDN prework through four native dispatches —
conv1d/carry/beta/expand — and structurally never consults it; runtime routing proof `69e7d8e`:
whole-prefill +0 launches at p=2048+4096 with the arm setter-armed vs the CubeCL fall-through
lane's +48 — the macOS scoping is PERMANENT, the env stays the fall-through lane's only arm; no
promotion owed, the Issue-1115 G1/G2 cells subsumed); the GEMV-occupancy per-backend scoping
precedent. Env `RIIR_PREFILL_PREWORK_FUSED`
is DEMOTE-only on Metal / promote-only elsewhere; `set_prefill_prework_fused` stays
the harness arm toggle. R1 (`deltanet_recurrence_smem_staged`) stays opt-in pending
its pre-authorized second M5 window (riir-ai Issue 1004).

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
