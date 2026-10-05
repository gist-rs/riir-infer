# 007 — Triadic Linear Attention: 3D tensor states for the GDN lane (paper distill + league lane intel)

**Status:** DISTILLED — pending owner decision (kernel-mining batch queued behind riir-refine Batch 201; triadic checkpoint = trigger-gated, no consumer until one exists).

Date: 2026-10-05 · filed from the M3, user-directed `@research` + `@distill` session.

## Source

- Paper: Sieberling, Runwal, Jin, Chin, Panda, Kim — **"Triadic Linear Attention: Three-Dimensional Recurrent States for Long-Context Sequence Modeling"**, arXiv:2609.36529v1 (2026-09-29, MIT-IBM Computing Research Lab / MIT).
- Kernels: `OliverSieberling/cute-triadic-gdn` **@ `2caa4098073da92c2ba0d573df6453ceb7dbce45`**, MIT — CuTe DSL, sm90 + sm100/sm103 (tested on B300). Quotes below re-verified at this pin before the clone was deleted.
- Training code: `OliverSieberling/TriadicLinearAttention` **@ `d69108ef0e41dac814cc3d918793d0b30132b832`**, MIT — minimal nanoGPT-style (train.py / model.py / configs / data prep). Both clones ephemeral, removed at close.
- Distill verdict on the kernel repo: riir-refine Research 236 + the queue-snapshot line (same date).

## Commit map (this session's landing, Session: m3-triadic-la, 1791196114)

- riir-train Plan 442: `396aa4cf` · riir-refine Research 236 + queue line: `6a95b24d` · riir-ai Issue 971 dated entry: `fbad82a2b` · this note + highwater: the landing commit of this file in riir-infer (hash in the session summary; not self-citable).

## TL;DR (commercial + league value)

Linear attention writes a rank-1 dyadic update `S += k⊗v` into a d×d matrix state; triadic LA writes the **triadic** product `S += k⊗k'⊗v` into a **d×E×d tensor state** and reads with TWO queries, `o = S×₁q×₂q' = Σ (qᵀk)(q'ᵀk')v`. An E-dim second key = **E-fold state capacity for two small projections** (E=8 adds 1.2% params at their shapes). Fully compatible with per-slice data-dependent forgetting, the delta rule, and chunkwise-parallel training. At 400M/1.3B: pushes the GDN-vs-Transformer perplexity crossover from ~10k to beyond 64k context, wins the state-matched comparison against larger-heads / wider-values / grouped-values / more-heads at both 2× and 4× (which barely help or degrade), and in a 3:1 GDN/GQA-8 hybrid, a triadic state (E=4) beats doubling the KV cache (GQA-4) on ppl + NIAH at roughly **half the memory at 64k**.

**Why this lane:** E=1 reduces exactly to Gated DeltaNet — the Bonsai-27B league arch. The paper's GDN baseline kernels follow **FlashQLA** (QwenLM's GDN kernel library — already recorded as lane prior art in riir-ai Research 367). The hybrid result is the long-context positioning argument for our own league: state capacity, not KV cache, is the cheaper axis at long context.

## The math (what transfers)

1. **Joint key + Kronecker factorization.** Flatten `κ = k⊗k'` (dim d·E) and the model is Gated DeltaNet with a d·E key. The chunkwise inner products then factorize: `(q⊗q')ᵀ(k⊗k') = (qᵀk)(q'ᵀk')`, so chunk masked attention stays C×C with d-dim keys — the joint form's C²·d·E cost collapses to C²(d+E) plus a C×C second-key mask `R' = Σ_e (q'_e k'_eᵀ) ⊙ Γ_e` (Eq. 7/9). The E-fold cost is confined to the STATE.
2. **Per-slice forgetting.** Each slice `S[:,e,:]` gets its own scalar gate `α_{t,e}` (Eq. 3) — within a slice the decay is scalar (GDN-class), across slices it is channelwise (GLA-class). With E small the gate projection stays negligible.
3. **Delta rule on two keys.** Erase the two-key readout before writing: `S += β k⊗k'⊗(v − S×₁k×₂k')` (Eq. 4); equivalently GDN on the d·E joint key with per-slice scalar decay (Eq. 5).
4. **Chunkwise form with both (App. A, Eq. 11).** One C×C UT-transform matrix `T` (from `K Kᵀ ⊙ R`, R = the decayed second-key mask) shared by all E slices; per-slice scalings ride `diag(k'_e ⊙ γ_e)` terms. Unlike DeltaNet's precomputed `W = TK`, they apply T to the residual because per-slice decay has no shared `W` analog.
5. **Numerical stability law (the anti-lever).** Every decay factor `γ^r_e/γ^s_e` is evaluated directly as `exp(diff of accumulated log gates)`; the difference is non-positive by construction, so **no clamp is ever needed**. The cheaper construction (factor around a mid-chunk reference, 2CE exponentials instead of C²E) creates positive exponents that overflow FP32 above e^88.7; their FP64 emulation of just that clamp measures rel. error 0.05 at within-chunk decay e⁻²⁰⁰ and 1.5 at e⁻⁶⁰⁰ — the cheap form is REFUSED on measured numerics, not taste.
6. **Ablations worth keeping:** keep d_k fixed (128), grow E (balancing d_k against E degrades); **non-negative** activation on k'/q' (softplus/sigmoid) beats SiLU/none — signed entries let reads and writes cancel across slices; L2-normalize k'/q' (E=1 then reduces exactly to the base mixer).
7. **Upcycling (paper §3.3, repo does NOT ship the surgery script — provenance split).** Expand a pretrained E=1 GDN to E=8 by copying the forget gate to every slice and initializing k'/q' projections + their convs from scratch, then long-context-extend: recovers ~½–¾ of the from-scratch triadic gain. The short-context data that dominates pretraining needs little state; enlarge the state at the extension stage.

## Kernels (what the repo ships, quotes at the pin)

- `gdn_joint_call(q, k, v, k2, q2, g, beta, scale, cu_seqlens)` — fwd + grads for all 7 inputs; E ∈ {1,2,4,8,12,16}; `T must be a multiple of 64` unpached, or packed varlen via `cu_seqlens` (state resets at every document start, one packed row). Verified against FP64 references at the 1.3B shape, rel ℓ₂ < 1%; every reduction follows a fixed order with **no floating-point atomics — bitwise reproducible gradients** (paper App. A).
- **State tiling:** "Our kernels therefore split the state of each head along the value axis into blocks of 32 columns, one per thread block, and each thread block keeps all E slices of its columns in registers for the entire sequence, so the full state is never held in one place." At E=8 a head's FP32 state is 512 KiB (2× a Hopper SM's register file); one 32-col block is 128 KiB.
- **Warp specialization (sm90 fwd, verbatim docstring @ pin):** "Forward recurrence for E <= 4 (Eq. 11), warp-specialized: producer, state, value and output warpgroups; stores the state at the start of every chunk in BF16 for the backward." At E=8, two state warpgroups hold 4 slices each over async tensor-core MMA, a producer warpgroup TMA-loads double-buffered smem and forms U, a fourth forms the output; per-role `warpgroup_reg_alloc/dealloc` rebalances registers.
- **`GJ_SAVE_MASKS=0`** (`ops/gdn_joint.py:170-173` @ pin): backward recomputes chunk masks with the forward's own kernel — "bitwise identical", one extra mask kernel per layer, 768 MiB saved at 128k tokens. Store-vs-recompute as a measured env knob.
- Training overhead (paper Fig. 4, H100): E=8 +28–30%, E=4 +14–15%, E=2 +9–11% fwd+bwd vs GDN; faster than the Transformer beyond 4k (E=8), 5.1× at 64k.
- **Decode is NOT measured in the paper.** Decode reads are E matvecs (E× the state traffic of GDN) — a triadic checkpoint's decode row would regress roughly E× unless kernels amortize it. The successor-arch threat is long-context-quality-weighted, not free. This caveat rides every lane-intel claim.

## Path 0 / three-track routing (per-track verdicts, never pooled)

| Track | Verdict | Home |
|---|---|---|
| (a) modelless inference | **No file.** Kronecker factorization + slice-decay stability are runtime math, but no modelless surface here consumes a linear-attention state (no product-key serving path, no LA hot path). Forcing an open primitive would be inventing a consumer. | — |
| (b) self-adaptive runtime | **Fusion idea only, novelty TBD** (below). | — |
| (c) model-based training | **Plan filed** — recipe + trigger + GPU-hours + GOAT gate design. riir-train Plan 442. | `../riir-train/.plans/442_triadic_gdn_state_upcycle_recipe.md` |
| Kernel substrate (riir-infer MOAT) | **This note** — successor-arch candidate for the GDN lane; no triadic checkpoint exists, so the lane consumes it via (i) the distill batch and (ii) the Issue 971 reference-design pool. | this file |
| Healer corpus (riir-refine, fusion priority #2 consumer) | **YES B− corpus (kernel_opt, all rules NotBenchable-annotated) / LEAGUE INTEL HIGH** — riir-refine Research 236 + queue line; batch queued behind Batch 201. | `../riir-refine/.research/236_cute_triadic_gdn_kernel_distill.md` |
| League lane intel | **HIGH** — see below; dated entry appended to riir-ai Issue 971. | `../riir-ai/.issues/971_4090_prefill_hold_defense_chunked_gdn.md` |

No-GD advocate findings: the stability law (5), the factorization (1), and the tiling law are the closed-form extracts — all three are kernel-side, not serving-side, hence they land as kernel_opt rules rather than katgpt-rs primitives. Model-based advocate findings: the recipe table (train-from-scratch E sweep; upcycling; softplus; per-slice gates; fixed-d_k) is filed in Plan 442. No advocate finding was discarded.

## League lane intel (Issue 971 mapping)

- Issue 971 (riir-ai) is the parked defense for the chunked-GDN prefill breaker: llama.cpp PR #26001 (+9.9–10.1% e2e @ISL≥2000 on ad102, the Bonsai H=48 class), defense = Plan 533 Phase 2-4 chunkwise-recurrence revival, reference implementation = the PR diff; the 09-18 interim added the fused single-pass thread (gdn-fused-v2, FLA-parity). **Trigger NOT fired as of the 10-03 watch** (zero chunked-GDN content in the fork).
- `cute-triadic-gdn` is a **second complete reference implementation** of the chunkwise GDN family (FlashQLA-following chunkwise schedule, fwd + full bwd, packed varlen, E=1 ≡ GDN) — for the revival, the CuTe sources are a readable spec of the same 3-stage math (UT transform → masked attention → state update) our cudarc port needs. Recorded in Issue 971 as a dated interim entry.
- Successor-arch watch: if any opponent ships a triadic-class GDN checkpoint, the long-context quality cells (ISL≥2000-4000 class) move against us at fixed params — the paper's own hybrid table is the sizing argument. Decode-side cost (E× state reads) is the mitigating fact; neither half is measured on our league shapes. The watch is recorded, not armed.

## Fusion (recorded, novelty TBD — no file, no issue)

The tensor-product view (Smolensky binding: value = filler bound to TWO roles, unbound by two contractions) is the same shape as our latent two-role bindings: a per-NPC associative memory that binds (context-key, role-key) → value would gain E× capacity at fixed state parameters, with per-role decay (the per-slice gates read like per-timescale memory channels — the paper's half-life analysis found slices spanning 0.2 → 1080 tokens, i.e. one head holds a multi-timescale memory hierarchy for free). Closest shipped cousins: `katgpt-sense::evolve_belief` (vector recurrent state), HLA scalars, `NeuronShard.style_weights`. None runs a sequence-indexed associative memory today, so there is no consumer to gate. Re-open trigger: any surface that needs per-entity fixed-size associative recall with two queryable roles.

## What's private vs open

The paper, both repos, and the extracted rules are public material. Nothing here touches riir-* private IP; the note itself is in the public substrate repo. Any future triadic port lands in `riir-infer` (public substrate) under a feature flag with G5-style parity gates (bit-drift ≤ 1e-3 vs the repo's own FP64-verified reference at our shapes) BEFORE any published number, exactly the laya-lane discipline.

## Validation protocol (if the owner ever pulls a trigger)

1. **Kernel-mining batch** (queued): compose the 2–3 kernel_opt rules per riir-refine Research 236 after Batch 201 lands; all rules NotBenchable-annotated (sm90/sm100 cuTe; our hosts are Metal/CubeCL + sm89 cudarc — no validator host exists).
2. **Issue 971 revival** (trigger-gated): the CuTe chunkwise sources join PR #26001 as reference material for the cudarc port; numerics gate = argmax-stability + league warm-pin FNV/argmax pins (NOT bit-identity — chunked reduction order differs), per the issue's own contract.
3. **Triadic checkpoint** (owner-gated, currently unaffordable at 27B): Plan 442 holds the recipe + GOAT gate; a ≤1.5B validation cell is the 4090-affordable shape if ever pulled.

## P0–P3

- P0 (this commit): research note + distill verdict + queue line + Issue 971 dated entry + Plan 442.
- P1 (queued, behind Batch 201): kernel_opt mini-batch (2–3 NotBenchable decision-content rules).
- P2 (trigger-gated): Issue 971 chunkwise revival consumes the repo as reference; league re-pin bout per the issue's own ladder.
- P3 (owner-gated): any triadic training run per Plan 442; decode-side E× caveat must be priced before any league claim.
