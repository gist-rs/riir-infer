# Issue 1126 — decode GDN core fusion: beta/decay + expand/L2 + fused recurrence in ONE launch (the Bench-1004 escalation rung 2 — the #398 conv-chain twin, the CUDA lane's real decode delta)

**Status:** CLOSED — MEASURED NEGATIVE (Bench 898: G1 bit-identical everywhere — kernel-level 5-step compounding + the real-model 512-step ARM-4 FNV exact; G2 FAIL −3.5%/−2.1% decode e2e, both windows agreeing). The kernel + test + env gate SHIP opt-in (`RIIR_GDN_CORE_FUSED=1`, default OFF — zero production change) with the record so the shape is not re-tried blind; the `recurrence_fused_body_impl` body-share refactor STAYS (the DRY win, bit-tested both ways). The census's remaining decode levers are named in the bench record: the 4→3 beta_decay+expand-only fusion (launch-bound with launch-bound, needs the paired-interleaved discipline), the attention-chain fusion, the embedding-dequant+inverse-FWHT fusion. Escalation defense continues there.

**Claimed:** shikuwa-4090-i1126, 1793021000, expires 2026-10-28T13:30:00+00:00 (DISCHARGED same session — lease spent on the implementation + the A/B)

## Rung-1 adjudication (the dedup grep, CUDA side — measured in code, not assumed)

Issue 1125's premise ("the decode consumer still pays a separate quantize pass") is **REFUTED on the cudarc/CUDA lane at the scored posture**: the folded graph decode path (`forward_from_x_devpos` → the layer fns, `ternary_deltanet_gpu_forward_cudarc.rs`) fuses EVERY activation quantize into a producer kernel — K1 `rmsnorm_sign_fwht_quantize` (attention/final-norm inputs), K2 `gdn_input_rotabq` (GDN input stage incl. the dense a/b GEMVs), K3 `gdn_out_norm_gate_permute_rotq` (GDN output stage), K4 `gate_sigmoid_sign_fwht_quantize` (o_proj input), K5 `swiglu_sign_fwht_quantize` (ffn_down input) — all promoted default-on at Bench 940 (2026-09-19), five weeks BEFORE the opponent's #395/#402 quartet. The mechanism 1125 files is **already shipped here**; 1125's live scope is the CubeCL/Metal lane (the split rotation set is still the Metal baseline — claimed by `zed-amo-i1125-rung` for exactly that).

The launch census that refutes it also names what we actually LACK (per GDN decode layer, the scored F32/hd128 posture):

| ours (launches) | theirs after b10826 |
|---|---|
| gdn_input_fused (1) + qkv gemv (1) + z gemv (1) | fused norm+FWHT+q8_1 (1) + gemvs |
| conv1d (1) | conv-chain fused (#398, default-on) |
| beta_decay (1 — grid 1×1, n_head threads: pure launch waste) | ↳ absorbed |
| expand_and_l2_normalize (1) | ↳ absorbed |
| recurrence_fused_hd128 (1) | their recurrence |
| gdn_out_fused (1) + out gemv (1) | fused + gemv |

#398 ("GDN decode conv-chain fusion", `GGML_CUDA_GDN_FUSION=0` disables — NVIDIA≥Turing, FIRES at sm_89, their +7.9% 9B decode H200) rode the scored b10822 build FIRST SCORED at Bench 1004. It is the named decode-shape delta we do not carry.

## The ask

`recurrence_f32_fused_expand_hd{64,128,256}`: ONE kernel replacing the three launches `beta_decay` + `expand_and_l2_normalize` + `recurrence_fused` on the decode F32 arm. Per (head, row) block: compute beta/decay inline (the beta_decay kernel's expressions verbatim — sigmoid, the ±20-branch softplus), read COMPACT qkv `[Q(n_k·hd) | K(n_k·hd) | V(n_v·hd)]`, L2-normalize Q/K inline (src head = head % n_k; the sequential c-sum + the sq_sum==0 guard verbatim), V verbatim — then run the Issue-742 fused body UNCHANGED (shared `__device__` body; identical values → identical FMAs).

- **Bit-identity by construction**: no fast-math (nvrtc default, sm_89), the f32 store→load round-trip the incumbent pays is exact, every per-value op order preserved. Gated empirically: kernel-level bit-exact test (chain vs fused, random + zero-head edges) + the real-model argmax-walk FNV `98412dc406c4ea19` + logits pin `63496f6688c8f742` EXACT.
- **G2**: decode e2e A/B on the real 27B at the manifest posture (one binary, env flips the dispatch — `RIIR_GDN_CORE_FUSED=0` incumbent / `=1` armed). Expected ~28 launches/token saved (2/GDN layer) ≈ +0.5-1.2% at the ~3 µs in-graph dispatch floor.
- **G3**: prefill neutrality by construction — the prefill path rides the `_rows` kernels; untouched.
- **G4**: launcher-only additions, zero allocations; the demote hatch restores every existing launcher.
- NOT in scope: the Half-state arm (opt-in, not the scored posture), the training forward, the attention-layer chain (split_qg/q-norm/k-norm/rope/append — a separate rung if the census warrants it), conv1d (true data dependency: in-place conv must complete before any qkv consumer — same-kernel fusion would need a grid barrier; refused).

## Non-goals / scope guards

- NOT the Metal lane (1125's live scope, amo's claim).
- NOT the prefill lane (sibling Issue 1124 — note: their WIP currently sits uncommitted in this worktree; this issue's edits are disjoint: `cudarc_kernels/deltanet.rs` + the forward dispatch only).
- NOT the lossy state-half successor (owner-gated, Plan 611/Bench 954).

**Refs:** riir-ai Bench 1004 (the 0.9730× decode RED); prism #398 `97ea5aeee1` (b10822, scored first at 1004); riir-infer Issue 742 (the fused recurrence — the body being extended); **riir-infer Bench 898 (the A/B verdict — this issue's record)**; Issue 1125 (the Metal-lane twin); riir-refine `.distill/001` (the watch handoff row).
