# Bench 934 — Issue 1146: ladder 4(c) matmul2d flash-attention A/B + the relaxed-OFF pp32k settle

**Status:** RECORD — 4(c) **PROMOTED dist-scoped** (wins at every length incl. 1.95× at pp32768, KL ≤ 6.7e-7, G1 identical); the `401af78` relaxed-OFF pp32k "sign flip" claim **PROVEN** (tensor 1.22× faster, guard-clean). Stacked posture now reads **774 tok/s at pp32768 = llama.cpp parity** (was 0.45×); pp4096 0.93×; pp512 0.76×. The issue's 6–8× gap is closed to 0.76–1.0×.
**Date:** 2026-11-02 17:16–17:46 +07
**Session:** zed-amo-1146-finish, 1793612565 (lease `3f4f828`)
**Issue:** `.issues/1146_prefill_throughput_gap_1box.md`
**Box:** bmo (Mac17,15, M5 Ultra, 30 cores, 96 GiB), AC, load 1.11 at start; runner pid 51651
**Model:** `Ternary-Bonsai-2-27B-PQ2_0.gguf` (league pin, 7,206,168,928 B)
**Build:** riir-infer `3f4f828` (clean detached worktree `/tmp/i1146f-wt` — the shared checkout carried a sibling's live WIP on `qwen_attention_prefill_sparse_cubecl.rs`, excluded by construction; katgpt-rs `6c7656193`), release, `-p riir-infer-dist --bin dist_stage --features gpu,bonsai2_hadamard`
**Rig:** `scripts/i1146_width_recurve.sh`, one binary both arms (env-only arms), w512, `I1146W_REPS=2`, per-pair GPU-quiet guard — **zero CONTAM across all three phases**

## Phase 1 — ladder 4(c): `RIIR_PREFILL_ATTN_METAL_TENSOR=1` (kernel `672b29c`)

Arm A = the promoted dist posture (tensor zero-copy + H4 + dist-relaxed, CubeCL attention);
Arm B = Arm A + the matmul2d flash-attention arm (env-only, highest precedence for routable
shapes). **Engagement proof:** all 10 arm-b cell logs carry `[Issue 1146] prefill attention =
matmul2d tensor arm`; zero arm-a logs do. **G1 CROSS-WIDTH PASS — identical 16-token streams
across arms** (the tolerance arm agreed end-to-end at pp1024).

| cell | a (promoted) | b (+attn tensor) | b vs a |
|---|---:|---:|---:|
| pp512 w512 r1 / r2 | 0.654 / 0.647 s | 0.624 / 0.620 s | **1.046×** |
| pp4096 w512 r1 / r2 | 6.233 / 6.247 s | 4.209 / 4.192 s | **1.483×** |
| pp32768 w512 r1 / r2 | 82.552 / 83.440 s | 42.302 / 42.315 s | **1.953×** |

The long-context bar (4(a)'s rejection lesson: "must WIN AT LONG CONTEXT, not just pp4096")
is met with margin everywhere — the gain GROWS with length, as the attention-share curve
(Bench 925: 37% at pp4096) predicts: the arm removes the matrix-unit-less attention cost and
its win compounds as KV history grows. This run's a-arm reproduces the shipped posture's
shape (Bench 923: 0.72 s pp512 / 7.13 s pp4096 → today 0.65 / 6.24; box and `8da2f81..3f4f828`
delta both read slightly faster — coherent, not order-flipped).

**The day's ladder (dist prefill, w512, bmo):**

| posture | pp512 | pp4096 | pp32768 | vs llama.cpp (pp512/pp4096/pp32k) |
|---|---:|---:|---:|---|
| morning 10-28: w128 simdgroup (Bench 911) | 3.04 s | 25.6 s | 96.1 s | 0.13 / 0.16 / 0.12× |
| + width 512 (919) | 2.19 s | 18.4 s | 183.9 s¹ | 0.21 / 0.23 / — |
| + tensor zero-copy (920) | 1.75 s | 15.24 s | ~95 s | 0.26 / 0.27 / 0.45× |
| + H4 (921) + relaxed dist-scoped (923) | 0.72 s | 7.13 s | 93.8–97.5 s | 0.65 / 0.55 / 0.45× |
| **+ attn matmul2d (this)** | **0.62 s** | **4.20 s** | **42.3 s** | **0.76 / 0.93 / 1.00×** |

¹ Bench 919's width-curve simdgroup pp32k row. llama.cpp rows from Bench 911 (pp512 1087.4 tok/s,
pp4096 1046.1, pp32768 772.0). Today's b-arm: 820.5–825.8 / 973.2–977.1 / 774.4–774.6 tok/s —
**parity at pp32768**.

## Phase 2 — per-family KL of the attention arm (`issue1146_relaxed_kl`, `I1146K_AXIS=attn`)

Ref = CubeCL attention chain, cand = matmul2d attention, tensor GEMM ON in both arms,
`RIIR_METAL_TENSOR_RELAXED=1` (the dist serving posture). Vacuous guard: **1024
tensor-attention launches** (hard in-test assert).

| family | n | KL mean | KL max (nats) | top1 | ΔNLL mean | NLL(ref) |
|---|---:|---:|---:|---:|---:|---:|
| code | 16 | 1.970e-8 | 9.550e-8 | 16/16 | −4.013e-5 | 0.316 |
| math | 16 | 1.173e-8 | 1.401e-7 | 16/16 | −1.277e-4 | 0.546 |
| prose | 16 | 1.684e-7 | 6.670e-7 | 16/16 | −1.473e-4 | 2.407 |
| zh | 16 | 1.801e-7 | 6.271e-7 | 16/16 | −1.699e-4 | 2.489 |

Max KL **6.67e-7 nats** — ~150× under the relaxed-class bar (1e-4), the same magnitude as the
lossless tensor GEMM's own accumulation noise (Bench 923's relaxed-GEMM rows: 1e-7/1e-6
class). top-1 16/16 in every family; ΔNLL all ≤ 1.7e-4 with one sign (negative — the cand
reads marginally LOWER NLL everywhere, i.e. pure noise, no family-conditional shift). The
e2e G1 identical streams close the behaviour axis: no observable divergence anywhere on the
deployed path.

## Phase 3 — relaxed-OFF pp32768 settle (the `401af78` claim)

Arm A = tensor OFF + relaxed OFF (simdgroup GEMM incumbent); Arm B = tensor ON, relaxed OFF.
`401af78` (per-buffer merged-interval CONTAINMENT for the zero-init cure) claimed it removed
a tensor-OFF→ON regression at pp32k; its three post-fix pairs were all CONTAM-flagged. These
four pairs are guard-clean (zero CONTAM the whole session):

| cell | a (tensor off) | b (tensor on) | b vs a |
|---|---:|---:|---:|
| pp32768 w512 r1 / r2 | 167.538 / 167.939 s | 137.424 / 137.253 s | **1.22× faster** |

The "sign flip" claim is **PROVEN**: with the containment fix, tensor is decisively faster at
pp32k even before relaxed. The original 3–11%-slower reading was the zero-init refill storm,
exactly as root-caused. (G1 CROSS-WIDTH PASS in this run too — identical streams.)

**Post-promotion functional check (same day, bmo, cold-boot posture):** the promoted binary (sha `56cc52fb…`) with no env reads pp4096 5.264 s; with `RIIR_DIST_PREFILL_ATTN_TENSOR=0` it reads 7.501 s — ratio 1.43×, matching the A/B arm ratio (1.48×); the demote literal restores the incumbent arm (env-switch law). Note: the `[Issue 1146] prefill attention = matmul2d tensor arm` announcement does NOT print in the promoted dist posture — it lives in the ENV-read arm of `prefill_attn_metal_tensor_enabled`, and the dist setter arms the flag silently (same shape as the relaxed setter); engagement is proven by timing, not the log line.

## Verdicts

1. **4(c) PROMOTED, dist-scoped**: `TernaryGpuStage` arms the matmul2d attention selector
   default-on before the forwarder compiles its pipelines; **`RIIR_DIST_PREFILL_ATTN_TENSOR=0`
   demotes** (the exact literal restores the CubeCL chain byte-path). Substrate stays opt-in
   (`RIIR_PREFILL_ATTN_METAL_TENSOR`) — same lossy-surface reasoning as Bench 923: a lossy arm
   must not silently move the riir-ai league's prefill outputs or the bit-exact gates, and the
   641-T5 protected window stays out of scope. A league-side flip is riir-ai's own decision on
   this record. G3 (decode) holds by construction: the selector has ONE call site, in the
   prefill attention ladder (`ternary_deltanet_gpu_forward.rs` `try_prefill_attn_metal_tensor`,
   called from `prefill_attention_layer_batched` only); decode attention never routes through it.
2. **Ladder closed.** With 4(c) promoted, every 1146 leg is measured: width (919), tensor
   GEMM (920), H4 (921), relaxed (923), embed (NULL, stays opt-in), attention (this). The
   remaining ~1.7× (pp4096 0.93×; pp512 0.76×) is kernel-internal (matmul2d tile geometry at
   prefill shapes + the FFN/DeltaNet GEMM residual) — a NEW-kernel follow-up lane, not an
   armed cell of this issue.
3. **The pp32k settle question is closed** — no further pair owed.

## Provenance

- Runner: `bmo:/tmp/i1146f/runner.log` → `i1146_finish/runner.log` (all phases, rc=0).
- Rig sessions: `i1146_finish/attn_ab_session.log`, `i1146_finish/pp32k_settle_session.log`
  (per-cell `rc=0`, zero CONTAM lines); KL raw: `i1146_finish/kl_attn.log`.
- Binary sha256s: `i1146_finish/bin_sha.txt` (dist_stage + kl_test, verified amo↔bmo identical
  pre-run); model sha256 in the same file.
- Box state: `i1146_finish/box_state.txt` (bmo, AC, load 1.11, 96 GiB).
- Width-rig provenance: `i1146_finish/attn_ab_provenance.txt`.
