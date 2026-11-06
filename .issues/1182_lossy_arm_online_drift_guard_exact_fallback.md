# Issue 1182 — online drift guard with exact-path fallback for lossy serving arms (the XERJ int8-guard pattern)

**Status:** open — adoption distill from XERJ Query-1 (`../riir-reflex/.research/016_XERJ_Query1_Pointer_Program_Serving.md` §3 row 3); serving-posture poc task, nothing wired; opt-in until a GOAT per the flag discipline

## Why

XERJ Query-1 serves int8 with a calibrated guard: when a computation drifts past a threshold the query falls back to the exact path; the int8 configuration was accepted only on 400/400 answer agreement with FP16 (blog §05, fetched 2026-11-06). Our lossy serving arms are gated **offline and at load time only**: per-family retention gates at promotion, league FNV/argmax tripwires (bench-side), Q8 sidecar BLAKE3 (load-time integrity, not numerics). Grepped this session: no online per-query drift detector or exact-path fallback ships (`riir-infer/**/*.rs`; the only "fallback" hits are an unrelated staging-chain comment and a temperature-table fallback).

The gap class: a lossy arm that passes its offline retention gate can still misbehave on an unforeseen input family at runtime — silently, until the next offline re-measure. A bounded online guard — sample a fraction of queries through the exact arm, sigmoid-threshold the divergence, demote the breached family at runtime with a counted fallback line — is the serving-lane counterpart of the retention gate.

**In-family cousin (the genus already ships):** speculative decoding verifies a lossy draft against the exact path and falls back on disagreement (`katgpt-speculative`, katgpt-rs) — but it verifies EVERY step (lossless by construction). The guard here samples a bounded fraction and demotes the breached family (lossy retained, overhead bounded) — different coverage/cost point on the same verify-against-exact spectrum. Published parallels: spec-decode verification and shadow/canary cross-checking in serving practice. (A TensorRT-style builder-time precision-constraint citation was considered and dropped: that is a per-layer BUILD choice, not an online per-query guard.)

## Tasks

- [ ] **T1. Pick ONE lossy arm for the PoC — the T0 decision point, recorded HERE.** Candidate arms, env switches verified this session: (a) the qwen38 dense Metal arm — `crates/riir-infer-gpu/src/qwen38_dense_cubecl.rs` L746–752 (`RIIR_QWEN38_DENSE_MM2D`, demote-only `=0` on macOS+metal_tensor_gemm) + `crates/riir-infer-gpu/src/gemm_dense_f16_metal_zero_copy.rs` L225–240 (`RIIR_QWEN38_DENSE_MM2D_RELAXED`, demote-only `=0`, the promoted 944 posture's relaxed precision); or (b) the dist-prefill tensor-attention arm — `crates/riir-infer-dist/src/stage_gpu.rs` L84–90 (`RIIR_DIST_PREFILL_ATTN_TENSOR=0` demotes; prefill-only by construction — decode never routes through the selector). Pick the arm with the widest input distribution in the league harness; record the pick + reason before any code.
- [ ] **T2. PoC bench** — inject synthetic drift into the arm (the tripwire posture); assert the guard fires on injected drift and never on the clean arm; report fallback rate + the tail-latency cost of the sampled exact-path cross-check.
- [ ] **T3. Threshold law** — sigmoid on the sampled divergence (never softmax), constant pinned; env posture follows the env-switch law (promote-only opt-in until a GOAT, then demote-only).
- [ ] **T4. Scope law** — this is a SERVING-lane device. The 20 Hz game hot path trades tail latency for a per-query cross-check: game-side consumers keep boot-time demote; record the caveat in the module docs and never wire the guard into a game tick path.
- [ ] **T5. GOAT gate** — guard armed vs unarmed on the league cells: retention unchanged, fallback counter = 0 on clean, fires on injected drift, tail p99 within a pinned bar. Pass → promote per the flag discipline; fail → demote loud.

## Provenance

- Source: XERJ blog §05 + §06 (<https://xerj.org/blog/xerj-query-ai-model>, fetched 2026-11-06); distill note `../riir-reflex/.research/016_XERJ_Query1_Pointer_Program_Serving.md`
- Published prior art (adoption, not novelty): speculative-decode verification (the in-family genus, `katgpt-speculative`) and shadow/canary cross-checking in serving practice; mechanism-level diff to the genus recorded in the Why section
- Verified in-tree this session: no online drift-guard / exact-fallback hits (grep over `riir-infer/**/*.rs`); Q8 sidecar BLAKE3 is load-time only (AGENTS.md); env switches for both candidate arms pinned at file+line in T1. Citation-gate note: riir-ai carries its own `.issues/1182` — always cite this issue as "riir-infer issue 1182", never a bare number across repos
