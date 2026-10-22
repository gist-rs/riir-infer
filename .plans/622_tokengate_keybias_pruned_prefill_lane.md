# Plan 622: TokenGate Substrate Lane — Per-Key-Bias Attention + Exact Pruned Prefill + Deterministic Gate Overlay

**Status:** Active — Phase 0 (not started); SECONDARY/PRIMARY ordering per Research 013 §4: this is the PRIMARY lane (runs in the serving hot path, no training required); riir-train Plan 469 rides this plan's Phase 1 seam.

**Date:** 2026-10-22
**Research:** [riir-infer/.research/013_TokenGate_Trainable_Token_Pruning_Prefill.md](../.research/013_TokenGate_Trainable_Token_Pruning_Prefill.md)
**Source paper:** [arXiv:2610.26754](https://arxiv.org/abs/2610.26754) — TokenGate: Adapting LLMs for Fast and Memory-Efficient Prefill with Learned Token Pruning (Stanford/Berkeley, 2026)
**Target:** `riir-infer/src/transformer/` + `riir-infer/src/deltanet/` + `crates/riir-infer-gpu/src/cudarc_kernels/` (+ CPU reference in `riir-infer/src/`); Cargo feature `tokengate` (opt-in, in the same commit as its `required-features` rows for any gated test/bench targets; **never default** — lossy-surface law) with env `RIIR_TOKENGATE` as the runtime switch INSIDE the feature (feature-off = code compiles away = the G3 off-arm; env-off inside the feature = runtime demote).
**Upstream provenance:** upstream repo `token-gate-llms/TokenGate` @ `6e51aac5be1bd84bfd747f43f3f67ed595072bc2` (Apache-2.0); forward-only kernel port — no backward code is ported (training side is riir-train 469)

---

## Goal

An opt-in prompt-skimming lane on the qwen35-hybrid serving path: at layer L a gate scores tokens, top-K are kept, layers > L process only kept tokens with per-key-bias attention. Three pieces: (1) the per-key × per-head additive attention-logit-bias kernel capability (absent from every workspace kernel today; also enables ALiBi/sinks/DASH-class selection), (2) the exact pruned-prefill serving quartet (pad-bias buckets, decode-time constant fold, side-stream KV gather, graph-per-length), (3) a deterministically-constructed gate overlay (seq-RMSNorm scores + qk-norm-derived magnitudes + `1 + γ(r−1)` rank cooling, γ=1) — zero training, exact no-op at K=full, honest quality band (≥ random-K floor; FastKV-class target with a measured FastKV arm; trained-gate accuracy NOT claimed). GOAT gate per Research 013 §5: byte-identity at the inert posture, ≤1.10× biased-call cost, feature-off byte-identity, zero-alloc gate path, four-arm acc-vs-K tables. **Verdict status: this plan's GOAT is a DISPOSITION until T4.1 measures the break-even** — no workspace consumer serves >4k contexts today, so the headline win is unproven at league postures.

**Bonsai/ternary-first (owner pivot rule):** the per-key-bias seam (Phase 1) lands in the SHARED attention kernels (`prefill_cuda_attention_fa.rs` / `cudarc_kernels/attention.rs` / CPU `attention.rs`) that serve BOTH the qwen35-hybrid full-attention layers AND the Bonsai/ternary league model's attention layers — the bias is mixer-agnostic on attention layers, and the gate overlay reads any layer-L mixer output including ternary GDN. The acc-vs-K harness and the T4.1 readout run on the **ternary/Bonsai league posture FIRST**, qwen35-hybrid second.

## Phase 1 — Per-key-bias attention seam (CORE)

### Tasks

- [ ] **T1.0** Kernel-inventory precondition (verdict round 2): enumerate EVERY attention kernel the Bonsai/ternary forward actually calls (candidates: `cudarc_kernels/attention.rs` — confirmed serving Bonsai-27B, head_dim 256; `prefill_cuda_attention_fa.rs`; the CubeCL prefill arm `qwen_attention_prefill_m32_kvf16_cubecl.rs`; `attention_causal_fused_cubecl.rs`; `attention_q8kv_cubecl.rs`) and decide per kernel: bias seam lands here, or NAME it excluded with the reason. Gates T1.1–T1.3 — a seam in one kernel only silently misses the posture that uses another.
- [ ] **T1.1** CPU reference: add `key_bias: Option<(&[f32], Layout)>` (per-key × per-head, additive post-scale pre-softmax) to the attention forward in `src/transformer/attention.rs` and the qwen35 gated-attention path in `src/deltanet/forward.rs` (`forward_attention_layer*`). `None` = unchanged code path (dispatch identity, not arithmetic identity).
- [ ] **T1.2** CUDA: extend `crates/riir-infer-gpu/src/cudarc_kernels/attention.rs` (+ the FA prefill arm `prefill_cuda_attention_fa.rs`) with a per-key bias input; forward only. CubeCL arm only if the CUDA lane is not the serving path for the target posture.
- [ ] **T1.3** G1: byte-identity harness — `None` and all-zero-bias vs incumbent across league shapes (reuse the multi-window walk family from the 948-class gates). Bucketed-vs-unpadded byte-identity test added in Phase 2 rides the same harness.
- [ ] **T1.4** G2: biased-call latency ratio ≤ 1.10× at matched shapes (`--release`, box provenance per `bench_preflight.sh`); G4: launch path alloc-free.
- [ ] **T1.5** Cargo feature `tokengate` gates the whole lane (compile-away off-arm); env `RIIR_TOKENGATE` demotes at runtime inside the feature; every gated `[[test]]`/`[[bench]]` target gets its `required-features` row in the same commit.
- [-] **T1.6** Metal bias arm (MSL kernel + laya-lane discipline): DEFERRED — the CPU reference is the G1 oracle and runs on the M3 dev box first; the league serving posture is 4090 CUDA (no M3 serving consumer for this lane today); the Metal arm lands when the CPU+CUDA seam is G1-proven and an M3 posture actually needs it. The four-arm acc-vs-K harness runs on CPU reference arms on the M3 in the meantime.

## Phase 2 — Exact pruned-prefill serving lane

### Tasks

- [ ] **T2.1** Top-K selection + gather per Research 013 placement rules: score at layer-L mixer output (any mixer type), prune only document tokens (non-prunable scaffold always kept), layers ≤ L keep full KV, layers > L process kept tokens only; GDN state split — layers ≤ L keep full-prompt state, layers > L build state from kept tokens.
- [ ] **T2.2** Pad-bias buckets: kept sequence padded to a multiple of 64, pad keys get −1e9 bias, pad rows discarded, caches cut back; test = bucketed-vs-unpadded byte-identity at every bucket width (K=full included).
- [ ] **T2.3** Decode-time constant fold: key ratios frozen at prefill ⇒ gate folds once into a per-layer additive mask (decode-appended keys stay 0); skip the q-multiply when s_eff ≡ 1 (dispatch identity).
- [ ] **T2.4** Optional: extend `forward_tokens_graph_walk` with a graph-per-padded-length capture of the layers- > L loop; static-address buffers for bias/scale/ratios (attr swap = recompile, measured upstream); side-stream KV gather if the serial gather shows in the G2 profile. Capture-length cutoff above which shapes are unique (do not capture).
- [ ] **T2.5** Layer-choice measurement: acc-vs-K at L ∈ {0, 1, mid} on one suite — pin the constant for OUR models (upstream used L=1 dense / L=3 hybrid; measure, don't copy).

## Phase 3 — Deterministically-constructed gate overlay

### Tasks

- [ ] **T3.1** Pure fns (local to riir-infer; promote to katgpt-core only on a second consumer): sequence-RMSNorm scorer normalization; rank-ratio cooling `s_eff = 1 + γ(r−1)`, `r = log(m_i)/log(max(2, p_i))`, γ=1 fixed (exact no-op at K=full for any γ — assert as identity test).
- [ ] **T3.2** QK-norm-derived magnitude table at load: `s_{l,h} = ‖g_q^h ∘ g_k^h‖₂/√d` from pretrained qk-norm gains (per head; upstream records learned gate_scale converging to the same range as the derived table — no training).
- [ ] **T3.3** Scorer family v0 with arms: (a) attention-mass received at layer L (phase-1 stat tap — the pass runs anyway), (b) hidden-state norm at L, (c) all-ones. Default arm = (a); the others are ablation rows.
- [ ] **T3.4** Four-arm acc-vs-K harness: no-prune ceiling / random-K floor / derived scorer / **FastKV-class arm (reported, non-gating)** — the paper-described FastKV selector (token scored by attention received from a window of the last queries, max-pooled, top-K) implemented on the SAME seam at the same layer L, so "FastKV-class" is a measured column, not an adjective. Log-uniform K sweep, on the NIAH-style harness + long-ctx ppl (reuse `needle_eviction_gate`, `row_logit_floor_ppl` instruments). GOAT = derived ≥ random-K on every cell; the FastKV column and the Pareto-vs-ceiling reported honestly. Selection-only diagnostic mode: clear bias AND scale so ablations measure selection alone (upstream measured the drift when only one is cleared).
- [ ] **T3.5** Random-K robustness census (co-adaptation measurement row): league ternary vs dense reference under identical random-K — a benchmark record, never a promotion claim (modelless law forbids base-weight mutation; quantization-aware training is the co-adaptation proxy being measured).

## Phase 4 — Verdict + league readout

### Tasks

- [ ] **T4.1** TTFT + peak-memory vs K on the league posture (pp benches + long-length rows; **Bonsai/ternary first** per the pivot rule) with box provenance; compare the quartet's win at random-K quality (scorer-agnostic) as the headline systems number. **Mandatory break-even row:** report the (K, context-length) cells where the quartet's overhead (bucket padding, graph capture, KV gather) exceeds its savings — expected at short lengths (league pp2048/pp4096 sit at the low end of the paper's 1.4–11.3× range); below break-even the lane must no-op via the dispatch identity, never slow the default path.
- [ ] **T4.2** Verdict write-up in `.benchmarks/`: the lane stays opt-in forever (lossy-surface law); promotion to default REFUSED by policy — the only re-open is the owner re-weighing the lossy law with the measured acc-vs-K table in hand (the Issue-971 absorption-counter posture).
- [ ] **T4.3** If Plan 469 produces a trained gate artifact: wire it as a frozen BLAKE3 sidecar in the `qv_lora` reader pattern behind the same feature+env gate; re-run the Phase 3 harness with a fifth arm (trained gate) — the trained-vs-modelless comparison the whole lane exists to adjudicate.

## Boundary notes

- Zero new external deps; no riir-* deps added (riir-infer stays upstream-clean).
- `attention_si.rs` keeps its key-independent affine — the per-key axis lands beside it, not inside it.
- Batched-recurrent right-padding audit is a separate pre-requisite: Issue 1107 (pads-to-front state no-op). If T2.1's gather batch rows, the Issue 1107 law applies to this plan too.
- Feature-gate hygiene: every new `#[cfg]`/env-gated target gets its `required-features`/runner row in the same commit; run `cargo clippy` (not check) on the gated files; `CARGO_TARGET_DIR=/tmp/p622` if the workspace lock is held.
