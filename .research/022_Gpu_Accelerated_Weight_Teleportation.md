# Research 022: GPU-accelerated Weight Teleportation (humans& — delta weight sync)

> **Source:** "GPU-accelerated Weight Teleportation" — Ziang Li & friends, humans& ai blog, 2026-11-02. https://humansand.ai/blog/gpu-accelerated-weight-teleportation (prior art it builds on: Miles v0.1, arXiv:2609.08368 §4.3 Disk-Delta Updates; open-sourced in Miles `github.com/radixark/miles` + SGLang `/update_weights_from_gpu_delta`)
> **Date:** 2026-11-03
> **Status:** Done — Gain (two-track; files filed, see §5)
> **Related Research:** katgpt-rs 439 (SAME TEAM — their NVFP4 RL blog, 2026-07: the bit-exact trainer-sampler numerics contract; this note is the TRANSPORT half of the same trainer-sampler problem) · riir-train Research 415 / arXiv:2608.03573 ("SFT Conflicts, RL Coexists" — RL deltas sparse; internal corroboration of the paper's premise) · riir-infer 004 (DQ disaggregated quantization), 015 (multi-host inference)
> **Related Plans:** none (issues filed: riir-infer 1169, riir-train 667)
> **Classification:** Public

---

## TL;DR

The blog makes RL trainer→sampler weight sync faster than intra-cluster RDMA by shipping **sparse bitwise-XOR delta masks** (lossless, dtype-agnostic) through a **double-compression + heterogeneous CPU/GPU decompression pipeline** (outer Zstd on CPU while serving continues; inner Snappy on Blackwell's fixed-function Decompression Engine), with a **double-buffered DE↔SM overlap** that streams layers so the sampler pause is 219.6 ms and staging never exceeds two layers. Every component is training-free systems engineering — the mechanism is published and open-source, so the verdict is **consume, not novelty**: a `weight_delta` transport primitive here (issue 1169) and a delta-checkpoint recipe in riir-train (issue 667). Blackwell-specific parts (DE, pinned-RAM→HBM DE input, DE↔SM ASIC overlap) die on our Metal/Ada fleet; the algebra, codec pairing, metadata plan, and version chains survive intact.

**Distilled for riir-infer (modelless, inference-time):**
`mask = old ⊕ new` over a contiguous byte view; `apply(old, mask) = new`; `apply∘apply = identity` (XOR involution — no FP, no rounding, works identically on F16/Q8/PQ2_0 packed layouts, and packed low-precision formats shrink the mask per weight). Sparse masks (RL/fine-tune deltas flip few bytes) compress losslessly; their measured pairing: Snappy inner + Zstd outer → 2.65 GB from 19.78 GB first-pass (13.4%) on a 445 GB NVFP4 checkpoint, 0.60% of model size. Receiver applies in place against a CPU-prepared layout plan (fused-kernel weight reordering handled by remapping the mask with the same permutation — pointwise alignment). Versioned delta chain with auxiliary bases (eval base, restart base); accumulated startup deltas use LZ4+Zstd once (smaller inner payload).

---

## 1. Paper Core Findings

| # | Finding | Numbers (their 445 GB NVFP4 GLM-5.3 proxy, 0.085% mutated/step) |
|---|---|---|
| 1 | Delta algebra: `old ⊕ new = mask`, `old ⊕ mask = new`; bitwise, dtype-agnostic, lossless | mask payload 0.60% of model size (Snappy+Zstd) |
| 2 | Double compression beats single GPU codec | Snappy 19.78 GB → +Zstd **2.65 GB** (13.4%); LZ4 5.36 GB → +Zstd 4.03 GB (75.2%) — Snappy's simpler copy commands recompress better under Zstd despite LZ4 winning solo |
| 3 | Heterogeneous decompression: outer layer on CPU, inner on hardware | rollout decoding continues during CPU Zstd + metadata prep (2.355 s overlap); DE reads pinned RAM → HBM directly |
| 4 | DE↔SM double-buffered per-layer streaming | sampler pause **219.6 ms**; staging = 2 layer buffers (no KV eviction); e2e < 4 s, beats intra-cluster RDMA full-weight sync |
| 5 | Metadata application plan (CPU-prepared) solves fused-layout misalignment | gate/up pairing in groups of 64 — mask remapped by the same permutation; one Triton launch for the whole model |
| 6 | Multi-version delta bases: steady-state, eval base, restart base; aux deltas overlap the hot path | startup accumulated deltas ride LZ4+Zstd once |
| 7 | Serving/deployment payoff | cross-datacenter deltas instead of full checkpoints; `release_state` opt-in; positioned as alternative to LoRA adapters for serving swaps (<0.23 s pause) |

## 2. Distillation

### 2.1 Naming disambiguation (grep-hazard)

This paper's "weight teleportation" = **network transport of weight deltas** (RL systems). It is NOT the model-merging literature's "teleportation" (permutation sync aligning models into one loss basin — Tatro et al. 2020; Ainsworth et al., *Git Re-Basin*). Future greps for `teleport` must not cross the streams; cite this note as "delta weight sync (transport)".

### 2.2 Path 0 inventory (advocate-merged; discards carry reasons)

| Component | Modelless-validable? | Analog ships? | Verdict |
|---|---|---|---|
| XOR delta algebra + apply | Yes — closed-form group op, provable identities | **No** (whole-artifact swaps only: `freeze.rs` MerkleFrozenEnvelope, `induced_cwm/hot_swap.rs` InducedCwmSlot, `arsenal_ops.rs` LaneSlot; LoraPair overlays mutate, not transport) | **File** → issue 1169 |
| Sparsity exploitation + two-regime fallback | Yes — combinatorial popcount | Measurement cousin ships: riir-train `delta_audit.rs` computes `sparsity_frac` (+ `min_sparsity_frac()`) over exported ΔW per Research 415 — measurement exists, **transport does not** (signal-diff: audit gates merges, never moves bytes) | **File** (Phase 0 kill branch in 1169 + riir-train 667) |
| Double compression + pair selection | Yes — fixed codecs; pair choice is a one-time calibration on our own mask distribution | No compression codec dep at riir-infer root (BOUNDARY.md allowlist pins the list) | **File as opt-in** — zero-new-dep RLE/varint core first; zstd/lz4_flex need BOUNDARY.md rows + owner act |
| Heterogeneous CPU-outer decompression | Yes — OS threads + staging; on M-series unified memory staging is already device-visible (stronger on Metal than the paper's pinned-RAM case) | No | **File** (receiver design note in 1169 Phase 3) |
| Blackwell DE + pinned-RAM→HBM DE input | Hardware-bound — DE is Blackwell-only; nvCOMP is CUDA-only, closed | n/a | **Discard** — dies on Metal/Ada; revisit only if a Blackwell box joins the fleet |
| DE↔SM double-buffer overlap | Pattern survives, silicon-bound specifics die | Generalizes: copy-engine↔SM on Ada, CPU-decode↔SM on Metal; apply kernel = wgpu compute + cudarc twin (sibling-transcription law) | **Record as pattern** in 1169; no Blackwell-specific code |
| CPU-prepared layout plan (mask remap under fused-kernel permutation) | Yes — deterministic permutation conjugation | Partially latent: our serving lanes re-pack whole vessels rather than patch packed weights in place; fused layouts (GLU fold, ktile64) live in riir-infer-gpu | **Defer** — no in-place packed-weight patch consumer today; revisit when one exists |
| Versioned delta chain + aux bases | Yes — monotone counter + lineage commitments | **Strong cousin, wrong payload**: `freeze.rs` v2 `parent_commitment`/`freeze_from_parent`/`verify_descent` (Issue 609) is already the commitment-chain DAG — it certifies whole-artifact descent; delta rows are a payload class under the same shape. Apply-gate vocabulary exists: `arsenal_ops.rs` `check_epoch_tag` (`EpochApply::{Advance, Idempotent}`, `SwapRefusal::{Downgrade, Fork}`) | **Fusion row** (§4); no separate filing — reuse the shapes when 1169 lands |
| Distributed compression ownership remap | Yes but scale-gated | n/a — collapses to `LayerRange` partition at our scale | **Discard** (multi-node compression coordination has no consumer at 1–2 boxes) |
| Checkpoint-as-handoff (disk round trip) | Anti-pattern lesson | — | **Record**: disk never in the delta hot path; their 6.041 s receiver figure is the pinned stranger baseline |

### 2.3 Hardware portability (technique stripped of substrate)

- **Dies:** Decompression Engine (Blackwell fixed-function; absent on Ada sm_89 and Apple GPUs), nvCOMP (CUDA-only), Triton-as-shipped.
- **Survives, portable:** CPU Zstd outer decode overlapping serving (zstd-rs; M-series unified memory makes staging GPU-visible zero-copy); bitwise-XOR apply as a wgpu compute kernel (Metal + CUDA via cudarc twin); two-slot per-layer streaming (no full-model output buffer, no KV eviction); BLAKE3-pinned base + mask integrity.
- **Their NVFP4 note helps us:** packed quant formats (our Q8, PQ2_0 nibble packing) shrink per-weight mask bytes further — the sparsity win compounds on ternary/PQ layouts.

### 2.4 Sparsity is a regime, not a law (pre-registered kill)

The premise (deltas flip few bytes) is corroborated internally — Research 415 / arXiv:2608.03573 found RL deltas sparse and near-orthogonal, and `delta_audit.rs` already measures `sparsity_frac` — but it is not guaranteed for OUR sources: LoRA-merged deltas compete with the adapter itself (at rank r an adapter is ~r·(d_in+d_out) per touched matrix — the honest weight_delta consumer there is full-param updates only); wholesale PQ re-quantization or full re-freeze is dense; and bf16/optimizer-state rounding flips low mantissa bits almost everywhere even for small VALUE updates (byte-space flipped-byte fraction ≠ value-space `sparsity_frac` — the gate measures byte-space). **Phase 0 of both filings is the measurement with a kill branch**: if compressed-mask payload ≥ 50% of **zstd(full artifact) at a pinned zstd level on the same bytes** (the real competitor is the compressed whole artifact, never raw bytes), close as NEGATIVE — no protocol gets built on a dense regime.

## 3. Verdict

**Tier: Gain** (two-track).

- **Track a/b (modelless transport):** mechanism is published + open-sourced (Miles arXiv:2609.08368, SGLang) — zero mechanism novelty is claimed. The actionable gain is measurable on our lanes: payload/transfer reduction for checkpoint and serving-weight updates, with G1 byte-identity trivially provable (XOR involution) and a pre-registered sparsity kill branch. → `riir-infer/.issues/1169`.
- **Track c (training recipe):** delta checkpoint artifacts + sparsity-keyed publish + crash-resume chain improve pipelines that exist TODAY at our scale (`checkpoint_safetensors.rs` + `atomic_replace.rs` save lane; the M3↔4090 `data_backup.sh` ~10 GB pull-all is the real transfer pain). The DE/RDMA/sub-200 ms halves are scale-gated (no Blackwell, no multi-node RL) and are explicitly out of scope. → `riir-train/.issues/667`.

**MOAT gate (riir-infer):** PASS — weights/loaders/transport is this repo's declared §Owns; the primitive is upstream-clean (BLAKE3 already an allowed dep; any codec dep enters only via a new BOUNDARY.md allowlist row). **MOAT gate (riir-train):** PASS as recipe adoption, not training-method novelty. **Not Super-GOAT:** Q1 fails on published prior art (the gate's precondition — the pinned integration claim is "delta-checkpoint weight sync for this workspace's weight lanes, consuming riir-train sparse deltas, distinguished from whole-artifact incumbents (MerkleFrozenEnvelope snapshot swap, LaneSlot epoch swap, LoraPair overlay) by payload = lossless sparse delta against a BLAKE3-pinned base" — and that claim's mechanism is prior art by construction).

## 4. Fusion (closest cousins → what the combination buys)

1. **× `riir-neuron-db` freeze lineage** (`parent_commitment`/`verify_descent`, Issue 609): a delta row IS a parent-child envelope whose payload is a compressed mask — the commitment DAG already exists; the delta payload slots under it. Buying: restart-from-base byte-identity audits for free (`verify_descent` over a delta chain). **Boundary:** riir-infer takes NO dependency on riir-neuron-db (the zero-riir-deps upstream law is this repo's boundary); it copies the SHAPE (a parent commitment over a BLAKE3 base digest) locally. A real dep edge would need a BOUNDARY.md row + owner act and would break upstream-cleanliness — shape-copy is the only path.
2. **× `riir-instinct` epoch gate** (`check_epoch_tag`, `EpochApply`/`SwapRefusal`): delta rows as epoch artifacts inherit Advance/Idempotent/Downgrade/Fork refusals — monotonic apply discipline comes free. **Boundary:** same law — the refusal vocabulary is re-declared locally in riir-infer (issue 1169 T1.3), never a dep edge. (Serving-tier adoption of delta publishes is an owner-gated future posture — the private serving rung may adopt; nothing filed there from this public note.)
3. **× `disaggregated.rs` two-copy container** (`DisaggregatedTernaryWeights::from_pair`): the receiver half already exists — apply delta to the idle copy, flip. No torn reads by construction.
4. **× `riir-infer-dist` PUSH lane** (Plan 623 multi-host): delta frames are a sibling payload class to `ActivationFrame` — same PUSH shape, one one-way hop per boundary.

## 5. Routing & filings

| File | What |
|---|---|
| `riir-infer/.research/022_Gpu_Accelerated_Weight_Teleportation.md` | this note |
| `riir-infer/.issues/1169_weight_delta_transport_poc.md` | track a/b PoC: weight_delta primitive + Phase 0 measurement + GOAT gates |
| `riir-train/.issues/667_delta_checkpoint_sync_recipe.md` | track c recipe: delta checkpoints + sparsity publish gate + crash-resume chain |
| `katgpt-rs/.research/439_NVFP4_RL_4over6_Adaptive_Block_Scaling_Pass.md` | sibling back-ref line (same-team chain: 439 = numerics half, 022 = transport half) |

## 6. GOAT gates (pre-registered, both filings)

- **G1 correctness:** `BLAKE3(thaw(old ⊕ mask)) == BLAKE3(new)` on F16/Q8/PQ2_0 fixtures including alignment/padding boundaries; `apply∘apply ≡ no-op`; **serializer-determinism sub-gate** — re-serializing an UNCHANGED checkpoint through the full loader/saver path must yield an all-zero mask (a non-bit-identical serializer can make the regime dense silently).
- **G2 payload:** compressed-mask payload < full-copy bytes AND < 50% of **zstd(full artifact) at a pinned zstd level, on the same bytes** on ≥2 real sparse sources (else the Phase 0 kill branch fires).
- **G3 no-regression:** full-copy path stays default and byte-identical when the feature is off; dense source auto-falls back to full copy.
- **G4 alloc:** apply path alloc-free over pre-allocated scratch.
- **Latency gate (receiver integration):** e2e delta update < 10% of full-reload wall-clock on a sparse source, GPU-exclusivity provenance recorded per the bench rules; ratios transfer across boxes, absolute numbers do not (their 445 GB/B300 numbers are a different box state).
