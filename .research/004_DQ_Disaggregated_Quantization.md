# Research 004: Disaggregated Quantization — Prefill/Decode Phase Specialization (DQ/QADD)

> **Source:** "Disaggregated Quantization: Specializing LLM Prefill and Decode" — [arXiv:2609.26333](https://arxiv.org/abs/2609.26333), Panferov, Kleinegger, Priyadarshi, Blankevoort, Alistarh (NVIDIA & ISTA), 2026-09-22. Code `IST-DASLab/disaggregated-quantization`; llama.cpp fork `IST-DASLab/disaggregated-llama.cpp` (MIT); checkpoints `ISTA-DASLab/Qwen3.8-27B-NVFP4-prefiller` (HF, apache-2.0). No `.raw` clone (paper distills over the wire; repos read via web — pinned by URL, not sha).
> **Date:** 2026-09-29
> **Status:** RECORD — Track (a) modelless/serving = **Gain** (`.issues/026`+`027`+`028` filed here); Track (c) model-based = **Gain** (`../riir-train/.plans/430_qadd_disaggregated_prefiller.md`, pre-registration, SECONDARY by serving-envelope fit); Track (b) self-adaptive = **Pass** (no runtime-latent angle). Not Super-GOAT — this consumes a published method; the landscape prior art is named in §6.
> **Controlling prior record:** `../riir-clippy/.research/217_arxiv_walk11_closeout.md` row 1 graded this paper "B+ league-intel HIGH — lane record; accuracy half → training-track (riir-train routing); no corpus rule". This note is the full-read deepening that walk row routed: it completes the riir-train routing (Plan 430) and extracts the serving/format intel the table row could not carry.
> **Related Research (this repo):** 001–003 unrelated lanes. **katgpt-rs 538** (GDN W4A4 survival mechanism — the `a_log`/`dt_bias` source). **riir-clippy 217** (the walk record above).
> **Related Plans:** riir-train 430 (QADD prefiller, filed this session).
> **Cross-ref:** riir-infer Issues 980 (gate-projection escape set — shipped) + local 026/027/028 (filed this session); `tests/issue879_gdn_quant_certification.rs`.
> **Classification:** Public.

---

## TL;DR

Prefill (compute-bound, T≫1) and decode (memory-bound, batch-1) reward different quantization — different **formats**, and even different **weights**. The paper makes phase-specialization concrete on three rungs: format disaggregation (quantized-compute prefill + weight-only decode — validated zero-training via PTQ up to 2.8T params), full disaggregation (separately trained prefill weights via QADD: the SFT label mask becomes a per-position pathway mask; response-only loss trains both pathways through the prompt-KV gradient path), and ODP (SSD-streamed prefill weights, amortized over prompt length). For this repo the four load-bearing facts are: **(1)** our serving is already format-disaggregated de facto (decode GEMV is weight-only; prefill is compute-keyed) — the paper validates that posture at frontier scale; **(2)** our 2-bit league artifacts (PQ2_0/Q2_0, 2.125 bpw) sit exactly in the gain-concentration band (+2.8 to +7.4 MMLU-Pro at IQ2-class decode); **(3)** our hardware inverts their headline constraint — a resident dual-checkpoint 27B posture fits both the 4090 (24 GB, at the edge) and the M3 (64 GB, trivially), so ODP is the *less* relevant half for us; **(4)** the GDN quant escape policy (gate projections + `A_log` + `dt_bias` frozen) is corroborated from the training side at 27B.

**Distilled for riir-infer (substrate, mostly inference-time):**
- **The most transferable rule is hardware-independent:** decode → weight-only, always. Batch-1 decode is memory-bound; activation quantization never pays there — skipping it is simultaneously an accuracy win *and* a small decode speedup (2–3% in their tables).
- **The phase-sensitivity measurement protocol is an extractable instrument:** damage ratio `R = Δacc(decode-only-quant)/Δacc(prefill-only-quant)` — a directional, falsifiable, zero-training assertion (R>1 on decode-heavy tasks, inverted on prefill-heavy). → Issue 026.
- **LUT grid design is an offline Lloyd-Max problem** under our single-level f16-per-128 scale (adding a tensor-level f32 scale is an explicit solver axis, never an assumption). The cheapest arm is encoder-only: `q2_0` already **decodes** a fourth code state (+2d) that its reference encoder never emits — a signed `d = ±amax/2` with sign-absorption activates it with no format change. → Issue 027 (T0 first, then the non-uniform Q2_0A, which must beat T0).
- **Dual-PTQ disaggregation** (two quantized copies of one checkpoint: compact decode-resident + higher-precision prefill) is the modelless ceiling of their trained result — and *measuring how much of the trained gain survives pure PTQ* is a clean decomposition experiment. → Issue 028.

---

## 1. Paper core findings (compressed)

- **Three schemes (their Fig. 2 ladder):** (a) *format disaggregation* — NVFP4 W4A4 prefill compute + NVFP4A16 weight-only decode; same storage, same prefill cost, decode 2–3% faster. (b) *full disaggregation* — separate prefill/decode weight sets trained jointly by QADD; prefill keeps hardware-native compute, decode keeps compact weights. (c) *ODP* — the prefill checkpoint streams from SSD through two device block buffers; buffer space carved from decode weights (idle during prefill) and restored before generation; compute overtakes SSD loading ≈8K context; <5% overhead above 16K.
- **QADD mechanism (the training novelty):** the SFT label mask (user turn = prefill, assistant turn = decode) selects the computational pathway per position in every quantized linear layer. The KL(teacher‖student) loss supervises response tokens only, yet prefill weights train — through the causal-shift boundary prediction and the prompt keys/values decode attends to. One forward-backward pass; STE fake-quant; FP32 masters; AdamW lr 3e-6 constant.
- **Accuracy results (Qwen3.8-27B, frozen Unsloth GGUF decoders + trained NVFP4 prefiller):** 1-bit decode +32.5 MMLU-Pro / +35.3 MMMU-Pro (more than doubles weight-only); 2-bit +7.4/+6.3; ~0 at 3-bit and slightly negative at Q3-class. Core Qwen3/Gemma3 family results: full disaggregation lifts LUT2 by +10.7 (decode-heavy) / +5.3 (prefill-heavy) on Qwen3.
- **Zero-training validation:** PTQ-only *format* disaggregation (skip activation quant on decode) improved 11/13 model-benchmark combos up to 2.8T params (6 significant at α=0.05, none significantly degraded).
- **Sensitivity asymmetry:** on decode-heavy tasks decode-only quantization is 2–4× more damaging than prefill-only (up to 7× on Gemma3-1B); the ordering reverses on prefill-heavy tasks. This is the paper's measurement protocol contribution.
- **LUT formats:** grids optimized to minimize expected quadratic error over N(0,1) under two-level scaling (one FP32 global + one FP8-E4M3 per 16 elements), constrained to contain 0 and +6, with the per-block scale **absorbing the sign of the block's max-magnitude element** (asymmetric grids): LUT3A16 8 levels, LUT2A16 4 levels, `b + 0.5` bpw.
- **GDN-specific (Appendix A.2):** the narrow gate projections `in_proj_a`/`in_proj_b` are shared and frozen in BF16 — "quantizing them destabilized training" — as are `A_log`, the time-step biases, and the recurrence dynamics; embeddings/lm-head shared+frozen. Qwen3.8-27B = 64 blocks, 48 gated-linear-attention + full attention every fourth.
- **Prefiller interoperability:** learned prefillers partially transfer across nearby decode bitwidths (IQ1_S decoder reaches 65.02 with the IQ1_M prefiller vs 61.54 with its own), but the pairing matters most at the lowest bits (+13.1 over RTN prefill at IQ1_S).
- **Honest caveats (their §5):** batch-one decode only; no highly-batched evaluation; multi-turn cache-policy robustness untested (decode-produced KV vs prefill-rebuilt KV for the same history can diverge); MoE models break ODP's economics (loading/compute ratio scales with active fraction).
- **Released-artifact honesty (verified by web sweep, not by the paper):** the fork's own README states the 1.78× headline needed custom FlashInfer kernels that were "too ugly to release"; the released fork achieves **≈1.3×**. ODP is Blackwell-only (sm_120/121) in the released code.

## 2. Path 0 inventory (three-track decomposition)

| # | Paper component | Ships here? (signal-diff) | Extraction / routing | Verdict |
|---|---|---|---|---|
| 1 | Format disaggregation (A16 decode) | **De facto yes** — the riir-ai decode path is an f16-scale weight-only GEMV (no activation quant); prefill is compute-keyed (smem GEMM + tiled flash + mma FA arm). Signal-diff: our decode consumes raw activations — the NVFP4A16-decode shape exactly. | Corroboration at 2.8T of our existing posture. No action. | Covered |
| 2 | Phase-sensitivity measurement protocol | **No** — no phase-isolated quant bench exists in our lanes. | Extractable bench design (2×2 phase matrix, damage-ratio directional assertion, KV-axis extension). | **Extracted → Issue 026** |
| 3 | LUT grid optimization (Lloyd-Max + sign-absorbing scale) | **No** — `q2_0` uses a symmetric amax-scale ternary encoder; the 4th code state is *decodable* (+2d, `q2_0.rs:99`) but **encoder-unreachable** (the reference encoder's `d = amax` never emits it) and **bridge-rejected** (`UnsupportedFourthState`, the TernaryGroupWeights repack). | Encoder-only signed-`d` asymmetric grid (T0) + offline solver + non-uniform Q2_0A; gain site = dense GGUF only; kernel cost = a second 2-bit decode kernel. | **Extracted → Issue 027** |
| 4 | Dual-PTQ disaggregated container (modelless ceiling of the trained result) | **No** — single-checkpoint serving only. | Two-copy container + in-process phase handoff; the PTQ-vs-QADD recovery measurement. | **Fusion idea → Issue 028** (PoC-gated; null is pre-registered acceptable) |
| 5 | ODP SSD streaming + crossover predictor | **No** — and hardware-honest demotion: M3 unified memory + fast internal NVMe make mmap the honest incumbent; the 4090 fits a *resident* dual-checkpoint 27B posture; their fork is Blackwell-only at ≈1.3× released. | Recorded (§5). The crossover formula `T* = (W_pf/BW_ssd)/t_tok_prefill` is a 30-minute calibration if ever needed — folded into 028's optional arm. | **Discarded-for-now** — reason: no consumer box in our fleet where ODP beats resident-or-mmap at our model sizes; re-opens if a f16-prefill copy (>24 GB) ever needs serving on the 4090 |
| 6 | GDN escape set (`in_proj_a/b` + `A_log` + `dt_bias` frozen) | **Covered** — `ternary_weights.rs` Issue 980 `gate_projections()` escape set ("neither rotated nor quantized"); `issue879` cert + Bench-870 recipe keep `a_log`/`dt_bias`/conv1d/norms f32. | Paper corroborates from the *training* side at 27B and adds the phase-sharing detail (shared+frozen across prefill/decode). New rule worth recording: any future disaggregated GDN serving shares the recurrence dynamics frozen. | Covered (corroboration) |
| 7 | QADD training recipe | **No** — `dl_qat.rs` (Plan 255) has STE + magnitude×direction decomposition + LoRA fields, but grep confirms **no teacher/KL term and no per-position phase mask**. | Path 0.5: applicable training paper → riir-train Plan. | **Extracted → riir-train Plan 430** (secondary) |
| 8 | GDN recurrent-state + KV handoff (their engine-to-engine contract) | **Partial** — `deltanet/minimal_activation_cache.rs` ships the state vocabulary; no disaggregated-serving consumer exists. | Collapses to an in-process handoff for us; folded into Issue 028 T2 (byte-identity gate: decode-from-handoff == in-process prefill). | Folded into 028 |
| 9 | Hardware-aware phase-posture decision table | **Partial** — serving already dispatches phase-keyed kernels; no formal posture table. | The synthesis rule (decode → weight-only always; prefill → lowest accelerated matmul) recorded here; a table constant is only worth shipping if a second consumer appears. | Note-level |

## 3. Verdict

**Tiers (per track, never pooled):**

- **Track (a) — modelless inference/serving: GAIN.** Three issues filed (026 instrument, 027 format lane, 028 serving container). Not GOAT yet — every candidate needs its measured gate first; the paper is the published source of each mechanism, so no novelty is claimed over it. The strongest novelty-adjacent candidate is the **grid lane** (Issue 027): an encoder-only signed-`d` asymmetric grid activating the code state the format already decodes (T0), then a non-uniform Lloyd-Max Q2_0A that must beat T0 — a *format-lane* contribution that would be ours, gated on winning per-family at matched bpw on the dense-GGUF arm.
- **Track (b) — self-adaptive: PASS.** No runtime-latent update, direction-vector, or freeze/thaw angle exists in the paper. One-line reason: phase specialization is a serving/training-axis concern, not a latent-state one.
- **Track (c) — model-based: GAIN → riir-train Plan 430** (pre-registration). Per the serving-envelope-fit rule this is the SECONDARY plan: QADD trains weights outside the serving hot path, while 026–028 act on artifacts and the serving path itself. Plan 430 says so in its Status.

**MOAT gate (riir-infer): public substrate moat — quant formats, PTQ rules, loaders.** In scope and on-bar: 027 is squarely the format lane; 026 is a measurement instrument the format lane needs; 028 is a loader/container feature. No MOAT conflict; no routing to game/chain/db surfaces. Healer (fusion priority #2): **no corpus rule** — confirmed independently of the walk record; these are format/serving mechanics, not kernel code-shapes (the one kernel-adjacent item, LUT table-gather decode, already has its in-corpus substrate via the `lut16-pshufb-byte-expansion-at-stage` rule).

## 4. Signal-diffs (the §3.6 defenses, one read each)

- **Issue 980 escape set vs the paper's frozen set:** ours is *inference-motivated* (ternary container excludes `in_proj_a/b`; `a_log`/`dt_bias` f32 per the audited recipe). The paper's is *training-motivated* at 27B and adds **sharing across phases**. Same set, different evidentiary basis — corroboration, and one new rule (phase-sharing) recorded above. Not a gap.
- **`q2_0` vs LUT2A16:** different mechanisms — ours is a symmetric-encoded 3-state ternary at 2.125 bpw (decoding 4 levels); theirs is asymmetric 4-level Gaussian-optimized at 2.5 bpw. The `q2_0.rs` header *already documents* the fourth code state as decodable-but-unreachable and anticipates grid variants ("the `Q2_g64` / `PQ2_0` variants may use it") — Issue 027 is that anticipation made concrete under our single-level f16-per-128 scale.
- **`dl_qat.rs` vs QADD:** grep for `prefill|decode|mask|phase|distill|teacher|KL` returns zero hits in `dl_qat.rs`. QADD's two novel components (SFT-mask pathway selection; frozen-teacher KL on response tokens) are absent — Plan 430 extends, not duplicates.
- **Serving posture vs format disaggregation:** our decode GEMV is weight-only (A16-equivalent) and prefill is compute-keyed — the paper's scheme (a) at f16-class precision. The paper adds the *decision rule* (never activation-quantize decode) and the *instrument* (026); the posture itself needs no change.
- **Landscape (phase-decoupled weights prior art):** OverFill (arXiv:2508.08446, COLM 2025) decouples prefill/decode weight sets via *pruning* (full prefill, pruned decode — opposite direction to DQ's specialized prefill); Decode-Branch Transformers (arXiv:2608.12385) allocates phase-specific MoE experts; SlimWise (arXiv:2609.34117, 6 days post-paper) prunes experts per phase + distills KV continuation. DQ's deltas over all three: quantization-native, the frozen-decoder **prefiller** recipe (train prefill toward an already-quantized artifact), and the PTQ-only format axis. None of this is claimed as our novelty — it bounds what 027/028 can claim.

## 5. League intel (perf-league axis)

- **No upstream absorption, no re-arm trigger** (verified 2026-09-29): llama.cpp upstream carries only *state-handoff* disaggregation (Issue #21266 open; PRs #25675 closed-unmerged, #27004 draft, #27058 open — all same-weights); vLLM carries PD infra + an open RFC #59111 on per-phase KV page-format conversion. No opponent serving-path change touches our tg128/pp2048/pp4096 rows.
- **Watch items for the weekly opponent-watch lane:** (a) the fork's `odp` GGUF arch upstreaming into ggml-org (that is the event that would change the opponent's prefill path); (b) llama.cpp #27004/#27058 maturing; (c) vLLM RFC #59111; (d) the four missing 3-bit prefiller checkpoints on the Hub (repo touched 2026-09-29 — possibly still uploading).
- **Honesty datum to carry beside any quoted number:** 1.78× TTFT is the custom-kernel number; the released fork states ≈1.3×. ODP is sm_120/121-only — neither our 4090 (sm_89) nor the M3 can run the released path at all.
- **Accuracy-axis experiment candidate (recorded, not scheduled):** the released `ISTA-DASLab/Qwen3.8-27B-NVFP4-prefiller` targets *our league model family* (Unsloth Qwen3.8-27B GGUF, IQ1_S…IQ2_S variants live). A with/without-prefiller MMLU-Pro run through their fork would put a third-party number on the disaggregation premise at our exact model — useful league context if the format-evolution watch ever fires.

## 6. Prior-art landscape (web-verified 2026-09-29)

- **No scooper:** nothing newer than 2026-09-22 does phase-specialized quantization (formats or weights). The post-paper window holds only adjacent concurrent work (SlimWise, MpFA arXiv:2609.33135 phase-adaptive attention precision, KITE arXiv:2609.27294, Crossflow arXiv:2609.27085 — all systems/architecture axes).
- **Earlier, missed by the paper's related work:** OverFill (arXiv:2508.08446), Decode-Branch Transformers (arXiv:2608.12385) — both phase-decoupled weight/computation allocation, neither quantization-based.
- **Term history:** Forys et al. (arXiv:2608.03741) used the exact phrase "disaggregated quantization" as a *simulation dimension* seven weeks earlier; the HF forum discussed full-prefill/quantized-decode splits as engineering practice in March 2026.
- **Reception:** no replication or critique yet (Semantic Scholar 0 citations); HF Daily Papers #2 of the day; small traction proxies.
- **Reopen triggers:** (1) a quantization-native phase-specialization paper lands (re-run §4 against it); (2) the `odp` arch upstreams into ggml-org (league re-arm per §5); (3) the missing 3-bit prefillers land with changed claims; (4) any published replication contradicting the 1-bit accuracy headline.

## 7. Fusion

- **Paper × Issue 980 × katgpt-rs 538 × issue879:** the GDN escape set is now corroborated from three independent directions (our inference policy, the W4A4-survival mechanism, NVIDIA's training-side finding). The residual new rule — *recurrence dynamics stay shared+frozen across disaggregated phases* — is recorded in §1 and 028's gate.
- **Paper × `q2_0.rs` fourth-state fact:** the format decodes code 3 as +2d but the reference encoder never emits it and the ternary bridge rejects it. The grid lane (Issue 027) is the fusion: the paper's Lloyd-Max methodology applied under *our* single-level f16-per-128 scale — T0 activates the state encoder-only; Q2_0A must then beat T0; the dense-GGUF arm is the only gain site and the kernel cost (a second 2-bit decode kernel past `UnsupportedFourthState`) is priced in T4.
- **Paper × our hardware class:** the paper's ODP exists because a second resident checkpoint is prohibitive *on their box*. On ours it is not (27B q4-prefill + q2-decode ≈ 21–22 GB on the 4090; trivial on 64 GB unified) — so the interesting posture for us is **resident dual-PTQ** (Issue 028), and the *measurement* of trained-vs-PTQ recovery becomes the interesting science.

## 8. Files created / routing

| File | What |
|---|---|
| `riir-infer/.research/004_DQ_Disaggregated_Quantization.md` | this note (`.highwater` 003→004) |
| `riir-infer/.issues/026_phase_sensitivity_quant_bench.md` | the instrument (falsifiable directional assertion) |
| `riir-infer/.issues/027_lut_grid_optimization_lane.md` | encoder-only asymmetric Q2_0 (T0) + Lloyd-Max grids + non-uniform Q2_0A; dense-GGUF gain site; kernel cost priced |
| `riir-infer/.issues/028_dual_ptq_disaggregated_serving.md` | resident dual-checkpoint container + PTQ-vs-QADD recovery measurement |
| `riir-train/.plans/430_qadd_disaggregated_prefiller.md` | the QADD pre-registration (secondary track; budget ladder + gates; 27B convergence explicitly not priceable) |

GOAT discipline: every issue carries its own gate; nothing promotes without a measured win per-family at matched bpw (the lossy-surface law); Plan 430's phase-mask mechanism gate is pre-registered to be able to fail.
