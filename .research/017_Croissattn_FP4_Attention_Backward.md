# Research 17: CroissAttn — A Complete FP4 Attention Backward

> **Status:** Done — Pass (inference substrate; no FP4 consumer, hardware-gated) + Gain (training track → riir-train Plan 473)
> **Source:** "CroissAttn: A Complete FP4 Attention Backward" — ICLR 2027 submission, under review, anonymous authors. PDF @ github.com/alrojo/fp4Attn commit `6faeb979` (no license declared; under double-blind review)
> **Date:** 2026-10-27
> **Related Research:** 004 (DQ disaggregated quantization → Plan 430 QADD), 007 (Triadic GDN tensor state), 009 (LittleBit sub-1-bit quant), katgpt-rs Research 439 (NVFP4 RL — the FP4-format precedent; this note is that note's re-open trigger arriving), riir-train Research 087 (QAT-LoRA precision mismatch — externally replicated by this paper's Table 4)
> **Classification:** Public

---

## TL;DR

CroissAttn runs **all six attention products** (2 forward + 4 backward) in native NVFP4 on Blackwell tensor cores — prior recipes (SageAttention3, Attn-QAT, Full-Stack FP4) each keep part of the backward in higher precision. The recipe law layer: straight-through gradient target with round-before-normalize + unrounded denominator, **two-operand stochastic rounding** (unbiased, closed-form variance), saved-factor recasts with fresh SR streams per contraction axis, score replay from saved codes, an auxiliary softmax correction that preserves the STE target in conditional expectation through the whole chain (Theorem 1), and a propagated-vs-fresh error decomposition (Theorem 2). Execution: query-row-owned accumulation + packed storage (0.5625 B/value); covering-scale rule checked before draws; a logical random identifier seeds every SR draw for exact replay.

**Verdict shape:** the *speed/memory* claims are Blackwell-bound — the fleet has zero FP4 tensor cores (4090 = Ada SM89: FP8/INT8 only; M3/Ultra: none), and emulated FP4 inverts the value. The *recipe laws* are format-agnostic and two of them have stack relevance today: the QAT lanes round ternary weights with RTN (SR is the unbiased alternative, first principled variance treatment we have for the ternary lattice), and `softmax_fix.rs` (Issue 959) lacks the unrounded-denominator unbiasedness lemma. **Training track = Gain** → riir-train Plan 473 (pre-registered: A0 quant-ready GPU softmax-backward port spec → A1 QAT SR-vs-RTN flag study → A2 owner-gated FP8-emulated pilot). **Inference track = Pass** — no FP4 consumer; recorded for the format shelf. Research 439's re-open trigger ("re-evaluate if we adopt FP4") is hereby evaluated: still hardware-gated, but the complete recipe now exists to copy the day Blackwell-class hardware joins.

## 1. Paper core

### 1.1 Format and quantized forward
NVFP4 = E2M1 payload + one E4M3 scale per 16 values + FP32 tensor multiplier. Forward: `s = Qf Kfᵀ/√d`; row max `m_i` fixed on the full row; `U = exp(s − m_i)`; **`Uf = Q(U)` rounded BEFORE normalization**; the denominator uses **unrounded** exponentials (`P = L⁻¹U` effective, `Pf = L⁻¹Uf` packed operand); `O = Pf Vf`. Unit row maxima anchor the exponentials' forward scale.

### 1.2 Straight-through gradient target (their Eq 2)
`dP⋆ = dO Vfᵀ`; `dV⋆ = Ufᵀ(L⁻¹dO)` (row normalization folded INSIDE the product so scales are exact pre-rounding); `dS⋆ = Jp(dP⋆)/√d` with `Jp(X) = P⊙(X − rowsum(P⊙X))`; `dQ⋆ = dS⋆ Kf`; `dK⋆ = dS⋆ᵀ Qf`.

### 1.3 Two-operand stochastic rounding backward (their Eq 5)
Every backward product consumes independently SR-quantized FP4 operands. SR (their Eq 3): for `x ∈ [a,b]` adjacent representable levels, `Pr(round up) = (x−a)/(b−a)`, so `E[x̂] = x`, `Var = (x−a)(b−x)` — unbiased with known variance. Saved factors are decoded and RE-quantized ("recast") into each backward product's contraction-axis layout with fresh independent SR streams (NVFP4 block scales group along the contraction axis — transposition changes scale groups, so old codes are wrong, not just stale). Score replay: probabilities recomputed from the SAVED forward codes — no fresh Q/K quantization in backward.

### 1.4 Auxiliary softmax correction and Theorem 1
`D⋆,i = ⟨dO_i, O'_i⟩` where `O' = P Vf` is computed in higher precision; `dŜ = P⊙(dP̂ − D⋆)/√d`. **Theorem 1:** given exact score replay, covering scales fixed BEFORE payload draws, independent streams per product, fresh unbiased draws downstream — `E[dV̂|F0] = dV⋆`, `E[dQ̂|F0] = dQ⋆`, `E[dK̂|F0] = dK⋆`: the STE target survives in conditional expectation through the entire 4-product chain. (The D-term is standard FA2; the twist is computing it on the dequantized `O'`.)

### 1.5 Error decomposition (Theorem 2, their Eq 9)
For a product with independently perturbed operands: `E||ÂB − AB||² = bias² + V_A·B̄² + Ā²·V_B + V_A·V_B` (entrywise powers, adjoining matrices multiplied). dQ error = propagated (through the affine correction map) + fresh product error. Measured on frozen Qwen3-8B tensors, 10⁴ draws: **propagated noise = 62.8% of query-gradient squared error**; mean-bias < 0.003 at 3σ; degenerate identities reproduce to 6e−17.

### 1.6 Execution
Query-row-owned (QRO): each query-tile owner accumulates the full query gradient locally in high precision, writes once — no global partial-dQ buffer, no T×T matrix. Packed storage 0.5625 B/value. Covering-scale rule (their Eq 8): `g_X ≥ max|X|/(6·448)`, block scales `ceil_{E4M3+}` — guarantees representability; decoded scales CHECKED before draws (a clamped SR draw is a biased draw, silently breaking Theorem 1's premise). Determinism: a logical random identifier (training step, micro-batch, layer, product site, operand…) seeds every SR draw.

### 1.7 Results
- B200 native kernels: fwd 1.4–1.7×, **bwd 1.5–2.2×** vs BF16.
- From-scratch 500M/1.7B @ 10B tokens: validation gap 0.0439/0.0451 nats (projections-only 0.020; attention adds ~0.024).
- SmoothQuant-style smoothing α=0.3 is ESSENTIAL: CroissAttn gap 0.355 → 0.128 nats without→with at 1.7B.
- Qwen3-8B continued 200M tokens: 0.056-nat gap; 7-task 74.9% vs 75.0% under BF16 serving.
- **Table 4 (train-in-serving-precision):** FP4-trained checkpoints lose less under FP4 serving than BF16-trained ones (8B: +0.0348 vs +0.0768 nats; 1.7B: +0.0544 vs +0.2214); FP4 training improves FP4-serving loss vs direct-cast (−0.0198 @ 8B, −0.1060 @ 1.7B).
- Long-context: 8×B200, 168k tok/s @32k ctx (1.60×); 512k ctx in 93.6 GiB/GPU at 1.79× step rate.

## 2. Path 0 decomposition (mandatory inventory)

| # | Component | Value = math or loop? | Modelless extraction (advocate A) | Coverage signal-diff (advocate B + greps) |
|---|---|---|---|---|
| 1 | Two-operand SR (Eq 3) | Closed-form rounding law | YES — ships anywhere, zero deps | **NO coverage**: zero SR hits in riir-train quantize paths (`dl_qat`, `kimi_k3_quantize` round ternary with RTN) and none in riir-infer/quant. Signal-diff: RTN is deterministic biased rounding; SR is unbiased with variance `(x−a)(b−x)`. REAL GAP. |
| 2 | Eq 9 error decomposition | Closed-form predictor | YES — prices product MSE from per-operand mean/variance before running it | NO coverage — no variance-accounting instrument anywhere. GAP (instrument-class). |
| 3 | Covering-scale rule (Eq 8) | Codec schedule | YES + the generic constant: covering bound = max_payload × max_scale for ANY (payload, scale) format pair (ternary×FP8, INT8×FP8, …) | PARTIAL: q8 mirror paths pick scales implicitly; no pre-draw coverage assert. GAP (law-class). |
| 4 | Unrounded-denominator lemma | Small theorem | YES — `E[P̂] = P` entrywise under U-SR with exact denominator; slots into the `softmax_fix.rs` family (Issue 959) | PARTIAL: softmax_fix.rs catalogs precision-cliff/log-space numerics; no unbiasedness-under-rounding lemma. GAP. |
| 5 | Auxiliary correction `D⋆` | Affine identity + per-row Cauchy–Schwarz bias certificate `\|⟨dO, O'−O⟩\| ≤ ‖dO‖‖O'−O‖` | YES — free in every format | Base D-term exists in standard backward math (attention_backward.rs); the dequantized-`O'` twist applies only when quantization exists. |
| 6 | QRO schedule + score replay | Execution pattern | YES (HW-agnostic scheduling) | No GPU softmax-attention backward exists at all (CPU-only today) — GAP gated on the A0 port (Plan 473). |
| 7 | Logical random identifier | Replayable-seed scheme | YES — BLAKE3/philox of the tuple | NO coverage. **Prerequisite** for any randomized quantization under the Issue-940 determinism law (a global RNG stream is unreproducible across micro-batch reordering). GAP (substrate-class). |
| 8 | Smoothing transform (α-fold) | Offline calibration | YES — the transform ships; the α=0.3 constant is REFUSED (NVFP4-2-bit-mantissa-tuned; FP8 wants smaller; sweep never copy) | NO coverage — verified: zero SmoothQuant/amax machinery in riir-train. GAP gated on activation-quant existence. |
| 9 | Train-in-serving-precision (Table 4) | External evidence | Citation | Weight-side QAT lanes already ARE train-in-serving-precision for weights; Table 4 externally replicates riir-train Research 087's precision-mismatch premise at attention-product granularity → PASS-Redirect into 087. |
| 10 | Speed/memory/validation numbers | Hardware measurements | No (honest refusal R1/R2) | Not portable; benchmark DESIGN (paired windows, bit-identity pins, GPU-exclusivity preflight) is already house style. |

## 3. Distillation and fusion

**The ternary-lattice insight (the panel's sharpest single finding):** ternary payloads {−1, 0, +1} are exactly representable, so SR is vacuous ON payloads — but it is exactly the right treatment for **group scales and activations**, where ternary-path error actually lives. And the QAT lanes' ternary weight rounding is currently RTN — the one place SR has an existing consumer (Plan 473 A1).

**Closest cousins (why none kills):**
- **katgpt-rs Research 439** (NVFP4 RL): linear-layer/RL recipe, verdict Pass with the re-open trigger this paper arrives to answer. No attention-backward content — complementary, not duplicative. Updated with the PASS-Redirect below.
- **riir-train Research 087** (QAT-LoRA fusion): its precision-mismatch premise is now externally replicated at 8B/1.7B scale (Table 4). Updated with the PASS-Redirect below.
- **`riir-train-engine::deltanet::attention_backward`**: softmax-attention backward for Bonsai's hybrid attention layers (GQA, QK-norm, RoPE, sigmoid gating) — CPU, full precision, RTN everywhere. The mathematical surface matches; the quantization/SR mechanism has zero coverage; GPU coverage is DeltaNet-recurrence only. First consumer, not a kill.
- **`katgpt-core softmax_fix.rs`** (Issue 959): fixed-form softmax numerics family (precision cliff, log-space diff, margin equilibrium). The unrounded-denominator unbiasedness lemma is the same genre and absent there. Extraction path recorded, not filed — no rounding-noise consumer exists in that crate today.

**Novelty scoping (verdict round 1):** SR on low-precision weight updates is old prior art (Gupta 2015 line) — the stack claim is the COMBINATION (two-operand SR + identifier seeding + ternary-hybrid QAT target), never "SR on weights" as such.

**Fusion answer (what paper × 439 × 087 × the QAT lanes produce that none alone has):** a quantization-ready spec for the hybrid model's attention backward that is format-parameterized (ternary/FP8/NVFP4), carries unbiased rounding + replayable seeding + pre-draw coverage asserts from day one, and is externally validated by Table 4's train-serve-gap economics. That is riir-train Plan 473 — not an open primitive, because the generic katgpt-core versions (rows 1–4, 7) have no second consumer yet; they ride the plan's first consumer and get lifted on a second one.

## 4. Adversarial panel (§3.5, one spawn round)

- **No-GD advocate:** 17 extractions; headline rows 1, 2, 3–4, 9, 14 above; 6 honest refusals (kernel numbers, nats, the α constant, silicon SR, theorems-as-specs, transfer claims). Key reframe: covering bound = max_payload × max_scale generalizes the FP4 constant `6·448` to any format pair.
- **Model-based advocate:** verified in-repo (a) softmax attention backward is CPU-only, (b) zero SmoothQuant machinery. Structural finding: the paper's direct consumer doesn't exist on GPU — everything quantization-shaped is gated on porting `attention_backward` to the 4090 (independently justified: every real 4090 training run round-trips attention grads through CPU). Top action: one pre-registered plan, staged A0/A1/A2; item 10 redirected as a citation into 087. On record: adopting FP4-attention *training* on this fleet today is pure cost with no payoff — the defensible adoption is the law layer + the QAT-SR flag + the quant-ready port spec.
- **Prior-art agent:** no scoop — nothing public does all-six FP4 attention backward as of 2026-10-27. Nearest rival arXiv 2609.04105 (HW-aware FA4, Direct-P): FP4 forward but FP8 backward, and every MXFP4 P/V training trajectory **diverged** — exactly the failure this paper's SR + correction claims to fix. "Two-operand SR" has no public prior-art hits as a named technique. Verified real: SageAttention3 (NeurIPS'25), Attn-QAT, Full-Stack FP4, Quartet/Quartet II, Chmiel "FP4 all the way" (SR on backward products), MixFP4, UE5M3, TetraJet-v2. Caveats: concurrent ICLR-27 subs not all visible; anonymous under-review source.

## 5. Verdict (per track)

- **Training track: Gain.** Path 0.5 applicable — the recipe-law layer has a today-consumer (QAT RTN→SR flag, 2–6 GPU-h GOAT) and the port-spec rider unblocks the rest. → **riir-train Plan 473** (pre-registration only; A2 owner-gated; FP4-native core deferred on hardware with explicit triggers).
- **Inference/substrate track: Pass.** No FP4 consumer; no activation-quant lane; the generic codec laws (rows 1–4, 7) ride Plan 473's first consumer rather than shipping as shelf-ware in katgpt-core. Recorded here for the format shelf.
- **Provenance law:** anonymous ICLR-2027 submission pinned at `6faeb979` — nothing promotes to production on this source until camera-ready or Plan 473's own A2 replication lands, whichever first.

## 6. Re-open triggers (consolidated)

1. **FP4-native lane:** Blackwell-class hardware joins the fleet (SM100 FP4 tensor cores), or a training ticket requiring >32k context lands with attention >~30% of step time.
2. **Smoothing:** the moment any lane quantizes activations (QAT activation quant, FP8 KV/attention). Not optional then; α re-tuned per format, never copied.
3. **Train-in-serving as a lane** (not a citation): an activation-quantized serving lane exists, or 087's fusion study measures a train-serve gap weight-side QAT fails to close.
4. **katgpt-core generic extraction** (SR/covering-scale/Eq-9/identifier as open primitives): a second consumer outside riir-train appears.
