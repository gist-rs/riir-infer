# Issue 1125 — decode fused-quantize rung: q8_1 written by the fused norm+Hadamard transform (the Bench-1004 escalation rung)

**Status:** OPEN — filed from the Bench-1004 bout; the 971 watch escalation ladder rung 1 (dedup grep + replicate), owner-gated for promotion only.
**Claimed:** zed-amo-i1125-rung, 1793016251, expires 2026-10-28T12:04:11+00:00
(scope: the fused-quantize rung implementation + G1 bit-identity / G2 decode e2e / G3 prefill-neutrality validation on amo, feature-gated opt-in; promotion to default-on stays owner-gated per the ask; runs on the same box as the i1113 re-cell lease — the two lanes share quiet-window discipline, measurement rows always preempt dev builds)


## The trigger (measured, not assumed)

Bench 1004 (2026-10-26, riir-ai `.benchmarks/1004_prism_b10826_bout.md`): the decode hold BROKE to **0.9730×** (ours 91.68 / opp 94.16) — the first sub-parity decode cell in the scored era. The opponent jump is +3.84% (their largest since #258) from prism b10819→b10826, dominated by the FWHT/q8_1 fusion family, DEFAULT-ON, PQ2_0-exact:

- **#395 `77a0affd87`** — quantize-to-q8_1 INSIDE the fused norm+Hadamard kernel for PQ2_0 consumers: the consumer-side standalone quantize launch is ELIMINATED (the kernel writes q8_1 rows in the consumer's layout directly).
- **#402 `c9c34bc9c4`** — o_proj/ffn_down Hadamard ⊗ q8_1 quantize fused, default-on (their +5-9% decode on 2B/4B H200; measured +3.84% at our 27B geometry together with the rest of the window).
- #401 `dc21b2cf42` — block-diagonal Hadamard truncation in the fused kernel (wide-block lanes).
- (b10822's #398 GDN conv-chain fusion default-on + #396 wider FWHT decode blocks rode the same scored build — first scored this bout.)

## The ours-side shape (dedup grep done — half-ships it)

Our folded decode lane (riir-infer; Bench-940's K1-K5 fused rotation kernels, `bonsai2_hadamard`) already fuses the whole-prefill rotation lane — but the decode consumer still pays a **separate quantize pass** where the opponent now fuses q8_1 quantization into the transform kernel. The mechanism is the producer-side twin of the Batch-288 rider #395 shape distilled upstream-side (riir-refine kernel_opt: "producer-side register + context-table face — q8_1 rows written in the consumer layout, the consumer lookup skips the quantize launch"). We ship the consumer half (the GEMV consumer reads q8_1); the producer half (the transform kernel writing q8_1 directly) is the gap.

## The ask

A fused quantize-into-rotation decode rung: the fused norm+Hadamard (rotation) kernel writes q8_1 rows (aliases block, use-count admission, cuBLAS-padded-view refusal — the Batch-288 admission scan shape) so the decode GEMV consumer's standalone quantize launch disappears. GOAT: G1 bit-identity (argmax-walk FNV `98412dc406c4ea19` + logits pin unchanged), G2 decode e2e vs the incumbent at the pinned posture, G3 prefill neutrality, G4 alloc. Feature-gated; promotion owner-gated on the league re-cell (the decode hold must re-cross 1.0×).

## Non-goals / scope guards

- NOT the lossy state-half successor (owner-gated, Plan 611/Bench 954) — this rung is modelless, bit-identical.
- NOT the prefill lane (sibling Issue 1124 carries the independent prefill-side MMQ GLU/gate-up lead from llama.cpp #29948) — this issue is decode-shape only.
- The opponent's #397 multi-column widening is Hopper-only (`pq2_mc_min = HOPPER ? 1 : 3`) — sm_89-inert; do not chase it.

**Refs:** riir-ai Bench 1004; prism #395/#401/#402 (b10826, tag `prism-b10826-c9c34bc`); riir-refine kernel_opt Batch 288 rider #395 (the distilled upstream face); riir-infer 1115/GEMV lineage; riir-refine `.distill/001` mining queue (the watch handoff row).
