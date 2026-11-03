# Plan 627: GLM-5.3-Flash 1M-context lane on M5 Ultra — arch support + LSA bucket indexer (owner-pulled)

**Status:** Filed + claimed — owner pull landed 2026-11-03 (this conversation): M5 Ultra fleet (amo/bmo 96GB now, +4×128GB planned) will run GLM-5.3-Flash with 1M context as the USUAL posture; LSA prefill is the owner-stated need. Design input: [`.research/020_LSA_Latent_Sparse_Attention.md`](../.research/020_LSA_Latent_Sparse_Attention.md); lane lineage: Plan 624 (sparse prefill arm) → bench 1157 → issue 1165 (τ=0.85 envelope promotion, ≥16K, Metal — the dense-side baseline this lane extends).

**Claimed:** zed-m3-lsa-distill, 1793698334, expires 2026-11-10T00:00:00+00:00

## The two mountains (honest scoping)

1. **Serving GLM-5.3-Flash at all** — 320B total / 18B active MoE, hybrid sparse+linear attention, MLA latent, mHC, natively multimodal. HF card lists SGLang/vLLM/TokenSpeed/Transformers/KTransformers/Unsloth — no llama.cpp, no MLX. On M5 Ultra the paths are: (a) OUR stack (riir-infer GLM lane — the league premise), (b) quantized single-box (ternary/PQ2 ≈ 63–80 GB weights — our Bonsai specialty), (c) FP8 across 4×128GB via the Plan-623 multi-host lane (quality-preserving).
2. **LSA prefill speedup** — the paper's own curve on THIS model (H200): 3.5×/attention-layer prefill @1M vs its trained sparse baseline; 22.7× is the 10M cell; vs a DENSE baseline the sibling row (Qwen) shows 8.2×/layer @1M. End-to-end @1M vs the sparse baseline is modest (1.09× CP8 / 1.51× TP8 in Table 3) — the honest expectation for OUR first rung is per-layer and dense-baseline-relative, measured, never promised.

## Goal

One pre-registered gate: **does the PCA-training-free LSA bucket indexer beat our dense prefill arm by ≥2× per attention layer at pp1M on m5ultra, at G1 accuracy parity on real-tensor long-context captures?** Win → promote; lose → demote loser, recorded negative (the HGA precedent shape).

## Phases

### Phase 0 — acquisition + landscape
- [ ] Download `zai-org/GLM-5.3-Flash` checkpoint (HF; verify SHA256 per-token, record BLAKE3 in `data/BASE_MODELS.md`-style pin — third-party pin law)
- [ ] Arch probe (the `plan402_gguf_probe` pattern for safetensors/config): layer map (sparse vs linear attention split), MLA latent dims, mHC placement, MoE expert count/active, rotary layout (paper says none — verify in config), tokenizer
- [ ] Survey community posture: MLX / llama.cpp GLM-5.3 support state (the serving-vehicle fork; llama.cpp = the league opponent)
- [ ] Memory budget doc: 1M KV under MLA ≈ 1 GB/layer/sparse-layer (paper datum) → total KV + weights per posture (96GB single-box ternary vs 4×128GB FP8 multi-host)

### Phase 1 — GLM arch lane in riir-infer (the big lift)
- [ ] `glm5` arch feature: safetensors loader (MoE + tied structures), text-lane forward skeleton
- [ ] MLA latent attention forward (CPU reference first, parity posture; then Metal via the 1146 tensor stack)
- [ ] Hybrid linear-attention layers (GDN-class recurrence — reuse the `deltanet/` lane) + mHC
- [ ] MoE routing + expert dispatch on Metal (grouped GEMM posture)
- [ ] `- [-]` vision tower — DEFERRED (text lane first; multimodal encode is a separate lane)
- [ ] Parity gate: CPU reference vs Metal, frozen captures, p-drift ≤ 1e-3 (the G5 discipline)

### Phase 2 — quantization posture
- [ ] FP8/quant eval on the weights; ternary/PQ2 lane applied to MoE experts (Bonsai recipe) for the single-box 128GB posture
- [ ] Lossy-surface law: per-family conditional retention gate (per the UQ/lossy promotion rule), never aggregate-PPL-only

### Phase 3 — long-context bench cells + accuracy harness (do not defer)
- [ ] League manifest expansion pp256k / pp512k / pp1M on m5ultra — owner-sanctioned this conversation (2026-11-03); llama.cpp baseline row only if it supports the model by then, else dense-self baseline
- [ ] Real-tensor long-context accuracy captures: needle + multi-hop trail tasks over REAL documents (random-key NIAH is BANNED — the HGA lesson, Plan 397 / Research 379); PaperTrail-style fabricated-trail construction is the reference design
- [ ] Box-state preflight on every cell (bench_preflight posture; rig runs via `agent-mail rig run` — bmo quiet runner for long cells, exclusive for benches)

### Phase 4 — LSA bucket indexer (the owner-visible speedup)
- [ ] `lsa_bucket` feature (default-off): RoPE-free/MLA-latent index space (the pinned `index_space.py` contract), d(idx)=64, pool 4:1, Gram-Ward buckets (n̄b≈8, unit-norm centroids), bucket cache (mmap-friendly — NeuronShard-adjacent layout)
- [ ] Flat search first: threshold top-k (sample→θ→compact→rescore→packed score‖id radix select — the App. F kernel shape) as wgpu/Metal compute or CPU SIMD (d=64 makes a CPU indexer viable for the poc)
- [ ] PCA training-free projections = the FIRST arm (zero training); distillation (W^idx_Q,K + τ) only if the floor misses G1 — rides the riir-train Plan 476 owner-gated lane shape
- [ ] Tree/beam search deferred until flat wins (paper: tree loses accuracy past 1M on GLM; decode tree only pays batched ≥1M)
- [ ] GOAT gate (pre-registered HERE, before any measurement): G1 accuracy parity on real captures (≥ dense-self on needle/trail families, per-family not aggregate) · G2 ≥2× per-attention-layer prefill @pp1M m5ultra vs dense arm (--release, exclusive rig, box-state quoted) · G3 flag-off byte-identical at ALL existing league cells (≤32k — the crossover law says the flag must not touch them) · G4 alloc-free hot path
- [ ] Promote-to-default only on G1–G4 + owner; demote the dense arm in the ≥256k cells if won (keep dense <16K per the issue-1165 conditions)

## Non-goals
- 10M posture (CP8/H200 class) — not on this fleet; revisit if a 10M consumer appears
- CuTe DSL kernel parity (H200/B300-specific) — our arms are Metal/wgpu; speedups will differ and are measured, not transcribed
- Training N-LSA-style from scratch — out of charter

## Sibling etiquette
 amo = builder; bmo = quiet runner for long cells; heavy benches exclusive-run. The i1167 packed-byte lane (claimed to 11-04) is untouched; if it lands first, the Phase-1 Metal kernels build on it.
