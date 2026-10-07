# Issue 1005 — eDLM (Drex DLM) inference lane: GGUF arch + segment block-causal mask + pointer head

**Status:** OPEN — filed from reflex `.research/008_Drex_DLM_SystemOne_Lane.md` (owner ask "riir-infer can infer dlm or not — file issue if not", 2026-10-07). Supersedes reflex-issue-073's "riir-infer eDLM out of scope" line (owner opt-in).

## The gap (measured, not guessed)

riir-infer ships a **dLLM/D2F lane** (`dllm` feature: `src/transformer/dllm.rs` —
`forward_bidirectional` + `forward_block_causal`, Gemma2 CPU + `riir-infer-gpu/gemma2_d2f`)
but **cannot load or serve Drex DLM** (`nace-ai/drex-dlm`, Efficient-DLM-8B backbone):

- `src/gguf_loader.rs` arch dispatch supports `gemma2` / `llama` / `qwen2` / `qwen35`
  (+ PrismML ternary, deltanet) — **no arch accepts `edlm`** (their GGUF arch key; Q8_0
  keeps the pointer head F16).
- The D2F block-causal mask is **position/block-based**; Drex needs a **segment-based
  branch mask**: `attend(i,j) iff j≤i and (seg[j]==0 or seg[j]==seg[i])` + `state_bidir`
  (state attends state both directions, never sees branches; branches causal-within,
  attend state, isolated across questions) — `code/kev/model.py::branch_mask_batch` at
  `6c63df2` is the reference.
- **No marker encode/pack** (`<q>`/`<opt>`/`</opt>`/`<decide>` specials, per-question
  branches, `option_isolation` with shared position ids) and **no pointer head**
  (256-dim q/k Linear at decide/opt markers, scaled dot, temperature from `head.pt`).
- Row-form fallback (state+branch per causal row, bidirectional state block) — the
  mask-parity escape for long/overflowing requests — does not exist either.

## Why (consumer)

1. reflex issue 073 (`--drex` comparison lane) currently requires THEIR runner (Python
   torch or the llama.cpp `edlm` fork). With this lane, OUR substrate serves the same
   weights — the measure-vs-serve split gets a third serving posture (ours), and the
   eDLM arch joins the loader's league coverage (new arch family: AR→diffusion converts).
2. T7 closes reflex Research-002's open GAP (candidate-branch prefix reuse on a CAUSAL
   backbone — second reference impl noted in reflex `.research/008`) on OUR substrate.

## Boundary (the rung split — the laya precedent governs)

Drex/eDLM spans the family ladder and each rung has its owner:

| rung | repo | for Drex? |
|---|---|---|
| **substrate** (loader, forwards, mask, head mechanics) | `riir-infer` — THIS issue | yes — rethink's out-of-scope row delegates "the encoder forward SUBSTRATE → riir-infer (public)" |
| **measurement** (harness comparison lane) | `riir-reflex` issue 073 | yes — harness-only, never the release set (the agentjev family law) |
| **product serving** (weights posture, arsenal seating, ESC arms) | `riir-rethink` (private forever) | **NO — license-blocked**: CC BY-NC weights can never serve the commercial product; reflex's out-of-scope row sends "TRAINED-ENCODER serving arms → riir-rethink" |
| **open serving stack** (L1 lane seating) | `riir-instinct` (public) | **NO for Drex**: weights barred from any release set (the agentjev family law); NC also blocks downstream commercial users of the open stack |

The pattern is the laya lane's exactly: substrate in `riir-infer-laya` (public), serving
posture + product in rethink (private), reflex measures. The rethink rung for an
eDLM-class model opens ONLY when a license-clean model exists (open-licensed or
self-trained) — that issue is filed THERE and CONSUMES this issue's substrate; nothing
serving-shaped is ever built here or in reflex.

## Phases

### Phase 1 — load + forward (CORE)

- [ ] **T1** `edlm` GGUF arch support in `src/gguf_loader.rs`: config extraction
      (block_count, head geometry, rms eps — Qwen3 tensor-name family), weight mapping
      onto the existing transformer structs, pointer-head tensors carried F16 (Q8_0) or
      from `head.pt` (BF16 safetensors path). **Substrate-first: check tensor-name
      overlap against the existing `qwen35` loader arm BEFORE writing anything new —
      the eDLM block tensors follow Qwen3 naming, so T1 may be a variant of that arm,
      never a parallel loader.**
- [ ] **T2** Segment-based block-causal attention mask (generalizes
      `attention::block_causal_t_n`): segment ids + `state_bidir`; additive mask
      builder usable from both the packed forward and row forms. Unit tests pin the
      4-quadrant law (state↔state bidir, state∤branches, branch→state, branch∤branch).
- [ ] **T3** Row-form fallback: per-question causal rows (state + branch), packed-vs-row
      parity test (their `test_rows_match_packed` analog, our fixtures).

### Phase 2 — decision readout

- [ ] **T4** Marker encode/pack (specials, per-question branches, strict over-context
      refusal — never silent crop; `option_isolation` variant with shared position ids
      and fixed decide position).
- [ ] **T5** Pointer head module: q/k projections, scaled dot, temperature; load from
      GGUF tensors / `head.pt`.
- [ ] **T6** End-to-end parity: system-one request → encode → probs against their
      published sample outputs (`examples/request.json` → billing 0.9641 / refund
      0.8548 / urgency 0.5955 @ `6c63df2`) + a fixture corpus run through their Python
      runner (G5-style discipline: distribution parity, per-option delta disclosed —
      their own cross-runner tolerance is 0.0050–0.0094 GGUF-vs-BF16).

### Phase 3 — serving perf (deferred, owner-gated)

- [ ] **T7** State-prefix KV reuse (exact by construction: state never sees branches):
      prefix cache + branch continuation rows; `prefix_min_tokens` heuristic (their 384
      attention-only / always-hybrid law). Closes the bekko-006 inference-side sibling
      + reflex Research-002 GAP.
- [ ] **T8** GPU eDLM forward (`riir-infer-gpu`): mask plumbing + parity vs CPU.

## License law

Weights CC BY-NC 4.0: local bench/comparison serving only; never a product lane, never
a distill teacher. Their repo code is MIT — reference reading only, no code copy (our
implementation is clean-room from the wire + published mechanics).

## Out of scope

- Training/fine-tuning the decision adapter (no recipe ships anyway).
- Their multi-decision `requests` wrapper (refused in their release too).
- Hosted Drex 1.5 (128K) — different artifact; this lane serves the open 32K release.
