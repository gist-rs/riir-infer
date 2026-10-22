# Issue 1111: `forward_qwen_deltanet_ternary_hidden` returns the ROTATED basis on a Hadamard-folded file, and its doc names the wrong width

**Status:** Open — filed 2026-10-23 from riir-train Plan 461 T2.0 (the Bonsai-2 latent-loop backend), measured on `Ternary-Bonsai-2-27B-PQ2_0.gguf`.

## Finding

`src/deltanet/ternary_forward.rs` — `forward_qwen_deltanet_ternary_hidden` documents its return as "the post-final-norm residual stream … a 12288-D (Bonsai-27B) … vector". Two defects on a Bonsai-2 (Hadamard-folded, `prism.hadamard.*`) file:

1. **Basis.** The body (`forward_qwen_deltanet_ternary_body`) copies the final-normed `x` into `scratch.hidden_copy` and then forward-rotates `hidden_copy` IN PLACE before the LM-head matvec (the folded head reads the rotated vector). The function returns `scratch.hidden_copy`, i.e. the ROTATED vector, not the primal-basis final-norm state. Measured (riir-train `tests/plan461_bonsai_backbone_parity.rs` P3): `rotate_inverse_inplace(returned)` matches the primal final-norm state to 1.5e-5 max-abs (|h| ≈ 104); the returned vector itself does not.
2. **Width.** Bonsai-27B's `n_embd` is **5120** (GGUF header), not 12288; the function returns `n_embd` values.

Consequence: a cosine/inner-product consumer (the dense embedder) is unaffected — the rotation is orthogonal — but any consumer that combines the returned hidden with primal-basis quantities (input embeddings, a primal-basis probe, an adapter trained on primal states) silently mixes bases. riir-train's backend keeps its own primal copy before rotating and does not use this function.

## Tasks

- [ ] **T1** Decide the contract: return the primal basis (copy before rotating — one extra `n_embd` copy), or keep the rotated basis and document it. Either way, state the basis in the doc.
- [ ] **T2** Fix the width in the doc (5120 for Bonsai-27B).
- [ ] **T3** A folded-file test pinning whichever basis is chosen (riir-train's P3 is the shape).
