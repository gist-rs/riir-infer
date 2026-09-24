# Issue 010 — gemma-2 fixture carries NO QK-norm: the 288-tensor GGUF + the forward both omit it (upstream Gemma-2 HAS it)

**Status:** OPEN — measurement/fixture-integrity finding from the Issue 883 P0 calibration prep (2026-09-25). No behavior change requested until a decision lands.

## Finding

`riir-train/data/gemma-2-2b-it-f16.gguf` is a **288-tensor** conversion (26 layers × 11 + 2 globals): **no `blk.N.attn_q_norm` / `blk.N.k_norm` tensors**. llama.cpp's standard gemma-2 conversion carries 340 (13/layer) — upstream Gemma-2 (all sizes) applies per-head RMSNorm to Q and K after projection, before RoPE (one of its headline architectural changes vs Gemma 1). The tensor probe (Python, GGUF header parse — see the session record) confirms 0 q/k-norm tensors; `gemma2.rs`'s forward correspondingly applies none, and `transformer/gemma4.rs`'s doc block states "attn_q_norm + attn_k_norm … are NEW vs Gemma 2" — **that claim is wrong** and has been load-bearing for this fixture's shape.

## Consequences

1. **The fixture is a mutated model, not upstream gemma-2-2b**: activations differ wherever QK-norm scales attention inputs (every layer). Any PPL/quality comparison of this stack against llama.cpp on a STANDARD gemma-2 GGUF is not apples-to-apples.
2. **Issue 883 P0's dashboard** (vk_calibration) runs on this artifact by necessity — self-consistent (loader, forward, and GGUF agree with each other), and V never passes through QK-norm even upstream, so ρ_l(V) transfers; but ρ_l(K)/ρ_l(V−K) describe THIS artifact (unnormalized K magnitudes). The bin prints this caveat beside every report.
3. gemma-4's loader/forward DO handle q/k norm (the mechanism exists in-tree to copy).

## Options (owner call)

- [ ] **(a)** Re-convert the GGUF properly (340 tensors — llama.cpp convert or a corrected internal path) + add q/k-norm to `forward_gemma2` / the f16 variant + this calibration lane's tap point moves post-QK-norm (883 trap 1's wrinkle, already anticipated). Fixture becomes upstream-faithful; any results pinned to the old artifact get re-run.
- [ ] **(b)** Keep the artifact as the house gemma-2 fixture, DOCUMENT it as such (this issue + the bin's caveat), fix the gemma4.rs doc claim. Cheapest; external-validity caveats stay.
- [ ] Either way: correct the "NEW vs Gemma 2" doc block in `gemma4.rs`.

## Trap (883 cross-ref)

Tap-point law: any standard-conversion gemma-2 must tap K post-QK-norm pre-RoPE (where the cache path stores it) — `gemma2_calibration.rs` documents this at the tap site.
