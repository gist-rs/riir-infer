# Issue 049: GDN delta-roll journal — rank-1 unwinding for GDN state checkpoints

**Status:** Open — **Claimed:** riir-refine-idle-4090, 1792515880, expires 2026-10-22T17:04Z
**Date:** 2026-10-20
**Source:** [katgpt-rs/.research/629_Depth_Memory_RLS_Depth_Mixing.md](../../katgpt-rs/.research/629_Depth_Memory_RLS_Depth_Mixing.md) (arXiv:2610.18809 "Depth Memory", state-unwinding trick) — distillation-derived fusion idea; novelty TBD, no code.

## Finding

The GDN state update is the **gated** delta rule: `S_t = S_{t−1}·diag(α_t) + β_t·(v_t − S_{t−1}·diag(α_t)·k_t)·k_tᵀ` (equivalently `S_t = S_{t−1}·diag(α_t)(I − β_t·k kᵀ) + β_t·v kᵀ`; the decay/gate is real — `decay_alpha: Vec<f32>` in `katgpt-attn/src/gdn2/types.rs`, init 0.99 **per key dimension**, so α is the diagonal matrix `diag(α_t)` applied on the right). Rank-1 machinery still applies, but unwinding is NOT verbatim the plain-delta-rule inverse: save `u_t = v_t − S_{t−1}·diag(α_t)·k_t` at write time; then `S_{t−1} = (S_t − β_t·u_t·k_tᵀ)·diag(α_t)⁻¹`. This works for any β (including β=1 with ‖k‖=1, where `(I − βkkᵀ)` is singular — the β-free form is what makes it safe). The α⁻¹ division is ill-conditioned exactly when decay is strong (`α → 0`), and f32 drift amplifies as `Π 1/α` over the unwound window. The paper's own mechanism uses the UNGATED form (`λ` enters the covariance rescale, not the state inverse) — the transfer here must carry the gate. **Conditioning gate: refuse to unwind past `min_j Π_t α_{t,j} < ε`** (per key dimension j; ε pinned from measurement) and fall back to a materialized snapshot for the remaining window.

## Proposed surface

A **journal** complementing the rollback-free verify path in `deltanet/tree_forward/`: verify deliberately never writes S₀ (rollback-free by construction), but post-commit writes DO land — retry paths, `kv_evict` eviction recovery, the disaggregated handoff lane. Today those cases have no cheap earlier-state recovery. The journal: store the final state + per-token `(u, k, β, α)` contexts; reconstruct any earlier state at `O(T·(D_v + D_k + 2))` bytes instead of `O(T·D_v·D_k)` snapshot storage — **~64× compression at D_k = D_v = 128** (journal 258 f32/token vs snapshot 16 384).

## Constraints / gates

- **Conditioning gate is load-bearing, not optional**: unwinding divides by per-key-dimension `α_{t,j}` columns; refuse past `min_j Π_t α_{t,j} < ε` (pin ε from measurement), snapshot fallback. A journal that unwinds through a near-zero-decay stretch silently amplifies drift by `Π 1/α`.
- f32 unwind is algebraically exact but NOT bit-exact — a G1 gate must measure and pin the replay drift (f64-accumulated escape hatch), same posture as katgpt-rs `marginal_rewind.rs`.
- The journal supports unwind; windowed read under gating: TBD at implementation, same α conditioning gate. The journal does NOT support re-derivation under different β/α schedules — contexts are schedule-specific; re-fit needs the raw `(k, v)` rows. Pin this in the module doc.
- Feature-gated (opt-in), zero-alloc, G4 canary; no interaction with the rollback-free verify invariant (verify stays write-free; the journal only covers post-commit paths).

## Related

- riir-infer issue879 (GDN quant certification) — the state surfaces it certifies are the ones the journal would snapshot.
- katgpt-rs `gdn_tree_verify` WS₀ folding — the same never-materialize algebra class, already GOAT-passed on the tree-verify axis (precedent, not coverage).
