# Issue 1155 — POC: exact slot-addressable GDN state (the Qwenputer identity on our kernels)
**Status:** Open — POC filed from riir-infer Research 018 (Qwenputer distill, 2026-10-29); unclaimed. Premise corrected in verdict round 1: exactness is a value-domain contract, NOT delta-rule algebra (G1 split into G1a/G1b accordingly).

## Premise

Qwenputer (divagr18/qwenputer @ `ca13310e`, MIT) runs a RISC-V CPU inside Qwen3.5-4B by driving its Gated DeltaNet layers as exact, addressable RAM: one-hot keys address one cell, gate = 1, write-β = 1, and — the load-bearing part — the payload lives on an **exact grid** (bounded integers) enforced by compile-time validators. ⚠ **The algebra alone does NOT give bit-exactness on our kernel**: `gated_deltanet_step_inplace` (`src/deltanet/forward.rs` L250-258) writes the addressed cell as `S[row,j] + b·(v[row] − S[row,j])` (the delta form). With b = 1 that is `a + (v − a)`, which is **not** `v` for arbitrary f32 — cancellation: a = 1e8, v = 1.0 → `0.0`. It IS `v` bit-exactly when payloads are bounded integers with |a|, |v|, |v−a| ≤ 2^24 (subtraction and re-addition exactly representable). The readout dot sums over ALL key columns, so the domain contract must also require **finite-only state** (`inf·0 = NaN` inside `simd_dot_f32` otherwise).

Our deltanet lane owns the same substrate (Bonsai-27B: 48 GDN layers; qwen3.8 arch twin family) and has never *proved* the identity — either direction. The `GdnStateJournal` (delta_roll, Issue 049) pins its f32 unwind with a drift floor because general state restore is "algebraically exact, NOT bit-exact" (its law 2). This POC establishes: (G1a) in-domain slots are bit-exact memory cells, (G1b) arbitrary-f32 payloads break it (negative control — pre-empting the Issue-825 false-negative "the identity does not transfer"), and (hypothesis) in-domain pinned slots restore bit-exactly through the journal under keep/halve.

**Orientation:** our state is `S[val_row, key_col]` (`row * key_dim + col`); a one-hot key addresses a key **column** j across all value rows — Qwenputer's "row" wording is transposed for our layout; assertions index columns.

Full construction-law set (L1–L8): `../.research/018_Qwenputer_GDN_Exact_RAM_Compiled_CPU.md`. Determinism law: any f32 fold in the proof follows katgpt-rs Issue 940 (sequential order, fixed tie rule).

## Tasks

- [ ] **G1a (CPU, in-domain battery):** over `gated_deltanet_step_inplace` — gate=1, β=1, one-hot key column, payloads on the exact grid (bounded integers: |a|, |v| and |v−a| all ≤ 2^24 — every intermediate exactly representable), finite-only state ⇒ every addressed cell round-trips **bit-exactly** and every non-addressed cell is **bit-unchanged**; sweep head counts and K/V widths.
- [ ] **G1b (CPU, negative control — must FAIL):** the same battery with arbitrary-f32 payloads ⇒ assert mismatches occur (include the cancellation pair a=1e8, v=1.0 → 0.0 as a named case). G1b passing (zero mismatches) is a harness bug, not a success.
- [ ] **Decay-selector semantics:** keep (g=1) and halve (g=0.5) are exact power-of-two state scalings; clear (g=0) zeroes the state exactly (destruction, forward-side). Halving-counter rows reach 2^-n exactly within the f32 exponent range (normals to 2^-126; subnormals to 2^-149 with reduced guarantee — an exponent bound, not a mantissa one).
- [ ] **G4:** zero-alloc hot path for the proof harness (counting allocator, own binary — the delta_roll G4 pattern).
- [ ] **G3:** byte-identical default posture (the POC ships as a gated demo/test; no serving-path change).
- [ ] **Journal interplay (HYPOTHESIS, pre-registered):** an in-domain pinned-slot region journaled + unwound via `GdnStateJournal` under **keep/halve** restores bit-exactly (exact scaling + same-grid subtraction). **Clear is out of scope** — journal law 1 refuses g=0 windows ("irreversible by construction"). A failure here is a RESULT (record drift + refine the hypothesis), never silently rewritten into the premise.
- [ ] Stretch (owner-gated): one wgpu parity cell at the same shapes; a ternary/PQ2 survival probe documenting which L1–L6 exactness contracts die under our league quantization (feeds the GDN W4A4 survival line, Research 538).

## Refs

- Research 018: `../.research/018_Qwenputer_GDN_Exact_RAM_Compiled_CPU.md` (provenance: blog 2026-10-22; repo pin `ca13310e`; model Apache-2.0, code MIT; L1 carries the kernel-level correction + counterexample)
- `src/deltanet/forward.rs` L244-262 (the delta-form write + Issue 594 orientation comment), `src/deltanet/delta_roll.rs` laws 1–2 (conditioning gate; "algebraically exact, NOT bit-exact")
- katgpt-rs Research 621 §M1/M8 (Handwired declaration/validator laws — the same discipline, independent substrate)
- riir-infer Research 007 (Triadic GDN — same lane, capacity axis), Research 538-class GDN quant survival
- katgpt-rs Issue 940 (float reduction-order determinism law); Issue-825 pattern (negative results recorded off false premises — G1b is the guard)
