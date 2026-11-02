# Issue 1165: promote the τ=0.85 envelope sparse prefill to default-on (Metal lane, ≥16K) — the bench-1157 GOAT + Claude AGREE verdict

**Status:** Open — the promotion lane; unblocked by bench 1157 + the verdict
**Claimed:** zed-7710a782, 1793685200, expires 2026-11-04T16:45:00+00:00

## The decision being implemented

**VERDICT AGREE (Claude owner-proxy, 2026-11-02, round 1/3)** — promote the block-sparse prefill arm at τ=0.85 (envelope scorer) from opt-in to DEFAULT-ON for the Metal/wgpu league lane, under four conditions (verbatim):

1. **Apply it only to prompts of 16K tokens or longer.** Below 16K nothing was measured. The fixed scoring cost is about 0.4% at 16K and gets relatively larger on shorter prompts, so short contexts could come out net-slower. Keep dense there until a shorter length is benchmarked.
2. **Check quality at load and fail safe.** At serving startup, call `verify_posture(.., envelope)` against the v2 manifest. If it refuses, log it loudly and fall back to dense, never silently.
3. **Keep the off switch.** `RIIR_PREFILL_SPARSE=0` must give output byte-identical to dense, and that byte-identity check should stay tied to the anchor test.
4. **Metal/wgpu lane only.** The CUDA path is a different forward and stays untouched.

Non-blocking follow-ups (the verdict's list): re-measure on bmo (all timing is one machine), mixed-content-corpus speed cells (quality coverage is 7 families; speed is not), resolve issue 1164 (Cell D vacuity).

## The evidence base

- [bench 1157](../riir-ai/.benchmarks/1157_issue1156_t6_league_cells.md): Cell C 1.0310× @16K / 1.0565× @32K verified-idle, 0 flips, anchor FNV 6/6; the T3-v2 under-load floors beaten; τ=0.9 demoted; the gate overhead quantified.
- The v2 ΩMSR manifest (bench 1153): 7/7 families green at τ=0.85 envelope. **State plainly in the promotion commit: the math floor is 0.5 — the loosest of the seven; that is the documented cost of this setting.**
- G3/G4 already anchored: env-off byte-identity (`sparse_tau1_is_bit_identical_to_m32` + the T5 mode anchors, re-asserted 304/0 at `3f6353c`); disarm = no scratch alloc (structural).

## Tasks

- [ ] **P1 — the default flip + length floor (riir-infer).** The arm defaults ON at `prompt_len ≥ 16384` on the wgpu/CubeCL lane; below it, dense (the m32 incumbent stays the sub-16K champion — MIN_P 8192 unchanged). The env becomes DEMOTE-ONLY per the env-switch law: `RIIR_PREFILL_SPARSE=0` (exact literal) restores the incumbent byte-identically; an exact-promote literal is no longer meaningful (already default) and must not exist.
- [ ] **P2 — the load-time posture gate (riir-infer).** At first sparse routing (or arm construction): load the v2 manifest from its canonical home + `verify_posture(.., envelope)` for the served τ at the served length; refusal = ONE loud line + dense fallback + a refusal counter (the bench-933 fail-closed pattern — the plain arm keeps no operator τ bypass in the default posture; an operator τ override re-opens opt-in semantics deliberately, never silently). The manifest's canonical non-test home: promote the fixture pair to a shipped path (or embed + digest-check — the regime-gate materials pattern); a missing manifest refuses to ARM (dense fallback, loud), never a placeholder.
- [ ] **P3 — the demote byte-identity test (riir-ai or riir-infer).** `sparse_demote_env_is_byte_identical_to_dense` at 16K: env `RIIR_PREFILL_SPARSE=0` (exact literal) vs sparse-off, FNV-equal — tied to the anchor test family. Plus the sub-16K dense default test (a <16K prefill at default posture runs zero sparse launches — the counters prove it).
- [ ] **P4 — the records.** The promotion commit message carries: the math-floor-0.5 disclosure, the verdict conditions, the bench-1157 citation. HISTORY rows both repos; the 1156 T6 row + this issue close together; the README feature-gate table moves the arm to default-on (Metal, ≥16K) with the demote literal documented.
- [ ] **P5 — the promotion bench row (GOAT closure).** One default-posture re-cell (16K + 32K, same protocol) after the flip, proving the default path IS the measured path (no drift between the opt-in evidence and the shipped posture). Record under a new bench number.

## Boundaries

- Metal/wgpu lane ONLY (condition 4) — no cudarc/CUDA surface change.
- The regime gate (E) is NOT part of this promotion (it missed the bars on the homogeneous stream; its lane stays opt-in).
- The m32 sub-16K incumbent is untouched.
