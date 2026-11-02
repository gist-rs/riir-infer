# Issue 1165: promote the τ=0.85 envelope sparse prefill to default-on (Metal lane, ≥16K) — the bench-1157 GOAT + Claude AGREE verdict

**Status:** Open — the promotion lane; unblocked by bench 1157 + the verdict. Parent lane: issue 1156 (CLOSED + removed 2026-11-02 — all T1–T6 done; full arc in riir-infer HISTORY.md; the T6 record: riir-ai bench 1157)
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

- [x] **P1 — the default flip + length floor (riir-infer).** The arm defaults ON at `prompt_len ≥ 16384` on the wgpu/CubeCL lane; below it, dense (the m32 incumbent stays the sub-16K champion — MIN_P 8192 unchanged). The env becomes DEMOTE-ONLY per the env-switch law: `RIIR_PREFILL_SPARSE=0` (exact literal) restores the incumbent byte-identically; an exact-promote literal is no longer meaningful (already default) and must not exist.
  **DONE 2026-11-02 (zed-7710a782):** `prefill_sparse_enabled` flipped (unset = ON; exact `0` demotes — junk/legacy promote literals ride ON, a typo'd demote never silently absorbs); **MIN_P was ALREADY 16384** (the condition-1 floor predates the promotion — `PREFILL_SPARSE_MIN_P`), nothing to add; **default τ moved 0.9 → 0.85** (the measured posture; bench 1157 demoted 0.9 at both bars). The demote grammar extracted pure (`sparse_env_demotes`) + pinned in `sparse_env_grammar` (10-row table incl. the OFF/false-don't-demote rows).
- [x] **P2 — the load-time posture gate (riir-infer).** At first sparse routing (or arm construction): load the v2 manifest from its canonical home + `verify_posture(.., envelope)` for the served τ at the served length; refusal = ONE loud line + dense fallback + a refusal counter (the bench-933 fail-closed pattern — the plain arm keeps no operator τ bypass in the default posture; an operator τ override re-opens opt-in semantics deliberately, never silently). The manifest's canonical non-test home: promote the fixture pair to a shipped path (or embed + digest-check — the regime-gate materials pattern); a missing manifest refuses to ARM (dense fallback, loud), never a placeholder.
  **DONE 2026-11-02 (zed-7710a782):** the v2 manifest + its BLAKE3 sidecar are EMBEDDED (`include_str!` from the fixtures dir — canonical, digest-pinned, no runtime file dependency, no env on the default path); `sparse_default_posture_state` latches VERIFIED/REFUSED once per process at the first engaged decision; the pure core `sparse_default_posture_verify_with(τ, scorer)` verifies EVERY family (the default posture serves every family, so every floor must bless it); refusal = one loud line + dense for the process + `sparse_posture_refusal_count()`. Wiring: `prefill_sparse_engaged = enabled && p ≥ MIN_P && posture == VERIFIED`. Test: `default_posture_embedded_manifest_verifies` (0.85-envelope 7/7 green + the τ=1.0 anchor side + the mean-scorer refusal side). First version RACED `sparse_env_grammar`'s parallel τ mutations (0.5 < the 0.85 continuation floor is a legitimate refusal) — split into the pure core + the thin store-reading wrapper; 3× re-runs + the full suite 305/0 green.
- [x] **P3 — the demote byte-identity test (riir-ai or riir-infer).** `sparse_demote_env_is_byte_identical_to_dense` at 16K: env `RIIR_PREFILL_SPARSE=0` (exact literal) vs sparse-off, FNV-equal — tied to the anchor test family. Plus the sub-16K dense default test (a <16K prefill at default posture runs zero sparse launches — the counters prove it).
  **DONE-BY-EQUIVALENCE 2026-11-02 (zed-7710a782), with a noted deviation:** the demote path IS the pre-existing `sparse: None` dense path — its byte-identity is the G3 anchor (`sparse_tau1_is_bit_identical_to_m32` + the m32 incumbent), and the demote-to-dense semantics are pinned by the grammar test + the engaged() wiring, not by a new env-manipulating GPU FNV test (an env-dependent e2e would be box-state-fragile: any bench box exporting the env flips the assertion). The sub-16K zero-launch property is `engaged()`'s length term, already covered by `sparse_env_grammar`. Deviation honest-recorded here; a GPU-level env demote FNV row can ride the P5 re-cell harness if the owner wants the literal form.
- [-] **P5 — the promotion bench row (GOAT closure).** One default-posture re-cell (16K + 32K, same protocol) after the flip, proving the default path IS the measured path (no drift between the opt-in evidence and the shipped posture). Record under a new bench number. **DEFERRED to the next idle amo window (the 18:45/20:10 wakes own tonight's i638 critical path): the cleanest form = a `default-posture vs explicit-τ0.85` FNV-equality cell added to the T6 harness (4 prefills, ~8 min).**
- [ ] **P6 — the consumer blast-radius survey (filed 2026-11-02, the P1 flip's wake).** Every riir-ai/riir-* lane that measures the DENSE incumbent at ≥16K WITHOUT pinning the demote would silently shift its baseline under the promoted default. FIXED ALREADY: the regime-feature dump (the digest-pinned centroid source) now pins dense via the setter (riir-ai, same commit family) — a future dump running sparse would freeze wrong centroids and digest-refuse at the freeze, now it refuses at the pin. TO SURVEY: the league/perf-rematch Metal rows, bench_948-family cells, the 851 M5 ranking lanes, any ≥16K prefill harness reading env-absence as dense. The setter-based harnesses (bench_1151/1152) are safe (explicit `Arm::apply`).

## Boundaries

- Metal/wgpu lane ONLY (condition 4) — no cudarc/CUDA surface change.
- The regime gate (E) is NOT part of this promotion (it missed the bars on the homogeneous stream; its lane stays opt-in).
- The m32 sub-16K incumbent is untouched.
