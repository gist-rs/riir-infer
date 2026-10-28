**Status:** OPEN — implementation in flight (this session)

## 2026-10-28 — Issue 1151: the promoted serving posture has no single home — `production` aggregate + posture floor

**Found:** the set of promoted serving rungs — the feature selection every serving binary must compile — is hand-typed per consumer with no single home. The miss class fired THREE times in nine days, each a consumer's hand-typed list omitting a rung the league/engine posture carried:

1. **Issue 1141** — dist ran the bare decode kernels (37.24 → 21.43–21.48 ms/token after repair, 1.74×) because the production decode posture lived only in riir-ai's `riir-gpu` defaults; `riir-infer-gpu` has NO defaults by design.
2. **Issue 1146** — same class one list over: chunked prefill + batched attention omitted from dist's `gpu` row → the SEQUENTIAL per-token paths (128 dispatches per layer), 6–8× behind (the Bench-911 per-frame floor class).
3. **Issue 1146 rung 2a (Bench 920, promoted 2026-10-28)** — the ktile64 `metal_tensor_gemm` rung was armed by the league harness "and no serving binary did" until hand-promoted into dist's list.

Correctness-bearing, not only perf: `dq_phase_bench`'s manifest comment documents that WITHOUT the kernel-selection features (`ternary_gemm_simdgroup` + both recurrence arms + chunked/batched prefill) the lane runs fallback kernels computing WRONG LOGITS (base arith acc ≈ 0.07 vs the 0.25–0.95 admissibility window). A silent feature omission is a wrong-answer bug wearing a perf regression.

**Fix (two halves):**

- **Single home** — a `production` aggregate feature in `riir-infer-gpu`: the GOAT-passed promoted serving rungs, exactly the set dist's `gpu` row hand-carries (+ the `ternary_gemv` base it already implies). Contract: joining the aggregate IS the promotion act (GOAT-gated); opt-in rungs stay out; demotion stays env kill-switches (cargo features are additive-only — consistent with the house demote-only law). Consumers forward ONE name; a new promoted rung lands in ONE list and every consumer inherits on the next build.
- **Detection** — `PRODUCTION_RUNGS` (cfg!-truthful per compilation) + `missing_production_rungs()` exposed from the substrate, and a dist posture-floor test (behind `required-features = ["gpu"]` + the `[[test]]` row — the R1 reader-protection pattern this crate already uses 7×) that reds NAMING any rung a hand-typed fork could drop. Catches the class at `cargo test`, not at a benchmark a week later.

- [ ] `production` aggregate feature row in `crates/riir-infer-gpu/Cargo.toml` (contract comment: joining = promotion; inert-platform note for the metal leg)
- [ ] `PRODUCTION_RUNGS` + `missing_production_rungs()` in a new ungated zero-dep substrate module + lib.rs re-export
- [ ] dist `gpu` row forwards `riir-infer-gpu/production` (the 9 hand-typed rows replaced; provenance carried into the aggregate comment)
- [ ] dist posture-floor test file + `[[test]]` `required-features = ["gpu"]` row
- [ ] `cargo clippy` at the default and `gpu` postures; floor test GREEN at `gpu` (asserts 0 missing, 10 named), loud-skip at default
- [ ] fold the Claimed line into the Lease-closed row; close the issue with the commit hash

**Not in scope (deliberate):** riir-ai's `riir-gpu` 1:1 mirror forwards (sibling repo, lane-specific cuda/metal postures — optional follow-up there); env-gated rungs (`RIIR_GEMV_WG_OCCUPANCY` etc. are demote-only env, they inherit without features).

**Claimed:** mac-m3-issue-1151, 1793198468, expires 2026-10-30T00:00:00Z
