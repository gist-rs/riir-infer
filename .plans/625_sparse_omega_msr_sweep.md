# Plan 625 — Issue 1156 T3: the per-family ΩMSR sweep + digest-pinned manifest

**Status:** In progress — zed-amo-i1156 (lease expires 2026-10-31T17:22:37Z; issue `.issues/1156_sparse_prefill_league_eval.md`)

## Why

T2 (Plan 624, `03838f0` + bench 1151) landed the block-sparse prefill arm with
G1 bit-identity + G2 perf (1.166× @16K / 1.310× @32K) and left the arm OPT-IN —
**the binding per-family floors are T3's** (the lossy-surface law: per-family
conditional retention, never aggregate-only; the bench-889 continuation-ppl
axis is the named killer — static selection there lost filler continuation at
K ∈ {0.5, 0.75} while looking flat in aggregate).

T3 delivers: the sweep instrument (τ grid × per-family holdouts), the
**ΩMSR manifest** (digest-pinned; the floors any T4/T6 gate run must
verify against), the planted-knee self-test (both directions), and the
per-family conditional retention read path for every subsequent gate run.

## ΩMSR definition (pre-registered here, pinned in the manifest)

**ΩMSR(family) = the bottom edge of the top-contiguous passing run of the
ascending τ grid** — the largest τ such that its row passes AND every LARGER
grid row also passes. Strictly ≤ the issue's literal "last t with retention
≥ 1−ε": at a non-monotone curve the literal rule would floor a τ whose
neighbor above failed (a near-tie lottery), so the contiguous-tail rule is
the honest pre-registration. Top row fails → None (no safe high posture —
loud, never a silent floor).

A grid row **passes** iff BOTH:
- **retention ≥ 1−ε** (pooled argmax agreement over the final-prompt probe
  + the teacher-forced continuation probes; ε = 0.05), AND
- **ppl excess ≤ band** where measured (Δppl = ppl_sparse − ppl_dense over
  the continuation decode; band = +0.25 ppl; the bench-889 axis).

τ = 1.0 is the identity anchor (bit-identical to m32 — the T2 G3 contract),
not a floor candidate. Grid (ascending): **[0.5, 0.6, 0.7, 0.8, 0.85, 0.9,
0.95]** — swept descending; retention resolution comes from probes, not grid
density: **130 probes/family/τ** (2 items × [1 final + 64 teacher-forced
decode steps]).

## Corpus (holdouts — never calibrated on)

7 families, 2 items each, **NEW seeds — disjoint from bench_1151/bench_948's
texts** (those informed T2's design read; these adjudicate it):
`needle` (4 planted facts at spread depths + filler narrative), `qa`
(multi-hop chains), `summarization` (long narrative → summary continuation),
`code` (new multi-function Rust-like source), `math` (multi-step worked
arithmetic), `induction` (kv-recall generator, NEW server/code vocab),
`continuation` (plain flowing corpus text — no planted salience; the
bench-889 coverage-vs-salience stressor). Every item carries a 65-token
family continuation (the decode-probe + ppl axis at zero extra prefill
cost). The `continuation` family additionally runs at **32K** (2 items ×
7 arms) — the "multiple K" axis against the league's pp16k/pp32k rows.

## Artifacts + homes

| Artifact | Home |
|---|---|
| ΩMSR substrate (types, knee calc, blake3 helpers, `load_and_verify_manifest`, `verify_posture`) + planted-knee self-tests | `riir-infer` `crates/riir-infer-gpu/src/sparse_omega_msr.rs` (knee calc ungated; blake3/manifest I/O gated `cubecl_runtime` — same posture as the arm) |
| Sweep instrument | `riir-ai` `crates/riir-gpu/tests/bench_1152_sparse_omega_msr_sweep.rs` (the bench-1151 vehicle extended: prefill + teacher-forced decode via the serving `set_input_token`/`forward_token` protocol) |
| Manifest + digest sidecar (committed gate input) | `riir-infer` `crates/riir-infer-gpu/tests/fixtures/sparse_omega_msr_manifest.json` + `.blake3` |
| Record | `riir-ai` `.benchmarks/1152_issue1156_t3_omega_sweep.md` |

## Gates (pre-registered)

- **G-T3.A (in-sweep anchors):** τ=1.0 FNV == dense on every anchored item
  (item 0 per family); positive control: τ=0.5 flips logits bits somewhere
  AND the sparse counters advanced.
- **G-T3.B (knee self-tests):** planted knee found; all-pass → grid top;
  never-pass → None; ε-boundary passes at exactly 1−ε; unsorted grid
  refused loud; non-monotone curve floors the contiguous tail (not the lone
  point above the gap); digest round-trip + tamper refusal. CPU lib tests.
- **G-T3.C (manifest):** in-run `load_and_verify_manifest` round-trip green;
  committed copy's digest verifies.
- **G-T3.D (per-family read, forever):** the record + manifest carry
  per-family per-τ rows — never an aggregate-only read; T4/T6 MUST route
  posture checks through `verify_posture` (one mechanism, no re-derivation).

**No promotion claim** — the arm stays opt-in; the manifest binds any T4/T6
default move (the Issue-750-T3 lossy law).

## Tasks

- [ ] **T3.1** `sparse_omega_msr.rs`: consts (grid/ε/band), manifest types
      (serde_json), `omega_msr` knee calc (top-contiguous rule),
      `blake3_hex`/`blake3_file_hex`, `manifest_body_json`,
      `load_and_verify_manifest` (digest + consts drift refusal),
      `verify_posture`; `pub mod` row in lib.rs; self-tests
      (planted-knee both directions + boundaries + tamper) green at lib
      posture.
- [ ] **T3.2** Sweep instrument `bench_1152_sparse_omega_msr_sweep.rs`:
      holdout corpus (7 families, new seeds), 16K block (7×2 items ×
      [dense + 6 τ] + τ=1.0 anchors), 32K continuation block, decode-probe
      protocol, JSONL progress rows (env `OMEGA_SWEEP_ROWS`), manifest
      writer + in-run round-trip verify. Compiles green at the
      `qwen38_wgpu,ternary_gemm_batched,metal_tensor_gemm` posture.
- [ ] **T3.3** Run the sweep (GPU-exclusivity preflight; box state
      disclosed; ~2–2.5 h on amo; league model pin).
- [ ] **T3.4** Manifest digest-pinned at the fixture path + committed.
- [ ] **T3.5** Record `1152_issue1156_t3_omega_sweep.md` (per-family table,
      ΩMSR, the continuation-ppl axis vs the bench-889 killer, honest
      verdicts); issue 1156 T3 row; plan checkboxes; commit + push both
      repos.

- [-] Per-family floors at 32K for the non-continuation families — T6's
      league-cell lane re-verifies if promotion ever needs 32K per-family
      evidence (today only the killer axis is dual-length).
