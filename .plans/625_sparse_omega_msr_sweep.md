# Plan 625 — Issue 1156 T3: the per-family ΩMSR sweep + digest-pinned manifest

**Status:** T3 + Protocol v2 + T4/T5 machinery DONE 2026-10-30 — zed-amo-i1156 (lease expires 2026-10-31T17:22:37Z; issue `.issues/1156_sparse_prefill_league_eval.md`; remaining: T4.4 freeze + T6 cells, both GPU idle-window-gated)

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

- [x] **T3.1** `sparse_omega_msr.rs`: consts (grid/ε/band), manifest types
      (serde_json), `omega_msr` knee calc (top-contiguous rule),
      `blake3_hex`/`blake3_file_hex`, `manifest_body_json`,
      `load_and_verify_manifest` (digest + consts drift refusal),
      `verify_posture`; `pub mod` row in lib.rs; self-tests
      (planted-knee both directions + boundaries + tamper) green at lib
      posture. (riir-infer `56068a3`; 12 ungated + 14 gated green)
- [x] **T3.2** Sweep instrument `bench_1152_sparse_omega_msr_sweep.rs`:
      holdout corpus (7 families, new seeds), 16K block (7×2 items ×
      [dense + 6 τ] + τ=1.0 anchors), 32K continuation block, decode-probe
      protocol, JSONL progress rows (env `OMEGA_SWEEP_ROWS`), manifest
      writer + in-run round-trip verify. Compiles green at the
      `qwen38_wgpu,ternary_gemm_batched,metal_tensor_gemm` posture.
      (riir-ai `7e004561f`; clippy -D clean at `ternary_gemm_batched`;
      note: the grid is 7 τ points [0.95..0.5] matching the substrate
      grid, and continuations were lengthened ≥65 tokens after the first
      dry run refused short-tokenized ones)
- [x] **T3.3** Run the sweep (GPU-exclusivity preflight; box state
      disclosed; ~2–2.5 h on amo; league model pin). (136 passes / 9520.6 s;
      anchors 8/8; counters +8640/+17280; rows at
      `/tmp/sparse_omega_sweep_rows.jsonl`)
- [x] **T3.4** Manifest digest-pinned at the fixture path + committed.
      (digest `a47b96e46557335a…`; in-run round-trip verified; sidecar
      `.blake3` alongside)
- [x] **T3.5** Record `1152_issue1156_t3_omega_sweep.md` (per-family table,
      ΩMSR, the continuation-ppl axis vs the bench-889 killer, honest
      verdicts); issue 1156 T3 row; plan checkboxes; commit + push both
      repos. (riir-ai `585b19cdd`)

- [-] Per-family floors at 32K for the non-continuation families — T6's
      league-cell lane re-verifies if promotion ever needs 32K per-family
      evidence (today only the killer axis is dual-length).

## Protocol v2 — the scorer re-open (the T3 named lever)

**Status:** DONE 2026-10-30 — zed-amo-i1156 (all V2 tasks landed; verdict: 7/7 families floored, envelope stays default; issue 1156 T3-v2 row)

T3's verdict refused blanket promotion and named the re-open lever: the
pooled-centroid scorer is blind to single fact rows (mean dilution — the T2
design doc's own named risk, measured). Protocol v2 swaps the scorer, keeps
EVERY floor constant (grid/ε/band), and re-runs the SAME sweep (same holdouts,
same corpus digest, same model pin) so the v1↔v2 read is paired:

- **The fix (Quest-class min/max envelope):** kernel 1 pools per-dim **max +
  min** K per block (a fact lives in the extremes, not the mean); kernel 2
  scores `band + envelope` where `envelope = Σ_d q̄_d·(q̄_d ≥ 0 ? kmax_d :
  kmin_d)` — an upper bound on `max_r(q̄·k_r)` over the block's rows, tight
  exactly on single-fact rows. The band (diagonal) and the select kernel are
  untouched; τ=1.0 stays score-independent (the bit-identity contract holds
  BY CONSTRUCTION).
- **Posture:** `RIIR_PREFILL_SPARSE_SCORER` — `envelope` (v2 DEFAULT),
  `mean` (v1 demote, exact literal; junk/unset → envelope). The committed v1
  manifest stays the MEAN scorer's binding floors; v2 lands its own manifest
  (`sparse_omega_msr_manifest_v2.json`) and `verify_posture` binds
  scorer == manifest scorer — a floor never blesses the other scorer.
- **GOAT:** the v2 sweep IS the gate — G1 per-family ΩMSR floors
  (pre-registered, unchanged); G2 the sweep's wall reads at τ=0.9 vs dense
  (envelope costs one extra pool butterfly + same-shape score FLOPs —
  expected ≈ 0 e2e); τ=1.0 anchors must stay bit-identical.

### Tasks

- [x] **V2.1** substrate: `SparseScorer` posture (env + setter + grammar
      tests); pool kernel min/max mode (finite sentinels f32::MAX/MIN — no
      infinity literals in kernel bodies); score kernel envelope mode;
      scratch kpool 2× (max | min halves); launchers + the dispatch caller.
      (riir-infer `7f4b00f`)
- [x] **V2.2** omega substrate: `OMEGA_MANIFEST_VERSION` 2 (reader accepts
      ≤ 2: v1 = scorer `mean` implicit — the demote posture's floors stay
      verifiable; NEWER refuses; v2 REQUIRES the `scorer` field);
      `verify_posture(.., scorer)` binding; self-tests for both versions +
      mismatch refusals. One canonical enum (`OmegaScorer`, ungated; the
      arm re-exports `SparseScorer` — the two-enum DRY slip caught by the
      instrument's round-trip assert at compile time, `123d4c4`).
      (riir-infer `003ff7f` + `123d4c4`)
- [x] **V2.3** instrument v2 (riir-ai): scorer pin (`OMEGA_SWEEP_SCORER`,
      default `envelope`), manifest `scorer` field, v2 fixture + v2 rows
      paths (the v1 artifact pair is never overwritten). (riir-ai
      `a3645644a`)
- [x] **V2.4** kernel tests green at the arm posture — incl. the planted
      single-row needle test (a needle block score ≥ 16× its mean-mode
      self — the mechanism proof) and hot-block dominance in BOTH scorers.
      (23/23 green at `ternary_gemm_batched`; clippy −D clean at
      `qwen38_wgpu,ternary_gemm_batched,metal_tensor_gemm` — incl. the
      pre-existing i1150 grouping break fixed grouping-only, `868dba6`)
- [x] **V2.5** run the sweep (GPU-exclusivity preflight; box state
      disclosed) → per-family verdict. (136 passes / 10954.75 s; DETACHED
      + polled after the watchdog killed the first foreground attempt at
      101 rows — lesson in the record; rows
      `/tmp/sparse_omega_sweep_rows_v2.jsonl`, attempt-1 kept as
      `.attempt1.jsonl`)
- [x] **V2.6** land manifest v2 + digest sidecar + record + issue/plan
      updates; the DEFAULT-scorer call rides the verdict. (riir-infer
      `b8b06eb` — digest `a4b55a15e0ef8774…`; verdict 7/7 families floored,
      math improved 0.9→0.5, no regression anywhere → **envelope stays
      DEFAULT**; record riir-ai `.benchmarks/1153_issue1156_t3v2_envelope_omega_sweep.md`)

## T4 + T5 — the regime gate and the mixed-mode wire

**Status:** LANDED (machinery) 2026-10-30 — zed-amo-i1156; the FREEZE + the
league cells are the remaining GPU lanes (idle-window-gated)

T4's premise (restored by T3-v2): the modelless regime gate has real floors
to consume. T5 gives the floors' decisions a mixed-mode execution wire.

- **T5 (fused mixed-mode):** the sparse m32 kernel gains a per-head mode
  word (`head_mode[head_idx] != 0` → DENSE — the skip branch dies for that
  head's cubes; cube-uniform, no divergent shuffle hazard). The launcher
  slices a padded per-layer buffer (`sparse_mode_layer_stride_words` =
  32-byte-binding-aligned strides — the wgpu offset law); the CPU mirror
  (`head_mode_words`) lets the dispatch skip the WHOLE sparse pipeline for a
  fully dense layer (pool/score/select too). Env
  `RIIR_PREFILL_SPARSE_DENSE_HEADS="0-3,7"` (junk anywhere demotes the
  whole knob loud — a partial list is the failure class); heads ≥ 62 refuse
  (the bitmask cap, fail-closed).
- **T4 (modelless regime gate, `regime_gate.rs`):** boundary-pooled hidden
  states ([`boundary_pool`]: first/last 100 rows of the chunk's residual
  stream, mean, concat — sigmoid margin, NEVER softmax) → two frozen
  centroids (digest-pinned sidecar pair, `parse_centroids_body` refusing
  kind/version/dim/margin drift) → per-chunk class (retrieval / flow /
  ambiguous) → τ from the T3 manifest at the CURRENT length through the NEW
  `sparse_omega_msr::conservative_floor_for_len` (exact bucket wins; between
  buckets the ceil bucket; past every bucket the tightest; a `None` bucket
  poisons at-or-above reads; unknown tags Err) + `verify_posture_at_len` —
  the length-aware arm of the ONE mechanism (the binding arm blesses
  τ=0.7 for continuation at 32K where the measured 32K row demands 0.85;
  the gate holds a length, so it consumes the strict arm). Ambiguous → the
  max of both class floors, belt-verified over ALL 7 families. A class with
  any missing/unmeasured/floor-less family routes DENSE (fail-closed).
- **The integration seam:** the gather kernel (`boundary_gather_rows_f32`,
  explicit-params bounds — the len-derived-binding law) + `run_regime_gate`
  (chunk-local, after the residual stream stages, before the first
  attention dispatch; armed-only alloc, G3). Materials cached once per
  process (OnceLock); a load failure = ONE loud line + refusal counter +
  the plain arm keeps the operator's env τ — never a silent change.
  Counters `regime_gate_counters()` = (decisions, dense_chunks, refusals)
  — the vacuous-guard triple for T6.
- **Honest scope:** the v1 dynamic range on the committed v2 manifest is
  τ ∈ [0.8, 0.85] at 16K, degenerating to 0.85 at 32K (the measured
  continuation row binds); the decision is per-chunk uniform across
  heads/layers — the per-head/per-layer WIRE is T5's, exercised by the env
  + tests; per-layer differentiation is the trained router's (Plan 476)
  upgrade path. `REGIME_GAIN` is a named placeholder pending the freeze.

### Tasks

- [x] **T5.1** kernel head_mode arg + skip line; launcher per-layer slice
      (32B-aligned padded layout); scratch mode buffer + CPU mirror;
      `parse_dense_head_mask` + env arm; layout helpers pinned to the
      alignment law. (riir-infer this commit)
- [x] **T5.2** the G1 battery green ON METAL: `all_dense_mode_ignores_the_mask_and_is_bit_identical_to_m32`
      (lossy τ=0.7 fixture mask proven lossy first — the non-vacuity
      guard), `all_zero_mode_tau1_stays_bit_identical_to_m32` (the T2 anchor
      through the T5 wire), `mixed_mode_equals_the_union_of_single_mode_runs`
      (heads {0,2,4} dense vs {1,3,5} sparse in ONE launch == the per-head
      union, fixture-proven dense ≠ sparse somewhere),
      `sparse_tau1_is_bit_identical_to_m32` re-green (the incumbent anchor
      post-change). Plus the pure suites (mask parse incl. junk/cap/inverted
      demotion, layout alignment, gather-vs-CPU bits). 298/298 lib green at
      `cubecl_runtime`; clippy −D clean at the lane posture + `production`
      check green. Contention disclosure: the sibling rethink sweep POSTs to
      amo's mammoth serve during the run — bit-identity gates are
      deterministic under serialized Metal execution (the exclusivity law
      governs TIMING gates; T6's wall cells still run verified-idle).
- [x] **T4.1** `sparse_omega_msr`: `OMEGA_LEN_BUCKETS`,
      `conservative_floor_for_len`, `verify_posture_at_len` + 6 planted
      tests (exact/ceil/past/None-poison/unknown-tag/32K-stricter-than-binding).
- [x] **T4.2** `regime_gate.rs`: boundary_pool (+ the gathered-buffer
      bit-equivalence pin), sigmoid/cosine, decide policy both directions +
      illegal-class-dense + ambiguous-conservative, floors from the COMMITTED
      v2 fixture (16k → (0.85, 0.8); 32k → (0.85, 0.85)), sidecar parser
      drift refusals, env arm + counters. 95/95 green at the no-features
      posture (the substrate runs everywhere).
- [x] **T4.3** the seam: gather kernel + `run_regime_gate` + the forward
      hook (staged-guard: only when the residual stream is staged) + the
      dispatch overrides (τ override, chunk-dense routing, per-layer
      dense skip) + `regime_gate_bnd_elems` (arming and alloc size in ONE
      place). `boundary_gather_matches_the_cpu_rows` green on Metal.
- [-] **T4.4** the CENTROID FREEZE pass — MACHINERY COMPLETE (riir-infer
      `7ff9a70`: the capture seam + `regime_centroid_freeze` CLI; riir-ai
      `d9eddba28`: the `bench_1156_regime_feature_dump` runner over the SAME
      byte-identical holdout corpus). Remaining = three steps in the next GPU
      idle window: (1) run the dump (~15 min dense on amo), (2) freeze via
      the CLI (the sidecar pair lands at the fixture path, digest-pinned),
      (3) calibrate `REGIME_GAIN`/margin on the dump's cosine spread. The
      gate refuses to arm until the pair exists — fail-closed, never
      placeholder centroids in the tree.
- [-] **T6** league cells at the measured postures (τ=0.85 / 0.9, now plus
      the mixed-mode + regime-gate postures once T4.4 lands) — the decision
      cell from the T3-v2 verdict table; verified-idle protocol, both box
      classes, preflight provenance; promotion owner-gated on the GOAT
      (perf AND per-family quality).

### T6 runbook (pre-registered — the idle-window cell plan)

**Preconditions:** verified-idle (`scripts/bench_preflight.sh` → quote the
`PROVENANCE:` line beside every number); no sibling serving lanes co-tenant
(the rethink sweep's mammoth/rethink serves are the known occupants — `ps`
before every cell); both boxes gitsync'd; league model pin
(`BONSAI_GGUF` → `Ternary-Bonsai-2-27B-PQ2_0.gguf`).

**Lane scoping (the i1149 routing-proof pattern):** the sparse arm is wired
in the wgpu/CubeCL forward (macOS lane — `prefill_attention_layer_batched`).
The 4090's scored prefill is the cudarc whole-prefill lane — a DIFFERENT
forward. Routing is proven by the launch counters, never assumed: on any
lane claiming to run the arm, `sparse_attn_launch_count` MUST move.
Expected: structurally N/A on the cudarc lane — any CUDA cell is a separate
wiring lane, NOT a T6 cell. The "both box classes" clause of the original
T6 row reads as: Metal cells here; the CUDA half is recorded N/A (or opened
as its own lane) with the routing proof, never silently skipped.

**Cell order (amo, --release):**

0. **T4.4 freeze first** (dense, calibration — quiet-box preferred, not
   strictly idle-gated): `bench_1156_regime_feature_dump` →
   `regime_centroid_freeze` → commit the sidecar pair → calibrate
   gain/margin on the dump's cosine spread. Cell E needs it; nothing else
   does — run it first so the window is not blocked on calibration later.
1. **Cell A — dense baseline** (`RIIR_PREFILL_SPARSE` unset): the pp16k /
   pp32k reference wall every other cell divides by.
2. **Cell B — τ=0.9** · **Cell C — τ=0.85**: 3 interleaved rounds
   median-of-ratios vs Cell A (the T2 thermal-order protocol); a τ=1.0
   anchor FNV fail-fast per cell (the G1 contract); counters engaged
   (pool/score/select/attn + regime counters for E).
3. **Cell D — mixed-mode** (`RIIR_PREFILL_SPARSE_DENSE_HEADS=0-4`, a
   quarter dense, τ=0.85): **REPORT-ONLY perf** — mixed-mode quality is
   UNMEASURED (the sweep pinned uniform-τ floors); its quality claim rides
   a follow-up mini-sweep or the gate's belt, never a perf cell. Do not
   read Cell D as a promotion candidate on perf alone.
4. **Cell E — the regime gate** (only if the freeze landed):
   `RIIR_PREFILL_REGIME_GATE=1` + the sidecar pair + `RIIR_OMSR_MANIFEST`
   at the v2 fixture; engagement proof = `regime_gate_counters()` decisions
   move AND every served τ carries its `verify_posture_at_len` belt green.

**Acceptance (pre-registered; promotion owner-gated):**

- **G2 (perf):** the served posture ≥ 1.02× @16K / ≥ 1.04× @32K
   verified-idle vs Cell A — the T3-v2 under-load reads (1.025×/1.049× at
   τ=0.85) are the FLOOR: idle must not read worse; a worse idle read is a
   demote signal, not a rounding note.
- **G1 (quality):** the served τ passes `verify_posture` 7/7 (0.85 and 0.9
   both do per the v2 manifest); a gate-driven lower τ carries its class
   belt green at the served length.
- **G3/G4:** env-off byte-identity (already anchored:
   `sparse_tau1_is_bit_identical_to_m32` + the T5 mode anchors) and disarm
   = no scratch alloc (structural: `sparse: None`). Re-asserted, not
   assumed, at the landed HEAD before the cells run.
- **The call:** τ=0.85 vs τ=0.9 (and any mixed/gate promotion) is the
   owner's on the GOAT — perf AND per-family quality together; demote the
   loser per the standing rule.
