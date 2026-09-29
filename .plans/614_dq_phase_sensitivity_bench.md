# Plan 614 — the DQ phase-sensitivity bench (Issue 026 T1–T5): the 2×2 activation-quant phase matrix on the league artifact

**Status:** OPEN — T0 CLOSED (round 3 **AGREE**, session c65b0114; rounds 1+2
REVISE incorporated). The pre-registration freeze commit landed (`d3300f2`).
T1+T2 DONE (both lanes' kernels + injections; G-i3 green on Metal + CUDA;
the D1 fallback lane is operative — no non-int8 prefill exists). T3 (the
runner) is the remaining build; then T5 (the 4090 run) + T6 (close-out).

Master: `.research/004_DQ_Disaggregated_Quantization.md` (arXiv:2609.26333 §2.2) · Issue `.issues/026_phase_sensitivity_quant_bench.md`. The issue's own gate: "Runs AFTER the 2×2 matrix definition review" — T0 below IS that review.

## SCOPE LABEL (round-1 fix: this is a TRANSFER test, not a replication)

This instrument measures the **transfer of the §2.2 sensitivity DIRECTION to
activation-only PTQ on a ternary PQ2_0 artifact**. It is NOT a replication of the
paper's protocol, which (a) quantizes weights+activations together to NVFP4 after
quantization-aware distillation from a BF16 base, and (b) reports family means over
0.6B–12B models. The paper's magnitudes are never quoted as expected values here.
Reference numbers pre-registered instead: decode-heavy asymmetry 2–4× (their §2.2
mean); **prefill-heavy reversal 1.1–4.1× holding on only 7 of 8 models** — a
prefill-heavy miss carries LESS weight than a decode-heavy miss in the source
itself.

## D1 — The matrix (round-1 fixes: symmetric base, phase boundary)

One weight artifact (`Ternary-Bonsai-2-27B-PQ2_0.gguf`), four cells differing ONLY
in where fake-quant passes run:

| cell | prefill activations | decode activations |
|---|---|---|
| `base` | f16-class (kernel-native, NON-int8 path) | f16-class (kernel-native) |
| `pf_aq` | fake-quant(grid) per linear input | f16-class |
| `dec_aq` | f16-class | fake-quant(grid) per linear input |
| `both_aq` | fake-quant(grid) | fake-quant(grid) |

- **Symmetric base (round-1 defect fix; round-2 conditions accepted):** the
  shipping prefill lane dynamically q8-quantizes activations (int8 CMMA). ALL
  cells therefore run prefill on the **non-int8 GEMM path**
  (`set_prefill_use_cmma_i8(false)`-class arm), so `base` is A16/A16 and Δacc
  measures full grid damage in both phases. Disclosed (round-2): **base is not
  the shipping configuration** — shipping prefill is A8, so a Δpf measured from
  A16 is an UPPER BOUND on shipping-visible prefill damage. T2's FIRST step is a
  NUMERIC acceptance predicate (round-2 fix — not "sane logits"): top-1
  agreement with the shipping A8 prefill ≥ **0.80** on 8 fixed 1024-token
  prompts AND the base arith-CoT accuracy inside the admissibility window
  (0.25–0.95). **The lane choice is LOCKED before the first accuracy cell and
  recorded in the bench record** (never chosen after seeing accuracy numbers).
  **Fallback** (only on predicate failure): shipping A8 prefill in every cell +
  a `dec_a8` control cell that must satisfy `|Δacc(dec_a8)| ≤ 2 items`
  (pre-registered), else the run is INSTRUMENT-FAIL.
- **Phase boundary (round-1 fix):** prefill processes ALL prompt tokens; its final
  logits produce generated token 1. The decode lane takes over from generated
  token 2 onward. Identical in every cell (no bias), but stated: in `dec_aq` the
  FIRST generated token is produced by the (clean) prefill lane's logits. On NIAH
  with ≤32 output tokens that token carries a large score share — disclosed per
  task in the record.
- `Δacc(phase) = acc(base) − acc(phase cell)`; `R = Δacc(dec_aq)/Δacc(pf_aq)`
  REPORTED, never tested (round-1 fix — see D4 statistics).

### Grids (round-2 scope note: A2 is the asymmetric 4-level ACTIVATION grid — 027-T0 SHAPE; it gates nothing in 027)

- **A2 = the asymmetric 4-level ACTIVATION grid (027-T0 shape)**, applied
  dynamically to activations — round-2 scope note: 027 itself is a weight-encoder
  lane whose gain site is the dense-GGUF arm and which does NOT consume the
  ternary artifact, so this grid gates nothing in 027; it is kept because it is
  the honest 2-bit-class asymmetric grid, not a strawman: per-128 block,
  `d = sign(argmax|a|) · f16(amax/2)` — **tie-break: lowest index wins** (round-2
  fix; parallel reductions on the two backends can otherwise pick different
  signs, and the tie case goes into G-i3 explicitly) — codes on `d·{−1, 0, +1, +2}`:
  `q = clamp(round(a/d), −1, +2)`, `aq = q·d`.
- **A4 (primary tier)**: per-32 affine 16-level — `min`,`max` in f32 per block;
  `s = f16((max−min)/15)`; `q = clamp(round((a−min)/s), 0, 15)`; `aq = q·s + min`
  (min in f32). No secondary scale quantization — stated divergence from q4_k's
  6-bit scale encoding (the dynamic-quant kernel convention).
- Bit-exact spec (G-i3's contract): rounding = **half-away-from-zero** (RTN
  encoder convention); all-zero block → `aq = 0` (d=0 guarded, never NaN); A4
  clamp is load-bearing (f16 rounding of `s` downward can push q to 16); blocks
  run **per token along the reduction dimension** of the consuming GEMM (per
  (token, 128-block) / (token, 32-block) independently — no statistics shared
  across tokens; the prefill [p×n] buffer and the decode [n] buffer block
  identically).

## D2 — Injection sites + escape set (round-1 fix: out-proj sites ADDED)

Per layer, per armed phase — on-device quantize→dequantize, in place, at FIVE
sites (basis recorded; folded model → the projections consume ROTATED copies):

| # | site | basis | consumed by |
|---|---|---|---|
| 1 | layer-input norm staging (`rot_scratch` after stage) | rotated | GDN qkv/z, attn q/kv |
| 2 | FFN-input norm staging (`rot_scratch` re-staged post mid-norm) | rotated | gate_up |
| 3 | `ffn_hidden` after the in-place fold rotation | rotated | down proj |
| 4 | attention `o_in` (attention output pre-out-proj) | primal | attn out proj |
| 5 | GDN `out_in` (lnorm output pre-out-proj) | primal | GDN out proj |

The primal `norm_x` is NEVER quantized (the in_proj_a/b dense escape set reads it).
Escape set (never quantized, both phases — same set both phases so no directional
bias; Issue-980 + paper agreement): `in_proj_a`/`in_proj_b`, embeddings, `lm_head`,
recurrence dynamics (`a_log`/`dt_bias`/conv1d/norms). If implementation finds a
site structurally fused beyond clean injection (e.g. inside a captured epilogue),
the site is dropped WITH the round-1 rule: any axis whose direction depends on the
missing site reads INCONCLUSIVE-v1 (no gate), never a recorded negative — and the
drop is disclosed in the record before the first accuracy cell.

## D3 — One artifact, two grid tiers (round-1 fix: gate scope)

Both tiers on the SAME ternary artifact — R is computed within a tier, so a single
weight artifact strengthens internal validity (only the grid changes). External
validity to dense weights is ZERO (ternary weight noise × activation noise
interacts; the artifact may have been trained with activation quantization) —
recorded. **Gate scope (round-2 correction): NEITHER 027 NOR 028 is gated by
v1.** 027 is a weight-encoder lane whose own gain site is the dense-GGUF arm
(the ternary artifact is explicitly NOT its consumer) and which carries no
phase premise; 028's premise is phase-specific WEIGHTS, which this matrix never
varies. What v1 DOES gate: Issue 026's own T5 standing rule (every future weight
format publishes its per-phase R before default promotion) and the confirmation
of the **never-activation-quantize-decode** serving posture. The dense-q4_k
paired lane (the 4090's `F:/models/qwen38-27b-dbirks-Q4_K_M.gguf` via
`Qwen38DenseForward`) is a recorded follow-up with its own injection surface.

## D4 — Task axes, statistics, admissibility (round-1 fixes throughout)

### Corpora (frozen before the first cell — BLAKE3-committed)

- **Decode-heavy (arith-CoT):** 48 synthesized deterministic items (multi-digit
  arithmetic chains; fixed 4-shot completion-format prefix — NO chat template, NO
  thinking tags — plain completion so Qwen3.8 `<think>` cannot eat the budget);
  prompt ≈ 60 tok; greedy ≤ 256 tok; EOS or cap stops. Parse rule: the LAST
  integer after `####` in the full output; commas stripped; **truncation (cap hit,
  no `####`) counts WRONG, no fallback**. Exact-match scoring.
- **Prefill-heavy (multi-needle NIAH):** 8 needles planted at spread depths per
  prompt, haystack from deterministic `chat_probe` pages (in-repo corpus), lengths
  {4096, 8192, 16384}; **32 prompts per length** (96 items; round-1 fix from 8);
  **ONE needle queried per prompt** (the kv_recall pattern — never "list all 8";
  8 values in ≤32 tokens is truncation-as-damage); query last; greedy ≤ 32 tok;
  **scoring: the FIRST needle-value substring in the output wins** (round-2 fix —
  if it is the target, correct; a distractor value or nothing, wrong).
- Determinism: seeded generators, no RNG; the generated corpus + haystack slices
  hashed (BLAKE3) and printed BEFORE the first accuracy cell; the hash, the GGUF's
  BLAKE3, the commit SHA, and box state go in the record.

### Statistics (round-1 fix: paired sign test, R reported only)

- The pre-registered test is on the **paired difference `Δdec − Δpf` per item**
  (same items in every cell): paired bootstrap 95% CI (10k resamples, seeded);
  **the CI's sign must hold** (decode-heavy: CI > 0; prefill-heavy: CI < 0).
  R is REPORTED with its bootstrap CI, never asserted. Direction only — magnitude
  never gated.
- **Minimum denominator:** if the smaller phase's Δ ≤ 2 items on an axis, R is
  "undefined" there and ONLY the paired sign is reported.
- **Pooling rule (round-2 fix):** primary NIAH verdict POOLED over **admissible
  lengths only** (a dead 16K arena never poisons the pool, and never gets
  dropped after the fact — inadmissibility is decided by the pre-registered
  window before any verdict); per-length tables always reported. **"Split"
  defined (round-2):** some admissible length whose OWN CI excludes 0 in the
  direction OPPOSITE to the pooled result → the verdict downgrades to
  "split — recorded, no gate" (point-estimate disagreements alone are noise,
  not splits).
- **Admissibility window (round-1 fix; round-2 keying):** an axis (per LENGTH —
  `acc(base)` has no grid) is admissible only if `0.25 ≤ acc(base) ≤ 0.95`.
  Outside → **INADMISSIBLE** (arena dead or saturated), not negative. Chance
  floors for scoring context: arith ≈ 0.05, NIAH single-of-8 ≈ 0.125.
- **Saturation threshold (round-2 fix — fires only when BOTH phases collapse):**
  tier-AXIS is SATURATED ⟺ `max(acc(pf_aq), acc(dec_aq)) ≤ chance + 0.05` OR
  `min(Δpf, Δdec) ≥ (acc(base) − chance) − 0.05`. The likely A2 outcome
  ("decode collapses, prefill survives") is a clean directional result, NOT
  saturation. SATURATED = instrument floor; recorded; gates nothing.
- **Exposure confound, disclosed:** decode:prefill token ratio recorded per task
  (arith ≈ 6:1 decode-heavy; NIAH ≈ 1:256 prefill-heavy) so "R tracks exposure"
  stays distinguishable from phase-specific sensitivity.
- `both_aq` is **report-only** (round-1 fix): its Δ is tabulated beside
  `max(Δpf, Δdec)` for the record; nothing is asserted on it.
- **Three-way outcome per axis (round-2 fix — the most likely A4 outcome at N=48
  is a CI straddling 0, which was previously undefined):** **HIT** = the paired
  bootstrap 95% CI on `Δdec − Δpf` excludes 0 in the predicted direction.
  **REVERSED** = the CI excludes 0 in the OPPOSITE direction — the ONLY recorded
  negative. **NULL** = the CI includes 0 — underpowered; recorded, no gate.
- **Label precedence (round-2 fix):** INSTRUMENT-FAIL > INADMISSIBLE > SATURATED >
  HIT/REVERSED/NULL — one label per (task, grid), first match in that order.

## D5 — Lane + box rules

- 4090-only (folded prefill is CUDA-only); prefill = cudarc whole-prefill lane,
  **`RIIR_PREFILL_CUDA_GRAPHS=0` in ALL FOUR cells** (round-1 fix); decode =
  CubeCL `forward_from_x` (knob consulted per dispatch). Chunked prefill for
  lengths > 4096 (GDN/KV state carries across chunks; assert `block_size`).
- Feature `dq_phase_bench` (default-off; G4 law — hot path untouched when off,
  cfg-gated blocks only). Knobs: `RIIR_DQ_FQ_PHASE=off|prefill|decode|both`,
  `RIIR_DQ_FQ_GRID=a2|a4` (AtomicBool + setter house pattern).
- **Launch counters PER PHASE (round-1 fix — phase-leak detection;
  round-2 corrected arithmetic):** `fq_prefill_launches` and `fq_decode_launches`
  are separate. Expected counts are EXACT: prefill = **(3 sites × 64 layers) +
  (site 4 × 16 attention layers) + (site 5 × 48 GDN layers) = 256 per chunk**;
  decode = **256 × (n_generated − 1) per item** (generated token 1 comes from
  the prefill lane's logits — D1 phase boundary; `n_generated` includes the EOS
  token when the loop stops on EOS, and every decode-lane step counts, including
  the one that produced EOS). The per-cell assertion is
  `counter == Σ_items 256·(len_i − 1)` computed from each item's ACTUAL
  generation length. `pf_aq` → decode counter EXACTLY 0 AND prefill counter ==
  expected; `dec_aq` → the reverse. A leak or count mismatch is INSTRUMENT-FAIL,
  never a quiet asymmetry — and the expected-count formula is frozen here, not
  edited after seeing counters.
- GPU exclusivity: the Issue-833 probe (vendored `gpu_exclusivity` module) —
  REFUSE under a resident compute app. Box-state provenance line in the record;
  AC power; solo run.

## D6 — Instrument gates + outcome-to-decision table (round-1 fixes)

- **G-i1 baseline identity (strengthened):** fake-quant OFF ≡ unmodified path —
  logits FNV identical on a fixed prompt, comparing the FEATURE-ON/knob-off build
  against a FEATURE-OFF build, both with graphs disabled.
- **G-i2 positive control + leak detection:** fake-quant ON changes logits (FNV
  differs); per-phase launch counters match the exact expected counts per cell.
- **G-i3 kernel correctness, per lane, one reference:** the CUDA kernel AND the
  CubeCL kernel each equal the SAME host reference bit-exactly on seeded tensors
  (both grids; includes zero-block, clamp-edge, ragged-tail, AND the A2 argmax
  TIE case — round-2 fix) — proving the two lanes' fake-quant agree with each
  other.
- **G-i4 determinism:** same cell twice → byte-identical greedy streams.
- **Stop rule:** after the first accuracy cell, no item/grid/protocol change; a
  cell re-runs ONLY on an instrument-gate failure (disclosed in the record).
- **Pre-registration freeze (round-2 fix):** THIS PLAN is committed before the
  first accuracy cell; the bench record quotes the commit SHA and this file's
  BLAKE3 beside the corpus and GGUF hashes. An uncommitted pre-registration can
  be edited after the results — committed is the only frozen state.

**Outcome → decision table (pre-registered; round-2 scope correction — NEITHER
027 NOR 028 is gated by v1):**

| outcome | standing consequence |
|---|---|
| A4 decode-heavy HIT | the never-activation-quantize-decode posture is CONFIRMED on our artifact; Issue 026's T5 standing rule is armed (future weight formats publish per-phase R) |
| A4 decode-heavy REVERSED | the ONLY recorded negative: activation-quant damage is prefill-concentrated on our formats — the §2.2 transfer fails; recorded in Research 004's lineage |
| A4 decode-heavy NULL | underpowered at N=48 — recorded, no gate; a follow-up N-raise is the recorded remedy, never a protocol edit |
| A4 prefill-heavy HIT | the phase-specialization axis reads real on our formats; 028's OWN T4 (its PTQ-vs-QADD recovery measurement — not Issue 026's T4) gets its instrument context |
| A4 prefill-heavy REVERSED | recorded negative for the reversal on our formats (the paper's own source held it on only 7/8 models — stated in the SCOPE LABEL) |
| A4 prefill-heavy NULL | recorded, no gate |
| A2 SATURATED | instrument floor at 2-bit — recorded; the likely "decode collapses, prefill survives" pattern is NOT this (it is a directional result) |
| A2 HIT/REVERSED/NULL | recorded as the 2-bit-tier read; secondary to A4 |
| any axis INADMISSIBLE | no gate from that axis; the window verdict is decided by base accuracy alone, before any phase comparison |
| INSTRUMENT-FAIL (any G-i gate or count mismatch) | no accuracy claim; cell re-run only per the stop rule |
| "split" (per the D4 definition) | recorded, no gate |

## Tasks

- [x] T0 — definition review: rounds 1+2 REVISE incorporated; **round 3 AGREE**
      (session c65b0114 — conditions applied: Status line, Grids heading,
      commit-before-T2). **T0's completion commit IS the pre-registration freeze
      (D6).**
- [x] T1 — `dq_fakequant` module: grid spec (A2 027-T0 + A4 affine), host reference
      (the G-i3 oracle), knob statics + setters + PER-PHASE launch counters.
      **DONE (0edf36c, e62f9d0, d32e29e lineage)**: pure spec + per-row host
      oracle (10 lib tests incl. the exhaustive f16-vs-`half` equality); CubeCL
      decode-lane kernels (bit-exact f32→f16→f32 RNE emulation in integer bit
      ops, half-away rounding — the GPU native round is half-to-EVEN —,
      full-workgroup smem init for ragged tails, lowest-index tie-break);
      CUDA prefill-lane kernels (nvrtc sm_89) as a dedicated `DqFqKernels`
      module. G-i3 GREEN on BOTH lanes and BOTH backends (Metal M3 + CUDA
      4090, 5/5 tests). Six kernel/oracle defects caught by the gates in
      landing (flat-vs-per-row host blocking — a spec violation —, uninit
      smem slots, half-even GPU rounding, f16 subnormal k=0 clobber,
      shift-overflow guard, an A4 division slip).
- [x] T2 — kernels + injections: DONE. Decode — the five `forward_from_x`/
      layer-fn choke points (rot staging ×2, ffn_hidden, attn_out,
      recurrent_out — all post-rotation, the folded basis). Prefill — the
      five `whole_prefill_inner` sites (normx ×2 with the GDN escape-set
      reorder under the feature, hid_b, attn_out_b, rec_b) via the
      lazily-compiled DqFqKernels. All `#[cfg(feature = "dq_phase_bench")]`;
      the 4090 build is green with the full feature set.
      ⚠ D1's "non-int8 prefill path" was found NOT TO EXIST (the cudarc lane
      ALWAYS activation-quantizes — both the q8 and the hi/lo arms are int8):
      **the pre-registered fallback is OPERATIVE** (shipping A8 prefill in
      every cell + the dec_a8 control cell). The lane-acceptance predicate
      (top-1 agreement ≥ 0.80 vs shipping + base-acc window) still runs
      first, per D1.
- [ ] T3 — runner bin `dq_phase_matrix` (feature-gated): corpus generators +
      BLAKE3 freeze, 4-cell × 2-grid (+dec_a8 control) driver, greedy generation
      + parsing + scoring, paired bootstrap CIs, R tables, admissibility/
      saturation classification, per-phase counter assertions, G-i1/G-i2/G-i4,
      GPU-exclusivity probe, JSON+MD out.
- [-] T4 — local verification: M3 half DONE (compile + clippy + lib tests +
      CubeCL G-i3 green); 4090 half DONE (full-feature build green + G-i3 5/5
      incl. the CUDA-lane arm). REMAINS: 4090 clippy + the runner's compile
      once T3 lands.
- [ ] T5 — the 4090 run (solo, GPU-exclusive, AC): both grids × both axes; write
      `.benchmarks/023_dq_phase_matrix.md` (live max is 022 — the `.highwater` 15
      is stale) with box state, exposure ratios, per-family/per-length tables, R
      + CI table, outcome classification per the D6 table, and the standing
      promotion rule (any future weight format publishes its per-phase R before
      default promotion).
- [ ] T6 — Issue 026 T1–T5 marks + verdict; HISTORY row; `.benchmarks/.highwater`
      repair (15 → 023) in the same commit.

## Budget

1–2 sessions (the issue's estimate). The 4090 run ≈ 2–3 h wall (48×256-tok arith
greedy × 8 cells + 96 NIAH prefills × 8 cells) — detached, solo.

## Pre-registered honest outcomes

- A4 confirms direction, A2 saturates → instrument stands; the 2-bit tier records
  its floor. Decode-heavy sign is the load-bearing assertion (the paper's own
  prefill-heavy reversal held on only 7/8 models).
- A4 decode-heavy misses → recorded negative; 027's phase premise collapses.
- Any instrument-gate failure → no accuracy claim, cell re-run only per the stop rule.
