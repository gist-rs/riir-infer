# Issue 016 — plan 611 S4: the Cubecl-posture G5 FAILS deterministically (top-1 0.33–0.46 vs 0.999)

**Status:** OPEN — the lane stays opt-in behind `laya-riir-cubecl`; every
number it produces is PROVISIONAL until this closes. Not a blocker for
S5's A/B harness *design*, but no Cubecl A/B number may be published
before the verdict flips.

Filed 2026-09-26 · session `riir-infer-m3-t7` · plan 611 S4
(reflex `.issues/008` T7).

## The observation

`g5_parity_cubecl_posture` (reflex `tests/laya_riir_parity.rs`, the G5
capture replayed through `RiirAgent::load_with_device(DeviceKind::Cubecl)`):

```
english:         26 forwards · top-1 12/26 = 0.461538 · prob drift 7.373e-1
typed-decisions: 26 forwards · top-1 12/26 = 0.461538 · prob drift 5.195e-1
multilingual:    36 forwards · top-1 12/36 = 0.333333 · prob drift 8.875e-1
gates:           top-1 ≥ 0.999 · drift ≤ 1e-3 → FAIL on all three
```

Exactly **12 forwards agree per checkpoint** while `act_probabilities`
drift is **0.000e0 everywhere** — `act_of` (a small fresh-buffer
matmul_w → add_bias_row → gelu → matmul_w → download chain) is EXACT,
so the small-chain machinery (upload → op → download, weights cache,
params cache) is sound. The divergence is in the ENCODER path or the
encoder→head handoff.

## What it is NOT (measured, not assumed)

1. **Not the wgpu dispatch-cap violation.** The first run also tripped
   `[issue 994] wgpu uncaptured error: Validation Error — dispatch
   group size [65536, 1, 1] > 65535` — the scores-mask broadcast at
   seq = 1024 (16·1024² = 2²⁴ elements → ceil/256 = 65536 workgroups).
   FIXED (riir-infer `d4ab87a`: chunked launch, 32768-cube chunks with a
   kernel-visible base offset). After the fix the error is gone AND THE
   NUMBERS ARE BYTE-IDENTICAL (12/26, same drifts to the digit) — the
   forward was never pool-poisoned; it was deterministically wrong all
   along. The two symptoms were independent.
2. **Not reduction-order drift.** Drift of 0.5–0.9 in probability space
   is structural, and the G5-class budget is 1e-3. Every op passed its
   isolated parity test (see below).
3. **Not environment/feature-unification.** The op smoke (9/9) passes in
   the same target dir; the failure reproduces in release AND debug with
   identical numbers.

## What IS covered green (op-by-op, `tests/cubecl_ops_smoke.rs` 9/9)

All matmuls (incl. real geometry 1×1024×3072 / 128×2048×512, offsets,
z-batched heads), the trait-default folds, add/add_bias_row/scale/relu/
gelu/glu, copy/copy_at, the composed `attention_forward_default` at a
sliding-window shape (8.9e-8), rope (1.2e-7), split/merge/gather
(exact), LN (S1a kernel), softmax, mask broadcast. Residency arms
(chain reuse, begin_pass invalidation, device-side copies) green.

→ The bug is in COMPOSITION at real geometry (28 layers, heads 16,
d 1024, i_sz 2624): something about slot reuse across layers, the
encoder→head handoff, or an interaction no isolated op test exercises.

## Repro

```bash
# live repos (needs the sibling reflex tree compiling)
git -C /Users/katopz/git/riir-reflex worktree add /Users/katopz/git/reflex.w-g5 develop
CARGO_TARGET_DIR=/tmp/t7_reflex cargo test --manifest-path \
  /Users/katopz/git/reflex.w-g5/Cargo.toml -p riir-reflex \
  --features laya-riir-cubecl --test laya_riir_parity \
  g5_parity_cubecl_posture -- --nocapture --test-threads=1
```

Weights cached under `~/.cache/riir-reflex/laya/` (no download needed).
Geometry (encoder_config.json): hidden 1024 / inter 2624 / heads 16 /
hd 64 / 28 layers / vocab 50368 (english+typed); 768/1152/12/22/256000
(multilingual).

## Next bisect levers (in yield order)

1. **Hidden probe**: run one fixture through the Cpu agent and the
   Cubecl agent and diff the ENCODER OUTPUT (the head is deterministic
   given hidden — act_of exactness proves the head machinery). The
   encoder return value is the cheapest intermediate to expose (a test
   can call `Encoder::forward_packed` directly with both backends — it
   is `pub` and takes `&dyn Backend`).
2. **Layer-count bisect**: the fixture states run through 28 layers;
   check whether agreement decays with depth (a per-layer corruption
   compounding) vs corrupting at one layer (a shape/edge bug that first
   bites at a specific weight shape).
3. **The 12-that-agree**: identify which 12 fixtures agree per
   checkpoint — hypothesis: the SHORT sequences agree (length-dependent
   partial-write or edge-tile bug). Correlate agreement with seq.
4. **Slot-reuse stress**: a smoke arm that runs the ENCODER op sequence
   twice over the same slot Vecs in one epoch (layer 2 reuse of layer
   1's slots is the one composition shape no test covers).

## Session marker

`Session: riir-infer-m3-t7, 2026-09-26` — commits `4e89963` (S3),
`06d22bd`-mirror `d4ab87a` (the mask fix), reflex `06d22bd` (the G5 arm).
