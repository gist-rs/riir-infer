# Plan 611 — T7: the op-layer unification — the encoder lane's Backend trait implemented over the CubeCL layer, the A/B vs the hand-tuned lanes, the verdict

**Status:** IN FLIGHT — S1 complete (a: `bc93e70`, b: `a5688a9`),
S2 complete (`a3f8928`), S3 complete (`4e89963`), S4 complete (wiring +
G5 arm; the S4 finding RESOLVED — 016's G5 FAIL was the `gather_rows`
residency class, fixed; top-1 = 1.000000 on all three checkpoints in
every run; a residual sporadic drift wobble filed as `.issues/017`, the
cubecl G5 arm `#[ignore]`d until it closes, all Cubecl numbers
PROVISIONAL). S5–S6 remaining; the S5 A/B may MEASURE at any time but
must not PUBLISH Cubecl numbers until 017 closes.
Feeds riir-reflex `.issues/008` T7 (the campaign's last open task) and
riir-infer `.issues/998` S8 (the same task, mirrored home). The reflex-side
issue 008 remains the campaign record; this plan is the execution home.

## The question T7 answers

Issue 008 P5 says: "the hand-MSL op layer and the CubeCL op layer unify
behind the Backend trait; A/B against the chart numbers; G5 at both
postures; the loser is deleted." Four years of context compress into one
honest question: **the laya lane carries THREE hand-tuned backends (Cpu,
Metal MSL, raw-CUDA cudarc) while riir-infer-gpu carries a second op layer
(CubeCL) for the engine's decode — the same ~60% op surface implemented
twice with different reduction orders. Does ONE portable CubeCL
implementation of the encoder's `Backend` trait replace the
per-device hand lanes, or does the A/B confirm the hand lanes are the
records and the CubeCL arm's value is portability + shared vocabulary?**

The expected answer, stated up front so the work is honest: the hand lanes
win. The Metal lane is the product of a full optimization campaign
(split-K + shape rule + `ln_rows_wide` + flash-attn rungs + fold epilogues
— reflex issue 020), and the raw-CUDA prefill family is likewise
campaign-tuned. A first CubeCL arm will not beat them. T7's REAL
deliverables are therefore:

1. **The GAP kernels land in `riir-infer-gpu` regardless of the verdict**
   (mean-centered LayerNorm — the engine's norms are RMS-shaped, no
   mean-centering exists; a non-causal / bidirectional-window attention
   path — the engine's attention family is causal/KV-shaped). These enrich
   the ENGINE — the DRY payoff that survives every A/B outcome.
2. **ONE op vocabulary**: the laya `Backend` trait becomes the op-order
   home the CubeCL layer also speaks; the engine's ops become reachable
   through the same trait (`issue 008`'s "one op layer" verdict row).
3. **A measured A/B** (not an assumed one) of CubeCL vs Metal vs Cpu on
   the laya fixtures, published beside the chart, with the verdict
   recorded and the arm's fate decided from it: keep as the
   portability/CI arm, or delete.

## Prior art (checked before writing — the substrate-first pass)

- `crates/riir-infer-gpu/src/matmul_cubecl.rs` — `MatmulCubeCL`, tiled
  f32 kernel, launch shape `C[M,P] = A[M,N] × Bᵀ[P,N]` — ALREADY the
  encoder's `matmul_w` shape (weight row-major `[out, in]`).
- `norms_cubecl.rs` — `RmsNormCubeCL` + `RmsNormBatchedCubeCL` +
  qk-fused; RMS-shaped only. **GAP: no mean-centered LN.**
- `elementwise_cubecl.rs` — the Split4/Situ elementwise family;
  `cubecl_runtime.rs` carries a gelu verify kernel.
- `cpu_reference.rs` — `softmax` (names `kernels/softmax.wgsl`),
  `rmsnorm`, `compare_f32` (the tolerance comparator the parity tests
  use).
- attention: `attention_cubecl/`, `attention_causal_fused_cubecl.rs`,
  `attention_q8kv_cubecl.rs`, the qwen prefill family — ALL
  causal/KV-shaped. **GAP: no non-causal path.** BUT the laya trait's
  `attention_forward_default` IS the reference non-causal op sequence —
  a CubeCL backend that implements the PRIMITIVE ops inherits
  non-causal attention composed, correctly, for free (speed later, if
  ever).
- rope: `rope_geglu_cubecl.rs` (the rope half of a fused kernel).
- runtime: `cubecl_runtime.rs` / `context.rs` / `buffer.rs` /
  `weight_buffer_cache.rs` / `params_cache.rs` (blake3-keyed
  params-handle cache — the natural home for `warm_weight`).
- the trait: `crates/riir-infer-laya/src/laya/riir/backend.rs` — ~25 ops,
  host-slice signatures, device-residency semantics documented per op
  (`begin_pass` slot invalidation, `download_into` as the ONE read
  barrier, `copy_at` device-side ranges, `set_row_segments` for
  shape-dependent kernel choice, `supports_packed_attention` /
  `needs_window_mask` posture gates).

Vocabulary-translation check (both spellings grepped): "layer norm" /
"mean-centered" / "layernorm" → only RMS forms in the gpu crate; the
encoder's `layer_norm_nobias_into` + "NO bias tensors anywhere in the
encoder" pin live laya-side. Confirmed GAP, not a naming miss.

## Constraints (the fence, the lane law, the G-gates)

- **Fence:** `riir-infer-laya` may path-dep `../riir-infer-gpu` (in-repo,
  allowed — the fence gate's F1 containment check); both crates are this
  repo's. Zero new external deps beyond what `riir-infer-gpu` already
  carries (cubecl/wgpu — already in its tree).
- **Feature:** `laya-riir-cubecl = ["laya-riir", "dep:riir-infer-gpu"]`
  (optional path dep; the cubecl tree joins ONLY behind it). Default
  builds never resolve it. NOT in `RELEASE_FEATURES` — the release set
  is unchanged.
- **DeviceKind:** a `Cubecl` variant behind the feature (the
  `Metal`/`Cuda` pattern), `LAYA_DEVICE=cubecl`, fail-loud when the
  feature is off (the existing refusal spelling).
- **G5:** the lane's parity gate must run at the new posture
  (top-1 ≥ 0.999, drift ≤ 1e-3 vs the frozen captures) BEFORE any
  published number cites it. Expected drift source: reduction-order
  differences in matmul/softmax/LN — the packed_forward_equiv budget
  (1e-4 metal / 1e-5 cpu classes) does NOT apply across backends; the
  G5 1e-3 budget does.
- **No default promotion:** Metal stays the macOS default and CUDA the
  non-macOS default regardless of the A/B, unless the CubeCL arm wins a
  REGIME (not a cell) — an outcome this plan prices as unlikely. The
  plan's success criterion is the VERDICT, not a promotion.

## Slices

- [x] **S1 — the GAP kernel + the skeleton (the engine payoff lands
      first).** **(a) LANDED 2026-09-26 (riir-infer `bc93e70`):
      `LayerNormMeanBatchedCubeCL`** — one workgroup per row, ONE
      strided walk accumulating Σx + Σx², two unrolled 256-thread smem
      tree reduces, `var = E[x²] − μ²` (the one-pass variance form; the
      kernel header documents the order divergence from the two-pass
      CPU reference — the 1e-5/1e-4 tolerances absorb it), the
      Issue-639 params discipline. 2 parity tests green (7×1024
      identity-gamma with the zero-mean/unit-var row claims + 3×64
      non-trivial gamma); full gpu lib 203/0; clippy `-D` clean at
      cubecl_runtime + default postures. **(b) LANDED 2026-09-26
      (riir-infer `a5688a9` + the ane.rs mechanical fix `e20ca91`):
      the `CubeclBackend` skeleton** — `laya/riir/cubecl.rs` behind the
      new `laya-riir-cubecl` feature (optional in-repo path dep,
      never a default): the Metal lane's residency model (weights
      (ptr,len) permanent; chain (ptr,len,epoch) touch-stamped;
      read-modify-write uploads on miss, write-first dsts get
      `client.empty`) on CubeCL's server-managed memory;
      `begin_pass` bumps the epoch + clears the chain WITHOUT a sync
      (tasks hold their own handle refs — the engine's no-sync
      decode residency); `download_into` resolves by base pointer +
      most-recent touch, panics on host-authored slices; the trivial
      op family (add / add_bias_row / scale / relu / gelu_erf /
      glu_gelu_gate / copy_into / copy_at) dispatched over new gpu
      elementwise launchers; every S2/S3 math op panics LOUD naming
      its slice. Two measured findings baked into docs: (1) the layout
      policy PADS allocations and returns handles with
      `offset_end = Some(slack)` — live ranges read via
      `size_in_used()`, never `size()`; (2) wgpu's 32-byte
      storage-bind alignment makes byte-offset handle views unusable
      for the forward's element-arbitrary offsets, so offset ops take
      PARAMS over whole-parent binds. Parity: gpu-side kernel tests
      (5 new, full gpu lib 208/0) + `tests/cubecl_ops_smoke.rs`
      op-by-op vs Cpu (exact for non-erf ops, 2e-5 erf-bearing) + the
      residency arms (chain reuse = x+2y not x+y; begin_pass
      invalidation panics deterministically; device-side copy slabs);
      runtime label resolved at construction (`wgpu<msl>` on this
      host) and printed — the plan's backend-selection pin. Clippy
      `-D` clean at default / cubecl_runtime / laya-riir-cubecl /
      --all-features; the [[test]] row refuses without the feature.
- [x] **S2 — the matmul family. LANDED 2026-09-26 (riir-infer `a3f8928`).**
      `matmul_w` → the SHIPPED `MatmulCubeCL` (derived-dims transB — exact
      because every call site guarantees whole exact-extent parents,
      `HeadScratch::fit` then exact dims; the weight rides the permanent
      cache, `warm_weight_2d`'s slot); `matmul`/`matmul_kt`/
      `matmul_kt_heads`/`matmul_heads` → TWO new offset+head-batched tiled
      kernels (`matmul_batched_transb_off_f32`, `matmul_batched_rr_off_f32`
      — the head batch rides the DISPATCH Z axis via `CUBE_POS_Z`, one
      launch per batch, not the priced per-head loop: it is simpler than
      the loop AND saves heads−1 dispatches; whole parents bound once, head
      slabs are in-kernel offsets, the whole-batch destination ONE
      `client.empty` slot); dims+offsets ride the params buffer as
      f32-encoded usize (f32-exact guard at 2^24) per the S1b
      alignment finding; `matmul_w_accum`/`matmul_w_glu` compose through
      the trait defaults (bit-identical folds, proven behaviorally);
      `set_row_segments` stays the trait's no-op v1. Parity: 2 gpu kernel
      tests (offsets over genuinely-padded parents + batched heads, max_err
      ~2e-6; full gpu lib 210/0) + 2 backend smoke tests (the full shape
      table — m ∈ {1,4,54,128,512} × the encoder geometry classes, heads ∈
      {1,4}, offsets arms; the fold pair) — 7/7 green, max drift 1.07e-4
      vs the 1e-3 budget. Measured finding: the z-dispatch works on
      wgpu<msl> first try (no 3D-dispatch precedent existed in this
      codebase). supports_packed_attention stays FALSE until S3 completes
      the forward surface — matmuls alone cannot run a forward.
- [ ] **S3 — the head ops + attention.** softmax_rows (the softmax
      kernel; check its row-length generality at d=1024/2048 — the
      head's option rows are short, the scores rows are seq-long),
      layer_norm_nobias_into → the S1 GAP kernel, apply_rope (the
      rope half of `rope_geglu` extracted or a small standalone kernel
      — rotate-half, cos/sin tables row-sliced exactly like the trait
      doc), split_heads/merge_heads (pure permutations — a kernel each
      or host-side ON DEVICE-CONSISTENT buffers only if the bytes are
      device-current; default: kernels, never host round-trips),
      gather_rows (the embedding gather). `attention_forward`: DO NOT
      override — inherit `attention_forward_default` (the composed
      non-causal sequence over the S1–S3 primitives);
      `supports_packed_attention` → `false` v1 (the packed path then
      keeps the loop on the CubeCL posture — the agent's existing gate,
      zero new machinery); `needs_window_mask` → `true` (the default —
      the composed body consumes the tensor).
- [x] **S4 — the agent wiring + G5 at the third posture. LANDED
      2026-09-26 (riir-infer: DeviceKind::Cubecl + `LAYA_DEVICE=cubecl`
      + the fail-loud refusal; `RiirAgent::load_with_device` — the
      explicit-device constructor the A/B harness needs, one process
      many postures, no env races; reflex `06d22bd`: the
      `laya-riir-cubecl` feature forward + the [patch.crates-io] rows
      the gpu tree needs cross-workspace (patches do NOT propagate; the
      lock also needed an explicit `cargo update -p wgpu-hal` — a patch
      whose version differs from the locked one is NOT auto-applied) +
      the G5 arm in `tests/laya_riir_parity.rs` (the body extracted into
      `g5_run(builder)`, the cubecl arm loads via the explicit
      constructor) + the harness DeviceKind arm. **THE G5 GATE FAILS AT
      THIS POSTURE** — top-1 0.33–0.46 vs 0.999, prob drift 5e-1 —
      - [x] **S4 — the agent wiring + the G5 arm (RESOLVED finding).**
      `DeviceKind::Cubecl` + `LAYA_DEVICE=cubecl` + fail-loud refusal +
      `RiirAgent::load_with_device` (the explicit constructor the A/B
      needs: one process, many postures). The reflex side landed the
      feature forward + the cross-workspace `[patch.crates-io]` rows +
      the G5 cubecl arm (reflex `bcb1c54`).
      THE 016 FINDING, RESOLVED 2026-09-26 (`riir-infer-m3-t7b`): the G5
      FAIL (top-1 0.33–0.46) was the **`gather_rows` residency class** —
      the backend bound the head's marker-gather SOURCE (the hidden
      state, an activation with stale host bytes) through the PERMANENT
      weight cache, uploading the stale zeros once; every marker row read
      zeros and all scorer logits collapsed to bias. The Metal lane's own
      `gather_rows` comment named the chain class correctly all along.
      Fix: the source binds `chain_buf`; regression arm
      `gather_rows_device_written` in `cubecl_ops_smoke`. Result: top-1
      1.000000 on all three checkpoints, EVERY run. A second fix landed
      from the issue-016 hidden probe: the S1a one-pass LayerNorm
      (`E[x²] − μ²`) cancellation at deep layers (residual ±4000) priced
      2.6e-6 → 7.7e-5 relative drift with depth — the kernel is now
      TWO-PASS (the CPU lane's exact form, `inv_dim` the candle scale);
      drift floor 3.64e-4 → 1.64e-4 at the encoder output. The probe
      (`Encoder::forward_probe` + `tests/cubecl_encoder_probe`) stays as
      the issue-017 instrument. RESIDUAL: a sporadic per-forward drift
      wobble (top-1 never flips; the outlier moves between checkpoints
      and runs; floor 3.7e-6/2.8e-5/6.4e-5 with 10–1000× excursions) —
      filed `.issues/017_cubecl_drift_wobble.md`; the reflex G5 cubecl
      arm is `#[ignore]`d until it closes. 016 removed (resolved; this
      section + 017 carry the record).]
- [ ] **S5 — the A/B harness + the verdict.**
      `tests/backend_ab.rs` (measurement-only, `#[ignore]`, the
      fold-A/B discipline): fixed fixtures (the G5 english + typed
      capture sequences + the 5-q packed shape), position-balanced
      interleave of Metal vs CubeCL vs Cpu row p50s (one process,
      three agents; the SAME fixtures drive the chart's torch row for
      the published comparison), preflight-gated in the run line, box
      state quoted. The record lands in
      `.benchmarks/NNN_t7_backend_ab/` with the chart comparison and
      the VERDICT paragraph: per-regime winner, the portability arm's
      keep/delete call, the GAP kernels' engine-side status (they stay
      regardless).
- [ ] **S6 — closure.** The verdict written back into riir-reflex
      `.issues/008` T7 + `.issues/998` S8 (mirrored), the loser
      deleted IF the verdict says so (expected: no deletion — the arm
      stays opt-in behind its feature as the portability/CI lane, or
      is removed wholesale if it is both slower AND unmaintained; the
      deletion criterion is written in S5's record BEFORE the numbers
      are read, so the call cannot be argued backwards), AGENTS.md +
      README feature tables updated both repos, HISTORY entries.

## Risks / priced unknowns

- **Device residency is the hard 20%.** The MSL lane's whole bug history
  (stale slots, torn reads, the layer-0 copy bug, the slab aliasing
  class) is residency management, not kernel math. CubeCL's runtime
  owns buffers differently (client + handles, not slot-by-(ptr,len));
  the Backend trait's host-slice signatures are slot-shaped. S1's
  skeleton must decide the mapping FIRST (a `HashMap<*const f32,
  BufferHandle>` keyed like the params_cache? a slot table mirrored
  from the Metal lane?) — that decision is the plan's biggest unknown
  and S1's exit criterion is exactly this: `begin_pass`/`download_into`
  /`copy_at` semantics proven on the smoke geometry before any math.
- **wgpu on macOS** (CubeCL's Metal runtime) vs hand-MSL: the
  comparison is fair only if the CubeCL arm gets a real Metal backend
  (wgpu→Metal), not a Vulkan-on-MoltenVK fallback — the context
  selection must be pinned in S1 and printed in every A/B line.
- **The head defer posture** assumes `download_into` barriers compose
  — true on the trait, but the CubeCL backend must keep reads ordered
  per slot; S4's packed gate covers it.
- **Effort floor:** S1–S4 is the price of admission (~the T4-move
  class); the A/B is cheap once they land. If S1 stalls on the
  residency mapping, the GAP-kernel half (S1a) still lands for the
  engine — the plan degrades gracefully by construction.

## What this plan deliberately does NOT do

- No shape-aware split rules, no fused attention kernel, no f16 — the
  hand lanes' campaign territory. The CubeCL arm is ONE clean portable
  implementation; if it approaches the hand lanes' numbers, THAT is
  the headline finding, not a tuning race.
- No engine call-site rewrite: the engine keeps its direct op calls;
  the trait is the laya lane's vocabulary that the CubeCL layer ALSO
  speaks (S1's kernels are engine-side and callable both ways).
- No promotion of the feature to default, and no `RELEASE_FEATURES`
  change.

## GOAT / gate mapping

- G1 (correctness): op parity (S1–S3) + G5 at the third posture (S4)
  before any A/B number is quoted.
- G2 (perf): the S5 A/B — measurement, not promotion.
- G3 (no regression): the feature is opt-in; default builds never
  resolve the gpu dep (the gate: `cargo check -p riir-infer-laya` at
  default features stays byte-comparable; the fence gate stays green).
- G4 (alloc): the backend's host-side path allocates only at
  `warm`/`begin_pass` (the slot table), never per op — asserted in the
  S1 smoke where the Metal lane asserts its own.
