# Plan 611 — T7: the op-layer unification — the encoder lane's Backend trait implemented over the CubeCL layer, the A/B vs the hand-tuned lanes, the verdict

**Status:** PLANNED — slices S1–S6, each independently landable; nothing started.
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

- [ ] **S1 — the GAP kernel + the skeleton (the engine payoff lands
      first).** (a) `layer_norm_mean_cubecl` in `riir-infer-gpu`: the
      mean-centered, bias-free LayerNorm kernel (`y = (x − μ) / σ ⊙ w`,
      row reduction Welford or two-pass — the CPU reference's exact
      reduction order documented in the kernel header; parity test vs
      `cpu_reference` at 1e-5, the norms test class) + a batched
      variant. (b) `CubeclBackend` skeleton in the laya lane
      (`laya/riir/cubecl.rs`, feature-gated): `name()` = `"cubecl"`,
      the trivial ops first (add / add_bias_row / scale / relu /
      gelu_erf / glu_gelu_gate / copy_into / copy_at) over
      `elementwise`-family launches, `begin_pass`/`download_into` over
      the runtime client, `warm_weight` → `params_cache` handles.
      Parity: every op vs the CPU lane on the smoke geometry
      (`metal_ops_smoke.rs`'s pattern — op-by-op, not forward-level).
- [ ] **S2 — the matmul family.** `matmul_w` → `MatmulCubeCL` (the
      shapes already match); `matmul` (a[m×k] @ b[k×n], both row-major
      — a transposed-B variant or a second kernel; pick by reading the
      tiled kernel's B indexing, never by transposing on the host);
      `matmul_kt` / `matmul_kt_heads` / `matmul_heads` (batched-over-
      heads; v1 may loop per head — correctness first, one dispatch
      later if the A/B shows it matters); `set_row_segments` no-op v1
      (one kernel for every shape — the honest v1; split-K-style
      shape rules are the hand lanes' territory and explicitly NOT a
      v1 goal). Parity per op per shape class (m ∈ {1, 4, 54, 128,
      512}, k/n the encoder geometry) vs CPU at the G5 drift budget,
      shape table in the test.
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
- [ ] **S4 — the agent wiring + G5 at the third posture.**
      `DeviceKind::Cubecl` + the env spelling + the fail-loud refusal
      when the feature is off; `RiirAgent::load`'s backend match arm;
      the head defer posture audited on the new backend (the drain
      batching is backend-agnostic by construction — `reads_of`/`act_of`
      go through `download_into` — but the packed gate runs at the
      CubeCL posture to prove it); `tests/laya_riir_parity` extended to
      run the G5 capture at `LAYA_DEVICE=cubecl` (feature-gated arm,
      loud-skip without the feature, the cache_reuse pattern).
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
