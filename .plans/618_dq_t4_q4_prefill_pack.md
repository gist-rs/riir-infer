# Plan 618 — Issue 028 T4 unblock: the Q4_K prefill pack (S1) + the q4 weight arm (S2) + the dual-PTQ measurement (S3)

**Status:** IN PROGRESS — S1+S2 landed (the pack exists and the container loads + runs it); the cudarc q4 GPU GEMV (S2 item 5) LANDED 2026-10-06; S3 (the measurement) remains.

Master: `.issues/028_dual_ptq_disaggregated_serving.md` (T4) + `.research/004_DQ_Disaggregated_Quantization.md`.
Consumer: `src/disaggregated.rs` (`load_pair` / `load_single_file` / `PhaseHandoff`, T1+T2+T3 landed `09d0dd5`).

## Why T4 is unblockable now (the science, settled this pass)

T4 was filed "BLOCKED on artifact: a q4-class prefill pack of the bonsai must be
QUANTIZED first (… on the unfolded source)". The unfolded-f16 source does not
exist on this box — **and it does not need to**: the requant source is the
PQ2_0 pack itself, and that is scientifically clean because the bonsai is a
**trained-ternary** checkpoint whose PQ2_0 payload is essentially exact:

- Issue 022's measured datum: re-round-tripping the league model's blocks
  through f16 reads damage ≈ 3.7–3.9e-8 (the f16 arm, `twt_ternarize_probe`)
  — i.e. the stored values are small-integer ternary times f16 group scales,
  not arbitrary 2-bit-quantized reals. ε(PQ2_0→f32) ≈ 0.
- Therefore `dequant(PQ2_0) → quantize_row_q4_k` carries **only ε(Q4_K)** —
  the same error a q4 pack quantized from an f16 original would carry. No
  double-quantization pollution; the dual-PTQ comparison (decode=PQ2_0 vs
  prefill=Q4_K) has exactly the paper's shape.
- The requantizer gates this premise on the artifact itself (S1's
  ternary-exactness check): if the residual-vs-ternary check ever reads
  non-zero beyond f16 epsilon, the pack production refuses and the premise
  is re-examined.

## Measured inventory (this pass, `scripts/probe_bonsai_tensors.py`)

`Ternary-Bonsai-2-27B-PQ2_0.gguf`: 851 tensors, 49 metadata keys, no
`qwen35.nextn_predict_layers` (identity block table tiles all 64 layers —
the collapse writer's stale-nextn check passes untouched).

| class | id | count | policy |
|---|---|---|---|
| block projections (`attn_qkv`, `attn_gate`, `ffn_*`, `ssm_out`, attn `q/k/v/o`) | 142 | 400 | **dequant → `quantize_row_q4_k`** |
| globals (`token_embd.weight`, `output.weight`) | 142 | 2 | byte-copy (the paper's shared+frozen embeddings/lm-head; `prism.hadamard.inverse_weight_names` semantics intact) |
| `ssm_alpha`/`ssm_beta` (a/b gate projs) | 30 BF16 | 96 | byte-copy — HALF the escape set, bit-shared by construction |
| `ssm_a`, `ssm_dt.bias` (+ norms, conv1d) | 0 F32 | 353 | byte-copy — the OTHER escape half + the dense fields |

Expected size: block projections ≈ 24.2B params × 4.56 bpw(Q4_K: 1444/256)
≈ **13.5 GB** + copies ≈ 0.6 GB → **~14 GB total** (the T3 budget table's
"q4_k ≈ 15 GB", confirmed). Escape-set verification (`verify_escape_set_shared`)
passes trivially: every non-requant tensor is byte-identical between the
copies.

## S1 — the requantizer (THIS PASS) — **COMPLETE 2026-10-06**

`examples/requant_q4_prefill.rs` (requires `twt_collapse` — the
`emit_collapsed_gguf` writer, which mirrors metadata in file order, carries
globals as byte-copies, and takes per-block supplied payloads):

- identity `CollapseSpec`: `blocks = (i, i+1, Merged)` for all 64 layers;
  projection suffixes → `TensorOut { Q4_K }` (new bytes), everything else →
  `TensorOut { parent type, parent bytes }`.
- `metadata_overrides`: `qwen35.block_count` = 64 (same value — the writer
  demands the override explicitly); `twt.layer_types` from
  `full_attention_interval = 4` (the loader's documented rule).
- Gates, all fail-closed:
  1. **ternary-exactness** (sampled, first requantized tensor): every
     dequantized value must land on `{−s, 0, +s}` of its group scale within
     f16 epsilon; refuses the run otherwise (the science premise above).
  2. **row-length divisibility**: every requant target's row length ÷ 256.
  3. **post-emit read-back**: re-open the pack; per sampled tensor,
     `dequant(q4)` vs `dequant(q2_0)` max-abs-err ≤ the analytic Q4_K bound
     (sub-block amax/15 + min-offset); geometry fingerprint equality vs the
     parent (the loader's own compat check would run this at T4 load anyway).
- Output: `../riir-train/data/Ternary-Bonsai-2-27B-Q4_K.pf.gguf` + a blake3
  sidecar (`.blake3`), the artifact record in this file.

### S1 landed (measured, the 4090 box)

- Run: `cargo run --release --features twt_collapse --example requant_q4_prefill`
  (staging 446 s + emit 142 s + verify; premises pre-checked by `--sample-only`).
- **Artifact:** `Ternary-Bonsai-2-27B-Q4_K.pf.gguf` — **14,428,236,928 bytes
  (14.43 GB)**, blake3
  `0a7a5605fadc37b482c0fddf6c28334244eea818b1716461bddb412f1a28309c`
  (sidecar beside the file). Requant payload 13.68 GB over 400 projection
  tensors; 451 copy-class tensors byte-identical (globals + the whole escape
  set); 851 tensors total, identity block table (64 layers), metadata
  mirrored in file order incl. `prism.hadamard.*` untouched.
- **Verification (exit 0, `--verify-only` re-runnable):** geometry plane
  (`general.architecture` + every `qwen35.*` key) equal INCLUDING the
  discriminant-tagged variant encoding — found by the first run's red:
  the writer's block-count override was re-widened to U64 while the parent
  stores U32, and `geometry_fingerprint` is discriminant-tagged, so the
  pair would have REFUSED at T4 load; the fix mirrors the parent's own
  value verbatim (identity requant = identity metadata encoding).
  Copy-class byte-identity PASS; all 400 requant tensors structurally
  Q4_K at the parent's exact shapes; sampled read-back (3 rows × attn_gate)
  max err 1.34–1.44e-3 within the analytic Q4_K sub-block bound.
- **Ternary-exactness premise PASS on the artifact** (the science): the
  bridge repack accepted every block (zero fourth-state codes) and the raw
  wire decode == `trit × f16 scale` on sampled blocks — the requant
  carries only ε(Q4_K), as pre-registered.
- En-route: `--hash-only` arm (the sidecar writer), the `probe_bonsai_meta.py`
  / `probe_bonsai_tensors.py` inventory scripts (GGUF v3 type table:
  7=BOOL, 10/11/12=U64/I64/F64 — the first hand-rolled table had 7=F64 and
  desynced the walk).
- ⚠ Known limit, disclosed: the PACK VERIFIES but the container cannot
  LOAD it yet — `load_ternary_proj` refuses Q4_K (the S2 work below); the
  metadata-plane compat (what `load_pair` checks before the prefill copy
  dequantizes) is proven equal by the verify gate.

RAM posture (disclosed): `emit_collapsed_gguf` materializes every `New`
payload before writing (the writer's `TensorOut.data: Vec<u8>`), so the run
peaks at ≈ 14 GB owned + the 6.7 GB mmap (page cache) — fits this box's
32 GB, disclosed as a one-shot artifact cost, not a serving cost.

## S2 — the q4-class weight arm — **COMPLETE 2026-10-06**

`load_ternary_proj` refused Q4_K; the ternary container's projections are
`katgpt_core::TernaryGroupWeights` (bit-planes). Landed as ONE enum at the
per-layer projection sites (the plan's option 1 — SOLID = one enum + one match
arm per site, never a duplicated forward):

1. **Weight vocabulary** — `ProjWeights { Ternary(TernaryGroupWeights),
   Q4K(Vec<BlockQ4K>, rows, cols) }` in `ternary_weights.rs` (the
   `GateProjWeights` precedent: tuple arm + dispatch methods). The 10
   per-layer projection fields move to it; `wte`/`lm_head` STAY bare ternary
   (the pack byte-copies the globals — Issue 980's globals contract) and the
   a/b gate projections stay `GateProjWeights` (byte-copied too). All the
   Q4K machinery is in this one file: `matvec_into` (fused dequant-dot
   `gemv_q4_k_row` per row, rayon across 16-row chunks at the ternary
   kernel's ≥256-row threshold — row-independent work, bit-stable under any
   worker count), `dequant_to_dense`, structural `invariants_hold` (block
   count == rows×cols/256), `bytes`.
2. **Forward** — `bitlinear` is the single CPU dispatch point (its match:
   Ternary = hook-or-SIMD exactly as before, Q4K = `matvec_into`); the GPU
   hook paths (`input_projections`, `ffn`) and the hook arm of `bitlinear`
   refuse a q4 projection LOUD (`as_ternary().expect(...)` — hooks pass
   `&TernaryGroupWeights`; silently dropping a hook would capture a wrong
   GPU graph). The lm_head call left `bitlinear` (it spells the ternary
   dispatch directly — the globals are never q4).
3. **Loader** — `load_proj(gguf, name)` dispatches on storage type:
   `Q2_0|PTQ1_0 → Ternary(load_ternary_proj)`, `Q4_K → Q4K(blocks.to_vec())`
   (shape + block-count checks; OWNS the bytes — the mmap borrow dies with
   the loader, mirroring the ternary arm's repack-to-owned). All 10 per-layer
   load sites moved to it; `load_ternary_proj` stays for the globals with an
   updated refusal naming the contract.
4. **Container** — `CopySet::Split` needed NO structural change: both copies
   are `QwenDeltaNetTernaryWeights`; the prefill copy simply carries Q4K arms
   where the pack quantized. `load_pair`'s existing gates (geometry
   fingerprint, layer_types, escape-set byte-identity) all pass unchanged on
   the dual-format pair.
5. **GPU arm — LANDED 2026-10-06 (the next pass, plan 618 item 5).** The
   cudarc Q4_K GEMV: `gemv_q4k_cuda_raw.rs` (new module, `ternary_gemv_cuda_raw`
   posture) — CUDA `gemv_q4k_dp4a` single + `gemv_q4k_dp4a_multi_persistent`
   (4-segment, the ternary multi's exact shape), one warp per output row,
   lane-strided activation blocks, consuming the SAME int8 + ascale buffers
   the ternary dp4a path quantizes into (the quantize step is format-
   independent — a q4 layer costs zero extra quantize launches). The
   min-offset term folds as a dp4a against 0x01010101; the sub-block scale
   pair decodes per activation block (`sb = (blk>>1)&7`, nibble class by
   `j` parity — sub-block-granularity, NOT byte-internal like q4_0).
   Wiring: `WeightBuffersCudarc` carries `format: WeightFormatCudarc
   {Ternary, Q4K}` + a `q4_blocks` byte payload (field-ADDITIVE — the
   riir-train-gpu backward reads `.codes`/`.wscale` and stays compiling);
   `upload_layer_weights_cudarc` dispatches `ProjWeights` per arm; the two
   GEMV primitives (`gemv_prequantized`, `gemv_prequantized_multi`)
   dispatch on format — the multi partitions a mixed batch into same-format
   launches (the dual-format pair is uniform-q4 per layer in practice);
   all 10 composite helpers pass the q4 handle through; `gemv_fused_into`
   (the negative-result apparatus) refuses q4 loud; the CubeCL whole-forward
   refuses q4 at upload (`q4_refuse` — the S2 hook law; the cudarc lane is
   the q4 GPU path). Also repaired: the S2 enum change had BROKEN this
   cfg-gated posture (27 compile errors — `ternary_gemv_cuda_raw` is
   compiled by no default lane; found by running the posture this pass).
   Gates: kernel parity 3/3 (exact host-mirror match + CPU-dequantizer
   layout pin on the mirror + CPU f32 reference at a 2% magnitude-
   referenced bound — measured 4.0e-4, ~50× margin; the value-referenced
   ratio on cancellation rows is meaningless, the S2 fixture lesson
   re-learned; multi == singles; accumulate contract), the e2e
   `q4_prefill_weights_run_the_gpu_forward_and_match_cpu` (all 10
   projections → Q4K through the GPU forward vs the CPU forward on the same
   arms: argmax equal, max|Δlogit| 0.0 on the discrete fixture, finite),
   full cudarc lib suite **357 passed / 0 failed** (5 ignored = the
   real-artifact arms), root lib 318 (the S2 baseline), issue-028 battery
   9/9, clippy clean at default + cudarc + cudarc+batched postures. The two
   macOS-only call-site blocks (metal_tensor_gemm upload, ane_prefill
   ladder) carry the same `q4_refuse` pattern but are compile-unverifiable
   on this box (blake3 build script needs a darwin C toolchain) — disclosed.
   **Fixture lessons (both caught by the structural legs, not by eyeball):**
   (1) Q4_K nibble alternation is at SUB-BLOCK granularity (sub-block 2p =
   low nibbles of bytes [32p..32p+32], 2p+1 = high nibbles of the SAME
   bytes) — the ternary kernel's byte-internal `__byte_perm` interleave is
   WRONG here; a first draft used it and the mirror-vs-dequantizer leg
   caught the class. (2) The multi kernel's first draft omitted the local
   row offset in the segment-mapped `gemv_q4k_row` call — row 1 of every
   segment silently computed row 0's weights; the multi==singles leg caught
   it. The structural legs (mirror pinned to `dequantize_row_q4_k`, multi
   vs singles) are the gate that pays.

En-route repairs (all compile-caught, none semantic):
- `convert_gate_proj`'s Ternary arm re-routed through a shared
  `convert_ternary_ref` (no extra clone on the `to_hybrid_dense` path);
- q4-arm additions to `to_hybrid_dense` (q4 dequantizes in both postures —
  the dense TRAINING container has no q4 arm);
- `for_each_ternary_site*` walkers now hand out `&ProjWeights` — the refit
  bin (`act_retention_walk`) takes `as_ternary_mut` (q4 sites refuse loud:
  refit needs bit-planes); the LmHead visit is preserved in the mutable
  walker via a temporary wrapper (the refit contract is ALL sites; the
  immutable twin — zero callers — drops it, documented);
- `bonsai2_rotation_load.rs` / `twt_bonsai_audition.rs` updated for the
  method accessors (`audition merges ternary candidates only` refusals);
- `DeltaNetTernaryLayerWeights` gained `#[derive(Clone)]` (small handles +
  dense f32 fields; the mixed-arm test needs one layer).

### S2 gates (all green)

- **Unit (lib, `deltanet_ternary_inference`)** — `q4k_arm_matvec_and_dequant_match_reference`:
  the Q4K matvec == dequantize-then-dot reference at 1e-3 (both chunk
  postures: 320 rows crosses the parallel threshold with a ragged tail);
  `dequant_to_dense` == the same reference; q4-vs-ternary aggregate sanity.
  **Fixture lesson (measured, not argued):** the first draft asserted a
  per-row q4-vs-ternary 10% envelope and FAILED — row 0 read q4 −2.0775
  (== f32 ref −2.0775 ✓) against ternary −2.4876, i.e. the TERNARY dot was
  19% off the f32 truth on this quant-hostile fixture. The cross-format
  check is now AGGREGATE (Σy within 5% of Σ|y|) — per-row dispatch truth is
  pinned by the exact dequant reference, and a row-shift/transpose bug blows
  the aggregate up. Also `ternary_arm_matvec_is_the_parallel_kernel` (the
  enum's Ternary arm == the pre-enum kernel, byte-equal) and
  `mixed_arm_layer_passes_invariants` (+ the wrong-block-count negative).
- **Always-on integration** — `dual_format_q4_prefill_pair_loads_and_runs`
  (issue028 battery): a synthetic dual-format pair (decode = PQ2_0 id 142,
  prefill = the same projections requantized Q4_K id 12 off the identical
  ternary content — the S1 copy policy at toy dims; new fixture helpers
  `synth_tensors_bonsai2_q4_prefill` + `q4_k_payload_from_ternary` in the
  shared kit). Gates: arm layout on both sides, the escape law at load,
  the phase split through the q4 arms end to end, boundary logits vs the
  all-ternary reference (relative L2 < 5%), finite decode logits.
- **Real-artifact load** (`#[ignore]`, release, the 4090 box) —
  `real_q4_prefill_pair_loads_and_prefills`: `load_pair` over the real
  6.7 GB PQ2_0 + 14.43 GB Q4_K.pf pair **PASSED the geometry-fingerprint +
  escape-set gates at production scale, 630 s / 702 s** across two runs
  (the 6.7 GB ternary repack + the 13.7 GB owned q4 copy; ~20.5 GB resident
  — the T3 budget table's "at the edge" row, measured). First run's red was
  the TEST's arm-layout assertion (layer 0 is DeltaNet — `attn_wq` is the
  empty arm; fixed to `layers[0].in_proj_qkv` + `layers[3].attn_wq`), never
  the loader.
  ⚠ **The real-pair PREFILL smoke did NOT complete** (disclosed): the
  second attempt printed the load gate then burned the session's entire
  30-min tool ceiling inside the prefill without finishing the 5-token
  prompt — consistent with the box PAGING (~20.5 GB resident + the 14.4 GB
  mmap page-cache pressure against ~24 GB free; the 630–700 s load itself is
  ~2–3× a clean-copy estimate, the same signature). The forward-path proof
  at real semantics is carried by the always-on synthetic gate (same code
  path, all arms); the real prefill smoke re-runs on a quieter box state and
  its s/token is a curiosity only — CPU host numbers do not price TTFT (the
  GPU arm / S3's deliverable).

RAM posture (disclosed): load peaks ≈ 20.5 GB owned + the 14.4 GB mmap page
-cache — fits the 32 GB box; the mmap pages are clean and evict while the
owned copies materialize.

## S3 — the measurement (the science deliverable)

- Arms, matched TOTAL storage per the issue: (a) single PQ2_0 (6.7 GB),
  (b) dual PQ2_0 + Q4_K (20.5 GB) vs a matched-storage single-checkpoint
  reference (q4_k single ≈ 14 GB + the difference in a deeper/other format —
  the issue's "matched TOTAL storage" pairing is fixed at run design time
  with the owner's 026 instrument vocabulary).
- Instrument: Plan 614's `dq_phase_matrix` family (the DQ phase-sensitivity
  bench) + the accuracy axis; TTFT at 4K/8K prompts once the S2 GPU arm
  exists (CPU host numbers do not price TTFT — measured this pass: the
  real-pair host prefill smoke paged before completing 5 tokens; see the
  S2 gate list).
- Pre-registered null: recovery ≤ measurement noise ⇒ container shelved,
  T4's decomposition number stands alone (the issue's PoC gate).

## Open at S2 close

1. ~~**The cudarc Q4_K GEMV (S2 item 5)** — the GPU prefill arm.~~ **LANDED
   2026-10-06 — see S2 item 5 above.** TTFT measurement on the real pair
   (S3's deliverable) is the remaining consumer of this arm.
2. **S3 itself** (above) — the accuracy half (per-family recovery vs
   matched storage) can start design any time against Issue 026's
   instrument vocabulary; the TTFT half now has its GPU arm.
3. The real-pair prefill smoke (optional, disclosed above) — a curiosity
   number on a quiet box, never a gate. The GPU q4 arm makes a real-pair
   GPU forward feasible (20.4 GB VRAM at the 4090's edge) — fold into S3's
   run design rather than running it standalone.

## Records

- S1 artifact: `Ternary-Bonsai-2-27B-Q4_K.pf.gguf` (blake3 sidecar; sizes +
  read-back verdict recorded here at landing).
- Housekeeping en route (this pass, `e993eff`): gpu-crate clippy `-D` clean
  at every posture (the pre-existing warnings the 035 session disclosed).
