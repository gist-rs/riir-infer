# Plan 618 — Issue 028 T4 unblock: the Q4_K prefill pack (S1) + the q4 weight arm (S2) + the dual-PTQ measurement (S3)

**Status:** IN PROGRESS — S1 landed this pass; S2/S3 scoped below.

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

## S2 — the q4-class weight arm (NEXT)

`load_ternary_proj` refuses Q4_K today; the ternary container's projections
are `katgpt_core::TernaryGroupWeights` (bit-planes). The prefill copy needs a
q4 arm:

1. Weight vocabulary: a `ProjWeights`-style enum at the
   `DeltaNetTernaryLayerWeights` projection sites (ternary | q4k), or a
   parallel prefill weight struct — decided at implementation time by which
   touches fewer forward call sites (the host ternary forward's matvec
   dispatch is the surface; SOLID = one enum + one match arm per site,
   never a duplicated forward).
2. Loader: the projection loader gains the `GgmlType::Q4_K` arm (dequant
   rows are already in `quant::q4k`; the container keeps blocks + does
   on-the-fly row dequant in the matvec — never materialize f32).
3. Host matvec: q4 row-dequant fused into the dot (or row-dequant-once per
   forward — prefill touches each row once per call).
4. Container widening: `CopySet::Split` prefill side takes the new weight
   type; `generate_greedy_disaggregated` routes the prefill phase through
   it. G1's byte-identity gate keeps its same-format arms; a new gate pins
   q4-prefill ≈ ternary-only within the measured ε4 bound (NOT bit-identity
   — the formats differ; the T2 bit-compare law applies to the HANDOFF, not
   across quantizations).
5. GPU arm (TTFT numbers): the cudarc prefill lane gains a Q4_K GEMV
   (the gemma2_cubecl lane's `GemvQ4KCubeCL` is the in-repo port
   reference; the cudarc deltanet lane has no q4 weight path today).

## S3 — the measurement (the science deliverable)

- Arms, matched TOTAL storage per the issue: (a) single PQ2_0 (6.7 GB),
  (b) dual PQ2_0 + Q4_K (20.5 GB) vs a matched-storage single-checkpoint
  reference (q4_k single ≈ 14 GB + the difference in a deeper/other format —
  the issue's "matched TOTAL storage" pairing is fixed at run design time
  with the owner's 026 instrument vocabulary).
- Instrument: Plan 614's `dq_phase_matrix` family (the DQ phase-sensitivity
  bench) + the accuracy axis; TTFT at 4K/8K prompts once the S2 GPU arm
  exists (CPU host numbers do not price TTFT).
- Pre-registered null: recovery ≤ measurement noise ⇒ container shelved,
  T4's decomposition number stands alone (the issue's PoC gate).

## Records

- S1 artifact: `Ternary-Bonsai-2-27B-Q4_K.pf.gguf` (blake3 sidecar; sizes +
  read-back verdict recorded here at landing).
- Housekeeping en route (this pass, `e993eff`): gpu-crate clippy `-D` clean
  at every posture (the pre-existing warnings the 035 session disclosed).
