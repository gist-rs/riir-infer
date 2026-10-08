# Issue 1005 — eDLM (Drex DLM) inference lane: GGUF arch + segment block-causal mask + pointer head

**Status:** OPEN — T8 CMMA FOLLOW-UP LANDED 2026-10-08 (this session, 4090): the tensor-core
GEMM arm for the four per-layer projections — cooperative-matrix f16×f16→f32 (`MatmulF16bCmmaCubeCL`,
new `matmul_f16b_cmma_cubecl.rs`), dispatch law **cmma at m ≥ 16, scalar below**, kill-switch
`EDLM_GPU_CMMA=0`; GOAT interleaved medians on BOTH runtimes: cmma wins 1.5× at m=18, 3.4–3.5×
at m=87, 7–8× at m≥512, loses 0.6× at m=8 (short branch rows stay scalar); parity green on both
runtimes with drift BYTE-IDENTICAL cross-backend (cmma kv 1.995e-3 / hidden 9.3e-6 on wgpu-spirv
AND native cuda); the activation-f16-rounding face (the tensor core consumes f16 A) is disclosed
+ per-posture gated (scalar kv pin 5e-4 unchanged, cmma pin 3e-3; hidden gate 2e-3 the binding
law, measured ≤9.5e-6). GOAT lane: `tests/edlm_matmul_cmma_goat.rs` (always-on real-shape parity
+ `#[ignore]` interleaved-A/B table). Prior T8: the GPU eDLM forward (`edlm_gpu` feature) —
f16-resident weights + multi-query GQA attention with per-row visibility bounds; tiny-weights
parity vs the CPU T7 cached path green on BOTH CubeCL runtimes (wgpu-spirv AND native CUDA on
the 4090, same drift class); the real-Q8_0 published-sample gate is env-gated `EDLM_GGUF`
(skip-loud unset — the 8.19 GB GGUF lives on the M3's SDXC1TB today, this box carries only the
BF16 shards; the M3 is NOT reachable from the 4090 — LAN timeout + Tailscale no-key — so the
real-Q8_0 cell stays an M3-side run). T7 LANDED 2026-10-08: state-prefix KV reuse — exact
(bit-identical on the real Q8_0) and **2.93×** on the published sample under load; the landing
note is on the T7 row.
Phase 2 LANDED 2026-10-08 (this session): T4 complete; T5's GGUF half complete (`head.pt` torch-pickle reader deferred — the GGUF embeds the same head); T6's published-constants leg PASSED on the real `drex-dlm-Q8_0.gguf` (87 tokens exact, worst drift 0.0103, argmaxes exact), its Python-runner corpus leg deferred (BF16 shards + torch oracle — a separate unit). T4 marker encode/pack — `EDLM_MARKERS` (the five reused Qwen specials, resolved via vocab lookup, never hardcoded ids; the ids pinned env-gated: 151659/151660/151648/151649/151661), `defuse_special_spellings` (the `<|name|>` → `<¦name¦>` anti-forgery rewrite, `[A-Za-z0-9_]+` names only, clean-room scanner), `EdlmLimits` (384/1024/2048 training + `serving(context)` posture), `encode_packed` (branch layout law pinned on a mock tokenizer: markers, seg, branch-local positions, opt_idx ends incl. span markers, strict over-context `ContextOverflow` refusal, non-strict state crop DISCLOSED via `PackedEncoding::state_truncated`, branch overflow refuses in BOTH modes, `option_isolation` shared-position + fixed-decide law); `PackedEncoding` grew `opt` + `state_truncated` (validate extended; hand-built test literals updated); `BpeTokenizer` gained `pub token_to_id` + `pub(crate) encode_no_special` (the `EdlmEncodeTokenizer` trait seams the encode off the concrete tokenizer — mock-tested). T5 pointer head: the GGUF-tensor load path (`edlm_pointer_head_from_gguf`, factored shared with the Phase 1 loader) + `EdlmPointerHead::question_probs` (Phase 1); **`head.pt` torch-pickle reader DEFERRED** — the serving artifact is the GGUF (the conversion embeds the same head); file a follow-up if a BF16-checkpoint path is ever needed. T6 end-to-end parity **PASSES on the real `drex-dlm-Q8_0.gguf`**: `EdlmGgufModel` (quant-resident mmap model — the forward dequants ONE layer at a time into a reused f32 scratch via the same `edlm_layer_forward` body the loaded path uses; O(file) evictable RAM + ~0.8 GB scratch vs ~32 GB fully-dequantized — the M3 box had ~9 GB free) + `forward_edlm_packed_streaming`/`forward_edlm_rows_streaming`; the packed-vs-row refactor is behavior-preserving (Phase 1 parity pins re-run green). Measured (94.8 s release, state_bidir=true = the released checkpoint posture): encode **87 tokens EXACT** (their published input-token count), team [0.9668, 0.0011, 0.0321] vs published [0.9641, 0.001, 0.0349], refund [0.1505, 0.8495] vs [0.1452, 0.8548], urgency [0.1968, 0.2179, 0.5852] vs [0.1876, 0.2169, 0.5955] — **worst drift 0.0103 (urgency), all three argmaxes exact** (billing / yes / Urgent); tolerance law = argmax EXACT + per-option < 0.015 (their own GGUF-vs-BF16 cross-runner spread: 0.0050 sample / 0.0094 255-option — same class; ours is a second compute path over the same Q8_0 blocks: f32-throughout vs their f16 accum). Packed-vs-row on REAL weights: hidden < 1e-4, probs < 1e-3. En-route heal (pre-existing at HEAD, all-features posture): `examples/requant_q4_prefill.rs` needless_range_loop (manual — healer declines the class). Gates: clippy -D at default/edlm/all-features/no-default + --all-targets; edlm lib **279/0** (+9 T4 tests), default lib 257/0; the T6 run is env-gated `EDLM_GGUF` (skip-loud unset).

Phase 1 T1–T3 LANDED 2026-10-08 (prior session): `edlm` feature (pure-local, default-off) + `src/transformer/edlm.rs` — T1 `load_edlm_weights_gguf` (arch `edlm`; metadata prefix `edlm.*` header-verified against the real `drex-dlm-Q8_0.gguf` 8.19 GB @ SDXC1TB/models/drex-dlm/q8_0 — 36L/4096/12288/32:8/128/151936/32768/1e-6/1e6; tensor map `token_embd`/`output_norm`/`output`(opt)/`blk.N.{attn_norm,attn_q,attn_k,attn_v,attn_output,attn_q_norm,attn_k_norm,ffn_norm,ffn_gate,ffn_up,ffn_down}`/`pointer.{q,k}.{weight,bias}`/`pointer.temperature.weight` (F16 weights, F32 norms+biases+temp; NE 1); NEOX RoPE ⇒ **NO Q/K unpermute** (the `qwen2` loader posture — the fork maps `LLM_ARCH_EDLM` to `LLAMA_ROPE_TYPE_NEOX`)); T2 `branch_mask` + `row_branch_mask` (the 4-quadrant law + `state_bidir` OR-ed independently of causality — the reference order base→bidir-OR→opts-AND→diag; pads invisible keys/dead queries; option-isolation conjunct) over `attention_head_masked` (new ungated substrate in attention.rs, eligibility discipline mirroring `attention_head_set_causal`); T3 `PackedEncoding`/`rows_of`/`forward_edlm_rows` + `forward_edlm_packed` (Qwen3 block: QK-RMSNorm per head pre-RoPE, SwiGLU, GQA, no KV cache, final-norm hidden states — no lm_head). **Packed-vs-row parity is EXACT** (worst diff 0.0 < 1e-5, both `state_bidir` postures) after two measured catches: the attention residual must be the PRE-norm stream saved per position (the Phase-A-saved normed-`xr` bug made state depend on the sequence's LAST token — single-question parity was blind to it, the two-question diagnostic pinned it; the diagnostic tests STAY as regression pins), and `forward_edlm_rows` must mirror the caller's `state_bidir` (parity holds in each posture, never across). Gates: clippy -D at default/edlm/all-features/no-default postures + `--all-targets`; default lib 257/0; edlm lib 270/0 incl. the env-gated real-GGUF header check (`EDLM_GGUF`, skip-loud unset). Weights fetched to SDXC1TB (CC BY-NC — local bench/comparison only); reference pinned `nace-ai/drex-dlm` @ `6c63df2` + their `llama.cpp` branch `edlm` @ `cdcf65d` cloned under `.raw/` (MIT code — reference reading only; rm when the lane closes).

## The gap (measured, not guessed)

riir-infer ships a **dLLM/D2F lane** (`dllm` feature: `src/transformer/dllm.rs` —
`forward_bidirectional` + `forward_block_causal`, Gemma2 CPU + `riir-infer-gpu/gemma2_d2f`)
but **cannot load or serve Drex DLM** (`nace-ai/drex-dlm`, Efficient-DLM-8B backbone):

- `src/gguf_loader.rs` arch dispatch supports `gemma2` / `llama` / `qwen2` / `qwen35`
  (+ PrismML ternary, deltanet) — **no arch accepts `edlm`** (their GGUF arch key; Q8_0
  keeps the pointer head F16).
- The D2F block-causal mask is **position/block-based**; Drex needs a **segment-based
  branch mask**: `attend(i,j) iff j≤i and (seg[j]==0 or seg[j]==seg[i])` + `state_bidir`
  (state attends state both directions, never sees branches; branches causal-within,
  attend state, isolated across questions) — `code/kev/model.py::branch_mask_batch` at
  `6c63df2` is the reference.
- **No marker encode/pack** (`<q>`/`<opt>`/`</opt>`/`<decide>` specials, per-question
  branches, `option_isolation` with shared position ids) and **no pointer head**
  (256-dim q/k Linear at decide/opt markers, scaled dot, temperature from `head.pt`).
- Row-form fallback (state+branch per causal row, bidirectional state block) — the
  mask-parity escape for long/overflowing requests — does not exist either.

## Why (consumer)

1. reflex issue 073 (`--drex` comparison lane) currently requires THEIR runner (Python
   torch or the llama.cpp `edlm` fork). With this lane, OUR substrate serves the same
   weights — the measure-vs-serve split gets a third serving posture (ours), and the
   eDLM arch joins the loader's league coverage (new arch family: AR→diffusion converts).
2. T7 closes reflex Research-002's open GAP (candidate-branch prefix reuse on a CAUSAL
   backbone — second reference impl noted in reflex `.research/008`) on OUR substrate.

## Boundary (the rung split — the laya precedent governs)

Drex/eDLM spans the family ladder and each rung has its owner:

| rung | repo | for Drex? |
|---|---|---|
| **substrate** (loader, forwards, mask, head mechanics) | `riir-infer` — THIS issue | yes — rethink's out-of-scope row delegates "the encoder forward SUBSTRATE → riir-infer (public)" |
| **measurement** (harness comparison lane) | `riir-reflex` issue 073 | yes — harness-only, never the release set (the agentjev family law) |
| **product serving** (weights posture, arsenal seating, ESC arms) | `riir-rethink` (private forever) | **NO — license-blocked**: CC BY-NC weights can never serve the commercial product; reflex's out-of-scope row sends "TRAINED-ENCODER serving arms → riir-rethink" |
| **open serving stack** (L1 lane seating) | `riir-instinct` (public) | **NO for Drex**: weights barred from any release set (the agentjev family law); NC also blocks downstream commercial users of the open stack |

The pattern is the laya lane's exactly: substrate in `riir-infer-laya` (public), serving
posture + product in rethink (private), reflex measures. The rethink rung for an
eDLM-class model opens ONLY when a license-clean model exists (open-licensed or
self-trained) — that issue is filed THERE and CONSUMES this issue's substrate; nothing
serving-shaped is ever built here or in reflex.

## Phases

### Phase 1 — load + forward (CORE)

- [x] **T1** `edlm` GGUF arch support in `src/gguf_loader.rs`: config extraction
      (block_count, head geometry, rms eps — Qwen3 tensor-name family), weight mapping
      onto the existing transformer structs, pointer-head tensors carried F16 (Q8_0) or
      from `head.pt` (BF16 safetensors path). **Substrate-first: check tensor-name
      overlap against the existing `qwen35` loader arm BEFORE writing anything new —
      the eDLM block tensors follow Qwen3 naming, so T1 may be a variant of that arm,
      never a parallel loader.** *(LANDED 2026-10-08 — loader lives in `src/transformer/edlm.rs` (`load_edlm_weights_gguf` + `edlm_config_from_gguf_metadata`), NOT gguf_loader.rs (that file is already 3.4k lines); reuses `GgufFile::dequant_f16_to_f32` (Q8_0 path) + the llama-layer weight structs by composition (`EdlmLayerWeights { base: LlamaLayerWeights, q_norm, k_norm }`); tensor names header-verified against the real Q8_0 GGUF; pointer tensors carried (`pointer.{q,k}.weight` F16 + F32 biases + `pointer.temperature.weight`); safetensors/`head.pt` path deferred to T5/T6.)*
- [x] **T2** Segment-based block-causal attention mask (generalizes
      `attention::block_causal_t_n`): segment ids + `state_bidir`; additive mask
      builder usable from both the packed forward and row forms. Unit tests pin the
      4-quadrant law (state↔state bidir, state∤branches, branch→state, branch∤branch). *(LANDED 2026-10-08 — `branch_mask` (bool `[l*l]`) + `row_branch_mask`; the eligibility PRIMITIVE is `attention_head_masked` in attention.rs (ungated substrate); tests pin the 4 quadrants, pads, diagonal, option isolation, buffer asserts; `state_bidir` is an independent OR, not causal-gated — the reference order base→bidir-OR→opts-AND→diag.)*
- [x] **T3** Row-form fallback: per-question causal rows (state + branch), packed-vs-row
      parity test (their `test_rows_match_packed` analog, our fixtures). *(LANDED 2026-10-08 — `PackedEncoding`/`BranchRow`/`rows_of` (layout-mismatch refusals) + `forward_edlm_rows`; parity EXACT (diff 0.0) in BOTH `state_bidir` postures on tiny random Qwen3-shape weights; the two diagnostic tests stay as regression pins (single-question = exactness, two-question = the cross-question contamination catch). `forward_edlm_rows` takes `state_bidir` — parity holds per posture, never across.)*

### Phase 2 — decision readout

- [x] **T4** Marker encode/pack (specials, per-question branches, strict over-context
      refusal — never silent crop; `option_isolation` variant with shared position ids
      and fixed decide position). *(LANDED 2026-10-08 — `EDLM_MARKERS`/`defuse_special_spellings`/`EdlmLimits`/`encode_packed`; layout law mock-tokenizer-pinned: markers·seg·branch-local pos·opt_idx ends·strict `ContextOverflow`·non-strict crop disclosed via `state_truncated`·branch overflow refuses both modes·isolation shared-span positions + fixed decide; `PackedEncoding` + `opt`/`state_truncated` (validate extended); `EdlmEncodeTokenizer` trait + `BpeTokenizer` impl (`token_to_id` added).)*
- [x] **T5** Pointer head module: q/k projections, scaled dot, temperature; load from
      GGUF tensors / `head.pt`. *(GGUF-tensor half LANDED 2026-10-08 — `edlm_pointer_head_from_gguf` factored shared with the Phase 1 loader; `question_probs` was Phase 1. **`head.pt` torch-pickle reader DEFERRED** — the serving artifact is the GGUF (the conversion embeds the same head); a BF16-checkpoint path would need a torch-pickle reader — file a follow-up only if ever needed.)*
- [-] **T6** End-to-end parity: system-one request → encode → probs against their
      published sample outputs (`examples/request.json` → billing 0.9641 / refund
      0.8548 / urgency 0.5955 @ `6c63df2`) + a fixture corpus run through their Python
      runner (G5-style discipline: distribution parity, per-option delta disclosed —
      their own cross-runner tolerance is 0.0050–0.0094 GGUF-vs-BF16). *(PUBLISHED-CONSTANTS LEG PASSED 2026-10-08 — `EdlmGgufModel` + `forward_edlm_packed_streaming`/`forward_edlm_rows_streaming` (one-layer-at-a-time dequant, O(file) mmap-resident); env-gated `sample_request_end_to_end_parity_env_gated`: 87 tokens EXACT, worst drift 0.0103 (urgency), team/refund ≤ 0.0053, all argmaxes exact, packed-vs-row on real weights hidden < 1e-4 / probs < 1e-3; tolerance law argmax-EXACT + 0.015 disclosed in the test. DEFERRED: the PYTHON-RUNNER CORPUS LEG — needs the ~16 GB BF16 shards + their torch runner as the subprocess oracle; file it as its own unit when the comparison lane (reflex 073) needs it.)*

### Phase 3 — serving perf (T7 LANDED; T8 open)

- [x] **T7** State-prefix KV reuse (exact by construction: state never sees branches):
      prefix cache + branch continuation rows; `prefix_min_tokens` heuristic (their 384
      attention-only / always-hybrid law). Closes the bekko-006 inference-side sibling
      + reflex Research-002 GAP. *(LANDED 2026-10-08 — `EdlmStateKv` (per-layer state
      K/V + the `state_ids` stale-cache content check + post-final-norm state hiddens)
      via `edlm_state_prefill{,_streaming}`; branch continuations
      `forward_edlm_branches_cached{,_streaming}` — branch queries attend
      `[state KV | own branch KV]` under a posture-INDEPENDENT eligibility law (state
      keys precede every branch query; `state_bidir` only widens state↔state, a
      prefill-only concern); one-call `forward_edlm_rows_cached{,_streaming}` with the
      `EDLM_PREFIX_MIN_TOKENS = 384` crossover (their attention-only law; the
      hybrid-always arm doesn't apply — this lane is dense); built on a
      behavior-preserving `edlm_layer_forward` split (`edlm_layer_kv` Phase A +
      `edlm_layer_attend` Phase B, `kv_base`/`key_count` params, allow attribute per
      house style). EXACTNESS PROVEN BIT-IDENTICAL, not tolerated: tiny weights both
      postures AND the real Q8_0 (cached row hiddens `assert_eq` plain row hiddens).
      Measured (2026-10-08, M3 Max, release, LOADED box — sibling cargo builds, ~4.9 GB
      free RAM, swap 6.1/7 GB): the published sample (18-token state × 3 questions)
      plain 157.3 s vs cached 53.7 s (prefill 11.2 + branches 42.5) = **2.93×**, winning
      even below the crossover because the row form recomputes the state per question —
      disclosed, never asserted (box state is part of the claim); FLOP-level expectation
      ~1.4× at this state size, growing with state length and question count. The 384
      crossover stays the reference default, caller-overridable; our rows-path
      measurement suggests lower crossovers win for K≥2 — a tuning note for the
      reflex-073 consumer. Stale-cache law: a cache whose state ids/length/geometry
      differ REFUSES (`EdlmStateKv::check_matches`) — never silently serves. Gates:
      clippy -D at default/edlm/all-features/no-default + `--all-targets`; edlm lib
      **285/0** (+6 T7 tests), default lib 257/0; the real-weights run is env-gated
      `EDLM_GGUF` (skip-loud unset).)*
- [x] **T8** GPU eDLM forward (`riir-infer-gpu`): mask plumbing + parity vs CPU.
      *(LANDED 2026-10-08 — `crates/riir-infer-gpu/src/edlm_cubecl/{mod,tests}.rs` +
      `matmul_f16b_cubecl.rs`, feature `edlm_gpu = ["cubecl_runtime", "riir-infer-core/edlm"]`.
      **The two design decisions, settled + measured:** (1) **upload path = Q8_0 → f32 EXACT
      host dequant → f16-resident** (~15.7 GB on the 8B release; fits the 24 GB 4090 and
      unified-memory M3) — every GEMM accumulates f32 (the f16 GEMV family's law), so the only
      numerics faces vs the CPU f32 path are the f16 weight rounding (rel ≤ 2^-11/weight) and
      the attention kernel's online-softmax order; quant-resident GPU dequant kernels stay the
      documented fallback (a bigger kernel, no measurement-lane need). (2) **parity target =
      CubeCL** (portable — one kernel set on wgpu-spirv AND native CUDA; the cudarc raw arm is
      decode-GEMV for quantized weights with no dense/f16 GEMM — building one would duplicate
      `matmul_cubecl`, the substrate-first refusal). **The decomposition (T7's, unchanged) makes
      the mask plumbing trivial: every phase is "query i attends keys 0..t_n(i)"** — state
      prefill `t_n = sl` (bidir) / `i+1` (causal), branch continuations `t_n = sl+i+1` over
      `[state KV | own branch]` — one kernel shape (`edlm_attn_multi_f32`, grid (n_head, sq),
      128-thread cubes == head_dim 8B shape, params buffer, combined `[keys | values]` layout,
      online softmax) serves both; NO general bool-mask kernel needed. Pipeline = the
      `llama_cubecl` hybrid: GPU GEMMs (`matmul_f16b_cubecl` — `matmul_tiled_f32` verbatim with
      the f16 cast at the B tile load, buffer-derived dims, `.issues/515` binding guards) + GPU
      multi-row rmsnorm (`edlm_rmsnorm_rows_f32`); CPU small-ops between the 4 readbacks/layer
      are the EXACT core helpers (`qk_norm_inplace` — made pub; RoPE, SwiGLU, residuals, the
      pointer head). Core diffs: `EdlmGgufModel.gguf` made pub (the f16 upload path dequants
      the same tensors from the same handle) + `qk_norm_inplace` pub. Q|K|V and gate|up are
      row-concat at upload (one GEMM + one readback each); wte never goes VRAM (host row
      gather). **Parity measured on f16-GRID tiny weights** (the CPU oracle runs the SAME
      rounded weights — kernel error isolated from the rounding face): kv drift ≤ 6.9e-5,
      hidden drift ≤ 1.3e-6, BOTH state_bidir postures, BIT-STABLE across dev/release AND
      wgpu-spirv/native-CUDA postures; kernel units (rmsnorm vs CPU 1e-5, attention vs plain
      softmax 2e-4 incl. a >1-tile vis case, GEMM 1e-4 rel) + the no-cache refusal pin. The
      measured catch: the first cut read V without the values-region offset (`kv_half`) —
      the single-tile attn unit caught it as ~0.5 drift, and the TINY KV gate caught the
      poisoning it fed into layer 2's captured cache (0.0011 vs the 5e-4 gate) — two
      instruments, one defect, both stay. **Real-Q8_0 gate = env-gated `EDLM_GGUF`**
      (`real_q8_parity_env_gated`: argmax EXACT + per-option < 0.015 vs the CPU streaming
      lane AND the published constants, drift + timing PRINTED with `runtime_name()`;
      heavy-test serialization + `memory_cleanup` per Issue 712) — skip-loud unset; this box
      has only the BF16 shards, the GGUF is M3-side. VRAM/footprint: 15.7 GB weights + ~1.2 GB
      KV@4096-state + transient activations. Disclosed v1 scope cuts: packed-mask/isolation
      forward (rows/cached semantics only — the CPU cached path's own caveat), CUDA graphs,
      CMMA tensor-core GEMM, GPU-resident KV carry, batched multi-branch — each a follow-up
      with its own bench. Gates: clippy -D at edlm_gpu/default/no-default postures
      + --all-targets; fence gate PASSED (no new deps); core default lib **257/0** unchanged,
      edlm lib 28/0 (filter), gpu edlm tests 7/0 dev+release+wgpu+CUDA.)
      *(**T8 CMMA FOLLOW-UP, same day, 4090 box:** `matmul_f16b_cmma_cubecl.rs` —
      the four per-layer projections dispatch through the cooperative-matrix
      tensor-core kernel (`gemm_ternary_cmma16` shape: 32-thread cubes, 2×2
      16×16 sub-tiles, f16 staging, f32 accumulators) whenever `m >= 16`;
      `EDLM_GPU_CMMA=0` forces scalar everywhere. **Measured GOAT (interleaved
      pairs, release, quiet GPU):** cmma wins on BOTH runtimes from m≈16
      (1.09×/0.85× noise-class), 1.46–1.55× at m=18–32, 2.6–3.6× at m=87,
      7.0–8.0× at m≥512; LOSES at m=8 (0.60–0.63× — staging ALU over dead
      tile rows), so short branch rows keep the scalar kernel. Absolute:
      the m=87 QKV GEMM 1744→496 µs on native CUDA. **Parity:** the
      activation-f16 rounding (the tensor core consumes f16 A — the second
      disclosed face beside weight rounding) moves the state-cache K/V drift
      6.5e-5 → 1.995e-3 — per-posture pins (scalar 5e-4, cmma 3e-3), the
      hidden gate 2e-3 is the binding law and measures ≤9.5e-6; drift is
      BYTE-IDENTICAL across wgpu-spirv and native CUDA. GOAT lane
      `tests/edlm_matmul_cmma_goat.rs` (real-shape parity always-on + the
      `#[ignore]` interleaved-A/B table; [[test]] required-features row per
      the R1 pattern). Gates: clippy -D at edlm_gpu / default /
      no-default (+ all-targets each), fence PASSED (no new deps), core
      default lib 257/0 unchanged, gpu edlm tests 7/0 at dev+release ×
      gpu edlm tests 7/0 at dev+release ×
      wgpu+CUDA. ⚠ ~~PRE-EXISTING at HEAD, not this change: `--all-features
      --all-targets` is RED on Windows in `dq_s3_matrix`~~ → **EXAMINED + FIXED
      2026-10-08 (4090 session):** the 5× E0308 were the S2 `ProjWeights` enum
      change (`8d92cac`/`91d0823`) missing ONE double-cfg-gated test —
      `test_speculative_run_invalidates_captured_graph`
      (`speculative_decode` + `cuda_graphs_forward`) still seeded the layer
      projections as bare `TernaryGroupWeights`; repaired to the
      `seed_live_backbone` pattern (`as_ternary_mut().expect("ternary
      fixture")`), and the test EXECUTES on the 4090: pos2 max_diff 0e0
      (bit-identical, seeds non-vacuous). The dead `container` binding in
      `dual_phase_switch_forward_matches_cpu_container` was swallowing the
      `from_parts` Result — now `.expect("container pairing must hold")` (the
      layer_types + escape-set laws are load-bearing). The `dq_s3_matrix` bin's
      8 warnings healed in the same pass (dead `built` init → break-with-value,
      dead `all_lengths_ok` + the redundant `dc`/`mc` outer guard removed —
      report text unchanged, the per-length find already skipped missing
      cells — `useless_format` ×2, `redundant_closure`, `unused_mut`,
      `#[allow(clippy::too_many_arguments)]` on `run_cell` per house
      convention). Gates: clippy -p riir-infer-gpu --all-features
      --all-targets GREEN (0 errors, 0 findings in touched files) + default
      posture GREEN; both named tests PASS on the 4090.) Remaining T8 perf follow-ups: k-blocked CMMA
      staging (the kb4/sg4 ladder), GPU-resident KV carry, batched
      multi-branch, CUDA graphs.)

## License law

Weights CC BY-NC 4.0: local bench/comparison serving only; never a product lane, never
a distill teacher. Their repo code is MIT — reference reading only, no code copy (our
implementation is clean-room from the wire + published mechanics).

## Out of scope

- Training/fine-tuning the decision adapter (no recipe ships anyway).
- Their multi-decision `requests` wrapper (refused in their release too).
- Hosted Drex 1.5 (128K) — different artifact; this lane serves the open 32K release.
