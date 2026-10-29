# riir-infer

The LLM inference substrate: weight loading, quantization, model
architectures, and compute kernels for model-based inference — upstream of
every engine, game, and application concern.

## Built on KatGPT-RS

The primitive core is [KatGPT-RS](https://github.com/katopz/katgpt-rs) (public, MIT); this repo adds loaders,
quantization and model architectures on top. From it (path deps
`../katgpt-rs`):

- `katgpt-core` — shared types, SIMD kernels, `float_order`, sigmoid, fitted
  anchor tables, SIMD-LUT dequant, ternary group-scale matvec, GDN tree
  verify, row-logit floor, and the opt-in inference features forwarded from
  the feature table (deltanet / gemma4 / ternary / dllm / belief drafter …).
- `katgpt-transformer` — gated MLP, delta routing, wall attention.
- `katgpt-speculative` — the DDTree / DFlash / Weaver speculative substrate.
- `katgpt-forward` — canonical clustered + standard LM head, top-k select.
- `katgpt-quant` (opt-in `turboquant`) — TurboQuant.
- `katgpt-attn` (opt-in `flashmemory_gqa` / `flashmemory_budget` /
  `flashmemory_trained_indexer`) — FlashMemory sparse GQA: the absolute-rule
  selector, the top-ρ budget selector, and the trained-indexer arm.

## The crate

`riir-infer-core` (repo root):

`riir-infer-gpu` (`crates/riir-infer-gpu`) — the GPU runtime + kernel layer: device context, buffer helpers, pool-poison detector, the CubeCL runtime (Metal/WGSL/SPIR-V via wgpu; CUDA behind `cuda_backend`), and the kernel families (GPU transpose; EXL3 trellis decode + block-Hadamard behind `exl3_gpu` — decode bit-exact vs the CPU reference, Hadamard gated in the FMA-contraction class, 71-87x wall over the CPU arm on the 4090). Optional companion crate — CPU-only consumers never resolve the GPU dep tree.

`riir-infer-laya` (`crates/riir-infer-laya`) — the pinned-checkpoint encoder lane: a Python-JSON byte writer (ungated), the tokenizer/config/weights (locate → verify → download) substrate, and a flat-`Vec<f32>` forward with CPU (`gemm`) and macOS Metal (MSL) backends behind the `laya-riir` / `laya-riir-metal` features, plus an opt-in portable CubeCL/wgpu backend (`laya-riir-cubecl`) over `riir-infer-gpu`'s op layer, and a macOS whole-graph Apple Neural Engine backend (`laya-riir-ane`): the `AneEncoder` loads a BLAKE3-digest-pinned artifact tree (content-addressed `.mlpackage` compile cache, an MLComputePlan gate re-run at load — 100% ANE-preferred ops or loud demotion) and serves the fp16 host gather under a strict bucket-coverage law; the runtime substrate consumed by riir-reflex's ANE routing tier. Plan 612 adds the e8 int8-table stack to the same feature: `LAYA_ANE_TABLE=e8` swaps the fp16 gather for a per-row-scaled int8 gather (~50% resident per checkpoint, G5 published in the plan; verified sidecar load — digest + byte-total + shape + checkpoint pairing + `serves_buckets` coverage, any mismatch refuses naming the converter command), opt-in pending the G1 parity + G2 perf gates (fp16 stays the default posture; env unset is byte-identical). The hand-tuned Metal lane is the default: in Bench 006 the CubeCL arm ran 5–8× slower than Metal and beat the CPU lane on short sequences. Optional companion crate — lane-free consumers never resolve the tokenizers/gemm tree.

`riir-infer-ane` (`crates/riir-infer-ane`) — the SHARED Core ML/ANE host (plan 619 T1.0): artifact digest/stage/sha-verify, the content-addressed `.mlmodelc` compile cache, `MLModel` load with compute units + multi-function `functionName`, and the `MLMultiArray` input/output helpers (dtype-agnostic, stride-aware readers). macOS-only by contract — a zero-dep stub compiles elsewhere, so enabling a consumer lane off-macOS resolves zero Apple deps. Consumed by `riir-infer-laya` (the `laya-riir-ane` posture) and `riir-infer-clef` — one host, never a second copy of the CoreML machinery.

`riir-infer-clef` (`crates/riir-infer-clef`) — the clef-text decision lane (plan 619): clef-text-0.6b (Apache-2.0, FluidInference's student of Cloudflare clef) answered IN-PROCESS on the Neural Engine behind the opt-in `clef-ane` feature (macOS-gated; compiles empty by default — the laya posture). The pinned artifact tree (HF `FluidInference/clef-text-0.6b-coreml`, sha256-verified download-on-demand) hosts a multi-function decoder/head package (buckets L256/L512/L1024/L2048); the Rust port (record encoder with Python-JSON rendering, fp16 RoPE/mask host math, head packing, per-question softmax over the artifact's own head convention) reproduces the official FluidUse runtime BIT-EXACT (plan 619 T2.2: 544/544 logits, 0 top-1 disagreements over 102 frozen captures) at 35–747 ms/record ANE (bench 884). Consumed by riir-reflex's `clef-local` comparison arm (reflex bench 152 board, bench 153 perm-probe: order-invariant by construction).

`riir-infer-audio` (`crates/riir-infer-audio`) — the audio model lane (issue 015): loaders + streaming serving state for PUBLISHED CoreML audio bundles on the shared ANE host, behind the opt-in `vad-ane` feature (macOS-gated; compiles empty by default — the clef posture). First lane: silero-vad (FluidInference `silero-vad-coreml`, blake3-dir-v1 digest-pinned both postures — the published `.mlmodelc` v6.2.1 through the host's precompiled load, and the `.mlpackage` source through the compile cache): 256 ms chunks over the carried context tail + LSTM hidden/cell state, ONE multi-output forward per chunk. PoC verdict (`.docs/004`): real speech reads 1.0000 vs every synthetic negative < 0.05, p50 0.163 ms/chunk (RTF 0.00064, 1500× headroom, CPU posture) — and the placement plan reads ALL-CPU on M5 Ultra/macOS 27 for every bundle probed (the ANE finding, owner-visible; the plan walk is not a usable ANE-placement authority on this silicon today).

- **Loaders** — GGUF + safetensors, mmap-backed, BLAKE3-checkable.
- **Quantization** — the q2k…q8kv GGUF family (incl. the Q4_K/Q6_K row
  formats + cudarc GEMV arms), q2_0 ternary, PTQ /
  TurboQuant paths, the dual-PTQ disaggregated serving container (opt-in
  `disaggregated` behind `deltanet_ternary_inference`; issue 028 T4 verdict
  NULL — shelved, record in HISTORY), EXL3 trellis decode + block-Hadamard
  (opt-in `exl3`;
  the multi-shard zero-copy pack reader, the CPU reference, and the
  bit-identical CPU fast arm — codebook LUT + rayon, 10.6-11.4x — land
  here; the GPU kernels live in `riir-infer-gpu` behind `exl3_gpu`.
  `Exl3Pack::open` is era-gated fail-closed on the pack's
  `quantization_config.version` against the validated set (`["1.4.2"]`)
  — legacy-era packs silently decode wrong (issue 001 §12.7, record in `.docs/001`); a
  deliberate unvalidated-era read is `Exl3Pack::open_unverified_era`).
- **Architectures** — gemma / llama / ternary / wall layers, deltanet,
  transformer, rope, speculative-decoding types (dflash); the dLLM decode
  lane (`gemma2_d2f`) + the FA-constrained exact-posterior sampler (opt-in
  `fa_constraint`, issue 035 P3 verdict NOT PROMOTED — stays opt-in, record
  in HISTORY); the FlashMemory top-ρ budget decode lane (opt-in
  `flashmemory_budget`, issue 1117): the Bonsai ternary decode routes its 16
  attention layers through sparse FlashMemory attention — per-layer top-
  `ρ·n` block selection by the centroid-dot score (the Bench-671 selector,
  katgpt-attn `flashmemory_budget_select`), an incremental per-layer block
  cache whose last-block update is bit-identical to a fresh rebuild, and the
  block-open force-refresh law (a newly opened block force-refreshes the
  selector that step, or tokens landing in it stay outside the selection
  until the next periodic refresh); projections/norm/rope/cache/gating/folded
  output projection stay verbatim dense, and ρ=1.0 decode is bit-identical
  to dense over 700 synth steps. End-to-end quality gate PASSED at 32K
  (Bench 893): per-layer fidelity survives 16 attention layers compounding
  through the 64-layer hybrid stack — ρ 0.15 (85% KV reduction): pooled
  ΔNLL +0.4048, top-1 70.56%; ρ 0.30: +0.2087, 79.91%; per-bucket quality
  IMPROVES with depth (ΔNLL 0.477→0.248 across the 4K buckets at ρ 0.15) —
  the Bench-671 depth worry does not compound. The speedup/retention
  crossover sits near ρ ≈ 0.2, so the serving operating point is ρ ≤ 0.15;
  the 128K perf+quality axis is in flight (issue 1117 T4).
- **CPU references** — the reference paths the GPU kernels are validated
  against.

No cognition, no training, no game or chain semantics. Consumed by
inference engines via path dependency.

## Build

```sh
cargo check                  # default features
cargo test --lib             # unit tests
cargo tree | grep -c 'riir-' # must print 0 — zero riir-* deps, by contract
```

Toolchain pinned by `rust-toolchain.toml` (1.99.0, the katgpt-rs issue-963 anchor).

## Contract

See `BOUNDARY.md` — the dependency + domain contract. Dependencies: the
public `katgpt-rs` crates + crates.io only; never a sibling `riir-*`
crate, never trained weights (loaders, not weights).

## License

MIT.
