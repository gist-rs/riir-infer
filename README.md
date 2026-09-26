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
- `katgpt-attn` (opt-in `flashmemory_gqa`) — FlashMemory sparse GQA.

## The crate

`riir-infer-core` (repo root):

`riir-infer-gpu` (`crates/riir-infer-gpu`) — the GPU runtime + kernel layer: device context, buffer helpers, pool-poison detector, the CubeCL runtime (Metal/WGSL/SPIR-V via wgpu; CUDA behind `cuda_backend`), and the kernel families (GPU transpose; EXL3 trellis decode + block-Hadamard behind `exl3_gpu` — decode bit-exact vs the CPU reference, Hadamard gated in the FMA-contraction class, 71-87x wall over the CPU arm on the 4090). Optional companion crate — CPU-only consumers never resolve the GPU dep tree.

`riir-infer-laya` (`crates/riir-infer-laya`) — the pinned-checkpoint encoder lane: a Python-JSON byte writer (ungated), the tokenizer/config/weights (locate → verify → download) substrate, and a flat-`Vec<f32>` forward with CPU (`gemm`) and macOS Metal (MSL) backends behind the `laya-riir` / `laya-riir-metal` features, plus an opt-in portable CubeCL/wgpu backend (`laya-riir-cubecl`) over `riir-infer-gpu`'s op layer. The hand-tuned Metal lane is the default: in Bench 006 the CubeCL arm ran 5–8× slower than Metal and beat the CPU lane on short sequences. Optional companion crate — lane-free consumers never resolve the tokenizers/gemm tree.

- **Loaders** — GGUF + safetensors, mmap-backed, BLAKE3-checkable.
- **Quantization** — the q2k…q8kv GGUF family, q2_0 ternary, PTQ /
  TurboQuant paths, EXL3 trellis decode + block-Hadamard (opt-in `exl3`;
  the multi-shard zero-copy pack reader, the CPU reference, and the
  bit-identical CPU fast arm — codebook LUT + rayon, 10.6-11.4x — land
  here; the GPU kernels live in `riir-infer-gpu` behind `exl3_gpu`.
  `Exl3Pack::open` is era-gated fail-closed on the pack's
  `quantization_config.version` against the validated set (`["1.4.2"]`)
  — legacy-era packs silently decode wrong (issue 001 §12.7, record in `.docs/001`); a
  deliberate unvalidated-era read is `Exl3Pack::open_unverified_era`).
- **Architectures** — gemma / llama / ternary / wall layers, deltanet,
  transformer, rope, speculative-decoding types (dflash).
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

Toolchain pinned by `rust-toolchain.toml` (1.98.1).

## Contract

See `BOUNDARY.md` — the dependency + domain contract. Dependencies: the
public `katgpt-rs` crates + crates.io only; never a sibling `riir-*`
crate, never trained weights (loaders, not weights).

## License

MIT.
