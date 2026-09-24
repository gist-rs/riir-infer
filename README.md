# riir-infer

The LLM inference substrate: weight loading, quantization, model
architectures, and compute kernels for model-based inference — upstream of
every engine, game, and application concern.

## The crate

`riir-infer-core` (repo root):

`riir-infer-gpu` (`crates/riir-infer-gpu`) — the GPU runtime + kernel layer: device context, buffer helpers, pool-poison detector, the CubeCL runtime (Metal/WGSL/SPIR-V via wgpu; CUDA behind `cuda_backend`), and the first kernel family. Optional companion crate — CPU-only consumers never resolve the GPU dep tree.

`riir-infer-laya` (`crates/riir-infer-laya`) — the pinned-checkpoint encoder lane: a Python-JSON byte writer (ungated), the tokenizer/config/weights (locate → verify → download) substrate, and a flat-`Vec<f32>` forward with CPU (`gemm`) and macOS Metal (MSL) backends behind the `laya-riir` / `laya-riir-metal` features. Optional companion crate — lane-free consumers never resolve the tokenizers/gemm tree.

- **Loaders** — GGUF + safetensors, mmap-backed, BLAKE3-checkable.
- **Quantization** — the q2k…q8kv GGUF family, q2_0 ternary, PTQ /
  TurboQuant paths.
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
