# Bench 002 — EXL3 GPU dequant: parity oracle + 4090 throughput (Issue 001 T7b)

**Status:** COMPLETE (2026-09-24) · method: measured GPU kernel runs + full
CPU-oracle compares · evidence chain: Issue 001 §16 · kernels:
`crates/riir-infer-gpu/src/exl3_dequant_cubecl.rs` (feature `exl3_gpu`)

## What this measures

The T7b deliverable: the 4090 GPU arm of the EXL3 dequant pipeline — three
CubeCL kernels (trellis decode; left Hadamard + row scale; right Hadamard +
column scale) producing dense f32 `W[in][out]` from a quantized layer's
trellis bitstream + channel scales, streamed in 128-column chunks
(VRAM-bounded intermediates, layer-size-independent).

Specimen: the LEAGUE MODEL's author pack — `turboderp/Qwen3.8-27B-exl3`
@ `SC_4.00bpw_H5_V6` (Bench 001's pack, sha256-verified there) — the 5
sample classes of the §15 CPU bench, smallest member of each.

Box: 4090 (24 GiB), i7-13700K host, Windows 11, AC power, GPU otherwise
idle. Release builds. Both CubeCL backends measured: `wgpu<spirv>` (default)
and native CUDA (`cuda_backend`).

## The two-tier parity oracle (and why not bit-parity)

Bit-parity against the CPU scalar reference is achieved for the **trellis
decode stage** — it contains zero floating-point arithmetic (integer
prefix/window math + the CPU's own LUT bytes uploaded verbatim). Pinned
synthetically across K∈{3,5}+cb0/mcg and K=4.5+mul1 (`gpu_decode_matches_cpu_bit_exact`)
and on the real pack (`real_pack_decode_bit_exact_probe`:
**147456/147456 probe elements bit-exact** on `model.visual.blocks.0.attn.k_proj`).

The **Hadamard stages cannot be bit-exact through CubeCL**: both available
backends contract `a*b + c` into fused multiply-add (the CUDA lane generates
CUDA C++ with plain `*`/`+` operators compiled by NVRTC at its default
`-fmad=true`; the wgpu lane contracts identically — verified by identical
divergence bits on both backends), and there is no per-kernel opt-out
(`CompilationOptions` carries no fmad flag; a global `-fmad=false` would
de-optimize every FMA-contracted GEMV kernel in the workspace — rejected).
FMA is one-directionally MORE accurate (one rounding per step instead of
two) — the same divergence class T4b itself adjudicated for the reference
implementation's own fp16 Hadamard intermediates (§12.6: rel-Frobenius
2.96e-08, "the expected class").

Gates (scale-aware; a near-zero sum whose ~1e-8 FMA error crosses the sign
makes raw ulp counts meaningless): max |diff|/max|W| ≤ 1e-5 and
rel-Frobenius ≤ 1e-5. Measured on every class, both backends:

| class | weights | rel-Frobenius | max \|diff\|/max\|W\| |
|---|---:|---:|---:|
| self_attn.o_proj (K4) | 31.5M | 2.60e-07 | 3.06e-07 |
| mlp.down_proj (K3) | 89.1M | 2.61e-07 | 2.01e-07 |
| linear_attn.out_proj (K5) | 31.5M | 2.59e-07 | 2.64e-07 |
| mlp.gate_proj (K3) | 89.1M | 2.61e-07 | 1.04e-07 |
| lm_head (K5) | 1.27B | 2.60e-07 | 5.03e-07 |

The uniform ~2.6e-7 is the FMA signature (deterministic, backend-stable);
gates hold with 20-100× headroom — a real regression (wrong codebook value,
wrong placement, missing scale) misses by orders of magnitude.

## Throughput (the T7b headline)

Wall = full driver call (scale expansion + trellis/LUT/H uploads + all chunk
launches + f32 readback). Kernel-only = the (reps−1)-run differential, which
cancels upload + readback (the serving-relevant number — weights stay
GPU-resident there). CPU-parallel = the T7a arm (16-thread rayon).

**Native CUDA backend (`cuda_backend`)** — the 4090 production lane:

| class | weights | GPU wall | GPU kernel-only | CPU-parallel | wall speedup |
|---|---:|---:|---:|---:|---:|
| self_attn.o_proj (K4) | 31.5M | 0.042 s (743 Mw/s) | 0.007 s (4749 Mw/s) | 0.473 s (66.5) | 71.4× |
| mlp.down_proj (K3) | 89.1M | 0.166 s (538 Mw/s) | 0.018 s (4983 Mw/s) | 1.292 s (69.0) | 72.2× |
| linear_attn.out_proj (K5) | 31.5M | 0.045 s (701 Mw/s) | 0.006 s (4955 Mw/s) | 0.459 s (68.5) | 72.4× |
| mlp.gate_proj (K3) | 89.1M | 0.167 s (534 Mw/s) | 0.018 s (5051 Mw/s) | 1.318 s (67.6) | 74.7× |
| lm_head (K5) | 1.27B | 1.940 s (655 Mw/s) | 0.217 s (5848 Mw/s) | 18.876 s (67.4) | 86.8× |

wgpu<spirv> backend (same parity, softer kernel throughput — lm_head
kernel-only 2757 Mw/s, o_proj 4645): see the test run record in Issue 001
§16; native CUDA is the lane the engine's 4090 decode path already uses.

Full-model extrapolation (pack ≈ 32.0 B quantized weights, Bench 001):
at the CUDA kernel-only rate (~4.8-5.8 Gw/s) ≈ **5.5-6.7 s** of pure GPU
dequant; at wall (~0.53-0.66 Gw/s, readback-dominated) ≈ 50-60 s end-to-end
vs the CPU arm's ~6.7 min. Per-layer dequant at load time is tens of
milliseconds — the load-path cost of an EXL3 27B model on the 4090 is now
bounded by readback bandwidth, not compute.

## Notes

- Dispatch is grid-strided (≤65535 workgroups, wgpu's per-dimension cap) —
  real layers need up to ~4.9M tile-threads; one launch covers any layer.
- VRAM peak ≈ 3 × 256 MB chunk buffers + trellis + tables — layer-size
  independent; the 1.27B-weight lm_head dequantizes in ~19 chunks without
  materializing the 5 GB f32 output on the GPU at once.
- The chunked path is parity-pinned against the single-chunk path
  (`gpu_chunked_matches_whole`) — chunking moves no bits.
- Findings paid for along the way (recorded so the next kernel doesn't
  re-pay): (1) a column-chunk readback is `[in × cols]` row-major — it must
  be scattered per-row into the output, never contiguously copied (the
  first draft produced exactly one correct row); (2) the parity harness's
  first-bit-mismatch diagnostic hid a ~100% element miss until the
  zero-count probe was added — count and localize, don't sample.
