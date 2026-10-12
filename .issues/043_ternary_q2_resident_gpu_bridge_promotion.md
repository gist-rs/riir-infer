# Issue 043 — `ternary_q2_resident` promotion: GPU uploaders accept raw-Q2 projections

**Status:** OPEN — filed 2026-10-13 from Bench 879 (opt-in verdict). Blocks promoting `ternary_q2_resident` into `deltanet_ternary_inference`.

## Context

`ternary_q2_resident` (Bench 879, `4d3988d`) keeps Q2_0 projections in GGUF
block layout (`ProjWeights::TernaryQ2`) on the raw-Q2 q8 kernel: bit-identical
to the bit-plane q8 arm, Bonsai-2-27B tg 0.931× (+7–8%), memory-neutral. It
stays OPT-IN because every GPU uploader takes `&TernaryGroupWeights` and
fails closed on a non-plane projection — promoting would turn every GPU build
loading through `load_qwen_deltanet_ternary_weights_gguf` into a panic:
`q4_refuse` + Metal `register_layer` (`ternary_deltanet_gpu_forward.rs`),
cudarc `proj_of`, `vram_budget::proj_gpu_bytes` (now loud wildcard arms).

## Tasks

- [ ] T1 — a plane view for GPU upload: `ProjWeights::planes() -> Cow<TernaryGroupWeights>`
  (borrow for `Ternary`, `TernaryQ2Weights::to_group()` for `TernaryQ2`), routed
  through the uploaders; pointer-keyed caches (`gemv_ternary_cubecl`,
  `gemv_ternary_cuda_raw`) must key on the SOURCE projection, not the
  temporary planes (a Cow-owned buffer's pointer is reused after drop).
- [ ] T2 — f32 arm on a Q2 projection (`set_ternary_q8_act(false)` panics
  today): either a raw-Q2 f32 kernel or a documented reload-with-planes path.
- [ ] T3 — re-run `ternary_q2_resident_goat` + one GPU forward smoke on the
  M3 (Metal) with the feature on; then promote (imply from
  `deltanet_ternary_inference`) per Feature Flag Discipline.
- [ ] T4 (small, independent) — FFN gate+up fused q8 call: both matvecs share
  `x`; one activation quantization + one fork-join per layer instead of two
  (est. ≤ ~1 ms/token of the 55.8 ms FFN — measure before keeping).
