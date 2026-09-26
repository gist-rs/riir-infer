# Bench 006 — plan 611 S5: the three-backend A/B (Cpu vs Metal MSL vs CubeCL) and the T7 verdict

**Status:** PRE-REGISTERED 2026-09-26. The verdict rules below were committed
before any number was read. Results pending.

Feeds plan 611 S5/S6 → riir-reflex Issue 008 T7 (the op-layer
unification) and Issue 998 S8 (mirror).

## Instrument

`crates/riir-infer-laya/tests/backend_ab.rs` (measurement-only,
`#[ignore]`; row requires `laya-riir-metal` + `laya-riir-cubecl`):

```sh
cargo test --release -p riir-infer-laya --features laya-riir-metal,laya-riir-cubecl \
    --test backend_ab -- --ignored --nocapture
```

- One process holds the three backends side by side.
- Every round runs all three arms and cycles through all six orders
  (12 rounds per row), so each arm sits in each position equally often.
- Verdicts use the MEDIAN of per-round PAIRED ratios (cubecl/metal and
  cubecl/cpu), never the absolute p50s.
- Table 1 is the encoder ladder: `Encoder::forward` plus read-back at seq
  16 / 54 / 128 / 317 / 512 on the english checkpoint.
- Table 2 is agent `system_one` on the typed checkpoint:
  - `5q_mid` runs each backend in its natural posture: Metal packed;
    CubeCL and Cpu per-question loop, since CubeCL's
    `supports_packed_attention` is false.
  - `1q_ctrl` is the loop path everywhere.
- Before timing, table 2 asserts output agreement against Cpu: the argmax
  must be exact and the rounded probabilities within 1e-3.
- Box state is quoted beside the numbers (preflight + load). The CubeCL
  rows are measured at BOTH settings of the Issue 019 question (the
  `c0dfa06` wgpu-hal Metal barrier ON and OFF). The Metal MSL and Cpu lanes
  do not go through wgpu, so only the CubeCL column can move.

## PRE-REGISTERED verdict

- A **regime** is ≥ 3 adjacent ladder lengths, or one agent case. **Winning
  a regime** means a paired-ratio median < 1.00 with ≥ 9/12 wins in EVERY
  cell of it. One winning cell is never a regime.
- **The hand Metal lane** is deleted ONLY if CubeCL wins EVERY regime (the
  whole ladder AND both agent cases). Otherwise it stays the macOS default.
- **The CubeCL arm** is DELETED if it loses to the CPU lane in every regime
  (cubecl/cpu median ≥ 1.00 everywhere), because a GPU lane slower than
  the CPU lane has no portability value. Otherwise it is KEPT opt-in behind
  `laya-riir-cubecl` as the portability/CI arm: not default, not in any
  release set, G5 gate armed.
- **Promotion:** none from this table. A CubeCL regime win over Metal names
  a candidate for a future gated plan and flips nothing.
- **The CPU lane** is the reference oracle and is never deleted.
- **The GAP kernels** (mean-centered LayerNorm; the non-causal attention
  composition) stay engine-side whatever the outcome.

## Results

_Pending — filled from the run, with box state._
