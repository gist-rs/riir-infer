# Bench 006 — plan 611 S5: the three-backend A/B (Cpu vs Metal MSL vs CubeCL) and the T7 verdict

**Status:** DONE 2026-09-26 — VERDICT: the hand Metal lane stays the macOS
default; the CubeCL arm is KEPT opt-in (portability/CI); nothing is deleted
or promoted. The rules were pre-registered at `fac0dfd`, before any run.

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

### Box state (quote beside every number)

M3 Max, macOS, aarch64, release profile, AC power, `pmset powermode 2`
(High Power). Harness binaries built from `fac0dfd` (plus the `c0dfa06`
revert for the barrier-OFF binary) in an isolated worktree.

riir-reflex `scripts/bench_preflight.sh` **REFUSED, on load only**:

```
PROVENANCE: power=AC Power load=10.81 swap=3386.19M canary=125.3us/best5 powermode=2(high)
```

- The GPU canary passed (125.3 µs vs the 141 µs reference: not throttled).
- Load during the three runs was 9.5 → 7.1, ambient: Zed ~170% CPU,
  fskitd, an active remote-desktop session. It did not settle.
- **These are loaded-box numbers.** The paired, permutation-balanced design
  is what keeps the RATIOS interpretable; the absolute p50s drift ~30% run
  to run (e.g. cubecl seq 512: 683 / 1082 / 991 ms).
- The CPU arm is the most load-sensitive column, since rayon competes with
  the ambient load. A quiet box would make the CPU faster and move every
  cubecl/cpu ratio UP.

### Table 1 — encoder ladder (english), paired medians, 3 runs

`c0dfa06` = the Issue 019 Metal barrier. "wins" = rounds out of 12 where
CubeCL was faster.

| seq | cubecl/metal, barrier ON r1 · OFF · ON r2 (wins) | cubecl/cpu, ON r1 · OFF · ON r2 (wins) | p50 ms, ON r1 (cpu · metal · cubecl) |
|---|---|---|---|
| 16 | 5.358 · 5.486 · 5.453 (0/12 all) | 0.631 · 0.603 · 0.559 (12/12 all) | 72.6 · 8.63 · 46.0 |
| 54 | 6.232 · 6.320 · 6.586 (0/12 all) | 0.698 · 0.588 · 0.538 (12/12 all) | 121.3 · 13.6 · 84.7 |
| 128 | 6.755 · 7.249 · 7.768 (0/12 all) | 0.719 · 0.649 · 0.601 (12/12 all) | 240.4 · 26.3 · 168.5 |
| 317 | 6.593 · 7.673 · 7.213 (0/12 all) | 0.710 · 0.899 · 0.894 (12 · 12 · 11/12) | 603.9 · 69.4 · 425.4 |
| 512 | 6.115 · 8.443 · 8.427 (0/12 all) | 0.643 · 0.955 · 0.898 (12/12 all) | 1073.9 · 113.1 · 683.3 |

### Table 2 — agent `system_one` (typed; mid question seq_len 179)

Natural postures: Metal runs `5q_mid` packed; CubeCL and Cpu run the
per-question loop. The inline agreement asserts passed in every run
(argmax exact; rounded probabilities within 1e-3 of Cpu).

| case | cubecl/metal ON r1 · OFF · ON r2 (wins) | cubecl/cpu ON r1 · OFF · ON r2 (wins) | p50 ms, ON r1 (cpu · metal · cubecl) |
|---|---|---|---|
| 5q_mid | 7.019 · 7.425 · 7.330 (0/12 all) | 0.907 · 1.061 · 1.005 (12 · 1 · 3/12) | 1578.7 · 206.5 · 1455.8 |
| 1q_ctrl | 6.099 · 5.555 · 5.119 (0/12 all) | 0.747 · 0.761 · 0.751 (12 · 12 · 11/12) | 385.9 · 52.8 · 292.5 |

### Verdict (read against the pre-registered rules)

- **The hand Metal lane STAYS the macOS default.** CubeCL wins no cell, let
  alone a regime: 5.1–8.4× slower, 0/12 wins in every row of every run. The
  Metal lane's lead is the reflex 020 optimization campaign (split-K, shape
  rule, flash-attn rungs, fold epilogues, packed attention) against one
  clean portable implementation. As plan 611 predicted, the first CubeCL arm
  does not approach it. The margin is ~6× where plan 611 hoped for "near",
  so there is no headline finding.
- **The CubeCL arm is KEPT opt-in** (`laya-riir-cubecl`; not default, not
  in any release set; G5 armed in riir-reflex `ccb5bd0`). The delete rule
  needs cubecl/cpu ≥ 1.00 in EVERY regime, and it fails:
  - The short ladder (16/54/128, three adjacent cells) is a CubeCL regime
    win over Cpu in all three runs (0.54–0.72, 12/12 each).
  - `1q_ctrl` is too (0.75–0.76, ≥ 11/12).
  - The long ladder (317/512) and `5q_mid` are ties or losses on the loaded
    box.
  - Load caveat: the short-regime margin is 28–46%, so the CPU would have to
    get 1.4–1.85× faster on a quiet box to flip it. That is not near the
    threshold, but it is a loaded-box verdict, and re-reading on a quiet box
    is cheap (one command).
- **Promotion:** none, by rule. There is no regime win over Metal.
- **The GAP kernels stay engine-side:** the two-pass mean-centered LayerNorm
  (`LayerNormMeanBatchedCubeCL`), the row-softmax (fenced at `2a34bd3`), and
  the encoder-lane permutation/rope/gather kernels. This is the part of T7
  that pays whatever the A/B said.
- **The barrier question (Issue 019)** was resolved separately: the
  barrier was reverted at `4205c12` (necessity and premise both refuted),
  so the **barrier-OFF column is the shipped configuration**. This table
  could not have priced it anyway: ON and OFF were separate binaries, not
  interleavable in-process, and run-to-run drift (up to ~30% in the long
  cells) swamps any per-rebind cost. The verdict above holds in every
  column.
