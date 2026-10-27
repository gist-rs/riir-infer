# Bench 907 — Issue 1145 T3: multi-cube norm+rotate staging G2 (idle bmo) + PROMOTION

**Date:** 2026-10-27, 19:15–19:25 (+07). **Tree:** `d1b6156` (the 1145 kernel +
wiring, default-OFF at bench time) with the probe binary built on amo
(same aarch64-apple-darwin, macOS 27.0.1 26A434 both boxes) and run on bmo —
bmo has no CLT in the SSH context (the box rebooted 1.5 h before the run;
`xcode-select` resolves nothing headless), so the established
build-on-amo/run-on-bmo pattern ships the release test binary; Metal kernel
compilation happens at runtime on bmo (OS Metal compiler — no CLT needed to
run). **Model:** `/Users/katopz/models/Ternary-Bonsai-2-27B-PQ2_0.gguf`
(7,206,168,928 B — bmo's copy of the league model; the repo-relative default
path does not exist on bmo). **Posture:** `gpu,bonsai2_hadamard` release,
`I1139_STEPS=64` (8 warm-up + 64 timed, ctx 72) — the Issue-1139 census
posture, digest-pinned.

**Box state (bmo, read at launch):** load 1.16/1.36/1.43, up 1:30,
`pmset -g therm` clean (no thermal/performance warnings), no sibling GPU
compute (`ps aux` clean of cargo/llama/riir/bench/train). amo pre-check run
under the same tree: 17.74 default / 17.35 MULTI (single runs, digest-pinned
both).

## Arms

A = default (`RIIR_NORM_ROT_MULTI` unset → incumbent
`ResidualAddRmsNormCubeCL`(mlp4) + `RotationCubeCL::launch_forward_copy`(r4),
2 launches per norm site). B = `RIIR_NORM_ROT_MULTI=1` → the multi-cube
fused kernel (one launch of `dim/hblock` = 5 cubes at 5120/1024, double-
buffered `x`/`x_alt`, host-side swap; the Issue-1136 no-owner race avoided
by construction). ABBA order across 3 pairs (A B B A A B), one process per
arm (the env is OnceLock-read).

## G1 — bit-identity (decode level)

`logits stream digest d06bffced92427cb · last token 279` in **all six arms**
(the 1139 pin, unchanged) + the `[norm_rot_multi] REACH` line in every B arm
(the Bench-768 vacuous guard — the only proof the arm dispatched, since the
outputs are bit-identical by construction). Kernel-level G1: 3 tests
to_bits-exact vs the incumbent pair on ALL THREE outputs (post-add x via
x_dst, norm_out, rot_out) at 5120/1024 (5 cubes), 2048/1024, and the edge
rows (zero-pin, ±4000, alternating) — 307/307 lib green at ternary_gemv.

## G2 — paired ABBA (idle bmo)

| pair | A default ms/token | B multi ms/token | Δ |
|---|---|---|---|
| r0 | 17.629 (56.72 tok/s) | 17.035 (58.70) | **+3.49%** |
| r1 | 17.458 (57.28) | 17.032 (58.71) | **+2.50%** |
| r2 | 17.518 (57.08) | 17.097 (58.49) | **+2.46%** |

Medians: A 17.518 → B 17.035 = **+2.83%**, 3/3 positive — **G2 PASS** at the
≥1.5% bar. G3: the wiring is decode-body-only and env-gated; prefill /
training / capture never dispatch the arm (structurally unchanged — the
digest identity covers the decode stream; 307/307 lib + workspace clippy
green). G4: zero per-token allocation — `x_alt` is one construction-time
buffer, the swap is a host-side Handle swap. Modelless ✓ (kernel scheduling
only).

Against the issue's ceiling (≈0.71 ms/token ≈ 4%): the gain is +0.48 ms/token
(2.83%) — the redundant whole-vector reduction per cube eats ~⅓ of the
theoretical launch-overhead saving, as priced by the issue ("cache-resident
after the first cube").

## Post-promotion confirm (amo, `RIIR_NORM_ROT_MULTI` default = ON)

- default (now multi): REACH fired, digest `d06bffced92427cb`, 17.168 ms/token
- `RIIR_NORM_ROT_MULTI=0` (demote hatch): no REACH, digest same, 17.622 ms/token —
  the incumbent pair byte-identical

## Verdict

**GOAT G1–G4 PASS, modelless → PROMOTED DEFAULT-ON 2026-10-27** (env
`RIIR_NORM_ROT_MULTI` demote-only, `dense_pair_env_semantics`: unset = ON,
`=0` restores the incumbent pair byte-identically). Scoreboard: Bonsai-2
1-box decode on idle bmo **17.57 → 17.04 ms/token (58.7 tok/s)** — the
Issue-1139 residual vs llama.cpp's 16.4 closes **1.07× → 1.04×**. 128
dispatches/token removed (the norm-site staging class), the largest single
rung of the 1139 launch-floor census. 1139's remaining above-bar classes
(`gdn_out_rot`, `gdn_proj_dense`, `ffn_rotate`) stay not-routed (launch-floor
attributed, fusion levers measured sub-bar in the 1139 T4 record).

**Refs:** `.issues/1139_m5_ultra_decode_gap_vs_llamacpp.md` (T4 census + this
issue's routing), `crates/riir-infer-gpu/src/norms_cubecl.rs`
(`residual_add_rmsnorm_rotate_multi_f32`), HISTORY.md entry.
