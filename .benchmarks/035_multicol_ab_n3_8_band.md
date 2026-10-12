# Bench 035 — issue 038 step 3: the multicol_ab paired bench, n∈[3,8] band (Ada / RTX 4090)

**Status:** RECORD — EXECUTED 2026-10-13 (idle-loop window unit; the free 4090 lane issue 038
was staged for). Verdict: **multicol beats the n1 incumbent across the WHOLE band, monotone
in n** (median same-rep paired ratio 0.971 → 0.431-equivalent; see table) — the mechanism
works as ported (`48d598e`); **the MMA prefill arm's handler contract dominates both GEMV
arms** (2.0–3.7×) at this contract, and per the module doc the WIRED GPU-resident A/B
(step 4's GOAT surface) is the decider for any MMA-vs-multicol question. The verify path's
incumbent today is the n1 loop — that is the comparison step 4's wiring gates on.

## Box state (provenance — Issue-021 law)

| Axis | Value |
|---|---|
| GPU | RTX 4090 24 GB (sm_89, Ada), KB5121794 post-era |
| GPU util / mem pre-run | 4% / 376 MiB (desktop composition only — `nvidia-smi --query-compute-apps` lists NO CUDA compute process) |
| CPU load / free RAM | 3% / 18.8 GiB |
| Transport | Tailscale SSH (LAN 192.168.1.33 port 22 timed out) |
| Power | AC (desktop) |
| Build | release, fresh 2m15s rebuild (fingerprint refresh), `CARGO_TARGET_DIR=E:\tmp\infer038`, worktree `E:\git\wt038infer` @ `0767c01`+`f1e5319` |
| Config | reps=30 warmup=3, arm order rotated per rep (plan-612 G2 pairing), shapes big (5120×17408) + small (1536×4096), n∈{3..8}, deterministic LCG weights |

## Verdict summary (median same-rep paired ratios — the runbook's verdict form)

| shape | n | mc/n1 | mc/mma | mc µs | n1 µs | mma µs |
|---|---|---|---|---|---|---|
| big | 3 | **0.971** | 2.209 | 391 | 405 | 178 |
| big | 4 | **0.862** | 2.575 | 497 | 551 | 188 |
| big | 5 | **0.859** | 2.965 | 589 | 684 | 198 |
| big | 6 | **0.809** | 3.204 | 678 | 823 | 207 |
| big | 7 | **0.796** | 3.529 | 766 | 977 | 216 |
| big | 8 | **0.766** | 3.746 | 889 | 1121 | 227 |
| small | 3 | **0.707** | 2.000 | 154 | 216 | 77 |
| small | 4 | **0.602** | 2.204 | 176 | 287 | 79 |
| small | 5 | **0.541** | 2.315 | 196 | 362 | 87 |
| small | 6 | **0.506** | 2.458 | 225 | 444 | 91 |
| small | 7 | **0.498** | 2.692 | 249 | 505 | 93 |
| small | 8 | **0.467** | 2.753 | 268 | 572 | 95 |

(weight GB/s columns in the raw output: n1 3→21 GB/s big / 8→3 small; mc 58→27 big /
11→6 small; mma 133→104 big / 22→17 small.)

## Reading

1. **vs the incumbent (n1 loop) — the step-4 gate:** multicol wins every cell, and the
   win GROWS with n exactly as the mechanism predicts (upstream measured 1.75–1.96× pure
   kernel at n∈{4,8}; here the handler contract dilutes big-shape wins to 1.03–1.31×
   because per-call host quantize + upload of 22.58 MiB dominates the arm's wall time,
   while the small shapes — 1.59 MiB — expose the kernel win at 1.41–2.14×).
2. **vs MMA:** at this contract MMA wins the whole band 2.0–3.7×. MMA's arm times the
   shipping prefill sequence (f32 upload → device quantize → GEMM → download); the GEMV
   arms pay host quantize + per-call transfers on their own APIs. This is NOT a
   wired-posture comparison — in a real driver the weights are GPU-resident and neither
   GEMV arm pays per-call upload. Any "MMA should own the band" conclusion needs the
   step-4 wired A/B, which is exactly what the issue gates promotion on.
3. **Handler-contract caveat** (the module doc's own ⚠, quoted): these are standalone-API
   wall times, not end-to-end decode latency; the pure-kernel parity with the n=1 kernel
   is by construction (bit-parity gates, bench 034-class + step-1/step-2 GREEN).

## Consequence for step 4

The dispatch-arm wiring (tree-verify driver + batched decode consumers, n∈[3,8], behind
the standing kill-switch pattern) is justified vs the path's incumbent: the n∈[3,8] band
currently pays n sequential n1 calls; one multicol call beats that at every measured
cell. GOAT-promotion (default-on) stays gated on the wired A/B + G1–G4 per the feature-flag
discipline — opt-in first.

## Refs

- issue 038 (the adoption lane; steps 1+2 GREEN 2026-10-11)
- upstream lead: PrismML-Eng/llama.cpp #306 (pin `2a42998c5`), distill Lead 12
- `crates/riir-infer-gpu/examples/multicol_ab.rs` (the instrument, landed 0.5-stage)
