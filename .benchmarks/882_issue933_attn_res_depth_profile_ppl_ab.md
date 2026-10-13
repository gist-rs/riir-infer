# Bench 882 — katgpt-rs Issue 933 T4: fixed-query modelless AttnRes ppl A/B on gemma-2-2b-it-f16 (the E5 probe)

**(Renumbered from 881: a concurrent session allocated 881 (`881_plan622_tau_sweep_gpu.md`) on origin while this run was in flight — their committed document keeps the number per the collision rule.)**

**Date:** 2026-10-13 (~07:2x +07) · **Box:** 4090RTX host (shikuwa, i7-13700K, 31.8 GB RAM) — **CPU-only probe** (GPU untouched)
**Provenance:** cpu_load=7% ram_free=21.6GB/31.8GB at launch (E:\tmp\933_run.meta.txt); no cargo/concurrent compute during the run (the box's only other activity: idle-loop doc work).

## The question (pre-registered)

katgpt-rs Issue 933 T4 (Research 619 E5, MSAR arXiv:2610.13630): can the
AttnRes mechanism be applied MODELLESSLY to a checkpoint that has NO AttnRes
weights (Gemma-2-class) — the learned per-sublayer `score_weight` query
replaced by a deterministic exponential-depth profile `score[i] = -λ·age[i]`?
MSAR's own conv-removal ablation (1.847→1.862 — value survives query
degradation) motivated the probe; the adversarial prior: real checkpoints
need task-specific lookback fixed queries cannot express.

**Bars (declared before the full run):** GREEN = some finite λ keeps
|Δppl| ≤ 1.0% vs the stock forward → the modelless posture is viable as an
opt-in candidate. RED = only the disarmed λ=∞ stays within tolerance →
record and close (the learned queries ARE the mechanism).

## Instrument

- **Arms:** `base` = stock `forward_gemma2_f16` (the shipped path) vs
  `forward_gemma2_f16_attn_res` (the fork — f16 layer stack verbatim + the
  three Kimi-semantics wiring points: b0 push at token start, depth-profile
  mix at BOTH half-layer inputs, block-sum push at block boundaries;
  S=4 layers/block per Kimi-K3; substrate
  `katgpt_transformer::attn_res::apply_attn_res_depth_profile`,
  katgpt-rs `ca35f9ca9`).
- **Frozen read:** chat_probe corpus (dir, blake3
  `44aabe8e…52a52`), first 4096 tokens, 4 chunks of `[BOS]+1024`, KV reset
  per chunk, teacher-forced next-token NLL at every position (4095–4096
  scored), paired per-token vs base on identical positions.
- **λ sweep:** ∞ (disarmed sanity) + 4, 3, 2, 1.
- **Gates before the run (all PASS):** substrate 5 new tests (λ=∞ bitwise
  identity; λ=0 exact uniform mean; recency monotone in λ; empty state;
  determinism) — 12/12 attn_res gates at `transformer_attn_res`; fork
  module 4/4 (`lambda_inf_is_bitwise_identical_to_stock` on synthetic
  5-layer weights + boundary accounting + cross-run determinism + finite-λ
  changes logits); clippy `-D warnings` clean both crates at the posture.

## Results

| arm | mean-NLL | ppl | Δppl% | mean-ΔNLL | worse% | max\|ΔNLL\| | tok/s |
|---|---|---|---|---|---|---|---|
| base | 2.42125 | 11.2599 | — | — | — | — | 8 |
| λ=∞ | 2.42125 | 11.2599 | +0.00% | +0.00000 | 0.0 | 0.0000 | 8 |
| λ=4 | 2.41477 | 11.1872 | **−0.65%** | −0.00648 | 59.1 | 2.5523 | 8 |
| λ=3 | 2.42115 | 11.2588 | −0.01% | −0.00010 | 62.5 | 4.0620 | 8 |
| λ=2 | 2.61650 | 13.6877 | +21.56% | +0.19525 | 71.6 | 6.1106 | 8 |
| λ=1 | 6.50637 | 669.3950 | +5844.93% | +4.08512 | 94.5 | 23.5792 | 8 |

(The 512-token smoke the run was shaped on read λ=4 −1.18% — the 4096-token
read converges to −0.65%; both directions agree, the magnitude tightens.)

## Verdict

**GREEN-candidate at λ∈{3, 4}** — the pre-registered bar (some finite λ with
|Δppl| ≤ 1.0%) is met TWICE, one of them improving over base. The λ landscape:

- λ=∞ exact identity (ΔNLL ≡ 0 at every one of the 4096 scored positions —
  the wiring verified on the real checkpoint).
- λ∈{3,4}: the within-tolerance band. λ=4 carries a real but thin net gain
  (mean ΔNLL −0.0065 over 59.1% worse-tokens — the 40.9% better tokens
  out-improve the worse ones; max per-token \|Δ\| 2.55 nats — this is NOT a
  no-op posture).
- λ≤2: the cliff (+21.6% at λ=2, catastrophic below) — mixing weights
  beyond a few percent toward past block sums destroy a residual stream
  the checkpoint never trained for it.

The mechanism story: at λ=4 the mix is x ← 0.982·x + 0.018·(recent block
sums) — a tiny depth-axis low-pass. The MSAR motivation holds in the weak
form: the mechanism TOLERATES query degradation (fixed depth profile) on a
non-AttnRes checkpoint. The surprising direction (slight IMPROVEMENT) is
consistent with mild smoothing of gemma-2's noisy residual/outlier
structure but is UNCONFIRMED — single corpus, single offset, 4096 tokens.

**Honest scope:** this is a VIABILITY signal, not a promotion claim. The
adversarial prior's blanket form ("the learned queries ARE the mechanism")
is refuted at the tolerance bar, but the improvement itself needs the
de-biasing reads before any production claim: (a) a second corpus slice /
offset, (b) a longer-token read for CI tightness, (c) a λ fine-sweep
(3.5–5) to locate the optimum, (d) a benchmark-regression read per the
issue's GREEN clause. The opt-in posture (riir-infer feature
`issue933_attn_res_probe`, default-off) IS the GREEN deliverable; any
promotion is owner-gated behind the confirm reads.

## Post-run notes

- Runtime ~70 min wall for 6 arms × 4096 tokens (≈8 tok/s/arm, CPU f16
  path, 26 layers × 2304 dims × 256k vocab).
- The corpus read is the chat_probe DIRECTORY (2.7M tokens available;
  first 4096 taken — deterministic given N; blake3 in the run header).
- The substrate fn (`apply_attn_res_depth_profile`) is gated by the
  EXISTING `transformer_attn_res` feature — no katgpt-rs feature-count
  change; the fork + bin live behind the new riir-infer
  `issue933_attn_res_probe` (opt-in, measurement-only).
- Committed: katgpt-rs `ca35f9ca9` (substrate + tests); riir-infer
  `<this commit>` (fork + gates + bin + this record).
