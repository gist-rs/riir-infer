# Bench 948 — Bonsai-2-27B sink-audit census (Issue 981 follow-up (b): the ternary-body capture hook)

**Verdict: LANDED — the flagship's census cell is UNDECIDED (f_attn_token0 0.1297, between the
chosen cuts 0.10/0.25); the audited policy action is KeepIncumbent (n_sink=4 stands), the safe
posture. G1 bit-identical across three runs. The capture seam is proven by the analytic
zero-weights identity on the ternary body (new lane tests).**

## What landed (the follow-up (b) precondition)

- **`sink_audit` feature now implies `deltanet_ternary_inference`** (the
  `scale_invariant_attn` implication precedent) — the census lane owns the ternary arm. A
  folded (Hadamard) file additionally needs `bonsai2_hadamard` at the RUN's build; the lane
  itself is rotation-agnostic.
- **The capture seam is shared**: `qwen_deltanet_ternary_layer_body`'s attention branch
  captures post-RoPE Q from `scratch.attention.q_buf` after
  `forward_attention_layer_ternary` returns (the same seam contract as the dense body —
  step 5 reads q_buf last, nothing writes it after; the Issue-594 output gate applies at
  step 6, AFTER the attention read, so the audit premise holds on the ternary serving body).
  cfg-gated `sink_audit`; every non-census caller leaves the capture `None` and pays nothing.
- **`run_census_ternary`** drives `forward_qwen_deltanet_ternary` over the same bounded
  HybridCache (Issue-929 sizing) and harvests through the SHARED `harvest_profile` — one
  harvest for both bodies (the ternary path writes f32 post-RoPE K/V into the same KVCache
  layout). The dense `run_census` is byte-identical in behavior (its three existing tests
  pass unchanged).
- **The `sink_audit` bin auto-dispatches** on the `token_embd.weight` tensor dtype
  (Q2_0/PTQ1_0 = ternary; the dense loader mis-shapes the PQ2_0 file with an in_proj_qkv
  row overrun — the dispatch must precede any weight load). The canonical payload is v1
  FROZEN (the hash inputs unchanged); the body lands in the report JSON + stdout only.
- **New lane tests**: `ternary_census_on_zero_weights_matches_the_analytic_uniform_profile`
  (the shared-harvest identity on the ternary body) + `ternary_census_is_deterministic`.
  All 5 census tests pass; clippy clean at `sink_audit` and `sink_audit,bonsai2_hadamard`.

## The measurement

```
model:   ../riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf (7.2 GB, PQ2_0, folded=true)
corpus:  ../riir-train/data/chat_probe (2,713,703 tokens; first 512 consumed)
body:    ternary (16 full-attention layers of 64; GDN layers have no softmax sink)
posture: release, --features sink_audit,bonsai2_hadamard; load 17.1 s, census pass 88.1 s
window:  N=512 → 196,608 observations (16 layers × 24 heads × 512 positions)

f_attn_token0  0.129721   (chosen cuts: ≤0.10 SinkFree, ≥0.25 SinkHeavy)
m_act          68.571     (disclosed, never verdict-bearing)
profile[0..8]  0.129721 0.009775 0.019283 0.006204 0.004223 0.003082 0.003602 0.004471
verdict:       UNDECIDED → KeepIncumbent (n_sink=4 stands; disclosed, never folded)
blake3(payload) dfca3cab96efe9de98e6b6fa04d96aac24e5f2dcdf023351535e6e6aabae786d
```

**G1: three runs, byte-identical blake3.** Report:
`.benchmarks/sink_audit_bonsai2_27b_report.json` (schema sink_audit_v1, `body: "ternary"`).

## Reading (honest)

- **The flagship is NOT SinkFree at the chosen cuts.** The 0.8B qwen35-hybrid read
  0.0764 → SinkFree (bench 947); Bonsai-2-27B reads 0.1297 — 1.7× the free cluster's top
  (0.073) and about half the sunk cluster's floor (0.297). The chosen cuts put it in the
  disclosed gap; the audited policy action (keep the incumbent pin) is IDENTICAL whether
  this is a family/scale trait or a measurement subtlety — no serving decision hangs on
  the classification.
- Plausibility: the paper's clusters are calibrated on Qwen-class ~1.7B dense baselines;
  Bonsai-2 is a 27B ternary hybrid (PQ2_0-quantized weights, Hadamard-folded) — a mid
  reading is a plausible family/scale/quantization effect, and the gate it ships (Issue
  594) is verified present in the ternary body (step 6) and at the GPU epilogue (T1
  sub-check, bench 947). NOT adjudicated as one of the paper's clusters — that is exactly
  what Undecided encodes.
- Position-0 mass dominance holds (0.1297 vs ≤0.019 for positions 1–7): IF the owner ever
  wants a leaner pin, the per-position profile supports `from_audit`'s leading-mass rule;
  Undecided keeps the incumbent by contract until the cuts are re-qualified on a
  ternary-family calibration pair (follow-up (a)'s real ungated arm + this census = the
  pair to calibrate on).

## Box state (disclosure)

M3 Max 16 cores, shared box (a sibling agent's editor dev-loop ran concurrently — the
census verdict is arithmetic, not latency, and G1 passed across contended runs: 3/3
identical blake3). macOS 27.2, on battery + charging during runs 1–3. Census pass timing
(88.1 s) is a throughput note only, never a league figure.

## Gates

- **G1** determinism: PASS (3 runs byte-identical) — dense and ternary lanes both carry
  the same-run-twice unit test too.
- **G2** discrimination: unchanged from bench 947's synthetic arms (planted 0.4716 vs free
  0.0236); the real-pair arms remain follow-up (a).
- **G3** policy correctness: unchanged (bench 947 switch grid; Undecided → KeepIncumbent
  is the disclosed cell, exercised by `from_audit`'s unit tests).
- **G4** alloc: the census capture is bounded and arena-shaped as before; the ternary body
  adds no per-step allocation (capture writes into pre-sized rows).
- Promotion: NOT applicable — the census is an offline diagnostic; the audited-policy
  promotion path still waits on T4's end-to-end cell (follow-up (c)).
