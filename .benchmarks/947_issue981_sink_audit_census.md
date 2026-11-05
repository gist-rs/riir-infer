# Bench 947 — Issue 981 sink-audit census: the gated-hybrid flagship reads SinkFree (T1 real-model cell)

**Status:** DONE 2026-11-05 (zed-m3-i981sinkaudit; katgpt-rs Issue 981 T1 + T2 + T4 reduced cell).
**Posture:** m3 (M3 Max, 16 cores), macOS 27.2, AC power, box load 4–9 during the census runs and
up to ~13 during the late needle-gate runs (the timed cells below are CORRECTNESS/retention reads,
not latency claims — the prefill wall-times moved with load and are quoted as evidence of run
completion only). Census bin: `--release`, feature `sink_audit`.

## What ran

1. **The census** (`sink_audit` bin, `--n 512`, corpus `chat_probe`, 2,713,703 tokens encoded;
   first 512 consumed):

   | checkpoint | arch | attn layers | f_attn_token0 | m_act | verdict | audited pin |
   |---|---|---|---|---|---|---|
   | **Qwen3.5-0.8B-Base-Q8_0** | qwen35 hybrid, gated (Issue 594 shape: attn_q = 2·q_dim) | 6 of 24 | **0.076425** | 35.96 | **SinkFree** (cut ≤ 0.10) | n_sink 4 → **1** (floor), window 256 carried |

   The profile drops cleanly: position 0 at 0.0764, positions 1–7 at 0.0039–0.0067 — no hidden
   secondary/delimiter sink concentration at 1–3 (the profile exists precisely to catch that).

   **This reproduces the paper's class ordering on a REAL gated checkpoint**: the paper
   (arXiv:2505.06708, Table 4) measures gated models at F-Attn 0.048–0.073 vs ungated
   0.297–0.467. The gated qwen35-hybrid reads 0.0764 — at the gated cluster's upper edge, and
   far below the sunk cluster. **G2 real-half: the gated arm of the pair.**

2. **G1 determinism, real model**: two census runs over the same checkpoint + corpus + window →
   BLAKE3 `c3467f92657f86b59d1615122a54b6cce1e882ab398cf21a157e590db680dffb` BOTH runs
   (byte-identical canonical payloads).

3. **T4 reduced-scale cell** (needle_eviction_gate @ 8k context, 4 needles, one budget arm,
   census armed inline via `--sink-audit 512`):

   | rig | policy | genNLL | retrieval | meanΔ |
   |---|---|---|---|---|
   | 25% budget, maxrecent | incumbent pin 4 | 3.191 | 2/4 | +0.599 |
   | 25% budget, maxrecent | **audited pin 1** | 3.191 | 2/4 | +0.599 |
   | 2% budget, maxrecent | no pin (`--sinks 0`) | **3.525** | 0/4 | **+1.669** |
   | 2% budget, maxrecent | **audited pin 1** | **3.523** | 0/4 | **+1.675** |

   - The 25% pair is BIT-IDENTICAL (same evicted=37032, same per-needle Δ to 7 decimals): at that
     budget the head rows survive on score alone, so the 4→1 pin reduction is quality-neutral —
     the T4 neutral direction, with the caveat that the cell never stressed the pin.
   - The 2% pair DIFFERS (the wire-through consumed the audited policy — position 0's retention
     moved genNLL 3.525 → 3.523 and needle-0 Δ 2.4627 → 2.4743): the audit's
     census → verdict → `from_audit` → `EvictorState` chain is proven end-to-end on the real gate.

## What did NOT run (disclosed, not folded)

- **The ungated G2 arm**: no ungated local checkpoint has a CPU forward. The dense qwen3.8 lane
  (the natural ungated read — no attention output gate in `qwen38_dense_cubecl`; only SwiGLU MLP
  gates) is GPU-only; the paper's released gated/ungated HF pair needs its own loader. The
  synthetic discrimination arms (bench_981, planted 0.4716 vs free 0.0236 — inside the paper's
  clusters) carry G2 on the arithmetic; the real-pair fixture is the follow-up.
- **Bonsai-2-27B-PQ2_0 through the dense census**: the dense qwen35 loader mis-shapes the
  ternary file's DeltaNet projections (in_proj_qkv row overrun panic, dot.rs:879) — Bonsai-2's
  serving path is the TERNARY loader + ternary forward body, which the capture hook does not
  ride yet. The flagship census is that hook's follow-up.
- **The 64k-class T4 cell**: the reduced cell (8k) is the runnable scale on this box in this
  window (~20–35 min/arm under load); the full-scale long-context quality cell is
  owner-adjacent runtime.

## Records

- katgpt-rs `923aef6a4` — the audit vocabulary + selector + bench_981 (all gates pass,
  alloc_tracking G4).
- riir-infer this commit — the census lane (capture + harvest + bin) + the needle-gate
  wire-through + the census unit tests (zero-weights analytic profile, determinism, short-corpus
  refusal).
