# Bench 911 — Plan 623 T0.3: the one-session 1-box arms (bmo idle: llama.cpp vs MLX vs riir-infer)

**Status:** RECORD — T0.3 CLOSED; decode arm competitive (0.87–0.98×), prefill arm LOSES 6–8× → the gap filed as `.issues/1146_prefill_throughput_gap_1box.md`
**Date:** 2026-10-28 (sessions 02:18–02:58 +07, four selective reruns of ONE harness)
**Session:** zed-amo-i623-t24; bmo driven over ssh (BatchMode key auth)
**Plan:** [`.plans/623_multi_host_pipeline_inference_lane.md`](../.plans/623_multi_host_pipeline_inference_lane.md) T0.3
**Instrument:** `scripts/plan623_t03_session.sh` — one harness, box-state preamble, per-cell logs (bmo `~/t03/<ts>/cells/`), GPU-quiet gate, JSON sidecar [911_plan623_t03_arms.json](911_plan623_t03_arms.json)
**Box:** bmo (Mac17,15, M5 Ultra, 96 GiB, macOS 27.0.1 26A434 — amo's twin), load 0.82–1.30 across all four segments, `ps` GPU-consumer gate green at fire
**Model:** `Ternary-Bonsai-2-27B-PQ2_0.gguf`, sha256 `3907dc16…62ec1` = the league pin EXACT (MLX arm rides the 2-bit affine+hadamard PACK — quant posture differs, disclosed not equalized, same as Bench 904)

## Arms

| engine | binary / runtime | posture |
|---|---|---|
| llama.cpp | the Bench-900 bundle (`build-rpc/bin/llama-bench`, prism-b10787) | `-ngl 99 -t 8`, 5 reps each row |
| MLX | `.raw/mlx-env` (mlx 0.32.0, mlx_lm 0.31.3 — the 904 pins) | `plan623_mlx_1box_bench.py`, greedy |
| riir-infer | `dist_stage` @ `3a8abae84620` (sha256 `9fdae0c8…dfbd`, built amo-side pre-`31fd00a` toolchain pin) | `--backend gpu --split 32`, loopback `127.0.0.1:17631`, `--prefill-chunk 128` (the shipped serving width) |

## The matrix (t/s; decode = 128 fresh tokens)

| cell | llama.cpp | MLX (pack) | riir-infer pipeline | riir-infer 1-box oracle | ours / llama.cpp |
|---|---:|---:|---:|---:|---:|
| **tg128** | **60.62 ± 0.74** | 57.9 (17.27 ms med) | **56.65** | **52.64**¹ | **0.93×** (0.87× oracle) |
| pp512 | 1087.36 ± 1.26 | 1058.1² | 140.2 (3.655 s) | — | **0.13×** |
| pp4096 | 1046.10 ± 0.10 | 1026.1 | 166.9 (24.534 s) | — | **0.16×** |
| pp32768 | 771.99 ± 0.08 | **REFUSED**³ | 96.1 (340.8 s; warm 340.9) | — | **0.12×** |

¹ The oracle prints END-TO-END incl. the 4-token prefill (`run_local_with`, no wire); its decode-only read is ≈55 t/s — within noise of the pipeline's 56.65. G1 PASS (stream 0 == oracle, token-exact).
² **Supersedes Bench 904's pp512 357.3** — that run carried first-call kernel-compile cost inside a 0.48 s cell; today's post-llama.cpp warm-cache read is the clean posture. tg agrees across sessions (57.9 today vs 58.1 in 904), pp4096 agrees (1026 vs 1005) — same compute, different compile amortization.
³ `metal::malloc: attempting to allocate 103079215104 bytes > maximum buffer size 62620631040` — the Bench-904 refusal reproduced in-session. Stock-MLX 32k prefill is N/A without chunked prefill; **riir-infer is the only engine that COMPLETES 32k prefill** (at 96 t/s).

## Findings

1. **Decode arm: competitive.** 0.93× pipeline / 0.87× oracle vs llama.cpp; 0.98× vs MLX. The multi-host lane's single-box base HOLDS at tg128 — the 2.37× aggregate at s=2 (Bench 908) stands on a base within ~7–13% of the incumbents, not on a blowout loser. Note the posture delta vs the league row (0.978× llama.cpp on amo): the league's bench_768 harness vs this dist oracle incl. prefill — different instruments, both honest, neither replaces the other.
2. **Prefill arm: the gap is real and wide** — 6.3× (pp512) to 8.1× (pp32768) behind llama.cpp, 7.6× behind MLX where MLX measures at all. This is the T0.3 clause firing: filed as **`.issues/1146_prefill_throughput_gap_1box.md`** BEFORE any further multi-host prefill claim.
3. **The per-frame floor is the gap, not the wire.** Chunked prefill at width 128 = 256 frames at P=32768: per-frame wall 1.331 s (pp32768), 0.914 s (pp512), 0.767 s (pp4096). The floor RISES with context (the every-4th-layer full attention attending over a growing KV) but the BASE floor (~0.77 s/frame = 6.0 ms/token through both stages) is what it is: a 128-row frame never engages prefill-shaped kernels — the MMA tiling, the GDN scan, and the stage handoff are all sized for batch-1 decode.
4. **pp4096 > pp512 per-frame (0.767 vs 0.914)** — warm-up amortization across 32 frames vs 4, before the KV-growth term takes over at 32k. The prefill cost model is floor + growing-attention, not linear-in-P.
5. **llama.cpp reproduces Bench 900 bit-close on this box** (tg 60.62 vs 60.85; pp512 1087.36 vs 1087.53) — same box, same bundle, five weeks apart: the incumbent baseline is stable, our gap numbers are trustworthy.
6. **Only-we-complete-32k** — MLX refuses, we don't (chunked prefill keeps intermediates bounded); the honest spin on the 32k row is capability, not throughput.

## Honest residue

- The riir-infer numbers are the DIST pipeline posture (two 32-layer stages over loopback); a fused 1-box runner would read the same kernels without the per-frame handoff — the oracle line covers decode only (`run_local_with` has no separate prefill readout; prefill cells ride the pipeline's own prefill line). Wire at loopback is noise at these walls (Bench 896: ~0.1 ms RTT vs 0.77 s frames).
- The binary predates `31fd00a` (the 1.99.0 toolchain pin, landed mid-session by the 4090 lane) — kernel semantics unchanged (pin commit touches lint tails + sockopts cfg), disclosed rather than rebuilt-mid-measurement.
- MLX pp512 supersession (finding 2) means Bench 904's 1-box table is stale on ONE cell; 904 stays as recorded — this bench is the correction of record.
-amo was NOT measured this session (its GPU lanes were live — the rethink elephant capture class); T0.3 asked for bmo-idle only.

## Session traps (paid for, pinned in the harness)

- **`nc -z` probe consumes the tail's single accept** — the tail serves ONE connection then exits; a port probe connects, gets accepted, EOFs the handshake ("failed to fill whole buffer"), and exits the tail before the real head arrives. Fix: poll the tail's own log for `listening on`.
- **`--ctx` must reach BOTH stages** — the tail's span gate checks activation spans against ITS OWN capacity; a head-only ctx left the tail at default 4096 and frame offset 4096 refused mid-prefill (`activation span 4096+128 > stage capacity 4096` — the gate working as designed).
- **bash 3.2 word-splitting**: `for w in ${CSV_VAR}` splits on spaces, never commas — the first selective rerun ran ZERO cells and printed COMPLETE in 13 s. `tr ',' ' '` first.
