# Issue 1139 — riir-infer 1-box GPU decode on M5 Ultra is ~2.2× behind llama.cpp (Bonsai-2-27B PQ2_0)

**Status:** Open — T1 DONE 2026-10-27 (the production funnel is ALSO ~27 tok/s on idle bmo → the gap is the kernel stack, not the stage path); T2/T3 open. Filed 2026-10-27 from Plan 623 T2.2b (indicative, NOT the formal T0.3 cell). Unclaimed.

## Why

Plan 623 T0.3's rule: *"If riir-infer's 1-box arm loses, file the gap as its own
issue first — the multi-host claim is meaningless on a losing single-box base."*
The T2.2b GPU-stage runs gave the first riir-infer decode number on an M5 Ultra
(bmo, idle — load ~1.4, no sibling GPU work):

| arm (bmo, Bonsai-2-27B PQ2_0, tg) | tok/s | source |
|---|---|---|
| llama.cpp prism-b10787 Metal solo | **60.9** | [Bench 900](../.benchmarks/900_plan623_t04_llamacpp_rpc_2box.md) |
| mlx pack runtime solo | 58.1 | [Bench 904](../.benchmarks/904_plan623_t04_mlx_arms.md) |
| riir-infer `TernaryGpuStage` full-range (T1.5 range funnel, per-step logits readback + host argmax) | **27.2** | `crates/riir-infer-dist/tests/two_stage_g1_gpu.rs` oracle line, 47 steps |
| riir-infer 2-stage GPU⇄GPU over loopback TCP, split 32 | 26.0 | same test, same run |

So the pipeline costs ~4% on top of a 1-box base that is itself ~0.45× the
incumbents. The 2-box lane cannot beat a 1-box incumbent until this closes.
The m3 decode campaign measures the same class (Issue 1134's 38.27 ms/token ≈
26 tok/s) — the per-rung work there (1127/1130–1137) is the lever family; this
issue is the M5-Ultra scoreboard pin those rungs are scored against.

## Tasks

- [x] **T1 — confirm on the production funnel. DONE 2026-10-27:** instrument
  `crates/riir-infer-dist/tests/decode_readback_probe_gpu.rs` (1-box `forward_token`,
  `--release`, full model, ctx 72, 8 warm-up + 64 timed steps). **bmo, AC power, load
  1.2–1.7, no sibling GPU work, 3 runs: 36.67 / 43.19 / 37.37 ms per token = 27.27 /
  23.15 / 26.76 tok/s** — the same ~27 as the stage seam, so the gap is the kernel
  stack (T2), not the stage path. Stage-only costs PRICED in the same run: idle logits
  readback 0.225–0.284 ms + host argmax over 248,320 logits 0.114–0.117 ms = **0.92–0.94%
  of the token** — the ceiling for an on-GPU argmax, BEFORE its own kernel cost; under
  the 1.5% G2 bar, so Plan 623 T2.2c(2)+(3) are DECLINED on this measurement (the
  production funnel pays the same readback, so the same verdict covers it).
  *Original brief:* re-measure with the 1-box
  `forward_token` decode loop (not the stage seam) on bmo, `--release`, box state
  recorded per the G2 law; rule out the stage-path-only costs (1 MB logits
  download + host argmax per step, `set_hidden_state`'s per-call device alloc).
  If the production funnel is also ~27 tok/s the gap is the kernel stack; if it is
  materially faster, the stage path's readback is a T2.2c item, not this issue.
- [ ] **T2 — attribute.** Per-dispatch census on the M5 Ultra (the Issue-1132
  instrument) vs the llama.cpp Metal graph for the same token: launches/token,
  bytes moved, the top-5 kernels by wall. Name the dominant class (launch floor vs
  bandwidth vs ALU) before proposing rungs.
- [ ] **T3 — route.** Fold the attributed levers into the existing rung issues
  where they already exist; file new rungs only for unclaimed classes.

## Non-goals

No multi-host work here (Plan 623 owns it). No gemma-2-2b cell (retired model).
