# Issue 034 — EXL3 Metal serving arm (Issue 001 §14 / reopen trigger r1: the OpenThai-SystemOne consumer)

**Status:** OPEN — filed 2026-10-02 from the OpenThai-EXL3 lane question (reflex session); the convert half rides [Plan 617](../.plans/617_exl3_openthai_convert_and_infer.md); unpriced until Phase-A data lands.

## Why this fires now (the recorded trigger)

Issue 001 (`.docs/001_exl3_trellis_format_support.md`, CLOSED 2026-09-25) closed with an
explicit handoff: *"Engine serving integration is reopen trigger r1, not an open task
here — file a new issue when a consumer appears."* The consumer appeared: the
OpenThai-SystemOne EXL3 lane proposal — `iapp/OpenThai-SystemOne` @ `5d04bcca`
(Apache-2.0), Qwen3.5-0.8B text tower + 256-slot classification head, currently a
fp32 comparison lane in riir-reflex (Benches 074/084/085/086). This issue is that
filing. It does NOT re-open §17.6's closed fused-GEMV verdict — it cites it and
scopes around it (below).

## What already exists — do not re-build

- **Reader:** `src/quant/exl3.rs` + `exl3_pack.rs` — bit-exact CPU dequant (numpy-oracle
  pinned), era gate `verify_pack_era` (known-good = `{"1.4.2"}`, `open_unverified_era`
  escape), `Exl3Residency` reporting.
- **Metal dequant:** T7b CubeCL kernels (`crates/riir-infer-gpu/src/exl3_dequant_cubecl.rs`,
  `exl3_gpu` feature) — macOS builds select cubecl `wgpu-msl`; decode stage BIT-EXACT;
  whole-pack bit-exact gate green **CUDA + Metal** (573/573 layers / 26.48 G weights);
  M3 non-CUDA tests 7/7 (plan 001 repair record). "No Metal" is FALSE for dequant.
- **4090 CUDA arm:** 71–87× wall vs the CPU arm (Bench 002).

## What is missing — this issue's scope

1. A SERVING path that consumes `Exl3Pack` end-to-end for a real model — the §14
   promotion trigger, never discharged (promotion stays REFUSED with trigger intact).
2. For this consumer: the OpenThai classifier forward (tower + 256-slot head + their
   decide contract, `permutations=1`) running over the pack — on Metal (M3) first,
   CUDA parity optional.

## The honest economics (§2 scoping clause + §17.6 scope note)

- §2 (load-bearing): EXL3's measured axis is **memory headroom / context ceiling, not
  decode throughput**. For a 0.8B single-shot classifier the pack buys ~0.4 GB on
  disk/transfer vs 1.6 GB bf16 / 3.2 GB fp32. **Dequant-once-at-load keeps RUNTIME at
  dense size (1.6 GB f16)** — the memory win only lands with packed-resident weights +
  dequant-in-forward (streamed per-layer or fused). Phase B prices that decision with
  measurements; it is not assumed in either direction.
- §17.6's fused-GEMV CLOSE was **decode-loop-scoped** (27B league model, ~0.95 s/step
  vs a 10–20 ms incumbent, decode-rate-bound, refuted by composition). A 0.8B
  single-shot forward is a different economics cell; the closure does not transfer by
  analogy in either direction. Phase B measures.

## Gates (all per-suite, never aggregate)

- **Bit-exact:** dequant parity vs the CPU reference on every layer of the new pack
  (plan 004's full-pack gate harness shape), Metal + CUDA both.
- **Numerics:** G5-style parity vs their fp32 server captures (the laya-riir
  precedent: frozen captures, byte-compare) before any lane number is quoted.
- **Board:** per-suite retention vs the fp32 pins — massive 0.9200, sib200 0.8382,
  xnli 0.8967/0.9000, wisesight 0.4750/0.4675 (Bench 074/084/086) — accuracy AND
  readout-ECE (calibration moves under quantization; the lossy-surface law,
  riir-ai Issue 750 T3 / Orthrus repro: aggregate flat while per-item flips).
- **§14 re-run** with runtime numbers if Phase B proceeds (its own trigger text).

## Non-goals

- **Encode on Metal:** exllamav3 quantization tooling is CUDA-only — convert runs on
  the 4090 (Plan 617 Phase A). No Rust encoder is scoped (writing one is a research
  project; the workspace reader stays read-only).
- Extending the era gate's known-good set without per-pack verification.
- A Rethink/riir-instinct product posture — DECLINED (see Plan 617 §Lane verdict):
  Thai product is owner-closed (C9), OpenThai latency (100 ms–1.6 s p50) fails every
  serve bar in the stack, and its EN strength is already captured teacher-side
  (synth-corpus openthai VETO, distill-teacher pass).
- The reflex-side lane file (`openthai-exl3`) — filed in riir-reflex when Phase B fires.
