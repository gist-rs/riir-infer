# Issue 034 — EXL3 Metal serving arm (Issue 001 §14 / reopen trigger r1: the OpenThai-SystemOne consumer)

**Status:** VERDICT DELIVERED 2026-10-02 — Phase A complete on the 4090; **A5 = NO-GO by the pre-registered letter (code_fixtures −3.13, ag_news +1.50; typed/sst5 at-bar) → Phase B does NOT fire**; A6 executed on the M3 (Bench 107, bekko-400M seated). Provisional-GPU disclosure recorded (sibling training shared the GPU; quiet-window re-confirm owed before any site publish). Owner decisions surfaced: (a) A5 re-scope = a NEW decision per the plan's own rule; (b) riir-reflex sib200 pool floor (560) is unsatisfiable at source (701 train rows) — floor-mispin. Plan 617 Phase B/C-partial below stays for the record. **FINAL (same day,
2026-10-02): the owner-delegated Claude verdict returned AGREE — NO-GO STANDS,
the lane closes, Phase B does not fire.** Re-open conditions recorded in Plan
617 C1 (paired-discordance bar; sib200 measured post-floor-fix — the reflex
`thai_sib200` floor was re-based 560→400 at `ff0b33a` the same day, justified
from the source measurement; consumer-tied). **§14 scope note (the verdict's):
this consumer's NO-GO does NOT discharge or falsify the §14 serving-arm mandate
— the reader, era gate and per-pack bit-exact gate are green; the issue-001
handoff ("file a new issue when a consumer appears") stays armed for FUTURE
consumers.** Reflex record: Bench 110 (`.benchmarks/110_openthai_exl3_convert_board.md`,
`3cad008`).

## Why this fires now (the recorded trigger — precise)

Issue 001 (`.docs/001_exl3_trellis_format_support.md`, CLOSED 2026-09-25) closed with the
handoff *"Engine serving integration remains separately open, not implied-complete"* and
*"file a new issue when a consumer appears."* The consumer appeared: the
OpenThai-SystemOne EXL3 lane proposal — `iapp/OpenThai-SystemOne` @ `5d04bcca`
(Apache-2.0), Qwen3.5-0.8B tower + 256-slot classification head, currently a
fp32 comparison lane in riir-reflex (Benches 074/084/085/086). This issue is that
filing. It does NOT re-open §17.6's closed fused-GEMV verdict — it cites it and
scopes around it (below).

**Citation precision (verdict amendment, 2026-10-02):** the trigger this discharge
path re-runs is **§14's own** — *a serving/GPU arm that consumes `Exl3Pack`
end-to-end AND measures the delivered gain* (its gate re-runs with runtime numbers,
per the T7c-4 update). **r1 is closest-but-not-literal**: its recorded text (§17.6)
requires *"4 bpw residency / >100k-token context on 24 GiB where decode speed is
secondary"* — a residency/long-context lane. This 0.8B single-shot consumer needs
neither (0.4 vs 1.6 GB is immaterial on any fleet box; short sequences). The earlier
draft's "r1 reopen" shorthand inherited the doc-001 close line's compression; the
precise mandate is the handoff sentence + §14, not r1's residency axis.

## What already exists — do not re-build

- **Reader:** `src/quant/exl3.rs` + `exl3_pack.rs` — bit-exact CPU dequant (numpy-oracle
  pinned), era gate `verify_pack_era` (known-good = `{"1.4.2"}`, `open_unverified_era`
  escape), `Exl3Residency` reporting.
- **Metal dequant:** T7b CubeCL kernels (`crates/riir-infer-gpu/src/exl3_dequant_cubecl.rs`,
  `exl3_gpu` feature) — macOS builds select cubecl `wgpu-msl`; **two-tier oracle:**
  decode stage BIT-EXACT, the Hadamard stages FMA-contraction tolerance-class
  (rel-Fro ~2.6e-7 vs ≤1e-5 gates, both backends); whole-pack gate **573/573 layers /
  26.48 G weights / 0 mismatches on BOTH backends** (4090 CUDA 555 s + M3 wgpu→Metal
  183 s — T7c-1c, §17.5); M3 module tests 7/7 (§16.5). "No Metal" is FALSE for dequant.
- **4090 CUDA arm:** 71–87× wall vs the CPU arm (Bench 002).

## The model's shape (load-bearing for A1/B2 — verdict amendment)

Qwen3.5 is a **gated-delta-net hybrid** (`chunk_gated_delta_rule` + causal-conv;
Research 003) — not a plain transformer. Consequences:
- A1's arch-support check should expect the hybrid as the LIKELY exllamav3 failure
  mode; the fail-loud STOP is armed — treat a clean convert as the surprise, not the
  baseline.
- B2 is NOT a generic transformer port: it must REUSE the qwen35-deltanet substrate
  this repo already carries (the `qwen35_deltanet_config_from_gguf_metadata` loader,
  the deltanet forward family, GDN chunked-prefill kernels) plus the head-side pieces
  Research 003 pins: the tokenizer's added special tokens, the `SlotHead`
  (`Linear(H→256)` read at every `<|ts_answer|>` hidden state; slot 255 = abstain;
  softmax over the k options + abstain), the per-question-type learned
  log-temperatures, the padded multi-option single forward, and the
  renormalized-probability decide contract.
- B1's dequant economics differ PER LAYER CLASS (fixed GDN state vs KV attention vs
  dense MLP) — measured per class, never averaged away.

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

- **Bit-exact:** decode-stage parity vs the CPU reference on every layer of the new
  pack (plan 004's full-pack gate harness shape), Metal + CUDA both; Hadamard stages
  at the recorded tolerance-class gates. Coverage floors pinned — a loader regression
  REDS, never a green zero.
- **Numerics:** G5-style parity vs their fp32 server captures (the laya-riir
  precedent: frozen captures, byte-compare) before any lane number is quoted.
- **Board:** per-suite retention vs the fp32 pins — massive 0.9200, sib200 0.8382,
  xnli 0.8967/0.9000, wisesight 0.4750/0.4675 (Bench 074/084/086) — under the
  PRE-REGISTERED bar of Plan 617 A5 (below), accuracy AND readout-ECE (calibration
  moves under quantization; the lossy-surface law, riir-ai Issue 750 T3 / Orthrus
  repro: aggregate flat while per-item flips).
- **§14 re-run** with runtime numbers if Phase B proceeds, under the cell-appropriate
  gain definition Plan 617 B4 pre-registers (the records' decode-step metric and the
  f16/q4 GGUF incumbent do not exist for this cell; §16's amendment is cited too).

## Non-goals

- **Encode on Metal:** exllamav3 quantization tooling is CUDA-only — convert runs on
  the 4090 (Plan 617 Phase A). No Rust encoder is scoped (writing one is a research
  project; the workspace reader stays read-only).
- Extending the era gate's known-good set without per-pack verification.
- A Rethink/riir-instinct product posture — DECLINED (see Plan 617 §Lane verdict):
  Thai product is owner-closed (C9), OpenThai latency (100 ms–1.6 s p50) fails every
  serve bar in the stack, and its EN strength is already captured teacher-side in the
  three recorded roles (synth-corpus agreement VETO; `--synth-teacher`;
  `--distill-teacher openthai` single-teacher fallback — Benches 083/089/104).
- The reflex-side lane file (`openthai-exl3`) — filed in riir-reflex when Phase B fires.
- **The Bekko SystemOne family (17M/68M/400M) — EXL3 structural NO** (owner ask
  2026-10-02): ModernBERT-compatible shared-prefix ENCODER, wrong model class for a
  causal-decoder format — exllamav3's converter refuses by design. 68M/17M already
  seated on the reflex board (Bench 103); the 400M board seat is Plan 617 A6 via the
  existing `--bekko` lane. A smaller-bekko-in-our-runtime idea would be a separate
  encoder-port filing (laya substrate + ModernBERT + the shared-prefix mask + three
  heads, G5-gated) — never this issue.
