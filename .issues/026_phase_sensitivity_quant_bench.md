# Issue 026 — phase-isolated quantization sensitivity bench (the DQ instrument)

**Status:** OPEN — bench-only instrument, no GPU training; falsifiable directional assertion (a direction miss = recorded negative, premise dead on our formats).

Master: `.research/004_DQ_Disaggregated_Quantization.md` (arXiv:2609.26333 §2.2). The paper's measurement contribution: on decode-heavy tasks, decode-only quantization is 2–4× more accuracy-damaging than prefill-only quantization (up to 7× on Gemma3-1B); the ordering REVERSES on prefill-heavy tasks. We have no phase-isolated quant bench; every format/serving decision in 027/028 needs this instrument first. Cheap to build, decisive either way.

## Tasks

- [ ] **T1 — 2×2 phase-matrix runner.** One weight artifact evaluated under {prefill: activation-quantized | weight-only} × {decode: activation-quantized | weight-only}. Formats: `q2_0` (our byte-floor) + `q4_k` (mid tier), on one 27B-class artifact (Bonsai-PQ2 for the decode arm; a q4_k GGUF for the paired arm). Activation-quant arms may fake it (quantize→dequantize activations per layer) — the matrix measures *error injection location*, not kernel speed.
- [ ] **T2 — task axes.** Decode-heavy: long-CoT generation (GSM8K-class or the league's reasoning suite). Prefill-heavy: long-prompt/short-answer (RULER-class retrieval at 4K–32K, or the harness's long-context suite). Define both precisely in the bench doc before the first run.
- [ ] **T3 — damage-ratio directional assertions.** `R = Δacc(decode-only)/Δacc(prefill-only)`: assert `R > 1` on the decode-heavy axis and `R < 1` on the prefill-heavy axis at both formats. Pre-registered: if the direction does not reproduce on OUR formats, the disaggregation premise is dead on our stack — record the negative, close the lane honestly, and 027/028 lose their motivation.
- [ ] **T4 — KV-axis extension arm.** Three independent axes the phase protocol separates: KV *written by* quantized-activation prefill vs KV *stored* precision (`q8kv`) vs KV *read* precision at decode. Pin which axis dominates on the decode-heavy suite.
- [ ] **T5 — bench record + promotion rule.** Record in `.benchmarks/` with box state. Attach the standing rule: any future weight format publishes its per-phase R before default promotion — a format that damages decode 4× more than prefill at equal bpw is misallocating bits where chat users feel it.

Budget: 1–2 sessions; accuracy evals on M3 or 4090 (no training). Runs AFTER the 2×2 matrix definition review, before any 027/028 accuracy claim.
