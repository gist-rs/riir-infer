# Issue 1180 — sink-audit census follow-ups: the real ungated G2 arm + the full-scale T4 cell

**Status:** OPEN — filed 2026-11-06, handed over from katgpt-rs Issue 981 (closed there). The katgpt-rs side is complete (the audit vocabulary `SinkAuditProfile`/`SinkAuditVerdict::classify` + `SinkWindowPolicy::from_audit`, katgpt-rs `923aef6a4`, T3 synthetic GOAT all gates PASS). Everything left runs on the census lane this repo owns (`sink_audit` feature + bin, Benches 947/948).

## Landed so far (do not redo)

- Bench 947: census lane (cfg-gated Q capture on `HybridForwardScratch`, full-attention layers only, bounded cache per Issue 929) + the `sink_audit` bin + `needle_eviction_gate --sink-audit N`. Qwen3.5-0.8B-Base (gated) reads `f_attn_token0` 0.0764 → SinkFree → pin 4→1; G1 bit-identical. Reduced T4 cell (8k / 4-needle): 25% budget audited-vs-incumbent bit-identical (neutral), 2% budget differs (the wire-through is consumed).
- Bench 948 (`7b427684`): ternary capture hook + `run_census_ternary`; Bonsai-2-27B-PQ2_0 reads 0.1297 → UNDECIDED → incumbent pin 4 stands (a family/scale/quant gap the chosen 0.10/0.25 cuts disclose, never fold).

Run any census: `cargo run --release --features sink_audit[,bonsai2_hadamard] --bin sink_audit <gguf>`.

## Tasks

- [ ] **T1 — the real ungated G2 arm.** The synthetic arms carry G2 on the arithmetic today. qwen3.8 dense has no attention output gate (the natural ungated arm) but no CPU forward; the paper's HF gated/ungated pair needs its own loader. Either (a) a GPU-path Q capture for qwen3.8 dense, or (b) a loader for the HF pair. Bar: the ungated arm reads SinkHeavy (≥ 0.25) and the gated arm SinkFree (< 0.10) on the same census code; an UNDECIDED read is disclosed, never folded. Together with Bench 948 it forms the ternary-family calibration pair if the cuts are ever revisited.
- [ ] **T2 — full-scale T4 cell** (64k-class long-context quality). On a SinkFree checkpoint: the reduced pin gives no long-context quality delta vs the incumbent (the GOAT half). On a SinkHeavy checkpoint (needs T1): the pin protects. Pre-registered gates as in katgpt-rs Issue 981 (now in katgpt-rs HISTORY.md); runtime is owner-adjacent, so route it through `agent-mail rig run` on a quiet rig.
- [-] **q8kv `sink_kv` sizing** — no code now: the q8kv sidecar is caller-sized and no caller populates it today (only the gemma2 kv_cache `Vec::new()`). The audited `n_sink` sizes it when a caller arms it.
