# Bench 953 — Issue 1180 T1: the real UNGATED G2 arm (gemma-2-2b sink census)

**Status:** DONE 2026-11-06 — T1 bar MET with one disclosed caveat (the pair is CROSS-FAMILY, not matched).

## What ran

New dense-body census lane (`transformer::sink_audit_dense`, under the `sink_audit` feature): the
existing m_Y probe (`AttnSpanProbe`) gained an optional leading-position profile, armed on every row.
It re-scores Q·K inside `attend_row` with the forward's own scale and softcap, so the distribution it
reads is the one the forward uses. `finalize_profile` turns it into the same `SinkAuditProfile` the
hybrid lane emits, so the verdict, `from_audit` and the canonical payload are shared code. A loud-zero
guard refuses when the profile mass ≠ layers·heads·N (a body that bypasses the probe would otherwise
read a false SinkFree). The `sink_audit` bin dispatches `gemma2` / `llama` GGUFs to this lane
(SentencePiece / BPE tokenizer, BOS prepended at position 0, as the serving path does).

`sink_audit ../riir-train/data/gemma-2-2b-it-f16.gguf --n 512` (corpus `chat_probe`, the bench 947/948 defaults).

| checkpoint | attention output gate | layers scanned | f_attn_token0 | positions 1–7 | m_act | verdict |
|---|---|---|---|---|---|---|
| **gemma-2-2b-it f16** (this bench) | **none (ungated)**, tanh softcap 50 | 26 of 26 | **0.407117** | 0.0011–0.0079 | 31.47 | **SinkHeavy** (≥ 0.25) → incumbent pin 4 stands |
| Qwen3.5-0.8B-Base Q8_0 (bench 947, re-run here) | sigmoid gate (Issue 594 shape) | 6 of 24 | 0.076425 | 0.0039–0.0067 | 35.96 | SinkFree (≤ 0.10) → pin 4→1 |
| Bonsai-2-27B PQ2_0 (bench 948) | gated, ternary | — | 0.1297 | — | — | Undecided |

The paper (arXiv:2505.06708, Table 4) reports gated 0.048–0.073 vs ungated 0.297–0.467. gemma-2
lands inside the ungated cluster; the gated Qwen lands at the gated cluster's upper edge. **G2 on
real checkpoints: the same census code separates the two classes (0.407 vs 0.076, 5.3×).**

## Gates

- **G1 determinism:** gemma-2 two runs → BLAKE3 `1a6d2d1b2af24e932760925303d2c1310920e93ecb7bcb6cba471a3f02cfef1d` both.
- **Refactor is byte-neutral:** the m_act helper was extracted and shared (`mean_layer_max_v_norm`); the Qwen3.5-0.8B
  census re-run reproduces bench 947's pinned BLAKE3 `c3467f92657f86b59d1615122a54b6cce1e882ab398cf21a157e590db680dffb`.
- **Unit arms (12 pass):** leading profile == per-key span masses and sums to one per head; dense census
  matches the analytic uniform profile; deterministic; unprobed forward refuses loud; short corpus refuses.
  `cargo clippy --release --features sink_audit --lib --bin sink_audit` clean.

## Caveats (disclosed, not folded)

1. **Cross-family, not a matched pair.** gemma-2 vs qwen3.5 differ in family, scale, tokenizer (gemma has a
   BOS, qwen does not) and softcap. The class separation is consistent with the paper but this bench does
   not isolate the gate as the cause. The paper's matched HF gated/ungated pair is still the only clean
   ablation; it needs its own loader and stays the open refinement.
2. **Softcap:** gemma-2 caps logits at 50·tanh(·/50); the probe reproduces it, so the mass read is the
   forward's own.
3. **N=512 vs sliding window:** gemma-2's alternating 4K sliding window never binds at N=512, and this
   forward is full causal anyway.

## Box state

m3 (M3 Max, 64 GiB), AC power, powermode 2, load average 95/77/65 (other agents' builds + seal lanes), ~10.5 GiB
free pages. Census pass 138 s and 198 s on the two gemma runs — wall time is load-bound and not a claim;
only the masses and hashes are.
