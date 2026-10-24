# Bench 888 — Issue 1110 T2: the β-reopen probe on the gemma2 d2f decode distribution

**Verdict: ROW DIES AT PROBE COST — the probe substrate has NO drift to act on.** On the only
real gemma2 checkpoint the stack carries (plain IT, not diffusion-trained), the incumbent d2f
decode is a structural **1-step termination-token collapse**: every masked position commits
`<eos>` (or `<end_of_turn>`) at step 0 with p ≥ 0.99993, so the walk converges immediately,
the census makes ZERO post-commit observations, and the drift trigger never has substrate.
The issue's own death condition ("if flip-moves never beat resample-moves on our shape, the
row dies honestly at probe cost") fires in its stronger form: there are no flip-moves at all.

**Status:** CLOSED — T2 executed; T1 correctly never built (no production change).

## The instrument

`crates/riir-infer-gpu/src/bin/issue1110_reopen_probe.rs` (`[[bin]]` required-features
`gemma2_d2f`, run at `--features gemma2_d2f,cuda_backend` — the 4090 CUDA posture). The
incumbent walk mirrors the disarmed `d2f_decode_gemma2` seams over public API only
(Issue-695-H7 prompt-tail clamp, temperature→softmax-once→max-confidence→cumulative draw);
at every denoise step, every already-committed position's mask-suppressed row is read
(`q_u`, top-1 id/prob, `rank_u`); the trigger is top-1 loss AND `q_u < η·p_top1`
(η-curve post-hoc; the η axis runs BELOW 1 — at η ≥ 1 the headroom condition is vacuous
because `q_u < p_top1 ≤ η·p_top1` always holds once the top is lost); first-fire events get
counterfactuals (flip = top-1 at fire; K seeded resamples from the fire-step distribution)
scored by teacher-forced re-forwards (mask-suppressed PLL at the event position).
Unit tests ×6 (pure helpers, no GPU) — softmax monotone/sums-1, topk order + first-index
ties, bounded-JSD zero-identity + ln-2 bound, the trigger's two-condition law + the
vacuous-η law, PLL modality, draw mass-concentration.

## Measured (4090, detached runs, GPU 44 °C / 5 % util at launch, no compute consumers;
`gpu_exclusive=false` in the PROVENANCE lines is the in-process nvidia-smi compute-apps
query returning desktop graphics processes on this box — the manual preflight showed none)

| posture | mask id | prompts | committed | p_at_commit min/median | tokens committed | steps to converge | obs rows | drift events |
|---|---|---|---|---|---|---|---|---|
| `pad_mask` (12 prompts, full) | 0 `<pad>` | 12 | 192 | 0.99993 / 0.99999 | `{1}` only | 1 (all) | 0 | 0 |
| `unk_mask` (4 prompts) | 3 `<unk>` | 4 | 64 | 0.93574 / 0.99996 | `{1, 107}` | 1 (all) | 0 | 0 |

- `<eos>` = id 1; `<end_of_turn>` = id 107. Every commit is a TERMINATION token.
- Converged 16/16 at every prompt in both postures; quality preset (block 16, steps 12,
  τ 0.8, temp 1.0, gemma IT chat template via `encode_with_bos`).
- No τ/temperature posture can produce multi-step refinement on this checkpoint: the
  confidence at masked positions is ~0.9999 for termination tokens, so τ ≤ 0.99 commits at
  step 0 (what we measured) and τ > 0.99 stalls forever (never commits). There is no
  intermediate regime — the collapse is structural, not a knob setting.

## Why (the root cause)

The `gemma2_d2f` lane (Plan 108) is the CubeCL inference vehicle for the D2F architecture,
but the stack's D2F TRAINING (katgpt-rs Plan 066 / riir-train Plan 068) trained only
micro-scale models (`micro_dllm`, vocab 27, mask = `vocab_size − 1`). No diffusion-trained
gemma checkpoint exists in the stack. `gemma-2-2b-it-f16.gguf` is a plain IT model: it has
never been trained to denoise mask-filled blocks, so at a `<pad>`/`<unk>`-filled position it
puts ~99.99 % on the sequence-terminator — exactly what an autoregressive chat model does at
an out-of-place filler token after a complete answer. The β-reopen premise (later denoise
steps re-inform earlier confidence commits) needs a mask-trained checkpoint to have any
substrate; this one has none.

En-route finding: precondition (b) of the issue was stale — `gemma-2-2b-it-f16.gguf` is
PRESENT on this box (5,235,213,856 bytes, dated Aug 8), no restore needed.

## What would re-open the row

A checkpoint with denoising training (a D2F-distilled gemma at real scale — riir-train
capacity — or any mask-trained LLM GGUF). Then T2's census runs unchanged: same bin, same
laws. The probe cost bought the answer for ~1 h of GPU; building T1's reopen memory first
would have shipped machinery with nothing to act on.

## Artifacts

- `pad_mask/reopen_probe.json` — full census (12 prompts, commits + topk snapshots, η-curve).
- `unk_mask/reopen_probe.json` — the mask-convention control (4 prompts).
- Provenance lines inside both JSONs (`captured_at_unix` 1792871950 / the unk run).

Session: zed-4090-1110-t2, epoch 1792870636.
