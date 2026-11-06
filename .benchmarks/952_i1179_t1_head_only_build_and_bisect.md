# Bench 952 — Issue 1179 T1: the head-only specdec acceptance cell — build half + bisect record

**Status:** IN PROGRESS — the harness + loader + probes landed (d3c5da8); the acceptance measurement is BLOCKED on the last draft divergence (the attention stage, isolated to identical-input ground truth below). T1 gate (≥2.5 tok/pass continues Path A) NOT yet decided.

## What landed

- `qwen38_specdec` feature + module: the uzu DFlash speculator consumed as a FROZEN vendor artifact (Path A of the issue): safetensors reader, uint4-affine dequant (group 32, zero-point nibble-packed along out), the incoherence basis fold (`W_orig = D_out·H32·Ŵ·H32·D_in` per 32-block, load-time), the CPU f32 draft block (norms Copy/Add semantics, separable causal conv with per-token coefficient deltas, GQA attention with per-head q/k norms + half-split rope 1e7/128, [value|gate] fused MLP SiLU), KV append (context_projection → context_norm → state_kv_projection → per-head key_norm + rope at absolute draft positions), blake3 pin at load.
- Forward capture: `forward_token_capture` (full logits + post-layer residual rows at [5, 19, 33, 47, 61]), `ingest_capture` (per-token prompt features + last-position logits), `checkpoint_state`/`restore_state` (full GDN+conv+KV snapshot for the verify loop's rollback).
- T1 harness bin `qwen38_specdec_t1`: fixture/vendor prompts, greedy + T=0.95 top-20 keyed sampling (`qwen38_spec::keyed_accept` — the 4090-proven decision layer, DRY), the cycle loop (draft 15 → verify 16 → keyed accept → restore+re-forward on partial), G1-T1 serial-identity gate.
- Bisect probes (examples/) + the uzu engine probe (`.raw/uzu` @ pin `1866d189`, patched: weaver-assert neutralized, stage hooks) — the ground-truth rig.

## Verified-correct pieces (against the uzu CPU engine, identical inputs)

| piece | evidence |
|---|---|
| artifact bytes | blake3 `f6c67be2…638748` == Bench-1167 pin; GGUF `ce14c955…5bd06a` ✓ |
| feature capture | layer-63 residual + final-norm + lm_head reproduces the forward's logits to 7 decimals |
| incoherence fold | `verify_fold`: max\|vendor-path − folded\| = 8e-6 on random inputs (qkvg projection) |
| premix norm stage | uzu probe vs mine: 11.000 vs 10.9813 (identical input) |
| conv_pre + kernel_proj | uzu probe (their premix + MY conv glue): 69.066 vs my pipeline 69.0523 |
| safetensors offsets | python cross-read of the scales tensor (the double-offset bug's detector) |

## Bugs found + fixed en route

1. **safetensors double offset** — `data` was already read post-header; indexing `data_offset + start` read garbage (NaN scales). Found by python-vs-rust cross-read.
2. **fold left-H columns** — the left-side block transform mixed only columns 0..32 instead of all columns per 32-row block. verify_fold: 6.26 → 8e-6.
3. **KV append missing key_norm + rope** — uzu's `append_projected_kv` routes the projected K through the attention prepare (per-head RMS + rope at the absolute position) BEFORE caching; my accept cached raw rows.
4. **rope applied to head 0 only** — my apply_rope treated [rows][4096] q as [rows][128]; heads 1..31 were un-roped (accept K row: same bug, now per-head).

## The open divergence (identical-input ground truth)

uzu's CPU engine (.raw/uzu probe) and my CPU draft, on the SAME block input (token 198 + 15 masks through MY mini-target tables, rope 0..16, stateless):

| stage | mine | uzu |
|---|---|---|
| L0 branch hidden max | 47.6 ([gate|value] order) | 1992 |
| L0 shortcut max | 1511 | 4896 |
| attention on the SAME conv_pre output | 87.09 (post-rope-fix) | 42.75 |
| their MLP on MY h2 | — | 21120 (my mlp_down 17.5k — order matches) |

The attention on an identical input differs ~2x — the remaining bug is inside my `attention_block` (qkvg fold verified; norms same-source; rope tables verified numerically except 4 entries at position 0 ch0-3 where uzu's table holds 0.0 — immaterial at 4/2048 entries). Suspects: the rotation-sign + hadamard KV transform (my f32 path skips it as score-preserving — re-examine), or a subtle mask/state detail. The probe rig is in place: `I1179_STAGE_DUMP=1` dumps their q/k/v code maxima; `/tmp/i1179_cycle0_dump.f32` + `/tmp/i1179_mini_target.safetensors` + the uzu `dflash_probe` test reproduce their side.

## Acceptance state

Not measured yet — j=1.0 on every cycle pre-bisect (garbage-in). The G1-T1 serial-identity gate PASSES (state discipline correct); the vendor acceptance reference on their own target: 3.6–3.8 (league fixture greedy, Bench 1167 R2). The `suffix_repetition_length=1` confound is NEUTRALIZED: read from their harness + kernel — it feeds a repetition-penalty ring, and their penalty = 1.0 = no-op.

## Box state

amo (M5 Ultra, macOS 27.2), AC, quiet; probes single-GPU; uzu probe CPU-only (72-133 s/run). Harness runs: target load ~20 s + prefill ~25 s + ~20 s/cycle (draft CPU + 16 serial verify steps + checkpoint/restore ~1 GB).

## FairnessManifest

speculator `model.safetensors` blake3 `f6c67be27ee8d81d7f133bf2724a969bd0b337cdbf2b7060e02803a219638748` (asserted, not just recorded); GGUF `ce14c95531256552644397d8c38691c0fe6f228cd2dd2d8cce13021ecc5bd06a`; emb-signs ranged-GET `~/.cache` n/a (fetched to `/tmp/i1179_emb_signs.i32`, optional flag, default OFF). The draft runs CPU f32 — the vendor runs bf16 + int8 KV: a disclosed precision divergence, secondary to the open bug.
