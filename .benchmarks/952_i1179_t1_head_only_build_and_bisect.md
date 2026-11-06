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

## The open divergence — CLOSED for the mixer, resolved as precision for the chains

**Verdict-round-1 root cause CONFIRMED + FIXED**: the qkvg slicing was section-major (`q = qkvg[..rows*4096]`) over a row-major `[rows][q|k|v]` buffer — row 0's q correct, every other row a q/k/v mix, k/v entirely rows 10-15's bytes (found by the verdict reviewer reading the code; verified by the probes). Post-fix:

| stage (identical input: token 198 + 15 masks, stateless) | mine | uzu |
|---|---|---|
| L0 shortcut max | 4881.43 | 4896.00 — **cos 0.9992** |
| L0 branch hidden max | 1762.85 | 1992.00 — **cos 0.9891** |
| harness draft ranks (fixture, cycle 0 row 0) | rank 34 (was 247120) | — |

**First acceptance numbers (post-fix, 24 cycles, fixture, greedy, argmax chain-15)**: mean j = **1.083** (histogram 22×j=1, 2×j=2) — G1-T1 serial-identity PASS. The draft commits real continuations (ranks 34-5516 typical) but is far from the ≥2.5 gate at the chain-only form.

**The gate comparison confound (next session's first cell)**: the vendor 3.6-3.8 tok/pass reference is the WEAVER tree (budget 16) on THEIR mirai-m target — no chain-only baseline exists on their target. The chain-vs-weaver gap on their own checkpoint must be measured (14.5 GB vendor checkpoint + the uzu engine, both already staged) before the 2.5 gate can be adjudicated for the argmax-chain form. Two residual precision divergences remain disclosed: bf16-vs-f32 (their degenerate repeat-chains on tiny contexts, e.g. [53769×15] at KV=0, vs my varied chains — a bf16 signature, cos agreement at L0 is 0.989-0.999) and the skipped int8-KV rounding.

## The pre-fix bisect record (historical)

The pre-fix table (all superseded by the per-row slicing fix):

| stage | mine | uzu |
|---|---|---|
| L0 branch hidden max | 47.6 ([gate|value] order) | 1992 |
| L0 shortcut max | 1511 | 4896 |
| attention on the SAME conv_pre output | 87.09 (post-rope-fix) | 42.75 |
| their MLP on MY h2 | — | 21120 (my mlp_down 17.5k — order matches) |

The attention on an identical input differs ~2x — the remaining bug is inside my `attention_block` (qkvg fold verified; norms same-source; rope tables verified numerically except 4 entries at position 0 ch0-3 where uzu's table holds 0.0 — immaterial at 4/2048 entries). Suspects: the rotation-sign + hadamard KV transform (my f32 path skips it as score-preserving — re-examine), or a subtle mask/state detail. The probe rig is in place: `I1179_STAGE_DUMP=1` dumps their q/k/v code maxima; `/tmp/i1179_cycle0_dump.f32` + `/tmp/i1179_mini_target.safetensors` + the uzu `dflash_probe` test reproduce their side.

## Acceptance state (updated)

First measured: mean j = 1.083 (24 cycles, fixture, greedy, argmax chain — see the post-fix numbers above; pre-fix it was j=1.0 garbage-in). G1-T1 serial-identity PASSES. Vendor reference on their own target: 3.6–3.8 (league fixture greedy, **Weaver tree**, Bench 1167 R2) — the chain-vs-weaver confound above must be closed before the 2.5 gate is adjudicated. The `suffix_repetition_length=1` confound is NEUTRALIZED: read from their harness + kernel — it feeds a repetition-penalty ring, and their penalty = 1.0 = no-op.

## Box state

amo (M5 Ultra, macOS 27.2), AC, quiet; probes single-GPU; uzu probe CPU-only (72-133 s/run). Harness runs: target load ~20 s + prefill ~25 s + ~20 s/cycle (draft CPU + 16 serial verify steps + checkpoint/restore ~1 GB).

## FairnessManifest

speculator `model.safetensors` blake3 `f6c67be27ee8d81d7f133bf2724a969bd0b337cdbf2b7060e02803a219638748` (asserted, not just recorded); GGUF `ce14c95531256552644397d8c38691c0fe6f228cd2dd2d8cce13021ecc5bd06a`; emb-signs ranged-GET `~/.cache` n/a (fetched to `/tmp/i1179_emb_signs.i32`, optional flag, default OFF). The draft runs CPU f32 — the vendor runs bf16 + int8 KV: a disclosed precision divergence, secondary to the open bug.
