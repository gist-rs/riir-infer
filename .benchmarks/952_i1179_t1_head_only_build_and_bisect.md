# Bench 952 — Issue 1179 T1: the head-only specdec acceptance cell — build half + bisect record

**Status:** IN PROGRESS — the harness + loader + probes landed (d3c5da8); the acceptance measurement is BLOCKED on the last draft divergence (the attention stage, isolated to identical-input ground truth below). T1 gate (≥2.5 tok/pass continues Path A) NOT yet decided.

## Session 2 (2026-11-06, zed-amo-specdec2, `8ce8eb1`) — the verdict checklist EXECUTED + the regime root cause

The four AGREE conditions, in order, and what each found:

1. **Vendor chain-only cell (condition 4) — DONE, the deciding reference exists.** Their engine (uzu 0.6.17 wheel, this box, league fixture prompt, greedy, 512-token outs ×3) with `speculator/shapes.json`'s M5-Ultra row patched to `construction_method: {"type": "Argmax"}` (their FIRST-CLASS chain form — `dflash_tfm.rs` at the pin; tree_budget 16 = root + 15 greedy chain rows, the same shape our T1 harness verifies). Result: **2.882 tok/pass ×3** (512/178 passes, identical). The Weaver control on the same box/prompt: **3.821/3.821/3.323** — matches the R2 vendor reference 3.6-3.8, so the Argmax patch demonstrably took effect. Driver: `scripts/i1179_chain_only_cell.py`. (Harness notes: the mirai checkpoint was staged from HF by parallel-range chunks — the xet single-stream wedges at ~12 GB; `model_by_path` needs an `encoding.json` in the dir — `[{"type": "hanashi", "name": "qwen3.8"}]`, LOWERCASE, read from the mirai registry row.)
2. **Per-position rank-0 (condition 3) — DONE, and the shape was the smoking gun.** 120-cycle raw-fixture greedy run: mean j = 1.242, G1-T1 PASS, but **rank-0 hit at draft position 1 = 0/120** rising monotonically to ~49% at position 15 — the signature of a degenerate/out-of-distribution regime, not a weak-but-healthy head (their 2.882 demands near-certainty at position 1).
3. **Real-context parity (conditions 1+2) — DONE, and the divergence is REGIME, not math.** The KV=4 fixture twin is non-adjudicating (their bf16 degenerates to repeat-chains there; our f32 varies). At the REAL 603-row state: their chain (CPU, our captured features, identical t_prev) `[248046, 271, 760, 1156, 279, 1785, …]` vs ours `[378, 4307, 279×4, 1785×4, …]` — total per-layer divergence (cos ≈ 0.00 at every layer, both layer_out AND shortcut). ROOT CAUSE: the RAW fixture prompt makes the target's greedy continuation start at `<|im_end|>` (248046) — embedding maxabs **0.047** vs content tokens' 0.43-0.53, and the draft MASK token 248070 is **0.011** — so the twin state (and every T1 cycle after it) sits in a post-EOS lane where the distilled head is queried out-of-distribution and both engines render noise (their per-layer maxima L0→L4: 1352/384/2352/16k/465k vs ours 1016/261/683/5745/87.6k — amplified rounding, not structure). The KV=4 case inverted for the same reason. Their CPU prepare kernel was verified symmetric meanwhile (the same ROTATION_SIGN_BITS word table + hadamard128 + int8 on q, k AND v; rope at absolute positions) — our skip-the-transform f32 path stays score-preserving; `accept()` matches their encode_accept stage-for-stage (projection → norm → kv projection → per-head key norm + rope).
4. **The regime fix + the deciding re-run: `--prompt chat`** (the bundled qwen3.8 template wrapper: `<|im_start|>user\n…<|im_end|>\n<|im_start|>assistant\n<think>\n` — the vendor's own 2.882 cell went through the chat template; its in_tokens 654 vs our 603). 120-cycle chat-regime run IN FLIGHT; the T1 gate verdict follows it. If chat-regime chain-only acceptance lands ≈2.5+, the draft is faithful AND Path A continues (the weaver form is then T2/T3 scope, the vendor's own chain-vs-weaver gap being 3.82 vs 2.88 on their target).

**The gate table so far (all chain-only, per-pass acceptance j):** their engine/their target/chat regime = **2.882**; their engine/their target/Weaver = 3.82; ours/our target/raw fixture = 1.242 (the degenerate lane); ours/our target/chat regime = **1.200** (`8ce8eb1` + the chat flag; G1-T1 PASS; rank0[pos1] = **0/120** in BOTH regimes — the structural signal).

## Session 2b (continued, same day) — the divergence LOCALIZED to the qkvg stage

Stage-by-stage exact comparisons at the REAL 603-row state (their CPU engine via the `.raw/uzu` probe rig @ pin `1866d189`, all dump files under /tmp/i1179_*):

| stage | result |
|---|---|
| accept kvproj (pre-norm/rope projected KV, rows 0-3 × 5 layers) | **EXACT, cos 1.0000** |
| cached K (post norm+rope, L0 row 0; theirs dequant + un-HS) | **EXACT, cos 1.0000** (max 6.111 vs 6.115 — int8 rounding only) |
| their out_projection ON MY preproj vs my final | **EXACT, cos 1.0000** (my attn_out fold is right — note: this test is basis-vacuous, it validates the pair, not the basis) |
| rope tables (formula + layout + write indexing, static read) | identical (half-split, 1/base^(2p/128), pairing (c, c±64)) |
| their prepare kernel (sign∘H128∘int8 on q, k, AND v; absolute-position rope) | symmetric ⇒ our skip-the-transform f32 path is score-preserving |
| **qkvg projection output (L0, same conv_pre)** | **DIVERGES: cos −0.48, magnitudes 4× (mine 123.9 vs theirs 31.25; ratio 3.96-4.13 across rows)** |

The chat-regime acceptance gap (1.20 vs their 2.882) and rank0[pos1]=0/120 are consequences: the divergence enters at the FIRST draft stage and compounds through the 5-layer block. The fold experiment: `I1179_FOLD_INONLY=1` (input-side-only fold) made it WORSE (cos ~0.00, magnitudes halved) — the both-sides fold stays. Their `RHTLinearWrapper` (rht_wrapper.rs + matmul post_rht at the pin): input RHT → matmul(Ŵ) → output RHT = exactly my both-sides fold algebraically — so the residual difference is in the DEQUANT ARITHMETIC the matmul feeds (uint4 code nibble order, zero-point packing/reading — the zp tensor is [160][3072 B] packing 6144 nibbles along output), or a wrapper activation-format subtlety (`try_new_with_input_preparation`'s Bf16 branch hands the INNER linear to the caller with a LinearInputPreparation — which path qkvg actually takes depends on `select_activation_format`). Dequant-convention brute force (12 combos, python, row-0 only) found no match — the search space needs the fold applied per combo and possibly a 5th axis (the matmul's own activation packing).

**Next session's first move (the rig makes each test ~5 min):** dump-side bisect of the qkvg dequant — compute row-0 qkvg for each convention WITH the fold (full 32-row blocks; the probe rig + `/tmp/i1179_convpre_l0.f32` + `/tmp/i1179_uzu_qkvg.f32` are the fixtures) against `/tmp/i1179_uzu_qkvg.f32`; candidates: zp nibble order, zp layout, weight nibble order, scale dtype/layout, or the wrapper's activation-format branch. Once qkvg matches (expect cos ≥ 0.999), the chat-regime T1 cell re-runs and the ≥2.5 gate gets its real verdict.

---

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

**The gate comparison confound + the verdict-AGREE conditions (next session's checklist, in order)**: the vendor 3.6-3.8 tok/pass reference is the WEAVER tree (budget 16) on THEIR mirai-m target — no chain-only baseline exists on their checkpoint. The vendor-target cell (14.5 GB vendor checkpoint + the uzu engine, both staged) DECIDES the gate: their chain-only j≈1 ⇒ confound; their chain-only ≥2.5 ⇒ a residual bug on our side (do NOT reword the gate). Verdict conditions, in order: (1) the q/k/v dump-vs-prepare parity check FIRST on the vendor target, before any acceptance number; (2) per-layer branch-cos check (attn-out, MLP-out, final hidden — L0 branch cos 0.9891 is 10× off the shortcut's 0.9992, max 12% low; if the gap decays across layers, hunt norm eps / rope base / GQA mapping / gate activation); (3) report rank-0 hit rate PER draft position (1..15) + raise the cycle count (n=24 is too small for a 3-decimal mean); (4) the vendor chain-only cell tells confound from bug. Two precision divergences remain disclosed: bf16-vs-f32 (their degenerate repeat-chains on tiny contexts, e.g. [53769×15] at KV=0, vs my varied chains — a bf16 signature, cos agreement at L0 is 0.989-0.999) and the skipped int8-KV rounding.

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
