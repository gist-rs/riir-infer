# Issue 1179 — the qwen3.8 specdec verify lane: the ABOVE-SERIAL half of the uzu beat (owner directive: >121 tok/s on bmo)

**Status:** OPEN — filed 2026-11-06 (session zed-uzu-distill@amo, epoch 1793928926); unclaimed. The serial half lives in Issue 1178 (T2 claimed zed-8c98fa2a@m3 to 11-08); this issue owns the SPECDEC multiplier — the piece that turns a ~38-50 tok/s serial into the 121.1-class headline. No filed task owned it before this issue: 1178 names it ("ABOVE-SERIAL") and hands the distill feed (Research 378 T0(e)); riir-ai Issue 755 is the drafter-blocked chat lane this re-opens UNDER NEW REGIME.

## The fire condition (Bench 1167 R2 decomposition, bmo M5 Ultra, 2026-11-06)

uzu 0.6.17 vendor-method reads **121.1 tok/s median** vs the published 129 (M5 Max) — REPRODUCED on our box. The number is NOT serial:

```
121.1 tok/s  =  3.18 tok/pass (shipped speculator, measured)  ×  ~38 serial-equiv passes/s
```

- uzu serial-equiv ≈ 38 tok/s ≈ 552 GB/s ≈ 67% of the serial roofline (the honest roofline the speculator multiplies; their R1 0.5.23 SERIAL cell read 52.9 = 94% — wheel-version-dependent).
- ours: decode 12.17 tok/s (Bench 950 re-anchor, 82.13 ms/step; pre-1176 was 8.61 — the league arms already bought ~29%).

**Why this lane exists (the arithmetic gate):**
- Specdec alone on our serial: 12.17 × ~2.5-3 net ≈ **30-36 tok/s** — 3-4× under the bar.
- Serial alone (Issue 1178's full prize, ~38-55): still under the bar without the multiplier.
- **Only serial × specdec together reach >121.** Beat arithmetic: serial 45-55 + acceptance ≥3.2 + draft cost ≤6-8 ms/cycle → cycle ≈ 25-33 ms → 3.2-3.8/0.025-0.033 ≈ **100-135 tok/s**. The beat bar therefore DEPENDS on 1178 T2/T3 landing serial ≥ ~40 (soft dependency — T1 acceptance pricing below unblocks immediately; the e2e bar does not).

## Why the 4090 chat specdec closure does NOT block this (the reopen record)

The qwen3.8 chat specdec lane closed NEGATIVE on the 4090 (three-wall decomposition, riir-train Issue 493 + Benches 824/942/943): Wall 1 cycle cost (Q2_K drafter +48% slower draft), Wall 2 acceptance (z-lab head 2.648 max, training ladder monotone DOWN), Wall 3 compose (746/779 refuted); the ≥100 tok/s ask needed mean-len ≥4.0 = "a head 1.5× better than z-lab". The issue's own reopen conditions: **"a new head family or a target swap"**.

**Both conditions are now met by measurement:**
1. **New head family** — uzu's shipped speculator measures **3.18 tok/pass median (3.6-3.8 on the league fixture)** on the SAME model family at T=0.95 (Bench 1167 R2 speculator stats). That is 1.2-1.4× the z-lab ceiling the closure named as the acceptance wall.
2. **Target-swap regime** — the bmo dense regime makes draft cost nearly free: a ~1.1 GB draft at M5-Ultra bandwidth ≈ 1.3-1.5 ms/token → k=4-6 draft ≈ 5-9 ms against an 82 ms serial step (6-11% overhead), vs the 4090 chat regime where draft cost was ~50%+ of the cycle. The closure's Wall 1 does not transfer.

Also in-house precedent: the 4090 DFlash2 loop banked **1.242×** (49.9 vs 39 spec-less, league chat row) — the verify-loop machinery class is proven in our stack; what is missing is the METAL DENSE wiring + a head with uzu-class acceptance.

**Modelless note (do not re-litigate):** n-gram/lookup specdec stays REFUTED for chat-class text — Bench 746 grid ("lookup empty on novel text") + Bench 824 Wall 3. Not re-opened. (Exact greedy specdec would be a pure speed lever with output-identity, but measured headroom on chat text is empty; the head path is the one with measured 3.18.)

## The two paths to a ≥3.2 tok/pass head

**Path A — consume the vendor artifact (fastest unblock; owner-gated posture):**
trymirai ships the speculator WITH the checkpoint: `trymirai/Qwen3.8-27B-M` on HF, **license apache-2.0**, speculator artifact **1,106,748,808 B, blake3 `f6c67be2…9638748`** (Bench 1167 R2 FairnessManifest). The MIT uzu engine source (Research 378, pin `1866d189`) documents the consuming format (`encodable_block/weaver.rs` + `dflash.rs` + the speculator-stats surface). Work = a riir-infer loader arm for the speculator format + the verify loop. **A1/A8 lineage holds:** weights-from-disk, never committed to any repo, blake3-pinned at load, the pin recorded beside every quoted cell.
⚠ **The unmeasured premise (T1):** their head was distilled against THEIR mirai-m 4-bit target; acceptance against OUR Q4_K_M GGUF target is UNMEASURED. Same-family, close-quant (their own card: "comparable to Unsloth UD-Q4_K_M") — plausible transfer, never assumed.

**Path B — train our own head (the durable answer):**
An EAGLE3/MTP-class draft head for Qwen3.8-27B, distilled in **riir-train on the 4090** (the standing GPU lane; riir-train 493 v1-v4 is the old family's training ladder — the new family's rung is the same unit with the new target class). Minted as an artifact; weights never enter repos; consumed by this lane's loader identically to Path A.

## Machinery inventory (consume, do not rebuild)

| piece | where | posture |
|---|---|---|
| `SPEC_MAX_K = 8` vocab-handle seam | `riir-infer-gpu/src/ternary_deltanet_gpu_forward.rs:1694` behind the `speculative_decode` feature | exists (GDN lane) |
| p-ROW batched verify family | `riir-infer-gpu/src/cudarc_kernels/attention.rs` (Issue 742 T9.9, "the Q4_K speculative verify port") | CUDA posture |
| multi-query verify attention | `attention_cubecl/mod.rs` (`attention_decode_folded`, seq_len_q > 1) | CubeCL |
| qwen3.8 DFlash2 + regime gate + accept calibration | riir-ai i1156 wave (`dspark_drafter.rs`, `dflash_config.rs`, `regime_gate.rs`, riir-gpu `relaxed_accept_calibration`) | CUDA/4090 lane, 1.242× banked |
| speculative fill (2-box form) | `riir-infer-dist/src/spec_fill.rs` (Plan 623 T5.1) | dist lane |
| **MISSING on Metal dense** | the verify pass on the qwen38_wgpu dense path (a k-position batched forward — can ride the existing prefill/tensor mm2d machinery + per-position logits), KV rollback, accept/reject sampling (T=0.95 top-k=20), the draft loader | **this issue** |

## Preregistered gates

- **G1 correctness:** the accepted greedy stream is **byte-identical** to serial greedy (speculative decoding with exact verify is output-identical, not approximate). FNV/argmax walk pins per the league law. The sampled arm (T=0.95 top-k=20) is measured separately as distribution-preserving — never claimed bit-identical.
- **G2 perf:** the vendor-method re-cell on bmo (the Bench-1167 harness twin: 1342-in prompt, T=0.95 top-k=20, 4096 out cap ×3, exclusive GPU via the queue, PROVENANCE + box-state lines mandatory, FairnessManifest rows). Intermediate bar **≥100 tok/s**; **BEAT bar > 121.1 median, same methodology** (the owner directive). A failed intermediate bar is a loud STOP, not a posture tweak.
- **G4:** no new steady-state hot-path allocations.
- **Feature posture:** `qwen38_specdec` opt-in, PROMOTE-ONLY env per the env-switch law; incumbent byte-identical when off. Promotion = GOAT + owner verdict.

## Tasks

- [ ] **T1 — acceptance price cell (Path A gate, unblocks NOW — speed-independent):** load the vendor speculator (blake3 `f6c67be2…9638748`), run draft+verify acceptance against OUR Q4_K_M target on the league fixture (`uzu_t2_prompt.txt`) + the vendor prompt, greedy AND T=0.95. **Decision gate: ≥2.5 tok/pass on our target continues Path A; below → Path B scoping opens, Path A closes with the record.**
- [ ] **T2 — the verify pass on Metal dense:** k-position batched forward reusing the prefill/tensor machinery + per-position logits readback; G1 greedy-identity harness vs serial (greedy walk digit-exact).
- [ ] **T3 — draft loop + accept/rollback + the sampled arm:** KV discipline at the accept boundary, T=0.95 top-k=20 accept/reject, the speculator-stats readout (tok/pass, passes/s) matching the vendor harness's own instrumentation.
- [ ] **T4 — e2e vendor-method re-cell on bmo + the record:** G2 bars above; row lands as WATCH/owner class in the league uzu block (the SCORED opponents remain llama.cpp + vLLM — this lane's bar is the owner directive, not a scoreboard gate; the Bench-831 format-cross law applies until a same-format cell exists).
- [ ] **T5 — (Path B, if gated) riir-train head-distillation plan:** plan-first in riir-train, 4090 GPU, the new head family rung; cross-link the artifact pin here.
- [ ] Cross-links at every landing: 1178 (serial — the beat arithmetic needs both), 1168 (host sync — the resident-greedy loop is the natural verify-loop host), riir-ai 755 (the drafter-blocked lane this re-opens), Bench 824/942/943/950/951/1167, Research 378 (T0(e) trie-tree feed).

## Boundary

riir-infer owns forward/verify/loader (this issue). riir-train owns any trained head (T5). riir-gpu/riir-ai own acceptance-calibration consumption. The league scoreboard stays llama.cpp-anchored. Weights never committed (A1/A8). Qwen3.8 base + the vendor checkpoint + speculator are Apache-2.0; the uzu loader reference is MIT — attribution notes in the loader arm, fixtures original.
