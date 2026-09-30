# Research 005: Phonon-2 / Neutrino-1 (Fermion Research) — Ternary-QAT League Intelligence + Audio-Lane Candidate

**Status:** RECORD — integration-distill addendum; the audio lane itself stays owner-DEFERRED (Issue 015, "audio until M5 Ultra", 2026-09-28). No plan, no new issue; one recon line added to Issue 015.
**Date:** 2026-09-30
**Filed from:** owner ask ("i think we have audio or audition somewhere (i think reflex), shall we use this one?").
**Sources (read over the wire, not cloned — no `.raw/` needed, nothing here required byte-level grep):**
- `https://fermionresearch.com/research/phonon-2/` (announcement, 2026-09-26)
- `https://fermionresearch.com/research/one-eighth-the-bits/` (Neutrino-1 ternary-QAT research)
- `https://github.com/fermionresearch/phonon` (repo page: Apache-2.0 code; Phonon-2 weights CC-BY-4.0 as a derivative of NVIDIA `parakeet-tdt-0.6b-v3`, also CC-BY-4.0; NOTICE files present)
- `https://huggingface.co/FermionResearch/Phonon-2` (weights; not fetched — license carried by the repo/announcement)

## TL;DR

Phonon-2 is the most accurate open **English ASR** model under 900 MB (164 MB, 5.21 % WER avg
over the Open ASR Leaderboard's seven English sets, 174× realtime on a MacBook Air) — a
**QAT'd derivative of the same Parakeet TDT 0.6B v3 our deferred audio lane already targets**
(FluidInference ships `parakeet-tdt-0.6b-v3-coreml`; Research 001). The owner ask resolves to
three separable answers:

1. **"We have audio or audition somewhere"** — corrected: `examples/twt_bonsai_audition.rs`
   is model *auditioning* (TWT calibration profiling), not hearing. The audio lane is
   Research 001 + Issue 015 (FluidAudio/CoreML PoC), **owner-deferred until M5 Ultra**.
   riir-reflex has zero audio.
2. **"Shall we use this one?"** — **not today, and not as the lane's first load**: (a) the
   owner defer stands; (b) Phonon-2 is **English-only** against our EN/TH product locale
   scope; (c) it ships **no CoreML bundle yet** ("coming soon") — the landed T2 design
   (pure-Rust `objc2-core-ml` load of a prebuilt `.mlmodelc`) fits FluidInference's bundles
   today. Recorded in Issue 015's recon as a pickup candidate for when its CoreML runtime
   lands (or via `fermion serve`, the OpenAI-compatible HTTP server, as a subprocess
   comparison lane).
3. **The real value is league intelligence**: the companion "one eighth the bits" paper is
   **Neutrino-1 8B ternary-QAT research that names Ternary-Bonsai-8B (65.75 MMLU) in its
   comparison chart against Neutrino-1's 72.1** — Fermion Research is a direct competitor to
   the Bonsai ternary lane, publishing a llama.cpp fork and their own engine. Their measured
   PTQ-near-chance result, state-occupancy statistics, and per-tensor-class precision
   assignments are directly consumable by our ternary league + trainer backlog.

## 1. What they ship (at fetched refs, 2026-09-30)

| Thing | License | What it is |
|---|---|---|
| **Phonon-2** | weights CC-BY-4.0 (derivative of NVIDIA parakeet-tdt-0.6b-v3) | 164 MB English ASR; encoder at ~2.1 bpw, **five learned levels** (0, ±a, ±b — two magnitudes per output row); 5.21 % WER avg; 174× RT on MacBook Air (MLX), 143× on 8 Zen 5 cores, 6,680× on H100 batch-128. CoreML runtime "coming soon". |
| **Phonon-1 family** | Apache-2.0 | 415 MB / Micro 285 MB English ASR from `Qwen/Qwen3-ASR-0.6B` (Apache-2.0), "trained at 2.4 bits per weight from the start". |
| **Neutrino-1 8B / 0.6B** | (weights repo; Qwen3-8B derivative) | Full-model ternary QAT: 72.1 five-shot MMLU from a 3.88 GB artifact (2.56 GB coded download); ~96 % knowledge retention vs Qwen3-8B. |
| **Engines** | Apache-2.0 | C CPU runtime (AVX-512 VNNI / AVX2 / NEON, int8 per-row requant), MLX (Apple GPU), CUDA (bucketed graphs), `fermion serve` (OpenAI-compatible HTTP), docker CPU/CUDA images, a llama.cpp fork. |

## 2. Audio-lane candidacy (Issue 015 pickup — recorded, deferred)

When the lane unblocks (M5 Ultra), the model choice gains a real competitor:

| Criterion | FluidInference `parakeet-tdt-0.6b-v3-coreml` | Phonon-2 |
|---|---|---|
| Load path | Prebuilt `.mlmodelc` / `.mlpackage` — fits our landed T2 (`objc2-core-ml`) | Own engines (MLX/C/CUDA); **CoreML "coming soon"** |
| Size / accuracy | ~220 MB Redux class; v3 full ~2.5 GB | 164 MB, 5.21 % WER (beats Parakeet Redux 178 MB at 5.69 on their harness, incl. under noise) |
| Languages | v3 = 25-eu-lang variants; EN | **English only** — conflicts with EN/TH product scope (Seal) |
| License | per-model (Parakeet CC-BY-4.0) | CC-BY-4.0 weights, Apache-2.0 code |
| Serving shape | streaming-stateful CoreML (our F1 design) | `fermion serve` HTTP (OpenAI-compatible) — subprocess-comparison-lane shape (the GLiNER/AgentJev/openthai pattern) |

Verdict for the lane: **FluidInference stays the integration target** (CoreML bundle + our
Rust load path); Phonon-2 becomes (a) the accuracy-per-byte bar to cite, and (b) an
HTTP-subprocess comparison lane if we ever measure ASR quality at all. Detta/pip/docker are
end-user tooling — usable by the owner today for personal dictation, not stack substrate.

## 3. Ternary-league intelligence (the transferable value — no audio needed)

Provenance note: every number below is **their measurement on their harness**; we have
reproduced none of it. Ternary-Bonsai-8B's 65.75 is a named row in *their* Fig. 4 — the 8B
class of the Bonsai family our league model (Ternary-Bonsai-2-27B-PQ2_0) belongs to.

1. **PTQ at 2-bit lands near chance; QAT is the whole game.** Published one-shot 2-bit
   conversions of 8B models score 24.2 / 24.7 on five-shot MMLU against 25.0 for random
   answers; models *trained in the target representation* score 47.24 / 65.75 / 72.1 at the
   same artifact size. This is the strongest external validation yet of our stack's premise
   (Bonsai ternary comes from the trainer, never PTQ'd) — and we hold the SAME law from our
   own side: riir-train Research 411 measured ternary Bonsai holding reasoning (thinking
   avg 80.49) while IQ2_XXS collapses (72.73) at 63% of its size, +7.8 points — "trained
   ternary is categorically different from aggressive post-training quantization". Their
   near-chance table and our collapse table are one law seen from two directions. Citation
   of record for refusing future "just round it" shortcuts (our lossy-surface law, one
   axis over).
2. **Five levels at ~2.1 bpw vs our three at 1.75.** We ship TWO ternary containers:
   PQ2_0 (gguf_loader id 142: f16 scale + 32 B packed 2-bit codes per 128 weights = 34 B /
   128 = **2.125 bpw**) and — the one that matters here — **PTQ1_0** (id 143, Issue 980 T6 /
   Plan 600 C5): `qs[24] + qh[2] + d:f16` = 28 B per 128 weights = **1.75 bpw of base-3
   dense-trit packing, 5 trits per byte** — the same packing family Phonon-2's encoder
   uses, minus their second magnitude bit. A real 5.93 GB `Ternary-Bonsai-2-27B-PTQ1_0`
   decode pack exists, and the GPU lane consumes it directly
   (`gemv_ternary_trit_rowtiled8`, Issue 628 T2 layout B, "1.75 bits/weight vs 2.125 →
   17.6% cut"). So the honest comparison is: **our 3 levels at 1.75 bpw vs their 5 levels
   at ~2.1 bpw** — their second magnitude costs roughly +0.35 bpw and is bought with QAT.
   That delta is a *training-recipe* item (riir-train), not an inference-substrate item:
   consuming 5-level weights needs the trainer to produce them; PTQ-ing 3-level trits into
   5 levels buys nothing (item 1).
3. **State occupancy is a stable prior.** Zero ≈ 62.6 %, ± ≈ 18.7 % each; stable within
   0.4 pp across a 14× parameter gap (0.6B vs 8B); FFN layers 2–4 run ~10 pp sparser
   (learned, not forced). Useful as sanity priors for our TWT / ternarize probes
   (`twt_ternarize_probe.rs`) and any future sparsity claims.
4. **The vocabulary is the expensive part.** int8 embedding tables = 32.1 % of the 8B
   artifact's bytes (47.5 % at 0.6B); embeddings/output-head/norm-gains stay higher-precision
   by class (lookup rows don't enjoy linear-sum error cancellation; output head decides on
   small logit margins). Our 27B carries a 248k vocab — the same composition pressure; any
   "shrink the model" lever that ignores the embed lane caps out at ~2/3 of the bytes.
5. **Knowledge survives better than behavior** (96 % general knowledge vs 79 % tool-use
   retention at 4.2× fewer bytes; one behavior axis — format discipline — *exceeded* the
   reference via behavior training). Same shape as our Issue-750-T3 per-family retention
   law: aggregates hide per-family flips.
6. **Coded-wire transport.** Their download codes the ternary lane near its information
   content (whole lane ships at 0.552 of raw; order-0 bound within 0.007 %; coded size
   correlates with zero-share at r = −0.92) and expands losslessly at load — serve 2-bit,
   download ~1.6-bit-equivalent. We already dequant-once-at-load for the ANE/GPU paths, so
   the expansion seam exists; what's missing is the coded container.

## 4. Technique census — signal-diff vs what already ships

| Their technique | Our shipped cousin | Diff (one read each) |
|---|---|---|
| Per-row int8 requant, epilogues written into next layer's 8-bit input (CPU) | `ane_prefill/requant.rs` (per-row int8 of ternary weights, ANE Form C) + kernel_opt `fused-transform-quantize-writer` (B182) | **Covered class** — theirs is the CPU-VNNI instance of the same fuse-into-quantized-input law |
| One-second-bucket CUDA graphs built at load; batch label-looping 16 steps/launch | kernel_opt `shape-bucketed-graph-cache-lru` (B189) + Issue-965 graphs lane | **Covered class** |
| GPU decode syncing with host every 16 steps, not per token | B182 split-submission / in-flight-cap discipline | **Variant** — decode-loop-shaped instance; recorded, not new |
| Unpack packed bytes once at load; decode never reads them | Our load-time dequant to f16/int8 (ANE/GPU paths) | **Covered** |
| Base-3 packing, 5 trits/byte | **SHIPPED HERE**: `PTQ1_0` (id 143, 1.75 bpw file format) + `gemv_ternary_trit_rowtiled8` (Issue 628 T2 layout B) | **Covered — shipped.** Their variant adds one magnitude bit per non-zero (5 levels); the packing itself is ours (round-1 review caught this note initially reading "not adopted" — the substrate-first vocabulary miss, recorded in §7) |
| 5-level two-magnitude QAT | riir-train Plan 255 (`lr_qat` / `lota_ternary` — adapter-level QAT); Bonsai full-model ternary from the trainer | **Training-track row** (see §5 F3) |
| Coded-wire transport (r = −0.92 zero-share law) | PTQ1_0 already ships the packed-on-wire → expand-at-load seam (5.93 GB file → ~7.2 GB container) | **Half new** — the entropy coder is the missing half (F1) |

## 5. Fusion candidates (recorded, not planned)

- **F1 — entropy-coded transport ON TOP of PTQ1_0:** PTQ1_0 already ships packed +
  expand-at-load; the new half is only the coder. Using their own occupancy prior
  (62.6 / 18.7 / 18.7), order-0 entropy ≈ 1.33 bits/trit against PTQ1_0's 1.625 packed
  bits/trit (qs 24 + qh 2 = 26 B per 128 trits) → ≈ 0.82× on the ternary lane — a much smaller win than their 0.55× raw-ratio
  (which prices against uncoded 2-bit). Low priority — our packs are local-disk today;
  matters only if/when models are distributed.
- **F2 — league watch:** FermionResearch's llama.cpp fork + Neutrino Engine are potential
  `perf-rematch` opponents (they publish 8B-class ternary serving numbers: 33.7 tok/s MLX,
  30.7 L4). **Owner-gated** — adding an opponent row is a perf-rematch lane decision, not
  taken here.
- **F3 — training-recipe rows (riir-train backlog):** (a) 5-level two-magnitude ternary QAT
  as a future Bonsai round lever — their 72.1-vs-65.75 edge is **a vendor-harness gap on a
  sibling Bonsai size (their 8B row), not measured on our 27B**, so it bounds nothing about
  Ternary-Bonsai-2-27B; it only says the lever is plausibly worth a round;
  (b) per-tensor-class precision assignment (embeds int8 / norms fp32) as a composition
  rule; (c) their "two independent behavior checks improved both axes" battery note. Single
  rows recorded here — batch into a riir-train plan when ≥3 accumulate (the §3.5 standing
  rule).
- **F4 — Phonon-1 as an encoder datapoint:** same size-class/role as our laya lane
  (`convaiinnovations/laya` — different lineage, NOT the same base model) but
  ternary-QAT'd at 2.4 bpw: a "what does QAT do to a 0.6B encoder" reference if the laya
  lane ever considers a ternary checkpoint. Not a reflex comparison lane (reflex measures
  classification over text, not ASR).

## 6. Verdict

**Gain — league-intelligence record + lane-candidate recon.** No Super-GOAT claim (nothing
novel of ours), no plan (audio owner-deferred; training rows wait for the backlog batch),
one recon line in Issue 015. Adoption of Phonon-2 refused-for-now on three grounds: the
owner's M5-Ultra defer, English-only vs EN/TH scope, and no CoreML bundle for the landed
T2 load path. The competitor analysis (§3) is the durable value: an external lab just
published (a) the PTQ-near-chance table that validates our from-trainer-ternary premise,
(b) a measured 8B-class gap over the Bonsai family name, and (c) state-occupancy +
per-class-precision priors our probes and trainer can consume.

## 7. Provenance

- Pages fetched 2026-09-30 (announcement + one-eighth-the-bits + GitHub repo page). No
  clone — the note records claims at page level; engine internals (C/MLX/CUDA paths) are
  described from their own announcement/README, **not source-verified**. If a future task
  mines their kernels or fork, clone to `.raw/` at a pinned sha first (§0.5).
- Internal-first sweep: `.research/` + `.plans/` + `*.md` grepped workspace-wide for
  `Phonon|Parakeet|Neutrino|FermionResearch|one-eighth` — hits only at katgpt-rs Research
  147 / Plan 446 (parakeet.cpp *decoder-side* phrase boosting — complementary, different
  repo/surface) and an unrelated "one eighth" prose match in riir-train Plan 341. riir-train
  QAT cousins read: Plan 255, Bench 446 (`lr_qat` vocabulary-translation record), Issue 561
  (the training-leads backlog pattern F3 joins); Research 411 read for the §3.1
  corroboration.
- **Round-1 review caught this note's own vocabulary-translation miss:** the §4 base-3 row
  initially read "not adopted" because the sweep grepped Fermion vocabulary in `*.md` and
  under-read the `quant/*.rs` substrate — `PTQ1_0` (1.75 bpw base-3 file format) and the
  Issue-628-T2 trit-packed GEMV shipped here all along. Fixed in place pre-commit; recorded
  as the standing lesson: a competitor-technique census greps the TECHNIQUE's shape
  ("5 trits per byte", "base-3", bpw arithmetic), not just the competitor's name.
- `*.rs` census: audio-input zero hits workspace-wide (Research 001 §3 stands);
  `twt_bonsai_audition.rs` is TWT calibration profiling, not audio.
