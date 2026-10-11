# Research 012: clef-text-0.6b CoreML — the ANE decision lane (FluidInference student of Cloudflare's Clef)

> **Source:** FluidUse `@ 0230049c4e81f8de53ae1086b20cd9106c511105` (Apache-2.0, clone-verified, removed after this note) · HF `FluidInference/clef-text-0.6b-coreml` @ revision `3c61a60954478ef385aa42d67d5764642dcc9135` (asset sha256 table pinned in §2) · HF `FluidInference/clef-text-0.6b` + `Cloudflare/clef` cards · Cloudflare blogs "Introducing Clef decision models" (2026-10-01) + "Clef-omni … faster … cheaper" (2026-10-09)
> **Date:** 2026-10-11
> **Status:** RECORD 2026-10-11 — verdict **Gain** (integration distill); plan filed (`.plans/619_clef_text_0_6b_ane_lane.md`). Filed from owner directive: "add ane lane and more … extract it like encoder to Instinct and call verdict".
> **Related Research:** 001 (FluidInference audio side), 002 (FluidUse e8 — the direct precedent for "official Swift runtime → our Rust lane" integration distills); riir-reflex `.research/005` (Clef/JDI crosswalk), `.research/007` (local clef-flash lane wire); riir-train `.research/464` (Clef post-training recipe vs our typed head)
> **Related Plans:** 612 (e8 table stack — the parity-gate + promotion-declined lesson this lane inherits); riir-reflex Plan 011 (the `--clef` HTTP comparison lane, CLOSED); riir-train Issue 623 (clef-flash 9B distill-teacher capture, ACTIVE on the 4090 — no collision, see §3)
> **Cross-ref (riir-reflex / riir-rethink / riir-instinct / riir-train):** reflex lanes + bench 138 (next); rethink downstream option (owner-gated, details private); instinct arena seats via the reflex runner; riir-train 623 teacher posture cross-option
> **Classification:** Public — every source is Apache-2.0 open weights + an open runtime; nothing here touches rethink-class product posture.

---

## TL;DR

`clef-text-0.6b` is FluidInference's **0.70B open-weight decision model** — a Qwen3-0.6B backbone + Cloudflare Clef's joint schema head, distilled from clef-flash (9B), answering typed `choice`/`score`/`noul` questions (exactly our `decision_wire` vocabulary, Jev-API compatible) in one prefill pass — published as **CoreML artifacts that place every decoder op on the Apple Neural Engine** at all four length buckets. The official runtime is Swift (FluidUse); the Rust host is the missing piece, and **we already ship 70% of it**: `riir-infer-laya/src/laya/riir/ane.rs` (objc2-core-ml, digest-pinned `.mlpackage` fetch, content-addressed `.mlmodelc` compile cache, host-side embedding gather) — the genuinely new host mechanics are the **multi-function package** (`MLModelConfiguration.functionName = "L{bucket}"`, verified present in our pinned `objc2-core-ml 0.3.2`) and the **two-model decoder→head pipeline** with span-mean head packing. This is an integration distill — Gain, exactly the Research-002 class — filed as Plan 619: a `crates/riir-infer-clef` lane on our substrate (one shared ANE host with the laya lane — never a second copy of the CoreML machinery), a frozen-capture parity gate (the laya G5 shape), a reflex **local in-process `--clef` posture** measured on the 9-suite board, and a neutral owner-gated downstream pointer (capability/cost details live in the private trail, not here).

**Distilled for riir-infer (public substrate):** the reusable host pattern is "weights live outside the graph": one multi-function decoder package (a function per length bucket, shared weights), the embedding table as a raw `embeddings.f16` the HOST gathers (tied — the same table feeds the head's lexical option rows), RoPE cos/sin and the causal+pad mask as host-computed INPUTS (bucket-agnostic graph), and the head consuming span-mean matrices (`q_mean`/`o_mean`/`o2q`/`lexical`/`type_oh` + masks). Every piece of that maps 1:1 onto what `ane.rs` already does for laya.

---

## 1. What CLEF is (and why it fits our wire)

Cloudflare's Clef family (Apache-2.0, open weights, Oct 2026): **decision models, not LLMs** — no output-token generation; a `state` + a schema of typed questions goes in, a **calibrated probability per option per question** comes out of one prefill. Variants: Clef 27B (Qwen3.8 backbone, 64k ctx), Clef-flash 9B (24k hosted ctx), Clef-omni 30B-A3B MoE (audio+video+image+text, 2026-10-09). Clef-omni on BANKING77 macro-F1 **94.8** / Clef **94.2** — suites we already measure (our board: banking77 H2 0.8540; clef-flash local lane 0.9540). The 0.6b **student** (FluidInference, KL-T2 + gold-CE distill, LoRA r64, merged): "good at routing/classification, **not a knowledge model**" (ARC-C 60.7 vs teacher 97.3) — i.e. positioned exactly where our suites live.

Wire compatibility is total: their `noul`/`choice`/`score` question types are our `decision_wire` `noul`/`choice`/`score`; the reflex `--clef` HTTP lane (Plan 011) already speaks it, and the harness's `AnswerTriple` mapping (choice→pick+probs, score→expected score, noul→true-prob) transfers unchanged to an in-process lane.

## 2. The pinned artifact + runtime contract (the port's ground truth)

All from FluidUse `@ 0230049c` (Sources/FluidUse/ClefText/*) + the HF revision `3c61a609…` sha256 table baked into `ClefTextModelStore`:

- **Files (~1.3 GB, never committed — fetch + verify):** `config.json` (hidden_size, head_dim, rope_theta, pad_id, vocab_size, buckets[256,512,1024,2048], max_questions=16, max_options=96, decoder/head filenames) · `tokenizer.json` (Qwen BPE, byte-fallback) · `embeddings.f16` = fp16 `[vocab, hidden]` raw, **tied** (also the head's lexical rows) · `Decoder.mlpackage` 848 MB fp16 — **one package, a function per bucket** (`L256…L2048`, shared weights) · `Head.mlpackage` 197 MB (same per-bucket functions; head inputs are length-dependent).
- **Decoder call:** inputs `hidden` `[1,L,D]` fp16 (host gather from `embeddings.f16`; pad rows = pad_id row), `cos`/`sin` `[L, head_dim]` fp16 (rotate-half duplicated-halves layout, computed host-side per bucket and cached), `mask` `[1,1,L,L]` fp16 additive (0 / −1e4; causal for real queries; pad queries see real tokens + earlier pads) → output `states` fp16 (strides handled) → converted to f32 `[L,D]`.
- **Head call:** `states` f32 `[L,D]`, `mem_mask [L]` (−1e4 past n), `last [L]` (one-hot at n−1), `q_mean [maxQ,L]` (1/|span| over each question's INSTRUCTION token span), `o_mean [maxO,L]` (same over each option's JSON span), `o2q [maxO,maxQ]`, `lexical [maxO,hidden]` (**mean of the tied embedding rows over the option span**), `type_oh [maxQ,3]`, `q_mask`/`o_mask` (0 / −1e4) → `logits`; **per-question softmax over that question's options** (the model's own head convention — fidelity to the artifact; our lane maps the result onto `decision_wire`, our sigmoid law governs OUR gates, never the artifact's reproduction).
- **Record encoding** (the training-time token stream, ported from Cloudflare's `joint_schema_model.py`): `<|im_start|>system\n{fixed system prompt}<|im_end|>\n<|im_start|>user\nSTATE:\n{state}` + `\n\nSCHEMA FIELDS:\n` + per-field `FIELD n / ID / TYPE / INSTRUCTION: {json}` + `ALLOWED OPTIONS:` + `OPTION k: {"option_id":…, "description":…}` + `END FIELD` + `\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:`. State renders as **Python-style JSON** (`sort_keys=True, separators=(",",":"), ensure_ascii=False`, Python `repr(float)`) — the fiddliest fidelity detail; long states truncate from the right and spans shift by `prefix+state` length.
- **Concurrency:** CoreML synchronous `prediction` is not thread-safe — their runtime is an actor; our equivalent is the laya `AneRuntime` ownership pattern (one `MLModel` per instance, no shared prediction).
- **Perf (their M5 Pro, published):** decoder@512: GPU 35 ms · **ANE 64 ms** · CPU-only 200 ms; ANE decoder+head by bucket 256→29 / 512→69 / 1024→196 / 2048→591 ms; 3-question 370-tok ticket 48 ms GPU / 77 ms ANE. CoreML-vs-PyTorch parity: 15/5,118 answers differ (0.3%, all near-ties), median prob drift 0.013.

## 3. Internal census (no duplication; one active sibling to respect)

- **reflex `--clef` lane (Plan 011, CLOSED):** HTTP client over the Jev-shaped wire — hosted forwarder posture + LOCAL `mlx-community/clef-flash-4bit` behind a python sidecar (bench 113/118/119; banking77 0.9800 smoke → 0.9540 full-N; p50 geomean 680 ms). Also a distill **teacher** (`TeacherForward::Clef`, riir-train Issue 623). Plan 619 ADDS the in-process ANE posture of the 0.6b student — it does not touch the HTTP lane or the teacher seam.
- **riir-train Issue 623 (ACTIVE):** clef-flash 9B teacher capture queued on the 4090 behind the plan-437 trainer. No collision: 623 is the 9B teacher on GPU; Plan 619 is the 0.6b student on the M3's ANE. Cross-option recorded in §5(c).
- **`laya/riir/ane.rs` + Plan 612:** the Rust CoreML host (bundle fetch/digest/compile-cache/`COMPUTE_UNITS_CPU_AND_NE`, host-side table gather + `e8` sidecar consumer arm). The 612 verdict — **e8 promotion DECLINED, G1/G2 red** — is the standing lesson for any clef-side table compression: measure, never assume.
- **Tokenizer:** the laya crate already consumes `tokenizer.json` via the `tokenizers` crate — the exact-reference path (no hand port of their `QwenBPETokenizer`).
- **CUA-S1-FORMS arena** (reflex Issue 035) and the laya ANE lane (reflex Plan 002, G5-ANE 76/76) — the frozen-capture parity discipline Plan 619 reuses.
- Workspace grep `clef` pre-note: all hits are the reflex lane / 623 / proposal-051 crosswalk rows — **no ANE-lane or 0.6b-student work exists yet**; zero hits for a Rust CoreML multi-function host.

## 4. Novelty — honestly scored

External: **not novel.** ANE-runnable text models via CoreML are an established ecosystem (ANEMLL, CoreML-LLM incl. EmbeddingGemma-300M at 99.8% ANE, apple/ml-ane-transformers, whisper.cpp CoreML); clef-text-0.6b ships official ANE artifacts + the official Swift runtime + ANE/GPU/CPU benchmarks. "Lance" was checked as a reading of the owner directive: **no Lance-format usage anywhere in the Clef/FluidUse chain** — "ane lane(c)" is the correct reading. Internal: **new for our stack** — the multi-function CoreML host, the decoder→head two-graph pipeline, and the first third-party decision model seated in-process on our substrate. 1 of 4 novelty-gate questions (new *for us*, not a new class) — the Research-002 precedent exactly: **Gain, integration distill.** No Super-GOAT claim; no "candidate" wording.

## 5. Fusion angles (why this is worth a lane, not just a port)

- **(a) Downstream serving consumers (owner-gated, private trail):** a rethink-side consumption option for the lane exists; its capability/cost framing and any arming decision live in the private trail — deliberately NOT elaborated in this public note. The public half of the decision is only Phase 3's honest board read, which any consumer would need anyway.
- **(b) The honesty axis extended to a third-party model:** the perm-probe (Bench 128 law) has never run against a Clef artifact; seating the student in-process makes it measurable (option-ordering spread on typed_decisions).
- **(c) riir-train 623 teacher posture:** FluidUse also ships **ClefFlashManager** (clef-flash 9B CoreML, GPU; their ~0.5 s vs our MLX p50 ~680 ms + python sidecar hop). A future 623 re-posture candidate — recorded here, owned by 623, not this plan.
- **(d) e8-for-clef:** `embeddings.f16` (297 MB) is the same shape the 612 sidecar pattern halves — but 612's G1/G2 reds say: only as a measured follow-up, never a rider.
- **(e) "and more" backlog (FluidUse matrix, same host pattern):** clef-vision-0.8b (image decisions — the multimodal step; vision-tower host math is the delta), decision-2.0 (kai/eos), vela-2 (span questions — a question class we lack), kev-0.8b, short-reply. Each is its own go/no-go; none ride this plan.

## 6. Verdict + MOAT gate

**Gain — integration distill.** Plan 619 (this repo): the `riir-infer-clef` crate + parity gate + bench. One-line reason: the strongest open small decision model, ANE-class latency, total wire compatibility — adoptable today on substrate we already own; not a new mechanism class, so no GOAT promotion claim, feature stays opt-in.

| Surface | Repo | What lands | Visibility |
|---|---|---|---|
| Substrate (crate, host, parity, bench) | `riir-infer` | Plan 619 Phases 0–2 | **public** |
| Comparison lane (local `--clef` posture, board read, perm-probe) | `riir-reflex` | Plan 619 Phase 3 (bench 138 next) | **public** |
| Arena seating | `riir-instinct` | via the reflex runner seam — no new instinct code for measurement; a `clef` arsenal row only if an owner seats it | public |
| Downstream serving consumers | `riir-rethink` | owner-gated pointer (§5a); nothing implemented here; details live in the private trail | **private forever** |

Tier check against the MOAT table: riir-infer row = "quant formats, kernels, **encoders**, loaders … zero riir-* deps" — in scope, zero new dep classes (objc2 set + `tokenizers`, the laya-crate precedent). Priority ladder: #3 (inference-perf league) is the served surface; the healer gains nothing (no healer surface consumes model weights — same determination as Research 002). Public/private: every file this note + plan create lands in PUBLIC repos and contains zero serving-posture/economy content — the downstream consumption option is one neutral owner-gated line (§5a); its details live in the private trail.

## 7. Provenance

- FluidUse clone: `git clone --depth 1` → `0230049c4e81f8de53ae1086b20cd9106c511105`, Apache-2.0 (LICENSE in-tree), read-only, **removed after this note** per the `.raw/` law. Quotes in §2 are from `Sources/FluidUse/ClefText/{ClefTextManager,ClefTextModelStore}.swift`, `ClefVision/{ClefVisionHost,ClefRecordEncoder}.swift`, `Tests/FluidUseTests/ClefTextTests.swift` at that sha.
- HF artifact: revision `3c61a609…` with the 9-asset sha256 table (config `af356b87…` … tokenizer `be756060…`) transcribed from `ClefTextModelStore.swift` — Plan 619 T0.2 verifies at fetch time (trust the checksum, re-derive the manifest).
- Web: Cloudflare blog (2026-10-09) read directly; HF card numbers via the prior-art sweep (subagent report in-session; ANE/GPU/CPU tables + parity figures from the card).
- Prior-art sweep (§4): ANEMLL / CoreML-LLM / apple/ml-ane-transformers / whisper.cpp-CoreML / ane-infer cited with URLs in the session sweep; "pflargh"/"sleek"/"Axis" returned zero hits (empty results are results).
- Internal-first: workspace `clef` grep (all hits read: reflex 011/623/proposal-051), `.research` 001/002 re-read, `ane.rs` + Plan 612 header re-read, `objc2-core-ml 0.3.2` registry grep confirming `setFunctionName:`. Numbering: `dual_allocation_gate` clean (0/0/0) before allocating `.research/012` + `.plans/619`.
