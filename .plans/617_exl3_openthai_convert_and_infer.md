# Plan 617 — OpenThai-SystemOne EXL3: convert (4090) + infer (Metal) — the r1 consumer lane

**Status:** IN PROGRESS — Phase A executing on the 4090 (GO received 2026-10-02); A1–A3 DONE, A4 (CUDA half) next. Lane-home verdict RECORDED below (Reflex comparison lane recommended, Rethink declined). AMENDED 2026-10-02 per the verdict ping-pong (AGREE with amendments; the dedicated claude_code reviewer backend lacked auth — independent sub-agent reviewer, claims verified against the records before applying): pre-registered A5 bar added; §14-primary/r1-not-literal citation fix; GDN-hybrid scope; §14 re-run gain definition; B2 substrate-reuse + dep-edge. Cheap-falsification-first order: Phase A buys the accuracy answer for ~half a day before any Metal serving work is priced. **A6 EXECUTED 2026-10-02 on the M3 (the plan's own "any box that meets the wall" clause): reflex Bench 107 — the 400M wins 6/9 EN suites vs the published modelless rows, beats the 68M on all 9 (overall +2.6 pt, flipping the family's −6.1), lane seat moved 68M→400M; A1–A5 remain the 4090 half-day.**

**Consumer:** `iapp/OpenThai-SystemOne` @ `5d04bcca` (Apache-2.0) — Qwen3.5-0.8B tower + 256-slot head; fp32 board pins at Bench 074/084/086 (massive 0.9200 · sib200 0.8382 · xnli 0.8967/0.9000 · wisesight 0.4750/0.4675).
**Substrate:** riir-infer Issue 034 (the §14-primary serving-arm trigger; r1 closest-but-not-literal — its recorded text requires 4 bpw residency / >100k context on 24 GiB, which this 0.8B single-shot consumer needs neither of); reader + Metal dequant already landed (`.docs/001`, two-tier oracle: decode BIT-EXACT / Hadamard tolerance-class).
**Expected size math (0.8B):** fp32 ~3.2 GB · bf16 ~1.6 GB · **EXL3 4.0 bpw ~0.4 GB** (optionally 3.0 bpw ~0.3 GB for the curve).

## Lane verdict (the worth question, recorded)

**Reflex+OpenThaiEXL3: YES (comparison lane). Rethink+OpenThaiEXL3: DECLINED.**

- OpenThai already lives in riir-reflex as a comparison lane; an EXL3 variant extends
  that lane family and quantifies the lossy question on the board with per-suite cells.
- Rethink is the PRODUCT lane: serving there means (a) reopening Thai for product —
  owner-closed (C9: "not focus in Thai but just for research sake"; reopen trigger
  documented), (b) serving a model whose latency (100 ms–1.6 s p50) fails every serve
  bar in the stack (reflex G2 ≤1 ms modelless; instinct 1 ms bag bar; encoder ms-class
  already record-only), and (c) riir-train vessel-minting work for cells that could not
  clear serve bars anyway — stacked recorded declines.
- OpenThai's EN strength (xnli 0.8967 / massive 0.9200) is best captured in Rethink the
  way the record already does: teacher-side in its three recorded roles — synth-corpus
  agreement VETO, `--synth-teacher`, and `--distill-teacher openthai` single-teacher
  fallback (Benches 083/089/104) — not by serving the teacher.
- EXL3's own axis (`.docs/001` §2) is MEMORY, not throughput — the consumer for a
  smaller footprint is the M3-resident dev/arena story (Reflex), not a GPU serving tier.

## The Bekko SystemOne family — EXL3 verdict (owner ask 2026-10-02)

Owner asked to add `hotchpotch/bekko-system-one-v0` **68M** and **400M** to this lane.
Verdict: **EXL3 = structural NO for the whole bekko family; the 68M is already seated; the 400M joins the board via A6.**

- **EXL3 structural NO:** bekko is a **ModernBERT-compatible shared-prefix ENCODER**
  (base `cross-encoder/ettin-reranker-400m-v1`; bidirectional; the state+instruction
  prefix never attends candidates, candidates never attend each other; Choice/Noul/Score
  heads over mean-pooled candidate branches). EXL3 is a CAUSAL-DECODER format and
  exllamav3's converter accepts decoder archs only — this is the wrong model CLASS for
  the format, not a conversion risk. No convert task is filed for bekko, ever, under
  this plan.
- **The size motive is also weak:** 68M ≈ 244–272 MB fp32; the author's own browser
  exports already exist (17M 29 MB / 68M 196 MB ONNX, INT8 row-wise embeddings + FP32
  blocks). A SMALLER bekko in OUR runtime would be a separate encoder-port plan
  (laya-encoder substrate + ModernBERT + the shared-prefix mask + the three heads,
  G5-gated) — a new filing if M3-resident bekko ever becomes a product want. Not here.
- **68M — already seated:** Bench 103 (17M/68M; owner wired the `--bekko` lane
  2026-10-01; 68M seats 9/9 EN dataset suites, wins 4/7 incl. xnli +15.3 / massive
  +8.7; distill follow-ups measured NEGATIVE in Bench 104). Nothing to add.
- **400M — the recorded follow-up (A6):** Bench 103 recorded it UNMEASURED
  ("CPU-infeasible ~2 h; the natural 4090-box follow-up"). A6 runs it through the
  EXISTING `--bekko` lane — zero code change (the lane takes `BEKKO_MODEL` env;
  Bench 103's env-override path proved it).

## Phase A — convert + cheap falsification (4090, ~half day–1 day)

- [x] A1 — env: exllamav3 venv under `.raw/` on the 4090 (Windows box: mind the
      git-over-SSH quirks — schtasks `gitsync` for repo syncs; `;` separators). Download
      `iapp/OpenThai-SystemOne` (weights cached under `.raw/hf`). Record the
      arch-support check — **expect the GDN HYBRID (gated-delta-net + causal-conv,
      Research 003) as the LIKELY exllamav3 failure mode; fail-loud STOP + record if
      the converter does not know it — a clean convert is the surprise, not the
      baseline.**
      **DONE 2026-10-02 (4090) — THE SURPRISE FIRED: arch check PASSES.** exllamav3
      1.5.3 (venv `.raw/exl3-env`; torch 2.14.1+cu130 — the PyPI Windows torch is
      CPU-ONLY, the +cu130 wheel comes from download.pytorch.org; JIT ext build needs
      ninja + CUDA_HOME + an EXPLICIT `vcvarsall.bat amd64` env — torch's internal
      MSVC detection produced C1060 heap-death at any parallelism, 14.41.34120 is the
      only installed toolset). `ARCHITECTURES` knows `Qwen3_5ForCausalLM`
      (GatedDeltaNet + interval-4 full attn + `interleaved_gate=True` — the exact
      hybrid); it does NOT know `OpenThaiSystemOneForDecision` (the wrapper). The
      pre-registered "GDN hybrid = likely failure mode" is REFUTED at the arch level;
      the wrapper arch remains the A2 surgery as planned. Checkpoint cross-check
      (`model.safetensors` header): 320 tower tensors under `model.layers.*` with the
      Qwen3_5 GDN key spellings (separate `in_proj_qkv`/`in_proj_z`/`in_proj_b`/
      `in_proj_a`, fused qkv already, conv1d [6144,1,4], A_log/dt_bias [16]) +
      full-attn q/k/v/o + q_norm/k_norm + NO lm_head tensor (tied embeddings in
      fact, though config declares false) + head trio (`slot_head.weight/bias`,
      `log_temperature`). Config: 24 layers, 8Q/2KV heads dim 256 (attn_output_gate
      true), linear 16K/16V heads dim 128, intermediate 3584, vocab 248339,
      rope_parameters{mrope_interleaved, sections [11,11,10], partial 0.25, theta
      1e7}, `mtp_num_hidden_layers: 1` with NO MTP tensors shipped.
- [x] A2 — tower extraction surgery: standard causal-LM checkpoint (config + tokenizer
      kept; the 256-slot head EXCLUDED — it stays bf16 beside the pack, tiny). Record
      BLAKE3 of the extracted checkpoint.
      **DONE 2026-10-02 — `scripts/plan617_extract_openthai_tower.py`.** 320 tower
      tensors renamed `model.*` -> `model.language_model.*` (exllamav3 Qwen3_5 key
      prefix; the Qwen3Next arch would have matched `model.*` but carries the FUSED
      in_proj_qkvz/ba spellings this checkpoint does not use), head trio excluded to
      `.raw/packs/openthai-head/openthai_slot_head.safetensors`. Config flattened,
      `architectures=["Qwen3_5ForCausalLM"]`, `tie_word_embeddings=true` (source
      declares false but ships no lm_head — exllamav3 alt-key path; the convert log
      confirms `Cloned lm_head from model.language_model.embed_tokens`),
      `mtp_num_hidden_layers=0` (none shipped). FULL byte-compare 320/320 vs source
      (not sampled). **BLAKE3 tower `462ccc445c88afb483a52ea624f4039dd81ebaee74a52e9a5c57e929523f6881`**
      · head `67007722f965f69968c9cbcacd711219bd363bf96d9339545d5a588584a0d6c1`.
      Extracted dir `.raw/hf/openthai-tower-qwen35-0.8b` (EXTRACTION_RECORD.json in
      dir).
- [x] A3 — convert @ 4.0 bpw (the verified fixture class; `head_bits`/H5 class to match
      the pin-era pack shape; optionally a 3.0-bpw pass for the size/quality curve).
      **Calibration mix recorded** (thai_wisesight + thai_sib200 + EN suites proportions
      — per-suite calibration sensitivity is the lossy-law exposure; §2). Output to
      `.raw/packs/`. Record achieved bpw + `Exl3Residency` numbers. Era-gate check:
      if the converter emits `quantization_config.version` ∉ `{"1.4.2"}`, extend the
      known-good set ONLY with per-pack verification — never silently.
      **DONE 2026-10-02 — clean convert (the surprise, fully).** `-b 4.0 -hb 5`
      (pin-era H5 class, codebook mul1), `-cd` the custom cal file — **calibration
      mix RECORDED** (`scripts/plan617_build_calibration.py`, seed 617, from the
      reflex canonical pool `.raw/datasets/`): 250×2048 rows, thai 116 (wisesight 87
      + sib200 29 = 46.4% — sib200 pool-capped) + EN 134 (ag_news/banking77/emotion/
      massive/sst5/xnli 21 each, prompt_injections 8 pool-capped); BLAKE3
      `ea2bfde47b054c86798e32d81c0fe56d9b7ca3eda49015c80b2afb07fcc88c4a`. All 24
      layers quantized clean (proxy_err ~1e-4, cos ~1e-8..1e-5, SQNR ~47 dB; per-layer
      4.0 bpw, lm_head 5.0). **Pack: `.raw/packs/openthai-tower-exl3-4.0bpw`,
      920,167,472 B (0.857 GiB)** — quantized layers ≈ 408 MB @ 4.0 bpw (the plan's
      ~0.4 GB holds); embed_tokens stays bf16 = 509 MB (half the pack — noted;
      consistent with the pin-era 27B pack's fp16 embed). **ERA-GATE CHECK FIRED:
      the converter emitted `quantization_config.version = "1.5.3"` ∉ known-good
      `{"1.4.2"}`** — extension of the known-good set is gated on A4's per-pack
      verification (the plan's own rule), one commit with the gate evidence.
      GPU state during convert: sibling riir-train plan435 CUDA training active
      (11.3/24.5 GiB VRAM) — convert is quantization compute, not a correctness
      gate; shared-GPU noted per the exclusivity rule's scoping.
- [ ] A4 — reader parity in-repo: pack opens through `Exl3Pack`, residency report,
      full-pack gate on Metal AND CUDA (plan 004's harness shape): **decode-stage
      bit-exact + Hadamard stages at the recorded tolerance gates** (the record's
      two-tier oracle). Coverage floors pinned. Repo test committed.
      **CUDA HALF DONE 2026-10-02 — `real_pack_openthai_bit_exact_full`
      (crates/riir-infer-gpu/src/exl3_dequant_cubecl.rs): 151/151 layers /
      751,435,776 weights / 0 bit mismatches (v1≡v2≡CPU oracle per layer), K4
      150 + K5 lm_head (mul1), wall 6.6 s on the 4090; floors pinned to THIS
      pack's metadata (151 groups / 751.3 M weights / both K classes).**
      **ERA SET EXTENDED: `KNOWN_GOOD_ERA_VERSIONS = ["1.4.2", "1.5.3"]`
      (src/quant/exl3_pack.rs) — the extension this gate's green run is the
      per-pack verification for, exactly per the A3 era-gate check's rule;
      era-gate unit tests 10/10 still green incl. the refuse side;
      post-extension the gate opens the pack through the DEFAULT Verify door.
      METAL HALF: the same test on the M3 — NOT run here (4090 box; M3 session
      owns that arm, same test name).**
- [ ] A5 — hybrid board re-run (the cheap falsifier): Python hybrid server on the 4090
      (exllamav3 tower forward → hidden states → their torch head + decide contract,
      `permutations=1`; the 084 disclosed-patch pattern for the dtype/numerics pin).
      Point reflex `--openthai` at it; run the 17-suite board + determinism pin.
      **PRE-REGISTERED RETENTION BAR (pinned BEFORE the run — the post-hoc-bar
      refusal; measured noise scale = 084's fp32 M3↔4090 cross-posture deltas of
      ≤0.8 pt):**
      - acc: |Δacc| ≤ **1.0 pt per suite** vs the fp32 pin;
      - calibration: Δreadout-ECE ≤ **+0.050 absolute per suite**;
      - determinism pin green (×2 byte-compare, the T2.3 law);
      - **mixed-outcome rule: ALL 17 suites must pass for Phase-B GO — ANY breach =
        NO-GO.** One pack serves every suite; a post-hoc suite-scoped carve-out is
        refused — an owner re-scope would be a NEW decision, not this plan's.
      → **GO/NO-GO for Phase B.** A retention failure here kills Phase B for the cost
      of half a day — that is the point of the order.
      **SERVER LANDED + SMOKE + PARITY PROBE GREEN (2026-10-02, commits `09b1eb2` +
      this one): `scripts/plan617_openthai_exl3_server.py`** — the fp32 lane's exact
      wire; tower forward over the pack (exllamav3, module loop minus the
      logits_output-capped lm_head, flash-attn params with the page-aligned
      batch_shape geometry, the recurrent slot FREED per request — single-shot
      requests leak slots otherwise); head half THEIR code verbatim (Formatter
      imported from the reflex checkout = byte-identical prompt by construction;
      slot-head bf16 matmul → f32, log_temperature per-qtype, option-count mask,
      their decode semantics). Wrapper keys ride `head_meta.json` (the converter's
      config.json carries only the tower). Smoke: billing/technical → billing @
      0.978. **PARITY PROBE (scripts/plan617_a5_parity_probe.py, 6 real states from
      the canonical pool, all three question types, both servers live on this box):
      6/6 top-1 agreement; max prob delta 0.008 (noul) – 0.074 (banking77), mean L1
      0.089 — the healthy 4-bpw signature, no rendering/position/temperature bug.**
      **BOARD: WAITING ON GPU EXCLUSIVITY — the sibling plan435 CUDA training has
      held the GPU at ~100% since 18:56; the pre-registered bar is a correctness
      gate and waits (rule + contaminated-latency law). The run command is staged:
      `OPENTHAI_SERVE_URL=http://127.0.0.1:8011 REFLEX_BENCH_HOST=4090-windows cargo
      run --release --bin harness -- --openthai --skip-laya --out .benchmarks/<N>_…`
      + the thai pair as the second run (the 084 two-run shape; fp32 pins for the
      bar = 084's OWN 4090 table — same box: massive 0.9200 / sib200 0.8382 / xnli
      0.9000 / wisesight 0.4675 / ag_news 0.8900 / banking77 0.6540 / typed 0.5360…).**
- [x] A6 — **(reflex-side, parallel, no EXL3 dependency) the bekko-400M board seat** —
      **EXECUTED 2026-10-02 on the M3 (not the 4090 — the box met the wall here:
      ~19 min for all 4,648 questions on CPU; the Bench-103 "~2 h CPU-infeasible"
      estimate refuted — it extrapolated the 17M compute ratio, the real
      shared-prefix batched ratio is far kinder). reflex Bench 107
      (`riir-reflex/.benchmarks/107_bekko_v0_400m_board/`, commit `e054d7e`):
      bekko-400M wins 6/9 EN dataset suites vs the byte-reproduced published
      modelless rows (ag_news 0.9150 · xnli 0.8600 · massive 0.9100 · typed
      0.6235 · sst5 0.4550 · code_fixtures 0.5938; reflex keeps emotion /
      prompt_injections / banking77), beats the 68M on ALL 9 (overall +2.6 pt
      vs the 68M's −6.1 — the family sign flips), and is the strongest lane on
      the board's EN suites overall (vs openthai 0.6205 and the Rethink hybrid
      0.6270 — read under the author-flagged family-familiarity caveat).
      Revision pinned to the FULL hash (resolved `4aeb85b9d4042d75d8b8adf6ff7ba9e4629510ba`
      via the HF API); comparator-posture law enforced (modelless rows byte-
      reproduced 9/9 first); determinism witness across two processes
      (code_fixtures 0.5938); license MIT; board publish acc-only (this run's
      box load 5→9.1, sibling active — the Issue-021 wall); the lane seat moved
      68M→400M (the 400M strictly dominates per-suite). Measurement-only
      posture unchanged.** (The original task text below.)
      - [ ] A6 — **(reflex-side, parallel, no EXL3 dependency) the bekko-400M board seat**
      (the Bench-103 recorded follow-up): on the 4090 (or any box that meets the
      wall), run the existing reflex `--bekko` lane with
      `BEKKO_MODEL=hotchpotch/bekko-system-one-v0-400m`, revision pinned to the FULL
      commit hash at run time (the card's short pin `4aeb85b`; the Bench-103
      `BEKKO_REVISION` convention) — all 9 EN dataset suites, `--nb-select
      --ridge-select` (+ `--oc-select` for typed_decisions), byte-reproducing the
      published modelless rows FIRST (the comparator-posture law, Bench 103 ⛔).
      Publish acc-only if the box state is unfit (the Issue-021 wall as before).
      **Caveats carried from the card + Bench 103:** bekko v0's generalization is
      author-flagged weak (training data shares dataset families with eval suites —
      read wins as family familiarity, never broad generalization); English-only (the
      two thai suites stay openthai/encoder-only); measurement-only posture — the
      lane is harness-side, never in a release set; record license as MIT (Bench 103
      addendum, verified).
      **PREP DONE (2026-10-02, CPU/network while the GPU waits): bekko venv at
      `riir-reflex/.raw/bekko-env` (torch 2.10.0+cu130 — CUDA on the 4090, the
      card's transformers 5.17 + sentence-transformers 6.1 pins); revision RESOLVED
      `4aeb85b` → FULL `4aeb85b9d4042d75d8b8adf6ff7ba9e4629510ba` (lastModified
      2026-09-30); weights cached (3.01 GB snapshot). RUN waits on the same GPU
      window as A5 (the board run is a measured gate).**

## Phase B — Metal serving arm (gated on Phase-A per-suite PASS; multi-day)

- [ ] B1 — packed-resident load path; measure dequant-in-forward (streamed per-layer)
      vs dequant-once-at-load for the 0.8B single-shot shape — runtime memory vs
      latency, both measured, **PER LAYER CLASS** (GDN fixed-state vs KV-attention vs
      dense MLP — the GDN hybrid makes one average a lie; §17.6 scope note: never
      transferred by analogy).
- [ ] B2 — forward port in Rust over the pack — **substrate-side in riir-infer** (the
      laya-riir precedent), REUSING the qwen35-deltanet substrate this repo carries
      (`qwen35_deltanet_config_from_gguf_metadata` loader, deltanet forward family,
      GDN chunked-prefill kernels) + the head-side pieces Research 003 pins
      (tokenizer special tokens, `SlotHead` Linear(H→256) at `<|ts_answer|>` states
      with slot 255 = abstain, per-question-type log-temperatures, the padded
      multi-option single forward, the renormalized-probability decide contract);
      reflex consumes via a re-export shim behind an opt-in feature — **name the dep
      edge and run the boundary check (`ci_boundary_contract.sh` via the
      boundary-guard skill) before landing; reflex's sibling-layout doc today names
      only `riir-infer-laya`.** G5 parity vs frozen captures from A5's server before
      any number is quoted.
- [ ] B3 — reflex lane `openthai-exl3` (comparison-lane family law; lane file filed in
      riir-reflex); 17-suite board re-run through the Rust path; determinism pin;
      site republish.
- [ ] B4 — §14 gate re-run with runtime numbers under the **pre-registered,
      cell-appropriate gain definition** (the records' decode-step metric has no
      decode loop here and no f16/q4 GGUF incumbent exists for OpenThai — define:
      bytes-at-load + runtime residency vs the fp32/bf16 server, and paired latency
      vs the fp32 server on the same box; cite §16's amendment as well as §17.6).
      **Expected posture pre-registered: STAYS OPT-IN even on full success** — a
      lane consumer is not a default-path gain; feature-flag promotion/demotion per
      GOAT.

## Phase C — verdict + records

- [ ] C1 — lane-home record re-affirmed or revised with evidence; promote/demote per
      results; doc-sync (riir-infer `.docs/001` cross-ref, reflex HISTORY entry when
      the lane lands).

## Explicitly deferred

- [-] Rethink/riir-instinct product posture (see Lane verdict above) — re-opens only if
      the owner reopens Thai product (C9 trigger) AND cascade economics ever clear.
- [-] Rust-native EXL3 encoder — research project; the reader stays read-only.
- [-] 3.0-bpw curve pass — only if 4.0 passes A5 and the size motive needs more room.
