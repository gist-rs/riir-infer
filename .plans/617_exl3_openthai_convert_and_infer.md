# Plan 617 — OpenThai-SystemOne EXL3: convert (4090) + infer (Metal) — the r1 consumer lane

**Status:** PROPOSED — awaiting Phase-A GO; lane-home verdict RECORDED below (Reflex comparison lane recommended, Rethink declined). Cheap-falsification-first order: Phase A buys the accuracy answer for ~half a day before any Metal serving work is priced.

**Consumer:** `iapp/OpenThai-SystemOne` @ `5d04bcca` (Apache-2.0) — Qwen3.5-0.8B tower + 256-slot head; fp32 board pins at Bench 074/084/086 (massive 0.9200 · sib200 0.8382 · xnli 0.8967/0.9000 · wisesight 0.4750/0.4675).
**Substrate:** riir-infer Issue 034 (the §14/r1 reopen); reader + Metal dequant already landed (`.docs/001`).
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
  way the record already does: teacher-side (synth-corpus openthai VETO, distill-teacher
  pass) — not by serving the teacher.
- EXL3's own axis (`.docs/001` §2) is MEMORY, not throughput — the consumer for a
  smaller footprint is the M3-resident dev/arena story (Reflex), not a GPU serving tier.

## Phase A — convert + cheap falsification (4090, ~half day–1 day)

- [ ] A1 — env: exllamav3 venv under `.raw/` on the 4090 (Windows box: mind the
      git-over-SSH quirks — schtasks `gitsync` for repo syncs; `;` separators). Download
      `iapp/OpenThai-SystemOne` (weights cached under `.raw/hf`). Record the
      arch-support check (exllamav3 model type for the tower) — **fail-loud STOP +
      record if the converter does not know the arch.**
- [ ] A2 — tower extraction surgery: standard causal-LM checkpoint (config + tokenizer
      kept; the 256-slot head EXCLUDED — it stays bf16 beside the pack, tiny). Record
      BLAKE3 of the extracted checkpoint.
- [ ] A3 — convert @ 4.0 bpw (the verified fixture class; `head_bits`/H5 class to match
      the pin-era pack shape; optionally a 3.0-bpw pass for the size/quality curve).
      **Calibration mix recorded** (thai_wisesight + thai_sib200 + EN suites proportions
      — per-suite calibration sensitivity is the lossy-law exposure; §2). Output to
      `.raw/packs/`. Record achieved bpw + `Exl3Residency` numbers. Era-gate check:
      if the converter emits `quantization_config.version` ∉ `{"1.4.2"}`, extend the
      known-good set ONLY with per-pack verification — never silently.
- [ ] A4 — reader parity in-repo: pack opens through `Exl3Pack`, residency report,
      full-pack dequant bit-exact vs the CPU reference on Metal AND CUDA (plan 004's
      harness shape). Repo test committed.
- [ ] A5 — hybrid board re-run (the cheap falsifier): Python hybrid server on the 4090
      (exllamav3 tower forward → hidden states → their torch head + decide contract,
      `permutations=1`; the 084 disclosed-patch pattern for the dtype/numerics pin).
      Point reflex `--openthai` at it; run the 17-suite board + determinism pin;
      **per-suite retention verdict vs the fp32 pins (acc AND readout-ECE)**.
      → **GO/NO-GO for Phase B.** A retention failure here kills Phase B for the cost
      of half a day — that is the point of the order.

## Phase B — Metal serving arm (gated on Phase-A per-suite PASS; multi-day)

- [ ] B1 — packed-resident load path; measure dequant-in-forward (streamed per-layer)
      vs dequant-once-at-load for the 0.8B single-shot shape — runtime memory vs
      latency, both measured (§17.6 scope note: never transferred by analogy).
- [ ] B2 — forward port in Rust (tower + head + decide contract) over the pack; G5
      parity vs frozen captures from A5's server before any number is quoted.
- [ ] B3 — reflex lane `openthai-exl3` (comparison-lane family law; lane file filed in
      riir-reflex); 17-suite board re-run through the Rust path; determinism pin;
      site republish.
- [ ] B4 — §14 gate re-run with runtime numbers (its own trigger text); feature-flag
      promotion/demotion decision per GOAT.

## Phase C — verdict + records

- [ ] C1 — lane-home record re-affirmed or revised with evidence; promote/demote per
      results; doc-sync (riir-infer `.docs/001` cross-ref, reflex HISTORY entry when
      the lane lands).

## Explicitly deferred

- [-] Rethink/riir-instinct product posture (see Lane verdict above) — re-opens only if
      the owner reopens Thai product (C9 trigger) AND cascade economics ever clear.
- [-] Rust-native EXL3 encoder — research project; the reader stays read-only.
- [-] 3.0-bpw curve pass — only if 4.0 passes A5 and the size motive needs more room.
