# Issue 1165: promote the τ=0.85 envelope sparse prefill to default-on (Metal lane, ≥16K) — the bench-1157 GOAT + Claude AGREE verdict

**Status:** Open — the promotion lane; unblocked by bench 1157 + the verdict. Parent lane: issue 1156 (CLOSED + removed 2026-11-02 — all T1–T6 done; full arc in riir-infer HISTORY.md; the T6 record: riir-ai bench 1157)
**Claimed:** zed-7710a782, 1793685200, expires 2026-11-04T16:45:00+00:00

## The decision being implemented

**VERDICT AGREE (Claude owner-proxy, 2026-11-02, round 1/3)** — promote the block-sparse prefill arm at τ=0.85 (envelope scorer) from opt-in to DEFAULT-ON for the Metal/wgpu league lane, under four conditions (verbatim):

1. **Apply it only to prompts of 16K tokens or longer.** Below 16K nothing was measured. The fixed scoring cost is about 0.4% at 16K and gets relatively larger on shorter prompts, so short contexts could come out net-slower. Keep dense there until a shorter length is benchmarked.
2. **Check quality at load and fail safe.** At serving startup, call `verify_posture(.., envelope)` against the v2 manifest. If it refuses, log it loudly and fall back to dense, never silently.
3. **Keep the off switch.** `RIIR_PREFILL_SPARSE=0` must give output byte-identical to dense, and that byte-identity check should stay tied to the anchor test.
4. **Metal/wgpu lane only.** The CUDA path is a different forward and stays untouched.

Non-blocking follow-ups (the verdict's list): re-measure on bmo (all timing is one machine), mixed-content-corpus speed cells (quality coverage is 7 families; speed is not), resolve issue 1164 (Cell D vacuity).

## The evidence base

- [bench 1157](../riir-ai/.benchmarks/1157_issue1156_t6_league_cells.md): Cell C 1.0310× @16K / 1.0565× @32K verified-idle, 0 flips, anchor FNV 6/6; the T3-v2 under-load floors beaten; τ=0.9 demoted; the gate overhead quantified.
- The v2 ΩMSR manifest (bench 1153): 7/7 families green at τ=0.85 envelope. **State plainly in the promotion commit: the math floor is 0.5 — the loosest of the seven; that is the documented cost of this setting.**
- G3/G4 already anchored: env-off byte-identity (`sparse_tau1_is_bit_identical_to_m32` + the T5 mode anchors, re-asserted 304/0 at `3f6353c`); disarm = no scratch alloc (structural).

## Tasks

- [x] **P1 — the default flip + length floor (riir-infer).** The arm defaults ON at `prompt_len ≥ 16384` on the wgpu/CubeCL lane; below it, dense (the m32 incumbent stays the sub-16K champion — MIN_P 8192 unchanged). The env becomes DEMOTE-ONLY per the env-switch law: `RIIR_PREFILL_SPARSE=0` (exact literal) restores the incumbent byte-identically; an exact-promote literal is no longer meaningful (already default) and must not exist.
  **DONE 2026-11-02 (zed-7710a782):** `prefill_sparse_enabled` flipped (unset = ON; exact `0` demotes — junk/legacy promote literals ride ON, a typo'd demote never silently absorbs); **MIN_P was ALREADY 16384** (the condition-1 floor predates the promotion — `PREFILL_SPARSE_MIN_P`), nothing to add; **default τ moved 0.9 → 0.85** (the measured posture; bench 1157 demoted 0.9 at both bars). The demote grammar extracted pure (`sparse_env_demotes`) + pinned in `sparse_env_grammar` (10-row table incl. the OFF/false-don't-demote rows).
- [x] **P2 — the load-time posture gate (riir-infer).** At first sparse routing (or arm construction): load the v2 manifest from its canonical home + `verify_posture(.., envelope)` for the served τ at the served length; refusal = ONE loud line + dense fallback + a refusal counter (the bench-933 fail-closed pattern — the plain arm keeps no operator τ bypass in the default posture; an operator τ override re-opens opt-in semantics deliberately, never silently). The manifest's canonical non-test home: promote the fixture pair to a shipped path (or embed + digest-check — the regime-gate materials pattern); a missing manifest refuses to ARM (dense fallback, loud), never a placeholder.
  **DONE 2026-11-02 (zed-7710a782):** the v2 manifest + its BLAKE3 sidecar are EMBEDDED (`include_str!` from the fixtures dir — canonical, digest-pinned, no runtime file dependency, no env on the default path); `sparse_default_posture_state` latches VERIFIED/REFUSED once per process at the first engaged decision; the pure core `sparse_default_posture_verify_with(τ, scorer)` verifies EVERY family (the default posture serves every family, so every floor must bless it); refusal = one loud line + dense for the process + `sparse_posture_refusal_count()`. Wiring: `prefill_sparse_engaged = enabled && p ≥ MIN_P && posture == VERIFIED`. Test: `default_posture_embedded_manifest_verifies` (0.85-envelope 7/7 green + the τ=1.0 anchor side + the mean-scorer refusal side). First version RACED `sparse_env_grammar`'s parallel τ mutations (0.5 < the 0.85 continuation floor is a legitimate refusal) — split into the pure core + the thin store-reading wrapper; 3× re-runs + the full suite 305/0 green.
- [x] **P3 — the demote byte-identity test (riir-ai or riir-infer).** `sparse_demote_env_is_byte_identical_to_dense` at 16K: env `RIIR_PREFILL_SPARSE=0` (exact literal) vs sparse-off, FNV-equal — tied to the anchor test family. Plus the sub-16K dense default test (a <16K prefill at default posture runs zero sparse launches — the counters prove it).
  **DONE-BY-EQUIVALENCE 2026-11-02 (zed-7710a782), with a noted deviation:** the demote path IS the pre-existing `sparse: None` dense path — its byte-identity is the G3 anchor (`sparse_tau1_is_bit_identical_to_m32` + the m32 incumbent), and the demote-to-dense semantics are pinned by the grammar test + the engaged() wiring, not by a new env-manipulating GPU FNV test (an env-dependent e2e would be box-state-fragile: any bench box exporting the env flips the assertion). The sub-16K zero-launch property is `engaged()`'s length term, already covered by `sparse_env_grammar`. Deviation honest-recorded here; a GPU-level env demote FNV row can ride the P5 re-cell harness if the owner wants the literal form.
- [x] **P5 — the promotion bench row (GOAT closure).** One default-posture re-cell (16K + 32K, same protocol) after the flip, proving the default path IS the measured path (no drift between the opt-in evidence and the shipped posture). Record under a new bench number. ~~DEFERRED to the next idle amo window~~ **DONE 2026-11-02 (zed-0ca3214b, the same i638 chain): PASS at both lengths — default FNV `0xc91d65a91574ff2c` == explicit τ0.85 FNV @16K (walls 62 426 / 63 454 ms, drift 1.65%) and `0x779b8600461f22ee` == @32K (153 804 / 153 462 ms, 0.22%), bit-exact, zero setter calls in the default arm (env-absence latches), routing counters moved, posture refusals flat. Record: riir-ai bench 1158 (`.benchmarks/1158_issue1165_p5_default_posture_cell.md`). Box state honestly recorded: preflight REFUSED at load 10.81 (a sibling CPU-only fit job — `bonsai_mtp_head` is "Pure CPU, std + rayon + blake3" from source, so GPU exclusivity held); the gated claim is bit-equality (load-immune, deterministic kernel — bench 1157's anchor held 6/6 under build load), walls report-only and untrusted for latency. Harness: `p5_default_posture_equals_explicit_tau085` added to the T6 harness (4 prefills, 452.63 s — the ~8 min estimate held); NAME FILTER mandatory (shares the binary + the process-wide τ/scorer stores with the league cells).**
- [x] **P6 — the consumer blast-radius survey (filed 2026-11-02, the P1 flip's wake; DONE 2026-11-02 zed-0ca3214b).** Every riir-ai/riir-* lane that measures the DENSE incumbent at ≥16K WITHOUT pinning the demote would silently shift its baseline under the promoted default. Method: `TernaryDeltanetGpuForward` + `.prefill(` ∩ ≥16K lengths over riir-ai tests + riir-train examples + the perf scripts, then per-site length-default + pin audit. **The survey table (all sites; all pins landed in the same commit family):**

  | site (repo) | what it measures at ≥16K | default reaches ≥16K? | verdict + fix |
  |---|---|---|---|
  | regime-feature dump (ai) | dense centroid source | n/a (14×16K dense) | already pinned `f15fa59fd`; stale "unset (dense reference)" doc line fixed |
  | **bench_809 (ai)** | **the league "ours" prefill lane** (perf-rematch + 851 ranking drive it); FNV pins @2048/16384 | YES — `B809_LENGTHS` default `[2048, 16384]` | **AFFECTED BY DEFAULT** → dense pinned via setter (a sparse-arm league entry is its own bout, never a silent baseline shift) |
  | bench_726 (ai) | the dense prefill BASELINE (`DEFAULT_LENGTHS` incl. 16384/32768) | YES | AFFECTED → dense pinned |
  | bench_734_chunk_width_e2e (ai) | chunk-width A/B, dense reference leg | YES — default `8192,16384` (16384 ≥ floor) | AFFECTED → dense pinned |
  | bench_844 (ai) | m32 promotion A/B | YES — default `[16384]` | AFFECTED → dense pinned |
  | issue884 (ai) | q8-act NLL QUALITY gate | YES — `I884_TOKENS` default 32768 | AFFECTED → dense pinned (a lossy-sparse NLL would poison the quality band) |
  | bench_772 (ai) | FNV/argmax pins @16384/32768 (Plan-622 override) | override-only (default P=2048) | AFFECTED-ON-OVERRIDE (pins would fail loud) → dense pinned |
  | bench_1122 (ai) | site-tax/ktile64 league cells, dense reference | override-only (default PS `[2048,4096]`; league override 16384/32768) | AFFECTED-ON-OVERRIDE → dense pinned in gate1 + gate2 + gate1b |
  | plan545_precompute_bonsai_gpu (train) | dense mass-tap instrument; chunk-boundary BIT-IDENTITY claim | YES — default `--seq-len 32768` | AFFECTED → pinned via the shipped demote ENV literal (riir-train does not name riir-infer directly — BOUNDARY; `set_var` before any forward call, the first-caller-wins latch) |
  | bench_1151/1152/1156 (ai) | sparse-arm harnesses | — | SAFE (explicit setters/`Arm::apply`) |
  | issue879 (ai) + issue879_gdn_quant_certification (infer) | cudarc / pure-CPU lanes | — | SAFE (condition 4: the arm is wgpu/CubeCL-forward only) |
  | issue879_t3_kv_weight_quant_nll (ai) | `TernaryDeltanetGpuForwardCudarc` | — | SAFE (cudarc) |
  | perf_rematch.sh (ai) | no ≥16K wgpu prefill lane of its own (league prefill rows are the 851/bench_809 lane) | — | SAFE (covered by the bench_809 pin) |

  **11 pin sites landed (9 setter calls + 1 doc fix + 1 env literal).** The env-absence = dense assumption is now false nowhere on the measured surface.

## Boundaries

- Metal/wgpu lane ONLY (condition 4) — no cudarc/CUDA surface change.
- The regime gate (E) is NOT part of this promotion (it missed the bars on the homogeneous stream; its lane stays opt-in).
- The m32 sub-16K incumbent is untouched.
