# Plan 623: Multi-host inference lane — layer-range stages, activation frames, TCP→RDMA transport, TP, speculative fill

**Lease closed:** zed-m3-i623-t32t33, 1793225492, was expires 2026-10-30T22:11:32+00:00 — LANDED the T3.3 MTU ladder + the T3.2 rkey semantics 2026-10-29 (`570ee79`+`5304716`, the landing note on T3.2 carries the detail; the T3.3 payload below on its task row). T3.2's MTLBuffer leg is code-complete (`verbs_metal`) and blocked on an amo-local Application-Firewall gate (owner sudo) — the residue rides the T3.2 task row.
**Lease closed:** zed-m3-i623-t3verbs, 1793220893, was expires 2026-10-30T20:54:53+00:00 — LANDED T3.1 + the T3.2 heap leg + the T3.3 small-message/striped rows 2026-10-29 (Bench 927; the landing note on T3.1 carries the detail). The open residues ride their task rows: T3.2's MTLBuffer leg + rkey semantics, T3.3's MTU ladder, T3.4, T3.5. (claim record: zed-m3-i623-t3verbs, epoch 1793220893, was expires 2026-10-30T20:54:53+00:00)
**Lease closed:** zed-m3-i623-t34gpucomm, 1793255400, was expires 2026-10-31T22:30:00+00:00 — LANDED T3.4 2026-10-29 (unit issue 1151 carries the full landing note: async-finish arm, all-feed-only deferral, G1 exact, s=2 decode window ~5%; open residues ride T3.5).
**Status:** Phase 2 PoC LANDED 2026-10-27 (T2.1 + T2.2 + T2.2b GPU stage + T2.3a pipelined prefill + T2.2c(1) range-scoped weights + T2.3b chunked batched prefill (promoted) — each stage holds only its own slice: `crates/riir-infer-dist`, D1 → (a)) — real 2-box TB5 runs, CPU⇄CPU and GPU⇄GPU, streams bit-identical to 1-box; 1-box GPU base gap filed as Issue 1139 — mostly BUILD POSTURE, fixed by Issue 1141 (dist `gpu` now carries the production decode posture + the workspace's cubecl-wgpu fork: 1-box stage decode 27.2 → 46.7 tok/s, 2-stage 26.0 → 45.2 tok/s on idle bmo; then Issue 1142's dense-pair loads-in-flight kernel, 1-box 51.4 tok/s; then Issue 1143's decode-norm loads-in-flight (`85c4da5`), 1-box 55.0 tok/s; then Issue 1144's radix-4 FWHT (`3147164`), 1-box 56.9 tok/s; residual 1.07× vs llama.cpp, attributed in Issue 1139 T4, next rung Issue 1145); Phase 1 (T1.1–T1.5) LANDED 2026-10-26

**Lease closed:** shikuwa-4090-i922-chattok, epoch 1793193000, was expires 2026-10-29T12:30:00+00:00 — LANDED Bench-917 follow-up 2 2026-10-28 (Bench 924): chat-shaped acceptance re-read, all THREE heads one session — plain head does NOT transfer (0.684→0.085 TF, dead for serving); mix/balanced transfers on TF (0.67/0.44); the FREERUN head (the 655 promotion candidate) is the best chat DRAFTER: E[m] 1.8725, spec 6.207 tok/s = 0.808× of ordinary, beating its own raw row — TF and draft-usefulness DECOUPLE cross-regime, rank by E[m] only; the offline `mtp_chat_tokens` tokenizer example ships; follow-up ladder re-ordered (re-fit = chat-trajectory capture, the 679 recipe; 2-box drafter = freerun head).

**Lease closed:** zed-amo-i623-mtpcons, 1793149690, expires 2026-10-30T01:08:10+00:00 — LANDED the T5.1d consumer half 2026-10-28 (Bench 917): `spec_mtp.rs` (`BmthHead` BMTH v1 reader + `MtpHeadDrafter`) + the runner roll seam (`DraftSource::propose_with_roll`, additive, token-level drafters byte-identical; G1 battery 11/11 incl. two new roll-path tests); in-posture acceptance MEASURED — teacher-forced per-slot 0.6841 at split 63 / 0.0591 at 48 / 0.0332 at the production 32 (fit basis 0.7929): the seat is split-63 ONLY; free-run stream degenerate (raw-text greedy collapse — the caveat and the follow-ups live in Bench 917); snapshot cost ~70× filed as Issue 1148; `--drafter mtp` wired into dist_stage.
(feat(plan-623 T5.1d): the trained MTP-head consumer — spec_mtp + the roll seam, in-posture acceptance MEASURED (Bench 917))
**Lease closed:** zed-amo-i623-t25residue landed the T2.5 residue 2026-10-28 (`a260b62` — the landing note on the T2.5 task carries the detail: measured link bw + tail probe rows; the commit was first made as `f610afc` and re-hashed onto the pushed claim by the manual rebase — same tree, new parent)
**Lease closed:** zed-amo-i623-g2fitsbox measured the G2 fits-one-box row 2026-10-28 (Bench 916; the instrument commit `3b286c8` carries the decode-only oracle window)
  (the two disclosed T2.5 residues, Bench 915 honest-notes: the tail answer-posture probe rows print + the link bw MEASURED on the wire (echo-ack bulk — the tail does zero compute, the v0 conflation concern does not apply); CPU-only, no GPU cells)
**Lease closed:** zed-amo-i623-admit landed the T2.4 ADMISSION SEAM 2026-10-28 (`07540f7`) — `Admission`/`AdmissionSource`/`StageFactory` + `run_head_multi_admit`, one wave engine over both doors, G1 4/4 (`tests/multi_stream_admission.rs`); the landing note on the T2.4 task carries the detail. T5.1's V0 design note (same session) is the scoping, not a held slot — the remaining T2.4 kernel micro-batching rung and the T5.1a–d slices stay unclaimed
**Lease closed:** zed-amo-i623-t21b landed T2.1b 2026-10-27 (the landing note on the task is its provenance) — the GPU cells (T2.3c/T2.4/T2.5/T2.7) stay unclaimed
**Lease closed:** zed-amo-i623-t21b landed T2.5 2026-10-27 (the landing note on the task is its provenance; T2.1b's lease closed the same day, `00a8858`) — T2.3c/T2.4/T2.7 cells stay unclaimed
**Lease closed:** m3-idle-623t15 landed T1.5 2026-10-26 (48 h lease from the claim; the landing below is its provenance) — the amo lane's Phase-0 elephant-capture constraint does not bind the M3, which is where T1.5 executed
**Lease closed:** zed-amo-i623-t24 landed T2.4 code 2026-10-27 (Bench 908 is the record; the landing note on the task carries the split — code DONE, GPU cells open) — the remaining T2.3c/T2.7 cells stay unclaimed
**Lease closed:** zed-amo-i623-exchange landed the T2.5 probe EXCHANGE 2026-10-28 (`22eb298`, wire-verified Bench 915 — the landing note on the T2.5 task carries the detail)
(the T2.5 admission EXCHANGE — the wire probe step: head asks the tail for its probe record at boot + the measured RTT, `plan_split` composes live; CPU-only, no GPU cells)
+ the 10-test G1 battery, all gates green (see the Phase 1 checkboxes). Phase 0 is decision-free (measurement only) and may
start once Issue 1109 T0 (bmo reachable) is done; Phase 1 substrate is decision-free
(riir-infer-core, no deps); Phase 2+ waits on [Proposal 042](../.proposals/042_multi_mac_distributed_inference.md) D1 (crate home); Phase 3+ on D2 (RDMA enabled).
**Lease closed:** zed-amo-i623-t03 claimed T0.3 2026-10-28T03:06+07 — 99 s AFTER zed-amo-i623-t24 pushed the completed T0.3 (`291d6f1`, Bench 911: all three engines × the four cells measured in one bmo-idle session, gap issue 1146 filed). The claimed unit was already discharged; the t03 lane should re-run `claim_lease_check` + fetch on its next turn and pick from the genuinely-open rungs (T2.4 kernel micro-batching, T5.1a–d). No work was duplicated — the claim predates no measurement.
**Lease closed:** zed-amo-i623-t24 landed T5.1a–c 2026-10-28 (`6c608dd`) — the landing note on the T5.1 task carries the detail (wire + snapshot rewind + runner mode, G1 7/7). The remaining T5.1d (a real drafter) is a riir-train lane; T2.4 kernel micro-batching stays with its own lane below.
**Lease closed:** zed-amo-i623-mb landed the T2.4 fused decode-batch v0 2026-10-28 (`03f0fe6`, Bench 912: G1 PASS ×2 builds, G2 +38.8–39.5% ABBA ×2 on idle bmo) — the claim above is that landing's provenance; the same-session continuation then REFUTED the arm on the 2-box wire and demoted it to opt-in ([Bench 913](../.benchmarks/913_plan623_issue1147_wire_refutation.md)); the kernel-level batch-2 rung is CLOSED for the wire (no engagement window under the incumbent cadence), loopback-rig residual only — issue 1147 closed.
**Lease closed:** zed-amo-idle-refine-cont landed Phase 1 (T1.1–T1.4) 2026-10-26 (`48b4b0d`) — the claim above is that landing's provenance.
Scope: Phase 1 substrate only (T1.1–T1.4, zero-GPU under the elephant-recapture posture); T1.5's Metal arm stays compile-only this window.
**Date:** 2026-10-22
**Research:** [Research 015](../.research/015_Multi_Host_Inference_exo_llamacpp_MLX_Distill.md)
**Issue 1109 CLOSED 2026-10-27** (file removed per the noise rule; T0–T4 all DONE — T4 = the owner's `rdma_ctl enable` ×2, verified by [Bench 905](../.benchmarks/905_plan623_phase3_jaccl_probe.md); the bring-up narrative lives in git history + `HISTORY.md`).
**Session note 2026-10-26 (zed-amo):** T0/T1/T3 DONE + T0.2 DONE ([Bench 896](../.benchmarks/896_plan623_t02_link_probe.md) — TB5 bridge 8.5–8.6 GB/s both ways, RTT ~94 µs @4 B, PP-hop ≈ 1.3–1.8% of the decode step, TP-over-TCP refuted arithmetically). RPC transport smoke over the bridge PASSED (handshake + set_tensor green). ⛔ T0.4 PREREQUISITE FOUND: the pinned research clone `19ab5a5` REFUSES the Bonsai GGUF (`invalid ggml type 142` — PQ2_0); the Bonsai-capable incumbent is the league fork `prismml-llama.cpp` @ `prism-b10787-73bfe6c` (`../riir-ai/.raw/prismml-llama.cpp`, PQ2_0 in its loader) — T0.4 starts by building THAT with `-DGGML_RPC=ON` on both boxes. **Session note 2, 2026-10-26 late (zed-amo): T0.4's llama.cpp-RPC arm EXECUTED — [Bench 900](../.benchmarks/900_plan623_t04_llamacpp_rpc_2box.md): solo amo 1012.9/59.1, bmo 1087.5/60.9 (pp512/tg128 t/s); 2-box layer-split 985.1/48.6 = tg −20% vs bmo solo (PP-over-TCP LOSES single-stream decode on identical boxes); `-ts` doesn't reach RPC; `-sm tensor` over RPC crashes (hadamard-sign view nil-buffer). T0.2 tailnet arm DONE — [Bench 901](../.benchmarks/901_plan623_t02_tailnet_arm.md): 0.025–0.028 GB/s, RTT ~9.4 ms = control-plane only. T0.4 remains open ONLY for the MLX/exo arms (weight-format convert caveat); T0.3 awaits a quiet-GPU window (sibling agent load 4–7 today — amo solo tg already carries ~3% load tax).**
**Target:** `src/` (LayerRange + ActivationFrame + state serializer), `crates/riir-infer-gpu/` (range-forward on the Metal/CubeCL paths), `crates/riir-infer-dist/` (D1-a; transport + stage runtime + planner). Cargo feature `multi_host` (opt-in, never default) on core; the dist crate is opt-in as a whole.
**Hardware:** amo (M5 Ultra, 96 GiB, macOS 27.0.1) ⇄ bmo (`Mac17,15`) — one TB5 cable @ 80 Gb/s + tailnet (amo `100.111.186.76`, bmo `100.100.117.44`).

---

## Goal

Run one model across amo + bmo from pure Rust, with output byte-identical to the
1-box riir-infer path (PP) or replay-deterministic (TP), and beat exo / llama.cpp RPC /
mlx-lm on the Proposal-042 scoreboard **measured on the same two boxes**.

## Phase 0 — Measure the incumbents and our own 1-box arm (no code in riir-infer)

- [-] **T0.1** Build/install the incumbents from the pinned `.raw/` commits on BOTH boxes
  (llama.cpp `rpc-server` + `llama-cli`/`llama-bench` with `-DGGML_RPC=ON`; mlx + mlx-lm;
  exo) — record exact commits and build flags. No Python dep enters riir-infer: the
  harness lives in `scripts/` as shell + a Rust bench driver. **DEFERRED 2026-10-26 —
  blocked on 1109: bmo is POWERED OFF (tailnet offline 9h, all TB ports no-device;
  adjudication in-issue) — both-box work needs bmo back on the bus first.**
  **UPDATE 2026-10-26 18:20 (bmo back): amo HALF DONE for llama.cpp — `.raw/llama.cpp`
  @ pinned `19ab5a5`, cmake Release `-DGGML_RPC=ON` → `build-rpc/` (`ggml-rpc-server`,
  `llama-cli`, `llama-bench`, `test-rpc-multi-server` all present, GGML_RPC=ON in
  CMakeCache). REMAINING: llama.cpp on bmo (no cmake/brew there yet — T3 toolchain
  first), mlx + mlx-lm + exo on both.**
- [x] **T0.2** Link microbench: TCP RTT + one-way latency for 4 B / 16 KB / 1 MB / 64 MB
  frames over (a) TB bridge, (b) tailnet; bandwidth for 64 MB. Rust std-only
  (`examples/link_probe.rs`). After D2: same over RDMA via `ibv_*` / MLX `allreduce_bench`.
  **DONE 2026-10-26 — bridge arm PASS, both directions: RTT median 94–96 µs @4 B,
  72–83 µs @16 KiB, ~330 µs @1 MiB, ~16.1 ms @64 MiB; 64 MiB push 8.617 GB/s (amo→bmo)
  / 8.541 GB/s (bmo→amo), all rounds tight. Record + derivation (PP hop ≈ 0.09–0.10 ms
  ≈ 1.3–1.8% of the decode step; TP-over-TCP REFUTED arithmetically — Phase 4 stays
  RDMA-gated): [Bench 896](../.benchmarks/896_plan623_t02_link_probe.md). Tailnet arm
  DEFERRED (bmo offline); RDMA arm behind D2.**
- [x] **T0.3** 1-box arms for the D3 models: riir-infer Metal vs MLX vs llama.cpp Metal —
  decode tok/s, TTFT @4k/32k. **If riir-infer's 1-box arm loses, file the gap as its own
  issue first — the multi-host claim is meaningless on a losing single-box base.**
  **DONE 2026-10-28 — [Bench 911](../.benchmarks/911_plan623_t03_arms.md): one bmo-idle
  session (four selective reruns of one harness, `scripts/plan623_t03_session.sh`),
  all three engines × {tg128, pp512, pp4096, pp32768}. Decode competitive: 56.65
  pipeline / 52.64 oracle vs llama.cpp 60.6 (0.93×/0.87×) and MLX 57.9 (0.98×);
  G1 PASS (oracle == stream, token-exact). Prefill LOSES 6–8× at every width
  (140/167/96 t/s vs 1087/1046/772) → the gap clause FIRED:
  `.issues/1146_prefill_throughput_gap_1box.md` (per-frame floor ≈ 6 ms/token,
  decode-shaped kernels at prefill shapes — fix directions + GOAT acceptance in-issue).
  Only-we-complete-32k: MLX REFUSED (Metal 62.6 GB buffer cap, the 904 refusal
  reproduced), riir-infer completes at 96 t/s. MLX pp512 supersession: 1058.1 here
  vs 904's 357.3 (904 carried first-call compile inside a 0.48 s cell; tg agrees
  57.9 vs 58.1) — 911 is the correction of record. Session traps pinned in the
  harness: nc -z eats the tail's single accept; --ctx must reach BOTH stages;
  bash 3.2 comma-splitting.**
  **Row inventory 2026-10-28 (zed-amo-i623-t24 — what exists, what the closing lane
  still owes):** decode tg128 rows exist per engine but NOT under one posture: llama.cpp
  solo bmo 60.9 (Bench 900), MLX 1-box bmo 58.1 + pp512 357.3 + pp4096 1005.3 +
  pp32768 REFUSED-by-Metal (Bench 904), riir-infer idle-bmo stage-path 56.9 (Issue-1144
  posture) + the league cells (amo, 0.978× llama.cpp at the league pin). ~~Owed: one
  bmo-idle session~~ DISCHARGED by Bench 911 (above).
  **⛔ MODEL SCOPE (owner, 2026-10-26): gemma-2-2b is RETIRED (riir-train Issue 617) — the
  1-box cells run on Bonsai-2-27B-PQ2_0 + qwen3.8-27B ONLY; never add a gemma-2-2b cell.**
- [-] **T0.4** 2-box incumbent arms: llama.cpp RPC (TCP; RDMA after D2), mlx
  (pack runtime; socket PP — ring p2p deadlocked, see Bench 904) + TP (ring
  collectives measured; jaccl after D2), exo. Every metric
  of the scoreboard; `.benchmarks/` record with both boxes' provenance.
  **exo arm N/A on the D3 models (scoped 2026-10-27, no bench): exo @ bc412e2
  has no prism_hadamard_qwen35 loader — its mlx engine auto_parallel.py
  patches the STOCK mlx_lm qwen3_5 classes only; loading the Bonsai-2 2-bit
  pack (custom model_type + Packed hadamard modules) is impossible without a
  model conversion that changes the measured artifact (same class as stock
  mlx-lm, whose load() the pack doc names unsupported). Re-open = exo gains
  a pack loader or a sanctioned stock-arch Bonsai conversion exists.** **llama.cpp-RPC
  arm DONE 2026-10-26 — [Bench 900](../.benchmarks/900_plan623_t04_llamacpp_rpc_2box.md):
  prism-b10787 built `-DGGML_RPC=ON -DGGML_BLAS=OFF` on amo, shipped as a binary bundle
  to bmo (NO CLT/cmake on bmo — rpath+openssl surgery on amo, ship, run); solo rows amo
  1012.9/59.1, bmo 1087.5/60.9 (pp512/tg128 t/s); **2-box layer-split 985.1/48.6 = tg
  −20% vs bmo solo** (per-step RPC round trips + pipeline sequencing — TB5/TCP PP LOSES
  single-stream decode on identical boxes; prefill −2.7% ≈ free); `-ts` does NOT reach
  the RPC device in this fork (both ts rows stayed amo-MTL, server-side proven);
  `-sm tensor` over RPC crashes the server (hadamard-sign view → nil buffer assert) —
  TP-over-RPC N/A in the incumbent. REMAINING: exo arm (scoped N/A on the D3

  models — see the T0.4 note below), T0.3 quiet-GPU cells, RDMA rows behind D2.**

  **MLX ARMS DONE 2026-10-27 — [Bench 904](../.benchmarks/904_plan623_t04_mlx_arms.md):
  1-box bmo 357.3/58.1 (pp512/tg128; decode PARITY with llama.cpp solo 60.9 =
  0.95×, prefill 0.33×); 2-box PP 43.0 t/s = −26% vs 1-box (socket exchange,
  32/32 split; role0 median agrees within noise); **mx.distributed ring p2p
  (send/recv) DEADLOCKED on 2 nodes in mlx 0.32.0** (zero bytes cross the wire;
  collectives fine — all_sum 64 MiB 3.7 GB/s vs 8.6 raw, floor 320 µs) —
  mlx-lm pipeline-over-ring inherits the gap. BOTH incumbents now measured
  LOSING single-stream decode on 2-box TB5/TCP PP. Pack re-downloaded post
  disk-clean via direct curl + files.json sha256 24/24; loader gap (pack
  schema v2 vs bundled v1 reader + language_model prefix + vision_tower
  weights) fixed OUTSIDE the pack in scripts/plan623_pack_artifact.py.
  amo 1-box + clean-window re-read: deferred to the T0.3 quiet window.
  **DISCHARGED 2026-10-27 ([Bench 906](../.benchmarks/906_plan623_t41_tp2_cell.md)
  §6): amo MLX 1-box pp 335.2 / tg 50.8 t/s (amo is the slower box; bmo 58.1
  stays the reference); clean-window socket-PP re-read 45.2 t/s — the −22..−26%
  PP loss HOLDS quiet (not load).** ⚠ **MLX-arm prerequisites found
  RECLAIMED on amo (2026-10-26 late scan): the HF-cache pack
  `models--prism-ml--Ternary-Bonsai-2-27B-mlx-2bit` carries only 556 KB of blobs — the
  big safetensors blob is GONE (re-downloadable from `prism-ml/Ternary-Bonsai-2-27B-mlx-2bit`,
  snapshot `fcba37d21`, integrity via the pack's own files.json 24/24 sha256 gate); the
  `.raw/mlx-env` venv's python binary is GONE too. AND bmo has NO working python3
  (`/usr/bin/python3` is the CLT stub — prints the xcode-select note) and no uv. The
  bmo-side MLX arm therefore needs: uv bootstrap on bmo (static binary, no CLT) →
  uv-managed CPython + venv (build on amo, rsync to the SAME absolute path — pyvenv.cfg
  is path-keyed) → `uv pip install mlx mlx-vlm tokenizers` → pack re-download (~10–15 GB).
  The 2-box PP arm then rides `mx.distributed` (jaccl/grpc ranks over SSH) — a dedicated
  bench window, priced here so it starts warm.**

## Phase 1 — Substrate in riir-infer-core (feature `multi_host`, zero new deps)

- [x] **T1.1** `LayerRange { lo, hi }` + range-forward on the CPU reference paths
  (`deltanet/ternary_forward.rs` `qwen_deltanet_ternary_layer_body` loop; gemma2
  `forward_gemma2_layers`). Embedding only on the stage with `lo == 0`, final norm +
  lm_head only on the stage with `hi == n_layer`. **G1:** stage(0..k) ∘ stage(k..n) ==
  full forward, bit-identical, every k, both archs. **LANDED 2026-10-26** (amo,
  zed-amo-idle-refine-cont): `LayerRange` + the gated stage entry points
  (`forward_qwen_deltanet_ternary_range`, `forward_gemma2_range`) over an UNGATED
  internal `*_body_range(lo, hi)` — the full forward DELEGATES to it, one loop, no
  drift surface. G1 PASS: every k on the 4-layer synth Bonsai-2 (decode loop, logits
  bit-exact) + through the ActivationFrame wire + gemma2 every k (softcapped, RoPE,
  GQA). **Design note:** the range compute is deliberately ungated (same compiled
  code as the full forward; the feature gates the LANE — wire types + public stage
  seams), matching the Issue-1117 extraction precedent.
- [x] **T1.2** `ActivationFrame` (`seq_id u32 | pos u32 | n_tok u16 | dtype u8 | len u32 |
  payload`), little-endian, bytemuck-castable payload; bf16 and f32 dtypes; decode-only
  fast path with no allocation (G4 alloc-free). bf16 on the wire is a LOSSY surface —
  f32 is the G1 posture, bf16 rides its own per-family retention gate (lossy-surface law).
  **LANDED 2026-10-26**: 4-byte `MAF1` magic + 16-byte header (20-byte wire prefix),
  field-wise LE encode (never a host-endian struct cast), RNE bf16 (`f32_to_bf16_rne` —
  NaN-preserving), decode_into reuses the caller buffer (capacity-stable assert in the
  battery). bf16 gate = idempotence THROUGH its own rounding (decode(encode(v)) ==
  decode(encode(decode(encode(v))))), never bit-identity; bf16-exact values survive
  exactly. Reserved byte refuses non-zero (fail-closed forward compat).
- [x] **T1.3** Per-range state serializer for `HybridCache` (KV rows + GDN state for the
  range's layers) — closes Issue 028's out-of-process gap; round-trip bit-identity test;
  wire size matches the `src/disaggregated.rs:44-52` budget. **LANDED 2026-10-26**
  (`HybridStateSegment`, `MHS1` v1): per-layer kind-tagged records (attention =
  key/value rows with per-layer lengths — sliding-ring layers carry their SHORTER
  physical buffers, the Issue-683 clamp — deltanet = full recurrent matrix + conv
  window), header carries `fill_pos` restored via `set_fill_pos` LAST (a refused
  restore never arms the tracker). G1: encode → FRESH cache → restore mid-run →
  continued decode logits bit-identical to the uninterrupted run.
- [x] **T1.4** Stage handshake record: arch, `geometry_fingerprint`, artifact BLAKE3,
  `LayerRange`, dtype — refuse on any mismatch (never a silent wrong-shape stage).
  **LANDED 2026-10-26** (`StageHandshake::verify_pair`): downstream verifies upstream —
  arch/geometry/artifact/dtype equality + RANGE TILING (upstream.hi == downstream.lo)
  + both ranges validate against n_layer. Negative battery covers every mismatch class
  (arch, geometry, artifact, dtype, gap, out-of-bounds).
- [x] **T1.4b (validation battery)** `tests/plan623_multi_host_g1.rs` — 10 tests, all
  green at the gated posture; clippy 0 at default / gated / no-default postures; lib
  347 passed (gated) / 269 (default); rustfmt clean on every touched file. Found
  (pre-existing, NOT this plan's): the issue028 `dual_format_q4_prefill_pair_loads_and_runs`
  gate red at pristine HEAD — filed as [Issue 1123](../HISTORY.md) (closed 2026-10-26, record in
  HISTORY.md: the TEST was stale — the TernaryQ2 promotion outran the battery; decode gate
  strengthened to all-sites ternary-arm, `e56dc0f`).
- [x] **T1.5** Range-forward on Metal/CubeCL: `TernaryDeltanetGpuForward::forward_from_x`
  / `mark_x_as_input` (the existing upload-x seam) and gemma2 `forward_layer_gpu` loop with
  `[lo, hi)`; G1 bit-identity vs the full GPU forward.
  **LANDED 2026-10-26** (m3, Metal/wgpu-msl, real models): the qwen decode funnel refactored to
  an UNGATED `forward_from_x_body_range(.., lo, hi, advance_pos)` (the full funnel delegates at
  `(0, n_layers, true)` — no second copy of the loop, the Issue-022 T3.1 structural law); the
  boundary law mirrors the CPU range contract — a stage's FIRST layer takes the standalone
  input RMSNorm (`layer_idx == lo` replaces `== 0`) because the producing stage applies its
  deferred FFN residual explicitly with the SAME `ResidualAddCubeCL` dispatch the full funnel
  runs after the loop (bit-exact by construction: the fused `residual_add_rmsnorm_f32`
  Phase-1 add IS `a[i] + b[i]`, one rounding, and the RMSNorm reduction tree is the same
  strided-256 walk in both kernels); final norm + folded lm_head gated on `hi == n_layers`;
  `advance_pos` false on stage passes (the stage runtime schedules position via the new
  `set_pos` seam — the CPU range's `pos` parameter twin). Seams (`multi_host`-gated):
  `forward_range_dispatch(LayerRange)` + `set_hidden_state(&[f32])` (fresh-mark protocol,
  width-refusing) + `read_hidden_state()` + `set_pos(usize)`. Gemma2: `forward_gpu_range`
  (per-layer bodies are self-contained → bit-identity structural; embed/head gated on
  lo/hi; delta_routing refuses loud — the hybrid CPU-side forward has no range form).
  **G1 EXECUTED AND GREEN — bit-exact, the expected class:** qwen (Bonsai-2 27B PQ2_0,
  block_size clamped to the decode bound per the Issue-994 VRAM gate) ALL 63 splits × 4
  tokens (gemma2 compile-only per the owner's model-priority law — see below);
  single-instance design (reference loop → `reset_state` → staged replay; state slots are
  absolute + ranges disjoint). Independent regression: `g1_032_folded_prefill_metal` green
  post-refactor (P=64: argmax 702/702, top-5 1.00, worst rel err 4.08e-6 — the decode arm
  runs the refactored funnel against the INDEPENDENT prefill path); CPU G1 battery 10/10;
  clippy clean lib+tests at the gated posture; default / no-default / all-features /
  wasm32 compiles clean. Gate: `tests/plan623_multi_host_g1_metal.rs`
  (`I623_T` / `I623_K` env overrides; `--ignored --test-threads=1 --release`).
  **⛔ OWNER RESCISSION (2026-10-26, same-day): the EXECUTED gemma2 G1 cell (gemma-2-2b f16,
  25 splits × 4 tokens, all bit-identical at the time it ran) was RESCINDED by the owner —
  gemma-2-2b is the RETIRED model class (riir-train Issue 617) and the workspace pivots on
  Bonsai/ternary. The gemma2 `forward_gpu_range` SEAM stays as compile-only scaffolding
  (it mirrors the CPU T1.1 `forward_gemma2_range` the amo lane landed; deleting it would
  desync the lane's own symmetry — its per-layer bodies are self-contained, so its range
  bit-identity is structural and needs no executed cell). The gemma arm was REMOVED from
  the G1 test; no new gemma-2-2b execution cells anywhere — not in this plan, not in any
  sibling.**

## Phase 2 — Stage runtime over TCP (D1 DECIDED 2026-10-27 → (a))

- [x] **T2.1** `crates/riir-infer-dist`: `Transport` trait (`send_frame`, `recv_frame_into`)
  + std TCP impl (`TCP_NODELAY`, `SO_SNDBUF/RCVBUF` sized for prefill chunks, vectored
  I/O, preallocated rings, bind to a named interface only). Peer authentication: pinned
  peer key (BLAKE3-keyed MAC over the handshake + frame counter) — not an RPC surface.
  **LANDED 2026-10-27 (PoC scope):** `transport.rs` — every frame MAC'd over
  `direction | kind | counter | payload` (anti-reflection + strict per-direction counter:
  replay/reorder/drop refuse; MAC checked BEFORE any header byte is interpreted;
  `MAX_BODY` cap), `TCP_NODELAY`, vectored writes, caller-reused recv buffer. Handshake
  wire form `handshake.rs` (LE by field) over the core `StageHandshake::verify_pair`;
  identity digests reuse `disaggregated::geometry_fingerprint` (no second fingerprint).
  - [x] T2.1b remaining knobs — **LANDED 2026-10-27** (zed-amo-i623-t21b):
    `dist/src/sockopts.rs` — `dial`/`listen`, the std-equivalent socket
    setup with `SO_SNDBUF`/`SO_RCVBUF` sizing (read back via `getsockopt`,
    the kernel's truth returned in `Applied`) and named-interface bind
    (macOS `IP_BOUND_IF`/`IPV6_BOUND_IF` via `if_nametoindex`, Linux
    `SO_BINDTODEVICE` — applied BEFORE bind/connect so the route is
    actually pinned; unknown interface refuses `NotFound`; non-unix
    targets fail closed on any knob). No new crates.io dep: the ~9 libc
    symbols are `unsafe extern "C"` declarations (edition-2024).
    CLI: `dist_stage --sock-buf BYTES` / `--iface NAME` (both roles; the
    applied values print; hostname resolution kept — `ToSocketAddrs`,
    std-equivalent). Frame buffers: the tail reserves its receive buffer
    + hidden staging to `frame_bound(chunk, hidden)` right after `Begin`
    — no mid-stream multi-MB realloc on the first chunked frame (the
    head's send path was already `with_capacity`-sized). G1: 4 new unit
    tests (buffer read-backs ≥ 512 KiB both sides + frames carry through
    the MAC'd transport; `lo0`-bound listener+dial round-trip on macOS —
    ifindex 1 read back both sides; bogus iface refuses both entry
    points; default options == std posture) + the CPU two-stage battery
    still 8/8 bit-identical. Clippy `-D warnings` at default +
    `bonsai2_hadamard` postures; rustfmt clean. Honest residue: the
    `gpu`-feature compile of the bin is cfg-neutral (edited lines ungated;
    the next 2-box cell rebuilds that posture anyway) and the G2 benefit
    of `--sock-buf` on 2-box prefill is UNMEASURED — that cell belongs to
    T2.7/T2.3c's measurement lane.
- [x] **T2.2** Two-stage runner: stage 0 = embed + layers [0,k); stage 1 = layers [k,n) +
  norm + lm_head + sampler → 4-byte token id back to stage 0. PUSH, no hub, one one-way
  hop per boundary per token. **G1:** 2-box greedy output byte-identical to 1-box over
  the Issue-028 battery prompts.
  **LANDED 2026-10-27 (PoC):** `runner.rs` (`run_head` / `run_tail` / `run_local` oracle)
  over the `StageCompute` trait (CPU stage `TernaryCpuStage` today; the T1.5 GPU seams
  plug into the same trait); bin `dist_stage`. G1 battery `crates/riir-infer-dist/tests/
  two_stage_g1.rs` 4/4 over REAL loopback TCP: token stream + every tail logit
  BIT-IDENTICAL to 1-box at every split k (synth Bonsai-2), wrong key refuses (MAC),
  foreign artifact refuses (handshake). **Real 2-box smoke (amo head [0,32) ⇄ bmo tail
  [32,64), TB5 bridge 192.168.10.x, Bonsai-2-27B PQ2_0, CPU both sides — no GPU, so no
  quiet-window wait):** handshake PASS (bmo's model file is byte-identical), 32-token
  greedy stream **byte-identical to amo's local 1-box run (`--full-check` G1 PASS)**,
  decode 16.3 tok/s, prefill 4 tok 0.28 s, load 6 s per box. SMOKE, not a G2 cell: CPU
  stages, token-by-token prefill, one shared-box run (amo load ~6, GPU held by the
  elephant capture), no 1-box CPU baseline timed. bmo binary = amo `--release
  --features bonsai2_hadamard` build shipped by scp (links only libc++/libiconv/
  libSystem — no CLT needed on bmo).
  - [x] T2.2b GPU stage `TernaryGpuStage` (`src/stage_gpu.rs`, feature `gpu`) over the
    T1.5 seams (`set_pos` + `set_input_token`/`set_hidden_state` →
    `forward_range_dispatch` → `read_logits`/`read_hidden_state`); `dist_stage --backend
    cpu|gpu --ctx N` (one generic role loop over `Box<dyn StageCompute>`; runner refuses
    `pos >= max_positions` on both sides). **LANDED 2026-10-27.** G1 gate
    `tests/two_stage_g1_gpu.rs` (`#[ignore]`, real model) executed on **bmo's idle GPU**
    (test binary shipped — no toolchain there): GPU⇄GPU over loopback TCP **bit-identical
    to the 1-box GPU forward at k = 1, 32, 63** (19 positions each), and the GPU token
    stream equals the CPU stream. **Real 2-box GPU run** (amo GPU head [0,32) ⇄ bmo GPU
    tail [32,64), TB5): `--full-check` **G1 PASS**, 32 tokens, 20.3 tok/s (amo's GPU
    was contended by the elephant capture — indicative only). Idle-bmo overhead read:
    2-stage 26.0 vs 1-box stage path 27.2 tok/s = **~4% pipeline cost** — but the 1-box
    base is ~0.45× llama.cpp's 60.9 on the same box → **`Issue 1139`**
    (the T0.3 rule: file the losing-base gap first).
  - [x] T2.2c(1) **range-scoped weights — LANDED 2026-10-27.** `WeightScope` (core
    `deltanet::ternary_weights`) + `load_qwen_deltanet_ternary_weights_gguf_scoped`: a
    stage reads ONLY its layer window from the mmap (+ `wte` iff first, `final_norm` +
    `lm_head` iff last; a tied `lm_head` still reads `token_embd`); out-of-scope layers
    are `DeltaNetTernaryLayerWeights::empty()`. The GPU forwarder needs no new
    constructor — empty layers upload as the existing 0-row dummies, and per-layer
    GDN/conv/KV state is now allocated for RESIDENT layers only (a dispatch reaching a
    skipped layer `expect`s on `None` — loud); `vram_budget` charges resident layers
    only. Guard: `QwenDeltaNetTernaryWeights::covers(&scope)` → `require_coverage` in
    both stage constructors (a slice handed to the wrong window refuses). Side fix:
    `invariants_hold` precedence — a 0-row `wte` used to short-circuit the per-layer
    check. **G1:** CPU battery 5/5 incl. the new scoped test (every split, head/tail
    slices bit-identical, wrong-window refusal, non-vacuity asserts); GPU G1 on **bmo's
    idle GPU, real Bonsai-2 27B, scoped per stage: bit-identical ×19 positions at
    k = 1, 32, 63**; working set (`forward_working_set_bytes`, ctx 20) **full 7.68 GiB →
    head 3.84 + tail 3.84 GiB at k=32** (0.46 + 7.22 at k=1). **Real 2-box GPU via
    `dist_stage` (amo head [0,32) ⇄ bmo tail [32,64), TB5, scoped): `--full-check` G1
    PASS**, 32 tokens, load 5.1 / 4.5 s per box, decode 18.2 tok/s (amo load ~7 —
    indicative only). The lane now buys capacity: a model up to ~2× one box's budget
    splits across two.
  - [-] T2.2c(2)+(3) stage per-token cost — **DECLINED 2026-10-27 on measurement**
    (`Issue 1139` T1, idle bmo,
    3 runs): the whole logits readback + host argmax is **0.92–0.94% of the token**
    (0.23–0.28 + 0.11 ms of 36.7–43.2 ms) — the ceiling for an on-GPU argmax before its
    own kernel cost, under the 1.5% G2 bar; the 20 KB `set_hidden_state` alloc is
    smaller still. Note for any successor: the shipped `ArgmaxCubeCL` tie rule is NOT
    first-index (strided lanes: a tie between index 1 and index 256 returns 256), so a
    re-open needs a tie-correct kernel to hold the Issue-940 law + the G1 bit bar. The
    stage's per-token lever is the 1-box kernel stack (Issue 1139 T2), not the seam.
  - [x] T2.2d **tokenizer front — LANDED 2026-10-27.** `dist_stage --text "..."` encodes
    with the GGUF's own `BpeTokenizer` (head only; the tail never tokenizes), `--chat`
    wraps the Qwen chat template, the head prints ids + decoded text and stops at the
    vocab EOS (`HeadOptions::eos`, kept in the output; `--no-eos` restores the full
    `--max-new`); the oracle (`run_local_with`) stops at the same id. (The Issue-028
    battery is itself token-id based — `[7, 13, 42, 100, 5]` — so the id path stays the
    G1 battery; the text front is the usability half.) CPU G1 8/8 incl. the EOS-stop
    test (pair + oracle cut at the same token, tail serves no position past it). **Real
    2-box GPU (amo head ⇄ bmo tail, TB5, scoped):** `--chat --text "What is the capital
    of France? Answer in one sentence."` → 20 prompt tokens, `"<think>\nThe user is
    asking a simple factual question … The answer is Paris.\n</think>\n\nThe capital of
    France is Paris.<|im_end|>"`, stopped at EOS after 42 tokens, **`--full-check` G1
    PASS**, decode 17.3 tok/s (amo shared — indicative).
- [x] **T2.3** Chunked pipelined prefill (chunk c on stage 1 while chunk c+1 on stage 0,
  double-buffered frames) + KV/GDN state lives on its own stage (no state traffic in
  steady state). (T2.3a + T2.3b landed 2026-10-27; T2.3c closed at the demotion
  2026-10-28 — the parent is complete.)
  - [x] T2.3a **per-position pipelined prefill — LANDED 2026-10-27**: new `FrameKind::Prefill`
    (activation, NO token reply) for prompt positions `0..len-1`; only the last prompt
    position and decode steps round-trip. The head computes pos t+1 while the tail
    computes pos t, and a long prompt can no longer back up token replies into a
    send/send TCP deadlock. CPU G1 battery still 4/4 bit-identical (every split, every
    position's logits). **2-box A/B (CPU stages, amo ⇄ bmo TB5, 64-token prompt,
    interleaved old/new/old/new, amo load ~5): prefill 3.989 / 3.967 s → 1.993 / 2.036 s
    = 2.0×** (the two stages now overlap fully); generated stream identical; decode
    unchanged (16.0–16.2 tok/s). "old" = the `d5f5936` crate built standalone.
  - [x] T2.3b **chunked multi-token frames — LANDED + PROMOTED 2026-10-27.** GPU seam
    `TernaryDeltanetGpuForward::prefill_range_chunk` (`multi_host` +
    `ternary_gemm_batched`): the 1-box chunk body `prefill_tokens_chunk` gained a
    private `StageChunk { lo, hi, hidden_in }` — layer loop `lo..hi`, embedding only
    when `lo == 0` (else the upstream rows upload into `x_b`), a non-last stage returns
    every row's hidden instead of the tail; the cudarc whole-prefill arm is skipped for
    a stage; every pre-623 caller passes `None` (byte-identical). Stage seam
    `StageCompute::prefill_chunk` (default = the per-position `step` loop; the GPU stage
    overrides it). Wire: new `Begin` frame (`prompt_len` + effective width — the
    attention-arm gates key on the WHOLE prompt length, Issue 782, so the tail must know
    it before chunk 0); prompt chunks ride `Prefill`/`Activation` frames with
    `n_tok > 1`. Width policy `runner::prefill_width`: chunk EVENED over the prompt
    (130 @ 128 → 65+65) and per-position under `min_batched = 64` — the measured Metal
    cost is ~0.9 s/chunk floor + 0.018 s/token vs 0.039 s/token per position
    (break-even ≈ 43). **G1:** CPU battery 7/7 (chunk 2/3/7 × every split, the
    `prefill_width` pins); GPU `tests/two_stage_prefill_gpu.rs` on idle bmo, real
    Bonsai-2 27B, scoped stages: **anchor — the full-range stage's chunked prefill is
    bit-identical to the PRODUCTION `prefill()` at the same chunk cap**; the 2-stage
    chunked stream + last-prompt + every decode logit bit-identical at k = 1, 32, 63
    (P = 64/16, 256/64, 256/256, 200/100). **G2:** 1-box P=256: chunk 16 (P=64)
    **0.60×** (loses), 64 → **1.34×**, 256 → **1.78–1.83×**; P=200/100 → 1.58×. **Real
    2-box (amo head ⇄ bmo tail, TB5, scoped GPU, P=256, ABBA, amo load ~3.8):
    per-position 6.83/6.95 s → chunk 64 5.73/5.49 · 128 5.34/5.30 (**1.30×**) · 256
    6.73/6.47 s**; `--full-check` G1 PASS on the chunked run. (Less than same-box 2.0×
    because T2.3a already overlaps the stages per position — chunk 128 balances batch
    efficiency vs overlap.) **Promoted:** the dist `gpu` feature now carries
    `ternary_gemm_batched` (the opt-in `gpu_prefill` it shipped behind is folded);
    `dist_stage --prefill-chunk` defaults to 128 on `--backend gpu`, 1 on cpu (chunking
    only forfeits the CPU path's per-position overlap). Side observation
    filed + CLOSED the same day as **Issue 1140** (removed per the noise rule; this is
    its record): decode read 4–6% lower after a batched prefill in the T2.3b cells
    (order-confounded, batched arm always first) and 10–24% in the 2-box cells (amo
    contended). The interleaved A/B (`decode_readback_probe_gpu.rs ::
    decode_after_batched_vs_per_token_prefill_interleaved`: fresh forwarder per arm,
    alternating order, P=256, 128 timed steps, 4 pairs, idle bmo on AC, load 1.6–3.1)
    reads **−1.35% median (25.52 → 25.17 tok/s), batched lower in 4/4 pairs** — real in
    sign, UNDER the 1.5% G2 bar, so attribution (pool scratch / prefill-left state) is
    not worth a rung; the production `prefill()` carries the same ~1% tax.
- [x] **T2.3c** Re-tune `runner::prefill_width`'s `min_batched = 64` break-even under the
  Issue-1141 posture: the per-position cost dropped more than the chunk floor (bmo, P=64:
  chunk 16 now **1.44×** 1-box / 1.16× split 32, was 0.60×) — re-measure the cost curve
  (per-position s/token vs chunk floor + s/token) and re-derive the threshold + the
  default 128 width; 2-box ABBA confirms.
  **DONE 2026-10-28 — CLOSED AT THE DEMOTION** (zed-amo-i623-t24,
  [Bench 909](../.benchmarks/909_plan623_t23c_prefill_width_curve.md)). Single-box grid
  (`prefill_width_curve_grid`, amo Metal, real 27B, GPU-exclusive, best-of-2 after
  warm-up): per-position **17.8 ms/tok** (was 39), chunk floor ~0.49 s @ w=16 — chunking
  wins at EVERY measured P with the right width (1.17× @ P=32 w=16 up to 2.35× @ 128
  single-chunk); default width 128 optimal everywhere. A 48 floor landed on that grid
  alone — then the **2-box ABBA (the task's acceptance cell) REFUTED it for the
  pipeline**: amo⇄bmo TB5 split-32 GPU both, P=64 per-position WINS ~8% (T2.3a's
  per-position overlap beats a whole-prompt chunk's serialisation), P=256 chunk-128 wins
  **1.47×** (T2.3b's 1.30× improved by the 1141+ stack). **`min_batched` stays 64**
  (revert landed same session, never a consumer posture in between); both curves in the
  runner doc comment; pins carry the 64 floor + the evening law; full dist suite green
  at every step. The single-box 48 optimum is recorded for future 1-box stage consumers.
  Residue: the crossover is bracketed in (64, 256), not pinned (P ∈ {96,128,192} cells
  if anyone ever needs the exact bound).
- [x] **T2.4** Micro-batch round-robin for ≥2 streams (fills the PP bubble); continuous
  batching admission; chunked prefill interleaved with decode (exo's prefill-blocks-decode
  gap). **FUSED DECODE-BATCH v0 LANDED 2026-10-28** (zed-amo-i623-mb, issue 1147,
  `03f0fe6`, [Bench 912](../.benchmarks/912_plan623_issue1147_fused_decode_batch.md)):
  the `BatchN {count}` hint + the `StageCompute::step_submit`/`step_finish` seam —
  the tail submits every ready stream's GPU chain BEFORE any readback (the shared
  queue runs them back to back, no inter-chain idle); the head hints only when every
  ready stream is a decode feed and drains its whole pass; `RIIR_DIST_PAIR=0` demotes
  byte-identically. G1 PASS ×2 builds (per-stream tokens bit-identical to the 1-box
  GPU oracle through the fused batch; CPU: paired == incumbent captures logit-for-logit,
  4/4 + 41 green). **G2: +38.8–39.5% aggregate decode window on bmo loopback s=2
  ABBA ×2 builds (78.2 vs 56.2 tok/s), = 1.68× the 1-box oracle (incumbent 1.20×).**
  dispatch-level (launches not reduced) — the kernel-level batch-2 rung
  (grid-dim batch, per-row reduction order pinned) stays OPEN with this +39% as its
  incumbent bar; the 2-box wire confirmation cell waits on a quiet amo window.
  Remaining sub-rungs: kernel micro-batching (above), the continuous-batching
  admission EXCHANGE (wire probe step, the planner's T2.5 residue).
  **⚠ WIRE REFUTATION 2026-10-28 (zed-amo-i623-mb continuation,
  [Bench 913](../.benchmarks/913_plan623_issue1147_wire_refutation.md)): the +39%
  does NOT transfer — on the real amo⇄bmo TB5 wire the armed arm reads 46.9–52.6
  vs incumbent 99.2–108.5 aggregate decode window (ABBA ×2, −52%): the BatchN
  wave forfeits the head⇄tail GPU overlap that IS the 2.37× bubble fill (the head
  must finish BOTH head-halves before sending either frame → whole-wave
  alternation, zero cross-box overlap). **`pair_decode` DEMOTED to opt-in**
  (default OFF; `RIIR_DIST_PAIR=1` arms — the loopback +39% stands for
  co-located rigs). **The kernel-level batch-2 rung is CLOSED for the wire**
  (the incumbent cadence never holds two ready chains — no engagement window);
  residual scope is the loopback rig only. The wire G1 rider PASSED (first
  2-box bit-identity: both streams == the 1-box oracle, `--full-check`).
  **CODE LANDED 2026-10-27** (zed-amo-i623-t24, [Bench 908](../.benchmarks/908_plan623_t24_multi_stream_round_robin.md)):
  `dist/src/multi.rs` — `run_head_multi`/`run_tail_multi` over ONE link; wire additions
  ADDITIVE (`BeginStream` seq|len|chunk + `TokenStream` seq|token; the Phase-1
  `ActivationFrameHeader.seq_id` already carried the stream id). Stream = stage-table
  index, ONE stage instance per stream (per-stream cache, Arc weights), table length =
  admission cap. Round-robin waves: one work item per non-done stream per pass, drain
  exactly one reply (FIFO route verified loud). G1: `tests/multi_stream_g1.rs` 7/7 —
  per-stream tokens bit-identical to the 1-box oracle at EVERY split, one-stream ==
  single-stream runner, B's prefill interleaves A's decode (both directions pinned),
  per-stream position cursors independent, unknown/duplicate-seq refusals; full dist
  suite 9+2+8 green, clippy `-D`, rustfmt clean, `--features gpu` compiles. Real-model
  loopback smoke (Bonsai-2 27B CPU, split 32, `--full-check`): G1 PASS both streams,
  aggregate 18.27 tok/s vs 1-box oracle 11.11 = **1.64×** (contended box, indicative).
  **2-box GPU G2 cell 2026-10-28** (the box freed; same session, ABBA, medians):
  amo⇄bmo TB5 split-32 GPU both, `--max-new 32`: aggregate decode window **109.5 tok/s
  = 2.37× single-stream (46.9)** at s=2 — per-stream rate unchanged-to-better (54.6 vs
  46.9, pure bubble fill) — and **saturated at s=3 (109.4)**: more streams partition
  the rate (36.5 per-stream), they do not add throughput. OPEN rungs: the admission
  seam (mid-flight join, serve-time shape), kernel-level micro-batching (batch-2 step
  on the tail — Phase 4's launch-bound finding makes it real), T2.7 scoreboard rows
  (DONE — Bench 910).
  **ADMISSION SEAM LANDED 2026-10-28** (zed-amo-i623-admit): `Admission`/
  `AdmissionSource`/`StageFactory` + `run_head_multi_admit` — one wave engine
  (`run_waves`) now drives BOTH doors: the static `run_head_multi` (borrowed table,
  `ClosedAdmission`, unchanged public signature — the 7-test G1 battery green
  unchanged) and the joining door (a `Vec<S>` pool + factory; an `Admission::Join`
  appends a factory-built stage at a pass boundary, sends its `BeginStream` there,
  reserves the joiner's frame bound (the T2.1b posture, joined edition), and refuses
  loud on empty prompt / over-capacity / foreign handshake). The TAIL needs no
  change — its loop already accepted a `BeginStream` any time and its static table
  IS the tail's admission cap (serve posture: prewarm the joiner's slot — the
  test-side lesson, a cap-1 tail refuses the join mid-run). G1:
  `tests/multi_stream_admission.rs` 4/4 — a stream joins MID-DECODE (its first
  tail frame lands after the resident stream began decoding), both streams'
  tokens bit-identical to the 1-box oracles THROUGH the join, the wire step census
  exact, Wait inert + Close drains, the three refusal classes, prewarmed slots
  beyond the initial table legal. Clippy `-D warnings`, rustfmt clean, full dist
  CPU suite 9+2+8+7+4 green. Honest residue: the idle-table `poll` contract is
  BLOCK (the engine never sleeps for it — a serve front-end blocks on its
  listener); a malformed join still refuses the RUN (validation belongs to the
  caller's admission front-end); kernel micro-batching stays the forwarder rung.
- [x] **T2.5** Planner: boot probe of per-layer cost per host (prefill + decode) and link
  RTT/bw (T0.2 code), split to equalise stage time under memory; manual override.
  **LANDED 2026-10-27** (zed-amo-i623-t21b) at SPLIT granularity (the honest PoC
  scope — per-LAYER hooks would touch the contended forwarder internals; candidate
  splits are what the cells measure anyway): `dist/src/planner.rs` —
  `probe_decode_spt` (timed decode on any `StageCompute`: warm-up + best-of-rounds
  median, capacity-refusing, logits-wide last stages safe) + `plan_split` (pure:
  per-token time at k = `max(head, tail) + hop`; hop = RTT + activation_bytes/bw —
  matches the T2.6 measured rows: TB5 +6 ms decode ≈ its RTT; memory axis runs
  BEFORE timing — head needs k layers of state, tail n_layer−k, either host
  over-budget refuses the candidate; a broken probe row (non-finite/≤0/out-of-range
  k) refuses, never trusted) + `LinkCost`/`HostProbe`/`SplitPlan` records.
  Bin: `dist_stage --plan-report 1,16,32,64` — per-candidate SCOPED loads, timed
  decode, one JSON record line (`host`/`n_layer`/`hidden`/rows), exit; the head's
  `k == n_layer` row IS the 1-box reference the T2.7 scoreboard needs. Two records
  + a link cost cross by operator/file; `--split N` stays the manual override and
  never consults the planner. G1: 5 planner unit tests (equaliser choice, RTT
  floor, both-host memory refusals, no-common-candidate, broken rows) + 2 e2e
  tests (`tests/planner_probe.rs`: real CPU-stage probes → plan → the formula
  exact; capacity/zero-step refusals). Clippy `-D warnings`, rustfmt clean,
  full dist suite 9+2+8 green. Honest residue: the automatic two-host probe
  EXCHANGE (wire step) is T2.4's territory (the multi-stream protocol work);
  real-box probe cells (GPU postures) are T2.7's measurement lane.
  **RESIDUE LANDED 2026-10-28 (zed-amo-i623-exchange, `22eb298`): the probe
  EXCHANGE is its own posture pair, not a serving-protocol entanglement —
  `dist/src/probe.rs`: strict-LE wire forms (`ProbeRequest`/`ProbeReply`, kinds
  13/14 additive; `ProbePing`/`ProbePong` 11/12 for the RTT), `measure_rtt`
  (median ping/pong over the SAME authenticated transport the serving session
  will use), tail `serve_probe_session` (answers until Shutdown; candidates
  validated 1..n_layer at the source; MAX rows/requests bounds) + head
  `run_probe_exchange` (measure → ask → model cross-check → `plan_split` →
  quote every input). Bin: `--answer-probe` (tail plan posture, never serves)
  + `--plan-wire CANDIDATES` (head: probes self, asks, prints the plan +
  `PLAN: --split K`) + `--link-bw` (the operator's CARRIED override — the
  default MEASURES the bw live, see the residue note below; RTT is always
  the measured half). The
  per-candidate probe loop is ONE shared `probe_candidates` (file arm + both
  wire postures — never three copies). G1: 5 module tests (strict decode
  arms, model-mismatch refusal, in-memory session) + `tests/probe_exchange.rs`
  3 loopback-TCP e2e (formula exact over the two live records, out-of-range
  candidate refused, lying-model reply refused); the lib hang the landing once
  had (missing Shutdown → serve recv blocks forever) is itself pinned by the
  test now sending it. **WIRE VERIFIED (Bench 915, CPU-only by design — the
  amo GPU held by sibling lanes): amo⇄bmo TB5, RTT 0.166 ms measured on the
  serving link, the tail's record crossed + cross-checked (n_layer 64 / hidden
  5120), plan composed live → `PLAN: --split 32` — the equaliser lands on the
  exact posture every measured 2-box cell runs (902/908/910/913). A serving
  tail refuses every probe frame at its Handshake expectation by construction
  (plan-mode postures only).** **RESIDUE 2 LANDED 2026-10-28
  (zed-amo-i623-t25residue): the two Bench-915 honest-notes — (1) the link bw
  is MEASURED on the wire by default (echo-ack bulk: `ProbeBw`/`ProbeBwAck`
  kinds 15/16 additive plan-mode; the tail echoes the received byte length —
  PURE I/O, zero compute, so the v0 "a bw probe would conflate tail compute
  with wire time" concern does not apply to the echo-ack shape; `bw = payload
  / max(elapsed, rtt)`, median of 3 rounds, 4 MiB bulk — the denominator
  floor keeps a loopback-class reading conservative, never inflated; the
  measured figure INCLUDES the transport's MAC cost per frame, i.e. it
  prices what a frame actually costs — the honest serving-path input);
  `--link-bw` stays the carried override (a peer predating kind 15 refuses
  loud — never a silent fallback); `ExchangeCtx.bw_bps` is `Option<f64>`
  (None = measure), the outcome carries the resolved value, and the boot
  line discloses measured-vs-carried. (2) the tail's answer posture prints
  its own per-candidate rows (`[tail] probe split k: …`) — both sides' raw
  lines on record. G1: +2 module tests (the measured in-memory exchange is
  formula-exact over the MEASURED bw; lying/truncated/wrong-kind ack
  refusals) + `tests/probe_exchange.rs` 4th loopback e2e (the measured-bw
  posture over real TCP; sanity band, never a perf claim). Full dist suite
  16+4+4+7+2 green, clippy `-D` clean.** Residual: the memory axis stays
  planner-level (budgets None on the wire — additive field if a v1 wants it);
  real-box GPU probe cells remain T2.7's measurement lane.
- [x] **T2.6** Tailscale posture: the same runner over the tailnet address; a measured row
  (the WAN/fallback lane exo cannot do — multicast discovery).
  **MEASURED 2026-10-27** — the unchanged `dist_stage` (no code change: the transport
  is plain authenticated TCP, so any routable address works). amo head ⇄ bmo tail,
  scoped GPU stages, P=256 (chunk 128 → 2×128), 32 decode steps, order TB/TS/TS/TB,
  amo load ~4 (shared — indicative): RTT **TB5 0.45 ms vs Tailscale 5.9 ms**; prefill
  TB 5.58 / 5.46 s · TS 5.68 / 5.91 s (**+4%** — chunked frames amortise the RTT);
  decode TB 16.62 / 16.15 · TS 14.82 / 14.94 tok/s (**−10%** ≈ +6 ms/token = the one
  activation→token round trip per decode step at the tailnet RTT). The WAN lane works
  and costs exactly its RTT on decode; the per-token round trip is the lever (T2.4's
  micro-batching / a speculative fill hide it).
- [x] **T2.7** G2 scoreboard rows (Phase-0 harness, same models): PP overhead vs 1-box;
  vs llama.cpp RPC-TCP, mlx-lm ring PP, exo PP-ring. `.benchmarks/` record.
  **DONE 2026-10-28** (zed-amo-i623-t24,
  [Bench 910](../.benchmarks/910_plan623_t27_scoreboard.md)): the scoreboard assembled
  from the measured rows — our PP s1 46–48 t/s (≈ −15% vs the Issue-1144 1-box 56.9),
  **s2 aggregate 109.5 t/s = 2.37× s1 (saturated at s3)** — vs llama.cpp-RPC 2-box
  −20% (Bench 900), mlx-socket PP −26% (Bench 904), mlx ring DEADLOCKED at 2 nodes
  (0.32.0), exo BLOCKED on the PQ2_0 weight format (Phase 0 note, unchanged). Verdict:
  single-stream PP loses 15–26% on EVERY family that measured it; round-robin
  multi-stream is the differentiator and no opponent row measures ≥2 streams (honest
  cross-family comparison stays at s1). Scope: the s2/s3 cells are duplicated-request
  rows (the T2.4 admission is the static table).

## Phase 3 — RDMA transport (needs D2)

- [x] **T3.1** Verbs FFI: `dlopen("librdma.dylib")` + the ~16 `ibv_*` symbols, the inline
  `post_send/post_recv/poll_cq` re-implemented via `ctx->ops` against the macOS 26.2+ SDK
  header; UC QP bring-up INIT→RTR→RTS with IPv4-mapped GID; bootstrap over the TCP
  side channel. Std-only.
  **LANDED 2026-10-29 (zed-m3-i623-t3verbs, `226cace`+`16fb64a`+two verbs_ping fixes;
  [Bench 927](../.benchmarks/927_plan623_phase3_verbs_bringup.md) is the record):
  `crates/riir-infer-dist/src/verbs/{ffi,mod}.rs` behind the opt-in `rdma` feature — the SDK
  header mirrored repr(C) with layout pins held to the C compiler's measured offsets
  (`tests/verbs_layout.rs`, 6/6 on the device-less m3 too); the four inline verbs ride the
  32-slot `ibv_context_ops` vtable (typed slots, poll_cq@88/post_send@200/post_recv@208);
  Arc ownership chain destroys child-before-parent by construction; MAC'd endpoint bootstrap
  (`no unauthenticated frame accepted` — the wrong-key negative REFUSES both sides before any
  QP traffic). FIRST RDMA TRAFFIC amo⇄bmo: UC QP pair, RTT p50 7.7 µs (TCP was 94–96 µs,
  12×) and 9.368 GiB/s one-way sustained 1 GiB (TCP bulk 8.6 GB/s). Provider laws measured:
  UC RTS mask is STATE|SQ_PSN only (TIMEOUT → EINVAL); posted-recv budget 4 MB OK / 5 MB
  rc=-12; UC credit flow-control stalls a dead peer's SQ as a poll TIMEOUT (never a WC
  error); GIDs are NOT IPv4-mapped on TB (index-0 fallback is the working path — the plan's
  "IPv4-mapped GID" assumption corrected in place). macOS-only by contract (the vtable is
  Apple's; rdma-core's evolved vtable is not claimed).**
- [x] **T3.2** Zero-copy probe: register an MR over a shared-storage `MTLBuffer.contents()`
  — if accepted, frames go GPU-buffer → wire with no memcpy (both incumbents bounce-copy
  twice); if refused, record the refusal and keep one memcpy per side.
  **HEAP LEG DONE (Bench 927): 67 MB registered, lkey granted. ✅ CLOSED NEGATIVE ON THE RKEY HALF
  2026-10-29 (`570ee79`, `verbs_write` probe, amo⇄bmo): RC is REFUSED AT CREATE (`ibv_create_qp(type 2)`
  returns NULL — the TB provider does not build the RC type, so RDMA READ is unreachable at any
  mask); rkey is granted **0x0 under EVERY access-flag combination** (the full grant matrix —
  LW, LW|RR, LW|RW, LW|RW|RR, even spec-invalid REMOTE_* alone — all register without refusal and
  all read rkey 0x0, lkey 0x01000000); and a UC RDMA WRITE ladder **STALLS AT THE FIRST POLL** —
  the WR is accepted (`post_send` rc=0) and no completion EVER arrives (no error WC either — the
  provider swallows the WRITE opcode in silence; SEND ops on the same QP complete normally). The
  zero-copy one-way shape is refused-by-silence at the provider: **SEND/RECV is the wire shape**;
  a heap bounce on each side is the measured cost of this provider posture. The MTLBuffer leg is
  CODE-COMPLETE (`verbs_metal`, `rdma_metal` feature — registration + GPU→wire + wire→GPU
  byte-identity with inline MSL fill/sum kernels) and UNBLOCKED the moment the binary is
  allowlisted: the macOS Application Firewall on amo silently drops the initiator's TCP SYNs to
  the unsigned target binary (nc/python/ssh connect to the same port; the Rust bin's accept never
  fires; samples show both sides healthy — initiator in read_exact, target in accept). Owner
  gate: `sudo socketfilterfw --add <bin> --unblockapp <bin>` (or sign it); the Metal-side premise
  verified independently (ObjC shared MTLBuffer + blit fill reads back through contents()).**
- [x] **T3.3** Small-message decode path (single WR, inline header) vs striped large frames
  for prefill; MTU probe (incumbents fix 1024 / 128 KiB strides).
  **✅ CLOSED 2026-10-29 (zed-m3-i623-t32t33, the `--mtu` override ladder on amo⇄bmo): the
  override ENGAGES cleanly (path_mtu=1024/2048 confirmed in the RTS line) and is
  PERFORMANCE-NEUTRAL at every shape — 128 KiB-stripe BW 9.342 / 9.337 / 9.294 GiB/s and
  1 MiB-stripe 9.229 / 9.295 / 9.277 GiB/s at MTU 4096/2048/1024 (−0.5% at the floor, within
  run noise); 64 B inline RTT is MTU-INSENSITIVE (single packet at every MTU; per-run spread
  6.0–10.1 µs at FIXED MTU — round 1's apparent MTU trend was order/variance artifact,
  refuted by repeats). **The posted-recv budget law is CORRECTED — the cliff is MTU-INDEPENDENT
  and BOX-LOAD-DEPENDENT**: identical outcomes at every MTU; today's edge = 3.75 MB PASS /
  4.0 MB FAIL under bmo's ~70 GB wired (flash_cmma bench + mammoth capture resident), vs
  Bench 927's 4 MB PASS / 5 MB FAIL the same day under quieter load; bytes-shaped (128 KiB×28
  = 3.5 MB/28 WRs passes at the edge), not count-shaped. THE LAW: never hard-code "4 MB is
  safe" — size the recv ring ≤3 MB, or runtime-probe the budget before sizing; the cliff
  moves with co-tenant wired memory.** (Small-message + striped rows: Bench 927.)
- [x] **T3.4** GPU↔comm sync: CPU comm thread on `MTLSharedEvent` signal values; no per-
  layer command-buffer commit on the critical path. (Unit issue
  Issue 1151 — LANDED 2026-10-29
  zed-m3-i623-t34gpucomm: the decode-feed async-finish arm (staged-copy read future
  through CubeCL/wgpu — the MTLSharedEvent correction recorded in the issue),
  all-feed-passes-only deferral (the Bench-913 wire-order class), demote-only env
  `RIIR_DIST_ASYNC_FINISH=0`; G1 hidden-bits exact ×6, s=2 all-feed decode window
  ~5% faster armed; per-layer commits: no such cadence exists in the wgpu encoder
  (≤32-task batches), the uniform-write flush analog left instrumented. Open
  residues: the literal comm-THREAD split + the tail logits arm ride T3.5's
  scoreboard harness.)
- [x] **T3.5** Scoreboard rows vs llama.cpp RPC-RDMA and exo/MLX jaccl. (jaccl prose
  comparison exists — Bench 905's 16–22 µs pipelined collective rate vs our 7.7 µs
  ping-pong RTT — but a like-for-like scoreboard row needs the T3.4 serving integration.)
  **LANDED 2026-10-29 (zed-m3-i971-housekeeping, [Bench 929](../.benchmarks/929_plan623_t35_scoreboard.md),
  issue 1153 carries the Claude-verdict fold + corrections): decode row — ours 2-box s1
  52.3–52.6 tok/s BEATS llama.cpp-RPC 2-box 48.6 (+7.6%, the Issue-1141..1146 rungs
  flipped T2.7's −20% reading); the ≤5% PP-overhead gate FAILS honest at −7.1% vs the
  in-run 1-box oracle 56.29 (localized: second-hop RTT + second dispatch, not host
  serialization). T3.4 arm WIRE-REFUTED (ABBA ×2, −48.5% decode window armed at s=2 —
  the drain serializes the pass; the +5% was a single-queue loopback artifact) →
  DEMOTED to promote-only (issue 1154, `6a1c524`). Prefill (the exo verdict fold):
  long-prompt warm cells — the ~2× overlap bar PASSES at the incumbent chunk (1.76×
  aggregate at c128), single-chunk postures land the serial 1.1–1.25× prediction, and
  exo's chunk-size bubble heuristic (`prefill_step_size // min(4, world_size)`,
  `generate.py:194`) does NOT transfer — finer chunks pipeline better on the
  activation pipeline; the 128 default stands. Decomposition (`RIIR_DIST_STEP_TIMING`):
  tail fwd+logits = 78% of the 19.05 ms/token wall, head-local <1% → BOTH T3.4
  residues DECLINED (comm-THREAD split, tail-logits arm); the verbs-Transport serving
  swap is NOT justified (the link leg is not binding at this split). Incumbents:
  llama.cpp RPC is TCP-only ("RPC-RDMA" N/A), exo N/A (PQ2_0), jaccl stays prose
  (serving forms dead by Benches 904/906). Harness lessons: foreground-chunked cells
  are the discipline (sequential-drift law; the detached-spawn EHOSTUNREACH class
  was RE-TESTED 2026-10-29 and refuted as a deterministic launch-context property —
  the signature is the issue-1152 posture drop, unproven for the incident; the harness
  now retries the spawn leg on the never-delivered empty-log signature — Bench 929
  addendum), four `set -u`
  unset-expansion deaths on bash 3.2, stale-log readiness race, macOS `seq -s,`
  trailing comma.)

## Phase 4 — Tensor-parallel (gate REFUTED 2026-10-27 by the T4.1/T4.2 cell — [Bench 906](../.benchmarks/906_plan623_t41_tp2_cell.md): the [Bench 905](../.benchmarks/905_plan623_phase3_jaccl_probe.md) RTT was the PIPELINED all_sum rate (16–22 µs); the TP dependency chain pays the SYNCHRONIZED per-exchange cost (183–277 µs measured) → 128 serialized exchanges/token ≈ +27 ms → TP-2 decode 18.2 t/s vs 1-box 58.1/50.8 (−69/−64%). Sharding correctness PROVEN (token-exact ×6 produce paths, replay-deterministic). Independent second negative: batch-1 decode is launch-bound (COMM_OFF floor −11..−14% from halving ~90% of weight bytes) — the TP memory thesis does not apply at batch-1 on this model/box class. Prefill parity (−1..−9%). TP stays a bench-lane harness, not a serving posture; the multi-host single-stream win path is PP (Phase 2) + RDMA (Phase 3) + speculative fill (Phase 5). Re-open triggers recorded in Bench 906 §9.)

- [x] **T4.1** Generic sharder from the arch layer table: column (q,k,v,gate,up), row
  (o,down) + allreduce; kv-heads divisibility check; ternary/quant packs split on
  block boundaries. **DONE 2026-10-27 (`scripts/plan623_mlx_tp2_bench.py`, mlx
  bench-lane harness over the pack runtime — NOT riir-infer-core Rust): column
  q/k/v(+fused per-head gate)/gate/up/GDN-in_proj; row o_proj/down_proj/out_proj;
  lm_head vocab-split + all_gather; embed/norms/GDN-small replicated. Hadamard
  block-locality law pinned: in-dim slices must be 1024-aligned → the MLP
  intermediate splits BLOCK-ALIGNED UNEVEN 9216+8192 (17408 = 17 blocks, odd
  count; the even split cuts block 8 across ranks). kv/GDN head divisibility ✓
  (12+2 / 8+24 per rank). N=1 arm = the no-comms sanity posture through the same
  code path (token-identical to the stock forward).**
- [x] **T4.2** Allreduce: fixed-order 2-rank sum (identical on both ranks → deterministic,
  Issue 940); overlap the o_proj allreduce with the MLP column matmuls; fuse residual add.
  **DONE 2026-10-27 (measured, Bench 906): 2-rank all_sum = one IEEE-commutative
  pairwise add → deterministic (held: token-exact, replay-identical ×2 runs ×2
  ranks). The OVERLAP lever is UNATTAINABLE at the mlx level — the jaccl op blocks
  the stream (comm-diag arm C: 277 µs with independent GPU work in-flight vs 21.9
  pipelined); overlap needs a side-thread p2p transport = Phase 3's own verbs lane
  (mlx p2p itself deadlocked, Bench 904 §4 + 905). Residual-add fusion moot at
  −69% decode.**
- [x] **T4.3** Scoreboard: single-stream decode vs exo TP-jaccl (claims "up to 1.8×") and
  llama.cpp `-sm tensor` RDMA. **DONE 2026-10-27 (Bench 906 §7): TP-2 jaccl 18.2
  t/s (−69%) — the exo claim reads 0.31× on mlx 0.32.0 collectives; `-sm tensor`
  unsupported over RPC (Bench 900); PP lanes: llama.cpp-RPC 48.6 (−16%), MLX
  socket 43.0–45.2 (−22..−26%). Multi-host single-stream verdict: PP + RDMA +
  speculative fill.**

## Phase 5 — Speculative fill

- [-] **T5.1** Draft on stage 0 (dflash/speculative substrate already in core), verify
  batch rides the pipeline as the micro-batch; acceptance-rate-weighted scoreboard row.
  Lossless by construction (verify is exact) — G1 byte-identity holds.
  **T5.1a–c LANDED 2026-10-28 (zed-amo-i623-t24, `6c608dd`): `dist/src/spec_fill.rs` —
  the VerifyStream/VerifyReply wire (one round trip per K drafts; K argmaxes cross,
  never K×vocab logits; the pending prediction rides the request so the tail
  adjudicates the SAME m the head does — a reply m mismatch refuses), the rewind via
  the Issue-028 `HybridStateSegment` snapshot/restore (T5.1b V0 RESOLVED as
  snapshot-based — byte-exact by construction, zero forward surgery; the GDN journal's
  rank-1 unwind stays the FAST-path follow-up rung; the GPU stage has no SpecStage impl
  — the dyn seat refuses at the type level), the runner pair (`run_head_spec` /
  `run_tail_spec`, the ordinary arm mirroring `runner`'s loop), the v0
  `RepeatLastDrafter` (acceptance ≈ 0 — the machinery lane, never a gain claim), and
  the G1 battery 7/7 (`tests/spec_fill_g1.rs`: full-accept == oracle, partial rewind
  m=0 AND m≥1 == oracle, repeat-last worst case == sequential byte-for-byte, shifted-
  alignment unit pins, codec refusals, tampered-m cross-check, snapshot round-trip).
  Bin: `--spec-k K` (1-stream, CPU-stage only — loud refusals otherwise). Traps paid:
  the rewind arm's extra recv_token deadlocked the pair (ONE reply per frame — the
  last in-loop reply IS the bonus); the MAC key is SHARED (mismatched test keys read
  as a tampered wire). Alignment note: the runner's cursor law makes acceptance
  SHIFTED (d_0 vs pending, d_j vs argmax_{j-1}) — a full accept commits K+1
  (drafts[1..] + argmax_{K-1}), a partial commits the re-forwarded prefix + bonus.**
  **V0 DESIGN 2026-10-28 (zed-amo-i623-admit — the substrate map is verified, the
  slices are ready to implement; nothing here needs re-scoping):**
  - **Why the verify batch is bit-identical (the G1 anchor):** a K-token verify
    forward IS a chunked-prefill frame — `StageCompute::prefill_chunk` is G1-proven
    bit-identical to per-position steps (T2.3b, CPU default impl AND the GPU
    `prefill_range_chunk`), so verifying K drafts through ONE `n_tok=K` frame reads
    exactly the logits sequential decode would have produced. Lossless by
    ADJUDICATION: accept the longest prefix of drafts whose argmax match the
    target's own per-position argmax; the output is the target's own tokens either
    way.
  - **T5.1a — wire (additive, `dist/src/spec_fill.rs`, NOT a multi.rs growth —
    the file-size law):** `VerifyStream` request frame (seq | n_tok | n_tok×u32
    draft tokens) + `VerifyReply` (seq | n_tok | n_tok×u32 per-position argmax).
    The TAIL computes the argmax per verify row from its own logits — K u32 cross
    the wire, never K×vocab logits. New tail loop arm beside the multi loop; the
    head adjudicates (longest common prefix) + decides commit-or-rewind.
  - **T5.1b — rewind (the load-bearing substrate question, half-answered):** GDN
    layers unwind via `GdnStateJournal::unwind_to` (core `deltanet/delta_roll.rs`,
    opt-in `gdn_delta_roll`, its own G1 drift pin + G4 gates, Bench 885 — the CPU
    stage does NOT wire the journal yet; wiring it into the stage is this slice).
    ⛔ ATTENTION KV (1-in-4 layers) has NO truncate today (`HybridCache` has only
    `reset()`): first verify the cursor-overwrite property (writes at pos j
    overwrite stale rows → a cursor reset is the truncate); if the length lives in
    per-layer implicit cursors, add `truncate_to` — EITHER WAY pin it with a test
    that re-forwards at the rewound position and compares bit-exact to a fresh
    cache. **LANDED 2026-10-28, MECHANISM SUPERSEDED (riir-infer `ddf4be0`, Issue 1148 close,
    [Bench 918](../.benchmarks/918_issue1148_exact_checkpoint_rewind.md)): the
    rewind is an in-process EXACT checkpoint (`SpecStage::checkpoint_state`/
    `restore_checkpoint`, retained buffers, live KV rows bounded by the runner
    cursor — the ternary forward never advances `fill_pos`, so the cursor is the
    only live bound), byte-exact by construction, zero steady-state alloc — no
    serialization. THE JOURNAL IS NOT WIRED: its measured 6.3e-6 unwind drift
    (Bench 885) fails Issue 1148's byte-exact ground rule — the wall outranks the
    mechanism named here (the wall is the lane's lossless G1: stream == sequential
    byte-for-byte, any drafter). The cursor-overwrite truncate law is PINNED at
    NaN-poison strength (`kv_cursor_overwrite_truncate_law_nan_poison`: rows past
    the rewound cursor are stale-but-unread, never restored). Re-cell: E[m] 1.6754
    (Bench 917's ≈1.68 reproduced), spec pair 5.03 tok/s vs ordinary 7.62 (0.66×;
    was ≈0.06×) — the 70× snapshot tax is O(1×) dead; the remaining 0.34× is the
    checkpoint + re-forward structure on a CPU pair whose hidden RTT is ~0.3 ms
    (the linked-pair posture — Tailscale/RDMA, T2.6's ~10 ms RTT — is where the
    lane's round-trip hiding pays).**
  - **T5.1c — the runner mode + the v0 drafter:** `DraftSource { propose(&mut
    self, ctx) -> &[u32] }` injected into the head; v0 impl = repeat-last-token
    (acceptance ≈ 0 — the honest disclosure: this proves the MACHINERY + the
    lossless G1, it proves NO gain; the acceptance-rate row is meaningless until a
    real drafter exists). G1 tests on the synth fixture: full-accept path
    bit-identical; rejection at j<K → unwind → the continued stream == the oracle;
    dumb-drafter worst case == the sequential stream byte-for-byte; wire refusals.
  - **T5.1d — the real drafter (the actual T5.1 completion condition):** an
    MTP/draft head for Bonsai-2 is a riir-train lane (GPU) — **FILED as riir-train
    Issue 654** (`654_mtp_draft_head_bonsai2_651d.md`); the v0 zero-shot
    tied-projection calibration floor LANDED (`5dbb0a3`); **the trained-head GPU
    rung LANDED 2026-10-28 (riir-train Bench 677): the head beats repeat_last
    (0.7929 vs 0.4150) and the modal drafter (0.7929 vs 0.6368) on the frozen
    read and clears the content GOAT-class bar (0.4333 ≥ 0.4); BMTH v1 artifact
    (60.7 MB, support 2963) on the 4090 box; remaining = the consumer (`impl
    DraftSource` over the BMTH bytes here, G1 battery re-runs unchanged) + the
    owner-gated promotion; the acceptance-rate-weighted scoreboard row rides
    the consumer lane.** **THE CONSUMER LANDED 2026-10-28 (Bench 917,
    `spec_mtp.rs` + the roll seam): in-posture acceptance MEASURED, not
    inherited — teacher-forced per-slot 0.6841 at split 63 (vs the 0.7929
    post-final fit basis), 0.0591 at 48, 0.0332 at the production split 32:
    the viable seat is SPLIT 63 ONLY (the basis cliff sits between 48 and
    63 — linear decodability lives in the final layers); projected serving
    E[m] ≈ 1.68 at k=2 there. The free-run E[m] (1.10) is stream-polluted —
    the model's raw-text greedy decode collapses into a (0, 271) period-2
    attractor (state-path cross-checked TF==pair; a model property, not a
    consumer defect); the serving-shaped row needs chat-shaped prompts
    (follow-up). The spec machinery's wall-clock is snapshot-bound (~70× vs
    the ordinary pair; k=1 == k=2 isolates the per-round HybridStateSegment
    snapshot ≈ 4 s — filed as Issue 1148, the fast path is the plan-named
    GDN journal unwind + KV truncate). `--drafter mtp --head PATH` wired
    into dist_stage (split guard refuses low seats loud). Promotion NOT
    claimed — owner-gated.** **T5.1d COMPLETE 2026-10-29 — the owner-gated
    promotion e2e EXECUTED (owner GO "654 lgtm"; riir-infer [Bench
    931](../.benchmarks/931_i654_promotion_e2e.md)): promotion REFUSED per
    the pre-registered gate — every replicated cell loses to plain decode
    (k=2 0.846×/0.946× paired, k=4 0.745×) with the CUDA-placed head
    (`BmthHeadCuda`, the gate's placement precondition satisfied + the
    placement delta measured small: 924's 0.808× CPU → 0.846–0.946×); the
    residual is STRUCTURAL (exact rolls = full steps ⇒ ~K× ordinary stage
    time per round; deeper K worse) and the head BEATS the modal comparator
    at every k. E[m] bit-stable across scorers (1.9592@k2 / 3.8400@k4 == the
    682 reads). The arm stays opt-in (`--drafter mtp`, `--head-device
    cuda|cpu`); the win-capable next rung (real-RTT linked posture + GPU
    stages + a GPU-resident checkpoint) is named in the bench record,
    machinery unbuilt. The head remains the serving-chat drafter candidate
    (riir-train Bench 682).** dflash
    itself (`core/src/dflash.rs`) is wired to the OLD transformer path
    (`TransformerWeights`/`MultiLayerKVCache`), NOT the ternary DeltaNet stage —
    "already in core" means the algorithmic cores (`katgpt_speculative`), not a
    drop-in drafter; do not assume one exists.

## GOAT gate (promotion = opt-in → default for the lane crate's consumers)

- G1 PP byte-identity vs 1-box (every phase); TP replay-determinism. **TP half
  VERDICT 2026-10-27 (Bench 906): G1 PASS (token-exact ×6 produce paths,
  replay-identical); G2 FAIL (−69% decode) → NOT promoted; the TP harness stays
  bench-lane. The Phase-4 gate REFUTATION also corrects the Bench-905 paper
  gate: pipelined ≠ synchronized exchange latency.**
- G2 `--release`, both boxes' provenance; beats the best incumbent on at least the
  capacity row and the aggregate-throughput row; never > 5% PP overhead on the fits-one-box row.
  **FULLY EVIDENCED 2026-10-28 (Bench 916): the fits-one-box row measured — loopback PP s=1
  vs the same-process 1-box oracle (GPU both postures, split 32, ctx 4096, amo, GPU-quiet):
  overhead −1.4..−1.6% across 3 runs (G1 PASS ×3) — ≤5% with margin, point estimate at/below
  zero (the negative sign is a posture note: two 3.6 GB stage instances vs one 7.2 GB, no
  win claimed); the capacity + aggregate clauses are Bench 910's scoreboard rows.**
- G3 feature-off byte-identity of riir-infer-core (the `multi_host` feature compiles away).
- G4 decode frame path alloc-free.
- Security: no unauthenticated frame accepted (negative test with a wrong key).
