# Plan 616 — Laya D2b: the Q8 residency tiers (host → device) with dequant-fused staging

**Status:** COMPLETE (2026-10-02 — all four phases executed; Phase 3
landed `323edbf`: the D2b adoption recorded via the no-change re-seat
(G5 both postures green vs the in-flight substrate; the live probe
reproduced Phase 2 exactly at HEAD), the /#sizes row re-measured at the
adopted q8 posture (reflex-site `78488db`, model 455,724,111 B, the
memory line the headline), and the Q4 format seam landed (decoder +
shared blocked-artifact converter + derived q4 staging kernels + the
bit-identity battery at Q4's own fidelity, ALL GREEN; Q4 retention
D1-priced separately before serving; PQ2 rides the same seam). Phase 0
(baselines) · Phase 1 (host residency 2.85×) · Phase 2 (device residency
4.74×, the fused staging, option (i) shipped) done as recorded below.
Plan of record:
`../riir-instinct/.issues/018_rethink_encoder_lean_goat.md` §Lane D2b.

## The measured premise (what exists today)

- D2a (Bench 0047, `10e33de`) made Q8 a STORAGE tier: the artifact halves
  the FILE (842.6 → 447.7 MB) but the runtime still materializes F32 —
  `weights::widen_q8_0` dequantizes at load into
  `Weights.data: Vec<f32>` (~1.68 GB host for english), and the Metal
  backend uploads F32 verbatim: `weight_t_buf` (device Wᵀ [k, n], the
  `matmul_w` form) + `weight_buf` (1D norms/biases). Device ≈ host ≈
  1.7 GB F32 today under the q8 posture — the storage win never reached
  memory.
- The weight inventory (english q8 artifact, 206 tensors): 126 Q8_0 (every
  ≥2D GEMM weight; per encoder layer wqkv [3072,1024] + wo [1024,1024] +
  wi [5248,1024] + mlp_wo [1024,5248] ≈ 14.75M elements × 26 layers) +
  80 F16 (the 1D norms/biases — stay F16→F32 per the house GGUF norm law,
  <1% of bytes). `tok_emb` [50368,1024] (206 MB F32) is deliberately
  HOST-ONLY (the token gather is host-side; it never reaches a device
  buffer — `Encoder::warm`'s doc).
- The GEMM dispatch tree (`Metal::run_sgemm_one`, batch-1): split-K
  (`sgemm_splitk`, m ≤ 32 under `SplitRule::WITH_MPS`) → the MPS arm
  (`run_sgemm_mps`, unsplit dense m ≥ `mps_min_m` — consumes the F32 Wᵀ
  buffer as a raw `MPSMatrix` operand) → the band rule
  (`sgemm_xwide` / `sgemm`, B-tile staging from Wᵀ).
- The staging identity that makes bit-identity achievable: `widen_q8_0`
  computes `w = d · f32::from(q)` with `d = f16-scale widened to f32`
  (f16→f32 is exact). An MSL staging that reads the same i8 + u16-scale
  and computes `float(q) * float(half_scale)` produces the IDENTICAL f32
  values — so a dequant-FUSED staging feeds the MMA the same tiles and
  every downstream accumulation is bit-identical by construction. This is
  the plan's central claim and every gate below proves it.

## The layout constraint (why this is kernel work, not a flag)

⚠ CORRECTED BY PHASE 1 (measured, the op gate caught the first draft):
the house Q8_0 blocks run over the FLAT tensor order — the converter's
`w.data.chunks(32)` — NOT per-row. "Blocks along k" holds only for
k % 32 == 0 (every big projection); a ragged k (the a0w class,
k = d + 4) has blocks CROSSING row boundaries and the tail block at the
flat end. The kernel's addressing is the flat formula (block = flat
element / 32, lane = flat element % 32) — no row_bytes/tail constants,
and the exact `widen_q8_0` walk per element.

A host-side Q8 TRANSPOSE (re-blocking Wᵀ) would
requantize — `d'·q' ≠ d·q` in general — and break the bit-identity claim
at the root. So the fused staging must read the NATIVE [n, k] layout and
dequant-then-transpose in-kernel: for a staged [BK, BN] B-tile, fixed-n
columns share 1–2 contiguous blocks (BK=64 aligned → 2 scales + 64 i8 per
column ≈ 68 contiguous bytes), consecutive threads take consecutive n —
a different staging shape with its own coalescing analysis, not a flag
flip. (The kernel_opt corpus' staged-B family — dequant-once-per-macrotile,
in-place-transpose-at-staging — is the pattern library; B186's regime
gates apply; the flat-block law above is the Phase 2 staging's
addressing premise.)

## The MPS tension (the one real design decision — priced, then decided by measurement)

MPS binds raw F32 MTLBuffers; it cannot dequant-fuse. Under device-resident
Q8 the options are:

| option | device memory | dense-shape perf (T13: MPS −26…−44% p50) | verdict |
|---|---|---|---|
| (i) MPS OFF under q8 | full win (~1.53 GB → ~0.41 GB GEMM weights) | dense m-ranges revert to our sgemm | **the default** — the q8 posture IS the memory posture; F16+MPS stays the speed posture; the split is honest and disclosed |
| (ii) keep F32 Wᵀ copies for MPS | ~no win (MPS consumes exactly the big GEMM weights) | kept | refused — defeats the lane |
| (iii) an MSL MPS-replacement dense GEMM with fused dequant | full win | unknown — T13 exists because Apple beat us there; the fused variant re-prices it | the Phase 2 stretch, only if (i)'s measured regression bites |

Phase 2's A/B measures (i)'s cost explicitly (the m 33–895 dense shapes
under q8, sgemm-fused vs today's MPS); if the regression is material AND
the GPU host is memory-rich, (iii) re-prices with its own bench — an owner
call at that point, not this plan's default.

**MEASURED (Phase 2, `examples/sgemm_q8_fused_probe`, 2026-10-02):**
option (i) shipped as the default with the cost priced per cell —
fused/mps 1.22–2.07× on the dense cells (m-scaling), fused/narrow
~1.15 (the dequant ALU), whole-forward 1.002× at the single-question
serving shape. The re-pricing condition above is MET with numbers:
long-prefill q8 serving is where (iii) would earn its bench — owner
call, not this plan's default. Until then `LAYA_Q8_DEVICE_F32=1` is
the documented escape at exactly those postures.

## Phases

### Phase 0 — measure the baseline (before any code) — **EXECUTED 2026-10-02**

- [x] Device + host residency at both postures, measured from the artifact
      headers + the warm-path set (`scripts/plan616_residency_probe.py`, the
      derivation committed): each checkpoint carries **421,205,504 Q8_0
      elements + 88,326 1D F16 elements**; device today (F32 widened) =
      **1.685 GB per checkpoint**; device-resident Q8 + F32 small =
      **0.448 GB (26.6%)** — a **3.76×** cut. Host (Phase 1) identical:
      1.685 → 0.448 GB. Lane B's two-worker posture: **3.37 GB → 0.90 GB**.
      (`tok_emb` is host-only today and stays host-only — not in either
      figure; its 206 MB F32 host copy is a separate Phase 1+ candidate if
      the gather ever moves on-device.)
- [x] The dense-shape MPS baseline (`sgemm_mps_probe`, m3 Metal, load
      2.18-2.31, ratios-only — the probe's own shared-box posture):
      **geo-mean mps/narrow 0.770 (m106) · 0.783 (m188) · 0.703 (m317) ·
      0.663 (m512) · 0.628 (m895) · 0.584 (m1700)** over the four real
      projection shapes (qkv/o/mlp-up/mlp-down), correctness
      bit-identical every cell. **Option (i)'s priced cost: −23%…−42%
      GEMM time on the dense shapes MPS serves today.** The Phase 2 A/B
      reads the fused-Q8 staging against exactly these cells (the staged B
      bytes halve under Q8 — the memory-bound staging gets cheaper, which
      is the mechanism that may partially offset the MPS loss).

### Phase 1 — host residency (cheap first rung, no kernel work) — **EXECUTED 2026-10-02**

- [x] Under `LAYA_WEIGHTS_VARIANT=q8` + Metal: skip the host F32
      materialization for the 126 Q8_0 GEMM tensors; keep the raw Q8 map
      (the loader retains it — `WeightData::Q8(RawQ8)`, never widened at
      parse) and dequantize-transpose on the DEVICE at warm time
      (`weight_t_buf_q8`: upload raw → the `q8_widen_t` load kernel →
      the SAME device F32 Wᵀ → the Q8 device buffer drops). Host ≈
      1.68 GB → ~0.5 GB; device unchanged; MPS unchanged (the q8-loaded
      Wᵀ is a plain f32 MTLBuffer to MPS); bit-identity proven (below).
      - The GEMM weights carry as `Weight2D { Dense(Vec<f32>) | Q8(RawQ8) }`
        through Encoder/Head; the forwards dispatch via `Weight2D`'s
        helpers; the Backend trait gains `warm_weight_2d_q8` /
        `matmul_w_q8` / `matmul_w_accum_q8` / `matmul_w_glu_q8` with
        WIDEN-ONCE defaults (`RawQ8.wide()` — the CPU/CUDA/CubeCL lanes
        resolve through it, byte-identical, no per-call widen); Metal
        overrides all four with the device path. One spine per op
        (`matmul_w_wb` family) — the f32 and Q8 entries share the body.
      - `tok_emb` + `type_emb` widen once at from_map and drop their raw
        bytes (host consumers: the gather, the bias rows) — the plan's
        tok_emb note stands (its 206 MB f32 is the Phase 1+ candidate).
      - THE LAYOUT LESSON (corrects this plan's own §layout-constraint
        prose): the house Q8_0 blocks run over the FLAT tensor order
        (the converter's `w.data.chunks(32)`), NOT per-row — the
        kernel's first draft assumed per-row blocking with a per-row
        tail and the op gate caught it (nn≠0 columns read the wrong
        scale wherever k % 32 != 0; blocks cross row boundaries). The
        flat addressing (`block = flat_element / 32`) is BOTH simpler
        (no row_bytes/tail constants) and the exact `widen_q8_0` walk,
        which is what makes the bit-identity claim structural per
        element rather than per shape. The plan's "blocks along k"
        sentence is true only for k % 32 == 0 — every big projection.
      - CPU-lane residency note: after first use the Q8 payload holds
        BOTH the raw bytes and the once-widened cache (~5.06 B/elt vs
        today's 4) — the CPU lane is the measurement/reference posture
        (serving is Metal), and `LAYA_Q8_HOST_F32=1` restores the
        exact pre-Phase-1 shape wherever the CPU posture carries
        production weight.
- [x] Gates: ALL GREEN —
      - `tests/q8_widen_identity.rs` (new, 5 tests): synthetic CPU
        identity (Q8 carrier vs Dense twin, encoder + head, bit-exact);
        Metal identity through the PACKED forward (mixed row segments →
        the fold arms AND the mixed-plan then_add fallback, warm path,
        bit-exact); first-miss (no warm call); the unfused stream
        (`with_folds(false, false)`); the load kernel's transpose read
        out through an identity matmul vs the host widen (k-tail
        shape).
      - `metal_ops_smoke`: q8 arms on the op level — `matmul_w_q8` vs
        the f32 carrier BIT-identical on Metal (25×768×2304 split-K +
        the 5×100×33 k-tail), Metal-vs-CPU at the file's 1e-3
        accumulation-order budget (the identity pair is device-vs-device;
        the first draft's device-vs-CPU bit assert was the wrong pair
        and the gate caught it within minutes).
      - THE LIVE ARTIFACT PROBE (the "one G5 checkpoint" gate, q8
        edition): `live_q8_artifact_device_widen_matches_host_widen`
        (#[ignore]d) loads the REAL english q8 artifact through BOTH
        paths in one process — served answers BYTE-identical
        (probs `[0.02236964, 0.013016701, 0.07665773, 0.3653884,
        0.5225676]`, conf 0.350442, act [1.0, 0.0]); RSS alone:
        **device-widen 1489 MiB vs host-widen 4238 MiB — a 2.85×
        whole-process host-residency cut** (the delta ≈ 2.7 GB ≈ the
        widened f32 the posture no longer holds; Phase 0 derived 1.685
        GB — agrees). Box state: m3, quiet-ish (the standing authority
        + serve processes), AC, 2026-10-02.
      - The standing F16 surface untouched: G5 parity green against the
        in-flight substrate (reflex `laya_riir_parity`, 2/2, 10.7 s),
        `metal_ops_smoke` 11/11, `metal_mps_gemm` 2/2 (+2 ignored
        measurement-only), `packed_forward_equiv` 4/4,
        `prefix_state_coupling` 2/2, `metal_fold_ab` 2/2, lib 48/48 at
        metal features, clippy `-D warnings` at laya-riir /
        laya-riir-metal / +laya-riir-cubecl (lib + tests).
- [x] Kill-switch `LAYA_Q8_HOST_F32=1` restores today's widen-at-load
      (read live per tensor in `Weight2D::from_weights`; the probe USES
      it as the control arm — one env, bit-restoring, residency
      restoring).

### Phase 2 — device-resident Q8 + dequant-fused staging (the real tier) — **EXECUTED 2026-10-02**

- [x] The Q8 device representation: per tensor ONE buffer in the native
      [n, k] block layout (numel/32 × 34 B ≈ numel × 1.0625 — 26.5% of
      F32), permanent-cache keyed like `weight_t_buf` (`weights_q8`,
      first-miss bare UPLOAD — no kernel, no transpose); `warm_weight_2d_q8`
      routes on the posture (resident = upload; kill-switch = the load
      kernel). **Measured: device_allocated 1654.9 → 348.8 MiB = 4.74×**
      (`live_q8_device_bytes_resident_vs_widened`, #[ignore]d, the real
      english map through the agent's own load path minus the tokenizer;
      Phase 0 derived 3.76× for the FULL checkpoint incl. the head — the
      encoder-only probe is consistent).
- [x] The fused-staging kernels: `sgemm_q8` (narrow geometry) /
      `sgemm_xwide_q8` (the single-wave band) / `sgemm_splitk_q8` — the
      SAME tile math as the f32 instances (A staging, MMA, stores,
      ragged-edge route line-for-line), with the B tile staged from the
      RAW blocked bytes: **k-fastest lanes** (`col = idx >> log2(BK)`,
      `kk = idx & (BK-1)` — the INVERSE of the f32 mapping; a warp reads
      32 contiguous quant bytes = one block, the 2-byte scale
      warp-uniform; the shared write strides the odd TBS so all 32 banks
      enumerate). The reduce epilogues are format-agnostic and shared
      verbatim; the T12/split-K folds run per format through one spine
      (`matmul_w_{,accum_,glu_}wb(WBuf)` — `WBuf::F32T | Q8Raw`, ONE body
      per op, never a second transcription).
- [x] MPS OFF under the q8 device-resident posture (option (i)): the
      fused dispatch never consults the MPS arm; the split rule falls
      back to the pre-T13 shape (`q8_split_rule()` — WITH_MPS re-based
      to DEFAULT, other pinned rules carry over; the master switch +
      `LAYA_METAL_SPLITK_MAXTGS` honored). ONE loud disclosure on the
      first unsplit fused dispatch naming the priced alternative and the
      switch back (per-dispatch would be noise at ~100 GEMMs/forward).
      Reach counters: `q8_fused_dispatches()` (the fused arm can never
      pass on the Phase 1 path) + `q8_widen_dispatches()` (the resident
      posture never builds the widened Wᵀ — the mechanism pin).
- [x] Gates: ALL GREEN —
      - THE BIT-IDENTITY BATTERY (`metal_ops_smoke`, the shared-tree
        law): fused-q8 vs the f32 carrier on a `.with_mps(false)`
        instance — byte-identical at 7 shapes (split-K, ragged n, k
        tails, the xwide pick (70,1024,2048), the flat-tail a0w class)
        + the fold entries (accum_q8/glu_q8 vs the unfused device
        streams — a host erf manual reference would differ in the last
        bits by construction) + the kill-switch arm (bit-identical on
        the DEFAULT tree incl. MPS; widen counter moves, fused frozen).
      - `q8_widen_identity.rs`: CPU bit-identity (unchanged), the dense
        Metal pair on the shared tree, first-miss, unfused stream, the
        NEW `q8_resident_matches_device_f32_posture` (Phase 2 vs Phase 1
        in one process: counters pinned, packed hidden states max abs
        **0.000e0** at the test geometry), the load-kernel transpose
        gate (now under the kill-switch, where the kernel actually
        runs).
      - **THE LIVE ARTIFACT PROBE** (`live_q8_artifact_device_widen_matches_host_widen`,
        #[ignore]d, three postures in one process): fused deterministic
        ×2 (byte-identical); Phase-1 tree byte-identical (device-f32 ==
        host-widen, probs/conf/act all exact); the option-(i) dispatch
        delta MEASURED at probs 5.36e-7 / conf 7.15e-7 / act exactly 0
        — two orders under the G5 1e-3 gate, the T13-class
        accumulation-order change (MPS vs our chains on the shapes the
        WITH_MPS/DEFAULT rules disagree about); RSS 1492 (fused) vs
        1881 (Phase 1) vs 4826 MiB (host-widen).
      - **THE DENSE-CELL A/B** (`examples/sgemm_q8_fused_probe`, the
        plan's asked-for A/B): mps vs fused-q8 vs f32 narrow on Phase
        0's exact cells (4 projection shapes × m 106–1700,
        position-balanced 15 rounds, 3 dispatches/round):
        **bit-identical at all 24 cells; fused/narrow 1.065–1.161 (geo
        ~1.15 — the in-staging dequant ALU; the T11 L2 finding says the
        halved B bytes buy nothing where re-reads were never the
        binding cost); fused/mps 1.22–2.07 growing with m (geo 1.42 @
        m106 → 1.96 @ m1700)**. The whole-forward single-question
        paired median: **1.002×** (PROVENANCE: power=AC, load 4.17,
        powermode 2-high) — the dense cells are a small share at the
        serving shape; long-prefill postures would feel the m-scaling
        and should price `LAYA_Q8_DEVICE_F32=1` or re-open option (iii)
        (owner call, the plan's own condition now MET with numbers).
      - G5 parity green against the in-flight substrate (reflex
        `laya_riir_parity`, 2/2, 27.4 s — the F16 surface untouched:
        Dense weights → `matmul_w` → MPS, byte-unchanged, so NO frozen
        cell moves; the per-device re-seat for the q8 posture is the
        no-change proof at this tier and the Phase 3 adoption record).
      - lib 48/48 + every metal suite (mps_gemm, fold_ab, fold_bits,
        splitk_ab, packed_forward_equiv, prefix_state_coupling); clippy
        `-D warnings` at laya-riir / laya-riir-metal /
        +laya-riir-cubecl (lib + tests + examples).
- [x] Kill-switch `LAYA_Q8_DEVICE_F32=1` falls back to Phase 1's
      device-F32 Wᵀ (read live per call in the three q8 entries + warm;
      two switches compose back to Phase 0). Gated in both postures.
- [x] Latency columns re-measured (the dense-cell A/B above + the
      whole-forward paired medians; PROVENANCE quoted; position-balanced
      interleaves both).

      THE PROBE FIXTURE LAW (caught by the probe's own bit assert): the
      f32 carrier arm must stage the DECODED values (`q8_fixture`
      returns bytes + decoded; the first draft transposed the RAW fill
      and the probe refused with 1e-3-scale diffs on every element — a
      fixture bug wearing a kernel-bug costume, the same law the
      identity tests encode as `dense_twin`).

### Phase 3 — adoption + re-seat + the Q4 seam — **EXECUTED 2026-10-02**

- [x] The re-seat: NO bit moved at any seated cell's surface — the F16
      path (Dense → `matmul_w` → MPS) is byte-untouched (G5 parity green
      at both postures against the in-flight substrate, CPU 27.5 s / Metal
      10.7 s; the live probe's B==C arms byte-identical), so the seated
      instinct/serve cells STAND — the re-seat is the no-change proof,
      recorded like 0047's. The q8 posture's own adoption re-seat: the
      live probe re-run at HEAD reproduced Phase 2's numbers exactly
      (fused deterministic ×2 byte-identical; Phase-1 tree byte-identical;
      the option-(i) dispatch delta 5.364e-7 probs / 7.153e-7 conf / act
      exactly 0 — two orders under G5; RSS 1491 / 1880 / 4825 MiB; device
      residency 348.8 vs 1654.9 MiB = 4.74×) — D2b is ADOPTED as the q8
      serving posture on that evidence.
- [x] The `/#sizes` row + note re-measured (reflex-site `78488db`): the
      rethink_encoder row's model bytes are now the ADOPTED q8 artifact
      (447,729,135 B recorded measurement + the three NLEH heads
      7,994,976 B = 455,724,111 B; engine 28,748,684 B unchanged — total
      484,472,795 B, 44% under the F16 row), and the note leads with the
      served posture's MEMORY line (device residency 348.8 MiB / 4.74×,
      RSS 1,492 vs 4,825 MiB) as the headline; the F16 HF-tree number
      (848,195,504 B) stays as the legacy reference. Self-test + live
      regeneration clean (`test_publish_sizes.py` 10/10, `--check` PASS).
- [x] The Q4 seam (the issue's Q4 second — the FORMAT tier only, its
      retention D1-priced separately): the staging decode's format
      constant is the seam, and it landed as
      - **decoder**: `RawQ4` / `WeightData::Q4` / `Weight2D::Q4` +
        `widen_q4_0` + `q4_blocked_len` (weights.rs) — the house Q8 law's
        4-bit shape: block-32, scale `f16(amax/7)`, signed grid [-7, +7]
        (both ±amax on the grid ends up to the f16 scale's own rounding,
        zero exact), GGML nibble order (even element low, odd high),
        18 B/block = 0.5625 B/elt — half the q8 tier. A `-8` nibble
        decodes too (never produced by the house converter, accepted for
        GGML interop).
      - **converter**: `laya-quant4` + `q4_artifact::convert_checkpoint`
        over the SHARED container machinery (`blocked_artifact.rs` —
        serialize / read-back proof / atomic commit / sidecar / variant
        resolution extracted from the q8 lane, so a third format lands
        as a decode + an encode loop, never a second transcription);
        the proof target is the in-memory fake-quant-Q4 grid, byte-exact;
        `LAYA_WEIGHTS_VARIANT=q4` resolves through the same env seam
        (both formats in one table, sidecar-verified, never auto-derived).
      - **the device tier through the format constant**: `metal.rs`'s
        `QFmt` enum (narrow/xwide/splitk/widen_t names per format) +
        `WBuf::QRaw { buf, fmt }` — the spines stay one body per op;
        the q4 MSL kernels are DERIVED from the shipped q8 texts by
        exact token replacement (the decode block + block width 34→18
        + names — the tile math, k-fastest lanes and ragged-edge routes
        are the q8 bytes verbatim; the token table asserts its own hit
        counts at generation, so a drifted token fails loud, never
        generates a q4 kernel that decodes like q8); `q4_widen_t` is
        the load kernel twin; `q4_fused`/`q4_widen` reach counters;
        the kill-switches (`LAYA_Q8_DEVICE_F32`, `LAYA_Q8_HOST_F32`)
        govern the raw-quant FAMILY (both formats), documented.
      - **the bit-identity battery at ITS fidelity** (ALL GREEN):
        `q4_widen_identity` 6/6 (CPU bit-identity; Metal packed-forward
        bit-identity incl. the fold arms + mixed-plan fallback; first
        miss; unfused stream; resident-vs-kill-switch counters + the
        dispatch delta G5-class; the `q4_widen_t` transpose read out
        through an identity matmul vs the host nibble widen at the k-tail
        shape) + the `metal_ops_smoke` q4 arms (the shared-tree battery
        at 7 shapes incl. the xwide pick + the k-tail + the flat-tail
        a0w class, the accum/glu folds vs the unfused streams, and the
        kill-switch arm). lib 56/56 (+8), every metal suite green, the
        q8 battery byte-unchanged (6/6, the shipped surface untouched),
        clippy `-D warnings` at metal / default / no-default / cubecl.
      - **PQ2** rides this same seam as the next format row (the Bonsai
        PQ precedent) — a decode + an encode loop + a QFmt row, no
        kernel transcription.
      - En-route, pre-existing at HEAD: the `sgemm_q8_fused_probe`
        example (Phase 2's own A/B) carried no `[[example]]` row and
        its `#![cfg]`-gated body compiled to a main-less example — the
        bare-clippy posture red on it; the required-features row landed
        with this work (the repo-birth gate discipline).

## Non-goals

- No ANE work (the q8×ANE refusal stands — a Q8 weights artifact feeds
  the Metal/CUDA lanes only).
- No host-CPU forward change (the CPU lane keeps widening; G5's
  reference is untouched).
- No per-request quantization — the artifact IS the posture (D2a's
  never-auto-derive law).

## Task order (the multi-session shape)

Phase 0 (one short session: measure + pin baselines) → Phase 1 (one
session: host residency + load kernel + gates) → Phase 2 (the kernel
sessions: staging rewrite per instance, A/B + bit-identity battery per
landing) → Phase 3 (adoption + re-seat + Q4 seam). Each phase commits
green and independently revertible.
