# Plan 616 — Laya D2b: the Q8 residency tiers (host → device) with dequant-fused staging

**Status:** PHASE 1 DONE (2026-10-02, host residency + the device load
kernel — all gates green, the live artifact probe byte-identical, RSS
1489 vs 4238 MiB); Phase 0 done; Phase 2 is the next task. Plan of
record: `../riir-instinct/.issues/018_rethink_encoder_lean_goat.md`
§Lane D2b ("device-resident Q8 buffers + dequant-fused kernels — the real
device-memory tier; per-device determinism re-seats apply THERE").

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

### Phase 2 — device-resident Q8 + dequant-fused staging (the real tier)

- [ ] The Q8 device representation: per tensor ONE buffer in the native
      [n, k] block layout (numel/32 × 34 B ≈ numel × 1.0625 — 26.5% of
      F32), permanent-cache keyed like `weight_t_buf`; `Encoder::warm`
      gains the Q8 warm path keyed on the agent's posture.
- [ ] The fused-staging kernels: `sgemm`, `sgemm_xwide`, `sgemm_splitk`
      gain a Q8-B instance (compile-time format constant — the Q4/PQ2
      forward seam; the decode table lives behind it) that stages from
      native-layout Q8 with in-staging dequant+transpose. The A-side, the
      MMA, the epilogues, and the T12/split-K folds are untouched — the
      staged tile VALUES are identical, so the outputs are bit-identical
      by construction.
- [ ] MPS OFF under the q8 device-resident posture (option (i)): the
      dispatch arm refuses with a loud one-line disclosure naming the
      priced alternative; `SplitRule` falls back to the pre-T13 rule.
      The Phase 0 baseline vs the fused-sgemm A/B prices the cost; (iii)
      re-prices only if it bites (owner call).
- [ ] Gates: the bit-identity battery (byte-identical vs F16 AND vs
      today's q8, both A/B'd in one session — the 0046 fresh-F16-witness
      pattern), G5 parity at the q8 posture (Metal AND CPU reference),
      the alloc-free G4 arms unchanged, and the **per-device determinism
      re-seat** — THIS is the gate Lane D2b exists to run: the frozen
      reads re-seat per device only if bits moved anywhere (the design
      claim is they did not; the gates prove it, and a moved bit is a
      loud FAIL, never an absorbed one).
- [ ] Kill-switch `LAYA_Q8_DEVICE_F32=1` falls back to Phase 1's
      device-F32 Wᵀ (two switches compose back to today).
- [ ] Latency columns re-measured on the fit box (the Issue-021 law —
      `bench_preflight.sh` PROVENANCE quoted); the win claim is MEMORY
      (measured in Phase 0's units); any latency claim rides the same
      A/B discipline as every lane here (paired, position-balanced,
      quiet-box).

### Phase 3 — adoption + re-seat + the Q4 seam

- [ ] If (and only if) any gate moved bits: the instinct/serve cells
      re-seat per device at the new numerics (the issue's "never
      grandfathered" law); if no bit moved: the re-seat is the no-change
      proof, recorded like 0047's.
- [ ] The `/#sizes` row + its note re-measure (the served posture's
      memory line becomes the headline number).
- [ ] The Q4/PQ2 second (the issue's): the staging decode's format
      constant is the seam — a new format lands as a decoder + an
      artifact converter + the same bit-identity battery at ITS fidelity
      (Q4 retention is D1-priced separately; PQ2 rides the Bonsai PQ
      precedent).

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
