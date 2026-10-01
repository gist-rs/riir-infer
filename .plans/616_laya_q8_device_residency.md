# Plan 616 — Laya D2b: the Q8 residency tiers (host → device) with dequant-fused staging

**Status:** PLANNED — scoped from the Lane C session (instinct issue 018's
amended order: C closed negative → D2b next). Not started; Phase 0 is the
first task. Plan of record: `../riir-instinct/.issues/018_rethink_encoder_lean_goat.md`
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

The Q8 bytes are [n, k] row-major with blocks along k (32 consecutive
k-elements of one output row share one scale). Today's staging consumes
Wᵀ [k, n] (`weight_t_buf` transposes at load; Issue-020 T4's coalesced
`b_cs == 1` branch). A host-side Q8 TRANSPOSE (re-blocking Wᵀ) would
requantize — `d'·q' ≠ d·q` in general — and break the bit-identity claim
at the root. So the fused staging must read the NATIVE [n, k] layout and
dequant-then-transpose in-kernel: for a staged [BK, BN] B-tile, fixed-n
columns share 1–2 contiguous blocks (BK=64 aligned → 2 scales + 64 i8 per
column ≈ 68 contiguous bytes), consecutive threads take consecutive n —
a different staging shape with its own coalescing analysis, not a flag
flip. (The kernel_opt corpus' staged-B family — dequant-once-per-macrotile,
in-place-transpose-at-staging — is the pattern library; B186's regime
gates apply.)

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

### Phase 1 — host residency (cheap first rung, no kernel work)

- [ ] Under `LAYA_WEIGHTS_VARIANT=q8` + Metal: skip the host F32
      materialization for the 126 Q8_0 GEMM tensors; keep the raw Q8 map
      (the loader already parses it — retain instead of discard) and
      dequantize-transpose on the DEVICE at warm time (`weight_t_buf`'s
      upload becomes a load kernel: read Q8 [n,k], write F32 Wᵀ [k,n],
      then the Q8 buffer drops). Host ≈ 1.68 GB → ~0.5 GB; device
      unchanged; MPS unchanged; bit-identity trivial (the device kernel
      computes the same `d·q` values — f16→f32 exact — the host widen
      did, and the transpose is value-preserving).
- [ ] Gates: byte-identical forward vs today at q8 (`metal_ops_smoke` arms
      + `packed_forward_equiv` + one G5 checkpoint); the CPU lane and the
      G5 reference keep widening on demand (the widening stays available —
      the resident host map is what skips it).
- [ ] Kill-switch `LAYA_Q8_HOST_F32=1` restores today's widen-at-load
      (one env, bit-restoring).

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
