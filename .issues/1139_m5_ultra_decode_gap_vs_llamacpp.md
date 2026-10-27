# Issue 1139 — riir-infer 1-box GPU decode on M5 Ultra is ~2.2× behind llama.cpp (Bonsai-2-27B PQ2_0)

**Status:** Open (scoreboard pin) — T1–T3 DONE 2026-10-27; Issues 1143 (`85c4da5`) and 1144 (`3147164`) CLOSED 2026-10-27; T4 DONE. Bonsai-2 PQ2_0 1-box decode on idle bmo: **37.24 → 17.57 ms/token (56.9 tok/s)** via Issue 1141 (build posture, 1.74×) + 1142 (dense-pair loads-in-flight, +9.9%) + 1143 (decode norm loads-in-flight, +7.3%) + 1144 (radix-4 FWHT, +3.3%). Residual vs llama.cpp's 16.4: **1.07×** — attributed (T4): 402 extra launches/token; the one rung above the bar is routed to Issue 1145. Close when 1145 closes.

## Why

Plan 623 T0.3's rule: *"If riir-infer's 1-box arm loses, file the gap as its own
issue first — the multi-host claim is meaningless on a losing single-box base."*
The T2.2b GPU-stage runs gave the first riir-infer decode number on an M5 Ultra
(bmo, idle — load ~1.4, no sibling GPU work):

| arm (bmo, Bonsai-2-27B PQ2_0, tg) | tok/s | source |
|---|---|---|
| llama.cpp prism-b10787 Metal solo | **60.9** | [Bench 900](../.benchmarks/900_plan623_t04_llamacpp_rpc_2box.md) |
| mlx pack runtime solo | 58.1 | [Bench 904](../.benchmarks/904_plan623_t04_mlx_arms.md) |
| riir-infer `TernaryGpuStage` full-range (T1.5 range funnel, per-step logits readback + host argmax) | **27.2** | `crates/riir-infer-dist/tests/two_stage_g1_gpu.rs` oracle line, 47 steps |
| riir-infer 2-stage GPU⇄GPU over loopback TCP, split 32 | 26.0 | same test, same run |

So the pipeline costs ~4% on top of a 1-box base that is itself ~0.45× the
incumbents. The 2-box lane cannot beat a 1-box incumbent until this closes.
The m3 decode campaign measures the same class (Issue 1134's 38.27 ms/token ≈
26 tok/s) — the per-rung work there (1127/1130–1137) is the lever family; this
issue is the M5-Ultra scoreboard pin those rungs are scored against.

## Tasks

- [x] **T1 — confirm on the production funnel. DONE 2026-10-27:** instrument
  `crates/riir-infer-dist/tests/decode_readback_probe_gpu.rs` (1-box `forward_token`,
  `--release`, full model, ctx 72, 8 warm-up + 64 timed steps). **bmo, AC power, load
  1.2–1.7, no sibling GPU work, 3 runs: 36.67 / 43.19 / 37.37 ms per token = 27.27 /
  23.15 / 26.76 tok/s** — the same ~27 as the stage seam, so the gap is the kernel
  stack (T2), not the stage path. Stage-only costs PRICED in the same run: idle logits
  readback 0.225–0.284 ms + host argmax over 248,320 logits 0.114–0.117 ms = **0.92–0.94%
  of the token** — the ceiling for an on-GPU argmax, BEFORE its own kernel cost; under
  the 1.5% G2 bar, so Plan 623 T2.2c(2)+(3) are DECLINED on this measurement (the
  production funnel pays the same readback, so the same verdict covers it).
  *Original brief:* re-measure with the 1-box
  `forward_token` decode loop (not the stage seam) on bmo, `--release`, box state
  recorded per the G2 law; rule out the stage-path-only costs (1 MB logits
  download + host argmax per step, `set_hidden_state`'s per-call device alloc).
  If the production funnel is also ~27 tok/s the gap is the kernel stack; if it is
  materially faster, the stage path's readback is a T2.2c item, not this issue.
- [x] **T2 part 1 — the posture factor. DONE 2026-10-27 (filed + closed as Issue 1141, this is
  its record).** T1's "it is the kernel stack" was half wrong: `riir-infer-gpu` has NO
  default features, and the production decode posture lived only in riir-ai's
  `riir-gpu` defaults (+ its vendored `cubecl-wgpu` fork carrying the uniform-content
  cache) — so every Metal build in THIS workspace ran the bare kernels on upstream
  cubecl-wgpu. Interleaved on idle bmo (AC, load 1.0–1.5), real Bonsai-2 27B,
  `decode_readback_probe_gpu` (ms/token): **A shipped 37.24 / 43.54 / 38.87 · B +rowpar,
  parallel, simdgroup, delta_routing, params_handle_cache (upstream wgpu) 31.49–31.73 ·
  C B − params cache + the fork 33.41–33.91 · D B + the fork 21.39–21.48 (46.7 tok/s)**.
  The two caches only pay COMPOSED on the M5 Ultra (C alone is a loss vs B — the
  opposite of the M3 call that kept `params_handle_cache` opt-in in riir-ai, Bench 771).
  riir-ai's own harness (`bench_763`, both caches forced ON, Bonsai-2) reads 21.93 —
  the same posture. **G1:** the caches are bit-identical (D vs D with
  `RIIR_UNIFORM_CACHE=0 RIIR_PARAMS_CACHE=0`: logits-stream digest `d06bffced92427cb`
  both); A→D changes logit BITS (rowpar/simdgroup reduction order — riir-ai's shipping
  numerics) with the greedy stream unchanged (last token 279 after 72 steps); dist GPU
  G1 tests re-PASS under D (splits 1/32/63 bit-identical, prefill anchor ==
  production `prefill()`). **Fix:** the fork's canonical home moved to
  `riir-infer/vendor/cubecl-wgpu-0.11.0-pre.2` (the wgpu-hal/cubecl-common precedent,
  Issue 1046) + the posture promoted into the dist `gpu` feature. riir-ai repoint +
  the M5U params-cache re-gate are filed in riir-ai.
- [x] **T2 part 2 — attribute the residual. DONE 2026-10-27.** riir-ai `bench_763` stage
  census at the composed posture (riir-gpu defaults + params cache, bmo idle, same binary,
  both artifacts): **Bonsai v1 Q2_0 16.77 ms/token (852 launches/token) vs Bonsai-2 PQ2_0
  21.92 (1254)** — v1 is at llama.cpp parity, so the whole residual is Bonsai-2-specific.
  The per-stage delta (Phase C, dispatch-only + drain, Bonsai-2 − v1): `gdn_proj_dense`
  **+2.49 ms** (48 launches at ~52 µs — the folded f32 a/b pair; absent in v1),
  rotations `gdn_out_rot` + `ffn_rotate` **+1.19 ms** (96 launches at ~12 µs, ≈0 in v1),
  input+mid norms **+0.92 ms** (same 128 launches, heavier per launch); everything else
  within ±0.2. The dense pair's ~52 µs/launch was the outlier: 96 planes over K=5120,
  ~160 strided loads per lane, one in flight at a time = load-LATENCY bound (2 MB/launch
  at ~38 GB/s). **→ Issue 1142 (filed + CLOSED, this is its record):**
  `gemv_plane_f32_pair_mlp8` issues 8 loads per lane before folding them in the SAME k
  order — bit-identical by construction. G1: unit `test_dense_pair_bit_identity` 18/18
  EXACT (both kernels × {48×5120, 5×100, 7×5000 main+tail} × wg {256, 64, 32}); real
  Bonsai-2 logits-stream digest `d06bffced92427cb` unchanged in every arm. G2 (bmo,
  interleaved, ms/token): depth 4 → 21.40 / 19.79 (+8.3%, 3/3), 8 vs 4 → 19.46 / 19.81
  (+1.7%, 3/3), 16 vs 8 → 19.30 / 19.45 (+0.75% — sub-bar, not taken); final default
  vs kill-switch 19.45 / 21.38 (**+9.9%**, 2/2). **PROMOTED DEFAULT-ON**, kill-switch
  `RIIR_DENSE_PAIR_MLP=0` (shared `dense_pair_env_semantics`), harness toggle
  `set_dense_pair_mlp`.
- [x] **T3 — route. DONE 2026-10-27.** The two remaining Bonsai-2-only classes are
  floor-class small-grid kernels (the norms 1134 already tried and failed to fuse; it
  measured an e2e LOSS on Metal). They are routed to **Issue 1143**: apply the 1142
  loads-in-flight pattern (bit-identical, no fusion) to the rotation/norm/small-grid
  GEMV kernels. No other open rung covers them; 1130/1133–1137 are closed.
- [x] **Issue 1143 record (filed + CLOSED 2026-10-27, `85c4da5`) — loads-in-flight on the
  remaining small-grid classes.** T1 read the kernels:
  - **Decode norm** (`residual_add_rmsnorm_f32`, 127 of 128 norm launches, both models):
    ONE 256-thread workgroup over dim 5120, so each lane walks 20 strided pairs, one
    DRAM round trip at a time. This is the 1142 shape, and the strongest candidate.
  - **FWHT forward and copy kernels:** 5–17 cubes, 4 fill loads per lane, then 10
    `sync_cube` butterfly stages.
  - **`gdn_out_gemv`:** the shared ternary GEMV over 5120 rows. Large grid, already at
    v1 parity, so not a candidate.

  Results:
  - **Norm:** `residual_add_rmsnorm_f32_mlp4` hoists 4 iterations' loads ahead of the
    unchanged fold. G1: unit `test_residual_add_rmsnorm_mlp4_bit_identity` to_bits EXACT
    (5120, ragged 5669, 768), and the real Bonsai-2 digest `d06bffced92427cb` is unchanged
    in every arm. G2 (idle bmo, interleaved): **19.52 → 18.17 ms/token (+7.3%, 3/3)**.
    Depth 8 is 0.4% slower than 4 (3/3), so 4 stays. **PROMOTED DEFAULT-ON**;
    kill-switch `RIIR_NORM_MLP=0`, harness toggle `set_norm_mlp`.
  - **FWHT fill hoist:** bit-identical (unit test plus the CPU-oracle tests), but only
    **+0.9% (18.21 → 18.04, 3/3)**. That is sub-bar, so it was NOT shipped and the code
    was dropped. The rotations are bound by barrier stages, not by loads, which is routed
    to **Issue 1144**.
  - Both 1143 arms off read 19.59 ms/token, consistent with the 1142 baseline.
- [x] **Issue 1144 record (filed + CLOSED 2026-10-27, `3147164`) — FWHT barrier stages.**
  - **Change:** `fwht_rotate_forward_f32_r4` and `fwht_forward_copy_f32_r4` (hblock 1024)
    run two butterfly stages per pass in registers over a per-thread quad. The lowest
    pair reads straight from global, the highest writes straight to global, and the
    middle three ping-pong between two smem halves. That is **4 barriers instead of 11**.
  - **Trap:** the first build was 1–2 ULP off. Metal compiles MSL with fast math, so a
    product feeding the next stage's add in a register fuses into an fma; the incumbent
    escapes this only because every stage round-trips smem. Fencing each product with
    `select(live, p, 0)` on a runtime-true predicate restores EXACT output.
  - **G1:** `fwht_radix4_matches_incumbent_bitexact` (both kernels, 5120/6144/17408, a
    3-row batch, magnitudes 1e4 and 1e-3), the CPU-oracle tests through the launcher, and
    the real digest `d06bffced92427cb` unchanged.
  - **G2** (idle bmo, interleaved): **18.15 → 17.57 ms/token (+3.3%, 3/3)**.
    **PROMOTED DEFAULT-ON**; kill-switch `RIIR_FWHT_R4=0`, harness toggle `set_fwht_r4`.
  - **Not ported:** the fused default-OFF twins (`gdn_permute_rotate_f32`,
    `swiglu_rotate_fused_f32`, `residual_add_rmsnorm_rotate_f32`) and the inverse
    (1 launch/token) keep the 10-stage loop.
- [x] **T4 — re-attribute the 1.07× residual. DONE 2026-10-27.**
  - **Census:** riir-ai `bench_763` (built from riir-ai develop with riir-infer
    `3147164`, riir-gpu defaults), idle bmo.
    - Bonsai-2: synced **18.03 ms/token**; dispatch-only 15.69 at 1254 launches/token.
    - v1 Q2_0: synced **15.40**; dispatch-only 13.59 at 852 launches/token.
    - Gap: **2.10 ms/token** over **402 extra launches**.
  - **Bonsai-2-only classes above the 1.5% bar** (Phase C Δ vs ALL, Bonsai-2 / v1):
    - norm staging: `input_norm` 0.80/0.46 + `mid_norm` 0.81/0.44 = **+0.71**
    - `gdn_out_rot` 0.54/0.00
    - `gdn_proj_dense` 0.54/0.00 (post-1142)
    - `ffn_rotate` 0.32/0.00
  - **Existing fusion levers re-measured** on `decode_readback_probe_gpu`, interleaved,
    digest unchanged in every arm:
    - `RIIR_GDN_PERMUTE_ROTATE_FUSED=1`: +0.5–0.8%.
    - `RIIR_SWIGLU_ROTATE_FUSED=1`: +0–0.5%.
    - Both together: −0.2%.
    - Radix-4 ports of both fused kernels: G1 EXACT, but still only +0.5% each, so they
      were NOT shipped.
    - `RIIR_NORM_ROT_FUSED=1` (1134): **−20%** (17.55 → 21.19, 3/3).
    - All sub-bar or negative; defaults unchanged.
  - **Routed:** the norm-staging pool → **Issue 1145** (multi-cube redundant-reduction
    norm + radix-4 staging, which needs a double-buffered `x`; ceiling ≈0.71 ms ≈ 4%).
  - **Not routed:** `gdn_out_rot`, `gdn_proj_dense` and `ffn_rotate` each sit at about
    the launch floor (≈5–11 µs × 48–96 launches); their fusion levers are measured
    sub-bar above.
- [x] ~~T2 (original brief)~~ superseded by T2 parts 1+2 above — Per-dispatch census on the M5 Ultra (the Issue-1132
  instrument) vs the llama.cpp Metal graph for the same token: launches/token,
  bytes moved, the top-5 kernels by wall. Name the dominant class (launch floor vs
  bandwidth vs ALU) before proposing rungs.
- [x] ~~T3 (original brief)~~ superseded — **route.** Fold the attributed levers into the existing rung issues
  where they already exist; file new rungs only for unclaimed classes.

## Non-goals

No multi-host work here (Plan 623 owns it). No gemma-2-2b cell (retired model).
