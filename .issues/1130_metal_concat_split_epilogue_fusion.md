# Issue 1130 — Metal concat-epilogue rung: the folded GDN qkv|z Split2 folded INTO the ternary concat GEMV's epilogue (the Bench-1007 named probe — unclaimed)

<<<<<<< HEAD
**Status:** IMPLEMENTATION BANKED + G1 PASS — 2026-10-26 ~23:1x +07: the complete rung landed from Session `zed-amo-1130-concatsplit` (a race window vs the lease claim below — the local fetch predating the push; donated, see the handoff note at the bottom). **The G2 e2e A/B + records belong to the lease holder.**
**Lease:** none as of filing — the i1127 bundle (attention-chain T1, FWHT T3) is the CUDA lane, disjoint.
**Claimed:** zed-m3-1130-concat-split, 1793030628, expires 2026-10-28T16:03:48Z
=======
**Status:** CLAIMED — 2026-10-26 ~23:4x +07, Session `zed-amo-1130-concatsplit` (implementation + G1 + the G2 e2e A/B this session; box load 2.6 falling — quiet window open).
**Claimed:** `zed-amo-1130-concatsplit` (Metal/CubeCL lane; the i1127 CUDA bundle is disjoint).
>>>>>>> ee96379 (feat(1130): Metal concat-epilogue split rung — the folded GDN qkv|z Split2 folded INTO the ternary concat GEMV's write epilogue (opt-in RIIR_GDN_CONCAT_SPLIT_FUSED))

## The trigger (measured, not assumed)

Bench 1007's post-promotion census (riir-ai, 2026-10-26 ~22:4x): the gdn_proj concat leg (`no-gdn_proj_concat` bit 9) reads **1.77 ms/token** = the 16384×5120 ternary concat GEMV **+ the Split2CubeCL region copy** (input_proj_out → qkv [conv_dim 10240] + z_buf [z_dim 6144]). The copy moves 64 KB/layer — pure movement, one launch per GDN layer (32/token). At the Metal per-dispatch floor class measured this session (~50-100 µs/launch in-pipeline for tiny kernels — the dense-pair rung's own evidence), the split is plausibly **0.6-1.6 ms/token of launch floor + a redundant 128 KB read+write pass**.

The dense-pair precedent (same session, promoted): ONE-launch fusion of movement into a neighboring GEMV, bit-identical by construction, measured +9.9/+9.1% e2e — the mechanism class is proven ON THIS LANE.

## The ask

Extend the folded-branch concat GEMV dispatch (ternary_deltanet_gpu_forward.rs, `gdn_proj` region, bit-9 leg) so `gemv_ternary_plane_rowtiled8`-class kernels write output rows DIRECTLY into the two consumer buffers: rows `[0, conv_dim)` → `qkv`, rows `[conv_dim, conv_dim + z_dim)` → `z_buf` — a row-range epilogue redirect (two output handles + one boundary comparand per plane), eliminating BOTH the `input_proj_out` staging buffer pass AND the Split2 launch. The unfolded branch (single full concat, no split) is untouched.

- **Bit-identical by construction**: same kernel math, same row→plane mapping, same values — only the destination address changes (movement-only, the Issue-1113-lever-1 discipline). G1: to_bits equality vs the incumbent chain (GEMV→staging→Split2), e2e FNV/argmax anchors UNCHANGED → dispatch-only rung, promotable on perf with EXACT anchors.
- Env gates (opt-in first, the 1129 shape): `RIIR_GDN_CONCAT_SPLIT_FUSED` + `RIIR_GDN_CONCAT_WG` if the sweep wants it (the 1129 micro read says wg is dead for small grids — pin 256).
- The census bits keep working: bit 9 (`gdn_proj_concat`) must strip BOTH the GEMV and the (gone) split either way.

## GOAT gates

- **G1**: fused-vs-chain bit-identity (to_bits) at the real shape (16384 = 10240+6144 rows × 5120, folded input) + a ragged boundary case; wg {256,64}.
- **G2**: decode e2e A/B on the folded 27B at the league posture (one binary, env flip, counter vacuous guard — the bench_1129 pattern); promote-if ≥1.5% with anchors EXACT. Expected: 0.6-1.6 ms/token → +2.5-7% class.
- **G3**: prefill neutrality by construction (the prefill path does not dispatch this leg).
- **G4**: no allocations on launch paths; the demote hatch restores the incumbent chain.

## Non-goals / scope guards

- NOT the ternary GEMV's own bandwidth (118 MB weights at ~79 GB/s effective — a different, bigger lever; the rowtiled8 access-pattern study, NOT this movement fusion).
- NOT the attention chain (i1127 T1: census-closed, ~1.0% of token) or the CUDA lane (disjoint).
- NOT the FWHT geometry (i1127 T3, claimed, in flight).
- NOT the Metal int8 consumer (T5-class, owned in the riir-ai 1113 lane).

**Refs:** riir-ai `.benchmarks/1007_issue1129_dense_pair_rung.md` (the per-dispatch floor evidence + this probe named); riir-infer issue 1129 (the promoted precedent + the launch-floor micro read); riir-infer issue 1127 (the sibling bundle — disjoint lanes verified); riir-ai issue 1113 (the defense; post-promotion re-cell 0.685×/0.677×).

## Handoff note (2026-10-26 ~23:2x +07 — Session `zed-amo-1130-concatsplit`, the race-resolution donation)

Race window: this implementation was built ~6 minutes after the lease claim above was filed
(this session's issue read at ~22:57 still showed `Claimed: none`; the claim push landed
23:04:07 +07 and the local fetch predating it missed that). Per the lease law + the Bench-649
GPU exclusivity, the rung's EXECUTION is yielded to the lease holder — the code is donated so
no work is duplicated:

- The rung is COMPLETE and G1-verified in-tree:
  - `gemv_ternary_plane_rowtiled8_f16scale_split` (gemv_ternary_scale_ab_cubecl.rs) — verbatim
    f16scale kernel math, redirected write epilogue (rows `[0, split_row)` → `out_a`/qkv, rows
    `[split_row, m)` → `out_b`/z_buf at `row - split_row`); branch hoisted to the epilogue,
    zero cost in the dot loop.
  - `GemvTernaryCubeCL::launch_split` — f16 lane ONLY (the shipping decode dispatch), occupancy-rule
    wg default (the same geometry `launch()` picks at this m → the A/B is like-for-like), returns
    `false` → exact staging-chain fallback; `RIIR_GDN_CONCAT_SPLIT_FUSED` opt-in +
    `RIIR_GDN_CONCAT_WG` sweep knob + the vacuous-guard counter.
  - Folded `gdn_proj` wiring: `launch_split` armed gates the Split2 off; bit 9 strips the GEMV
    and the (then-gated) split either way.
  - G1 PASS (`test_concat_split_bit_identity`): to_bits EXACT vs the `launch()`+`Split2CubeCL`
    chain at the real folded shape (16384 = 10240+6144 × 5120) + boundary-straddling (24×128,
    split 15) + doubly-ragged (13×100, split 5), wg {256,128,64} ×3 repeats, NaN-sentinel full
    coverage, counter +1/launch, demote hatch OFF→false. Verify:
    `cargo test -p riir-infer-gpu --features "cubecl_runtime ternary_gemv" --release test_concat_split_bit_identity -- --nocapture`
- The e2e A/B harness is banked at riir-ai `445cc4e5a`:
  `crates/riir-gpu/tests/bench_1130_concat_split_e2e_ab.rs` (ABBA, one model load, the
  bench_1129 pattern, composed-caches regime; the facade re-exports
  `set_concat_split_fused`/`concat_split_launch_count`).
- One deliberate divergence from this issue's text: the harness does NOT pin the wg — both
  arms ride the Issue-1114 occupancy rule (the incumbent dispatches wg 128 at m=16384), so the
  measured delta is the split elimination alone; `RIIR_GDN_CONCAT_WG` remains the sweep knob.
  The 1129 micro read's "pin 256" advice applied to THAT rung's geometry.
- Holder's run: verify G1, then the A/B per the harness header (Bench-649 exclusivity; the
  census expectation is +2.5-7% from the 1.77 ms/token leg). On GOAT (both rounds ≥1.5%),
  promotion flips `concat_split_fused_enabled` to the demote-only default + a league re-cell.
