# Issue 1129 — Metal dense-pair rung: the folded GDN a/b f32 GEMVs fused into ONE launch + the wg-occupancy geometry the dense path never rode (the Bench-1006 census cut)

**Status:** OPEN — filed 2026-10-26 ~21:4x +07 from the Bench-1006 census follow-up (issue 1113 defense lane).
**Claimed:** zed-amo-i1129-pair, 1793024895, expires 2026-10-28T14:28:15+00:00 (same session as the 1113-gemvrung lease; the two scopes are one lane)

**⟲ 2026-10-26 ~22:5x +07 zed-amo: RUNG COMPLETE + PROMOTED DEFAULT-ON — G1 bit-identity ×6, G2 e2e GOAT ×2 (+9.88/+9.07% decode ABBA, vacuous guard exact), the wg rung measured DEAD (kept as a knob).** [riir-ai Bench 1007](../../riir-ai/.benchmarks/1007_issue1129_dense_pair_rung.md): the micro sweep read the pair at a per-dispatch floor (~105-120 µs, FLAT across wg {256,128,64,32}) — geometry refuted, fusion confirmed (−43% pair cost at the micro level); the e2e A/B (riir-ai `bench_1129_dense_pair_e2e_ab`, in-process ABBA, one model load) read OFF 26.30→ON 23.71 / OFF 26.19→ON 23.81 ms/token with G1 logits bit-identical across 4 arm pairs (24 tok × 248320 floats) and the counter exact (OFF +0/+0, ON +6000/+6000). Post-promotion: ALL 23.51→21.29 ms/token combined, the census dense-leg Δ 4.73→2.56 ms (halved — the prediction). `RIIR_GDN_DENSE_PAIR_FUSED` default flipped to demote-only ON (env-semantics unit pins updated: unset=ON, any non-1 set value demotes). **Promotion adjudication rides the reviewer verdict on this session's thread** (the cd03cea7 posture); the league decode re-cell (~+7-9% expected, tok/s re-pins only, anchors UNCHANGED) rides the next quiet window. Remaining scope: the `cuda_backend` CubeCL arm (the Issue-1115-shaped A/B — the scored 4090 lane is cudarc, untouched).

## The trigger (measured, not assumed)

Bench-1006 census refinement (this session, 2-bit gdn_proj split, measurement-only, mask=ALL bit-identical): the gdn_proj region's 6.79 ms/token is NOT the ternary concat — the **dense f32 escape-set a/b pair carries 4.73 ms = 20.1% of the ENTIRE decode token** (no-gdn_proj_dense row), vs 1.77 ms for the 16384×5120 ternary concat + split (no-gdn_proj_concat row). Legs partition the region (1.77 + 4.73 = 6.50 ≈ 6.79).

The shape: m = n_v_heads = 48 rows × n_embd 5120, f32 weights ≈ 983 KB per GEMV, 64 launches/token, ≈74 µs per launch — ~13 GB/s effective. The geometry: `gemv_plane_f32` at wg 256 → `ceil(48/8)` = **6 workgroups** on a ~64-core GPU. This is the exact small-m pathology the Issue-1114 occupancy rule fixed for the ternary path (m=5120: 80→320 wgs) — and the dense path never rode it (Bench-1006 named cut (b), confirmed live). Bench-996's "the sweep did not test it, latency-floor tiny shapes" assumption is now REFUTED by the census: the tiny shapes are 20% of the token.

The mechanism class is proven upstream-lane: the CUDA folded graph fuses these same dense a/b GEMVs into its input kernel (K2 `gdn_input_rotabq`, Bench-940 promoted), and the 4→3 launch-bound fusion rung measured direction-positive +0.6–1.0% e2e on CUDA (Bench 899). The Metal/CubeCL lane is the one still paying 2 launches per layer for the pair.

## The ask

1. `gemv_plane_f32_pair` (riir-infer-gpu `gemv_cubecl.rs`): ONE launch computing both a and b — uniform per-plane split (a-planes / b-planes), per-row math VERBATIM `gemv_plane_f32` (lane-stride-32 loop + `plane_sum`) → **bit-identical by construction** (same per-row reduction order as the incumbent pair; the `gemv_fma_contract` cfg_attr mirrored). e2e anchors unchanged → dispatch-only rung, promotable on perf with EXACT anchors.
2. wg-parameterized dispatch (`launch_pair` + `launch_plane_wg`): at m_total=96, wg 64 → 48 wgs (8× today's 6+6); sweep {256,128,64,32} per-launch.
3. Env gates (opt-in, default OFF = incumbent): `RIIR_GDN_DENSE_PAIR_FUSED` (1/on/true) + `RIIR_GDN_DENSE_PAIR_WG` (u32, default 256 until swept). GOAT gates below; promotion to default rides the league re-cell law.

## GOAT gates

- **G1**: fused pair vs the two singles **bit-identical** (f32 to_bits) at the real shape (48×5120) + ragged (m_a=5, m_b=3, n=100) × wg {256,64,32}.
- **G2**: decode e2e A/B on the real folded 27B at the league posture (one binary, env flips the dispatch, counter as vacuous guard) — promote-if ≥1.5% e2e (the Bench-995 bar) with FNV/argmax anchors EXACT both arms.
- **G3**: prefill neutrality by construction — the prefill path does not dispatch these GEMVs; untouched.
- **G4**: launcher-only additions, zero allocations in the launch path.

## Non-goals / scope guards

- NOT the ternary concat (1.77 ms — the smaller leg; the 74 µs-class per-launch study here may inform it later, separate rung).
- NOT the CUDA lane (its K2 already fuses the pair).
- NOT atomics/split-K partials (non-deterministic accumulation order — violates the FNV pin law).
- NOT the recurrence/conv chain (1126's lane, measured negative at rung-1 on CUDA; the 4→3 remainder is theirs).

**Refs:** riir-ai `.benchmarks/1006_issue1113_gdn_subclass_census.md` + the census-refinement run (this session, lands with the record); riir-ai issue 1113 (the defense); riir-infer issue 1126 (the CUDA twin lane + the 4→3 precedent); Issue 1114 (the occupancy rule the dense path never rode).
