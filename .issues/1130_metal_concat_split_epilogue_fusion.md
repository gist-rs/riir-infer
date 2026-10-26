# Issue 1130 — Metal concat-epilogue rung: the folded GDN qkv|z Split2 folded INTO the ternary concat GEMV's epilogue (the Bench-1007 named probe — unclaimed)

**Status:** OPEN — filed 2026-10-26 ~23:0x +07 from the Bench-1007 defense continuation (issue 1113 lane, Metal/CubeCL side).
**Claimed:** none — unclaimed; the i1127 bundle (attention-chain T1, FWHT T3) is the CUDA lane, disjoint.

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
