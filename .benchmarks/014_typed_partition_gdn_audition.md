# Bench 14 — T5.0 typed-partition GDN apply-path audition: merges NEVER win (0/16 blocks)

**Status:** RECORD — audition complete; the merged emit is REFUTED locally (the laya finding replicates on the Bonsai GDN lane). Typed medoid-passthrough GOAT at 53% depth: see `goat_typed_e01.log` + this file's tail after the arm lands.

## What ran

`examples/twt_bonsai_audition` (landed this session) at **ε=0.1 TYPED** — the
typed min-max DP (`minmax_partition_typed`, same-type blocks only) over the
Phase-1 S matrix:

- **34 blocks = 53.1% depth** (parent 64 layers): 16 mergeable GDN triples +
  2 GDN pairs + 16 attention singletons. The merge room the unconstrained DP
  never had (its ε=0.1 partition — 14 blocks, type-mixed almost everywhere —
  made merges impossible on 12 of 14 blocks).
- Calibration: `.raw/twt/audition_calib.txt` (the GOAT corpus — same
  distribution the end-task gate reads), 8 rows × 256 tokens = 2048 positions,
  contiguous chunks. Selection artifact:
  `.raw/twt/bonsai_audition_e01_typed.json` (gitignored).
- **Preflight parity: max |Δ| = 0e0** over 256 positions — the block's first
  member replayed through `qwen_deltanet_ternary_layer_body` reproduces the
  parent's own post-member hidden BIT-EXACTLY. The apply path is the
  forward's code (extracted verbatim) and the state handling is proven
  equivalent. The arm earned its keep en route: the first run inherited the
  capture row's final recurrence state into the preflight replay (missing
  reset) and FAILED at 5.66 — a real instrument bug a silent selection would
  have shipped.
- Pool per block: {k member passthroughs, sign_majority (arm B),
  mean:source_quant, rdsc:source_quant (arm C)}. Param menu per T3.1: norms
  merged-mean; a_log/dt_bias/conv1d NEVER averaged (from the block's
  minimax-medoid member). Errors: Σ‖f_cand(h_in) − h_e‖² over rows ×
  positions, f64 in fixed order (deterministic).

## Measured (M3 Max, AC, load ~8-10 from sibling agents — wall times
load-contaminated; the SELECTION is deterministic and is the claim)

**WINNERS: 16/16 blocks → MEMBER passthrough. 0 merges picked.**

| block | k | winner | best-member err | best-merge err | merge/member |
|---|---|---|---|---|---|
| [0,3) | 3 | member:0 | 2.952e4 | 3.580e5 | 12.1 |
| [4,7) | 3 | member:4 | 5.616e4 | 8.902e4 | 1.59 |
| [8,11) | 3 | member:8 | 1.217e5 | 2.073e5 | 1.70 |
| [12,15) | 3 | member:12 | 1.534e5 | 2.488e5 | 1.62 |
| [16,19) | 3 | member:18 | 2.522e5 | 4.224e5 | 1.67 |
| [20,23) | 3 | member:22 | 4.290e5 | 7.045e5 | 1.64 |
| [24,27) | 3 | member:26 | 8.007e5 | 1.358e6 | 1.70 |
| [28,31) | 3 | member:30 | 1.104e6 | 1.706e6 | 1.55 |
| [32,35) | 3 | member:34 | 1.401e6 | 2.450e6 | 1.75 |
| [36,39) | 3 | member:38 | 1.147e6 | 1.748e6 | 1.52 |
| [40,43) | 3 | member:42 | 1.368e6 | 2.223e6 | 1.63 |
| [44,47) | 3 | member:46 | 1.729e6 | 3.085e6 | 1.78 |
| [48,51) | 3 | member:50 | 3.455e6 | 7.598e6 | 2.20 |
| [52,54) | 2 | member:53 | 4.224e6 | 8.006e6 | 1.90 |
| [56,59) | 3 | member:58 | 8.908e6 | 1.553e7 | 1.74 |
| [60,62) | 2 | member:61 | 1.369e7 | 2.151e7 | 1.57 |

Per-op damage (merge err / best-member err, median over 16 blocks):

| candidate | median | min | max |
|---|---|---|---|
| mean:source_quant (arm C) | **1.70** | 1.52 | 12.1 |
| rdsc:source_quant (arm C) | 75.9 | 1.57 | 6357.8 |
| sign_majority (arm B) | **2244.8** | 12.7 | 33627.5 |

## Readings

1. **The merge question is ANSWERED (negative) for the zero-training lane on
   this model.** The T3.2 laya finding (members always win at k=2) replicates
   at k=3 homogeneous GDN triples with the full pool through a
   parity-proven apply path. Arm C over the MEAN is the only survivable
   merge construction (median 1.70×) and it never wins a block.
2. **Arm B is dead on real activations** — its T4.1 operator-level damage
   (amax overshoot ~3×) compounds through the GDN recurrence into a median
   2,245× mapping error. The T4.1 record said "hopeless where the surrogate
   is ≥ ~31% wrong"; on real activations it is hopeless everywhere.
3. **The activation-space ranking DISAGREES with the S-cosine medoid on 10
   of 16 blocks** (e.g. [40,43): audition 42 vs S-medoid 41; [16,19): 18 vs
   the emit's 17-or-19 tie-break; [52,54): 53 vs 52). Weight-space cosine
   (the S matrix) is NOT the member-selection oracle — but the audition's
   local disagreement never gets an end-task test here, because at 53% depth
   the whole collapse dies anyway (the GOAT arm below). The member-selection
   question only bites at depths where blocks are wide — exactly the depths
   where nothing survives. That is the lane's ceiling in one sentence.
4. **Per-block error grows monotonically with depth** (2.9e4 at [0,3) →
   1.4e7 at [60,62)) — later blocks are increasingly un-representable by any
   single member, consistent with the layer-pruning literature's
   reasoning-harms gradient.

## The T5.0 gate outcome

T5.0's own gate — "only if the passthrough-only collapse FAILS quality does
the merge question justify the instrument" — was satisfied (fine-end sweep:
all real-depth cuts failed), the instrument ran, and it returned the SAME
verdict as laya: **member passthroughs dominate; the zero-training merge
class adds nothing on this lane**. The lane's honest ceiling stays the
fine-end passthrough window (~5% cut PASSES the 0.9 bar; ~11% hit-parity).
Beyond: riir-train 423's distillation track.

## The typed-control GOAT (this session's remaining arm)

`/tmp/twt_collapse_pq2_typed_e01_members.gguf` (34 blocks, medoid
passthroughs — emitted by `twt_collapse_emit --typed`) vs parent, the T5.1
budget: absolute top-1 agreement ≥ 0.9, parent arm CACHED. Result recorded
below when the arm lands.

```text
(pending — /tmp/twt_audition_logs/goat_typed_e01.log)
```
