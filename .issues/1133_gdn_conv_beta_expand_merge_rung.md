# Issue 1133 — GDN decode rung: conv1d + beta/decay + expand/L2 in ONE launch (the last unmerged launch-bound boundary in the GDN chain)

**Status:** MEASURED-NULL, SHIPPED OPT-IN (Bench 905, 2026-10-27 ~02:2x +07) — G1 bit-identical everywhere (kernel-level at 2 geometries + the edge block; real-model walk/logits/stream FNVs EXACT in every window), G2 **2/4 pairs positive (max +0.48%), pooled ≈ 0% ± 0.5% — sign consistency FAILS the 899 bar; NOT promoted, the T4 staging list gains no row**. Ships `RIIR_GDN_CONV_MERGE=1` default OFF with the record (the 898 disposition shape: the null is kept so the shape is never re-tried blind) + a named re-open trigger (the 512-thread-block variant) that only fires if the ladder budget is still short after T3/T6 land. The mechanism: the ~62 µs/token dispatch saving is offset by the source-block occupancy loss the conv→expand dependency forces (smem staging is the only non-grid-barrier shape; 16 blocks/4096 threads vs the incumbent expand's 72/18432). Dev-iteration lessons (the −25% element-walk serialization; the negative-count clamp) recorded in the bench. Claim scope DISCHARGED for execution; the file stays as the record.
**Lease closed:** shikuwa-4090-i1133convmerge, epoch 1793032572, expires 2026-10-28T16:36:12Z — work COMPLETE (the rung shipped opt-in `RIIR_GDN_CONV_MERGE` + measured-null Bench 905 + recorded in the same session); reshaped from the leftover Claimed line per the Issue-943 fold law (nothing remains to coordinate).

## The finding (arithmetic from measured records, no new instrument)

- **Bench-900 measured the in-graph node dispatch cost e2e**: removing 64 attention-chain nodes saved ~85 µs/token ⇒ **~1.3 µs/node** — the census's ns-class premise was assumed away (Issue 1127's reconciliation; the e2e A/B is the deciding instrument).
- **Bench-899 validated the arithmetic**: its merge removed exactly 48 nodes (1/layer) and measured +0.6–1.0% e2e with −1.5% GPU busy.
- **This rung removes the same 48-node class**: 1 node/layer × 48 GDN layers ≈ **62 µs/token dispatch ≈ 0.57% e2e**, plus a small exec-overlap saving (conv 1.96 µs + fused-bde ~3 µs separate → one launch; the kernels are launch-bound, min≈med≈avg tight floors in the census).
- The ladder context: T1 (+0.77%) and B899 (+0.6–1.0%) are the staged opt-ins; T3 (in flight, ≥3.5%) and T6 (filed, ~2.6%) are the big levers. This rung is an 899-class staging row for the owner-gated T4 window — it does not re-cross the +2.71% bar solo and is not filed as if it did.

## Why the Bench-898 conv1d refusal does NOT cover this boundary

The 898/1127 non-goal reads: "conv1d stays its own launch — true in-place data dependency: every qkv consumer must observe the whole conv, and same-kernel fusion would need a grid barrier." That refusal is about fusing conv INTO THE RECURRENCE (its q/k/v reads span all heads → cross-block → grid barrier). This rung's consumer-side is the EXPAND kernel, whose Q/K L2 reads are **per-source-head local** (head_dim channels) and whose V copy is **per-channel 1:1** — the conv→expand dependency is solvable INTRA-block by staging (design below). The recurrence stays its own launch (the 899 lesson kept: never fold work into the recurrence's warp blocks).

## The design (bit-identity by construction, per the Issue-940 law)

`conv_beta_decay_expand_fused_f32` — one launch replacing `launch_conv1d` + `launch_beta_decay_expand_fused`:

- **Grid** = `ceil(conv_dim / 256)` × 256 threads (the conv kernel's own grid — the expand work rides the same blocks; no idle blocks: every block owns conv channels, source-blocks add Q/K expand, V-blocks add V copy).
- **Phase 1 (per-channel-private, zero cross-thread hazard, no sync needed before it completes):**
  - conv for channel c computed WITHOUT the shift-first reorder trap: the incumbent shifts the state window then dots ascending k (`Σ state'[k]·w[k]`, `state'[KS-1] = input`). The identical product sequence accumulates in the identical order directly from the UNSHIFTED window: `Σ_{k<KS-1} state[k+1]·w[k] + input·w[KS-1]` — same adds, same order ⇒ bit-identical. SiLU expression verbatim (`sum / (1 + expf(-sum))`).
  - Reads `input[c]` into a register BEFORE any write (the state append needs the RAW pre-conv value — the incumbent reads it before overwriting too).
  - Writes: `qkv[c]` in-place (same value the incumbent leaves), the conv-state shift+append (per-channel-private row), smem stage `smem[t] = post-conv value` for the Q/K expand phase.
  - V channels (`2·n_k·hd ≤ c < conv_dim`): ALSO write `expanded[2·n_v·hd + (c − 2·n_k·hd)]` directly (1:1 channel→out-elem, verbatim copy semantics).
  - Leading threads (`idx < n_head`): beta/decay verbatim from the 899 kernel (reads a/b/a_log/dt_bias — no conv dependency).
- `__syncthreads()` (one).
- **Phase 3 (Q/K expand, partitioned by SOURCE block):** block b owns the fully-contained source heads of its channel range (256/hd heads per block; requires `(n_k·hd) % 256 == 0 && (n_v·hd) % 256 == 0` — the league posture (16,48,128) = 2048/6144 both ✓). Each owned source head s serves out-heads `{s, s+nk, s+2nk, …} < nv` (the `nv % nk == 0` gate is already the 899 dispatch gate). Per out element: the sq_sum loop verbatim over the SMEM-staged values (same values the incumbent read from post-conv compact ⇒ same adds ⇒ same bits), zero-guard branch verbatim, `compact[src] · inv_norm` verbatim. Each `expanded` element is written by EXACTLY ONE block (out-head→src-head is a function) ⇒ no cross-block write hazard.
- **Misaligned geometries REFUSE at dispatch** → the wiring falls back to the incumbent chain (the Issue-1124 GLU posture-gate precedent: `mlp % 128 == 0`). Kernel-level tests pin both a passing aligned shape and the refusal path.

## Dispatch precedence + env

`core_fused` (898, takes the whole core) > **`RIIR_GDN_CONV_MERGE=1`** (this rung — supersedes 899; implies its beta/decay+expand semantics) > `RIIR_GDN_BETA_EXPAND_FUSED=1` (899) > incumbent chain. `=0`/unset byte-identical to the incumbent (the demote hatch). One-shot `[conv_merge] REACH:` stderr line (the Bench-768 vacuous-guard law). The dispatch lives inside `forward_deltanet_layer` — the graph-capture lane records it at capture, the eager lane runs it directly; one wiring site serves both (the 899 shape).

## Discipline (carried from 899/900, unchanged)

- G1: kernel-level bit-identity vs the 2-launch chain (`launch_conv1d` → `launch_beta_decay_expand_fused`) at the production geometry (16,48,128), an off-shape, the zero-vector-head L2-guard edge, and all three softplus ±20 branches — plus the real-model argmax-walk FNV `98412dc406c4ea19` + logits[::97] FNV `63496f6688c8f742` EXACT at `B734_ROUNDS=3` in EVERY A/B window.
- G2: paired-interleaved A/B on the league vehicle (`bench_734_t5_decode`, riir-gpu, `--features bonsai2_hadamard`, one binary env-dispatched, `B734_WARM_SECONDS=10`), ≥3 pairs, direction on BOTH axes (e2e + sync µs), REACH line confirmed in every armed window, box-state + exclusivity gate honored (Issue-834; verified-idle protocol).
- Ships opt-in, default OFF; promotion rides the owner-gated T4 window like every rung in this ladder; the loser is kept as the hatch, never deleted.
- If direction-NEGATIVE: record the negative (the 898 precedent), keep opt-in default-OFF so the shape is never re-tried blind.

## Non-goals / scope guards

- NOT the recurrence boundary — 898 measured folding into it NEGATIVE; the recurrence stays its own launch.
- NOT the rotation file (`deltanet_rotation_cudarc.rs`) — the i1127-T3 lease is live there; i1131 (T6) executes after it clears on the landed T3 pattern.
- NOT the prefill lane (`conv1d_rows` batched path, `prefill_cuda_full.rs`) — different vehicle, the league decode cell is the target.
- NOT the dense-qwen lane (`qwen38_dense_cudarc.rs` `forward_gdn_layer`) in this rung — the league model is the ternary Bonsai-2; the dense lane inherits the kernel via the launcher if its posture ever wants it (same gate applies), but its A/B is not this record.

**Refs:** Bench 902 (the census; conv 0.8% + expand 1.0% exec share), Benches 899 + 900 (the node-dispatch arithmetic + the paired-A/B discipline), Bench 898 (the refusal this design routes around, and the negative-record precedent), Issue 1127 (the ladder + T4 window), Issue 940 (reduction-order law), Issue 1124 (the posture-gate fallback precedent), Bench 768 (the REACH law).
