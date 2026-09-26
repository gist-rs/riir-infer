# Issue 018 — plan 611 S4 follow-up: the Cubecl posture's residual prob-drift
# wobble (sporadic per-forward, top-1 never flips)

**Status:** RESOLVED 2026-09-26 (session `riir-infer-m3-t7c`) — TWO
independent defect classes, both landed: the **within-kernel** smem
write-after-read in the row-softmax kernels (`2a34bd3`) and the
**between-dispatch** missing Metal memory barriers in the vendored
wgpu-hal fork (`c0dfa06`). The G5 cubecl posture is bit-stable:
top-1 1.000000 ×3 checkpoints ×5 runs, drift 3.092e-6 / 2.233e-6 /
5.187e-6 — the deterministic reduction-order floor (~200× under the
1e-3 gate), byte-identical across runs. The reflex `#[ignore]` re-arms.

## The resolution record (what the two classes were)

**Class A — in-kernel (sibling session `2a34bd3`):** both softmax
kernels read `smem[0]` (the global max) and later overwrite slot 0
with the sum partial — without a `sync_cube()` between the read and
the overwrite, a slow SIMD group read the SUM as the max. Sparse,
scheduling-dependent, inside one kernel — invisible to every
inter-op instrument.

**Class B — between-dispatch (this session, `c0dfa06`):** wgpu-core
emits a barrier request for every conflicting dispatch pair
(`flush_bindings` → `drain_barriers` → hal `transition_buffers`), but
the Metal hal implemented the two transition functions as EMPTY
bodies (upstream 30.0.x state, not a fork regression) — and Metal
does not order memory across dispatches WITHIN one compute pass.
Every write→read chain between cubecl kernels sharing a pass (split →
rope → scale → score GEMM → mask → softmax) was unbarriered.
Fix: `memoryBarrierWithScope` at BOTH emission sites — the
transition functions (usage-CHANGE pairs) and `set_bind_group` when a
compute pass is open (consecutive same-state in-place chains —
rope→scale on q, add_mask→softmax on scores — never produce a
transition; a set_bind_group precedes every rebind).

**How B was localized** (the deep probe, `LAYA_PROBE_DEEP=1`): the
probe's repeat arm became a loop (8 passes, first-divergent-tag) and
deep mode sinks the attention internals — `L11.scores` diverged
7.77e-1 with `q`/`k`/`v` clean at 1e-6, naming the score GEMM as the
first kernel AFTER an unbarriered write→read pair. Fires went from
~every pass (8/8, 5/8, 3/8) to 0/18 ×3 runs; the first-pass
cpu-vs-cubecl tables went run-dependent to byte-identical.

**The remaining drift is NOT a defect:** the deterministic floor is
two valid f32 summation orders (the CPU lane's blocked `gemm` vs the
GPU kernel's sequential-k accumulation) on cancellation-heavy dots —
the −21821 residual outlier at token 6 carries a one-time ~0.34
absolute offset (≈100 ulp), laundered by the final LayerNorm to the
observed ~1e-6 prob drift. Bit-stable, ~200× under the G5 gate.

## Why lever 2 (sync-per-forward) was answered WITHOUT running it

The probe serializes per OP (every sink drains) — and the wobble fired
under it (2.288e1 at L27.h_mlp, 016-era; the deep probe's L11/L3
scores fires). A per-forward drain is coarser than a per-op drain, so
it could not have pinned the wobble; the fire sites were inside
single ops (Class A) and inside single composed ops (Class B). The
`LAYA_CUBECL_SYNC_PASS` knob was built, answered negative by this
reasoning, and removed before landing.

## Standing notes

- The barrier fix protects every wgpu-compute consumer sharing the
  fork (the engine decode lane runs the same no-sync residency at ~835
  dispatches/token) — the decode lanes should re-bench once; the
  per-rebind barrier cost is bounded by real pipeline depth and
  measured by plan 611 S5's A/B.
- Every hypothesis in the "what it is NOT" list below held (the
  gather, the LN cancellation, cross-forward overlap, host-authoring,
  within-dispatch write races were all excluded by measurement — the
  two live classes were the smem read fence and the inter-dispatch
  ordering).

---

## Pre-resolution record (kept for the trail)

Filed 2026-09-26 · session `riir-infer-m3-t7b` · plan 611 S4 follow-up.

## Context — what 016 was and what fixed it

016's failure (top-1 0.33–0.46, prob drift 0.5–0.9) was the
`gather_rows` residency class: the CubeCL backend bound the head's
marker-gather SOURCE (the hidden state — an ACTIVATION, device-current
with stale host bytes) through the PERMANENT weight cache, uploading the
stale host zeros once; every marker row read zeros and all scorer logits
collapsed to bias. The Metal lane's identical comment named the correct
class all along. Fixed: the source binds `chain_buf` (epoch-keyed, hit =
the live device slot). Regression arm added to `cubecl_ops_smoke`
(`gather_rows_device_written`: LN writes the source via a write-first op,
host bytes stay zeros, the gather must bind the live slot).

The two-pass LayerNorm (replacing the S1a one-pass `E[x²] − μ²`, whose
cancellation the issue-016 probe priced at deep layers — residual values
±4000) took the drift floor down 2.2× and removed the length-correlated
class.

## The residual wobble (this issue)

G5 cubecl posture, 6+ runs of the same binary (`laya_riir_parity`
`g5_parity_cubecl_posture`), top-1 = 1.000000 on all three checkpoints
EVERY run; prob drift per checkpoint:

```
run A: english 3.092e-6 · typed 1.581e-4 · multilingual 5.187e-6   PASS
run B: english 6.727e-3 · typed 1.081e-4 · multilingual 5.187e-6   FAIL(en)
run C: english 3.688e-6 · typed 2.956e-4 · multilingual 1.018e-3   FAIL(ml)
run D: english 3.445e-3 · typed 2.838e-5 · multilingual 2.091e-3   FAIL(en,ml)
run E: english 8.568e-4 · typed 3.293e-4 · multilingual 6.357e-5   PASS
```

The per-checkpoint drift FLOOR is tiny (3.7e-6 / 2.8e-5 / 6.4e-5) and
reproduces; sporadically a checkpoint reads 10–1000× its floor. The
outlier MOVES between checkpoints and runs — the corruption is
per-forward, sporadic (~5–10% of forwards), never argmax-flipping.

## What it is NOT (measured)

1. **Not the gather class** — fixed, smoke-armed (see above).
2. **Not the LN cancellation class** — two-pass now; the drift floor
   dropped and the length-correlation is gone.
3. **Not cross-forward overlap** — the per-question G5 path's last
   `download_into` (`act_of`) drains the stream before the forward
   returns; `begin_pass` starts clean.
4. **Not host-authoring staleness** — the issue-016 probe audits every
   host-authored read (gathered, rope, mask, act_in): each is written
   before its first bind, once per pass.
5. **Not within-dispatch write races** — every kernel audited
   write-complete (full edge tiles, full row walks) with disjoint
   element ranges; rope is pair-threaded race-free by construction.

## What it IS consistent with (unproven hypotheses, ranked)

1. **A within-forward dispatch-ordering hazard at the CubeCL/wgpu
   runtime level** (cubecl 0.11.0-pre.2, wgpu 30 + the vendored
   wgpu-hal fork — a pre-release stack): a queued kernel reading a
   buffer another queued kernel is still writing, materializing only
   under scheduler timing (device load, queue depth). The probe's
   per-op drains make it rare there (1 fire in 4 runs), the G5's
   unsynced forward makes it commoner.
2. **A CubeCL server slot-reuse hazard**: `begin_pass` drops ~100
   chain handles per forward; if the server's block free-list hands a
   freed block to a new `client.empty` while a queued task still
   references the old handle, the recycled bytes leak into a read.
   (The engine's decode lane uses the same residency pattern with no
   observed wobble — but it never drops handles mid-stream the way
   `begin_pass` does.)

## The instruments (already in-tree)

- `tests/cubecl_encoder_probe` (laya crate, `laya-riir-cubecl`):
  CPU-vs-CubeCL per-op diff at real geometry (seq 400) + a
  begin_pass-separated repeat pass with per-tag diff printing — the
  repeat arm fires `REPEAT-DIFF <tag> <diff>` lines when the wobble
  hits (observed once: 2.288e1 at `L27.h_mlp`).
- `Encoder::forward_probe` (encoder.rs, cfg-gated): the per-op sink the
  probe drives; mirrors `forward_packed` exactly.

## Next levers (in yield order)

1. **Catch the wobble in the probe's repeat arm** (loop N passes, print
   the first divergent TAG): the first divergent op names the kernel or
   the binding path. Run it under GPU load (another agent building) to
   raise the fire rate if needed.
2. **Sync-per-forward A/B**: drain the stream in `begin_pass` (one
   blocking read of a 1-element buffer — `client.sync()` is DynFut, so
   a tiny `read_f32` is the pragmatic drain) and re-run the G5 ×5. If
   stable, the hazard is cross-forward after all (re-examine 3 above);
   if not, it is within-forward (lever 1).
3. **Minimal repro against the runtime**: export the failing op
   sequence (layer loop at seq 512, one pass, no drains) into a
   riir-infer-gpu test and bisect dispatch counts — a candidate for a
   cubecl upstream report (the stack is pre-release).
4. **Pin/repin**: if 2 pins it, the A/B (plan 611 S5) prices the drain;
   the lane's keep/delete verdict can then weigh portability vs the
   sync cost honestly.

## Session marker

`Session: riir-infer-m3-t7b, 2026-09-26` — the gather fix + two-pass LN
+ this filing land in one push; the reflex-side `#[ignore]` on
`g5_parity_cubecl_posture` lands beside it.
