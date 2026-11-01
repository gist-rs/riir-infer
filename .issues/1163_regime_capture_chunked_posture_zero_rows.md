# Issue 1163 — the T4.4 regime-feature capture writes NOTHING in the dump runner at the chunked posture

**Status:** open — filed 2026-11-01 (zed-amo-i638t1b) after 4 dead dump runs (~12 GPU-h).
**Claimed:** zed-amo-i638t1b, 1793479094, expires 2026-11-02T09:30:00+00:00

## Symptom

`bench_1156_regime_feature_dump` (riir-ai, the T4.4 freeze pass's runner half)
completes all 14 dense league items then panics at the `total > 0` assert —
`[1156-dump] total 0 rows`, the dump JSONL never created. 4/4 runs dead on amo
(01:0x, 04:2x, 05:1x, 08:3x +07; runs 2–4 AFTER the riir-ai worktree was ff'd
to origin/develop `639d57080` and run 4 with `ternary_attention_batched_prefill`
EXPLICITLY on). Each attempt cost ~3 h because the dump target compiles the
sweep module in and the sweep re-runs to completion before the panic.

## Proven

- **Seam works at the single-chunk posture.** A scratch probe (warmup + ONE
  `fwd.prefill` call) wrote rows: p4096 → 1 row; p16384 with
  `RIIR_PREFILL_CHUNK_MAX=0` (single call) → 1 row. Armed/setter/gather/
  read_one/append/counter all function there.
- **Feature set NOT the cause** (refuted): run 4 carried
  `ternary_attention_batched_prefill` explicitly (cargo tree confirms it rides
  riir-gpu's DEFAULT feature list anyway); `nm` shows `boundary_pool`,
  `regime_capture_append` and friends present in the test binary.
- The capture block sits inside `prefill_tokens_chunk_once`
  (`ternary_deltanet_gpu_forward.rs` ~L10356), gated
  `regime_capture_armed() && x_staged` under
  `cfg(all(cubecl_runtime, ternary_gemm_batched, ternary_attention_batched_prefill))`.

## Leading suspect

The read arm is **silent**: `if let Ok(bytes) = self.client.read_one(bnd) { … }`
— an `Err` is swallowed with no log and no counter. A wgpu map/read failure on
a freshly `empty()` handle right after the gather launch (a race the small
probe shapes win but the 4096-token chunk loses) would produce exactly this
zero-row signature. Second candidate: something about the chunked driver's
per-chunk entry that skips the block (but `is_final_chunk=true` at p=4096
equals the probe shape, which works — weakens this branch).

## Fix plan (T1)

1. Loud `Err` arm + a per-item `assert!(rows > 0)` + one
   `pollster::block_on(self.client.sync())` retry before the read.
2. Split the dump target from the sweep module (or gate the sweep behind a
   flag) so a dump attempt costs 12.5 min, not 3 h.
3. Re-run the dump → freeze → calibrate (the T4.4 remaining steps, plan 625).

## Cost so far

~12 GPU-hours of dead runs on amo; the T6 Cell E (regime gate) is blocked on
the centroid pair this dump feeds. Cells A–D are NOT blocked (they don't read
the capture) but share the box protocol.

## Evidence

Scratch analysis + probe sources: amo `/tmp/i1156_t44/` (`next_steps.md`,
`regime_capture_bug.md`, `probe_capture.rs` copy); dead-run logs
`dump.log`/`dump2.log`/`dump3.log`/`dump4.log` in the same dir. The sweep
side-effect: the v2 manifest fixture was rewritten by the re-runs (box_state
line + continuation-32k floor 0.85→0.80 — a LEGIT measurement delta, uncommitted);
stashed as `stash@{0}` in riir-infer pending its own verdict note.

**Numbering note:** first filed as 1162 locally without the pre-allocation gate run — 1162 was already allocated upstream (zed-m3-idle1031b, the ternary_gemv-alone compile breakage, filed f9140a4 / fixed 2c694cb / closed d24608c); renumbered 1162→1163 per first-allocation (dual_allocation_gate lesson re-paid).
