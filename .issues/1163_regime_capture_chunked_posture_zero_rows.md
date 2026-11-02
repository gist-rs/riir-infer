# Issue 1163 — the T4.4 regime-feature capture writes NOTHING in the dump runner at the chunked posture

**Status:** ROOT CAUSE FOUND + FIX LANDED 2026-11-01 (zed-amo-i638t1b) — found STATICALLY (no GPU run burned); awaiting the dump re-run to validate end-to-end, then close.
**Claimed:** zed-amo-i638t1b, 1793479094, expires 2026-11-02T09:30:00+00:00

## Root cause (2026-11-01, static analysis — the read_one race was the WRONG suspect)

`CAPTURE_PATH` was a **`OnceLock<Option<PathBuf>>`**, and the dump runner's
flow is warmup-then-arm:

1. Runner warms up **disarmed** → the capture block's first
   `regime_capture_armed()` call runs `get_or_init` → env unset → the OnceLock
   stores **`None`**. A `OnceLock<Option<T>>` holding `None` is INITIALIZED.
2. `set_regime_capture_path(&dump_path)` → `OnceLock::set(Some(...))` **fails**
   (already initialized) → `let _ =` swallowed the Err.
3. Every dump item: `armed() == false` → the capture block is skipped → 0 rows
   → the `total > 0` panic at hour ~3.

This explains every observation from the 4 dead runs: feature-set-irrelevant
(full-feature runs died identically), `nm` symbols present (the block compiled
but never executed), the scratch probe WORKING (it armed before any `armed()`
read, or via env), and warmup rows never appearing. The **compiled-out-cfg**
posture (a build without `ternary_attention_batched_prefill`) dies the same
signature with even less diagnostics — same assert, zero messages.

Two more silent zero-row classes found on the way: the append's swallowed
`open()` Err (a missing parent dir) and a genuinely-failed boundary
`read_one` (still possible; now loud + counted + retried).

## Fix (landed 2026-11-01)

**riir-infer** (`regime_gate.rs`, `ternary_deltanet_gpu_forward.rs`):
- `CAPTURE_PATH` → `Mutex<Option<PathBuf>>` + a one-shot env seed
  (`CAPTURE_ENV_SEEDED`): the env seeds the default on the first `armed()`
  read; the setter **overwrites at ANY time** and fences the seed so a later
  read cannot clobber it. The warmup-then-arm flow now works.
- `REGIME_CAPTURE_COMPILED` const mirroring the capture block's cfg — runners
  assert the posture in milliseconds instead of discovering a compiled-out
  block at hour 3.
- Append: parent dir ensured per call; open/writeln failures are named
  (`eprintln`) + counted (`regime_capture_write_failures`).
- Boundary read: one sync-fenced retry, then a named failure +
  `regime_capture_read_failures` counter.

**riir-ai** (`bench_1156_regime_feature_dump.rs`):
- Asserts `REGIME_CAPTURE_COMPILED` first (ms-scale posture gate).
- **Armed smoke** before the 14 items: one dense 2048-token prefill, asserts
  rows > 0 — a dead seam now fails in seconds, not at hour 3. Smoke rows are
  truncated; the census rides a counter offset.
- Per-item assert `rows > 0` + a final failures==0 assert.
- Parent dir ensured before arming.
- Module doc: the test-NAME FILTER is now load-bearing in the documented
  command — unfiltered `--ignored` also executes the bench_1152 sweep compiled
  into the same binary (~2.5 h on the same GPU; a startup reminder line prints
  too).
- Regression tests: `capture_seam_end_to_end_after_disarmed_warmup` (triple
  posture: arm-after-disarmed-read, append lands on disk, counters reconcile,
  missing parent self-heals) + the slim setter regression at weaker postures
  (the two are cfg-exclusive so the global statics are never raced between
  parallel tests). `cargo test -p riir-infer-gpu --features
  cubecl_runtime,ternary_gemm_batched,ternary_attention_batched_prefill --lib
  regime_gate` → 21/21 green.

## Symptom (historical)

`bench_1156_regime_feature_dump` (riir-ai, the T4.4 freeze pass's runner half)
completed all 14 dense league items then panicked at the `total > 0` assert —
`[1156-dump] total 0 rows`, the dump JSONL never created. 4/4 runs dead on amo
(01:0x, 04:2x, 05:1x, 08:3x +07; runs 2–4 AFTER the riir-ai worktree was ff'd
to origin/develop `639d57080` and run 4 with `ternary_attention_batched_prefill`
EXPLICITLY on). Each attempt cost ~3 h because the dump target compiles the
sweep module in and the sweep re-runs to completion before the panic.

## Evidence

Scratch analysis + probe sources were at amo `/tmp/i1156_t44/` (DELETED from
/tmp 2026-11-01 morning — after the 7:39 dump run started; the evidence lived
only there, a lesson for /tmp residency). The root cause above needs none of
it — it is visible in the runner/seam sources at the cited commits.

**Numbering note:** first filed as 1162 locally without the pre-allocation gate run — 1162 was already allocated upstream (zed-m3-idle1031b, the ternary_gemv-alone compile breakage, filed f9140a4 / fixed 2c694cb / closed d24608c); renumbered 1162→1163 per first-allocation (dual_allocation_gate lesson re-paid).
