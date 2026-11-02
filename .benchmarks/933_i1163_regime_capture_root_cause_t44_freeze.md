# Bench 933 — the regime-capture zero-rows bug (root cause + fix) and the T4.4 freeze/calibrate landing

**Status:** COMPLETE — 2026-11-02 (zed-amo-i638t1c, amo). The i1163 root cause was found STATICALLY; the fix landed and validated end-to-end; the T4.4 freeze pass (plan 625) completed on the first post-fix run.

## The bug and the fix (issue 1163, riir-infer `77ed153` / riir-ai `e4644a2d6`)

Four dump runs died (~12 GPU-h) with `total 0 rows`. The filed suspect (a
silent `read_one` Err arm) was WRONG — static analysis found the real chain:
the runner warms up DISARMED → the capture block's first `regime_capture_armed()`
call seeds `CAPTURE_PATH` (a `OnceLock<Option<PathBuf>>`) with `None` from the
unset env → the post-warmup `set_regime_capture_path` is a swallowed no-op
(OnceLock already initialized) → every dump item sees `armed() == false` →
0 rows → the `total > 0` panic at hour 3. A `OnceLock<Option<T>>` holding
`None` is INITIALIZED — the type carried the trap.

Fix (riir-infer):
- `CAPTURE_PATH` → `Mutex<Option<PathBuf>>` + a one-shot env seed; the setter
  overrides at ANY time and fences the seed (the warmup-then-arm flow works).
- `REGIME_CAPTURE_COMPILED` const mirroring the block's cfg (a compiled-out
  block is named in milliseconds, not hours).
- The append self-heals its parent dir; open/writeln failures counted
  (`regime_capture_write_failures`); the boundary read retries once behind a
  sync fence, failures counted (`regime_capture_read_failures`).

Fix (riir-ai runner): ms-scale posture assert; an ARMED SMOKE (2048-token
dense prefill) before the items — a dead seam fails in seconds with the
failure counters in the message; per-item `rows > 0` asserts + a final
failures==0 gate; the test-NAME FILTER is load-bearing in the documented
command (unfiltered `--ignored` also executes the bench_1152 sweep compiled
into the same binary — the ~2.5 h per dead run).

## Validation (the first post-fix run, amo, release, 2026-11-02 12:08)

```
[1156-dump] smoke: 1 rows — seam alive, dump begins
[1156-dump] needle#0 p16384: 4 chunk rows captured (66.2s)      (…14 items…)
[1156-dump] total 56 rows -> /tmp/regime_features.jsonl
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 1 filtered out
finished in 953.38s
```

56 rows = 14 items × 4 chunk rows (p16384 chunks into 4 — the capture fires
per chunk call, the chunked posture that NEVER captured before). All 7
families × 8 rows; failures == 0. Runtime 953 s vs ~3 h per dead run (the
name filter removed the sweep sidecar). Regression tests in regime_gate
(`capture_seam_end_to_end_after_disarmed_warmup` + the slim setter test,
cfg-exclusive so the global statics never race): 21/21 green at the triple
posture.

## T4.4 freeze + calibrate (plan 625 step 0 — landed in the same pass)

- Freeze: `regime_centroid_freeze /tmp/regime_features.jsonl
  crates/riir-infer-gpu/tests/fixtures/regime_centroids.json --gain 10.0
  --margin 0.20` → 56 rows → 2 classes, dim 10240; pair +
  BLAKE3 `fb319c6fd2b0955d02bc08c05525fb6a2867233191f579137ffc48cc48cfb0aa`
  committed to the fixtures dir.
- CALIBRATION READ (computed on the dump's cosine spread before freezing):
  raw-margin profiles per family (centroids = per-class means):
  needle +0.141..+0.145, qa +0.126..+0.134, summarization +0.125..+0.137
  (cleanly retrieval); math −0.150..−0.165, code −0.077..−0.113 (cleanly
  flow); **induction −0.052..−0.072 (routes FLOW-side — misclassified);**
  **continuation +0.105..+0.114 (routes RETRIEVAL-side — misclassified).**
  The boundary-pool feature axis does not carry the ΩMSR class labels for
  induction/continuation. Raw separation 40/56 (71.4%).
- **Calibrated pair: gain 10.0 / margin 0.20** — the tested pair at which
  ZERO retrieval-family rows take the loose floor: the margin band
  (|m| < 0.0847 in raw-margin units at gain 10) bands induction's
  −0.052..−0.072 into AMBIGUOUS → conservative (max) floor → safe;
  code's −0.077 row also bands (1 row conservative, the rest dense);
  needle/qa/summarization keep sparse (worst +0.125 > 0.0847);
  math stays dense. Continuation misroutes sparse-side (+0.105..+0.114
  past the band) — the SAFE direction (it pays dense, it never loses
  correctness). At the placeholder margin 0.1 (band ±0.04) induction would
  route dense — the DANGEROUS direction — hence the recalibration.
- Routing census at the calibrated pair: sparse 32 / dense 15 / AMBIGUOUS 9
  / wrong-side: flow→sparse 8 (continuation, safe direction),
  retr→dense 0.

## Cost + honesty

Dead-run cost: ~12 GPU-h (4 runs, pre-fix). Post-fix: 16 min to a green dump.
The `read_one` race theory in the issue text was wrong and is superseded by
the OnceLock root cause (the block never RAN). The /tmp evidence dir
(`/tmp/i1156_t44/`) was wiped by an amo power-cycle mid-morning — the root
cause needed none of it; lesson: evidence in /tmp on amo is one power-cycle
from gone.

## Next (T6 cells, plan 625)

Cells A–E stay window-gated (verified-idle amo, the thermal-order protocol);
Cell E's prerequisites (the sidecar pair + the calibrated gate) are now in
place. Arm posture for Cell E:
`RIIR_PREFILL_REGIME_GATE=1 RIIR_REGIME_CENTROIDS=crates/riir-infer-gpu/tests/fixtures/regime_centroids.json RIIR_OMSR_MANIFEST=<the v2 fixture>`.
The standing ΩMSR manifest quarantine (riir-infer `stash@{0}` — the
continuation-32k floor 0.85→0.80 re-measurement) is UNCHANGED and still
owner-gated (a floor change is an acceptance-gate change, never a drive-by).
