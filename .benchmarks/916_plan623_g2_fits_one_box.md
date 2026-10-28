# Bench 916 — Plan 623 G2: the fits-one-box PP-overhead row (loopback s=1 vs the 1-box oracle, same run, same build)

**Status:** RECORDED — the G2 gate's last open clause measured: **loopback PP s=1 carries NO decode overhead vs the 1-box arm (−1.4..−1.6% across three runs — inside noise, ≤5% with margin); G1 PASS ×3**
**Date:** 2026-10-28
**Session:** zed-amo-i623-g2fitsbox, 1793148942
**Plan:** [`.plans/623_multi_host_pipeline_inference_lane.md`](../.plans/623_multi_host_pipeline_inference_lane.md) — the G2 gate line ("never > 5% PP overhead on the fits-one-box row"); the arithmetic precedent is [Bench 896](896_plan623_t02_link_probe.md) §"PP stage hop" (0.09–0.10 ms hop vs a 5–7 ms decode step ≈ 1.3–1.8%, derived, never measured until now)
**Code:** `3b286c8` — `run_local_with_report` (the `--full-check` oracle prints its decode-only window; the comparator is the DECODE rate, the same window the pipeline arm prints, never the prefill-diluted end-to-end figure)

## Why this row exists

The G2 gate demands the pipeline HARNESS (protocol + copies + sequencing) cost
≤5% when the model fits one box — the physical-link cost is separately priced
(the 2-box TB5 s=1 −15% is the wire's RTT, Bench 910; a link property, not a
harness defect). Until now the assertion rested on Bench 896's arithmetic. This
cell measures it: both stages on ONE box over loopback TCP, the 1-box oracle
loaded in the SAME process right after the pipeline run — one run, both arms,
same GPU clocks, same build, G1 compared per run.

## Rig

- **Box:** amo (`Mac17,15`, M5 Ultra, 96 GiB, macOS 27.0.1). **GPU-quiet gate
  green at fire: 0%** (ioreg) for every run; load 5.31 carried (Zed Dev 97% —
  the editor + agents; no compute jobs — disclosed). Thunderbolt peer bmo idle.
- **Build:** release `--features gpu,bonsai2_hadamard` @ `3b286c8`, bin sha256
  `e235896466eca0c1…`.
- **Model:** `Ternary-Bonsai-2-27B-PQ2_0.gguf` (7,206,168,928 B), split 32,
  GPU BOTH stages, **loopback TCP** `127.0.0.1:18777`, streams 1, ctx 4096,
  prompt `7,13,42,100`, `--max-new 32`, `--no-eos`, incumbent cadence (no
  `RIIR_DIST_PAIR`), authenticated frames (`RIIR_DIST_KEY`).
- The `[norm_rot_multi] REACH` line fired both sides (the promoted 1145 posture
  is the arm under test); the [Issue 864] KV working-set advisory printed once
  per head run (steady-state unaffected, per its own text).

## Results (3 runs, ABCA-order repeats of the same rig)

| run | PP s=1 decode tok/s | 1-box decode tok/s | oracle prefill | overhead |
|---|---|---|---|---|
| a | 55.33 | 54.52 | 0.135 s | **−1.49%** |
| b | 55.44 | 54.68 | 0.129 s | **−1.39%** |
| c | 55.88 | 54.99 | 0.128 s | **−1.62%** |

G1 `stream 0 == 1-box stream` **PASS ×3** — the split pipeline is
token-identical to the 1-box oracle on every run.

End-to-end oracle figures (incl. prefill) for the record: 44.05 / 44.58 /
44.82 tok/s; PP prefill 0.379–0.400 s (the 4-token prompt pays two wire hops
+ the stage handoff — the T2.3c lane's business, measured at scale there; not
this row's axis).

## Verdict

- **The fits-one-box G2 row PASSES with margin**: measured loopback PP overhead
  is **−1.4..−1.6%** — the ≤5% bar holds with ~6.5 points of headroom, and the
  point estimate sits at or slightly below zero.
- **The consistent negative sign is a POSTURE note, not a claimed PP win**: the
  pipeline arm runs two 3.6 GB stage instances where the oracle runs one 7.2 GB
  (each stage holds half the KV working set); ~0.5 tok/s ≈ 0.18 ms/token is
  within cross-posture noise and no mechanism study stands behind it. The claim
  is exactly the gate's: no harness overhead. A "PP is faster" claim would need
  its own controlled study — not filed, not asserted.
- **G2 is now fully evidenced**: the capacity + aggregate-throughput clauses
  are Bench 910's scoreboard rows (s2 aggregate 2.37× s1 vs the opponents'
  single-stream PP rows); this cell closes the last open clause.

## Honest scope

- One box (amo), one model class (Bonsai-2 27B PQ2_0), one split (32/32), one
  ctx (4096), s=1 only — the fits-one-box row is a one-box assertion by
  definition; the 2-box rows live in Benches 908/910/913.
- The 1-box comparator is the same-process oracle (`--full-check`), i.e. it
  inherits the pipeline run's warm GPU state — the conservative direction for
  an overhead gate: a cold-clock 1-box arm would read SLOWER, inflating the
  apparent PP overhead; the warm same-run posture removes that bias.
- CPU load 5.31 (the editor) rode every run identically on both arms; the
  decode windows are GPU-bound (86–87% GPU util during the runs).

**Session:** zed-amo-i623-g2fitsbox, 1793148942
