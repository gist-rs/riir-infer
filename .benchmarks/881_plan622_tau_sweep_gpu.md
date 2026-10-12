# Bench 881 — plan 622 App-H τ-sweep (the GPU FAST lane, τ ∈ {1, 10, 100})

**Status:** COMPLETE — verdict **Fail** at every swept τ; the App-H tolerance row is
recorded (5% pinned = the G3 bar); the sweep closes plan 622's Phase-5 App-H
task and feeds the owner-gated promotion decision (both axes now fully measured).

- 2026-10-13, the m3-max idle loop (the claim row `b67f0e66` in
  riir-refine `.distill/001_mining_queue_snapshot.md`; walk #14 ran concurrently
  M3-side, zero conflict).
- Verdict assembly = `katgpt_attn::si_probes::assess_tau_sweep` (the landed
  Phase-5 modelless half — this bench is the ppl measurement loop it was
  written to consume).

## Box state (provenance — Issue-021 law)

| Axis | Value |
|---|---|
| GPU | RTX 4090 24 GB (sm_89, Ada), KB5121794 post-era |
| GPU util pre-run | 5% / 372 MiB (desktop only) |
| Build | **the bench-873 build REUSED VERBATIM** — release exe `E:\tmp\tgt-si620post\release\si_zero_shot_ppl_gpu.exe` (built 2026-10-13 02:19, the `fd16ab6`+harness-fix lane; NO rebuild — same-binary comparability with bench 873's τ=10 rows is the point) |
| Runner | `E:\tmp\si622_tausweep.cmd` via schtask `si622tau` (two sequential runs, `DONE-EXIT-0` 06:28:11 box time, ~31 min total) |
| Config | `--arms base,si,si-cal --cal-sidecar E:\tmp\si622-cal-4090.bin --max-length 2048 --min-length 128 --rungs 16 --score-window 128 --tolerance 0.05 --tau {1.0,100.0} --retrieval`; corpus wikitext2_test 2049 tokens (BOS incl.) — identical to bench 873 per-run |
| Exclusivity | box idle at start (gitsync drained first); the walk #14 pass ran M3-side only |

## The sweep table (pooled ppl over 128-token scored spans, base = 7.2499 both runs — bit-consistent with bench 873)

| τ | si pooled | si Δppl | si-cal pooled | si-cal Δppl | si retrieval | si-cal retrieval |
|---|---|---|---|---|---|---|
| 1.0 | 10.8399 | **+49.52%** | 10.8675 | **+49.90%** | **4/4** (zebra+river @ 1024+2048) | 2/4 (zebra both) |
| 10.0 (bench 873) | 10.5846 | **+46.00%** | 11.4233 | **+57.56%** | 3/4 | 2/4 |
| 100.0 | 9.1015 | **+25.54%** | 14.0809 | **+94.22%** | **4/4** | **0/4** |

The pooled deltas ARE the in-dist rows: every scored rung (128→2048) sits inside
the training context, so the pooled row is the in-distribution read — recorded
as such, not silently widened.

## `assess_tau_sweep` verdict (tolerance 5.0% = the G3 bar, disclosed)

Rows (si arm, the paper-faithful loop's primary; si-cal carried alongside —
the calibrated arm is the paper's QK-norm posture):

```
[{tau: 1.0, delta: 49.52}, {tau: 10.0, delta: 46.00}, {tau: 100.0, delta: 25.54}]
→ TauSweepVerdict::Fail([(1.0, 49.52), (10.0, 46.00), (100.0, 25.54)])
```

si-cal rows (same tolerance): `[(1.0, 49.90), (10.0, 57.56), (100.0, 94.22)]` —
also all-offending. **No swept τ buys the length-generalization win without
in-dist regression at this checkpoint.**

## The physics the table shows (the sweep's real finding)

1. **si-only in-dist damage DECAYS with τ** (+49.5% → +46.0% → +25.5%) while
   retrieval stays perfect at the extremes (4/4 at τ=1 and τ=100) — the
   schedule's length-temperature ramp widens with τ, leaving short-range
   logits closer to untouched, exactly the a₀²=1 boundary law's prediction.
   The τ=100 si arm is the least-bad zero-shot posture on BOTH axes measured.
2. **Calibration does not transfer across τ** — si-cal is the paper-faithful
   posture (per-head σ from the frozen sidecar measured at load), but its
   in-dist damage GROWS with τ (+49.9% → +57.6% → +94.2%) and its retrieval
   collapses at τ=100 (0/4). The frozen σ table was measured at the un-si
   forward; under a wide ramp the normalized logits no longer sit at the
   Gaussianity premise the schedule assumes — the calibrated arm's failure is
   the premise drift, not a bug in the sidecar (digest `95ef…` verified both
   runs, same slots 1536).
3. **τ=1 is the worst in-dist posture with the best retrieval** — the tight
   ramp distorts EVERY distance bin the model actually uses in-dist
   (per-rung worst +93.85%), while the needle at distance ~2000 survives
   because the ramp concentrates mass exactly there. The quality axis is
   distance-shaped, not uniform.

## What this closes and what it arms

- Plan 622 Phase 5's App-H task: the ppl loop is DONE (this record) feeding
  the landed `assess_tau_sweep` — the task's modelless half was landed
  2026-10-11, this is its measurement half. Verdict recorded: Fail at 5%.
- The owner-gated promotion decision now has BOTH axes swept: G2 perf-neutral
  (bench 878), G3 quality FAILs reproduce at every τ with the τ-shape above,
  G1 schedule properties + G4 alloc-free (bench 932 CPU legs). The feature
  stays opt-in; the si lane's remaining value is the TRACK-B (training)
  integration question (riir-train Plan 449), not zero-shot adoption.
- The FAST re-gate lane cost on record: ~15.5 min/run all-arms+retrieval —
  any future si-touching landing re-gates for that, not the CPU lane's 2 h/arm.

## Records

- `E:\tmp\si622-tausweep.log` (the run; box-side, regenerable)
- bench 873 (the τ=10 cells this sweep extends; same build)
- `.research/610_Scale_Invariant_Attention.md` (the App-H protocol source)
- katgpt-rs `.plans/622_scale_invariant_attn.md` (the plan this closes the
  League/App-H row under)
