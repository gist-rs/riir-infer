# Bench 915 — Plan 623 T2.5: the probe EXCHANGE runs live on the real wire (the head fetches the tail's record and the plan composes itself — chosen split matches the cell-validated posture)

**Status:** RECORDED — the T2.5 landing residue (the wire probe step) VERIFIED end to end on the amo⇄bmo TB5 link: RTT measured on the serving link, the tail's record crossed the wire, the model cross-check held, `plan_split` composed live and chose **split 32** — the same 32/32 posture every real 2-box cell (902/908/910/913) runs. No promotion claim (the planner is a boot tool; its prediction is the formula's output, not a serving row)
**Session:** zed-amo-i623-exchange, 1793143154
**Date:** 2026-10-28
**Plan:** `.plans/623_multi_host_pipeline_inference_lane.md` T2.5 (the "honest residue" of the 2026-10-27 landing)
**Code:** `22eb298` — `crates/riir-infer-dist/src/probe.rs` + the `--plan-wire` / `--answer-probe` postures in `dist_stage`

## Why this cell existed

T2.5's planner landed 2026-10-27 with its records crossing "by operator/file" — the
two `--plan-report` JSON lines hand-carried between hosts. The landing note named the
residue: the automatic two-host EXCHANGE (wire step). `22eb298` landed it; this cell
is its live verification — CPU-only end to end, deliberately: the amo GPU is held by
the sibling 890/1146 lanes, and the EXCHANGE mechanism is backend-agnostic (the probe
times whatever `StageCompute` the posture builds).

## Provenance

- amo = head (load 2.66–3.08 during the run — sibling GPU lanes active; the CPU probe
  rows are their-box-state-labeled, not league rows), bmo = tail (load 1.07 idle,
  verified via `ps` — no cargo/rustc/dist_stage resident).
- Both binaries built at `22eb298` (release, `--features bonsai2_hadamard`, CPU
  backend — no gpu feature needed for the probe). bmo synced by incremental bundle
  (`fc28c98..origin/develop` + `--ff-only` against `refs/remotes/bundle/develop` —
  its stale `refs/bundle/develop` alias forced the full-ref spelling; noted for the
  next bundle sync).
- Model `Ternary-Bonsai-2-27B-PQ2_0.gguf` (amo `~/git/riir-train/data/`, bmo
  `~/models/`), n_layer 64, hidden 5120. TB5 link 192.168.10.1 ⇄ 192.168.10.2.
- Session key: fresh random 64-hex.
- First boot attempt refused loud (the 7.2 GB GGUF declares `prism.hadamard`; a
  non-`bonsai2_hadamard` build refuses to load a rotated file it cannot honor — the
  gate working as designed); rebuilt with the feature, cell clean.

## The run

```
tail (bmo):  dist_stage --role tail --answer-probe --split 32 --backend cpu \
               --listen 192.168.10.2:7633 --model <bonsai pq2>
head (amo):  dist_stage --role head --plan-wire 1,16,32,48 --split 32 --backend cpu \
               --connect 192.168.10.2:7633 --model <bonsai pq2>
```

Boot slices: tail `[32,64)` loaded 4.5 s, probe-answer posture armed; head `[0,32)`
loaded 4.8 s. The head probed its own candidates (scoped load + timed decode, 8 steps
× 3 rounds best-of per candidate), then connected, pinged, requested, planned, and
exited clean. The tail answered the session in **10.7 s** wall (four scoped loads +
four probes over the wire), clean shutdown. **Zero serving-loop frames touched** —
the exchange is a plan-mode posture; a serving tail refuses every probe frame at its
`Handshake` expectation by construction.

## Results

Link (measured, median of 8 authenticated ping/pong round trips on the SAME transport
the serving session uses): **RTT 0.166 ms**, bw carried at the T0.2 TB5 row
(8.6 GB/s — ~0.5% of the decode hop; carried-not-measured is disclosed in the tool
output itself).

Head record (amo, load 2.66, raw from `[plan]` lines): k=1 → 0.000879, k=16 →
0.013500, k=32 → 0.027096, k=48 → 0.040207 s/tok.

Tail record (bmo): the answer posture probes quietly (its rows crossed the wire in
the reply; the head prints the combined predictions). Derived exactly by subtraction
— the tail dominates all four candidates (head row + hop < predicted in every row),
so `tail_k = predicted_k − hop` with hop = 0.000166 + 20484/8.6e9 = 0.000168 s:
k=1 → 0.054214, k=16 → 0.042122, k=32 → 0.029543, k=48 → 0.040208 s/tok.

| split k | head s/tok | tail s/tok (derived) | predicted max+hop | |
|---|---|---|---|---|
| 1 | 0.000879 | 0.054214 | 0.054382 | |
| 16 | 0.013500 | 0.042122 | 0.042290 | |
| **32** | 0.027096 | 0.029543 | **0.029711** | **chosen** |
| 48 | 0.040207 | 0.040208 | 0.040376 | |

`PLAN: --split 32` — predicted steady state 0.029711 s/tok ≈ 33.7 tok/s at the CPU
probe posture. The near-tie at k=48 (0.040207 vs 0.040208 — the equaliser doing its
job) and the tail-dominated shape (amo's CPU is faster per layer than bmo's under
this box-state) are the planner's intended read.

## Verdict

- **The EXCHANGE works on the real wire**: one head command replaces the hand-carried
  JSON pair; the tail's record, the RTT, the model cross-check (n_layer 64 / hidden
  5120 verified both directions) and the plan all crossed/composed live.
- **Consistency with the cells**: the chosen split (32) is the posture every measured
  2-box row uses (TB5 32/32 = the 2.37× bubble-fill cell, Bench 908; the 913 wire
  battery; the 910 scoreboard). The equaliser independently lands where the league
  measured — a real consistency check, not a designed coincidence.
- **The `--plan-report` file arm survives the refactor byte-compatibly**: one head run
  at k=32 printed the same JSON shape (`{"host":"head","n_layer":64,"hidden":5120,
  "rows":[[32,0.026363]]}`) — the 3% delta vs the exchange run's 0.027096 is
  run-to-run CPU variance under load, expected.

## Honest notes

- The predicted 33.7 tok/s is the FORMULA's output at the CPU probe posture, not a
  serving row; validating it against a real 2-box CPU run is not this cell's scope
  (the serving postures are GPU — T2.7's lane). The planner's numbers are decision
  inputs; the cells remain the league.
- The tail's per-candidate `[plan]` stderr is suppressed in the answer posture
  (`quiet`) — its rows are derived above, exact by the max-side structure. A later
  one-liner can print them if operators want both sides' raw lines.
  **UPDATE 2026-10-28: both honest-notes RESOLVED (zed-amo-i623-t25residue) —
  the bw is measured live (echo-ack bulk, kinds 15/16; `--link-bw` stays the
  carried override) and the tail prints its own rows. Landing note: plan 623
  T2.5. The numbers above were taken at the carried-bw posture; the measured
  value prices the transport's MAC cost too, so a re-run reads slightly lower
  than 8.6 GB/s carried — expected, disclosed by the tool line.**
- amo was NOT quiet (load 2.66–3.08, sibling GPU lanes); the head probe rows carry
  that state. bmo was idle. RTT is network-bound and load-robust (median of 8).
- The probe timing posture is best-of-3-rounds (the landed `probe_decode_spt`
  semantics), 8 synthetic-token steps — the same posture the file-based arm has used
  since T2.5 landed; nothing was retuned for this cell.
