# Issue 029 — NIAH builder undershoot crash + Phase-D-only report writer (Bench 012 follow-ups)

**Status:** OPEN — filed 2026-09-30 from the Bench 012 run (issue 013 T2): the
NIAH phase crashed at trial 0 and the structured report was never written.

## Defect 1 — `build_niah_trial` undershoot hard-bails

`.benchmarks/012_t2_run.log`:

```
Error: niah trial 0 (sunset1000)
Caused by:
    niah build: token budget 972 < target 1023 (grow the filler)
```

`src/transformer/vk_harness.rs::build_niah_trial` sizes the body char budget as
`seq_len * 42 / 10 + 64` (~4.2 chars/token, "measured on the gemma SP
tokenizer") and relies on the shrink path to correct any error. The shrink path
absorbs OVERSHOOT only — undershoot is `bail!`. The filler pool tokenizes at
~4.49 chars/token on this fixture (trial 0, depth 0.25, password sunset1000) →
4364 chars → 972 tokens, 51 short of `body_target = seq_len − 1`.

**Fix (bounded grow-retry):** wrap the build in a retry loop that multiplies
`body_chars` by 5/4 on undershoot (cap ~8 retries; each retry re-encodes and
re-runs ALL existing guards — prefix monotonicity, verify-decode of the
password span, shrink-inside-post-needle-filler, post-cut tail sanity). The
guards make the retry safe: a re-segmenting tokenizer or a boundary break still
fails loud. Overshoot keeps taking the existing shrink path unchanged.

## Defect 2 — the report writer runs at Phase D only

`kv_plus_ladder.rs` writes `--report` once, after the NIAH phase. A crash in
ANY earlier phase loses the structured report AND the in-memory per-arm detail
(win shares, flip distributions, the ρ_l(V−K) dashboard) — Bench 012's gates
were adjudicated from log summary lines alone. **Fix: write the report
incrementally after every arm completes** (the .cmd header already claims
"rewritten after every arm" — make the code match the contract).

## Follow-up lane

- [ ] Grow-retry fix in `build_niah_trial` + a unit test that builds a trial at
      a chars/token ratio that undershoots (the 4.49 case) and still produces a
      valid `seq_len`-token trial with the password intact.
- [ ] Incremental report write per arm in `kv_plus_ladder.rs`.
- [ ] `--niah-only` flag: load `--table <path>` (skip calibration), skip the
      ladder/grid/validation, run only Phase C4 + D, write a NIAH-only report.
- [ ] Rerun NIAH off `.benchmarks/012_kv_table_residual.bin` (sha
      `3a0f5333d6bafd63…`) — arms {f16, k-0.00, k-0.50, k-1.00} (the k-sched
      arm is moot: G-D recorded the schedule overfit) — and append the G-E
      direction rows to the Bench 012 gate doc. ~2.1 h at the observed 4 fwd/s.
