# Issue 029 — NIAH builder undershoot crash + Phase-D-only report writer (Bench 012 follow-ups)

**Status:** RESOLVED 2026-09-30 — Defect 1 fixed + Defect 2's scope landed as
`--niah-only` (commit `d9acd5a`), rerun executed (`.benchmarks/012b_niah_only_report.md`,
rc=0, 6/6 trials built past the exact crash point); the G-E rows are appended to the
Bench 012 gate doc. The fixture-gated unit test LANDED `e474f62` (2026-09-30, 5-trial
guard grid, loud-skip without the fixture). Remaining open (deliberate): the
incremental per-arm report write — below, unexecuted; the instrument gap is documented
in the gate doc's finding 4.
**In flight (sibling session, uncommitted WIP in the live worktree — do not
duplicate):** ~~the grow-loop fix in `build_niah_trial`~~ landed; ~~the
`--niah-only` wiring~~ landed.

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

- [x] Grow-retry fix in `build_niah_trial` — **LANDED `d9acd5a`, ratio-adaptive form**
      (multiply `body_chars` by the measured deficit `body_target/total + 256`, re-encode,
      bounded loop; all downstream guards — prefix monotonicity, verify-decode, shrink —
      run on the final text unchanged). Validated live by the 012b rerun (6/6 trials built
      past the exact crash point).
- [x] Fixture-gated unit test for `build_niah_trial` — **LANDED `e474f62`
      (2026-09-30)**: `transformer::vk_harness::tests::niah_builder_fixture_gated_guards`
      behind the module's own `fitted_v_tables` cfg — 5-trial grid mirroring the
      kv_plus_ladder NIAH schedule (row 0 IS the Bench-012 crash shape), asserts exact
      `seq_len`, BOS head, `answer_pos`, tail decode post-cut, password-span
      verify-decode, and the needle contiguous post-cut; loud SKIP without the
      gitignored gemma-2 fixture (`RIIR_INFER_SP_GGUF` override). Validated 3/3 at
      `--features fitted_v_tables` + 192/192 default.
- [ ] Incremental report write per arm in `kv_plus_ladder.rs` — still open (a real
      refactor of the Phase-D report block into a re-renderable fn; deliberately NOT
      half-landed at the end of the session that found it).
- [x] `--niah-only` flag: load `--table <path>` (skip calibration), skip the
      ladder/grid/validation, run only Phase C4 + D, write a NIAH-only report — **LANDED
      `d9acd5a`** (the reduced report marks the ladder/gates sections ABSENT by construction
      instead of rendering empty tables).
- [x] Rerun NIAH off `.benchmarks/012_kv_table_residual.bin` (sha
      `3a0f5333d6bafd63…`) — arms {f16, k-0.00, k-0.50, k-1.00} (the k-sched
      arm is moot: G-D recorded the schedule overfit) — and append the G-E
      direction rows to the Bench 012 gate doc — **DONE** (`.benchmarks/012b_niah_only_report.md`,
      6382 s wall under the sibling trainer's load; G-E rows in the gate doc).
