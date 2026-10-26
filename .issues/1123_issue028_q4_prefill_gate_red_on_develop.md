# Issue 1123 — issue028 dual-format q4-prefill gate red on develop (`ProjWeights::Ternary` assert)

**Status:** Open — filed 2026-10-26 from the plan-623 Phase 1 validation sweep (the
pre-existing-bug note; NOT a plan-623 regression — verified below).

## Finding

`cargo test --test issue028_disaggregated_handoff --features deltanet_ternary_inference`
fails on `develop` @ `7a01f05`:

```text
test dual_format_q4_prefill_pair_loads_and_runs ... FAILED
thread 'dual_format_q4_prefill_pair_loads_and_runs' panicked at
tests/issue028_disaggregated_handoff.rs:479:5:
assertion failed: matches!(dec.layers[0].in_proj_qkv, ProjWeights::Ternary(_))
```

The other 8 tests in the battery pass.

## Verified NOT a plan-623 regression

The failing assert was reproduced at PRISTINE HEAD `7a01f05` in a detached worktree
(`/tmp/riir-infer-623-verify`, since removed) with an isolated
`CARGO_TARGET_DIR` — the plan-623 working-tree changes (range-forward refactor,
`multi_host` feature) were not present. The test reads the dual-format q4
container path (`synth_tensors_bonsai2_q4_prefill` → loader projection-kind
dispatch) — an area the Issue-1117 flashmemory lane and the Issue-028 T4 S2
history both touch.

## Suspects / next read

1. Did the q4-K payload shape change under the q4 prefill synth
   (`quantize_row_q4_k` side) while the loader's dispatch expectation
   (`layers[0].in_proj_qkv` must stay ternary on the DECODE copy) stayed
   fixed? Read the assert's intent first — the decode copy must be ternary;
   if the synth now leaks q4 into the decode set, the FIXTURE is wrong; if
   the loader misroutes, the LOADER is wrong.
2. `git log -p -- tests/common/synth_bonsai2.rs src/quant/` around the last
   q4-related landing.
3. Likely custody: the Issue-1117 lane (claimed, live) — coordinate before
   fixing if their WIP overlaps `src/quant/` or the loader dispatch.
