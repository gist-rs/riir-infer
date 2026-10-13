# Issue 047: lib test binary crashes at teardown under the parallel harness — `deltanet_inference`-only posture

**Status:** OPEN — filed 2026-10-13 (4090 box, found while landing Issue 046's required-features row, which made this posture a readable build). Pre-existing: reproduces with Issue 046's changes stashed.

## The finding

`cargo test --features deltanet_inference --lib` (the PLAIN deltanet posture —
no ternary stack) exits with `STATUS_STACK_BUFFER_OVERRUN` (0xc0000409) at
TEARDOWN, after every test has already printed `ok`:

```
test result: ok. 312 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out
process didn't exit successfully: riir_infer_core-*.exe (exit code: 0xc0000409)
```

## Evidence (2026-10-13, 4090 / Win11, develop @ a4ed1e6 + issue-046 files)

| posture | harness | result |
|---|---|---|
| default features | parallel (`cargo test --lib`) | **ok. 264 passed; clean exit** |
| `deltanet_inference` | parallel (`cargo test --lib`) | all pass, then **0xc0000409 at teardown** — FLAKY: 2 crashes, 1 clean across 3 runs |
| `deltanet_inference` | `--test-threads=1` (binary direct) | **ok. 312 passed; exit 0** (184 s) |
| `deltanet_inference` | `--test-threads=2 / 4 / 8 / 16` | **ok. 312 passed; clean** every run |
| `deltanet_inference` | parallel, Issue-046 changes stashed | **same crash** — pre-existing |

So: posture-specific (only the plain-deltanet build), concurrency-dependent
(clean at every FIXED thread count tried; flaky only at the default count =
`available_parallelism()`, 24 logical CPUs here), post-test (the harness
summary prints before the death when it prints at all — one crash lost even
the summary to the fastfail). Classic shape for a late thread touching freed
globals at process teardown — rayon pool teardown racing static destruction,
or a detached test thread outliving its statics. FLAKY, so a filter-bisect
will not converge cheaply; reproduce first with repeated default-count runs
before any bisect (expect ~2/3 crash rate).

## Not caused by

- Issue 046's `[[test]]` row + `on_decode_pool` cfg (stashed-reproduces).
- The spec test itself (integration target, not in `--lib`).

## Suspicion order (who to bisect)

The ~48 tests this posture adds over default are the `deltanet::*` unit tests
(forward/reference/rotation/weights/minimal_activation_cache). Under
`--test-threads=1` they all pass — the crash needs INTERLEAVING, so the pair
matters, not one test. Suggested method: `cargo test --features
deltanet_inference --lib -- --test-threads=N` for N ∈ {2,4,8} to find the
minimum thread count, then bisect by filter subsets (run the binary in
`target/debug/deps/` directly with `--skip`/substring filters; no cargo
lock needed).

## Why nobody saw it

The plain `deltanet_inference` lib posture is not in any standing gate: the
repo's traveled postures are default (CI) and the ternary/full stacks
(integration tests with their own required-features). Issue 046's row made
`--features deltanet_inference` a build a reader will actually run.
