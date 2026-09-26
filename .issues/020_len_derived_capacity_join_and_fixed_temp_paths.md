# Issue 020 — len_derived CAPACITY join + 2 fixed temp-path sites (the katgpt-rs sweeps' standing red here)

**Status:** OPEN — filed 2026-09-26 from the katgpt-rs drift sweeps (run at katgpt-rs `7cf2fc908`, the Issue-902 landing, on the 4090 box; riir-infer clean at HEAD `28b1566`). Measured; not fixed. Owner: this repo.

## Finding 1 — LIVE CAPACITY joined finding (len_derived ceiling 0)

```
CAPACITY crates/riir-infer-gpu/src/encoder_lane_cubecl.rs:361
         gather_rows_f32  rows_handle  [len: rows_host.len(]
```

The joined shape: HALF A — the kernel derives a dimension from a bound
buffer's **declared size** (`out.len()` inside `gather_rows_f32`); HALF B — a
bind whose declared size can exceed the live range. If a caller ever binds a
persistent / capacity-sized `rows_handle` (a struct field created
`client.empty(capacity)`), the declared size lies about the live range and the
kernel reads never-written memory — no panic, no NaN, a measured
identically-zero result (the canonical story: katgpt-rs
`len_derived_binding_audit.py`, riir-ai `3e00c93e0`).

**Repair direction (read the bind site, not the kernel):** `GatherRowsCubeCL::launch`
already takes `x_len`/`out_len` explicitly — the `rows` extent is the one
derived from `rows_host.len()` at bind time. If every current caller creates
`rows_handle` exactly `rows_host.len() * 4` bytes, the finding may adjudicate
as PERSISTENT-UPSTREAM (a struct-field bind, pinned on the katgpt-rs EYES
list with the field's creation read) rather than a defect — that judgement is
this repo's owner's. If any caller binds a larger persistent buffer, bind the
live count instead (`client.create_from_slice` of the ids, or a right-sized
sub-handle).

**Reproduce / validate after the fix:**

```bash
cd ../katgpt-rs && DOCS_GATE_PARTIAL_CLONE=1 python scripts/len_derived_drift_sweep.py --no-stability
```

The `riir-infer` row must read `CAPACITY=0` (the joined count is what the
`max_findings 0` wall asserts).

## Finding 2 — 2 fixed temp-path sites (shared_temp_path ceiling 0)

```
crates/riir-infer-laya/src/laya/riir/ane.rs::riir-laya-ane-cache
src/quant/exl3.rs::exl3-pack
```

`env::temp_dir().join("fixed_name")` is safe against sibling tests in one
binary and truncated by any concurrent PROCESS running the same code — two
concurrent runs share one `/tmp` (the measured 1-in-24 failure,
katgpt-rs Issue 832). Repair recipe (the workspace form):

```rust
std::env::temp_dir().join(format!("name_{}", std::process::id()))
```

**Reproduce / validate after the fix:**

```bash
cd ../katgpt-rs && DOCS_GATE_PARTIAL_CLONE=1 python scripts/shared_temp_path_drift_sweep.py
```

The `riir-infer` row must read `fixed=0`.

## Context

Both findings are why the katgpt-rs sweeps carry a standing red on this repo
since the carve moved the GPU kernel layer here — everything else in those
sweeps was closed at katgpt-rs `26ba6afdd` (the Issue-902 landing: the EYES
re-address + the measured floor re-pins). This issue is the actionable record
so the red is not a mystery line in someone else's output.

Katgpt-rs carries NO pin for these — the walls are 0 and the findings are
live. Fixing this issue is what turns both sweeps green on this repo.
