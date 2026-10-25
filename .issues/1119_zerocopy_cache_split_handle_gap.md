# Issue 1119 — the zero-copy weight cache covers the FUSED handles only; the prefill dispatches read the SPLIT handles — the metal-tensor path is structurally dead at 10 of 12 sites, and the fall-through is SILENT

**Status:** OPEN — filed 2026-10-25 (found live by the i1122 gate1 cell's green-zero; fix LANDED same session, the cell re-run is the acceptance test)
**Claimed:** zed-m5ultra-1122, epoch 1761447191, expires 2026-10-27T03:30:00+00:00

## The defect (measured, then root-caused in source)

`riir-ai` Issue 1122 gate1 (`gate1_site_isolated_tax`, first live run 2026-10-25 18:57–19:03,
`ok. 1 passed`) printed:

```
G1 single-site: argmax base=248046 site=248046, max_abs 0.000e0 — PASS   ← bit-identical
[gate1] site-timing rows: 0
ffn_gate in-pipeline dispatch: n=0 mean=0.000ms
```

Exact-0 G1 + zero timing rows + n=0 dispatches = the "single-site" posture ran the
identical simdgroup code. The site mask, the timing instrument, the count floor — all
fine; the ROUTE was never taken. Cause chain (all in
`ternary_deltanet_gpu_forward.rs`):

1. The zero-copy branch requires `w.zerocopy_cache` on the DISPATCHED weight handle.
2. The construction-time cache list caches `in_proj_concat, out_proj, gate_up_proj,
   down_proj, attn_wq, attn_wkv, attn_wo` — the FUSED/old-handle set.
3. The prefill dispatches read `in_proj_qkv/in_proj_z/in_proj_a/in_proj_b` (GDN sites),
   `gate_proj/up_proj` (FfnGate/FfnUp sites — `PrefillSite::FfnGate` at the
   `&layer_w.gate_proj` call), `wq/wkv` (AttnQ/AttnKv locals) — DISTINCT
   `TernaryHandle` fields (from_weights, not aliases of the fused ones).
4. Overlap = `out_proj` + `down_proj` only. Everywhere else `w.zerocopy_cache` is
   `None` → the `let (Some, Some)` fails SILENTLY → fall-through to simdgroup.

The silent branch is the same class Issue 727 H3 fixed for the host-round-trip path
("a mis-configured A/B would measure CubeCL twice without a trace") — the zero-copy
path never got its twin.

## Blast radius on the recorded verdicts

- **Bench 998** (the lever-3(a) e2e closure, 0.55–0.79× at pp2048/pp4096): its arms
  flipped the same posture flags. The split-handle FFN dispatch predates the run
  (present at riir-infer `1c33b97` 10-24 20:05; Bench 998 ran ~21:00), so the closure
  measured a pipeline where the metal-tensor path could engage at out_proj/down_proj
  only — NOT at the FFN gate/up sites that own ~60% of the prefill wall (cell-1,
  Bench 995). The 0.55–0.79× numbers are REAL for the state they measured; the SCOPE
  of the verdict ("the class loses e2e") over-generalizes by exactly the engaged-site
  axis. This is the concrete mechanism i1122's "the per-launch→e2e tax is unexplained"
  was probing.
- **i1122 gate2** (ktile64 e2e, first live run 19:06+): same partial-engagement state
  (its verdict also carries real contamination advisories — separate axis, see the
  i1122 registering note).

## The fix (LANDED this session, uncommitted → commit ref on close)

1. Construction cache list += `in_proj_qkv, in_proj_z, in_proj_a, in_proj_b,
   gate_proj, up_proj` (the split set) — ~+150–250 MiB GPU-side cache on the
   PQ2_0 league model, construction-time only.
2. `warn_zerocopy_cache_missing_once(site)` — per-site one-time loud warn when a
   flagged+admitted site has no cache (the 727-H3 twin); fall-through behavior
   unchanged.
3. Clippy clean at `metal_tensor_gemm,ternary_gemm_batched` (`/tmp/i1122_fix`, 27s).

## Acceptance (what closes this issue)

`gate1_site_isolated_tax` re-run at a quiet window must print
`[gate1] site-timing rows: >0` and `ffn_gate ... n>0` — the instrument engaged.
Then the i1122 gate-1 tax read (mean_ms vs the Bench-1113 cell-2 absolutes) is
ADMISSIBLE for the first time. Until then every "metal-tensor vs simdgroup" e2e
number on this tree is a partial-engagement measurement.

## Boundary note

Fix lives HERE (the forward + cache are riir-infer-gpu substrate); riir-ai's
bench_1122 test needed no change — it correctly exposed the gap. Consumer record:
riir-ai `.issues/1122` (the campaign issue).
