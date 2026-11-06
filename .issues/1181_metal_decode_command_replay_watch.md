# Issue 1181 — Metal decode command-replay watch lane (fusion idea, pre-registered gates)

**Status:** WATCH-CLASS FUSION IDEA — novelty TBD (published prior art: lithos-metal, cubecl #1993, CUDA Graphs); pre-registered gates below; likely-demotes on measurement. Do NOT implement before T1's measurement passes its bar.

**Unclaimed.** The trigger that arms this lane: the riir-ai issue 1165 owner flip (`RIIR_DECODE_RESIDENT` default) lands AND a T1 measurement prices per-token host-encode cost above the pre-registered demote bar.

## Pinned claim

metal-graph's `Graph -> GraphExec -> Launch` lifecycle (robbiemu/metal-graph @ `0cd855b7d23c44aa0f7511aa6f01f694c00da39`, Apache-2.0) is the freeze/thaw pattern applied to GPU command streams (instantiate = freeze+validate+plan, launch = thaw/use, patch = deterministic hot-swap of contract-validated slots). The un-owned application in OUR stack is a **Metal decode replay arm**: our CUDA prefill already ships a graphs arm (DEFAULT-ON, "+10.1% on the 4090", `crates/riir-infer-gpu/src/prefill_cuda_full.rs`), but the Metal serving lane (`crates/riir-infer-laya/src/laya/riir/metal.rs`) encodes per-op command buffers with no replay arm, and the resident decode substrate (plan 626 `resident_begin/resident_enqueue/resident_harvest` on `TernaryDeltanetGpuForward`, T1–T5 CLOSED 2026-11-05) still encodes per launch.

**This is NOT a novelty claim** — the mechanism is published prior art:
- **lithos-metal** (riir-refine kernel_opt corpus rule `gpu-resident-control-loop-static-icb-replay`, B223; research note `riir-refine/.research/248_lithos_metal_distill.md`) — GPU-resident decode control loop under static Metal ICB replay.
- **cubecl #1993** (riir-refine kernel_opt corpus rule `metal-icb-replay-rewrite-private-copy`, B319, upstream @ `c1d0125`) — the replay-degradation tax (10→23 ms over 20 unchanged replays on Apple drivers) and the remedy: a per-replay blit-copy of the commands into a private ICB + a cross-stream last-replay event fence.
- **CUDA Graphs** — the pattern's canonical form; our own prefill arm above.
- metal-graph itself — another instance, C-first.

## The two-sided precedent (why this is watch-class, not a plan)

- **Prefill WON on CUDA:** `RIIR_PREFILL_CUDA_GRAPHS` default-on in `prefill_cuda_full.rs` (+10.1% on the 4090, per the module doc).
- **Decode pot measured SMALL on 4090:** "the submit-amortization lane demotes (334 µs, no 1 ms pot)" (riir-ai `perf_rematch_league.md`, 4090-CUDA post-KB-era steady-state). That figure is a **demote example only, never a Metal bar** — different box, backend, and era.
- **Metal: unmeasured.** The M3/M5 decode rows are behind (0.888×/0.863×/0.842× at the 11-03 re-cell) with kernel-level levers named (ffn_gate_up) — graph-level orchestration is named nowhere. Plan 626's resident path already removed the queue-drain idle; what remains is the per-token host ENCODE cost, which nobody has measured.

## Gates (all pre-registered — run in order; stop at the first demote)

- [ ] **T0 — raw-Metal / boundary pre-gate.** The replay arm MUST ride the EXISTING metal-rs dep edge in riir-infer-laya (`metal = "0.31"` + `objc2 = "0.6"`, both optional, Cargo.toml:74-75 — the lane's MPS path already imports `metal::`). wgpu-hal / cubecl-wgpu expose no `MTLIndirectCommandBuffer` (zero hits across riir-infer vendor/ + src/ + crates/, checked 2026-11-06). If the arm lands outside riir-infer-laya, the dep-edge move is a BOUNDARY.md allowlist question resolved BEFORE any code.
- [ ] **T1 — encode-share measurement (the demote gate).** Measure per-token host ENCODE cost (command-encoding µs/token on the existing per-op encode path) vs decode wall µs/token at the post-1165-flip posture. bench_preflight PROVENANCE line quoted + box state (free RAM, power mode, commit-vs-limit) per the owner's box-state law — this is encode cost, NOT total host overhead (plan 626 already removed the idle). **Pre-registered demote bar: DEMOTE unless measured encode ≥ 1.0 ms/token.** Bar anchoring: 1.0 ms ≈ 6.3% of the amo M5-Ultra resident loop (15.92 ms/tok, plan 626 T4, ADJUDICATED 2026-11-03 on amo, quiet). The 1.0 ms absolute bar is box-independent; a T1 run on M3 (or any other box) re-derives the PERCENTAGE there, never the absolute bar.
- [ ] **T2 — BUILD the B319 remedy (cost line in the break-even estimate).** Implement the per-replay private-copy rewrite (the cubecl #1993 shape: blit-copy the commands alone into a private ICB before EVERY replay + the cross-stream last-replay event fence) in the replay arm itself. riir-infer's vendored cubecl 0.11.0-pre.2 carries NO ICB/capture path and `cubecl-metal` is ABSENT from the lockfile (checked 2026-11-06); the remedy lives only at upstream tip (`c1d0125`) and must be BUILT, not linked. Cost: a few µs/replay per B319's source — non-zero, per-replay, and priced into T4's break-even.
- [ ] **T3 — eligibility = fully-static subgraphs only.** Nodes with ANY patchable field or scalar binding are INELIGIBLE for the replay group; every binding needs a declared resource range with known access (the riir-refine corpus's B223/B319 vocabulary; the metal-graph `mg_icb_dispatch_eligible` shape). Multi-node groups fall back to direct encode until hazard-safe grouping is proven.
- [ ] **T4 — patch contracts + break-even evidence.** Patchable slots validate against DECLARED resource contracts (range/access), per-aspect patchability flags, declared grid ceilings; the arm reports `break_even_launches = instantiate_cost / per_launch_savings` and a savings ≤ 0 result is a LOUD no-break-even verdict (the evidence-suite law; riir-refine research 383 candidates 2-3 carry the full rule shapes for a future corpus batch — riir-refine plan 350, Batch 341, staged unclaimed).

## Cross-references

- riir-refine research 383 (the distill verdict; its lane-intel row is the conversation-side record; the correction note there supersedes the "our tree carries the remedy" claim in its Drops row AND in the same-day zed-m3-metalgraph queue row).
- riir-refine plan 350 (Batch 341: the metal-graph corpus distill, staged unclaimed behind B339/B340).
- riir-infer plan 626 (the resident decode substrate, CLOSED with the owner flip pending) + riir-ai issue 1165 (the consumer half + league re-cell).
- riir-ai `perf_rematch_league.md` (the 4090-CUDA demote example + the decode league rows).

## Provenance

Source inspected 2026-11-06; full clone at the pin, removed at the riir-refine verdict landing; every quote verified at the pin that session (riir-refine research 383 is the quote authority — this issue re-states gates, not quotes). Verdict path: the katgpt-rs `research` skill §5 Claude ping-pong, 3 rounds (REVISE → corrections → AGREE), session `83888082-92ea-43eb-b98e-923c61149253` — the REVISE round caught the false "our substrate carries the remedy" claim now corrected in research 383.
