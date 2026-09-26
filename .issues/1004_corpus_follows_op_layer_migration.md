# Issue 1004 — riir-ai inference-adjacent corpus follows the op-layer migration (2026-09-26 sweep verdict: zero misplaced today)

**Status:** OPEN — trigger-gated on riir-infer Plan 611 T7 (op-layer unification, in flight) + Issue 998 S8 (edge-drop); the 2026-09-26 sweep found ZERO further docs/benches to move today.

## The 2026-09-26 sweep (owner ask: "can we move more?")

After the 2026-09-26 adoption batch (998/1003/610/041/870/871/002 — see Issue 998),
a full sweep of riir-ai `.benchmarks/ .docs/ .issues/ .plans/ .proposals/
.research/` for remaining infer-substrate material found **zero misplacements**.
Everything infer-LOOKING in riir-ai measures a harness that lives in riir-ai today:

| Cluster (riir-ai paths) | Harness home (verified 2026-09-26) | Moves when |
|---|---|---|
| Issue 879 T3 KV×weight-quant NLL (`.benchmarks/872_/874_/876_…`) | `crates/riir-gpu/tests/bench_874_issue879_t3_kv_weight_quant_nll.rs` | riir-gpu KV/attention kernels migrate to riir-infer-gpu (Plan 611 T7) |
| Issue 884 prefill-MMQ format rungs (`.benchmarks/878_/882_…`, `.plans/562_/572_`) | `prefill_mmq_v2` riir-gpu ingest surface | same |
| Issue 980 Bonsai-2 rotation 4090 (`.benchmarks/940_…`, `.plans/600_/602_`) | riir-gpu cudarc lanes (`forward_from_x_devpos`, `new_graph_ready`); consumes riir-infer-core `deltanet::rotation` config only | same |
| GDN/prefill/decode campaigns (issue734 family 700–749; 583–720; 487/606/609/692) | `crates/riir-gpu/{tests,src,benches}` + `crates/riir-engine/benches` | same |
| Perf league (`.benchmarks/rematch/`, `.docs/09_performance/`, 877/941/954) | riir-gpu + `scripts/perf_rematch.sh` — gate-pinned (`watch_coverage_gate`, league watch rows) | only with the league lane itself; the gate pins re-point in the same commit |
| Kimi-K3 research 327–332 | the notes' own Routing lines say katgpt-rs (`katgpt-attn` mla/kda, `katgpt-transformer` moe); the kernels (`kda/mla/moe_cubecl.rs`, `kimi_k3_gpu_forward.rs`) are riir-gpu's; program = riir-ai Proposal 032 | OWNER-GATED, and NOT to riir-infer — the candidate destination is katgpt-rs (public) per the Routing lines |
| Research 085 (quant-outlier / LoRA guard) | LoRA-training context (riir-ai/riir-train) — riir-infer is modelless | never (wrong domain for riir-infer) |
| Early engine benches 023–033; `.docs/02_crates/riir_gpu*.md`; `12_inference/` | riir-engine/riir-gpu; `12_inference/` was a misnomer holding the Quest Style Bridge doc (renamed `12_quest_style/` 2026-09-26) | with the kernel migration / never (misnomer fixed) |

## The rule (the 870/871 precedent)

A bench/doc moves to riir-infer iff the harness/instrument it measures lives in
riir-infer **that day**. Bench 871 moved because its T1 cert target
(`issue879_gdn_quant_certification`) is a riir-infer test; benches 872/874/876
stay because their T3 harness is a riir-gpu test. Same rule for docs.

## Tasks

- [ ] When Plan 611 T7 lands a migrated op set in riir-infer-gpu: re-run this sweep, move the corresponding bench records + docs, re-point citations in the same commit.
- [ ] When Issue 998 S8 edge-drop lands: move the league corpus — or re-point the gate pins first (`watch_coverage_gate` + `scripts/perf_rematch.sh` name riir-ai paths).
- [ ] Owner decision (separate, not this issue's default): research 327–332 routing — katgpt-rs is public; moving internal research notes there is a Research 003 posture call.

## Refs

- riir-infer Issue 998 (repo promotion) · 1003 (carve remains) · Plan 610 (slice 1) · Plan 611 (T7 op-layer unification, in flight 2026-09-26)
- riir-ai Proposal 032 (Kimi-K3 native support program)
