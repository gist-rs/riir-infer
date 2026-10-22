# Proposal 042 — Multi-Mac distributed inference: seal1 + seal2 over Thunderbolt 5 / Tailscale, on par with or beating exo / llama.cpp RPC / MLX

> **Status:** PROPOSED — filed 2026-10-22 (owner directive: "if exo, llama.cpp can do it,
> we must be able to do it too — both riir-train and infer (focus on infer, separate
> plan for train later) … aim to on par or beat them all"). Owner decisions D1–D3 below
> are OPEN; Plan 623 Phase 0 (measure the incumbents) is decision-free and may start.
**Branch:** `develop` · **Distill:** [Research 015](../.research/015_Multi_Host_Inference_exo_llamacpp_MLX_Distill.md) (exo `bc412e2`, llama.cpp `19ab5a5`, mlx `c6abf0d`, mlx-lm `0d4d3d0`, all under `.raw/`)
**Execution:** [Plan 623](../.plans/623_multi_host_pipeline_inference_lane.md) · **Hardware bring-up + seal2 routing:** [Issue 1109](../.issues/1109_seal2_cluster_bringup_and_routing.md)
**Training half:** DEFERRED to a separate riir-train plan (owner: "separate plan for train later") — see §Train.

## TL;DR

Two Mac Studio M5 Ultra boxes (`seal1` = this box, 96 GiB; `seal2` = `Mac17,15`, RAM
unmeasured) are cabled by ONE Thunderbolt-5 link (negotiated 80 Gb/s, measured on seal1
2026-10-22) and also share a tailnet. Every incumbent that pools two Macs (exo, llama.cpp
RPC, mlx-lm sharded) does it with one of two physics:

- **Pipeline-parallel (PP)** — split layers; buys **capacity** (models bigger than one
  box) and **multi-stream throughput**; single-stream decode is never faster than one box.
- **Tensor-parallel (TP)** — split every matmul; buys **single-stream speed** (~2×
  memory bandwidth) but pays ~2 allreduces per layer per token, so it only wins over
  **RDMA** (TB5 + macOS `rdma_ctl`), not TCP.

riir-infer has the layer loops (`forward_from_x`/`mark_x_as_input` on
`TernaryDeltanetGpuForward`, gemma2 `forward_layer_gpu`, the `forward_gpu_logits_handle_max_layer`
layer-cut precedent) and an in-process prefill→decode handoff (Issue 028,
`src/disaggregated.rs`) but **zero multi-host substrate** — no transport, no activation
byte format, no collectives (substrate-first survey 2026-10-22). This proposal adds them
as a pure-Rust, std-only lane and defines "beat" as **measured on the same two boxes,
same model, same quant**, against all three incumbents.

## What "on par or beat" means (the scoreboard — Plan 623 G-gates)

| metric | regime | incumbents measured | win condition |
|---|---|---|---|
| single-stream decode tok/s | model fits one box | riir-infer 1-box Metal, MLX 1-box, llama.cpp 1-box | PP: ≤ 5% slower than 1-box (overhead bound); TP-RDMA: faster than exo-TP-jaccl and llama.cpp `-sm tensor` RDMA |
| single-stream decode tok/s | model needs BOTH boxes | exo PP (ring + jaccl), llama.cpp RPC (TCP + RDMA), mlx-lm `--pipeline` | ≥ the best incumbent |
| aggregate decode tok/s @ 2/4/8 streams | either | exo (continuous batching), llama.cpp server slots | ≥ best incumbent |
| TTFT @ 4k / 32k prompt | either | all three | ≥ best incumbent (chunked pipelined prefill) |
| cold start → first token | either | all three | ≤ best incumbent (no weight upload, local mmap) |
| correctness | always | — | **G1: PP output byte-identical to the 1-box riir-infer path** (Issue 028 gate shape); TP: fixed-order sum, replay-deterministic (Issue 940) |

Every row carries box provenance (`riir-reflex/scripts/bench_preflight.sh`), `--release`,
and both boxes' state — a latency number without box state is not a measurement.

## Design (one paragraph per layer; detail in Plan 623)

1. **Substrate (riir-infer-core, no new deps):** `LayerRange { lo, hi }` on the forward
   entry points (CPU reference first, then Metal/CubeCL); an `ActivationFrame` byte format
   (`seq_id u32 | pos u32 | n_tok u16 | dtype u8 | payload`) + a `HybridCache` /
   per-range KV+GDN state serializer (closes the "no serializer" gap Issue 028 left
   for an out-of-process handoff); the stage handshake reuses `geometry_fingerprint`
   (`src/disaggregated.rs:90`) + the artifact BLAKE3.
2. **Transport + stage runtime (new crate, see D1):** std-only TCP (`TCP_NODELAY`,
   blocking vectored I/O, preallocated frames) over the TB bridge or the tailnet; stage
   PUSH, no hub; last stage samples and returns a 4-byte token id; ≥2-stream micro-batch
   round-robin; chunked pipelined prefill. Phase 3 adds an RDMA transport behind the same
   trait (verbs via `dlopen("librdma.dylib")`, the macOS TB5 path every incumbent uses).
3. **Planner:** probe per-host per-layer cost + link RTT/bandwidth at boot, split to
   equalise stage time subject to memory (exo splits by RAM only; llama.cpp by free mem).
4. **TP (Phase 4, gated on Phase 3's measured RDMA RTT):** generic column/row sharder
   from the arch layer table; fixed-order 2-rank sum.
5. **Speculative fill (Phase 5):** the verify batch IS the micro-batch, so PP decode
   stops idling half of each box — the lever none of the three incumbents has.

## Owner decisions (OPEN)

- **D1 — where the transport lives.** riir-infer's BOUNDARY.md owns "weights, quant,
  architectures, kernels, loaders" — not networking; riir-ai owns `riir-net`
  (tokio/axum, game-oriented) and riir-infer may not depend on any foreign `riir-*`.
  - **(a) RECOMMENDED:** new in-repo crate `crates/riir-infer-dist` — std-only (no
    tokio, no new crates.io dep; RDMA via `extern "C"` dlopen, no `libloading`), opt-in,
    never a default, macOS RDMA arm target-scoped. Rationale: llama.cpp keeps ggml-rpc
    inside ggml and MLX keeps distributed inside mlx — the collective is part of the
    inference substrate, and the per-token latency budget cannot afford riir-net's async
    runtime. Needs a BOUNDARY.md **Owns** row (lane-widening precedent: owner-gate D8,
    the audio row) — landed in the same commit as the crate, never before the ruling.
  - (b) substrate-only here (LayerRange + ActivationFrame + state serializer), transport
    in riir-ai on `riir-net::BinaryTransport`. Cost: riir-infer cannot self-test the lane;
    every public consumer (reflex/instinct) inherits riir-ai.
- **D2 — RDMA enablement.** `rdma_ctl status` = `disabled` on seal1 (2026-10-22); enabling
  needs `rdma_ctl enable` from the **Recovery-mode Terminal** on BOTH boxes + reboot, and
  incumbents require the Thunderbolt Bridge OFF on the RDMA port. A physical, owner-only
  act. Phases 0–2 run on TCP and do not need it; Phase 3–4 do.
- **D3 — the scoreboard models.** Recommended: (i) **Gemma-2 9B/27B** (all three
  incumbents + riir-infer's CubeCL gemma2 path — the apples-to-apples row); (ii) the
  **league Bonsai/Qwen3.5-hybrid ternary** posture (riir-infer's own moat; llama.cpp
  `qwen35` arch runs it, MLX/exo arm only if they load it); (iii) one **capacity row** —
  a model that does NOT fit 96 GiB at its quant, chosen after seal2's RAM is measured.

## Train (deferred, recorded so it is not lost)

The same transport (D1) carries gradient/LoRA-delta allreduce for riir-train; per the
modelless-first mandate the first train consumer is **not** backprop but distributed
calibration/capture (activation capture, act-aware fits) and the riir-train Issue 629
teacher lane if it outgrows one box. A separate riir-train plan is cut AFTER Plan 623
Phase 2 lands (the transport must exist and be measured first).

## seal2 and riir-rethink's open issues

Asked: "can we use seal2's GPU to solve all issues in riir-rethink/.issues?" — measured
answer in [Issue 1109 §Routing](../.issues/1109_seal2_cluster_bringup_and_routing.md):
seal2 directly unblocks the **hardware-gated** items (016 and 025 were "M5-Ultra-gated";
the hardware now exists — the remaining gate is the owner's serve/deploy call, which no
GPU can make), can host the pplx rig for 035 T4 and the 031 T4 probe, and is the natural
second node for 032's teacher lane via the deferred train plan. 024 and 034 are **not
GPU-blocked** (owner gates / a dead candidate lane) — seal2 does not move them.

## Risks

- **PP disappointment:** a model that fits one box decodes slower over PP — by
  physics, not by implementation. The scoreboard reports the 1-box arm next to every
  PP row so the capacity/throughput claim is never misread as a latency claim.
- **RDMA on macOS is new** (TN3205 UC queue pairs, ≥ 26.2; this box runs 27.0.1).
  Zero-copy MR over an `MTLBuffer` is a hypothesis; the fallback is one memcpy per side.
- **Riir-infer's single-box Metal arm must be competitive first** — if it loses to MLX
  on one box, no network cleverness recovers it; Plan 623 Phase 0 measures this first.
