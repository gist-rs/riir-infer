# Plan 623: Multi-host inference lane — layer-range stages, activation frames, TCP→RDMA transport, TP, speculative fill

**Status:** Proposed — 2026-10-22. Phase 0 is decision-free (measurement only) and may
start once Issue 1109 T0 (bmo reachable) is done; Phase 1 substrate is decision-free
(riir-infer-core, no deps); Phase 2+ waits on [Proposal 042](../.proposals/042_multi_mac_distributed_inference.md) D1 (crate home); Phase 3+ on D2 (RDMA enabled).
**Date:** 2026-10-22
**Research:** [Research 015](../.research/015_Multi_Host_Inference_exo_llamacpp_MLX_Distill.md)
**Target:** `src/` (LayerRange + ActivationFrame + state serializer), `crates/riir-infer-gpu/` (range-forward on the Metal/CubeCL paths), `crates/riir-infer-dist/` (D1-a; transport + stage runtime + planner). Cargo feature `multi_host` (opt-in, never default) on core; the dist crate is opt-in as a whole.
**Hardware:** amo (M5 Ultra, 96 GiB, macOS 27.0.1) ⇄ bmo (`Mac17,15`) — one TB5 cable @ 80 Gb/s + tailnet (amo `100.111.186.76`, bmo `100.100.117.44`).

---

## Goal

Run one model across amo + bmo from pure Rust, with output byte-identical to the
1-box riir-infer path (PP) or replay-deterministic (TP), and beat exo / llama.cpp RPC /
mlx-lm on the Proposal-042 scoreboard **measured on the same two boxes**.

## Phase 0 — Measure the incumbents and our own 1-box arm (no code in riir-infer)

- [ ] **T0.1** Build/install the incumbents from the pinned `.raw/` commits on BOTH boxes
  (llama.cpp `rpc-server` + `llama-cli`/`llama-bench` with `-DGGML_RPC=ON`; mlx + mlx-lm;
  exo) — record exact commits and build flags. No Python dep enters riir-infer: the
  harness lives in `scripts/` as shell + a Rust bench driver.
- [ ] **T0.2** Link microbench: TCP RTT + one-way latency for 4 B / 16 KB / 1 MB / 64 MB
  frames over (a) TB bridge, (b) tailnet; bandwidth for 64 MB. Rust std-only
  (`examples/link_probe.rs`). After D2: same over RDMA via `ibv_*` / MLX `allreduce_bench`.
- [ ] **T0.3** 1-box arms for the D3 models: riir-infer Metal vs MLX vs llama.cpp Metal —
  decode tok/s, TTFT @4k/32k. **If riir-infer's 1-box arm loses, file the gap as its own
  issue first — the multi-host claim is meaningless on a losing single-box base.**
- [ ] **T0.4** 2-box incumbent arms: llama.cpp RPC (TCP; RDMA after D2), mlx-lm
  `sharded_generate --pipeline` + TP (ring; jaccl after D2), exo (PP + TP). Every metric
  of the scoreboard; `.benchmarks/` record with both boxes' provenance.

## Phase 1 — Substrate in riir-infer-core (feature `multi_host`, zero new deps)

- [ ] **T1.1** `LayerRange { lo, hi }` + range-forward on the CPU reference paths
  (`deltanet/ternary_forward.rs` `qwen_deltanet_ternary_layer_body` loop; gemma2
  `forward_gemma2_layers`). Embedding only on the stage with `lo == 0`, final norm +
  lm_head only on the stage with `hi == n_layer`. **G1:** stage(0..k) ∘ stage(k..n) ==
  full forward, bit-identical, every k, both archs.
- [ ] **T1.2** `ActivationFrame` (`seq_id u32 | pos u32 | n_tok u16 | dtype u8 | len u32 |
  payload`), little-endian, bytemuck-castable payload; bf16 and f32 dtypes; decode-only
  fast path with no allocation (G4 alloc-free). bf16 on the wire is a LOSSY surface —
  f32 is the G1 posture, bf16 rides its own per-family retention gate (lossy-surface law).
- [ ] **T1.3** Per-range state serializer for `HybridCache` (KV rows + GDN state for the
  range's layers) — closes Issue 028's out-of-process gap; round-trip bit-identity test;
  wire size matches the `src/disaggregated.rs:44-52` budget.
- [ ] **T1.4** Stage handshake record: arch, `geometry_fingerprint`, artifact BLAKE3,
  `LayerRange`, dtype — refuse on any mismatch (never a silent wrong-shape stage).
- [ ] **T1.5** Range-forward on Metal/CubeCL: `TernaryDeltanetGpuForward::forward_from_x`
  / `mark_x_as_input` (the existing upload-x seam) and gemma2 `forward_layer_gpu` loop with
  `[lo, hi)`; G1 bit-identity vs the full GPU forward.

## Phase 2 — Stage runtime over TCP (needs D1)

- [ ] **T2.1** `crates/riir-infer-dist`: `Transport` trait (`send_frame`, `recv_frame_into`)
  + std TCP impl (`TCP_NODELAY`, `SO_SNDBUF/RCVBUF` sized for prefill chunks, vectored
  I/O, preallocated rings, bind to a named interface only). Peer authentication: pinned
  peer key (BLAKE3-keyed MAC over the handshake + frame counter) — not an RPC surface.
- [ ] **T2.2** Two-stage runner: stage 0 = embed + layers [0,k); stage 1 = layers [k,n) +
  norm + lm_head + sampler → 4-byte token id back to stage 0. PUSH, no hub, one one-way
  hop per boundary per token. **G1:** 2-box greedy output byte-identical to 1-box over
  the Issue-028 battery prompts.
- [ ] **T2.3** Chunked pipelined prefill (chunk c on stage 1 while chunk c+1 on stage 0,
  double-buffered frames) + KV/GDN state lives on its own stage (no state traffic in
  steady state).
- [ ] **T2.4** Micro-batch round-robin for ≥2 streams (fills the PP bubble); continuous
  batching admission; chunked prefill interleaved with decode (exo's prefill-blocks-decode
  gap).
- [ ] **T2.5** Planner: boot probe of per-layer cost per host (prefill + decode) and link
  RTT/bw (T0.2 code), split to equalise stage time under memory; manual override.
- [ ] **T2.6** Tailscale posture: the same runner over the tailnet address; a measured row
  (the WAN/fallback lane exo cannot do — multicast discovery).
- [ ] **T2.7** G2 scoreboard rows (Phase-0 harness, same models): PP overhead vs 1-box;
  vs llama.cpp RPC-TCP, mlx-lm ring PP, exo PP-ring. `.benchmarks/` record.

## Phase 3 — RDMA transport (needs D2)

- [ ] **T3.1** Verbs FFI: `dlopen("librdma.dylib")` + the ~16 `ibv_*` symbols, the inline
  `post_send/post_recv/poll_cq` re-implemented via `ctx->ops` against the macOS 26.2+ SDK
  header; UC QP bring-up INIT→RTR→RTS with IPv4-mapped GID; bootstrap over the TCP
  side channel. Std-only.
- [ ] **T3.2** Zero-copy probe: register an MR over a shared-storage `MTLBuffer.contents()`
  — if accepted, frames go GPU-buffer → wire with no memcpy (both incumbents bounce-copy
  twice); if refused, record the refusal and keep one memcpy per side.
- [ ] **T3.3** Small-message decode path (single WR, inline header) vs striped large frames
  for prefill; MTU probe (incumbents fix 1024 / 128 KiB strides).
- [ ] **T3.4** GPU↔comm sync: CPU comm thread on `MTLSharedEvent` signal values; no per-
  layer command-buffer commit on the critical path.
- [ ] **T3.5** Scoreboard rows vs llama.cpp RPC-RDMA and exo/MLX jaccl.

## Phase 4 — Tensor-parallel (gated on T3.5's measured RDMA RTT: go only if 2·L·RTT < ½ the 1-box decode step)

- [ ] **T4.1** Generic sharder from the arch layer table: column (q,k,v,gate,up), row
  (o,down) + allreduce; kv-heads divisibility check; ternary/quant packs split on
  block boundaries.
- [ ] **T4.2** Allreduce: fixed-order 2-rank sum (identical on both ranks → deterministic,
  Issue 940); overlap the o_proj allreduce with the MLP column matmuls; fuse residual add.
- [ ] **T4.3** Scoreboard: single-stream decode vs exo TP-jaccl (claims "up to 1.8×") and
  llama.cpp `-sm tensor` RDMA.

## Phase 5 — Speculative fill

- [ ] **T5.1** Draft on stage 0 (dflash/speculative substrate already in core), verify
  batch rides the pipeline as the micro-batch; acceptance-rate-weighted scoreboard row.
  Lossless by construction (verify is exact) — G1 byte-identity holds.

## GOAT gate (promotion = opt-in → default for the lane crate's consumers)

- G1 PP byte-identity vs 1-box (every phase); TP replay-determinism.
- G2 `--release`, both boxes' provenance; beats the best incumbent on at least the
  capacity row and the aggregate-throughput row; never > 5% PP overhead on the fits-one-box row.
- G3 feature-off byte-identity of riir-infer-core (the `multi_host` feature compiles away).
- G4 decode frame path alloc-free.
- Security: no unauthenticated frame accepted (negative test with a wrong key).
