# Research 015 — Multi-host LLM inference: exo, llama.cpp RPC, MLX distributed — distill + beat-levers

**Date:** 2026-10-22 · **Filed for:** [Proposal 042](../.proposals/042_multi_mac_distributed_inference.md) / [Plan 623](../.plans/623_multi_host_pipeline_inference_lane.md) / [Issue 1109](../.issues/1109_seal2_cluster_bringup_and_routing.md)
**Sources (checked out to `.raw/`, gitignored, shallow):**

| repo | commit | date | licence |
|---|---|---|---|
| `exo-explore/exo` | `bc412e2` | 2026-10-13 | Apache-2.0 |
| `ggml-org/llama.cpp` | `19ab5a5` | 2026-10-22 | MIT |
| `ml-explore/mlx` | `c6abf0d` | 2026-10-22 | MIT |
| `ml-explore/mlx-lm` | `0d4d3d0` | 2026-10-22 | MIT |

All `file:line` cites below are relative to `.raw/<repo>/` at those commits.

⛔ **None of the three repos publishes a 2-Mac tokens/s number as text.** exo ships
Jeff Geerling chart images (4× M3 Ultra) and the README claim "up to 1.8× on 2 devices,
3.2× on 4" (TP + RDMA, `exo/README.md:25-27`); MLX claims RDMA is "an order of
magnitude lower latency than ring" (`mlx/docs/src/usage/distributed.rst:200-203`);
llama.cpp has no RPC/RDMA figure at all (one tuning note: 128 KiB stride best of
32/64/128/256, `ggml/src/ggml-rpc/transport-apple.cpp:27`). **The baseline is therefore
something we MEASURE on seal1+seal2 (Plan 623 Phase 0), never a quoted number.**

## 1. The three stacks in one table

| axis | exo | llama.cpp RPC | MLX / mlx-lm |
|---|---|---|---|
| control plane | Zenoh router + custom IPv6-multicast discovery (`rust/networking/src/discovery.rs`), event-sourced state, bully-ish election (`src/exo/shared/election.py`) | client is a HUB; one `rpc-server` per remote, sequential single-client accept loop (`ggml-rpc.cpp:2817-2826`) | `mlx.launch` + hostfile JSON; ssh-driven discovery (`python/mlx/_distributed_utils/config.py`) |
| data plane | delegates to MLX distributed (ring / jaccl) | own protocol, 23 cmds, packed host-endian structs, 296-B `rpc_tensor` descriptors (`ggml-rpc.cpp:36-82`); TCP or Apple RDMA (`transport-apple.cpp`) | ring = TCP sockets, spin threads (`mlx/distributed/ring/ring.cpp`); jaccl = libibverbs via `dlopen("librdma.dylib")` (`jaccl/lib/rdma.cpp:39-61`) |
| partitioning | PP split ∝ free RAM (largest-remainder, `src/exo/master/placement_utils.py:47-122`) or TP via 11 hand-written per-arch sharders (`auto_parallel.py:456-767`) | `--split-mode layer` ∝ free mem / `--tensor-split`; RPC devices get the FIRST layers (`src/llama-model.cpp:1640-1699`); `-sm tensor` meta-device TP | PP reverse-split (`mlx_lm/models/pipeline.py:16-38`); TP column(q,k,v,gate,up) / row(o,down)+`all_sum` (`models/llama.py:231-259`) |
| per-token decode traffic (PP) | send `[B,1,hidden]` + blocking `mx.eval` both sides + **all_gather of last hidden to every rank** (`auto_parallel.py:120-194`) | fire-and-forget SET_TENSORs (hidden + pos + KQ mask + KV idx) → GRAPH_RECOMPUTE (12 B) → **one blocking GET round trip** per split boundary | `recv_like` → layers → `send` → `all_gather(h)` (`deepseek_v3.py:347-364`) |
| per-token decode traffic (TP) | ~2 `all_sum`/layer, hidden×2 B each | ~2 allreduce/layer, server↔server on port+1000, bf16 ≥32K elems, **blocking + backend sync** (`ggml-rpc.cpp:2224-2403`) | 2 `all_sum`/layer → ~160 latency-bound allreduces/token at 80 layers |
| weights | each node downloads its layer-range FILES from HF (whole-file granularity) | **client uploads weights over the wire every start**; >10 MiB tensors hash-cached (byte-at-a-time FNV, blocking round trip; `:313-321, :840-846, :1160`) | lazy load + drop unused layers; PP fetches only needed files (`mlx_lm/utils.py:759-794`) |
| RDMA (TB5) | JACCL; needs full RDMA mesh, macOS ≥ 26.2, `rdma_ctl enable` in Recovery, TB bridge OFF, identical OS builds (`README.md:277-304`) | UC QPs, 16×128 KiB ring/direction, **full 128 KiB frame per SEND even when ~16 KB used**, 2 memcpys, pure spin-poll (`transport-apple.cpp:30-48, 403-482`) | UC QPs, MTU 1024, two-sided SEND, **bounce buffers 4 KB–512 KB + memcpy both ends**, 2nd cable unused below 512 KiB (`rdma.h:17-23`, `mesh_impl.h:13`) |
| GPU↔comm sync | `mx.eval` fences around every send/recv; `MLX_METAL_FAST_SYNCH=1` | Metal buffer `is_host=false` → malloc + GET + memcpy + set (`ggml-backend.cpp:634-648`) | collectives run on the CPU stream (`ring.cpp:431-433`) → GPU→CPU fence each side; fast-synch documented "not reliable, can deadlock" (`distributed.rst:366-374`) |
| prefill/decode | chunked staggered prefill ≥4096 (`generate.py:163-278`); prefill BLOCKS decode (`TODO.md:7`); optional disaggregated prefill (msgpack KV chunks over TCP) | PP overlap only across ubatches inside one `llama_decode` (`GGML_SCHED_MAX_COPIES=4`); single-stream decode serial across hosts | no micro-batching; PP buys capacity, not speed |
| speculative decode | none | (local only) | none in the distributed path |
| security | — | "fragile and insecure, never on an open network"; no auth, cache trusts client hashes (`tools/rpc/README.md:3-5`) | — |

## 2. The physics (why "beat" has two different meanings)

- **Pipeline-parallel (PP) single-stream decode is serial**: token latency =
  Σ stage compute + (stages−1) × hop. With the model fitting on one box, PP is
  ALWAYS slower than one box for one stream. **PP buys CAPACITY** (a model larger
  than one box's unified memory) and **aggregate throughput** only if ≥2 streams /
  micro-batches keep both stages busy.
- **Tensor-parallel (TP) buys SPEED** (each box reads half the weights → decode is
  memory-bandwidth-bound → up to ~2× minus collective cost), but pays 2 allreduces
  per layer per token. At ~64 layers that is ~128 latency-bound collectives/token:
  a 50 µs TCP hop costs ~6.4 ms/token (kills the gain at small models); a ~5–10 µs
  RDMA hop costs ~0.6–1.3 ms (keeps it). **TP is only worth it over RDMA.**
- Per-token PP payload is tiny: hidden × 2 B (bf16) ≈ 8–16 KB. Bandwidth is
  irrelevant at decode; **latency + sync overhead is everything.** Prefill is the
  bandwidth case (T × hidden × 2 B per boundary per chunk).

## 3. Beat-levers (each is a Plan 623 task; none is a quoted win until measured)

1. **Zero weight traffic + zero download:** both hosts mmap the SAME GGUF/safetensors
   locally (one rsync of the artifact, BLAKE3-checked) and load only their layer range;
   the `geometry_fingerprint` (blake3, `src/disaggregated.rs:90`) + artifact digest are
   exchanged at handshake. Beats llama.cpp's per-start upload and exo's HF fetch.
2. **Stage-to-stage PUSH, no hub, no request/response:** the producer sends the
   activation frame the moment its last layer retires; the consumer is blocked in
   recv. One one-way hop per boundary per token (llama.cpp: blocking GET round trip;
   exo/MLX: send + fence + extra all_gather).
3. **Logits on the last stage only, broadcast a 4-byte token id** (+ pos) back to
   stage 0 — not an all_gather of the hidden state to every rank (exo, mlx-lm).
4. **Static per-stage decode graph, locally-derived masks/positions:** wire frame =
   `{seq_id u32, pos u32, n_tok u16, dtype u8, hidden[n_tok × n_embd]}`; no graph
   or descriptor traffic (llama.cpp: 296-B descriptors, re-serialisation per shape).
5. **Cost-model split, not RAM-proportional:** probe per-host per-layer latency
   (both prefill + decode) and link RTT/bandwidth at boot, split to equalise stage
   time subject to memory. exo has no probing despite its README (`TODO.md:4`).
6. **Fill the pipeline bubble:** (a) ≥2-stream micro-batch round-robin; (b)
   speculative decode where the verify batch is the micro-batch (the workspace already
   owns dflash/speculative substrate); (c) chunked prefill interleaved with decode
   (exo: prefill blocks decode).
7. **Transport ladder:** `std::net` TCP (`TCP_NODELAY`, blocking `writev`/`readv`,
   pre-sized buffers, no per-message alloc) over the Thunderbolt bridge → same over
   Tailscale (WAN/fallback; exo cannot cross Tailscale — multicast discovery) →
   RDMA (verbs via `dlopen`, small-message inline/one-WR path for decode, large
   striped frames for prefill, MR registered on the shared-storage `MTLBuffer` to
   drop both bounce memcpys — **a hypothesis to verify on macOS's verbs**, Plan 623 T3.2).
8. **GPU↔net sync without a full command-buffer round trip:** a CPU comm thread
   waiting on `MTLSharedEvent` values, never a fence on the critical path per layer.
9. **Generic TP sharder from tensor metadata** (column/row from the arch's layer
   table) instead of 11 hand-written per-arch strategies; fixed-order 2-rank sum
   (a+b in rank order on both ranks) → **deterministic, replay-safe** (Issue 940
   determinism law).
10. **Security by construction:** the activation/control frames are not an RPC
    surface (no ALLOC/SET at attacker-chosen addresses); peers authenticated by a
    pinned peer key over the link; bind to the TB bridge / tailnet interface only.

## 4. What we do NOT copy

- llama.cpp's client-as-hub RPC model and weight upload.
- exo's Python process-per-runner + per-step `all_gather` of task IDs/cancellations
  (a Rust rank-0 scheduler pushes the schedule once).
- MLX's CPU-stream collectives and the "fast synch" knob documented as deadlock-prone.
- JACCL's full-mesh requirement (2 hosts = 1 link — moot today; matters at 3+).

## 5. Open measurement questions (owned by Plan 623 Phase 0/3)

- Real RTT of a 16 KB message over the TB5 bridge TCP vs RDMA on seal1↔seal2.
- Does macOS `librdma` accept an MR over an `MTLBuffer` `.contents()` pointer
  (shared storage)? If not, the floor is one memcpy per side.
- Single-box decode tok/s for the target models on riir-infer's Metal path vs MLX vs
  llama.cpp Metal — the multi-host claim is meaningless if the single-host arm loses.
