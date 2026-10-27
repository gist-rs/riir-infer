# Bench 900 — Plan 623 T0.4: llama.cpp-RPC 2-box arm (prism-b10787, amo⇄bmo TB5)

**Status:** LANDED (llama.cpp-RPC arm of T0.4; MLX/exo arms open — see §Remaining)
**Date:** 2026-10-26
**Plan:** [623](../.plans/623_multi_host_pipeline_inference_lane.md) T0.4 · **Issue:** [1109](../HISTORY.md) T0.4
**Fork:** `prismml-llama.cpp` @ `prism-b10787-73bfe6c` (`../riir-ai/.raw/prismml-llama.cpp`, shallow clone) — the PQ2_0-capable league fork; the pinned research clone `19ab5a5` cannot parse type-142.
**Build:** `cmake -B build-rpc -DGGML_RPC=ON -DGGML_BLAS=OFF` (BLAS OFF — see Trap 1), Release, arm64, ggml 0.21.0.

## Box state (both ends)

| | amo | bmo |
|---|---|---|
| SoC | M5 Ultra, 96 GiB, 30 cores | M5 Ultra, 96 GiB, 30 cores (Mac17,15) |
| OS | macOS 27.0.1 26A434 | macOS 27.0.1 26A434 (identical build) |
| link | TB5 bridge `192.168.10.1/24` ⇄ `192.168.10.2/24` ([Bench 896](896_plan623_t02_link_probe.md): 8.5–8.6 GB/s, RTT ~94 µs @4 B) | same |
| load | `load avg 4.19–7.12` (sibling agent cargo builds concurrent — GPU idle) | `load avg 0.99–1.15` (near-idle) |
| GPU consumers | Zed only (exempt per house rule) | none |

Model: `Ternary-Bonsai-2-27B-PQ2_0.gguf` (6.70 GiB, qwen35 27B, PQ2_0 2.13 bpw grp128) — amo `riir-train/data/`, bmo `~/models/` (sha256-verified identical, T3).

## Deployment shape (T0.4's real cost — three traps paid for)

bmo has **no Xcode CLT** (`clang`/`install_name_tool` are the stub dialog), **no cmake, no brew** → no native build. The amo build ships as a binary bundle (`build-rpc/bin/`: `ggml-rpc-server`, `llama-cli`, `llama-bench` + all `libggml*` dylibs) with rpath surgery done on amo:

- `install_name_tool -add_rpath @executable_path <binary>` on every executable **and** every dylib that links a non-system absolute path; `codesign -s - -f` after every tool invocation (ad-hoc re-sign).
- amo's homebrew `openssl@4` dylibs ship alongside; `libllama-common.0.2.0.dylib` → `@rpath/libssl.4.dylib`+`@rpath/libcrypto.4.dylib`; `libssl.4.dylib` itself → `@rpath/libcrypto.4.dylib` (it referenced the Cellar path).
- **Law: never patch the live `/opt/homebrew` copy — patch a bin-dir copy and ship that.**

### Traps paid for

1. **`-DGGML_BLAS=OFF` is MANDATORY for the RPC server.** The default build enables Accelerate BLAS; the RPC server's internal scheduler routed `RMS_NORM` to the BLAS backend → `ggml_backend_blas_graph_compute: unsupported op RMS_NORM` → `ggml_abort` server-side, client sees `Remote RPC server crashed or malformed response` mid-decode. Metal+CPU only, fixed.
2. **dyld chain on the target box**: three successive `Library not loaded` aborts (executable rpath → `libllama-common` openssl → `libssl`'s own Cellar libcrypto). All fixed amo-side; bmo can't even run `install_name_tool`.
3. **LLAMA_NO_CLT law:** any box without CLT cannot build NOR patch Mach-O binaries — every load-command fix must happen on the build box, then ship.

## Results (llama-bench, pp512/tg128, -t 8, -ngl 99, qwen35 27B PQ2_0)

| row | backend | pp512 t/s | tg128 t/s | Δ tg vs best solo |
|---|---|---:|---:|---|
| amo solo (load 4–7) | MTL | 1012.87 ± 6.51 | 59.13 ± 0.19 | −2.8% |
| bmo solo (idle) | MTL | 1087.53 ± 1.09 | **60.85 ± 0.48** | baseline |
| **2-box layer-split** (default even split, client amo) | MTL,RPC | 985.07 ± 4.30 | 48.58 ± 0.45 | **−20.1%** |
| 2-box `-ts 1/0` | MTL,RPC (ts 1.00) | 1025.70 ± 1.82 | 57.96 ± 0.17 | INVALID — see below |
| 2-box `-ts 0/1` | MTL,RPC (ts 0.00/1.00) | 1007.11 ± 7.36 | 56.36 ± 1.76 | INVALID — see below |
| 2-box `-sm tensor` | — | — | — | **SERVER CRASH — unsupported** |

## Findings

1. **2-box PP over TCP tg128 = 48.6 t/s = −20% vs the bmo solo** (and −18% vs loaded amo solo). The T0.2 estimate (PP hop ≈ 1.3–1.8% of the decode step) priced ONLY the activation hop; the measured cost includes the per-step RPC round trips (set_tensor inputs + graph_compute + get_tensor outputs ≈ 3 RTTs) and pipeline sequencing. Prefill pays almost nothing (−2.7% vs amo solo; pp is compute-dense and the hop amortizes).
   ⛔ **Verdict for Proposal 042: 2-box llama.cpp PP over TB5/TCP LOSES single-stream decode on identical boxes.** The multi-host lane's decode win must come from latency, not throughput: PP is only a win when neither box alone holds the model (memory-bound), or in batched/concurrent serving where the pipeline overlaps. Single-stream 27B-on-96-GiB never needs PP.
2. **`-ts` does not reach the RPC device in prism-b10787** — both ts rows provably stayed on amo Metal (bmo's server log shows only the 2 `dummy_kernel` device-init compiles, zero real kernels; ts column rendered `1.00` / `0.00/1.00` but device 2 never received tensors). All-remote and biased-split cells are UNMEASURABLE via llama-bench flags in this fork. The genuine split row is the default no-ts layer split (bmo compiled the real kernel set there).
3. **`-sm tensor` over RPC crashes the server**: `ggml_metal_buffer_get_id: tensor 'prism.hadamard.signs.17408 (view)' buffer is nil` → `ggml-rpc.cpp:1101` buffer-bound assert. PrismML's hadamard-sign view tensors cannot cross the RPC buffer boundary. **TP-over-RPC does not exist in the incumbent** — the T4.3 "llama.cpp `-sm tensor`" comparison arm is only viable local-box, or after an upstream RPC protocol extension. Scoreboard row: N/A (crash), recorded as the incumbent's TP ceiling.
4. llama-bench table `backend` column prints `MTL,RPC` even when all compute stayed MTL — the column lists REGISTERED backends, not participating ones; participation must be verified server-side (kernel compiles / set_tensor traffic). **A green 2-box row is not proven by the client's table.**
5. bmo solo (60.9) > amo solo (59.1) on identical silicon — amo's concurrent agent load costs ~3% on tg; quiet-box re-cells should expect amo ≈ bmo.

## Costs

- Build (amo, -j24, incremental over the earlier BLAS config): ~4 min. Ship (bin bundle ~8 MB + dylibs): seconds over TB5.
- Each llama-bench cell: ~90 s (model load + pp512×5 + tg128×5 repeats).

## Remaining (T0.4 open rows)

- **MLX arm**: `mlx-lm sharded_generate --pipeline` (PP) + TP ring — needs `uv` + `mlx-lm` on both boxes, PQ2_0 GGUF is not an MLX weight format → needs an MLX-format Bonsai convert (or its llama.cpp GGUF bridge). Owner model-priority call if conversion cost is nontrivial.
- **exo arm**: PP + TP — pip install exo; same weight-format caveat.
- **T0.3 1-box cells** (league-harness e2e shapes, quiet-GPU window) — open, needs the quiet window.
- **RDMA rows**: gated on D2 (owner).
