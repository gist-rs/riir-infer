# Bench 904 — Plan 623 T0.4 MLX arms: 1-box, 2-box PP, mx.distributed ring probe

**Date:** 2026-10-26/27 (amo + bmo, TB5 bridge)
**Session:** zed-amo-t04-mlx
**Plan:** [623](../.plans/623_multi_host_pipeline_inference_lane.md) T0.4 (mlx arms)
**Model:** `prism-ml/Ternary-Bonsai-2-27B-mlx-2bit` @ `fcba37d21`
(prism_hadamard_qwen35, 64 L, hidden 5120, inter 17408, GDN interval 4,
2-bit affine g128 + hadamard block 1024; vision tower present-but-unused by the
text bench). Pack integrity: files.json sha256 23/23 present-files verified
incl. the 8.595 GB safetensors; 24/24 present after `.gitattributes` restore.
The pack was re-downloaded after the disk-clean reclaimed the blobs — via
direct curl (`hf download --force-download` hit a stall-then-bogus-exit-0 +
dangling-symlink path, recorded in §5).

## 0. Verdict line

**2-box PP over the TB bridge LOSES single-stream decode for the second
incumbent too: MLX-socket PP 43.0 t/s vs MLX 1-box 58.1 t/s (−26%)**
(llama.cpp-RPC PP was −20%, Bench 900). MLX 1-box decode is AT PARITY with
llama.cpp solo (58.1 vs 60.9 t/s, 0.95×) while its prefill is 3× slower
(357 vs 1087 pp512). **mx.distributed ring p2p (send/recv) is DEADLOCKED on a
2-node ring in mlx 0.32.0** — the PP exchange rides a plain TCP socket instead;
ring collectives work and cost 2–4× over raw TCP (all_sum 64 MiB = 3.7 GB/s
vs 8.6 raw).

## 1. Setup (both boxes, same absolute paths)

- uv 0.13.5; venv `/Users/katopz/git/riir-infer/.raw/mlx-env` (cpython
  3.12.15 via uv), pack runtime pins: `mlx==0.32.0 mlx-lm==0.31.3
  mlx-vlm==0.6.3 transformers==5.5.0`. amo built it; bmo = rsync of the
  uv-python dir + the venv at the SAME absolute path (pyvenv.cfg is
  path-keyed) — bmo has no CLT, zero compilation, imports verified
  (`mx.default_device() = gpu`).
- Pack at `.raw/bonsai2-mlx-2bit/` (explicit dir, not the HF cache — see §5);
  rsync amo→bmo 8.6 GB ≈ 4 min (~35 MB/s effective over the ssh channel),
  bmo sha256 = `130de5925082c168b7866b2e…` == files.json.

## 2. The loader gap (recorded, pack untouched)

The pack ships `config.json` **schema_version 2** but its bundled
`runtime/artifact.py` reader accepts only schema **1** — the pack's own
quickstart refuses to load it. The v2 fields are additive metadata
(hadamard_config / tensor_namespace / gdn_activation_layout / components /
vision keys); the reader consumes `text_config` + `modules`, and all 402
module records are shape-identical to v1. Two more deltas vs the bundled
reader: tensors carry a `language_model.` prefix (VL-style serialization;
text-namespace keys survive a prefix strip collision-free) and the file
carries `vision_tower.*` weights (excluded from the strict text load).
**Fix lives OUTSIDE the pack** (pack bytes stay sha256-gated):
`scripts/plan623_pack_artifact.py` = the bundled reader + schema gate
{1,2} + prefix strip + vision-key filter, each commented at the site.
Both boxes run the same copy (rsync twin).

## 3. 1-box rows (bmo idle; amo deferred — elephant capture holds its MPS)

| arm | pp512 t/s | tg128 t/s | tg ms median | source |
|---|---|---|---|---|
| MLX 1-box (bmo) | **357.3** | **58.1** | 17.217 | `plan623_mlx_1box_bench.py`, greedy |
| MLX 1-box TTFT (bmo) | pp4096 **1005.3** | — | — | same script, I623_PP=4096; **pp32768 REFUSED by Metal** (single-buffer max 62.6 GB < the 103 GB intermediate — stock-MLX 32k prefill N/A without chunked prefill; its own finding) |
| llama.cpp solo (bmo), Bench 900 | 1087.5 | 60.9 | — | prism-b10787, PQ2_0 |
| ratio MLX/llama.cpp | 0.33× (512) / 0.92× (4k) | **0.95×** | | |

Quant postures DIFFER (2-bit affine+hadamard pack vs PQ2_0 GGUF) — the rows
are plan-internal comparability, not an equated quant study. Decode parity at
2-bit is the expected memory-bound result; MLX prefill is compute-bound and
3× behind llama.cpp's MMA prefill.

## 4. 2-box PP row + the mx.distributed findings

**Transport verdict first:** `mx.distributed` ring (backend="ring", 2 ranks,
TB bridge) — `all_sum`/`all_gather` work; **`send`/`recv` p2p deadlocks on a
2-node ring in mlx 0.32.0**: both ranks post (send posted / recv posted), zero
bytes cross the wire (netstat queues empty on all four hops), both wait
forever in `array::wait()`. Minimal repro: `.raw/min_p2p.py`. The ring's own
comment at ring.cpp:520 says recv checks sockets in the OPPOSITE order of
send precisely for the 2-node same-neighbor case — the pairing is consistent
on paper and dead in practice at 0.32.0. **Verdict: mlx ring p2p N/A on this
topology; mlx-lm pipeline-over-ring inherits it.**

**PP harness** (`scripts/plan623_mlx_pp2_bench.py`): layer-split 32/32 over
the pack runtime (rank0 = embed + layers[0:32], rank1 = layers[32:64] +
norm + head), exchange over ONE direct TCP socket (amo→bmo connect; amo's
application firewall blocks inbound to unsigned python — see §5 — so the
connect direction is chosen to be amo→bmo; one socket carries hidden
amo→bmo and tokens bmo→amo, bf16 raw bits, TCP_NODELAY). Caches stay
rank-local (GDN ArraysCache / FA KVCache) — recurrent state never crosses
the wire. Greedy sampling on rank1; gen_head repeats (= deterministic on a
ramp prompt, no NaN blowup).

| arm | tg128 t/s | tg ms median | Δ vs its own 1-box |
|---|---|---|---|
| MLX 2-box PP (bmo half, clean box) | **43.0** | 23.236 | **−26%** vs 58.1 |
| MLX 2-box PP (amo half, loaded box) | 23.4 ms median | | agrees with role1 within noise |
| llama.cpp-RPC 2-box, Bench 900 | 48.6 | | −20% vs 60.9 |

Prefill: role1 wall 0.631 s / role0 wall 0.294 s → end-to-end ≈ **325 t/s**
(−9% vs bmo solo 357). The +6 ms/token PP cost is serialization of the two
GPU half-forwards (the socket hop is ~0.1-0.3 ms of it) — the same
"pipeline sequencing, not wire" conclusion as Bench 900's RPC row.

**Ring collective probe** (`plan623_mlx_distributed_probe.py`, full JSON
`904_plan623_t04_mlx_ring_probe.json`): all_sum 4 B = 325 µs median /
hidden-10 KiB = 322 µs / 1 MiB = 1.05 ms (2.0 GB/s) / 64 MiB = 36.2 ms
(**3.7 GB/s vs 8.6 raw TCP**) / all_gather 1 MiB = 832 µs. The ring's
collective floor (~320 µs) alone exceeds a socket PP hop by ~30×; ring-based
TP stays unattractive on this link class at any split (consistent with Bench
896's TCP TP arithmetic refutation).

**Verdict:** the Proposal-042 conclusion now holds for BOTH incumbents —
2-box PP on identical boxes over TB5/TCP **loses single-stream decode**;
the multi-host win must come from (a) the memory-bound case (model that
cannot fit one box — 27B-on-96-GiB never hits it) or (b) lower per-step
hop counts / RDMA (D2-gated). riir-infer's own lane therefore wins or loses
on step-serialization efficiency (ActivationFrame + 1 hop) and RDMA, not on
incumbent comparisons alone.

## 5. Environment traps (all recorded so they never re-fire)

1. **macOS application firewall on amo** (state=1) blocks INBOUND to
   unsigned binaries — the uv venv python silently times out on inbound
   connects (Bench 896's `link_probe` works because it is individually
   allowlisted). bmo: firewall OFF. No passwordless sudo on either box →
   no allowlist path. The lawful pattern used here: **amo always the TCP
   CLIENT**, bmo the listener; loopback + ssh -R relay where bmo→amo was
   unavoidable (ring init). JACCL/RDMA is the real fix and is D2/owner-gated
   (recovery-mode `rdma_ctl`).
2. **hf download stall + bogus exit-0:** `hf download --force-download`
   stalled twice (499 MB, 2.0 GB), left orphaned `.incomplete` fragments,
   then exited 0 printing the snapshot path while the final blob never
   materialized (dangling `blobs/<etag> -> blobs/05/<sha>` symlink into a
   nonexistent sharded dir; xet cache 17 MB). Remedy: direct `curl -L -C -`
   with a retry loop into an EXPLICIT pack dir (70 MB/s vs hf's stall), then
   the files.json sha256 gate. The HF-cache snapshot dir was left with its
   small files only; the bench uses `.raw/bonsai2-mlx-2bit/`.
3. **Launch-order race on the shared ring hostfile:** rank1 connects-first
   to rank0's entry; if rank0's listener isn't up, sshd's -R relay accepts
   then drops the relayed connect → rank1 hangs on a dead socket with no
   error. Rule: **rank0 first, poll LISTEN (lsof), then rank1.**
4. **macOS rsync (openrsync)** rejects `--info=stats1`; and `--relative`
   with absolute sources recreates absolute trees under the destination
   (junk `git/riir-infer/.raw/Users/...` on bmo — removed).

## 6. Artifacts

- `904_plan623_t04_mlx_ring_probe.json` — ring probe rows (rank0)
- `904_plan623_t04_mlx_pp2_r1.json` — the clean 2-box PP row (role1/bmo)
- `scripts/plan623_mlx_1box_bench.py` · `plan623_mlx_pp2_bench.py` ·
  `plan623_mlx_distributed_probe.py` · `plan623_pack_artifact.py` (rsync
  twins on bmo at the same paths)
- amo-side rows (1-box + a clean-window 2-box re-read): **deferred to the
  T0.3 quiet window** (elephant capture holds amo MPS until ~10-27); the
  role0 row's median agreeing with role1 within noise is the interim
  cross-check.
