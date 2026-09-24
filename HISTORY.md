# HISTORY.md — riir-infer

Durable records for resolved questions and closed lanes (the noise-reduction
convention: the record lands here, hash-pinned; open work lives in `.issues/`
and `.plans/`). Created 2026-09-23 at the first record.

## 2026-09-24 — Issue 002 CLOSED: the CUDA backend for the laya lane (the 4090 bench row, 17–60× the CPU posture)

Landed `9b52cb1`/`99f156e`→rebased `e99d767`: the lane's third compute
backend, `laya-riir-cuda` — cudarc 0.19 (`driver`+`nvrtc`, `cuda-13030`+
`fallback-dynamic-loading`, target-scoped `not(macos)`, one workspace version
with `riir-infer-gpu`'s raw-CUDA lane), CUDA C compiled to PTX at construction
(NVRTC, arch `sm_89`). The Metal backend's architecture ported verbatim:
permanent `(ptr,len)` weight cache, epoch-keyed chain slots with `begin_pass`
invalidation, `download_into` prefix-read barrier (a `CudaView` slice —
`memcpy_dtoh` asserts on the longer parent slot), lazy async submission on one
stream. ONE strided batched sgemm (64×64×32 tile, 512 threads, fp32 FMA;
the weight binds row-major `[n,k]` directly — the B-tile loader maps lanes
along whichever stride is 1, so NO device transpose cache) + the 14 tail
kernels at the CPU lane's exact semantics. Attention v1 = the trait DEFAULT
op sequence, fully device-side (attention <6% of forward FLOPs at the pinned
geometries — a fused flash kernel is a follow-up rung, tracked below).
`CudaSlice::clone()` is a device-to-device COPY in cudarc (not a refcount
bump like Metal's `Buffer`) — the caches hold `Arc<CudaSlice<f32>>`.

Two kernel defects found by the op gate before any G5 run: the staging loops
loaded 1024 of 2048 tile elements at 512 threads (the Metal kernel's 1024-
thread q<2 shape copied straight over), and the `b_cs==1` staging branch
dropped the `n0` column offset (columns ≥ 64 served tile 0's B — the tiny
shapes passed, n=128 failed from column 64 on; a ones/identity ladder
localized it in two runs).

Gates (all green on the 4090 box, CUDA 13.3 / driver 610.62):
- `cuda_ops_smoke`: every backend op vs the CPU free fns — bit-exact data
  movement, 1e-7…1e-4 reductions;
- consumer-side G5 at `LAYA_DEVICE=cuda`: english 26/26 top-1 drift 1.863e-6,
  typed 26/26 2.471e-6, multilingual 36/36 2.894e-6 — the Metal drift class,
  ~500× under the 1e-3 gate, GREEN FIRST RUN;
- `packed_forward_equiv` at the cuda posture (the same-day concurrent
  packed-forward landing composed cleanly — `copy_at` added at the rebase).

Measured (fixture rows, same-session A/B on this box): english 210.4→17.5 ms
(12.0×), multilingual 95.0→9.0 ms (10.6×), typed 208.4→17.4 ms (12.0×) —
every checkpoint BELOW the M3 Metal row (28.3/12.2/28.3 ms) at v1. The full
15-suite bench refresh + the site publish record lives in the consumer repo
(riir-reflex `.issues/026`, `.benchmarks/026_4090windows_cuda/`). Follow-up
rungs (open, unordered): flash-attention port (the MSL two-pass online-softmax
form), tile ladders for the m<64/n≤1024 shapes, CUDA graphs for the
per-op dispatch overhead — each G5-gated at the cuda posture before any
number replaces a published one.

## 2026-09-23 — crates.io publication: keep `publish = false` until the vendor patches upstream (owner-gates menu v2 row 2)

Owner verdict: the crate and `crates/riir-infer-gpu` stay **closed to
crates.io** — this is a HARD blocker, not a preference. Both vendor forks under
`vendor/` are load-bearing (`[patch.crates-io]` in the root manifest):

- `cubecl-runtime` — carries the #1359 drop-queue fix.
- `wgpu-hal` — carries the VRAM accessors the GPU code probes.

A `[patch.crates-io]` section does not survive publication: a crates.io
consumer would build against the UNPATCHED upstream crates — the drop-queue bug
and the missing accessors included. Publication becomes available the day the
vendor deltas land upstream (the forks shrink to zero and the patches drop out
of the manifest).

Boundary note: this repo stays upstream of the engine regardless — the public
funnel for the stack's primitives remains `katgpt-rs` (its katgpt-core family
publishes), not this repo.

Session: owner-gates-m2, 1790121600
