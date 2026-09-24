# HISTORY.md — riir-infer

Durable records for resolved questions and closed lanes (the noise-reduction
convention: the record lands here, hash-pinned; open work lives in `.issues/`
and `.plans/`). Created 2026-09-23 at the first record.

## 2026-09-25 — Issue 004 CLOSED: the sgemm tile ladder — block-fit floors, the narrow single-question win

The `.issues/002` v1 backend ran ONE sgemm instance (64×64×32, 512
threads) for every GEMM shape. The lane now ships THREE instances —
`sgemm_narrow` (32×64×64, 512 thr, staging A[32][65]+B[64][65] = 24 960 B),
`sgemm_wide` (the v1 kernel, renamed) and `sgemm_xwide` (64×128×32,
1 024 thr, staging A[64][33]+B[32][129] = 24 960 B) — picked per call by
**BLOCK-FIT on the SM count**, a floor MEASURED on this box, not ported:
the M3 Metal lane's `m < 256` threshold does NOT transfer to the 128-SM
4090 (it is subsumed by the block-fit arithmetic on the real population).

**The cliff that sets the floor** (the probe's CUDA arm,
`sgemm_shape_timing`): at m=106 the narrow instance wins −15.7 % at
n=2048 — exactly 128 blocks, one per SM — and LOSES +46 % at n=2560 —
160 blocks: static block scheduling strands 32 SMs at 2× work while 96
idle after one. Every instance whose grid exceeds the SM count pays that
straggler tail, so the pick is: narrow iff its grid (× batch) fits one
wave; xwide iff m≥256 ∧ n≥2048 ∧ its grid fits (the multi-wave zone —
gate/up at n=5248, 205 blocks — reverts to the proven wide instance:
readings sat inside the instrument's measured ±8-10 % two-context
artifact band). Per-output accumulation stays k-ascending in ONE thread
on every instance, so the ladder is result-identical by construction —
and measured: 13/16 bench suite-lane rows bit-identical vs the 028 run
(every single-question suite; the three typed_decisions rows wobble 1
case in 2000 within that lane's pre-existing `determinism_ok: false`
variance, on record since the 026 v1 run).

Measured wins: per-shape narrow −13.7..−19.6 % (n=1024/1536/2048, the
m=4/m=1 head tails −12..−17.5 %); xwide QKV (120 blocks) −6.8..−8.5 %
across three runs; forward-level A/B on the fixture rows (ABAB,
single-backend-per-process): english −10.4 % · multilingual −14.1 % ·
typed −8.4 %. Published row refresh (reflex `.benchmarks/029`): the
single-question suites −6..−14 % p50, packed suites flat by the
conservative floor. Kill-switch `LAYA_CUDA_LADDER=0` (wide everywhere —
the A/B posture, never a silent default).

A launch defect fixed in passing: the v1 form passed the staging
footprint as DYNAMIC shared memory on top of the kernels' STATIC
`__shared__` arrays — harmless at wide's 2×16 768 B, but the new
instances' 2×24 960 B crosses the 48 KB static default and the launch
dies `CUDA_ERROR_INVALID_VALUE` (caught by the first smoke arm — the
static-smem constant never reached the launch). All instances now
launch with dynamic smem 0; the footprint constants live on as
compile-time bounds.

Gates at the final floors: `cuda_ops_smoke` (with the new boundary arms
— m=33/255/256, n=65/2047/2048/2080, k=33/63/65, m=321 — every tile
edge on every instance) · consumer G5 at the cuda posture ·
`laya_batch_parity` · `packed_forward_equiv` · clippy −D warnings both
repos. Follow-up rungs stay open: CUDA graphs for per-op dispatch
overhead; the packed multi-wave zone has NO measured win yet — split-K
or an occupancy-tuned instance is the open question, not another tile
size.

## 2026-09-25 — Issue 003 CLOSED: CUDA flash attention — the packed-path zeros defect fixed + the fused rung

The `.issues/002` v1 posture ran `attention_forward` through the TRAIT
DEFAULT op sequence on CUDA. That default SLICES host memory
(`&qkv[qkv_off..]`) — correct at offset zero (the slice IS the parent
the device op wrote → same `(ptr,len)` chain key → hit → device-current)
and SILENTLY WRONG at non-zero offsets: the packed multi-question
forward's slice is a NEW key → chain MISS → uploads the host bytes,
which under the write-first discipline are STALE (the parent was written
device-side only; the host vec holds its `resize(.., 0.0)` zeros).
**Every multi-question case's attention ran on zeros at v1.**

The evidence was the published bench, not the code: reflex
`.benchmarks/026_4090windows_cuda` vs `018_4090windows_run` (CPU, same
box) — typed_decisions english 0.3575→0.2690, multilingual 0.3490→0.2690,
typed **0.7445→0.2690** (−47.5 pt), code_fixtures 0.5417→0.2917 (2
q/case), while every 1-question-per-case suite was byte-identical
(ag_news 0.9500, banking77 0.4980). The consumer-side G5 passed green
because its fixture rows are single-question (offset zero — the correct
path); `laya_batch_parity` (the multi-question gate) had not been run at
the cuda posture. The 026 close-out's "accuracy byte-identical on every
lane" claim was wrong for the multi-question suites — corrected in the
consumer repo's bench doc the same day.

The fix IS the rung: the Metal lane's one-pass online-softmax flash
kernel (MSL_FLASH, the reflex Issue 020 T10 rung-3 form) ported to CUDA C
at plain fp32 FMA — ONE dispatch per layer over the packed qkv (split,
rope, q-scale, scores, sliding window, softmax, value mix, head merge
in-kernel; the seq² scores parent never exists; **offsets bind at
dispatch**, so the packed forward is the unbatched kernel's exact math —
Metal's design, which is why the Metal lane was immune). 256 threads,
one block per (32-row query block, head); shared staging
tq[32][65]/tk[64][33]/tv[32][65]/ts[32][33]/tacc[32][65]+mrow/lrow/arow
= 38 016 B dynamic smem; the online rescale α = expf(m_old − m_new) is
exactly 1.0f when the max does not move. Kill-switch `LAYA_CUDA_FLASH=0`
→ the reference sequence, which now PANICS on non-zero offsets (the
Metal guard — the silent zeros are now a loud breach, and
`supports_packed_attention` answers false there so the agent takes the
per-question loop). `needs_window_mask` mirrors the armed path (the
encoder stops building `[seq,seq]` masks on the fused lane).

Gates green on this box, CUDA 13.3 / driver 610.62, GPU clear of compute
consumers (GUI apps only — the exempt class):
- `cuda_ops_smoke` + 3 new arms: fused full (seq 1/9/37/64/129 — every
tile edge) drift 1.2–1.8e-7; sliding (w8@64/w4@37/w16@130) 1.2–1.8e-7;
  the PACKED-offsets arm (two sequences at non-zero qkv/rope/out offsets,
  the encoder's whole-parent call shape) 1.2e-7 — GREEN FIRST RUN.
- `packed_forward_equiv` gained a CUDA arm (non-macOS,
  `laya-riir-cuda`-gated) — the gate class that catches the zeros defect
  at the substrate level; verified it FAILS LOUD under
  `LAYA_CUDA_FLASH=0` (the fallback's offset guard panics).
- Consumer-side at `LAYA_DEVICE=cuda`: `laya_batch_parity` — 26+26+36
  batched multi-question forwards, top-1 1.000000, drift ≤ 5.1e-5
  (~20× under the 1e-3 gate) — THE gate that would have caught v1; G5
  parity english 3.3e-6 · typed 1.3e-6 · multilingual 4.7e-6 (the
  online-softmax restructure's expected class, ~200× under the gate).
- Latency (fixture rows, short seqs — flash vs `LAYA_CUDA_FLASH=0`):
  english 17.5→16.2 ms (−7.4%), multilingual 9.0→8.5 (−5.6%), typed
  17.6→16.6 (−5.7%). The long-seq suites gain far more (the seq² scores
  traffic ~6×`heads·seq²·4B` per layer never exists, and the windowed
  key walk cuts attention FLOPs ~2.4× at window 64) — measured in the
  consumer repo's refreshed bench.

The bench refresh + the published-numbers correction live in the consumer
repo (reflex `.issues/028` — renumbered from 027 after a same-window
dual-allocation; the record is reflex HISTORY §2026-09-25 bench 028).
Remaining follow-up rungs (open, each G5-gated at the cuda posture): the
sgemm tile ladder (closed same day as `.issues/004`), CUDA graphs for
per-op dispatch overhead.

Session: 4090-cuda-flash, 2026-09-25

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
