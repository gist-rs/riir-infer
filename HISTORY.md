# HISTORY.md — riir-infer

Durable records for resolved questions and closed lanes (the noise-reduction
convention: the record lands here, hash-pinned; open work lives in `.issues/`
and `.plans/`). Created 2026-09-23 at the first record.

## 2026-09-25 — Issue 009 CLOSED: software-pipelined narrow staging — NEGATIVE at the gate, and the probe's tiny-op floor exposed

The `.issues/008` follow-up ("double-buffered staging / cp.async — latency
hidden, not bandwidth saved") was built as the REGISTER form and answered
**NO on the pre-registered gate** — with two findings that outlive the rung.

**The rung** (built, measured, reverted): `sgemm_narrow_pipe` — the same
32×64×BK64 tile/threads/fragment/grid as `sgemm_narrow`, k-loop software-
pipelined: tile t+64's guarded loads issue into 12 registers at loop top, a
compiler fence (`asm volatile("" ::: "memory")` — measured load-bearing,
see below) pins their issue order, tile t's compute (~1300 cy) hides their
latency, stores land after the post-compute barrier. Double-buffering both
operands needs 51 456 B > the 48 KB static cap, so B ping-pongs (`tb[2]`)
while A register-defers into the single `ta` — 43 136 B total. Bit-identical
on every probe shape both postures, both runs; all gates green (smoke with
8 pipe-posture arms incl. the ragged/epilogue paths, packed_equiv 4/4, lib
41/41).

**Finding 1 — the fence iteration**: the first build measured FLAT (median
−1.8 % on the narrow rows) because nvcc SINKS the register loads to just
before their consumers (a register-pressure choice), collapsing the window
— the fence pins program order and is mandatory for this form. With it the
probe still read −1.5 % (−1.2..−2.3 on the m=106/45 rows).

**Finding 2 — the probe's tiny-op floor (the real yield)**: across the
narrow-zone rows, FLOPs vary 400× (1×1028×256 = 0.5 MFLOP at 43.5 µs →
106×1024×1024 = 222 MFLOP at 45.5 µs) while time varies 1.06× — every
row is `≈ 40-41 µs launch/WDDM floor + flops/50 TFLOP/s`. The
`sgemm_shape_timing` tiny-op rows are ~90 % submission floor and CANNOT
resolve kernel-level rungs in the narrow zone; `.issues/006`'s "head tails
−13..−17 %" rows were the same floor class (the float4 win was real but the
row magnitudes were floor-shifted), and 008's reg4 reading only registered
because a 4× warp cut punched the kernel ~75 µs PAST the floor. Future
narrow-zone rungs must gate at the FORWARD level (launches amortize in the
deep queue) or add a same-kernel floor-subtraction arm to the probe.

**The honest instrument**: forward-level paired env-flip (`laya_fixture_timing`,
english — the narrow-heavy checkpoint, 3 alternating pairs, quiet box):
off 12.6/12.6/12.6 ms → on 12.4/12.4/12.5 ms row p50 — **−0.8..−1.6 %,
reproducible 3/3** (p90 and ms/question improve too; the alternation kills
drift). The pipelined kernel IS genuinely faster — by ~1.6 %, not the
predicted ~25 %: TLP across 16 warps already hides most of the staging
convoy at the phase boundary, and the residual (2 barriers + the store
phase ≈ 1.5-2 % of the loop) is what the pipeline actually recovered.

**Verdict** (the pre-registered gate binds): both instruments read under
the ≥3 % bar → NO-GO. `cuda.rs` + the smoke arms + the reflex probe env
dance reverted byte-identical to the pre-rung state; the kernel is not kept
behind a switch because no posture clears the bar. The cp.async form
(1 sync/iter, no register round-trip) remains the recorded next lever,
priced by this record at ≤ ~4 % forward (the barrier + store residual) —
not worth a rung unless the forward-level instrument shows the narrow share
growing.

## 2026-09-25 — Issue 008 CLOSED: the narrow reg4 rung — NEGATIVE, the narrow zone is TLP-bound, not bandwidth-bound

The `.issues/007` open question ("the NARROW instance (1×4 fragment, ~20 %
ceiling, m<256 single-wave zone) is the next relative laggard — a narrow
reg4 arm is the recorded follow-up rung") is answered **NO on measurement**,
and the measurement's real yield is the mechanism: **the narrow zone's
binding constraint is thread-level parallelism, not shared-memory
bandwidth** — register blocking, which trades warps for bytes-per-FMA, is
the wrong currency exactly where the ladder runs 1 block per SM.

**The rung** (built, measured, reverted): `sgemm_narrow_reg4` — the 007
family 4×4 fragment (16 accumulators, 4 LDS.32 + 1 LDS.128 per 16 FMAs =
2 B/FMA vs the 2-acc narrow's 5, ceiling 20 %→ 50 % on the 007 roofline
arithmetic). The fragment-area law forces a 32×16 warp tile; on the narrow
32×64×BK64 tile that is a 1×4 warp grid = **128 threads** (vs 512). Same
tiles, same staging footprints, same grid arithmetic — every block-fit
property carries over; per-output accumulation k-ascending in one thread
(bit-identical held on EVERY probe shape, both postures, both runs).

**Measured** (`sgemm_shape_timing`, the `LAYA_CUDA_REG4` A/B axis — the
narrow rows went live for the first time; two full runs, quiet box, GUI-class
load only): every true narrow-served row regressed **+41..+177 %** —
106×1024×1024 50→125 µs (+150/+139 %), 106×2624×1024 119→332 µs
(+178/+171 %), the n=1536/2048 crossover rows +139/+152 %, short-seq
45×1024×1024 +151/+157 %, the m=4/1 head tails +118/+41 %, act a0 +113 %.
Ten times outside the ±8-15 % instrument band — no `--control` arm needed to
adjudicate. The wide/xwide-class rows (n ≥ 2560 at m=106, banking77, the
packed zone) re-measured the 007 rung at −2.6..−20 % on both runs — the
consistency check; the one historically noisy row (packed O 424) flipped
+12 %→−12 % between runs exactly as the recorded band predicts.

**The mechanism, read off the kernel's own structure**: the narrow zone's
grids are ≤ 128 blocks BY CONSTRUCTION (the block-fit pick) — 1 block per SM,
so the SM's latency hiding is the block's warps and nothing else. The
k-loop is sync→stage→sync→compute: staging is 48 global loads per thread
per k-tile (A 16 + B 32 at 128 threads), and with 4 warps the DRAM/L2
latency of a stage is exposed almost fully — 16 warps (512 threads) at
least keep 4× the loads in flight and 4× the barrier-arrival slack. The
2 B/FMA bandwidth win is real but irrelevant: the measured narrow ceiling
was never the 20 % roofline — at m=106 the 2-acc narrow computes at ~5 %
of fp32 peak (50 µs for 222 MFLOP), i.e. the zone sits ~4× under its own
bandwidth ceiling already. Cutting warps scaled the wall time almost
exactly as TLP arithmetic predicts (512→128 threads ≈ 4× offered parallelism
→ 2.3-2.8× wall on the load-dominated loop).

**Verdict** (the pre-registered gate's NO-GO path): no partial win, no
carve-out — every true narrow row lost. `cuda.rs` reverted byte-identical
to the pre-rung state (the BK48 precedent — only the docs carry the
negative); the kernel is not kept behind a switch because there is no
posture in which it wins.

**The follow-up rung this record opens** (replacing the 007 pointer):
the narrow zone needs LATENCY hidden, not bandwidth saved —
**double-buffered staging / `cp.async`** (prefetch k-tile t+1's A/B while
computing t, sm_89's async copy path bypassing registers) is the recorded
next lever. Note the symmetry with the 007 record: double buffering was
correctly dismissed for the BANDWIDTH-bound wide zone and is exactly the
right repair for the LATENCY-bound narrow zone — the two zones' rooflines
diverge, so their rungs must too.

## 2026-09-25 — Issue 007 CLOSED: the register-blocking sgemm rung — 4×4 fragments, −9..−21 % kernel on every wide/xwide shape

The `.issues/006` follow-up rung ("register blocking / double-buffered
staging — the instances sit at 25-30 % of fp32 peak") landed as REGISTER
BLOCKING, and the choice between the two candidates was settled by the
arithmetic, not taste: after the float4 rung the inner loop is
**2×LDS.32 (A) + 1×LDS.128 (B) ≈ 6 SM-cycles of shared-memory bandwidth per
8 FMAs ≈ 2 SM-cycles of FFMA capacity** — a ~33 % shared-bandwidth roofline
at full occupancy, matching the measured 25-30 %. The lane is BANDWIDTH-
bound, not latency-bound, so double buffering (a latency repair) buys
nothing; register blocking raises the ratio.

**The rung**: `sgemm_wide_reg4` + `sgemm_xwide_reg4` — the SAME tiles and
grids as wide/xwide at HALF the threads (256 / 512), warp tile 32×16 (warp
grid 2×4 / 2×8), thread fragment **4 rows × 4 cols = 16 accumulators**:
4 LDS.32 + 1 LDS.128 per 16 FMAs = **2 B of smem reads per FMA (the 2-acc
instances' 3)** → ceiling 50 %. Same grids mean every measured ladder
property (block-fit cliffs, straggler tails) carries over untouched; xwide
additionally rises to 3 blocks/SM = 48 warps (the 1 024-thread instance
capped at 32). Per-output accumulation stays k-ascending in ONE thread —
the result-identity law is untouched. Kill-switch `LAYA_CUDA_REG4=0` holds
the 2-acc posture (the `LAYA_CUDA_LADDER` contract).

**Measured** (`sgemm_shape_timing`, the A/B axis moved to
`LAYA_CUDA_REG4`; two full runs + `--control`): every wide/xwide-served
row negative in run 2 — banking77 zone −9.4..−16.1 %, the packed
multi-wave zone −9.4..−21.4 % (O m=1268 175→151 µs, down 436→345, QKV
477→387, gate/up 735→637), the m=106/45 n≥2560 rows −10..−22 %. Run-1's
lone positive (424 QKV +3.5 %) flipped to −9.4 % in run 2 — inside the
instrument band, which the `--control` arm (SAME-kernel pairs) measured at
−14.9..+10.1 % on this box this hour (the sibling session's builds — wider
than the recorded ±8 %, why every row was read against it). Narrow-served
rows flat on both runs (both postures route narrow — the design's
consistency check). Live-row median ≈ −12..−13 %, gate was ≥3 %.

Forward level (paired same-binary env-flip, 3 alternating pairs):
english 13.1→12.4 ms row p50 (−5.3 %), typed 12.9→12.4 (−3.9 %),
multilingual flat — the dilution is structural: at fixture seqs only the
n≥2560 projections route wide-class (the rest narrow, unchanged; attention
and the row kernels untouched).

**Published row refresh** (reflex `.benchmarks/031`, 15 suites PASSED,
host 4090-windows): every suite ≤ 030's p50 — median −6.2 %, best −9.0 %
(the packed typed_decisions trio), the short fixed-overhead suites flat.
Accuracy 14/17 rows BIT-IDENTICAL; the three typed_decisions rows wobble
3-7 cases in 2000 — that lane's pre-existing `determinism_ok: false`
variance (false in 026/028/029/030 AND 031, on record since v1).

Gates at the landing: `cuda_ops_smoke` 4/4 (the ladder-boundary + ragged
arms now route through the reg4 kernels at the default posture) ·
`packed_forward_equiv` 4/4 · lib 41/41 · consumer G5 at the cuda posture
2/2 (top-1 1.000000 ×3, prob drift ≤ 3.3e-6 — the lane's own class) ·
`laya_batch_parity` 1/1 · clippy clean both repos. Remaining upside on
record: the reg4 instances should now sit near ~40 % of fp32 peak (the
50 %-ceiling minus overheads), and the NARROW instance (1×4 fragment,
~20 % ceiling, m<256 single-wave zone) is the next relative laggard — a
narrow reg4 arm is the recorded follow-up rung, measured before built
(the `.issues/007` §Open question).

## 2026-09-25 — Issue 006 CLOSED: the float4 sgemm rung — every instance's B loads collapsed, −7..−17 % on every suite

The `.issues/004` open question ("the packed multi-wave zone has no measured
win — split-K or an occupancy-tuned instance") is closed with the question
REFRAMED and answered: the zone's problem was never the straggler tail —
it is LOAD-ISSUE THROUGHPUT. The probe's appended packed-zone population
(m=424 ≈ 4×106, m=1268 ≈ 4×317) measured the wide instance at **12-16.5
TFLOP/s = 15-20 % of the 4090's 82.6 fp32 peak** (gate/up at m=1268:
13.65 GFLOP in 829 µs), 2.5-3× under the cuBLAS-class roofline — and the
inner loop's own shape explains it: **6 smem loads per 8 FMAs**, with the
four B-fragment loads CONTIGUOUS in the staging tile.

**The rung**: every instance's B staging row pads to a 16 B multiple
(wide 65→68, narrow 65→68, xwide 129→132 — col0 is a multiple of 4 at
every call site, so `&tb[kk*pad + col0]` is float4-aligned BY CONSTRUCTION)
and the four B loads collapse to ONE `reinterpret_cast<float4>` — 6 loads
per 8 FMAs becomes 3 (narrow's 1×4 fragment: 5 → 2). float4 loads change
no arithmetic order, so the instances stay result-identical by construction
(verified BIT-IDENTICAL on every probe shape, both arms) and every ragged
edge is staging-side-safe (out-of-range elements stage as 0.0f; stores stay
guarded per element).

**Measured** (`sgemm_shape_timing`, in-process base/ladder A/B with the
experiment arm on the would-be-wide slots, two full runs): −10.3..−25.2 %
on the multi-wave zone rows, −5..−14 % on the single-question wide rows,
the m=4/m=1 head tails −13..−17 % (those are narrow-served — narrow's own
float4 win). Landed (both arms new kernels), the absolutes vs the morning
baseline: gate/up m=1268 839→727 µs (−13 %), QKV m=1268 536→445 (−17 %),
down m=1268 470→406 (−14 %), m=317 QKV 158→125 (−21 %). Forward level
(fixture probe, cross-binary same-box same-session): english 15.1→12.9 ms
row p50 (−14.6 %), multilingual 7.2→6.3 (−12.5 %), typed 15.1→12.8
(−15.2 %).

**The published row refresh** (reflex `.benchmarks/030`, 15 suites
PASSED, host 4090-windows): every suite −6.8..−16.7 % p50 — the packed
suites TOO this time (typed_decisions 109→100/59→55/109→99, banking77
28→25, code_fixtures 31→28) — the multi-wave zone's first measured win,
which was the `.issues/004` open question. Accuracy: 13/16 rows
BIT-IDENTICAL; the three typed_decisions rows wobble 1-4 cases in 2000
within that lane's pre-existing `determinism_ok: false` variance (false
in 026, 028, 029 AND 030 — on record since v1, independent of every
kernel rung).

Gates at the landing: `cuda_ops_smoke` 4/4 (tile-edge + ragged arms on the
new kernels) · `packed_forward_equiv` 3/3 · lib 41/41 · consumer G5 at the
cuda posture 2/2 (top-1 1.000000 ×3, drift ≤ 5.1e-5 class) ·
`laya_batch_parity` 1/1 · clippy clean both repos. No kill-switch added —
the float4 form IS the wide/narrow/xwide kernels now (same tile geometry,
same pick, same result chain; `LAYA_CUDA_LADDER=0` still holds the ladder
A/B posture). Remaining upside on record: the instances still sit at
~20-24 TFLOP/s ≈ 25-30 % of peak — the next rung (register blocking /
double-buffered staging) is a bigger redesign, not landed today.

## 2026-09-25 — Issue 005 CLOSED: CUDA graphs — NEGATIVE, the lane is GPU-bound (submit fully hidden)

The standing follow-up rung (recorded at the close of 002/003/004: "CUDA
graphs for per-op dispatch overhead") is closed on measurement, with the
pre-registered go/no-go gate of `.issues/005` §2 answered NO by a DIRECT
device-timeline reading rather than by the submit/wall ratio alone.

**The instrument (ships, `LAYA_CUDA_STATS=1`)**: `begin_pass` stamps t0 and
records a timing event at the head of the now-idle stream; the pass's FIRST
`download_into` (the pipeline-draining sync) records the end event, syncs,
and prints `submit` (pre-sync − t0: the whole CPU path — host embedding
gather, H2D uploads, slot `cuMemAlloc`s, every launch call), `wall`
(post-sync − t0), `gpu` (the device-timeline elapsed between the two events
— the GPU critical path of the prefix, inter-kernel gaps and alloc-induced
stalls INCLUDED), plus the submit-path decomposition (upload count/time/
bytes, alloc count/time, accumulated at the call sites). Zero cost when the
env is unset — the events are only created under the flag, and the stats-off
posture re-measured byte-stable (english 15.1 ms / multilingual 7.2 ms row
p50, identical to the pre-instrumentation readings).

**The measurement** (fixture probe, all three checkpoints, reps=3, GPU
exclusive — GUI apps only, the owner-call exemption):

| row class | submit | wall | gpu | uploads | allocs |
|---|---|---|---|---|---|
| english p50 | 4.6-5.2 ms | 12-15 ms | **= wall ±0.3 %** | 6x ~0.07 ms | 18-25x ~0.2 ms |
| multilingual p50 | 1.7-2.8 ms | 6.1-7.2 ms | **= wall ±0.1 %** | 6x ~0.04 ms | 19-24x ~0.13 ms |
| typed p50 | 4.5-6.5 ms | 12.8-15.7 ms | **= wall ±0.3 %** | 6x ~0.08 ms | 19-24x ~0.2 ms |
| long rows (63-133 ms) | 16-32 ms | 63-133 ms | **= wall** | ≤0.8 ms | ≤1.9 ms (one 6.8 ms malloc hiccup) |

**`gpu == wall` on every row of every checkpoint.** The device timeline
fills the entire wall: the CPU submit path — all of it, launches, uploads,
allocs — executes entirely INSIDE the GPU's execution window. The
pre-registered GO premise ("the CPU path co-determines the wall") is false
here, and the ratio arm of the rule (multilingual max 0.46 < 0.5, median
~0.33) concurs. A graph replay would remove CPU work that is already free;
the remaining win is bounded at GPU-side inter-kernel gap reduction
(~300 launches × ~0.5-1 µs ≈ 0.15-0.3 ms ≈ 1-2 % of wall) — under the
lane's measured noise band (the `sgemm_shape_timing` control's ±8-10 %
two-context artifact band; the ±6 % idle per-round spread) — while the
signature-keyed arena + pinned-slot + staging-refresh machinery would add
exactly the stale-replay correctness surface `.issues/003` exists to
prevent, and per-signature capture costs more than it saves on novel
shapes (the serving stream's one-shot signatures). The pass TAIL (act head:
host math between three data-dependent downloads) is unreachable by graphs
by construction.

Two observations recorded for the future (the reopen triggers, both on the
instrument, one command away):
1. **The launch path costs ~15 µs/call** (submit minus gather/upload/alloc
   over ~300 launches) — expensive per call but FULLY HIDDEN at every
   current geometry. If the kernels ever get much faster (the ladder rungs
   compound, or a bigger GPU), the submit path becomes the wall and the
   ratio flips — re-run the probe before reopening, the number decides.
2. **Per-pass allocs are ~0.2 ms and uploads ~0.05-0.1 ms** — both already
   hidden; no slot-pooling or pinned-staging rung is warranted either (the
   cheaper remedies the issue §3 pre-identified die by the same evidence).

Gates at the close: `cuda_ops_smoke` 4/4 · `packed_forward_equiv` 3/3 · lib
41/41 · consumer G5 at the cuda posture 2/2 (top-1 1.0, drift ≤ gate) ·
`laya_batch_parity` 1/1 · clippy clean both repos · stats-off parity rows
byte-stable. No published number changes (nothing shipped that moves
them); the instrument is debug-only.

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
