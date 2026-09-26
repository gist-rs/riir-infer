# HISTORY.md — riir-infer

Durable records for resolved questions and closed lanes (the noise-reduction
convention: the record lands here, hash-pinned; open work lives in `.issues/`
and `.plans/`). Created 2026-09-23 at the first record.

## 2026-09-26 — Issue 018 CLOSED: the Cubecl-posture drift wobble was a softmax write-after-read race (fix `2a34bd3`)

Plan 611 S4's residual: the laya G5 gate at the cubecl posture held top-1
1.000000 in every run, but the prob-drift magnitude sporadically read
10–1000× its per-checkpoint floor, the outlier moving between checkpoints
and runs (9/20 G5 runs failing). The gate was `#[ignore]`d and every Cubecl
number held PROVISIONAL.

- **Root cause:** both CubeCL softmax kernels (`softmax_f32`,
  `softmax_rows_inplace_f32`) read the reduced max from shared slot 0 and
  then, with **no barrier**, let the sum phase write `smem[tid] =
  local_sum`, so thread 0 could overwrite slot 0 while a slower SIMD group
  was still reading the max. Scheduling-dependent, so worse under load.
  One `sync_cube()` after the read. The same commit fixes a latent chunked
  launch defect: rows past `MAX_WG_X` re-normalized rows 0.. (the chunk's
  first row now rides `params[1]`; heads·seq > 32768 only, i.e. long
  context).
- **How it was localized:** the issue's own lever 1 (the encoder probe's
  repeat loop, per-op drains). Every fire FIRST diverged at an `L*.attn`
  output, and the row-softmax is the only reduction inside the composed
  attention. A scan of every `let … = smem…[0usize]` read followed by a
  shared write before the next barrier found exactly the two softmax
  sites. The deltanet and two-pass-LN hits were false positives: only
  thread 0 consumes the deltanet value, and LN keeps separate arrays.
- **Refuted first, measured:** hypothesis 3 / lever 2 (a stream drain per
  `begin_pass`): 10 interleaved pairs, excursions to 5e-2 in both arms.
  Zero-filling every write-first slot (`client.empty` recycles pool bytes):
  excursions to 2.4e-2. Neither the drain nor the pool was the class.
- **Measured fix (M3, release, AC, load 12.8–16.6):** repeat-loop probe
  **43/120 fired passes at HEAD vs 0/120** with the fix (same tree,
  interleaved); G5 cubecl **10/10 PASS, every run bit-identical** at the
  floor (english 3.092e-6 · typed 2.233e-6 · multilingual 5.187e-6), then
  3/3 through the re-armed gate (riir-reflex `ccb5bd0`). Regression arms in
  `elementwise_cubecl::tests` (chunked-offset parity + a 200-rep
  bit-identical repeat at the 16×400 score geometry) both fail 5/5 with the
  fix reverted. gpu lib 215/0; clippy `-D` workspace `--all-features
  --all-targets` green.
- **The second class is contested (Issue 019).** A concurrent session
  (`riir-infer-m3-t7c`) landed `c0dfa06`, which emits Metal memory barriers
  from the vendored `wgpu-hal`, and recorded 018 as two classes (`270740c`).
  The measurement above was taken in a clean worktree at HEAD plus ONLY
  `2a34bd3`, with no barrier code present, so **A alone was sufficient for
  the observed wobble**. Class B's localizing evidence (`scores` diverging
  with q/k/v clean) is read after the in-place softmax, which is A's site.
  Its premise also does not match the fork's encoders, which never set a
  dispatch type and so are serial. Necessity and per-rebind cost are
  tracked in Issue 019. Nothing reverted.
- Cubecl numbers are no longer provisional; plan 611 S5 may publish.
- The issue file is removed; its full text (the pre-resolution trail + the
  two-class record as written by `riir-infer-m3-t7c`) is at `270740c`.

## 2026-09-26 — the riir-ai carve docs adopted into this repo (Issues 998 + 1003, Plan 610, Proposal 041, Benches 870 + 871, doc 002)

The riir-infer formation records moved home to the subject repo (owner noise-reduction pass on riir-ai), **numbers kept verbatim** so every existing "riir-ai Issue 998 / Plan 610 / Proposal 041" citation resolves to the same document:

- `.issues/998_riir_infer_repo_promotion.md` + `.issues/1003_riir_infer_carve_remains_4090.md` — both OPEN (998: S8 remains, gated on riir-reflex 008 P2/T4; 1003: 4090-executable content done, remains = the owner-gated S6b edge-drop franchise + S8). riir-ai's issue ledger keeps 998/1003 allocated; its `.highwater` is unchanged.
- `.plans/610_riir_infer_gpu_carve_slice1.md` — IN PROGRESS (S8 remains).
- `.proposals/041_riir_infer_model_based_inference_split.md` — the founding split proposal (Phase 1 landed; Phase 2 executing via riir-reflex Issue 008). First file in this repo's new `.proposals/` ledger.
- `.benchmarks/870_ternary_recipe_audit.md` + `.benchmarks/871_issue879_t1_gdn_quant_certification.md` — the Issue 879 GDN quant-survival records (871's harness is `tests/issue879_gdn_quant_certification.rs` here since the carve; Issue 879 itself was resolved in the riir-ai ledger, record in riir-ai HISTORY.md).
- `.docs/002_riir_infer_core.md` — the crate doc from riir-ai's docs book (renumbered 002 in this repo's `.docs/` ledger).

Cross-repo relative links inside the moved files were re-pointed; prose paths inside them remain written from the riir-ai side at writing time (each file carries a Moved header note saying so). riir-ai side: index rows struck + BOUNDARY.md/AGENTS.md citations re-qualified + HISTORY.md tombstone, same window.

## 2026-09-25 — Issue 001 CLOSED: EXL3 (trellis-coded) weight format — reader complete, fused-GEMV lane closed at this tier

The full record (§1–§17, section numbers unchanged, so every `Issue 001 §N`
citation in `src/quant/exl3*.rs`, `riir-infer-gpu`, the benches and plans
still resolves) moved to
[`.docs/001_exl3_trellis_format_support.md`](.docs/001_exl3_trellis_format_support.md).

- **Banked (opt-in `exl3`, not promoted):** the safetensors-side grouped
  reader + `quant/exl3.rs` (T2 Option A, no `GgmlType` coupling), the CPU
  reference and LUT+rayon fast arm (`725a8a5`, 10.6–11.4×), the pack loader
  (`bd93a7b`), the CubeCL GPU decode v1/v2 (Bench 002; 71–87× wall on the
  4090), the sound bench harness (`9989e9f`), the whole-pack bit-exact
  gate (`425e0c5`: 573/573 layers, 26.48 G weights, 0 mismatches on CUDA
  AND Metal), and the fail-closed era gate at open (`55c154f`, known-good
  `{"1.4.2"}`, `open_unverified_era` escape).
- **The axis that is real is residency** (Bench 001: 3.4× vs f16; context
  0 → ~117k on 24 GiB), not decode throughput. Decode runs 21–29 Gw/s and is
  NOT bandwidth-bound; the binding mechanism is UNMEASURED.
- **Closed at this tier:** T7c-2a/2b/3 (fused GEMV). By composition,
  26.5 Gw/step ÷ ~28 Gw/s ≈ 0.95 s/step against a 10–20 ms incumbent, so
  §14's delivered-gain trigger cannot be met. Reopen triggers r1–r4 are in
  §17.6. r1 (a serving consumer of the residency axis) is the owner of the
  "engine serving integration" line, and it needs a NEW issue when it fires.

## 2026-09-25 — T10 rung 2 (Metal): the attn_rope hoist pre-pass — default-off, promotion probe pending

`attn_rope` now derives the Q/K rope ONCE per layer into a packed
`[2, seq, d]` device scratch (`5ef7442`), and `flash_attn`'s staging copies
the rotated K instead of re-rotating it per (query block × head × key
tile). Opt-in `LAYA_METAL_ROPE_HOIST=1`; the in-kernel rope arm stays the
shipped default and is bit-identical by construction (same expressions,
same order) — promotion follows the quiet-box position-balanced A/B
(instrument archived: riir-reflex `.benchmarks/033_rope_hoist_ab/`), not a
flag flip.

The landing's own defect is the reusable lesson: widening the buffer list
moved the flash staging to `[[threadgroup(11)]]` while the host kept
binding its length at index 9 — the unbound-pointer class. Small smoke
shapes passed on allocation luck; full-model G5 failed DEGENERATE (uniform
probs, agreement 12/26), which is what surfaced it. Fixed (9 → 11) in the
same commit: a shape-widening edit must move the host bind and the kernel
binding together.

Gates (both postures, hoist on/off): `metal_ops_smoke` 7/7 ×2,
`packed_forward_equiv` 4/4 ×2, `packed_same_shape_gate` 1/1 raw-bit,
lib 41/41; consumer G5 metal top-1 1.000000 ×3 checkpoints, drift ≤ 6.1e-6
×2; `laya_batch_parity` ×2; clippy `-D` ×3 feature postures.

Session: issue020-metal-lane, 1790285773

## 2026-09-25 — two silent MiniCPM5 (llama-arch GGUF) defects, found bringing it up as Issue 011's long-context fixture

Both defects produced finite, plausible output that was wrong, and both were settled against a reference implementation (HF `tokenizers` / `transformers` fp32) rather than by reasoning.

- **RoPE pairing — `0b26b9a`.** llama.cpp's converter (`LlamaModel.permute`) stores llama Q/K rows in the interleaved order (GGUF row `2j+s` = HF row `s·hd/2 + j`). This crate's RoPE, CPU and `riir-infer-gpu` alike, is rotate-half, and `load_llama_weights_gguf` never un-permuted. On MiniCPM5-1B, tinyshakespeare BOS+256, riir ppl went **259.93 → 69.0547**, against **69.0547** from HF fp32 on the same ids. Fix: `rope::unpermute_interleaved_rows`, llama loader only (gemma2/qwen2 GGUFs are NEOX and unpermuted).
- **BPE digit rule — `175f67f`.** `BpeTokenizer` hard-coded Qwen2's one-digit-per-pre-token rule for every GGUF. llama-bpe groups `\p{N}{1,3}`. Text without numbers tokenizes identically, which is why a 1000/1000 tinyshakespeare diff passed. After the fix, 54 271 of 54 271 ids match on number-dense text. How it was isolated: the passkey needle failed 0/2 at base, and HF `transformers` scoring riir's EXACT prompt ids failed on the same digits, so the fault was in the ids and not the forward. Now 2/2, answer ppl 2.81 → 1.21.
- **Downstream:** every consumer that re-exports this loader or tokenizer inherits both fixes. The riir-train canon cross-arch benches (422/423/424/426/427/605) paired Gemma-2 with this MiniCPM5 forward, so their verdicts are unverified: riir-train Issue 571.
- **Instrument kept:** `row_logit_floor_ppl --dump-tokens N`. In ppl mode it prints the corpus ids; in needle mode it prints each prompt as `score_from|ids`. It is the diff against a reference that found both defects.

## 2026-09-25 — Issue 010 CLOSED (REFUTED): Gemma-2 has no QK-norm; the 288-tensor fixture is upstream-faithful

Issue 010 (`95f52fb`) claimed `riir-train/data/gemma-2-2b-it-f16.gguf` was a
mutated conversion: 288 tensors with no `attn_q_norm`/`attn_k_norm`, against a
supposed llama.cpp standard of 340 (13/layer), and that upstream Gemma-2 applies
per-head q/k RMSNorm. **That premise is false.** Checked against upstream
HuggingFace `models/gemma2/modular_gemma2.py`: `Gemma2Attention` defines no
`q_norm`/`k_norm`, and bounds scores with `attn_logit_softcapping` (tanh, cap 50),
which this repo's forward already applies (`attention_head_softcap`). Per-head
QK-norm arrived in Gemma-3, where it replaced softcapping. The 11 tensors/layer
(attn_norm, q, k, v, o, post_attention_norm, ffn_norm, gate, up, down,
post_ffw_norm) × 26 + 2 globals = 288 is the standard shape.

- The fixture and forward are upstream-faithful. Option (a), re-converting, was
  never needed. Nothing pinned to the old artifact needs a re-run.
- `transformer/gemma4.rs`'s "q/k norm … NEW vs Gemma 2" doc was **correct** and
  stays.
- Corrected in the same commit: the `vk_calibration` bin's caveat 1 (doc and the
  two printed lines) and the `gemma2_calibration.rs` tap-law paragraph. For
  gemma-2 the pre-RoPE K tap is exactly the cache's K. Issue 883 trap 1's
  post-QK-norm wrinkle applies to gemma-3/4-class stacks only.
- Why it happened: a tensor count was compared against a remembered "standard"
  rather than against the upstream model definition. Check the reference source
  before filing a fixture-integrity finding.

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

## 2026-09-25 — CalibrationTables promoted substrate-side (883 P0 Kimi fixture rider)

The gemma-2 harness's layered table builder moved upstream:
`katgpt_core::fitted_anchor_table::LayeredVkCalibration` (with
`VkLayerTables`) is now the ONE builder for both 883 P0 fixtures — this
repo's gemma-2 dashboard and katgpt-rs's new Kimi-K3 dashboard (katgpt-rs
Bench 889, real weights: ρ(V)≈ρ(K)≈ρ(V−K) per MLA layer — the "coupled
through one latent" signature; KDA layers = fixture-class null, no KV
cache). `gemma2_calibration.rs` re-exports it under the historical name
`CalibrationTables` — zero behavior change, the `vk_calibration` bin
compiles + clippy-clean unchanged. The tap forward, corpus loader, and
dashboard format stay here (gemma-specific); the row-map/triplet/scratch
plumbing is substrate-owned (DRY: one builder, two fixtures).

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

## 2026-09-25 — Issue 020 T6 CLOSED NEGATIVE: the encoder's host side is 1–1.6% of forward wall (measured before building)

T6 proposed pooling the per-forward allocation churn — ~60 MB of host `Vec`s
rebuilt every forward (`encoder.rs` `Scratch::new()` + `gathered`/`h`/`out`/
rope tables) plus every activation device buffer freed by the per-pass chain
clear (`metal.rs` `begin_pass_impl`). The 09-25 head scratch-pool rung (built,
measured, moved nothing, reverted — the consumer repo's issue-020 follow-up)
already said the head is dispatch+GPU bound; this measurement settles the
encoder's half the same way, and it was taken BEFORE building the pool (the
discipline the head rung paid for).

Instrument: `crates/riir-infer-laya/tests/metal_host_gpu_split.rs` — a
`#[ignore]`d, `required-features = ["laya-riir-metal"]`-gated, measurement-only
probe (`--ignored --nocapture`). It splits one `Encoder::forward_packed` at the
real english geometry (d 1024, 28 layers, intermediate 2624) into `enq` (the
wall of the forward alone — the body contains no sync, so that is the whole
host side: MSL dispatch encoding, the chain uploads + destination slots, the
host `Vec` churn) and `sync` (`download_into`'s commit + wait — the GPU side).
The decision rule was recorded in the probe doc before measuring: pooling can
shrink only part of `enq`; if `enq` is a small fraction of the wall, T6 closes
NEGATIVE; a host-bound reading under load defers to a quiet box.

Measured 2026-09-25, M3, AC, load 11–12 (falling), 88% RAM free, one sibling
CPU bench (~6 cores, memory-bandwidth pressure — which inflates BOTH the host
reading and the GPU's unified-memory reads), no GPU consumers, 9 rounds/shape,
2 warmups:

- seq188: enq p50 **0.80 ms** (min 0.64) · sync p50 **48.7 ms** (min 36.1) — host share **1.6%**
- seq512: enq p50 **1.45 ms** (min 1.29) · sync p50 **141.6 ms** (min 132.6) — host share **1.0%**
- packed2x256: enq p50 **1.37 ms** (min 1.22) · sync p50 **135.2 ms** (min 130.7) — host share **1.0%**

The host reading is load-INFLATED (CPU contention inflates the host, never the
GPU), so the verdict is robust in the recorded direction: a quiet box only
shrinks the 1.0–1.6%. Allocation pooling can shrink only part of that share —
dispatch encoding stays — and cannot move case wall. The per-pass chain clear
STAYS (its staleness guard is load-bearing: host-authored buffers are rebuilt
per forward at recycled heap addresses, the Issue-015 class). The probe stays
as the standing instrument for any future host-side rung claim.

Also landed in the same commit: the `[[test]]` required-features row for the
new target (the T1.1e repo-birth law — the row keeps a feature-less selection
skipping loudly instead of printing a green zero).

Session: issue020-t6, 1790323200

## 2026-09-25 — the narrow sgemm's shape is the measured local optimum (T7 occupancy axis refuted, both arms)

The question a future kernel reader will ask: narrow stages A[32][65] +
B[64][65] = 24 960 B — one threadgroup per core — why not shrink the staging
so 2–3 co-reside and hide the staging-load latency? Measured from the
consumer's `sgemm_shape_timing` probe (reflex `ebe667e`, the record lives in
its issue-020 T7 section): two challengers behind a temporary
`LAYA_METAL_SGEMM_VAR` flag — **bk32** (BK 32, same 32×64 tile, 12 544 B → 2
TGs/core) and **bn32** (32×32 tile, 8 448 B → 3 TGs/core), both keeping the
k-ascending per-element chain (every row bit-identical to `sgemm`).

bk32 LOST 17–33% on every resolvable cell, growing with k exactly as the
barrier model predicts (BK 32 doubles the k-loop's two barriers); bn32 lost
harder (15–45% — the same doubling plus halved B reuse). The 2–3× co-residency
gain is strictly smaller than the barrier cost. At BK 64 a 2-TG fit would
break the bank-conflict padding (stride 65) or the 8×8 block structure (BN ≤
24 idles 4 of 16 simdgroups); wide (BM 64) already lost zero-for-zero in the
band rung. The variant code never landed — this record and the reflex issue
carry the negative, the BK=48 precedent.

Session: issue020-occ, 1790323200

## 2026-09-25 — the sgemm MMA-roofline probe: the narrow instance is staging-bandwidth-bound (reflex Issue 020 T7 follow-up)

`crates/riir-infer-laya/examples/sgemm_roofline.rs` (new, measurement-only,
`[[example]]` required-features row per the T1.1e law): the f16-axis
discriminator. A verbatim copy of the shipped narrow kernel (CPU-drift-checked
on the ragged cell) against an MMA-only twin — same inner k-chunk, staged
ONCE, `reps` iterations of the 8-wide chunk with no re-staging and no
barriers, FLOPs matched by reps = k/64, position-balanced rounds in one
process. Measured (AC, load 2.4–3.1): 317×3072×1024 — narrow 3.13 TF/s vs
roofline 5.11 (+63%); 1024×3072×1024 — 4.88 vs 9.21 (+89%); 1024×8192×1024 —
4.37 vs 10.24 (+134%). The traffic math closes: per k-tile each threadgroup
stages 24 KB (A 8 + B 16), so cell 3's B re-reads alone are ≈1.6 GB ≈ the
measured 3.93 ms wall at ~400 GB/s. Verdict: NOT MMA-bound — the f16 lever
is the B-OPERAND BYTES (halving staging + device reads; predicted ~1.4–1.7×
on the hot cells), not the f16 MMA; double-buffering is dead with it
(bandwidth-bound, not latency-bound). Numerics note for the rung: f16 weight
rounding is ~4.9e-4 relative per term (< the 1e-3 G5 gate) but promotion is
an Issue-750-T3 lossy-surface call — per-family retention, never the
aggregate.

## 2026-09-25 — f16-B staging REFUTED at kernel level (the roofline probe's own follow-up arm)

The roofline verdict (staging-bound, +63..+134% MMA-only headroom) predicted
the f16-B lever: halve the B-operand bytes (16 of 24 KB staged per k-tile),
halve the dominant traffic, ~1.4-1.7x on the hot cells. Measured (the probe
extended with a `sgemm_hb` arm — `device const half*` B, converted to f32 at
staging, everything after the staging bit-identical; f16 seed via a
round-to-nearest-even `f32_to_f16`; plumbing verified against the f32 arm
within the 2e-2 rounding gate): **flat within ±2% on every cell** — 317x3072x1024
−1.1%, 1024x3072x1024 +1.9%, 1024x8192x1024 −0.8% (the deep-k cell where the
model predicted the most). Load 8.9-9.5 (sibling resumed) — irrelevant: both
arms inflate together and the ratios are the decision axis.

Mechanism, now measured rather than modeled: B (1024x3072 f32 = 12.6 MB, f16
= 6.3 MB) FITS IN L2 on this GPU (~32 MB), so B's per-m-tile re-reads were
never DRAM traffic — the binding cost is the L1/threadgroup-issue path
(TG writes + simdgroup loads + the barriers ordering them), which halving
bytes does not relieve. Same mechanism as the 09-24 coalescing negative
("sector waste already absorbed by L2/MLP"), one level deeper.

Consequences: the f16-B BACKEND rung (f16 transposed weight cache + dispatch
+ G5 re-gate + the Issue-750-T3 per-family retention walk) is dead before
being built — days of lossy-surface work for a measured ±0%. With this, five
axes are refuted at kernel level (occupancy bk32/bn32, BK=48 barriers,
coalescing x2, f16-B) and the roofline gap (+63..+134%) is the staging-issue
path itself, which none of the tried geometries reaches. Narrow's shape is
the measured local optimum on this hardware/toolchain. Reopen triggers: a
Metal/toolchain change exposing direct-to-MMA staged layouts, or an L2-
oversized working set (n > ~4096 changes B's residency class — the packed
path's n grows with question count, worth re-probing there first).
