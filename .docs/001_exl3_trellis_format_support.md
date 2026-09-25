# Doc 001 — EXL3 (trellis-coded) weight-format support: the Issue 001 record

**Status:** CLOSED 2026-09-25 — relocated from `.issues/001` (the noise-reduction rule; section numbers preserved so every `Issue 001 §N` citation in code and docs still resolves here). Every task is done or closed at this tier: T1–T7 + T7c-1a/1b/1c/1d + the T7c-4 era gate DONE; T7c-2a/2b/3 CLOSED-AT-THIS-TIER with reopen triggers r1–r4 (§17.6); the §14 promotion stays REFUSED with its trigger intact. Engine serving integration is reopen trigger r1, not an open task here — file a new issue when a consumer appears. Closing summary: HISTORY.md § 2026-09-25 Issue 001.
**Status at close (verbatim, for the record):** OPEN (reader complete; fused-GEMV lane CLOSED) — T1–T7 COMPLETE 2026-09-24 (T4b native oracle BIT-EXACT; T7a CPU 10.6-11.4×; T7b GPU 4090 71-87× wall / 4.7-5.8 Gw/s kernel, [Bench 002](../.benchmarks/002_exl3_t7b_gpu_dequant.md)). **T7c: T7c-1a/1b/1d/1c DONE 2026-09-24/25** (v2 bit-exact; extraction kill criterion fired + confirmed by the sound harness — extraction CLOSED; LUT gather = the one real 1.25×; decode wall 21-29 Gw/s, NOT bandwidth-bound at ~3.5 GB/s of ~1008 GB/s — binding mechanism UNMEASURED, candidates: dependent-load latency / occupancy-ILP / LUT-gather serialization; whole-pack bit-exact gate green 573/573 layers / 26.48 G weights, CUDA + Metal). **T7c-2a/2b/3 CLOSED-AT-THIS-TIER 2026-09-25 by the owner-delegated Claude verdict ping-pong (session `36dc0297`, round-3 REVISE upholding CLOSE) — the fused GEMV's step time is computable from landed measurements (26.5 Gw/step ÷ ~28 Gw/s ≈ 0.95 s vs the 10-20 ms incumbent) and §14's trigger asks for delivered gain vs that incumbent; reopen triggers r1-r4 recorded at §17.5. T7c-4 SPLIT: the era-gate-at-open half LANDED 2026-09-25 (`verify_pack_era` on `quantization_config.version`, fail-closed, `open_unverified_era` escape, known-good = {"1.4.2"}); the promotion half closes UN-DISCHARGEABLE at this tier — §14 stays REFUSED with its trigger intact, now backed by a real refusal instead of a doc comment.** Engine serving integration remains separately open, not implied-complete.
**Owner:** unassigned. **Filed:** 2026-09-24. **T1–T4 executed:** 2026-09-24 (4090 box).
**Origin:** riir-clippy Research 207 (`.research/207_qwen38_exl3_dgx_spark_distill_verdict.md`),
lane-intel axis. The corpus half of that verdict is riir-clippy Plan 170 and is
independent of this issue — neither blocks the other.

> ⚠ **Read the scoping clause (§2) before writing any promise into a plan.**
> On the one measured source available, EXL3's bpw advantage converts to
> **memory headroom and context ceiling, not decode throughput**. An issue
> written around expected decode wins would be promising the wrong axis.

## 1. Why this repo, and why this is the only repo

`BOUNDARY.md` puts *"the quantization zoo (q2k → q8kv, q2_0 ternary,
PTQ/TurboQuant)"* under `riir-infer-core`, and the domain test — *is this
model-based inference substrate (weights, quant, architectures, kernels,
loaders)?* — answers YES without qualification. A weight format is the
canonical example of what this repo owns.

Measured at filing time: **zero** `exl3` / `exllama` / `trellis` / `qtip`
occurrences anywhere in this repo. The format is absent; the home is
unambiguous.

The mirror check was also run and is recorded so it is not re-derived:
**katgpt-rs is the wrong home by its own contract** — its domain test is *"is
this a modelless inference primitive with no riir dep?"*, and a weight-format
loader is not a modelless primitive. Do not file a sibling issue there.

## 2. Scoping clause — what EXL3 buys, and on which axis (LOAD-BEARING)

This is the transferable finding from Research 207 candidate #3, and it is a
**warning rather than an encouragement**. Measured on one NVIDIA DGX Spark
(GB10, 128 GB unified aarch64) serving a Qwen3.8-Flash-Next EXL3 pack, same
build and same harness, 3.05 bpw vs 4.05 bpw:

| quantity | 3.05 bpw | 4.05 bpw | reading |
|---|---:|---:|---|
| decode @4k (tok/s) | 52.22 | 48.70 | inside the sample spread |
| decode @32k (tok/s) | 50.53 | 51.31 | inside the sample spread |
| cold prefill (tok/s) | 1,142 | 1,137 | inside the sample spread |
| resident | 79 GiB | **100.81 GiB** | **+22 GiB** |
| utilisation forced | 0.80 | **0.92** | — |
| context ceiling | 262,144 | **131,072** | **halved** |
| MemAvailable | 17–18 GiB | **3.4 GiB** | — |
| draft acceptance | 68% | **74%** | a *speculation* win |

**Mechanism:** on a trellis/codebook format the per-weight dequantization cost
barely grows with K, so decode is bound by **dequantization compute** rather
than by bytes moved — and the usual *"lower bpw ⇒ faster decode"* heuristic
**inverts**. The defensible promise for this repo is therefore **residency and
context ceiling**, with acceptance as a secondary speculation-side effect.

⚠ **This table is n=1: one box, one author, one model, one cell per number.**
It is enough to refuse a wrong promise and not enough to make a right one. Any
GOAT gate this issue eventually feeds must re-measure on our own silicon.

**The memory-side escape, if a pack does not fit** (candidate #4): moving a
large embedding/lookup table off-device to memory-mapped checkpoint views makes
every lookup a host synchronization point, so it cannot sit inside a full CUDA
graph — the run drops to PIECEWISE-only with the lookup as the splitting op.
The important part is that the **graph-mode downgrade is priced independently
of the mechanism that forced it**: PIECEWISE alone measured 5–7% on the same
model with the table still resident, so the gather's own cost is the residual,
not the whole ~10%. Net on that source: ~10% slower, +15 GiB headroom, and 262k
context reachable where the resident table cannot reach it.

## 3. What the format is, and where the math lives

- **Format:** EXL3, from [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3)
  (MIT) — see that repo's `doc/exl3.md`. Packs are **safetensors**-borne, not
  GGUF.
- **Math:** EXL3 is a streamlined variant of **QTIP** (Cornell RelaxML, NeurIPS
  2024, [arXiv:2406.11235](https://arxiv.org/abs/2406.11235)) — trellis-coded
  vector quantization with **procedural codebooks** (the codebook is computed,
  not stored) and **incoherence processing** (a Hadamard-style rotation applied
  before quantization, inverted after).
- **Lineage is already read on the research side** — katgpt-rs Research 502
  places QTIP in the AVQ2 / affine-lattice / VPTQ / additive-quantization
  lineage. This issue **inherits** that literature read rather than opening one.
- **Measured-configuration reference:** `vcruz305/Qwen3.8-Flash-Next-EXL3-DGX-Spark-recipe`
  @ `e0ebee8ddc4391b5e66d86fdce993b0756e00986` (MIT, Victor Cruz). The pin is
  load-bearing — that source, its exllamav3 fork and the `vllm-exl3` plugin all
  move; re-verify every quote at the pin.

## 4. The dispatch shape is ALREADY adjudicated — do not re-litigate it

riir-ai Research 360 (`.research/360_M4_Prefill_Engine_Metal_Kernel_Distill.md:118`)
already names **"EXL3 codebook"** as one of six formats in a Universal Quant
Router, and adjudicated that router shape a **positive instance of B29
`enum-dispatch-separate-kernel-per-variant` + B23 — "No action."** So the shape
of adding a format is decided; only the format itself is missing.

This repo's existing dispatch matches that shape already:
`src/gguf_loader.rs` carries `enum GgmlType` (line 47) with `match` arms per
variant (line 555 and line 669), and `src/quant/` is one module per format
(`q2k.rs` … `q8kv.rs`, `q2_0.rs`, `ptq1_0.rs`).

⚠ **But the GGUF enum is the wrong seam for EXL3.** An EXL3 pack is safetensors
— the tensors arrive through `src/safetensors_loader.rs`, which today handles
**BF16 only** (it dequantizes BF16 → f32 and nothing else) and carries a
`dtype: String` per tensor rather than a typed enum. An EXL3 layer is several
safetensors entries (trellis codes plus the scale / rotation sidecars), so the
seam is **a grouped-tensor reader over the safetensors metadata**, not one more
`GgmlType` arm. Whether the two loaders should converge on one typed quant enum
is an open design question this issue does not pre-decide.

## 5. Tasks

- [x] **T1 — Format spec read at the pin.** Read `doc/exl3.md` at a pinned
  turboderp-org/exllamav3 commit and record the on-disk layout here: tensor
  naming, the per-layer entry set, bit packing, and exactly which sidecars a
  dequantization needs. Record the pin. Nothing below starts before this.
  → **DONE, record in §10.** ⚠ The T1 wording above cited `doc/exl3.md` —
  that file does NOT exist at the pin (stale citation, corrected in §10.0);
  the spec was read from the implementation + `doc/convert.md`.
- [x] **T2 — Decide the loader seam.** Grouped-tensor reader in
  `safetensors_loader.rs` vs a typed quant enum shared with `GgmlType`. §4
  states the constraint; the choice is an owner/design call and the reason goes
  in this file, not in a commit message.
  → **DECIDED 2026-09-24: Option A (safetensors-side grouped reader +
  `quant/exl3.rs`), NO GgmlType coupling** — Claude-verdict ping-pong round 1
  REVISE→incorporated, round 2 AGREE. The decision record with the full
  rationale + four seam conditions: **§11**.
- [x] **T3 — CPU reference dequantization** (`src/quant/exl3.rs`), scalar and
  correct before fast. The procedural codebook and the incoherence rotation are
  the two pieces with no analogue in the existing zoo.
  → **DONE 2026-09-24** (landed behind the opt-in `exl3` feature, per the
  workspace promotion rule — T6 decides default-promotion). Record in **§12**:
  `Exl3Codebook` (all three cbs, exact f16 semantics + one documented
  `__hfma` emulation caveat), `Exl3K` (integer + half-integer from
  `trellis.shape[-1]/16`), the tail-biting ring reader in **tensor-core
  element order** (a T1 addendum the record now carries), Sylvester-128
  Hadamard, packed-sign unpack, the zero-copy `Exl3Layer<'a>` honoring all
  four §11.3 conditions, and `detect_exl3_layers` over the safetensors
  metadata map (`TensorMeta` → `pub(crate)`). 11 tests: independent
  numpy-computed codebook pins, ring round-trips (int + half-K), tail-biting
  wrap, tensor-core permutation bijection + spot checks, Hadamard
  orthogonality/symmetry, sign unpack, layer validation (markers/K/dims),
  and a full-dequant cross-composition. clippy `-D warnings` clean at both
  feature postures; 197/186 tests green (feature on/off).
- [x] **T4a — Real-pack validation (DONE 2026-09-24; the cross-implementation
  oracle half of T4).** Dequantize a real pack and compare — record in
  **§12.5**. **T4b (REMAINS): the exllamav3-native oracle** — run
  exllamav3's own dequant on the same tensors (torch + CUDA ext build on
  the 4090; the MSVC fix at the pin suggests Windows is supported) and
  compare per-tensor numeric error per §6 (never text equality).
- [x] **T4b — exllamav3-native oracle (DONE 2026-09-24; see above).**
  The env WAS built same-session (torch 2.14+cu130 + the ext JIT under MSVC
  14.41/nvcc 13.3; one build workaround + one clone-local build-flags patch,
  both documented in §12.6) and the comparison ran — record in **§12.6**.
  **VERDICT: BIT-EXACT on a pin-era pack** — and the old-pack discrepancy
  that almost read as a spec bug resolved as a PACK-ERA mismatch (§12.7).
- [x] **T5 — Re-measure §2 on our silicon** (4090 / M3) before any promise
  about residency or throughput enters a plan, a README or a league row. The §2 table
  is n=1 on hardware we do not have.
  → **DONE 2026-09-24 (4090 box), record in §13**: loader wiring landed
  (`Exl3Pack`, `bd93a7b`), the LEAGUE MODEL's own author pack downloaded
  (16.35 GB, sha256-verified) + 4 more bpw branches measured metadata-only;
  residency + context-ceiling tables recorded; the §12.7 era-gate applied
  end-to-end (native oracle: rel-Frobenius ≤1.3e-07 over 5 layer classes
  incl. lm_head). NO throughput claim licensed (T7's arms are the
  denominator) — §13.6 states exactly what is and is not established.
- [x] **T6 — GOAT gate + feature flag**, per the workspace promotion rule: ship
  behind an opt-in feature, bench the claim, promote only on a measured gain on
  the axis §2 says is real.
  → **DECIDED 2026-09-24 (record in §14)**: G1–G4 evaluated on T3–T5
  evidence; the residency gain is measured ([Bench 001](../.benchmarks/001_exl3_t5_residency_context.md))
  and modelless — but the feature **STAYS OPT-IN**: promotion requires the
  gain to be delivered ON the promoted path, and nothing on any default
  build path consumes EXL3 until T7 wires a serving/GPU arm. Named
  promotion trigger in §14.
- [x] **T7 — SIMD / GPU arms**, only after T3–T5. Scope unknown at filing.
  → **T7a (CPU arm) DONE 2026-09-24 (`725a8a5`, record in §15)**: codebook
  LUT + rayon parallel dequant, bit-identical to the reference,
  10.6–11.4× measured on the 27B pack. **T7b (GPU arm) DONE 2026-09-24
  (record in §16, [Bench 002](../.benchmarks/002_exl3_t7b_gpu_dequant.md))**:
  three CubeCL kernels (decode / left-H / right-H, grid-strided,
  chunk-streamed), decode stage BIT-EXACT + Hadamard stages gated in the
  FMA-contraction class, **71-87× wall / 4.7-5.8 Gw/s kernel-only** vs the
  CPU arm on the 4090. The §14 promotion trigger remains (engine serving
  integration + delivered-gain measurement — engine-repo work).

## 6. The correctness gate has a precondition — measure the comparator first

Research 207 candidate #1 (`aa-control-prices-the-fidelity-instrument`) is
directly load-bearing for T4 and is recorded here so T4 does not have to
rediscover it:

**Before using output-text equality as a fidelity gate, run the A/A control —
the same configuration against itself — and compare its disagreement rate to
the A/B's.** On the source measured, the serving engine was non-deterministic
at temperature 0 with a fixed seed (**4 distinct outputs from 5 identical
requests** on an open-ended prompt, 1 of 5 on a constrained one), and
**spec-on vs spec-on scored the same 2-of-6 identical as spec-on vs spec-off**
— the control was indistinguishable from the test, so the text diff was
measuring engine noise and nothing else.

Two consequences for T4:

1. Prefer a **counted, direction-known signal** (per-tensor numeric error
   against the reference dequantizer; acceptance counters) over text equality.
2. The divergence is **prompt-class-conditional** — open-ended generations
   diverge, tightly constrained ones do not — so a fidelity suite built only
   from constrained prompts reports a **clean instrument that has no power**.

## 7. Explicitly NOT in scope — each with its closer

- **Absolute tok/s from the source** (79 / 62 / 53 native, 52.22 @4k vLLM,
  1,142 cold prefill). GB10 aarch64 unified 128 GB, a 512-expert top-10 MoE.
  Our league cells are 4090 / M3 on Bonsai ternary. Format-, model- and
  box-crossed rows are never a headline and never a gate change.
- **MTP / speculation economics** (acceptance 68–74%, draft-depth sweeps,
  dynamic-draft policy). riir-ai Issue 721 closed the tree-masked batched-verify
  path as a **structural** negative (G2 FAIL cross-substrate on M3 Metal *and*
  4090 Vulkan: verify cost measured ~linear in T, only ~1.4–1.5× weight-read
  amortization, perfect-drafter bound ~21×), and left riir-ai Issue 717's G3a
  re-gate CLOSED with named reopen triggers — the live one being a
  weight-stationary T-amortized ternary GEMM. A *higher-precision target
  agreeing with its draft more often* is evidence **for** that verdict, not a
  trigger: acceptance was explicitly not the binding axis.
- **The vLLM / SGLang plugin surface** (`vllm-exl3`, `sglang-exl3`,
  `--quantization exl3`, tensor-parallel size 1 only) — not our runtime.
- **Serving plumbing** (TabbyAPI, OpenAI-wrapper shims, preflight scripts) —
  not this repo, by `BOUNDARY.md`'s hard fences.
- **`--gpu-memory-utilization` ceiling lore** from the source — box-specific
  operational tuning for a machine we do not have.
- ⛔ **The Research 120/139 arena numbers do NOT join to this source.** That
  arena is Qwen 3.8 **27B** (64 layers, hidden 5120, vocab 248320) on
  MLX/Metal; this is Flash-Next on CUDA/GB10 (48 layers, hidden 2560, 512
  experts top-10). Same family name, different geometry, different silicon,
  different runtime — **do not let the shared name join them.**

## 8. Open questions

- Does EXL3 support belong behind one feature flag, or does the incoherence
  rotation warrant its own (it is reusable — `bonsai2_hadamard` already exists
  in this repo and may share machinery)?
- Is a real EXL3 pack obtainable on our boxes at a size worth loading? T1–T4
  need one; a synthetic fixture can carry T3 but not T4.
- Does this repo want a **typed** quant enum spanning both loaders (§4 T2), or
  does the GGUF/safetensors split stay as it is?

## 9. Provenance and license hygiene

Source chain, three deep, each MIT: the recipe → the author's own exllamav3
fork (PR #2, fork `master` @ `523ecd3`) → turboderp-org/exllamav3 → QTIP
(arXiv:2406.11235) for the math. **No third-party code is pasted into this
repo.** Any fixture is an ORIGINAL minimal reproduction; the reference
implementation is consulted for the FORMAT, and the format is a fact about
bytes on disk, not an expression.

## 10. T1 record — the EXL3 on-disk format as read at the pin

Executed 2026-09-24 (4090 box). Everything below is read from the source at
the pin, not from prose; each subsection names the file the fact came from.

### 10.0 The pin, and one corrected citation

- **Pin:** turboderp-org/exllamav3 @ `6b84a21b6f1e5da3f291b9e1019061f0de788279`
  (tip at read time, dated 2026-09-20, *"MSVC fix (__builtin_popcount)"*).
  Clone at `.raw/exllamav3` (kept until T4 — it is the reference oracle and
  the fixture source; re-verify every quote at the pin).
- ⚠ `doc/exl3.md` (the path this issue's T1 cited) **does not exist at the
  pin** — `doc/` carries only `convert.md`, `env_vars.md`, `optimize.md`.
  The format facts below come from the implementation: `modules/quant/exl3.py`,
  `modules/quant/exl3_lib/quantize.py`, `exllamav3_ext/quant/pack.cu`,
  `exllamav3_ext/quant/codebook.cuh`, `exllamav3_ext/quant/exl3_dq.cuh`,
  `modules/quant/exl3_lib/ngram_codec.py`, `modules/linear.py`, `doc/convert.md`.

### 10.1 Container and detection

- A pack is a **directory of safetensors shards** (standard HF-style layout;
  `config.json` + shards; the n-gram table ships as a standalone
  `ngram_embedding.safetensors`). No GGUF involvement.
- A linear layer at key `K` is EXL3 iff the safetensors metadata carries the
  group `[{K.sv | K.svh}, {K.su | K.suh}, K.trellis]` (`modules/linear.py`
  `is_exl3_storage`).

### 10.2 Per-linear tensor set (the dequantization's complete inputs)

| entry | dtype | shape | required | role |
|---|---|---|---|---|
| `K.trellis` | int16 (bit-packed codes) | `[in_features/16, out_features/16, 16*K]` | YES | the codes (§10.4) |
| `K.suh` | fp16 | `[in_features]` | one of su/suh | per-input-channel scale (§10.6) |
| `K.svh` | fp16 | `[out_features]` | one of sv/svh | per-output-channel scale |
| `K.su` | int16 (packed ±1) | `[in_features/16]` | — | LEGACY packed-sign spelling of suh (bit i of word j = sign of channel 16j+i; `1 − 2·bit`) |
| `K.sv` | int16 (packed ±1) | `[out_features/16]` | — | legacy spelling of svh |
| `K.mul1` | int32 marker | `[]` (0-dim scalar; 1 elem) | opt. | selects codebook cb2; value = `0x83DCD12D` as u32-be-int (`codebook_mul1_mult`). REQUIRED for half-integer K. The DEFAULT modern pack (`doc/convert.md -cb`: "mul1 (default)") — **real-pack-verified 2026-09-24: Terra3312/GLM-5.3-Flash-EXL3-4bpw-MUL1 carries .mul1 on all 887 groups of shard 1, I32 0-dim, value exactly 0x83DCD12D** |
| `K.mcg` | int32 marker | `[]` (0-dim) | opt. | selects codebook cb1; value = `0xCBAC1FED` (`codebook_mcg_mult`) |
| `K.bias` | fp16 | `[out_features]` | opt. | plain bias |
| `K.scale` | — | — | removed | loader passes `None`; `assert scale is None, "scale is no longer used"` |

**Bitrate is self-described by shape:** `K = trellis.shape[-1] / 16`
(`LinearEXL3.__init__`), integer or half-integer; a half-integer K
(= ka + 0.5, alternating ka/ka+1-bit steps, period 16) is legal only with
`.mul1` present. Per-tensor K ∈ 1..8 (±0.5); the CONVERTER's `-b 3.75`
style averages are achieved by per-layer allocation, not fractional-word
tiles (a tile at half-integer K is still a whole number of u16 words:
`16·K` words).

**No on-disk global scale exists.** The quantizer's global scale search and
`out_scales` are folded into `suh`/`svh` (see `refit_scales`, §10.6).

### 10.3 The three procedural codebooks (verbatim ops, `codebook.cuh`)

All three map a 16-bit code word `x` (zero-extended to u32) to one fp16
value. `lop3(a,b,c,0x6a)` = `(a & b) | c`; `half2(x)` reads the u32 as two
fp16 lanes; the result is the sum of the two lanes (fp16 add).

- **cb0 "3inst"** (no marker): `x = x·89226354 + 64248484 (mod 2³²)`;
  `x = (x & 0x8fff8fff) | 0x3b603b60`; `v = half16(x[15:0]) + half16(x[31:16])`.
- **cb1 "mcg"** (`.mcg` marker): `x = x·0xCBAC1FED (mod 2³²)`; same
  lop3-half2; `v = half16(low) + half16(high)`.
- **cb2 "mul1"** (`.mul1` marker, the default): `x = x·0x83DCD12D (mod 2³²)`;
  `s = dp4a(x, 0x01010101, 0x6400)` (byte sum + 0x6400); take `s[15:0]` as
  an fp16 BIT PATTERN (0x6400..0x67FF = 1024.0..2047.0); `v = fp16(s[15:0]) · (1/147.7) + (−10.39)`
  (half-precision constants `0x1eee`, `0xc931`).

### 10.4 Trellis tile structure and bit packing

- Weight matrix is quantized **transposed**: `[in_features, out_features]`
  (`quantize_exl3` docstring: "row major shape (in_features, out_features)").
- Tiles are **16×16**: tile (a, c) covers in-rows `[16a, 16a+16)` ×
  out-cols `[16c, 16c+16)`; the 256 weights are ordered row-major with the
  IN offset as the row (`W4[a,:,c,:].reshape(-1, 256)`).
- Per tile: **256 codes of K bits each** — the low K bits of each weight's
  16-bit trellis state — packed MSB-first into `16·K` uint16 words
  (`pack_trellis_kernel`, `pack.cu`): weight i's code occupies stream bits
  `[i·K, (i+1)·K)`, first code bit at the TOP of the first word.
- **Word endianness (main trellis):** the ring is stored so that a
  **little-endian u32 read of each consecutive u16 PAIR yields the
  contiguous MSB-first stream** (first code bit = bit 31 of the first u32).
  Mechanically (`pack.cu`): the packer accumulates native u16s `sp[j]`
  (MSB-first within each word) and flushes through `SWAP16(x) =
  __byte_perm(x, 0, 0x1032)` on the u32 pair, which EXCHANGES the two
  u16 halves' positions while leaving each word's internal bytes native —
  so on disk the u16 sequence is pairwise-transposed: `[sp1, sp0, sp3,
  sp2, …]`. Verified by deriving the K=8 case through both kernels
  (`pack_trellis_kernel` flush + `unpack_trellis_kernel`'s unswapped u32
  funnel reads agree iff this is the layout): with codes c₀..c₃, the first
  stored u32 (LE) = `(c₀<<24)|(c₁<<16)|(c₂<<8)|c₃`. ⚠ A u16-at-a-time
  reader must account for the pair transposition; a u32 reader gets it
  for free.
- **Trellis state = a 16-bit sliding window over the code ring, and the ring
  is TAIL-BITING:** the decoded value of weight i is
  `codebook(16 stream bits ending at bit (i+1)·K − 1, mod 256·K)` — i.e.
  the last 16 code-bits cyclically, so the initial states come from the END
  of each tile's own ring (confirmed three ways: the dq kernels' ring
  indexing `ptr[i % (bits·8)]` with the `+256·bits` offset; the encoder's
  "single tail-biting rings" wording; and `ngram_codec.py`'s explicit
  header, §10.5).

### 10.5 The n-gram embedding special case (PLE models)

Huge hashed n-gram tables (e.g. 320M × 160) are quantized as **160-weight
tail-biting rings over cb2 only** (`ngram_codec.py`, header verbatim):

> "A packed row is (1 + 10*K) little-endian uint16 words (stored as int16):
> word 0 holds the fp16 row scale's bit pattern, the remaining words hold
> the 160*K-bit ring bitstream where stream bits [i*K, (i+1)*K) are the low
>K bits of position i's code […] where state_i is read as the 16 ring bits
> ending at stream bit (i+1)*K - 1 (mod 160*K)."

⚠ Note TWO layout differences from the main trellis: (a) word order/endianness —
little-endian u16s in stream order here vs the pairwise-transposed u32
stream there (§10.4); (b) **bit order within each word** — LSB-first here
(`ngram_codec.pack_rows`: stream bit m → word bit m), MSB-first there.
Plus per-hash-head bias vectors outside the ring.

### 10.6 Incoherence processing (what dequant must apply)

Reference dequant path (`LinearEXL3.get_weight_tensor`), for
`W_rot = reconstruct(trellis)` in `[in, out]` layout:

```
w = block_hadamard_left (W_rot, n=128, scale 1/√128)   # per-128 block along in
w = w · suh[:, None]                                     # per-input-channel fp16 scale
w = block_hadamard_right (w, n=128, scale 1/√128)       # per-128 block along out
w = w · svh[None, :]                                     # per-output-channel fp16 scale
```

- The Hadamard is **Sylvester's construction** recursively (`util/hadamard.py
  get_hadamard`; Paley fallbacks exist for non-powers-of-2; 128 = pure
  Sylvester), matrix `H/√128`. Both sides use `had_k = had_n = 128`, so
  in/out features must be 128-divisible for EXL3 layers (the fused
  reconstruct kernel asserts it).
- `suh`/`svh` are **per-channel fp16 SCALES, not just ±1 signs**:
  `refit_scales` post-quantization REFITS both in the Hessian metric
  (alternating closed-form: per-output col `c_n = (q_nᵀHw_n)/(q_nᵀHq_n)`,
  per-input row solves `((QQᵀ)∘H) r = rowsum(Q∘(HW))`); the converter's
  `out_scales` (default on) folds output scales into `svh`. The packed
  `su`/`sv` ±1 spellings are the legacy pure-sign era; when present the
  loader unpacks them to ±1 fp16 and any suh/svh alongside wins.
- **Full formula:** `W = diag(suh) · (I_{in/128} ⊗ H₁₂₈/√128) · W_rot ·
  (I_{out/128} ⊗ H₁₂₈/√128) · diag(svh)` (+ optional bias at the linear
  output). The quantizer applies the exact inverse before encoding
  (rotation + seed-derived sign flips), per `unrotate_H`.

### 10.7 Layer-class facts that shape a loader (`doc/convert.md`)

- `lm_head` (output layer): integer K 1..8 (default 6). MTP layers: integer
  K, or 16 = stored unquantized (fp16 tensors, not EXL3). Vision: same rule.
- The DEFAULT codebook is **mul1** — required by exllamav3's optimized
  int8-GEMV / CPU-offload paths; a Rust reader should treat cb2 as the
  common case and cb0/cb1 as compatibility.
- Every dimension of an EXL3 layer is 16-divisible (tiles), and both
  in/out are 128-divisible (Hadamards).

### 10.8 What T3 (CPU reference) must implement, minimal set

1. safetensors metadata group scan (detection, §10.1) + K-from-shape.
2. Trellis unpack: per tile, reconstruct 256 16-bit ring windows from the
   `16·K` byte-swapped words; per weight `v = codebook(window)` (all three
   cbs; cb2 first-class).
3. Half-integer K: alternating ka/ka+1-bit steps (period 16, mask 0xAAAA) —
   the ring-window arithmetic generalizes; verify against a real pack.
4. Sylvester-128 Hadamard both sides + suh/svh channel scales (+ su/sv
   unpack fallback).
5. The n-gram table path (little-endian rows, per-row fp16 scale) — can be
   deferred; standalone shard, only PLE models carry it.

### 10.9 Open items carried to T2 (the seam decision)

- The safetensors loader today is BF16-only with a `dtype: String` per
  tensor (§4 of this issue). EXL3 needs: grouped-tensor reads keyed by the
  metadata (several entries per linear), K/marker derivation, and a
  128-divisibility assert — none of which fit `GgmlType` (GGUF's enum), but
  the dequantized output (fp16 `[in, out]` → engine wants `[out, in]`?) is
  the same shape every other quant produces post-dequant. Evidence for the
  verdict: the GGUF seam stays untouched; the EXL3 reader is a SAFETENSORS-
  side grouped reader; convergence to a shared typed enum remains the open
  design question (T2's actual decision).

## 11. T2 record — the loader-seam decision (2026-09-24, verdict-adjudicated)

**Decision: Option A.** An EXL3 grouped-tensor reader on the safetensors
path + `src/quant/exl3.rs` as the first safetensors-side quant module,
holding a typed `Exl3Layer` (trellis + channel scales + markers + K +
codebook tag). `GgmlType` is untouched. Adjudicated by the Claude verdict
ping-pong (round 1 REVISE — rationale rewritten + four conditions added;
round 2 AGREE; every line citation in the round-1 verdict verified against
the tree before recording).

### 11.1 Why NOT `GgmlType` — the load-bearing rationale

Not a fit argument but a **namespace argument**: `GgmlType` is not a typed
quant enum, it is a **wire-format ID parser** — its discriminants ARE
ggml's on-disk numeric type IDs (`F32 = 0`, `BF16 = 30`, `Q2_0 = 42`,
`PTQ1_0 = 143`), consumed by `from_id(id: u32)` straight off the GGUF
header. EXL3 has **no ggml type ID** — an arm would mint a discriminant
inside a numeric namespace upstream ggml allocates. This repo carries the
burn precedent IN THE ENUM ITSELF: `gguf_loader.rs` documents BF16 having
been mapped to 29 (= upstream's `IQ1_M`), which "meant no real BF16 GGUF
could be opened at all", plus a `42 | 142` fork-renumbering arm. Arity
confirms independently: the enum's whole method surface is
`block_info() -> (block_bytes, weights_per_block)` + `tensor_bytes(n)` — a
single-tensor byte calculus an EXL3 arm cannot answer (K comes from
`trellis.shape[-1]/16`; one layer's data spans 3–7 entries).

### 11.2 Enum convergence — deferred, with ONE valid trigger

The future convergence shape is **wrap, never flatten**: a top-level
`QuantFormat::{Gguf(GgmlType), Exl3(Exl3Codebook), …}` — flattening would
destroy the wire-ID property `from_id` and the `42|142` arm depend on.

⚠ **riir-ai Research 360's Universal Quant Router is NOT a trigger for
loader convergence here** — it adjudicated a KERNEL-layer dispatch over
*dequantized* formats in riir-ai; it can ship and consume dense fp16
without either loader in this repo changing a line. Importing it as a
trigger would carry an adjudication across a boundary it does not govern.

**The single valid trigger: a second safetensors-borne quantized format
lands in this repo** — at which point `safetensors_loader.rs` has two
group-scan detectors racing over one metadata map and needs a
discriminant. That is the only condition that actually forces the question.

**No false binary:** the blessed B29 `enum-dispatch-separate-kernel-per-
variant` shape is not being deferred — it already lives at the level where
variance actually exists: `Exl3Codebook::{Cb0, Cb1Mcg, Cb2Mul1}` over the
three procedural codebooks (§10.3) IS that shape. Declining to put it at a
level with one arm is not deferring the pattern.

### 11.3 The four seam conditions T3 must honor (verdict round 1, binding)

1. **Lazy by construction — do NOT inherit the loader's eager-dequant
   contract.** `safetensors_loader.rs` today mmaps and immediately
   dequantizes BF16 → owned `f32` (`load_tensors_from_shard →
   BTreeMap<String, Vec<f32>>`). §2 says EXL3's defensible axis is
   **residency + context ceiling** — an `Exl3Layer` that eagerly expands
   to dense f32 at load is WORSE than BF16 and forfeits the only axis §2
   says is real. The struct holds zero-copy views (`&[u8]` over the mmap /
   borrowed slices) and dequantizes AT USE; T5 cannot measure an axis T2
   foreclosed.
2. **Seam split at metadata-vs-format.** Group scan / detection is
   safetensors-METADATA work (loader side); K-from-shape, marker→codebook
   resolution, and dequant are FORMAT work (`quant/exl3.rs`). `exl3.rs`
   consumes the metadata map (e.g. `&BTreeMap<String, TensorMeta>`)
   rather than EXL3 knowledge growing inside the loader — which requires
   making `TensorMeta` (currently private, `dtype: String`) visible to
   the quant module as part of T3's first commit.
3. **Carry the marker's u32 VALUE, not a `bool`.** §10.2 pins the content
   (`mul1` = `0x83DCD12D`, `mcg` = `0xCBAC1FED`). Selecting a codebook on
   name-presence alone decodes an unexpected pack with the wrong codebook
   — silently numerically wrong, no panic; T4's oracle would be the only
   catcher. `Exl3Layer` preserves the word so T3 verifies it loudly
   (mismatch ⇒ refuse the pack).
4. **`quant/mod.rs` doc line updated** when `exl3.rs` lands — "k-quant
   formats from the GGML ecosystem" becomes false the moment the first
   safetensors-side quant joins.

No trait boundary yet — a trait over one implementor is the same one-arm
mistake as the enum (verdict, non-blocking note).

**Two T3 implementation notes from the round-2 AGREE (non-blocking,
recorded so they don't live only in the verdict thread):**

- ⚠ **The mmap is a function local** — `load_tensors_from_shard` binds
  `Mmap::map` inside the function and drops it at return, so nothing can
  borrow from it across the call boundary. A zero-copy `Exl3Layer<'a>`
  needs the `Mmap` hoisted to an owner that outlives the layers
  (loader-owned map handing out borrows, or `Arc<Mmap>` on the layer).
  The realistic failure mode for condition 1 is not disagreement — it is
  a borrow-checker fight in hour one resolved by "just copy it", which
  silently reinstates the eager contract §11.3 refuses.
- ⚠ **Shard spans:** the loader is per-shard, one EXL3 linear is 3–7
  entries, and §10.5 puts the n-gram table in a standalone shard. Whether
  a group's entries are shard-local is a question a REAL pack answers —
  check in T3 rather than assuming (if groups span shards, more than one
  mapping stays alive per layer → `Arc<Mmap>`).

Process note carried forward (round 2): a verdict round's outcome is
written in the commit that FOLLOWS the verdict, never in the file that
requests it — this §11 record was committed after round 2 returned AGREE,
which is the ordering to keep.

## 12. T3 record — the CPU reference dequantization (2026-09-24)

**Landed:** `src/quant/exl3.rs` behind the opt-in `exl3` feature
(`Cargo.toml` `exl3 = []`; promotion is T6's call). Supporting edits:
`TensorMeta` → `pub(crate)` with its `dtype` field now actually READ
(§11.3 cond. 2 — `quant/mod.rs`'s "GGML ecosystem" doc line updated per
cond. 4).

### 12.1 One ADDENDUM to §10 discovered during T3 (load-bearing for T4)

⚠ **The trellis ring is stored in TENSOR-CORE ELEMENT ORDER, not
row-major.** The quantizer pre-permutes tiles into the CUDA encoder's
layout and — its own comment — "[undoes] permutation on reconstructed
tiles, but **keep[s] indices in tensor core layout**"
(`exl3_lib/quantize.py`, the `tensor_core_perm` build). `frac.cu`'s
header confirms and defines it: for ring position `p = t·8 + j`, the tile
element is `in_off = (t%4)·2 + (j&1) + 8·((j>>1)&1)`, `out_off =
(t>>2) + 8·((j>>2)&1)` (`frac_perm`). A decoder that walks the ring in
row-major order reconstructs a PERMUTED matrix — shapes and stats look
plausible, every weight is wrong. §10.4 now implies this via the frac.cu
quote; this section states it explicitly because T4's oracle comparison
is where a silent permutation bug would otherwise surface as "EXL3
quality is terrible" rather than as a layout error.

### 12.2 What shipped

- `Exl3Codebook::{Cb0, Cb1Mcg, Cb2Mul1}` — the three procedural codebooks
  in exact f16 semantics (integer wrapping ops + one fp16 add). The B29
  enum-dispatch shape lives HERE (§11.2's "no false binary").
  `from_markers` VERIFIES the raw u32 words and refuses mismatches loudly
  (§11.3 cond. 3) — never a bool.
- `Exl3K` — integer + half-integer bitrate, derived from
  `trellis.shape[-1]/16` (words%16: 0 → int, 8 → half); `bits_for_step`
  implements `frac_d` (`ka + bit(i mod 16) of 0xAAAA` — even steps short,
  odd steps long, period 16).
- Ring reader: LE-u32-read → MSB-first bitstream; 16-bit windows ending
  at `S(p+1)`, cyclic (tail-biting); placement through
  `ring_pos_to_tile_element` (§12.1).
- `sylvester_hadamard_128()` — cached `H/√128`, built by 7 doublings from
  `[[1]]` (verified against the pin's `hadamard_1.txt` base + Sylvester
  recursion; orthogonality/symmetry asserted in tests).
- `Exl3Layer<'a>` — ZERO-COPY byte views (§11.3 cond. 1), validates dims
  (128-divisible), trellis length, K, markers, half-K⇒mul1;
  `dequantize_f32()` = trellis decode → left block-Hadamard → row scales
  → right block-Hadamard → column scales, scalar O(in·out·128).
- `detect_exl3_layers(&BTreeMap<String, TensorMeta>, &markers)` — the
  loader-side group scan (dtype-validated: trellis I16, markers I32);
  `#[allow(dead_code)]` until T4/T5 wire the loader.

### 12.3 Numerics caveats T4 must adjudicate

1. **cb2's `__hfma` emulation**: computed as `f32(h)·f32(inv) + f32(bias)`
   rounded once to f16 (product exact in f32; two roundings total vs a
   true fused op's one). cb0/cb1 are bit-portable (single fp16 add). If
   T4's per-tensor numeric error shows a systematic cb2 offset, this
   emulation is the first suspect — the fix is a software hfma or f64
   accumulation for the final rounding.
2. **The Hadamard convention** (Sylvester from `[[+]]`) is inferred from
   the pin's data files + recursion; a real pack pins it absolutely.
3. **Half-integer step direction** (`0xAAAA`: even steps `ka`, odd `ka+1`)
   follows `frac_d` verbatim; a real half-K pack confirms the tile-word
   parity derivation end-to-end.

### 12.4 Validation at landing

`cargo test -p riir-infer-core --lib` = 186 green (baseline unchanged);
`--features exl3 --lib` = 197 green (11 new, non-vacuous by name:
  codebook_pins_match_independent_numpy, codebooks_finite_and_bounded,
  ring_roundtrip_integer_k, ring_wraps_tail_biting, half_k_bits_and_words,
  half_k_roundtrip, ring_perm_is_bijective_and_matches_spec,
  sylvester_hadamard_128_properties, sign_unpack_matches_bit_order,
  layer_validates_dims_markers_and_k, dequant_matches_reference_composition).
clippy `--all-targets -D warnings` clean at BOTH postures. The known-answer
pins were computed independently (numpy f16 via `uv run --with numpy`, from
the §10.3 spec ops — `.raw/exl3_pins.py`, kept with the clone).

### 12.5 T4a record — real-pack validation (2026-09-24, 4090 box)

**Oracle fixture:** `async0x42/Qwen3-8B-exl3_4.0bpw` (the smallest dense
EXL3 pack on HF; Qwen3-8B, single `model.safetensors`, `quantization_config
= {quant_method: exl3, bits: 4}`). Fetched SURGICALLY — header (100,264 B)
+ one layer's byte ranges via HTTP Range, no 4 GB download:
`model.layers.0.self_attn.k_proj` (in=4096, out=1024, K=4, **cb0** — this
2025-05 pack predates markers; 253 groups, all suh+svh fp16 form, no
su/sv, no markers; lm_head K=6/96-words, 252 layers K=4/64-words).

**Result:** the Rust `Exl3Layer::dequantize_f32` vs the independent numpy
oracle (`.raw/exl3_layer_oracle.py`, written from §10, no Rust involvement)
over all 4.19M weights: **relative Frobenius error 3.486e-13, max abs err
2.68e-7 on max|W| = 0.954** — machine precision; every layout fact in §10
is CORRECT on real data (bit order, u16-pair transposition, tail-biting
ring, tensor-core element permutation, cb0 codebook, Sylvester-128
Hadamard, scale composition). Final-W stats sane for LLM weights
(mean −1.3e-5, std 0.053). Gate: `cargo test --features exl3 --lib
real_pack_oracle_k_proj -- --ignored` (env-gated on the fixture at
`%TEMP%/exl3-pack/`; skips loudly when absent).

**Modern-pack marker validation:** `Terra3312/GLM-5.3-Flash-EXL3-4bpw-MUL1`
shard-1 header: `.mul1` on **all 887 groups**, `I32` **0-dim scalar**
(§10.2 corrected: shape `[]`, not `[1]`), and the range-fetched word is
**exactly `0x83DCD12D`** ✓. All 887 groups carry trellis+suh+svh+mul1
within the SAME shard — groups are shard-local in this pack (the §11.3
round-2 question, answered for one pack; the loader should still not
ASSUME it — the index.json decides per pack).

**What T4a does NOT prove** (honest scope): both implementations were
written from the same §10 spec, so a SPEC misreading shared by both would
pass. The exllamav3-native oracle (T4b) is the remaining independent
authority — its env cost is recorded above.

### 12.6 T4b record — the exllamav3-native oracle (2026-09-24, 4090 box)

**Env built same-session** (kept at `.raw/exl3-venv`, ~3 GB, gitignored):
uv venv (py3.12) + torch 2.14.0+cu130 + the clone JIT-built under MSVC
14.41 (BuildTools 2022) + nvcc 13.3, arch sm_89. Two build findings,
both recorded for the next Windows build:

1. **MSVC C1060 (heap exhaustion) on `bindings.cpp`** — root cause: the
   HUGE inherited MSYS environment block; cl.exe fails even single-threaded
   with it, compiles clean in a minimal env. Workaround: shrink `os.environ`
   BEFORE `import exllamav3` (`.raw/build_ext_clean_env.py` keeps the
   recipe) — INCLUDE/LIB must be set explicitly (torch's msvc detection
   needs more than a bare PATH).
2. **Clone-local BUILD-FLAGS patch** (`.raw/exllamav3/exllamav3/ext.py`):
   `/Zc:preprocessor` dropped, `/bigobj` added — build flags only, zero
   algorithm/kernel source touched; the oracle's math is unmodified.

**Oracle specimen (pin-era):** `Terra3312/GLM-5.3-Flash-EXL3-4bpw-MUL1`
shard 1, `model.language_model.layers.0.self_attn.o_proj`
(in=8192, out=4096, K=4, **cb2/mul1**, suh+svh) — range-fetched
surgically (16.8 MB of a 22-shard pack).

**RESULTS (all four gates green):**

| gate | result |
|---|---|
| state words vs their `unpack_trellis` (tile 0,0) | **256/256 exact** |
| tile(0,0) cb2 values + my tensor-core placement vs their `reconstruct` | **256/256 exact** |
| FULL w_rot (8192×4096 = 33.5M weights) vs their `reconstruct` | **BIT-EXACT (max_abs = 0.0)**, rel-Frobenius −7.4e-08 |
| FULL W (Hadamards + suh/svh) vs their `get_weight_tensor` | rel-Frobenius **2.96e-08**, max_abs 1.8e-4 on max\|W\|=0.284 (their fp16 intermediates vs f32 — the expected class) |

**The §10 spec — ring layout, u16-pair transposition, tail-biting windows,
tensor-core element order, cb2 `__hfma` emulation, Sylvester-128 Hadamards,
channel scales — is CONFIRMED BIT-EXACT against the reference
implementation on pin-era data.** T4 is COMPLETE (T4a cross-implementation
+ T4b native oracle). Chain of evidence: Rust ≈ numpy at 3.5e-13 (T4a);
numpy = native bit-exact (T4b) — Rust ↔ native transitively confirmed.

### 12.7 ⚠ The pack-era finding (a T4b by-product — READ before picking an oracle pack)

The FIRST oracle specimen (`async0x42/Qwen3-8B-exl3_4.0bpw`, created
**2025-05-10**) FAILED the native comparison: rel-Frobenius 0.184, with my
cb0-of-verified-words values ABSENT (250/256) from their `reconstruct`
output — while their own `unpack_trellis` on the same tile agrees with my
decode 256/256. Diagnosis path (all measured, `.raw/exl3_t4b_diag*.py`):
no bit-alignment, bit-order, codebook, or window-shift variant of the tile
bytes reproduces their values — meaning current `reconstruct` reads that
old pack differently than current `unpack_trellis` does. Git history:
the pack was made one day after commit `456f14f` ("Fix regression",
2025-05-09); the format's layout machinery evolved after (tensor-core
permutation, codebook defaults, packing) with no compat guarantee —
exllamav3 carries no format version marker, so OLD PACKS SILENTLY DECODE
WRONG (or at least differently) under current code.

> **Correction 2026-09-25 (T7c-4 landing): the sentence above — "exllamav3
> carries no format version marker" — is FALSE as written, and the era
> gate landed on the truth.** The TENSOR format self-describes nothing,
> but the pack CONFIG self-describes plenty: the writer has stamped
> `quantization_config.version` (its own `__version__`) into the pack's
> `config.json` since versioning was established (`conversion/compile.py`
> at the pin writes `"version": __version__`), and the real 27B pack
> carries `"version": "1.4.2"`. The legacy-era specimen predates the
> stamp (exllamav3 `version.py` read `"0.0.1"` at 2025-05-11) — exactly
> the absent-version class. The shipped detector candidate named below
> (unpack-vs-reconstruct probe) was therefore never built: it needs the
> exllamav3 Python env at open, and the config-key gate does not. Wired as
> `verify_pack_era` at `Exl3Pack::open` — fail-closed, escape
> `open_unverified_era`; see §17.5 T7c-4.

**Consequences:**
1. For any future oracle or production read: use pin-era packs only; the
   pack creation date vs the pin is a load-bearing compatibility axis the
   format does not self-describe. (A candidate detector: compare
   `unpack_trellis`-consistent words against a small `reconstruct` probe —
   divergence ⇒ legacy pack, refuse loudly.)
2. T4a's numpy-vs-Rust agreement on that old pack remains valid (both
   implementations of the CURRENT spec, self-consistent) — but the T4a
   fixture is a LEGACY pack; the modern specimen is the authoritative one.
3. Filed as an upstream curiosity only — turboderp's format, his call;
   our loader refuses-by-marker already covers the marker dimension, and
   the era dimension is ours to gate when T5/T6 wire a real loader.

## 13. T5 record — §2 re-measured on our silicon (2026-09-24, 4090 box)

**Method:** every residency byte below is measured from REAL pack bytes
(safetensors headers of actual shards — not arithmetic from config
geometry); the context ceiling is an arithmetic projection from those
measured bytes with stated assumptions (§13.3), NOT a served measurement —
no engine integration exists yet (T7's arms are that integration). Box
state (the G2 box-state rule): i7-13700K + RTX 4090 24 GiB, CPU ~8% load,
GPU idle, AC power.

**Specimen:** `turboderp/Qwen3.8-27B-exl3` @ branch `SC_4.00bpw_H5_V6`
commit `516bf129059031c6da9416768ea6b7a1be00a8fc` — the format AUTHOR's
own pack of the LEAGUE MODEL (Qwen3.8-27B: 64 layers hidden 5120, vocab
248320; a HYBRID model — 48/64 layers `linear_attention`, only 16
full-attention every 4th, 4 KV heads × dim 256; MTP 1 layer; SigLIP-ish
vision tower). `quantization_config`: exl3 **version 1.4.2**, bits 4.0,
head_bits 5, mtp_bits 4, vision_bits 6, codebook mul1 — modern defaults
throughout; branch lastModified 2026-08-27, pin 2026-09-20 → **pin-era**
(§12.7's gate). Full download 16,349,968,660 B; shard-1 sha256 verified
against the HF LFS hash (`89403623…e1b79044`). On disk at
`.raw/packs/qwen38-27b-exl3-4bpw` (gitignored, kept for T6/T7).

### 13.1 The loader (landed `bd93a7b`)

`src/quant/exl3_pack.rs` (feature `exl3`): `Exl3Pack::open` (index.json |
single-file | `*.safetensors` dir) mmaps all shards, merges the
name→(shard, meta) table, reads marker words off the maps, detects groups;
`layer(key)` returns zero-copy `Exl3Layer` borrows (§11.3 cond. 1).
**Groups span shards in this real pack**: `lm_head.trellis` is in shard 2
while `lm_head.suh/svh/mul1` are in shard 1 — the §11.3 round-2 open
question, answered by measurement; the merged-table resolution handles it.
`residency()` is the byte-accounting instrument; 5 fixture tests
(synthetic 2-shard pack with a spanning group + legacy su/sv group,
single-file, garbage-marker refusal, duplicate-tensor refusal) + the
measurement test. Clippy `-D warnings` clean at BOTH feature postures.

### 13.2 Residency — measured bytes (the §2 axis that is real)

The 4bpw row is exact on-disk measurement of the downloaded pack
(`Exl3Pack::residency`); the 2.0/3.0/5.0/6.0 rows are metadata-only over
HTTP-Range header fetches of the same repo's other pack branches
(`.raw/exl3_bpw_sweep.py`; byte counts are shape-determined, so exact for
any pack era). dequant-f16 = (quantized + dense params) × 2 B.

| pack | total GiB | achieved bpw | per-layer K | quantized GiB | dense GiB | dequant-f16 GiB |
|---|---:|---:|---|---:|---:|---:|
| 2.00bpw | 10.039 | 2.232 | 2.0–6.0 | 6.763 | 3.275 | 51.75 |
| 3.00bpw | 12.871 | 3.167 | 3.0–6.0 | 9.595 | 3.275 | 51.75 |
| **SC_4.00_H5_V6** | **15.227** | **4.087** | **3.0–6.0** | **12.601** | **2.626** | **51.95** |
| 5.00bpw | 18.535 | 5.037 | 4.0–6.0 | 15.259 | 3.275 | 51.75 |
| 6.00bpw | 21.367 | 5.972 | 4.0–6.0 | 18.091 | 3.275 | 51.75 |
| (f16 baseline) | — | 16.0 | — | — | — | 51.95 |

Model total ≈ 27.8–28.2 G params (26.02–26.48 G quantized + 1.41–1.76 G
dense; the SC recipe quantizes MORE tensors than the plain branches —
dense 2.63 vs 3.28 GiB). The 4bpw pack's measured interior: trellis
12.586 GiB + scales 14.8 MiB + markers 2.2 KiB + group bias 0.5 MiB;
573 groups over 3080 tensors. **The f16 model (51.95 GiB) does not fit a
24 GiB 4090 at all; every EXL3 variant does.** Same-model GGUF q4-class
would land ~16 GiB (not measured here — no GGUF of this model on box);
Bonsai-27B PQ2_0 (7.17 GB) is a DIFFERENT model and quant family — size
context only, never joined (§7).

### 13.3 Context ceiling on the 4090 (24 GiB) — arithmetic projection

KV/token (f16): 16 full-attn layers × 2 × 4 KV-heads × 256 × 2 B =
**64 KiB/token** — the hybrid architecture's advantage (48 linear-attn
layers carry a FIXED ~151 MB state (f32, `mamba_ssm_dtype`), not
per-token KV). Assumptions (stated, not measured): 1.5 GiB engine
overhead (activations/CUDA graphs) + 0.15 GiB linear-attn state; ceiling
= (24 − pack − 1.65) GiB ÷ 64 KiB. Model cap `max_position_embeddings` =
262,144.

| pack | weights GiB | context ceiling (tok) |
|---|---:|---:|
| 2.00bpw | 10.04 | ~201,600 |
| 3.00bpw | 12.87 | ~155,300 |
| SC_4.00 | 15.23 | ~116,700 |
| 5.00bpw | 18.54 | ~62,500 |
| 6.00bpw | 21.37 | ~16,100 |
| f16 | 51.95 | 0 (does not fit) |

**§2's residency/context direction CONFIRMED on our silicon** (2.0 vs
4.0 bpw: −5.2 GiB resident → +73% context ceiling, 201k vs 117k tokens);
the source's magnitudes (its +22 GiB / halved ceiling on 128 GB unified)
are ITS box's, not ours. q8_0 KV would ~double every ceiling row — an
engine knob, not a format fact.

### 13.4 CPU reference dequant throughput (the §2 mechanism axis, baseline)

Release, single thread, i7-13700K at ~8% box load: **6.1–6.5 M
weights/s** across all 5 sampled classes (o_proj 31.5M w in 5.01 s; mlp
89.1M w in 14.5 s; lm_head 1.27B w in 197 s; scalar O(in·out·128)).
Extrapolated full-model single-thread dequant ≈ **70 min** — the
reference posture denominator T7's SIMD/GPU arms replace. Zero NaN, sane
stats on every sampled layer (mean ~0, max|W| 0.34–1.35).

### 13.5 Era-gate on the full pack (the §12.7 detector, applied)

`.raw/exl3_t5_oracle.py` (T4b's env + machinery) vs the Rust exports on
THE SAME downloaded pack, 5 classes:

| layer | K | rel-Frobenius | max_abs |
|---|---|---:|---:|
| linear_attn.out_proj | 5 | 1.26e-07 | 1.0e-03 |
| lm_head (1.27B w) | 5 | **0.0** | 2.3e-04 |
| mlp.down_proj | 3 | 9.7e-08 | 5.4e-04 |
| mlp.gate_proj | 3 | 1.1e-07 | 1.6e-04 |
| self_attn.o_proj | 4 | **0.0** | 4.8e-04 |

The fp16-intermediate class T4b documented (their Hadamard+scales in
fp16 vs our f32); two whole layers agree EXACTLY. **Pin-era confirmed;
the loader decodes this pack correctly, and the mixed-K recipe (mlp K=3,
attn 4–5, lm_head K=5) reads end-to-end.**

### 13.6 What T5 licenses — and what it does NOT

**Licensed** (for plans/README/league rows, our box, this model class):
- Residency: exact measured bytes at 5 bpw points (§13.2).
- Context ceiling: the §13.3 projection with its stated assumptions.
- The §2 residency/context DIRECTION on our silicon (lower bpw ⇒ smaller
  resident ⇒ higher ceiling), n=5-branches + arithmetic.

**NOT licensed:** any decode/prefill throughput or tok/s claim (no
serving path until T7 — and §2's own data had decode inside sample
spread); any quality claim (T6's GOAT gate, cf. turboderp's KLD tables
on the same repo — not read by us, not citable as ours); any
MTP/acceptance claim (closed negative, §7); the `utilisation forced`
row (a serving-config artifact of the source's runtime, not a format
property). GGUF-vs-EXL3 same-model comparisons stay OPEN until a
qwen3.8-27B GGUF lands on a box (then: same model, same box, both
formats, our loader — the honest league shape).

**M3 note:** the byte table is silicon-independent; the ceiling table
recomputes per memory budget (the formula + constants are in §13.3).
Re-measuring on the M3 means running the same measurement test there —
not done this session (pack is on the 4090 box; a re-download or copy
is the cost). The 4090 numbers stand on their own.

## 14. T6 record — the GOAT gate verdict (2026-09-24)

The workspace promotion rule: opt-in feature → bench the claim → promote to
default only on a measured, modelless gain on the §2-real axis.

| gate | verdict | evidence |
|---|---|---|
| G1 correctness | **PASS** | T3 numpy pins + ring/Hadamard/bijection tests (§12.4); T4a real-pack cross-impl 3.49e-13 (§12.5); T4b native-oracle BIT-EXACT (§12.6); T5 full-pack 5-class oracle ≤1.3e-07 incl. the 1.27B-weight lm_head (§13.5). |
| G2 gain (residency — the §2-real axis) | **PASS, measured** | [Bench 001](../.benchmarks/001_exl3_t5_residency_context.md): 4.09 achieved bpw vs 16 f16 → 15.23 vs 51.95 GiB (3.4×) on the league model; 5 bpw points measured; context ceiling 0 → ~117k tokens at 4bpw on a 24 GiB box. The gain is arithmetic on the format (modelless — no training, no calibration data of ours). |
| G3 no-regression | **PASS** | Pure local feature gate (`exl3 = []`, no dep forwards); default build byte-identical in surface (clippy `-D warnings` clean at BOTH postures; 186/201 tests green at flag-off/on). |
| G4 allocation | **PASS** | Zero-copy by construction (§11.3 cond. 1): mmap'd shards, borrowed slices, `residency()` reads metadata only; `dequantize_f32` allocates only its documented output (at-use materialization). |

**Verdict: STAYS OPT-IN. Default-promotion REFUSED at this gate — not on
gate failure but on the promotion rule's own wording: the gain must be on
the path promotion turns on.** Today no default build path (and no engine
integration) consumes EXL3; promoting now would compile an unconsumed
reader into every build — surface with zero delivered gain. The reader is
also era-gated by necessity (§12.7): a default-on surface that happily
opens LEGACY packs (which decode wrong) is a hazard an opt-in surface
keeps explicit.

> **Update 2026-09-25 (T7c-4):** the era hazard named in the paragraph
> above is now a REAL refusal at `Exl3Pack::open` (`verify_pack_era`,
> fail-closed on `quantization_config.version`, `open_unverified_era`
> escape) — no longer a doc comment the caller must remember. The
> promotion trigger below stays INTACT and un-discharged: the fused-GEMV
> arm that would have measured the delivered gain is CLOSED at this tier
> (§17.6); on any reopen trigger r1-r4, this gate re-runs with runtime
> numbers as written.

**Promotion trigger (named, one condition):** T7 lands a serving/GPU arm
that consumes `Exl3Pack` end-to-end AND measures the delivered gain on
that path (bytes moved per decode step, or context headroom in a real
serving config, vs the f16/q4 GGUF incumbent on the same model + box) —
at which point promotion re-runs this gate with the runtime numbers and
the §12.7 era-gate wired into the loader's open path (refuse legacy
packs loudly at open, not decode).

## 15. T7a record — the CPU fast arm (2026-09-24, 4090 box, `725a8a5`)

**What landed:** `codebook_lut()` (65536-entry f32 tables, memoized from
`decode_f16` — the codebook is a pure function of the code, so the table IS
the same math precomputed) + `Exl3Layer::dequantize_f32_parallel()` (rayon
over DISJOINT row strips per stage; every output element's accumulation
order is unchanged → **bit-identical** to the scalar reference, pinned by
`parallel_matches_scalar_bit_identical` across integer-K, half-K (mul1),
and legacy su/sv spellings, compared via `to_bits` — NaN-payload aware).

**Measured** (release, 16-thread i7-13700K, the 27B pack, real layers,
parity asserted per class):

| class | weights | scalar | parallel | speedup |
|---|---:|---:|---:|---:|
| self_attn.o_proj (K4) | 31.5M | 5.12 s / 6.1 Mw/s | 0.479 s / 65.6 Mw/s | 10.7× |
| mlp.down_proj (K3) | 89.1M | 14.71 s / 6.1 | 1.336 s / 66.7 | 11.0× |
| linear_attn.out_proj (K5) | 31.5M | 5.26 s / 6.0 | 0.462 s / 68.1 | 11.4× |
| mlp.gate_proj (K3) | 89.1M | 13.78 s / 6.5 | 1.299 s / 68.6 | 10.6× |
| lm_head (K5) | 1.27B | 197 s | 19.42 s / 65.5 | 9.9× |

Full-model parallel extrapolation ≈ **6.7 min** (vs ~70.7 min scalar).
Box: CPU ~8% load, GPU idle, AC.

**Fixture lesson (worth carrying):** the parity test's first draft fed RAW
random bytes as the f16 scales — random 16-bit patterns include NaN/Inf,
which put NaN in BOTH outputs; `NaN != NaN` failed `assert_eq` while the
parity was never wrong. Float-parity tests compare `to_bits`, never `==`.

**T7b (GPU arm) remains** — the 4090 dequant kernel (CubeCL path lives in
`crates/riir-infer-gpu`; exllamav3's own CUDA kernels are the reference
shape) + engine integration; it is also the §14 promotion trigger. The
~11× CPU arm is the correctness oracle for that kernel: bit-parity
against the same scalar reference.

## 16.5 The T7b module-orphan repair (2026-09-24, M3 session)

The `3ce2d75` rustfmt sweep accidentally dropped BOTH the
`pub mod exl3_dequant_cubecl;` declaration (`lib.rs`) and the `exl3_gpu`
feature row (`Cargo.toml`) — the landed module was ORPHANED at HEAD for
~1h10m (`af881ac` → this repair): present on disk, compiled to nothing,
`--list` showed 0 of its tests, the feature unresolvable. An M3 session
discovered it while landing a parallel T7b attempt (closed TWIN per the
owner call; the duplicate died in the working tree, its independent
readings — Metal-lane decode bit-parity + the wgpu-hal
`fast_math_enabled` FMA-contraction mechanism — both AGREE with this
record). **Repair:** declaration + feature row restored verbatim from
`af881ac`; validated on M3 Metal/wgpu-msl — `exl3_dequant_cubecl::tests`
7 passed / 3 ignored (CUDA-only rows) — the orphaned tests run again.
Standing guard for the class: a sweep commit touching `lib.rs` or
`Cargo.toml` must name every deletion against HEAD before landing; and
"module file exists" is not "module compiles" — the check is
`cargo test --features exl3_gpu --lib exl3_dequant_cubecl:: --list`.

## 16. T7b record — the 4090 GPU arm (2026-09-24, 4090 box)

**What landed** (`crates/riir-infer-gpu/src/exl3_dequant_cubecl.rs`, feature
`exl3_gpu = ["cubecl_runtime", "riir-infer-core/exl3"]`): three CubeCL
kernels — `exl3_trellis_decode` (one thread per (tile, ring position),
closed-form prefix sum, MSB-first window with ring wrap, tensor-core-order
scatter), `exl3_left_hadamard` (+ fused row scale), `exl3_right_hadamard`
(+ fused column scale) — grid-strided (≤65535 workgroups: wgpu's
per-dimension dispatch cap; real layers need up to ~4.9M tile-threads),
driven by `Exl3DequantCubeCL::dequant_layer_f32` over 128-column chunks
(VRAM-bounded intermediates, layer-size-independent; lm_head's 5 GB f32
output never materializes on the GPU at once). Core additions:
`Exl3Layer::{trellis_bytes, suh_f32, svh_f32}` (the upload surface; the two
dequant fns now share the scale-expansion helpers — the DRY fold).

**The two-tier oracle** (the §15 aspiration was bit-parity; measured
reality split it):

1. **Decode stage — BIT-EXACT.** Zero floating-point arithmetic; the CPU's
   own LUT bytes uploaded verbatim. Pinned synthetically (K3/cb0, K5/mcg,
   K4.5/mul1) and on the real pack (147456/147456 probe elements on
   `model.visual.blocks.0.attn.k_proj`).
2. **Hadamard stages — FMA-contraction class.** Both CubeCL backends
   contract `a*b+c` (the CUDA lane generates CUDA C++ plain operators under
   NVRTC's default `-fmad=true`; wgpu contracts identically — same
   divergence bits on both backends); there is no per-kernel opt-out, and a
   global `-fmad=false` would de-optimize every FMA-contracted GEMV in the
   workspace (rejected deliberately). FMA is one-directionally MORE
   accurate — the same class T4b adjudicated for the reference
   implementation's own fp16 Hadamard intermediates (§12.6). Gates
   (scale-aware; ulp counts are meaningless across sign flips of near-zero
   sums): max |diff|/max|W| ≤ 1e-5, rel-Frobenius ≤ 1e-5. Measured:
   **rel-Fro 2.59-2.61e-7 on every class** (the FMA signature — uniform,
   backend-stable), 20-100× under the gates.

**Measured** (the LEAGUE MODEL author pack, release, native CUDA backend;
full tables + the wgpu lane in [Bench 002](../.benchmarks/002_exl3_t7b_gpu_dequant.md)):

| class | weights | GPU wall | kernel-only | CPU arm (T7a) | wall speedup |
|---|---:|---:|---:|---:|---:|
| self_attn.o_proj (K4) | 31.5M | 743 Mw/s | 4749 Mw/s | 66.5 Mw/s | 71.4× |
| mlp.down_proj (K3) | 89.1M | 538 Mw/s | 4983 Mw/s | 69.0 Mw/s | 72.2× |
| linear_attn.out_proj (K5) | 31.5M | 701 Mw/s | 4955 Mw/s | 68.5 Mw/s | 72.4× |
| mlp.gate_proj (K3) | 89.1M | 534 Mw/s | 5051 Mw/s | 67.6 Mw/s | 74.7× |
| lm_head (K5) | 1.27B | 655 Mw/s | **5848 Mw/s** | 67.4 Mw/s | **86.8×** |

Full-model extrapolation: **~5.5-6.7 s** pure GPU dequant (kernel-only),
~50-60 s end-to-end with readback, vs the CPU arm's ~6.7 min. Load-time
dequant is now readback-bound, not compute-bound.

**Fixture lessons paid for** (carried forward): (1) a column-chunk
readback is `[in × cols]` row-major — scatter per-row into the output,
never copy contiguously (the first draft produced exactly one correct row
out of 256); (2) parity harnesses must COUNT and LOCALIZE misses — the
first-bit-mismatch diagnostic hid a ~100% element miss; (3) the wgpu
dispatch cap (65535 workgroups/dimension) bites at real layer sizes —
grid-stride every kernel that maps thread-per-element.

**What T7b does NOT do** (the honest scope line, **AMENDED 2026-09-24 by §17**): ~~the
§14 promotion trigger requires a SERVING arm that consumes `Exl3Pack` end-to-end
(weights GPU-resident, converted to the engine's dense f16 GEMV path)~~ — the
two-stage "dequant → dense f16 GEMV" framing is SUPERSEDED for the trigger
discharge by §17's fused-GEMV lane (the structural byte-traffic argument:
any dequant-then-GEMV path moves ≥ pack bytes + 2× dense bytes per decode
step — strictly more than fused — on the exact metric §14 names, at every
model size and under every scheduling trick, including layer-at-a-time
streaming). The delivered-gain measurement (§17.4) + the kernels compose
here; the engine serving integration remains separately open, never
implied-complete. The 16.35 GB pack + the `.raw/exl3-venv` oracle env stay
on disk for that work.

## 17. T7c — the fused trellis GEMV (the §14 trigger discharge lane, verdict-adjudicated 2026-09-24)

**Status:** ACTIVE — filed 2026-09-24 (4090 box) after a 3-round Claude verdict
ping-pong (AGREE; session `c56e3f15`). The §14 promotion trigger discharges at
GEMV level **in this repo**; engine serving integration stays separately open.

### 17.1 The path decision (what the verdict approved)

**Supersession rests SOLELY on the structural byte-traffic argument** (the
verdict's condition 1 — the residency argument was dropped as refutable):
any dequant-then-GEMV path writes the dense weights then re-reads them, so
it moves **≥ pack bytes + 2× dense bytes per decode step — strictly more
than fused — on the exact metric §14 names (bytes moved per decode step),
at every model size and under every scheduling trick** (including
layer-at-a-time streaming dequant into scratch).

**The fused design:** for `y = W·x` with
`W = diag(suh)·(I⊗H)·W_rot·(I⊗H)·diag(svh)`, the Hadamard transforms
collapse onto the VECTORS — `v = H·(svh⊙x)` per 128-block of the input,
`y = suh⊙(H·t)` per 128-block of the output (7n ops per 128-block =
n·log₂128, negligible against the GEMV) — leaving the weight-side work as
a GEMV through `W_rot` with INLINE trellis decode: per weight, one
closed-form decode (integer prefix math + 65536-entry f32 LUT gather,
bit-exact) + one FMA. Reads only pack weight bytes per step (≈13.8 GB for
the 27B league model at 4.09 bpw — the per-step read is WEIGHT bytes;
16.35 GB is the whole pack incl. scales/unquantized tensors). Plane-per-
output-column organization mirroring `gemv_q4k_cubecl.rs` (plane_sum
reduction, no atomics); the 16 columns of a tile share the tile's ring
words → L2 absorbs the ≤16× re-read (the decode kernel's measured
locality precedent).

**In-boundary:** BOUNDARY.md §Owns already names riir-infer-gpu's EXL3
kernels; the incumbent comparison needs only this repo's existing
`gemv_f16_cubecl` + `gemv_q4k_cubecl` — no sibling dep on the critical
path (the verdict's procedural point).

### 17.2 The measured risk + the discriminating bench (T7c-1)

T7b's decode kernel runs at **4.7–5.8 Gw/s** — a naive fusion inherits
~5.5 s per 27B decode step. Mechanism candidate: the inner loop
(`exl3_dequant_cubecl.rs:137-141`) does 16 per-bit `% ring_bits` modulos
**and 16 separate global loads** per weight. The mitigation (v2
extraction): every `ring_bits` value is a multiple of 128 **by
construction** (`stream_bits_per_tile()` = 256·ka or 256·ka+128 — a type
invariant, not an observation), so `window16` = 1–2 u32 loads +
shift/mask + conditional-subtract wrap, zero modulo.

**The bench is DISCRIMINATING, three arms** (verdict condition 2):

| arm | what it isolates |
|---|---|
| A1 | v1 kernel as-is (the 5.8 Gw/s baseline) |
| A2 | v2 extraction (byte-aligned windows, modulo-free) — isolates the modulo+per-bit-load mechanism |
| A3 | v2 + LUT gather replaced by a constant — isolates the surviving `lut[w]` gather (1 load/weight) |

**Expected bound (a DERIVATION, not a measurement — arm A3 converts it):**
the LUT gather survives v2 at 1 load/weight vs v1's ~17 loads/weight
(16 window + 1 gather), so the achievable speedup is capped near **~17×
(≈100 Gw/s)** — the next wall, written here with its inputs so the bench
replaces it rather than inherits it as a measured constant.

**Kill criterion (written BEFORE the bench — verdict condition 3):** if
A2 lands under ~25 Gw/s, that is **a bound on the extraction, NOT a
falsification of fused GEMV** — record it as a bound, leave the lane open
(the Issue-825 lesson: a mechanism inferred from a consistent number is
not a falsification).

### 17.3 Gates (kept separate — verdict condition 4)

1. **v2-vs-v1 decode: BIT-EXACT over the WHOLE real pack** (the T5
   full-pack 5-class oracle shape, never a sample).
2. **Fused GEMV: accumulation-order tolerance** vs the CPU reference
   (`dequantize_f32` then matvec) — the T4b/T7b FMA-contraction class;
   never pooled with gate 1 (a decode regression must not hide inside a
   tolerance).

### 17.4 The delivered-gain measurement (discharges §14's FIRST branch)

Bytes moved per decode step + step time, measured on the same box, through
the SAME layer weights/geometry in three GEMV paths: EXL3 fused (this
lane), the repo's f16 GEMV kernel over f16 weights, the repo's q4k GEMV
kernel over q4k-encoded copies of the same weights. The 117k-token
context ceiling is NOT re-cited as delivered gain — §14's G2 already
banked it as arithmetic on the format (Bench 001); residency appears only
as already-banked context. **Every figure carries its box state** (free
RAM, commit-vs-limit, concurrent jobs — multi-session 4090).

### 17.5 Tasks

- [x] T7c-1a: v2 extraction kernel (integer-only, word-aligned windows,
      conditional-subtract wrap) — bit-exact vs v1 AND vs the CPU reference
      tile decoder pinned by `gpu_decode_v2_bit_exact_vs_v1_and_cpu`
      (K2/K3/K4.5/K5 × Cb0/Cb1Mcg/Cb2Mul1, CUDA, green 2026-09-24) +
      per-layer bit-exact on the real pack (the bench's built-in gate, green).
- [x] T7c-1b: the discriminating bench (A1/A2/A3) on the real pack's layers
      (CUDA lane, release, 4090; box state: ~500 MiB VRAM in use, 20% util
      idle desktop baseline, no concurrent GPU compute, CPU ~5%, 9.3/31.8
      GiB RAM, AC power) — **KILL CRITERION FIRED for the extraction class,
      as pre-registered**: A2 (v2 word-aligned) shows NO reliable win over A1
      (v1) — the ratio flips run-to-run (1.56×/2.78× → 0.44×/0.51× on the
      same layers across consecutive runs) and the best-of-5 interleaved
      rounds cannot separate them. **The modulo mechanism theory is NOT
      confirmed**: v1's 16 L1-cached loads + modulos cost ≈ the same as v2's
      branchy 2-load path. Recorded as a BOUND on the extraction approach,
      not a falsification of fused GEMV (§17.2's own wording). ⚠ The harness
      itself (reps-differential with per-chunk readback) proved too noisy at
      this scale — lm_head (5 chunks) produced impossible artifacts (972
      Gw/s, `inf`) in 2 of 3 runs; single-pair runs oscillated ±3× (gate_proj
      v1 68.9 → 9.1 Gw/s consecutive). The plausible stable reading across
      all runs: decode runs ~25-55 Gw/s on the small layers in BOTH arms —
      **5-10× above the T7b record's 4.7-5.8 Gw/s** (which included the
      Hadamard passes + different measurement shape), but far below the
      ~17×-derivation ceiling, and NOT enough headroom to conclude fused
      GEMV reaches q4k-class decode on its own. The A3 (no-LUT) rows were
      also inconsistent (fastest on 2 layers, slowest on 2) — no reliable
      LUT-gather bound either.
- [x] T7c-1d (NEW, blocked T7c-2) — **DONE 2026-09-25 01:44 (4090, plan 003)**: the stable harness LANDED (`bench_arm_stable` + `StableBench` + `TooFastToTime`, commit `9989e9f`; driver `real_pack_decode_bench_stable`) and the RE-RUN completed (exit 0, 520.7 s, REPS=64 SAMPLES=30, pack = the §13 4bpw author pack). **Design note (honest scope):** true CUDA-event timing is unreachable in cubecl 0.11 — the CUDA server's raw `CUstream` is private, its `Fence` exposes no elapsed — so the sound primitive is **sync-bracketed system-time sampling** with an enforced ≥5 ms/pass work floor (`TooFastToTime` refuses below it), a 10% within-run spread gate, and a 15% cross-method agreement gate vs the retained differential. **Results (peak Gw/s, spread; A1/A2/A3 = v1 / v2 / v2-no-LUT):**

| layer | weights | A1 | A2 | A3 | A2/A1 | A3/A1 |
|---|---:|---:|---:|---:|---:|---:|
| L11 o_proj | 31.5M | 21.8 (4.2%) | 21.5 (4.1%) | 27.8 (9.8%) | 0.99 | 1.27 |
| L0 down_proj | 89.1M | 21.4 (6.2%) | 21.9 (4.2%) | 27.7 (5.1%) | 1.02 | 1.29 |
| L0 out_proj | 31.5M | 21.4 (13.4% UNST) | 21.3 (4.7%) | 27.3 (8.1%) | 1.00 | 1.28 |
| L0 gate_proj | 89.1M | 23.0 (5.7%) | 22.3 (5.2%) | 28.8 (4.6%) | 0.97 | 1.25 |
| lm_head | 1.27B | 21.0 (2.0%) | 20.3 (1.4%) | 26.6 (2.8%) | 0.97 | 1.27 |

**Verdicts (the decision the task named):**
1. **The T7c-1b kill criterion for the extraction class is CONFIRMED by the sound harness** — A2 ≈ A1 on every layer (0.97–1.02×). The modulo/loads mechanism was never the bottleneck; v1 stays the decode kernel; no further extraction work.
2. **A3 is the first reliable signal of the real cost: the LUT gather is a consistent 1.25–1.29× on EVERY layer** (26.6–28.8 vs 21–23 Gw/s). An arithmetic-LUT (A3-class) variant is the only remaining extraction-class win, and it is bounded at ~28 Gw/s.
3. The ceiling arithmetic stands refuted as a target: even at A3's ~28 Gw/s the decode is ~3.5× below the ~100 Gw/s (~17×) derivation — the wall is NOT the window math NOR the gather; at 13.8 GB of trellis bytes per 27B step the DRAM/L2 read of the codes themselves is the binding term (28 Gw/s ≈ 3.5 GB/s of codes at 0.125 B/weight — far under the 1 TB/s HBM). The honest reading: **inline decode cannot reach q4k-class GEMV by extraction alone; the lane decision (proceed-with-A3 / pivot to tensor-core trellis decode per the format's own design intent / close) is the OWNER call this table feeds.**
  [Correction 2026-09-25, per the closure verdict round 3: the original wording above concluded "the wall is LATENCY-class: the per-weight dependent-load chain, not bandwidth" — that mechanism was reached by ELIMINATION (bandwidth ruled out by the 3.5 GB/s-vs-1008 GB/s arithmetic), not measured; nothing measured the dependent-load chain itself, and occupancy/ILP limits and LUT-gather serialization remain live candidates. The lane decision this table fed was adjudicated 2026-09-25 by the owner-delegated verdict ping-pong (session `36dc0297`): **CLOSE at this tier** — see the T7c-2 closure block below.]
4. Harness validity: 14/15 rows under the 10% spread gate (one A1 row at 13.4% printed UNSTABLE and is excluded from verdicts); lm_head — the only layer where the pass is big enough to amortize the readback — is the only cross-agree=YES row, which is itself evidence the differential method was unsound at small scales (the reason it was replaced). v2-vs-v1 bit-exactness re-asserted per layer (§17.3 gate 1 at bench scope, green).
5. Box state: RTX 4090 24 GiB, AC, GPU at the ~5% idle desktop baseline through the run (checked pre-launch; 503 MiB in use, no compute apps); the bench window was 01:35:20–01:44:50 (+0700); a CONCURRENT reflex harness (another agent's lane) started 01:51 — AFTER the run; numbers carry that provenance.

  Blocked-on note (the process-level lesson, recorded because it cost two dead runs): a detached-over-SSH process on the 4090 is reaped when the launching SSH session tears down — the run only survives via `schtasks /create /sc once` + `/run` (session-independent). Any future long remote bench launches through a scheduled task, never `Start-Process`.
- [x] T7c-1c: full-pack bit-exact gate — **DONE 2026-09-25 03:57 (+0700, 4090 CUDA, plan 004, exit 0 in 555 s)**: `real_pack_v2_bit_exact_full` + the core `decode_w_rot_f32` extraction (commit `425e0c5`), whole pack via scheduled task. **v2 ≡ v1 ≡ CPU reference on EVERY layer: 573/573 layers, 26,481,917,952 weights, 0 bit mismatches (both v2-vs-v1 and v2-vs-CPU), wall 551.9 s.** The CPU oracle is `Exl3Layer::decode_w_rot_f32` — the decode-only LUT stage extracted from `dequantize_f32_parallel`; the scalar `dequantize_f32` stays UNTOUCHED as the independent oracle (`parallel_matches_scalar_bit_identical` pins helper ≡ scalar; core suite 17 passed; GPU module 7 passed on M3 Metal before the run). **Coverage is the gate's payoff — the synthetic fixtures cover K 2/3/4.5/5 only:** K3 34 layers / 2.55 G w · K4 305 / 19.67 G · K5 66 / 3.78 G · **K6 168 layers / 480 M w — the vision tower (`model.visual.*`) decoded for the first time**; the table independently confirms the pack's own quantization_config (vision_bits 6 → K6, mtp_bits 4 → mtp K4, head_bits 5 → lm_head K5, 1.271 G w bit-exact). All 573 groups are Cb2Mul1 — this pack is mul1 throughout, so Cb0/Cb1Mcg remain synthetic-fixture-covered only. Coverage floors pinned in the gate (≥500 plans / ≥26 G weights / ≥3 K×codebook classes) — a loader regression REDS, never a green zero. Box state (agent-measured at launch): RTX 4090 24 GiB idle desktop baseline (20% util, 503 MiB, 37.5 W, no compute apps), AC; run window 03:45–03:57 via scheduled task (the reaping lesson; a git-bundle fetch — the box's github fetch hung twice). **M3 Metal DOUBLE-BACKEND CONFIRMATION (same gate, 03:57–04:12 +0700): PASSED — 573/573 layers / 26,481,917,952 weights / 0 mismatches on the wgpu→Metal codegen path too, wall 183.3 s.** The pack (16,349,968,660 B) was copied to the M3 and is byte-count-verified against §13's recorded download size; coverage table identical (K3 34 · K4 305 · K5 66 · K6 168, all Cb2Mul1). The 555 s (CUDA) vs 183 s (Metal) wall gap is a MEASUREMENT-SHAPE note, not a throughput claim: the 4090's per-layer cost is dominated by PCIe chunk readback, which M3 unified memory does not pay — the same latency-class reading §17.5 T7c-1d recorded for the decode wall itself. Box state: AC plugged, battery 100%; only idle co-tenants (the mmorpg authority at ~1.7% CPU, a reflex serve process at ~0%) — no GPU compute during the run. **§17.3 gate 1 is closed with double-backend evidence; the T7c-2 proceed/pivot/close decision remains the OWNER call.**
- [-] T7c-2a: vector-side Hadamard transform kernels (input `v =
      H·(svh⊙x)` per 128-block; output `y = suh⊙(H·t)` per 128-block).
      **CLOSED-AT-THIS-TIER 2026-09-25** (with T7c-2b/3 below) by the
      owner-delegated Claude verdict ping-pong — session `36dc0297`,
      round 3: `REVISE` upholding the CLOSE recommendation; every revision
      executed in this same landing (see the closure block + the T7c-4
      era-gate record).
- [-] T7c-2b: the fused GEMV kernel `gemv_exl3_cubecl` (plane-per-output-
      column, inline v2 decode + FMA; `Exl3Handle` = GPU-resident trellis
      words + LUT + H + scales per layer) + the accumulation-order
      tolerance oracle vs CPU. **CLOSED-AT-THIS-TIER 2026-09-25.**
- [-] T7c-3: the delivered-gain bench (bench number NOT pre-allocated —
      take the next free per `.benchmarks/.highwater` at reopen): bytes/step
      + step time across EXL3-fused / f16 / q4k on the same weights + box,
      box state recorded; §14 gate re-run with the runtime numbers.
      **CLOSED-AT-THIS-TIER 2026-09-25.**

### 17.6 T7c-2 closure record (2026-09-25, verdict ping-pong session `36dc0297`)

**The decision:** CLOSE the fused-GEMV lane at this effort tier. Upheld by
the reviewer's round-3 verdict (a `REVISE` whose revisions were executed,
not a rejection of the close): (A) proceeding was refuted BY COMPOSITION,
not pessimism — the fused GEMV changes byte traffic, not the decode rate,
and the decode rate is the binding term; 26.5 Gw/step ÷ ~28 Gw/s ceiling ≈
0.95 s/step against a 10-20 ms incumbent is ~50×; §14's trigger asks for
measured delivered gain vs that incumbent, so the bench's verdict is
already computable from landed measurements — three GPU-days to print a
refusal buys nothing. (B) the hand-CUDA/tensor-core pivot stays rejected
for its stated reasons (week+ uncertain effort, no consumer — engine
serving integration separately open; 4090 windows owed to armed league
lanes) and is sequenced behind a POC reopen trigger (r4) instead of a
cold-start port.

**The measured bound this closure rests on** (§17.5 T7c-1d, sound
harness): decode 21-29 Gw/s across five real-pack layers and three
structurally different kernel arms; NOT bandwidth-bound (~3.5 GB/s of
codes vs ~1008 GB/s HBM). **The binding mechanism is UNMEASURED** —
candidates: per-weight dependent-load latency, occupancy/ILP limits,
LUT-gather serialization. The consistency of 21-29 Gw/s across three
arms is itself evidence, and it does not select among the candidates
(the round-3 correction to verdict 3 above).

**Reopen triggers (any one fires):**
- **r1 — a real consumer of the residency axis:** an engine serving
  integration that needs 4 bpw residency / >100k-token context on 24 GiB
  where decode speed is secondary (prefill-amortized long-context lane).
  EXL3's banked win stays §2/Bench 001 (3.4× vs f16; context 0 → ~117k).
- **r2 — upstream movement worth mining:** exllamav3 ships a production
  decode/GEMV kernel reference, or a league opponent adopts an
  EXL3-class format (competitive need).
- **r3 — the Mac Ultra context lane:** unified memory changes the
  readback economics (the T7c-1c full-pack gate already ran 3× faster on
  Metal for exactly this reason; the Metal backend is bit-exact-validated).
- **r4 — a two-stage probe proves the wall falls, cheap-first:** FIRST a
  bounded occupancy/ILP sweep on the EXISTING v1 kernel (cubes-in-flight
  × per-thread weight count — hours-class; a FLAT sweep confirms the
  closure, a RISING one reopens cheaply); the cp.async/smem-staged POC
  ONLY if the sweep is flat and the lane still matters. (The original
  single-stage cp.async-POC wording presumed the unmeasured latency
  diagnosis — corrected per the round-3 verdict.)

**What closure keeps banked:** the loader, CPU reference + fast arm, GPU
v1/v2 decode kernels, double-backend whole-pack bit-exactness (573/573
layers), and the sound bench harness — nothing in any reopen trigger has
to be rebuilt.

- [x] T7c-4: era-gate wiring at `Exl3Pack` open — **DONE 2026-09-25 (the
      hygiene half; commit recorded in §17.7)**: `verify_pack_era` reads
      `quantization_config.version` from `config.json` (embedded) and/or
      the standalone `quantization_config.json` spelling, checked against
      `KNOWN_GOOD_ERA_VERSIONS = ["1.4.2"]` (the validated pin-era pack;
      admission rule = the T7c-1c full-pack bit-exact gate on a real pack
      of that version). Fail-closed in every unvalidated direction:
      absent version (the pre-versioning legacy class — exllamav3 was at
      "0.0.1" in 2025-05, the era of the §12.7 specimen), unknown version
      (error names the OBSERVED value), disagreeing spellings, present
      `quant_method ≠ "exl3"`. Escape: `Exl3Pack::open_unverified_era` —
      the deliberate unvalidated-era read, spelled at the call site.
      Fixture battery: 5 new tests (absent/unknown/standalone/method/
      disagreement) + the 4 existing fixtures now carry known-good
      configs (a `cfg(test)` bypass was refused — test posture = shipped
      posture) + `real_pack_era_gate_opens` (env-gated pass-side arm,
      run green on the real pack: 573 groups). **Blind spot, stated on
      the gate itself:** pass side validated at n=1 (the pin-era pack),
      refuse side at n=0 (no legacy specimen on disk) — fail-closed is
      what makes n=0 acceptable. The `su/sv`-vs-`suh/svh` split was
      measured NOT an era axis (legacy signs are a SUPPORTED
      representation path, Group B/cb0, round-trip-pinned) and is not
      consulted. **The promotion half of T7c-4 closes UN-DISCHARGEABLE at
      this tier:** §14 stays REFUSED with its named trigger intact — no
      consuming arm exists, and the fused-GEMV arm that would have
      measured the delivered gain is closed above. The opt-in mitigation
      §14 leaned on ("an opt-in surface keeps the hazard explicit") is
      now a REAL refusal at open, not a doc comment.

### 17.7 Landing record (2026-09-25)

Code: `src/quant/exl3_pack.rs` — era gate + escape + 5 new tests + the 4
existing fixtures re-configured (known-good era configs), landed at
**`55c154f`** — clippy `-D warnings` clean at default AND `--features exl3`
postures,
`cargo test --features exl3 --lib exl3` 22 passed / 0 failed (3 ignored:
2 real-pack env-gated + the pre-existing ngram arm), real-pack
pass-side arm green (573 groups). The §12.7 marker-sentence correction
landed in the same commit. No files under `crates/riir-infer-laya`
(the concurrent sibling lane); no bench number consumed (T7c-3's slot
released — next free per `.highwater` at reopen).
