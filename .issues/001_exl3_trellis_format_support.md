# Issue 001 — EXL3 (trellis-coded) weight-format support in the quantization zoo

**Status:** OPEN — T1 (format spec) + T2 (seam decision) DONE 2026-09-24; T3–T7 pending.
**Owner:** unassigned. **Filed:** 2026-09-24. **T1/T2 executed:** 2026-09-24 (4090 box).
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
- [ ] **T3 — CPU reference dequantization** (`src/quant/exl3.rs`), scalar and
  correct before fast. The procedural codebook and the incoherence rotation are
  the two pieces with no analogue in the existing zoo.
- [ ] **T4 — Correctness gate against an independent oracle.** Dequantize a
  real pack and compare against exllamav3's own output for the same tensors.
  ⛔ **Do NOT use output-text equality as the fidelity gate** — see §6.
- [ ] **T5 — Re-measure §2 on our silicon** (4090 / M3) before any promise about
  residency or throughput enters a plan, a README or a league row. The §2 table
  is n=1 on hardware we do not have.
- [ ] **T6 — GOAT gate + feature flag**, per the workspace promotion rule: ship
  behind an opt-in feature, bench the claim, promote only on a measured gain on
  the axis §2 says is real.
- [ ] **T7 — SIMD / GPU arms**, only after T3–T5. Scope unknown at filing.

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
| `K.mul1` | int32 marker (1 element) | `[1]` | opt. | selects codebook cb2; content = `0x83DCD12D` as u32-be-int (`codebook_mul1_mult`). REQUIRED for half-integer K. The DEFAULT modern pack (`doc/convert.md -cb`: "mul1 (default)") |
| `K.mcg` | int32 marker (1 element) | `[1]` | opt. | selects codebook cb1; content = `0xCBAC1FED` (`codebook_mcg_mult`) |
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
