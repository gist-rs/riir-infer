# Issue 001 — EXL3 (trellis-coded) weight-format support in the quantization zoo

**Status:** OPEN — scoping issue, no implementation committed.
**Owner:** unassigned. **Filed:** 2026-09-24.
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

- [ ] **T1 — Format spec read at the pin.** Read `doc/exl3.md` at a pinned
  turboderp-org/exllamav3 commit and record the on-disk layout here: tensor
  naming, the per-layer entry set, bit packing, and exactly which sidecars a
  dequantization needs. Record the pin. Nothing below starts before this.
- [ ] **T2 — Decide the loader seam.** Grouped-tensor reader in
  `safetensors_loader.rs` vs a typed quant enum shared with `GgmlType`. §4
  states the constraint; the choice is an owner/design call and the reason goes
  in this file, not in a commit message.
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
