# Bench 872: Plan 620 Phase A — the flag-0 bit-identity gate, THREE-PART GREEN (4090 box)

**Status:** RECORD — Phase A bit-identity gate **PASSED 2026-10-12 10:01–10:36** (all three parts). The hooked build's flag-0 decode path reproduces the incumbent build BYTE-FOR-BYTE on the real league checkpoint. The plan's load-bearing law ("flag-off = bit-identical") is now measured, not asserted.

## What ran

The three-part procedure from plan 620 §Phase A, on the 4090 (Windows, CUDA, GPU-idle window after another lane's job drained):

- **Instrument:** `si_flag0_probe` (the recording bin, no si surface) at
  `--features ternary_gemv_cuda_raw,cuda_graphs_forward,riir-infer-core/bonsai2_hadamard`
  — the feature set WIDENED from the plan's original two (the league GGUF is
  Hadamard-folded; the loader loud-refuses without the feature; the plan's
  command predates the hardening). SAME set both sides.
- **Sides:** PRE = worktree `E:\git\wt-si620pre` @ `4801622` with ONLY the probe
  file overlaid from `origin/develop` (one instrument, two builds — the overlay
  is the same-file-same-bytes rule, and the probe compiles unchanged at the
  incumbent commit); POST = worktree `E:\git\wt-si620post` @ `origin/develop`
  (`aeae7d2`: decode si hook `4b4ff16` + Phase D prefill hook `61eb815` + the
  probe's cache-window clamp `aeae7d2`).
- **The clamp (aeae7d2, load-bearing for the run):** the probe originally built
  the forward at the GGUF's native block 262144 and OOMed a 24 GB card with
  17.5 GB free; the clamp (`prefill + greedy + 32` = 224, the harness's
  row_logit_floor shape) is forward-math-neutral for sequences inside the
  window and applied IDENTICALLY at both commits.
- **Provenance:** schtasks `si620flag0` + `E:\tmp\si620_flag0.cmd`; log
  `E:\tmp\si620-flag0.log`; recordings `E:\tmp\si620-flag0-{pre,post}hook.txt`
  (17 lines each, MD5 `a1f7caa54b38f446a913bf898d981e3c` BOTH SIDES); target
  dirs `E:\tmp\tgt-si620{pre,post}`. Attempts 1–5 are the honest trail:
  loader refusal (feature set), OOM ×2 (the clamp defect + another lane's
  16.8 GB job — never killed, waited out), a broken `for /f` wait-gate parse
  (nounits output is one comma-free token), and the box's mid-run reboot
  (the sibling's docs(039) wgpu-#10027 window-run restart — the darkness was
  THEIR experiment, not an outage).

## Part 1 — the probe stdout byte-compare: IDENTICAL

`fc /b prehook.txt posthook.txt` → **"FC: no differences encountered"**. Every
line load-bearing per the protocol: `block clamped to 224` (the clamp line, both
sides), eager prefill digests @16/@128, eager greedy-64 id chain + rolled
digest, the graph-lane (devpos) twins of all four, and `eager==graph ids: YES`:

```text
eager prefill digest @16:  4be351478ceaf1098ed09d3933f8d36c04aa290b3e1225df9e220d0096d34b00
eager prefill digest @128: 7c406a367bc0ec15b0af193869d242ba8b75557d61824daee002fc7c6ce73eb6
eager greedy64 rolled digest: 654fa632cff82013a0dc81c1492322f15c7b31f1352ff337ec5418d6096342ce
graph prefill digest @16:  4be351478ceaf1098ed09d3933f8d36c04aa290b3e1225df9e220d0096d34b00
graph prefill digest @128: 7c406a367bc0ec15b0af193869d242ba8b75557d61824daee002fc7c6ce73eb6
graph greedy64 rolled digest: 654fa632cff82013a0dc81c1492322f15c7b31f1352ff337ec5418d6096342ce
eager==graph ids: YES
```

(The identical eager/graph digests at PRE and POST are also a quiet cross-check
of the clamp's math-neutrality: the graph lane captures through the same
clamped-cache geometry and lands on the same bytes.)

## Parts 2+3 — the in-build unit gate + the decode matches_cpu twins: GREEN, values equal

`cargo test -p riir-infer-gpu --release --lib --features <set> -- --nocapture
attention_decode` at BOTH worktrees (schtasks `si620flag0tests`):

| side | result | max_diff values |
|---|---|---|
| PRE (4801622 + probe overlay) | **7 passed; 0 failed** (0.36s) | hd=128 n_pos=5: **2.980232e-8** · hd=256 n_pos=3: **5.960464e-8** · devpos_vs_scalar: **0.0000e0** |
| POST (aeae7d2) | **8 passed; 0 failed** (0.34s) | hd=128: **2.980232e-8** · hd=256: **5.960464e-8** · devpos_vs_scalar: **0.0000e0** · + the POST-only unit gate `test_attention_decode_si_flag_off_bit_identity_and_affine` ok, its affine check `max_diff=1.7695e-8` (one-ULP class) |

The twins' max_diff VALUES are identical across builds — the kernel-vs-CPU
agreement did not move by a single ulp. PRE correctly runs 7 (no si unit gate
exists at the incumbent commit); POST runs 8 with the unit gate green (bit-exact
disarmed-vs-dummy + identity-LUT wiring + the affine value check).

## Disclosure

- The broad `matches_cpu` test filter (first tests attempt) caught an UNRELATED
  red at this posture: `gemv_ternary_multicol_cuda_raw::tests::
  test_multicol_matches_cpu_within_int8_tolerance` (mean_rel 1.1e2 — the
  multicol lane's known in-flight class, plan 620's coordination note). Not a
  Phase-A surface; the narrow `attention_decode` filter is the gate's scope.
- GPU-exclusivity: the final runs had the GPU idle (the wait-gate held the run
  until memory.used < 12000 MiB). Earlier attempts' OOMs were capacity
  failures, never partial-results readings.

## What follows

The Phase-A checkbox flips. Plan 620's remaining box work: the G3 GPU re-run
(base-arm cross-check vs the CPU base arm — bench 936's base rows) + the G2
pp2048/tg128 cells (si-ON vs flag-off ±3%) — the GPU is idle NOW; the G3 GPU
harness bin is already landed (`si_zero_shot_ppl_gpu`, retrieval lane included).
Then plan 622's League cells + the owner-gated promotion decision (which bench
936 already bounds: the zero-shot pairing FAILED on CPU, so promotion is off
the table pending the τ sweep).

Session: si620-continuation, 2026-10-12 (gate execution + record; the probe
bin + protocol landed in the prior sessions).
