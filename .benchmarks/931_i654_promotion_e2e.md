# Bench 931 — Issue 654 promotion e2e: the CUDA-placed head measured, the gate REFUSED (no win), the structural loss priced

**Date:** 2026-10-29 · **Box:** shikuwa-4090 (Windows 11, i7-13700K 24 threads, power plan "High performance") ·
**Build:** `cargo run --release` (`riir-infer-dist`, features `bonsai2_hadamard,spec_mtp_cuda`), isolated `CARGO_TARGET_DIR=C:/t654promo` ·
**Posture:** the Bench 924 in-process 2-stage CPU pair over loopback TCP, split 63, ctx 4096, prompt 768 (chat tokens), gen 192 ·
**Model:** `riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf` (the league artifact) ·
**Head:** `data/bonsai_mtp/chat/head_freerun/bonsai_mtp_head_v1.bin` (BLAKE3 `25932a3e…`, Bench 682's promotion candidate), sidecar-verified at load ·
**Scorer:** `--head-device cuda` — `riir-infer-gpu::cudarc_kernels::bmth::BmthHeadCuda` (one block per support row + the Issue-697 CPU-exact first-index argmax; per score one 20 KB H2D upload, two launches, one 8-byte readback). The `SCORER:` line named CUDA placement on every run (the 654 gate's refuse-loud law: no silent CPU fallback posture exists in this binary). ·
**Runs:** 5 harness invocations, 07:45–08:35Z; GPU 4% / 491 MiB (desktop only) before each run; no compute apps; zed + desktop processes only (the box's everyday load).

## The verdict (the pre-registered gate, applied as pinned)

**Promotion REFUSED — the arm stays opt-in, default-off.** The gate promotes only on a replicated free-run
**tokens/s win vs BOTH plain decode AND the modal drafter**; every cell LOSES to plain decode:

| posture | mtp_head | ordinary pair | ratio | modal | repeat_last |
|---|---|---|---|---|---|
| k=2, run 1 | 1.700 tok/s · E[m] 1.9592 | 2.010 | **0.846×** | 1.010 · 1.3643 | 0.602 |
| k=2, run 4 | 1.726 tok/s · E[m] 1.9592 | 1.824 | **0.946×** | 0.928 · 1.3643 | 0.607 |
| k=4, run 5 | 1.548 tok/s · E[m] 3.8400 | 2.077 | **0.745×** | 0.447 · 1.3643 | 0.346 |
| (runs 2–3 partials) | mtp 2.271 / ordinary 2.358 cross-invocation — not paired, not scored | | | 0.945 | |

- The mtp arm BEATS the modal comparator at every k (1.73 vs 0.93 at k=2) — the drafter axis holds; the
  promotion axis (vs plain decode) fails in every replicated cell. Acceptance is reported beside the speed
  throughout, per the gate.
- **E[m] is bit-stable across every run and every scorer**: 1.9592 at k=2 and 3.8400 at k=4 — EXACTLY the
  Bench 682 consumer reads (CPU scorer). The CUDA fold's argmax agreed with the CPU sequential fold on
  every scoring round of ~300 rounds — the fold-order disclosure in `bmth.rs` measured empirically silent
  at this shape. The placement changed the speed, never the drafts.
- **The placement delta is real but small**: Bench 924's CPU-scorer ratio 0.808× → CUDA-scorer 0.846–0.946×
  (paired, same box, same day: the ~6.2 ms CPU head score removed from every round). The residual loss is
  STRUCTURAL, not placement.

## Why the loss is structural (the math the measurements confirm)

The roll-chain drafter's cost model at split 63 (head = 63 layers, tail = 1 layer), per verify round:

```
round = ck(head) + K × roll(head) + K × consume(tail) + RTT
tokens_accepted ≤ K   (E[m] ≤ K by construction)
```

Rolls MUST be exact full steps — an approximate roll would break the lane's lossless G1 (the verify rows
would diverge from the sequential oracle) — so the head stage spends ~K× ordinary per-round stage time for
≤K tokens, plus the cursor-growing checkpoint (O(cursor) live rows) and the rejected-draft re-forwards. At
k=2 with E[m]=1.96 the round reads ~2× ordinary stage-time for 1.96 tokens; deeper K is WORSE (k=4: 0.745×
— Bench 682's "deeper K lowers wall-clock tok/s" reproduced verbatim at the GPU-head posture). No drafter
quality can flip this posture; the modal comparator's defeat was never the binding constraint.

**Where a win remains mathematically available (the named next rung):** a LINKED cross-machine posture with
real RTT ≫ (rolls + ck)/round — K tokens per round trip amortize the wire, ~0.5×RTT per token saved at k=2
full-accept. On loopback / home-LAN (~0.3–2 ms RTT) the win is absent (measured here and in 918/924); it
needs RTT in the tens of ms — and GPU stages with a GPU-RESIDENT checkpoint (the current ck is D2H/ memcpy
into host Vecs; GPU-resident ck machinery does not exist yet). Filed as the 654 next rung: **build the
GPU-resident checkpoint + run the linked-posture e2e across a real-RTT wire**; until then the spec lane
stays what the v0 doc always said — machinery + lossless G1, no serving claim.

## Instrument findings (recorded, not fixed here)

- **The box runs ~3.2× slower than the Bench 918-era reads**: `spec_rewind_timing --bonsai` at HEAD reads
  decode step 455.5 ms / ck 30.1 ms / ckr 30.0 ms (head [0,63), cursor 600) vs Bench 918's 143.75 / 12.24 /
  12.34 on the same box. The memcpy-class ck scaled by the same factor — a box MEMORY/thermal state change
  (post-924 reboot suspected; power plan is "High performance"), not a code regression. ABSOLUTE tok/s here
  are therefore NOT comparable to Bench 924's rows; the ratios within each run are the currency, and every
  ratio verdict above is paired within its own invocation.
- The ordinary arm varies 1.82–2.36 tok/s across invocations (±13%) under the box's everyday load; the mtp
  arm varies 1.70–2.27. Two complete paired tables (runs 1, 4) + the k=4 cell carry the verdict; runs 2–3
  are disclosed partials.
- `cargo build` on the 4090 caught the cudarc API mismatch pre-run (E0308: `CudaContext::new` takes
  `usize`), fixed in `b769ccc`; the i654 feature commit `06c184a` had also carried an orphaned
  `[[test]]` row from the shared worktree (the i1150 lane's file was untracked) — removed in `4c77521`,
  the row returns with the i1150 lane's own landing (`7e20540` now carries file + row). Both landed
  before this bench's build.

## Gates

- **G1**: `tests/spec_fill_g1.rs` 14/14 (12 + 2 new ModalDrafter pins) — the lossless battery stays green
  through the CUDA-scorer drafter (drafter-agnostic by construction, now also measured).
- The placement precondition is SATISFIED and disclosed (`SCORER:` line on every run) — the e2e measured
  the DRAFTER, and the drafter still lost on the only runnable posture.
