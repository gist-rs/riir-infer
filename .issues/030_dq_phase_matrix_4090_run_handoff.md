# Issue 030 — Plan 614 T5 handoff: the DQ phase-matrix 4090 RUN (queued, box was CPU-busy)

**Status:** OPEN — queued 4090 task. T5 was attempted 2026-09-30 and **deferred on the owner's stop: the box was CPU-busy** (owner observation; the plan's own GPU-exclusive precondition could not be honestly certified at that moment). This file is the turnkey pickup — every command is pre-registered in [Plan 614](../.plans/614_dq_phase_sensitivity_bench.md); nothing needs re-derivation.

## What remains (Plan 614 T5 + T6, feeding Issue 026 T1–T5)

Plan 614 T0–T3 are DONE (round-3 AGREE freeze `d3300f2`; both lanes' kernels + injections G-i3 green on Metal M3 + CUDA 4090 5/5; the runner release-green on the 4090). What is left is the RUN + close-out:

1. **Pre-run diligence (the plan mandates GPU-exclusive):**
   - `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv` — GUI processes (dwm/explorer/Zed/Docker UI) are exempt; ANY cargo/rustc/python-CUDA compute consumer = wait.
   - The OpenThai SystemOne uvicorn pair (loopback :8002, `openthai_cpu` venv) is CPU-only — not a GPU conflict, but it IS CPU load; re-check overall box idle before launching.
   - ⚠ **A sibling session is active in this repo on Issue 029** (anti-duplication marker `4d7304e`; grow-loop + `--niah-only` WIP). Their work is in the MAIN checkout `E:/git/riir-infer` — the run uses the ISOLATED worktree, so there is no file collision, but coordinate if they start GPU work.
2. **Refresh + build (cache was warm at deferral; only docs/script commits since):**
   ```powershell
   cd E:/git/dq614-wt
   git fetch origin develop; git reset --hard origin/develop
   cargo build --release -p riir-infer-gpu --no-default-features `
     --features dq_phase_bench,ternary_gemv_cuda_raw,ternary_gemm_batched --bin dq_phase_matrix
   ```
   (Worktree state at deferral: refreshed to `4d7304e`, clean, 0 dirty. Build logs convention: `F:/wt/dq614-build.*.log`.)
3. **Lane check FIRST (pre-registered):**
   ```powershell
   $env:DQ_LANE_CHECK_ONLY='1'; $env:RIIR_PREFILL_CUDA_GRAPHS='0';
   $env:BONSAI_GGUF='E:/git/riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf';
   cargo run --release -p riir-infer-gpu --no-default-features `
     --features dq_phase_bench,ternary_gemv_cuda_raw,ternary_gemm_batched --bin dq_phase_matrix
   ```
4. **Then the FULL matrix** — same command without `DQ_LANE_CHECK_ONLY`, **detached + logged** (the plan's own requirement; do NOT run it inside an interactive ssh session — a dropped connection should not kill a GPU run). Log beside the build logs (`F:/wt/`).
5. **T6 close-out:** read the four cells (`base` / `pf_aq` / `dec_aq` / `both_aq`) + the damage ratios R (Issue 026 T3's directional assertions: `R > 1` decode-heavy, `R < 1` prefill-heavy — pre-registered reference band decode 2–4×, prefll-heavy reversal weaker per the plan's SCOPE LABEL); write the `.benchmarks/` record with box state; verdict into Issue 026 (T5) + Plan 614; **delete `E:/git/dq614-wt` after T6** (the plan's own cleanup clause).

## Why deferred (the honest record)

- 2026-09-30 ~07:45 ICT: pre-flight found the GPU compute-idle (2%, 655 MiB, GUI-only) but the owner observed **high CPU on the box** and stopped the kickoff. The build was started and aborted by the stop; **verified no orphaned cargo/rustc/dq_phase processes remain** and the worktree was left clean at `4d7304e`. Nothing half-run: no matrix cells were measured.
- The plan's own precondition ("GPU-exclusive, detached+logged") plus the workspace rule (riir-ai AGENTS §GPU exclusivity — load average is NOT a proxy) makes waiting the correct call, not a deferral of convenience.

## 2026-09-30 ~10:0x +07 — pickup pre-flight RE-CHECK: precondition still unmet (documented, not run)

An idle-loop session picked this up ~2h15m after the deferral: GPU compute still
clear (nvidia-smi compute-apps = GUI-only, all exempt) but **CPU 83–84%** —
the documented OpenThai uvicorn pair is burning ~15 cores (pid 29012, 271
cumulative CPU-hours — an actively-served batch, not an idle server) beside
`kv_plus_ladder.exe` (~3 cores, the riir-train plan-612 KV-table lane's bench
over gemma-2-2b). "Re-check overall box idle" fails → the run stays correctly
deferred; nothing measured, nothing touched, the worktree remains clean at
`4d7304e`. For the next pickup: both consumers are sibling lanes with unknown
durations — re-run the pre-flight as written above; if the box has calmed,
proceed verbatim (steps 2–6 need no re-derivation). Note: the 2026-09-30
boundary-contract 162nd run flagged `dq614-wt` as contract rot (27th discovered
repo, no CANONICAL row) — expected shape, self-resolves at this run's own T6
deletion clause; do not "fix" it separately.

## 2026-09-30 ~13:1x +07 — second pickup re-check: ONE consumer cleared, the blocker stands (uvicorn)

The Batch-193 conversion session re-checked before taking the next queue
item: GPU compute still clear (GUI-only, exempt) but **CPU 73-74%** — the
OpenThai uvicorn pair (pid 29012, `openthai_systemone.server:app` :8002)
measured **16.11 cores** over an 8s window (the dominant consumer, ~24
logical procs). **`kv_plus_ladder` has FINISHED** — no longer in the top
consumers — so the earlier note's "both consumers" is stale by one; the
go/no-go verdict is unchanged (uvicorn alone saturates the box's CPU
headroom). Nothing measured, nothing touched, the worktree remains clean
at `4d7304e`. Re-run the pre-flight as written; the run proceeds verbatim
(steps 2-6) once the box calms.

## 2026-09-30 ~17:05 +07 — third pickup re-check: blocker UNCHANGED (uvicorn)

The next idle-loop session re-ran the pre-flight verbatim: GPU compute
clear (22%, 632 MiB, GUI-only, exempt) but pid 29012 (`python`, the
OpenThai SystemOne uvicorn pair) measured **15.7 cores over an 8s window**
(box load 64–68%). `kv_plus_ladder` remains finished. The go/no-go verdict
is unchanged — the run stays deferred; nothing measured, nothing touched,
the worktree remains clean at `4d7304e`. For the next pickup: re-run the
pre-flight as written; steps 2–6 proceed verbatim once the box calms.

## 2026-09-30 ~19:00–23:00 +07 — FOURTH pickup: box calmed; the lane check ran for the first time and root-caused SEVEN defects (all fixed on develop)

Pre-flight PASSED (uvicorn 0 cores, box 8-10%, GPU GUI-only). The lane
check — never before executed (T3's release-green was build-only) — was run
four times and each failure root-caused + fixed at source:

1. **`63700b0`** — the pinned GGUF declares `prism.hadamard`; the core
   loader refuses without `bonsai2_hadamard` (Issue-980 guard).
   `dq_phase_bench` now forwards it.
2. **`babd69d`** — `[734-arm8-gate] blocked by: feature:
   ternary_deltanet_chunked_prefill` (RIIR_PREFILL_CUDA_TRACE named it);
   the cudarc whole-prefill arm hard-requires chunked + attention-batched
   prefill. Both forwarded.
3. **`5ba91d1`** — `gemm_dense_ab_batched` panicked on the SECOND ragged
   prompt (x.len 1556480 vs p*n 1530880): grow-only staging (the `grow!`
   macro) hands it a capacity-sized buffer; the kernel reads only p rows.
   Capacity semantics (>=) + a NaN-poison regression test (25 excess rows;
   4090-verified 3.4e-8).
4. **`cd800a5`** — the two qrot entry points carried the same exact-eq
   asserts (the next calls in the same path); aligned to the
   fwht_rotate_copy_batched capacity convention.
5. **`b2fc1d5`** — with fallback kernels selected (no simdgroup/recurrence
   features) the lane ran ~10× slow; forwarded the engine's
   kernel-selection features so the bench BUILDS the shipping lane it
   measures.
6. **`7c578b7` + `d753e85` + `feb631e`** — THE BIG ONE: the arith corpus
   gold was LEFT-TO-RIGHT while the model — and the few-shot shots
   themselves — use standard ×-before-+/- precedence. 35/48 items
   mislabeled (73%); the model was answering its own prompts correctly
   while gold disagreed (specimen: `4359 + 592 - 198 * 96 * 8`, v1 gold
   3650304 = LTR, model −147113 = standard). **THE LANE WAS NEVER BROKEN.**
   v2 gold = eval_standard_precedence; unit tests pin the specimen +
   re-derive every item's gold from the RENDERED prompt via an independent
   evaluator.
7. **`32f4395`** — first NIAH entry OOM'd ([issue 994] wgpu OoM → pool
   poison, fail-loud working): the block_size clamp was INVERTED (`.max`
   kept the full 32768 block = 4 GiB KV; Issue-864's advice never applied).
   Fixed to `.min(max_len + 64)`.

v2-corpus lane check (in flight at this writing): base(1) arith landed
**6/48 = 12.5%** — BELOW the 0.25 admissibility floor. This is now the
MODEL's genuine ternary-arithmetic rate (instrument verified correct), not
an instrument defect; difficulty tuning after seeing accuracy is exactly
what the pre-registration freeze forbids. Per the plan's outcome table the
decode-heavy axis reports INADMISSIBLE (no gate from that axis) and the
NIAH per-length windows carry the run's value. The en-route
shot-4-regeneration pattern on negative-gold items (the shots are all
positive-result; v2 generates '-' freely) is noted for the record — a
corpus-distribution observation, not a defect.

Also en route (housekeeping): M3 incremental caches cleaned (134 G freed,
disk 85→202 Gi); the 4090 detached-run mechanism is SCHEDULED TASKS
(`schtasks`), not Start-Process — ssh session teardown kills the session
job object's children (measured: log created, no done-marker, batch dead).

## 2026-10-01 ~01:30 +07 — THE STATE LEAK: fix #8 was the root of everything; lane check EXIT 0; THE MATRIX IS RUNNING

Retraction + resolution of the fourth pickup's own readings: the v2-corpus
lane check still failed G-i4 (base(2) niah@4096 12/32 vs base(1) 27/32)
and the arith decline WITHIN cells (4/5 early → 2/43 late) was the tell —
degradation ACCUMULATED across items. **The bin never called
`fwd.reset_state()` between items** (`a07fffd`): the GDN recurrent state
carries IN PLACE across prefill calls, so every item after the first
started from the previous item's final state. The shot-regeneration
outputs, the 12.5% arith reading, the base(2) collapse — all state
pollution, not model behavior. Every pre-fix accuracy observation is
retracted.

With the reset (`a07fffd`), the lane check **EXIT 0**:
- base arith acc = **0.9583** (46/48) — above the 0.95 window ceiling by
  0.008 (a borderline the record classifies; the damage axis has full
  downward headroom)
- NIAH near-ceiling: base(1) 27/32@4k · 31/32@8k · **32/32@16k**; base(2)
  MATCHED (32/32@4k, 31/32@8k) — G-i4 byte-identical green, G-i1 green
- Clean outputs also run ~2× faster (polluted generations looped to the
  256-token cap; clean ones hit EOS early) — the matrix estimate returns
  to the plan's 2–3h class

**The full matrix fired 2026-10-01 ~01:30 +07** (both grids, all cells,
DQ_OUT=F:\wt\dq614-matrix, schtasks-detached, logged). T6 close-out
follows its completion: read the four cells + damage ratios R, write
`.benchmarks/023_dq_phase_matrix.md` with box state, verdict into Issue 026
+ Plan 614, delete `E:/git/dq614-wt`.

## Non-goals

- Do not touch Issue 029's lane (sibling-owned WIP).
- Do not re-derive the matrix definition — T0's round-3 AGREE freeze (`d3300f2`) is the review of record.
- Issues 027/028 accuracy claims stay gated on this run's instrument (Issue 026's own ordering).
