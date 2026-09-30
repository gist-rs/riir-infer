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

## Non-goals

- Do not touch Issue 029's lane (sibling-owned WIP).
- Do not re-derive the matrix definition — T0's round-3 AGREE freeze (`d3300f2`) is the review of record.
- Issues 027/028 accuracy claims stay gated on this run's instrument (Issue 026's own ordering).
