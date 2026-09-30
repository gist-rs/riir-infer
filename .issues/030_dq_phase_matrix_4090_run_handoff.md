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

## Non-goals

- Do not touch Issue 029's lane (sibling-owned WIP).
- Do not re-derive the matrix definition — T0's round-3 AGREE freeze (`d3300f2`) is the review of record.
- Issues 027/028 accuracy claims stay gated on this run's instrument (Issue 026's own ordering).
