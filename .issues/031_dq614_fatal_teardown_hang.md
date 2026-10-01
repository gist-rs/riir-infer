# Issue 031 — dq_phase_matrix FATAL teardown hangs inside process exit (CUDA driver detach-time cleanup)

**Status:** OPEN — filed 2026-10-01, defensive fix landed same day; root-cause confirmation pending an error-path repro (owner: any next matrix session)

The v1 matrix run (`a07fffd`, 2026-10-01 ~06:0x +07) measured all 9 cells,
then FATALed at the final gate (`G-i2 positive control FAIL`, the
phase-blind control — fixed separately in `23bbff5`). The FATAL line
printed and the process then **hung inside process exit for >10 minutes**:

- RAM climbed **4.7 → 8.8 GB** over ~10 min (~7 MB/s sustained allocation
  rate — an active loop, not a passive wait).
- GPU memory was released (the driver teardown progressed partway).
- The process never exited; it was **killed by hand (PID 45132)**.
- No further log output after the FATAL line.

Recorded in Issue 030 (v1 FATAL section) as "the error-exit path needs its
own fix". The hang is not cosmetic: **it produced the zombie-looking states
that a sibling session swept, killing a live re-run** (Issue 030
§"SESSION-CROSSING EVENT"). Any future error-path run will re-create the
same trap until this is fixed.

## Code path

`crates/riir-infer-gpu/src/bin/dq_phase_matrix.rs`:

```rust
fn main() {
    if let Err(e) = run() {
        eprintln!("[dq614] FATAL: {e}");
        std::process::exit(1);   // ← the hang site (by elimination)
    }
}
```

The FATAL line itself reached the log, so the hang is **after** the print,
i.e. inside `std::process::exit(1)`. `process::exit` runs the CRT atexit
table and then `ExitProcess`, which sends `DLL_PROCESS_DETACH` to every
loaded DLL **with the loader lock held**. The CUDA driver's detach-time
context cleanup is the leading suspect: with a live cudarc context
(model weights, pinned buffers, in-flight stream state from the failed
gate) the cleanup can block indefinitely while every other DLL's detach
handler queues behind the loader lock. The sustained RAM growth suggests
a retry/allocate loop inside that cleanup rather than a clean wait.

Hypotheses, ordered:
1. CUDA driver (nvcuda.dll) `DLL_PROCESS_DETACH` context cleanup blocking
   on live stream/context state — fits "GPU freed partway + RAM climbing".
2. cudarc static state interacting with exit-time teardown.
3. A CRT atexit handler registered by another dependency spinning.

Confirming the exact frame needs a Windows-side repro with a debugger or
WER dump — deliberately NOT run while the matrix occupies the box. The
defensive fix below does not require the confirmation; it removes the
ability to hang at all.

## Fix (landed 2026-10-01)

The error path must not trust the CRT exit. `hard_exit(code)`:

1. Flush stdout/stderr explicitly (the FATAL line + any buffered report).
2. On Windows: `TerminateProcess(GetCurrentProcess(), code)` — terminates
   immediately, skips atexit **and** `DLL_PROCESS_DETACH`, cannot hang.
3. Non-Windows: `std::process::exit(code)` fallback (no observed hang
   there; the driver lane is Windows-only in practice).

`std::process::abort()` was considered and rejected: it can surface a WER
dialog on a headless box — a second hang vector.

## Defense in depth (ops side, apply at close-out)

The runner `.cmd` wrappers (`E:/git/_sync/dq614_{chain,matrix}.cmd` and
`dq614_m2.cmd`) should carry a watchdog: after the log contains `FATAL`,
kill the pid if it is still alive after N minutes (the code fix makes this
a no-op; the watchdog covers any future error path that reintroduces a
slow exit). Apply when the cmd files are regenerated at Issue 030
close-out — do NOT edit `dq614_m2.cmd` while the v2 run is live.

- [x] File the issue (this file) — 2026-10-01
- [x] Land the `hard_exit` defensive fix in `dq_phase_matrix.rs`
- [x] Cross-reference from Issue 030's unowned-hang note
- [ ] Root-cause confirmation (Windows repro/dump) — only on a quiet box,
      never while a matrix run is live
- [ ] Watchdog arm in the runner `.cmd` wrappers at Issue 030 close-out
