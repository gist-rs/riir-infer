# Issue 031 — dq_phase_matrix FATAL teardown hangs inside process exit (CUDA driver detach-time cleanup)

**Status:** OPEN — filed 2026-10-01, defensive fix landed same day; root-cause confirmation pending an error-path repro (owner: any next matrix session). Prior-art web survey appended 2026-10-01 (§Prior art below: the class is upstream-documented driver-side teardown, config-dependent; repro guidance sharpened — sticky-error capture + context-reset A/B).

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

## Prior art (web survey, 2026-10-01 idle session — zero-box, read-only)

Searched the CUDA-driver-teardown hang class. Findings for whoever runs the
error-path repro:

1. **The class is documented upstream and is driver-side, config-dependent.**
   [NVIDIA forums #49680](https://forums.developer.nvidia.com/t/use-of-driver-api-in-dlls-causes-a-hang-on-exit-on-some-configurations/49680)
   (2017, Win7/driver 369.3/CUDA 8): a CUDA program using driver APIs hangs
   **after `main` returns**, must be killed from the task manager, and only on
   some configurations — the exact observed shape (FATAL line printed → no
   further output → killed by hand). Their resolution was giving the driver's
   cleanup a well-defined context (explicit `cuCtxPush/Pop` around GPU work in
   the library). Modern moral for us: **explicitly destroying/resetting the
   context before exit** is the upstream-sanctioned graceful alternative to
   skipping teardown.
2. **The landed fix matches documented exit semantics.** [NVIDIA forums
   #201239](https://forums.developer.nvidia.com/t/terminating-a-multi-threaded-cuda-program-that-uses-cusolver-exit-vs-exit/201239):
   CUDA registers **atexit handlers invoked by `exit()` but not `_exit()`**.
   `hard_exit`'s `TerminateProcess` skips atexit AND `DLL_PROCESS_DETACH` —
   the documented-strongest form of the same escape. Prior art validates the
   mechanism; nothing to change.
3. **Sharpened hypothesis 1 — exit-time cleanup of an ERROR-state context.**
   Our FATAL followed a failed gate (a kernel-output mismatch), i.e. a possible
   sticky CUDA error at exit. [NVIDIA forums
   #263505](https://forums.developer.nvidia.com/t/how-to-re-init-the-context-after-cudaresetdevice-now-error-cudaerrorcontextisdestroyed/263505):
   a context corrupted by a kernel execution error (700-class) does **not**
   fully recover via `cudaDeviceReset`. Every prior-art hang above is a PASSIVE
   wait — **ours allocates (~7 MB/s)**, which fits a retry/allocate loop inside
   cleanup-of-a-broken-context better than a clean block. The novel bit (RAM
   climb) remains ours to explain; the repro is still owed.
4. **cudarc-specific reports: none found** (negative search across cudarc
   GitHub/HN/dependents) — hypothesis 2 (cudarc static state) has no upstream
   attestation either way; keep it behind the debugger evidence.

**Repro additions for the matrix session** (cheap, alongside the WER dump):
- At the hang, capture any pending sticky error (`cuCtxGetLastError` /
  `cudaGetLastError` from a second attach) — its presence would confirm (3).
- A/B the error path once: explicitly drop/destroy the cudarc context (or
  `cuDevicePrimaryCtxReset`) BEFORE `hard_exit` — a graceful exit would
  confirm the mechanism and yield an optional clean-error-path fix;
  `hard_exit` stays the backstop regardless (a corrupted context may not
  reset cleanly — #263505).
- [ ] Root-cause confirmation (Windows repro/dump) — only on a quiet box,
      never while a matrix run is live
- [x] Watchdog arm in the runner `.cmd` wrappers — **LANDED 2026-10-02 (idle
      session, the close-out precondition met: Issue 030 closed, no run
      live).** Machine-local `E:/git/_sync/dq614_watchdog.ps1` (the
      regeneration copy below), started detached by a `start /B` line
      inserted before the exe launch in all three wrappers
      (`dq614_chain.cmd` → lanecheck log · `dq614_matrix.cmd` → matrix log ·
      `dq614_m2.cmd` → matrix2 log; one-line diffs, verified against
      backups). Semantics: poll the run log every 30 s for `FATAL`; once
      seen, if the image is still alive after the 600 s grace →
      `Stop-Process -Force` by image name (safe: the ALIVE guard admits at
      most one instance) + write `dq614_watchdog.fired` beside the log for
      forensics; exits quietly when the process dies first (the normal
      hard_exit path — no-op by design) and self-caps at 8 h.
      Measured both arms on this box: FATAL + live process (ping.exe
      stand-in) → killed after grace + marker written; FATAL + process gone
      → quiet exit in ~35 s (one poll cycle). No-op today: `hard_exit`
      already removes the observed hang — this covers a future error-path
      regression only.

      Regeneration copy (the wrappers and this script are machine-local;
      the durable text lives here):

      ```powershell
      param(
          [Parameter(Mandatory = $true)][string]$Log,
          [string]$ImageName = 'dq_phase_matrix.exe',
          [int]$GraceSeconds = 600,
          [int]$MaxHours = 8
      )
      $ErrorActionPreference = 'SilentlyContinue'
      $procName = $ImageName -replace '\.exe$', ''
      $deadline = (Get-Date).AddHours($MaxHours)
      $fatalSeen = $false
      $fatalAt = $null
      while ((Get-Date) -lt $deadline) {
          if (-not $fatalSeen) {
              if ((Test-Path $Log) -and (Select-String -Path $Log -Pattern 'FATAL' -Quiet)) {
                  $fatalSeen = $true
                  $fatalAt = Get-Date
              }
          }
          else {
              $alive = Get-Process -Name $procName -ErrorAction SilentlyContinue
              if (-not $alive) { exit 0 }
              if (((Get-Date) - $fatalAt).TotalSeconds -ge $GraceSeconds) {
                  Stop-Process -Name $procName -Force -ErrorAction SilentlyContinue
                  $marker = Join-Path (Split-Path $Log -Parent) 'dq614_watchdog.fired'
                  "FATAL seen $fatalAt ; killed $ImageName at $(Get-Date) after ${GraceSeconds}s grace" |
                      Out-File $marker -Encoding utf8
                  exit 0
              }
          }
          Start-Sleep -Seconds 30
      }
      ```
