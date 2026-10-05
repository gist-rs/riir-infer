@echo off
rem run_t4_deferred_cells.cmd - riir-infer Issue 013 T4 kernel-half run
rem (the Issue-012 scheduled-task recipe: launches OUTSIDE the agent process
rem tree so a watchdog kill of the agent cannot take the measurement down).
rem
rem Cells (the pre-registered T4 protocol, issue 013):
rem   A - context ~1025: --seq-len 1024 --tg-pairs 4 --tg-prefill 1024
rem   B - window edge ~4097: --seq-len 4096 --tg-pairs 2 --tg-prefill 4096
rem Both cells run the 4-arm deferred mode: full-cache control, eager-l1
rem (the Bench 016 continuity anchor), def-l0 (the bitwise arm), def-l1
rem (the production candidate). The decision rule reads the WINDOW-EDGE
rem deferred ratio: <= 1.20 re-fires the katgpt-core promotion; > 1.20
rem holds.
rem
rem Runs the PREBUILT exe (target-rel-t4\release\kv_reconstruct_gate.exe,
rem built from the committed tree at fe75497 BEFORE this task was
rem registered - no rebuild inside the task, so a sibling's late source
rem edits cannot alter the measurement).
rem
rem Reports: .benchmarks\017a_t4_deferred_cellA_report.md (cell A),
rem .benchmarks\017b_t4_deferred_cellB_report.md (cell B); log
rem .benchmarks\017_t4_deferred_run.log.

cd /d E:\git\riir-infer
echo === T4 deferred cells launch (%DATE% %TIME%) === > .benchmarks\017_t4_deferred_run.log

rem -- Guard: no other cargo/rustc/kv_ process may hold the box --
rem (the cell timings are paired-interleave ratios, but the wall and the
rem min_us rows are box-sensitive; the 016 recipe ran exclusive).
tasklist | findstr /I "cargo rustc kv_reconstruct kv_plus" >nul
if %ERRORLEVEL% EQU 0 (
  echo === REFUSED: a cargo/rustc/kv_ process is already running at %DATE% %TIME% === >> .benchmarks\017_t4_deferred_run.log
  tasklist | findstr /I "cargo rustc kv_reconstruct kv_plus" >> .benchmarks\017_t4_deferred_run.log
  exit /b 1
)

echo === box state at launch === >> .benchmarks\017_t4_deferred_run.log
systeminfo | findstr /C:"Total Physical Memory" /C:"Available Physical Memory" >> .benchmarks\017_t4_deferred_run.log
wmic OS get FreePhysicalMemory,TotalVirtualMemorySize /format:list >> .benchmarks\017_t4_deferred_run.log 2>nul
wmic cpu get loadpercentage /format:list >> .benchmarks\017_t4_deferred_run.log 2>nul

rem -- The exe must exist (built from the committed tree before registration) --
if not exist target-rel-t4\release\kv_reconstruct_gate.exe (
  echo === REFUSED: prebuilt exe missing - build target-rel-t4\release from the committed tree first === >> .benchmarks\017_t4_deferred_run.log
  exit /b 1
)

echo === CELL A: context ~1025 (%DATE% %TIME%) === >> .benchmarks\017_t4_deferred_run.log
target-rel-t4\release\kv_reconstruct_gate.exe ^
  ..\riir-train\data\gemma-2-2b-it-f16.gguf ^
  ..\riir-train\data\chat_probe ^
  --table .benchmarks\012_kv_table_residual.bin ^
  --eval-tokens 2048 --seq-len 1024 --skip-g1 ^
  --tg-pairs 4 --tg-prefill 1024 --tg-decode 64 ^
  --recon deferred ^
  --report .benchmarks\017a_t4_deferred_cellA_report.md ^
  --box-note "4090 workstation i7-13700K 16-core, CPU lane, AC; box exclusive (schtasks); T4 4-arm deferred mode; exe from fe75497" ^
  >> .benchmarks\017_t4_deferred_run.log 2>&1
echo === cell A exited rc=%ERRORLEVEL% (%DATE% %TIME%) === >> .benchmarks\017_t4_deferred_run.log

echo === CELL B: window edge ~4097 (%DATE% %TIME%) === >> .benchmarks\017_t4_deferred_run.log
target-rel-t4\release\kv_reconstruct_gate.exe ^
  ..\riir-train\data\gemma-2-2b-it-f16.gguf ^
  ..\riir-train\data\chat_probe ^
  --table .benchmarks\012_kv_table_residual.bin ^
  --eval-tokens 4096 --seq-len 4096 --skip-g1 ^
  --tg-pairs 2 --tg-prefill 4096 --tg-decode 64 ^
  --recon deferred ^
  --report .benchmarks\017b_t4_deferred_cellB_report.md ^
  --box-note "4090 workstation i7-13700K 16-core, CPU lane, AC; box exclusive (schtasks); T4 4-arm deferred mode; exe from fe75497" ^
  >> .benchmarks\017_t4_deferred_run.log 2>&1
echo === cell B exited rc=%ERRORLEVEL% (%DATE% %TIME%) === >> .benchmarks\017_t4_deferred_run.log

echo === T4 deferred cells complete (%DATE% %TIME%) === >> .benchmarks\017_t4_deferred_run.log
exit /b 0
