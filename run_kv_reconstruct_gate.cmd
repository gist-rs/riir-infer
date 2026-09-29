@echo off
rem run_kv_reconstruct_gate.cmd - Issue 013 T3 full run (the Issue-012
rem scheduled-task recipe: launches OUTSIDE the agent process tree so a
rem watchdog kill of the agent cannot take the measurement down).
rem
rem CHAINED AFTER T2: this script WAITS for the T2 process
rem (kv_plus_ladder.exe) to exit before building or running, so the box is
rem exclusive and the G2 paired timings are not polluted by the T2 job.
rem
rem Consumes T2's table artifact .benchmarks\012_kv_table_residual.bin
rem (BLAKE3-pinned; refuses loud if missing).
rem Report: rewritten after every phase at
rem .benchmarks\013_t3_reconstruct_report.md

cd /d E:\git\riir-infer
echo === box state at launch (%DATE% %TIME%) === > .benchmarks\013_t3_run.log

rem -- Guard: wait for the T2 process to exit (max ~6 h) --
set /a waits=0
:waitloop
tasklist /FI "IMAGENAME eq kv_plus_ladder.exe" 2>nul | find /I "kv_plus_ladder.exe" >nul
if %ERRORLEVEL% EQU 0 (
  set /a waits+=1
  if %waits% GTR 720 (
    echo === GUARD TIMEOUT: kv_plus_ladder.exe still running after 12h — refusing to start (%DATE% %TIME%) === >> .benchmarks\013_t3_run.log
    exit /b 1
  )
  ping -n 61 127.0.0.1 >nul
  goto waitloop
)
echo === T2 process exited after %waits% min; box is ours (%DATE% %TIME%) === >> .benchmarks\013_t3_run.log

rem -- Guard: the T2 table artifact must exist --
if not exist .benchmarks\012_kv_table_residual.bin (
  echo === REFUSED: T2 table artifact missing — T2 must have died before freeze (%DATE% %TIME%) === >> .benchmarks\013_t3_run.log
  exit /b 1
)

systeminfo | findstr /C:"Total Physical Memory" /C:"Available Physical Memory" /C:"System Up Time" /C:"OS Name" >> .benchmarks\013_t3_run.log
wmic OS get FreePhysicalMemory,TotalVirtualMemorySize /format:list >> .benchmarks\013_t3_run.log 2>nul
tasklist | findstr /I "cargo python kv_ vk_ riir" >> .benchmarks\013_t3_run.log 2>nul

rem -- Run the PREBUILT exe (built from the committed sources before this
rem task was registered; the T2-recipe discipline — no overnight rebuild,
rem so a sibling's late source edits cannot alter the measurement) --
target-rel\release\kv_reconstruct_gate.exe ^
  ..\riir-train\data\gemma-2-2b-it-f16.gguf ^
  ..\riir-train\data\chat_probe ^
  --table .benchmarks\012_kv_table_residual.bin ^
  --eval-tokens 2048 --seq-len 1024 ^
  --tg-pairs 12 --tg-prefill 128 --tg-decode 64 ^
  --report .benchmarks\013_t3_reconstruct_report.md ^
  --box-note "4090 workstation i7-13700K, CPU lane, AC, box exclusive (T2 exited)" ^
  >> .benchmarks\013_t3_run.log 2>&1
echo === run exited (%DATE% %TIME%) rc=%ERRORLEVEL% === >> .benchmarks\013_t3_run.log
