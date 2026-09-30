@echo off
rem run_kv_niah_only.cmd - Issue 029 NIAH-only rerun (the Issue-012/013 recipe:
rem launches OUTSIDE the agent process tree so a watchdog kill cannot take the
rem ~2h measurement down).
rem
rem Consumes T2's table artifact (.benchmarks\012_kv_table_residual.bin) — no
rem second calibration. The ladder/grid/validation phases are SKIPPED (--niah-only);
rem they live in the parent run's log (012_t2_run.log) and its gate doc.
rem Report: rewritten after every phase at .benchmarks\012b_niah_only_report.md

cd /d E:\git\riir-infer
echo === box state at launch (%DATE% %TIME%) === > .benchmarks\012b_niah_run.log
systeminfo | findstr /C:"Total Physical Memory" /C:"Available Physical Memory" /C:"System Up Time" /C:"OS Name" >> .benchmarks\012b_niah_run.log
wmic OS get FreePhysicalMemory,TotalVirtualMemorySize /format:list >> .benchmarks\012b_niah_run.log 2>nul
tasklist | findstr /I "python cargo kv_ vk_" >> .benchmarks\012b_niah_run.log 2>nul

target-rel\release\kv_plus_ladder.exe ^
  ..\riir-train\data\gemma-2-2b-it-f16.gguf ^
  ..\riir-train\data\chat_probe ^
  --table .benchmarks\012_kv_table_residual.bin ^
  --niah-only --niah-trials 6 --seq-len 1024 ^
  --report .benchmarks\012b_niah_only_report.md ^
  --box-note "4090 workstation i7-13700K, CPU lane, AC; sibling python trainer active (steady-state since 2026-09-29)" ^
  >> .benchmarks\012b_niah_run.log 2>&1
echo === run exited (%DATE% %TIME%) rc=%ERRORLEVEL% === >> .benchmarks\012b_niah_run.log
