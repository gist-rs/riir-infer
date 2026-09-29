@echo off
rem run_kv_plus_ladder.cmd - Issue 013 T2 full run (the Issue-012 scheduled-task
rem recipe: launches OUTSIDE the agent process tree so a watchdog kill of the
rem agent cannot take the ~12h measurement down).
rem
rem Arms: f16 | k-0.00 k-0.50 k-1.00 (+ k-sched when the grid arms)
rem Slices: cal [0..61440) eval [61440..73728) search [73728..74752) of the
rem chat_probe token stream (T1's fixture; the search chunk is the new
rem held-out schedule-search slice).
rem Table artifact: .benchmarks\012_kv_table_residual.bin (BLAKE3-pinned;
rem T3's reconstruction lane reuses it — no second 2h calibration).
rem Report: rewritten after every arm at .benchmarks\012_t2_kv_ladder_report.md

cd /d E:\git\riir-infer
echo === box state at launch (%DATE% %TIME%) === > .benchmarks\012_t2_run.log
systeminfo | findstr /C:"Total Physical Memory" /C:"Available Physical Memory" /C:"System Up Time" /C:"OS Name" >> .benchmarks\012_t2_run.log
wmic OS get FreePhysicalMemory,TotalVirtualMemorySize /format:list >> .benchmarks\012_t2_run.log 2>nul

target-rel\release\kv_plus_ladder.exe ^
  ..\riir-train\data\gemma-2-2b-it-f16.gguf ^
  ..\riir-train\data\chat_probe ^
  --cal-tokens 61440 --eval-tokens 12288 --search-tokens 1024 --seq-len 1024 ^
  --dump-table .benchmarks\012_kv_table_residual.bin ^
  --niah-trials 6 ^
  --report .benchmarks\012_t2_kv_ladder_report.md ^
  >> .benchmarks\012_t2_run.log 2>&1
echo === run exited (%DATE% %TIME%) rc=%ERRORLEVEL% === >> .benchmarks\012_t2_run.log
