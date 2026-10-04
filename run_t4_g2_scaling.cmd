@echo off
rem run_t4_g2_scaling.cmd - riir-infer Issue 013 T4: the P3 read-path G2
rem scaling cells (the pre-registered T4 protocol: two chained cells, A then
rem B, one process each, box otherwise idle; the katgpt-rs-side promotion
rem decision reads the WINDOW-EDGE cell B per the decision rule in the issue).
rem
rem Cell A - context ~1025:  seq 1024, 4 pairs, prefill 1024, decode 64
rem Cell B - window edge ~4097: seq 4096, 2 pairs, prefill 4096, decode 64
rem (--skip-g1: the G1 record is Bench 013's; G3 still runs and still gates.)

cd /d E:\git\riir-infer
echo === T4 box state at launch (%DATE% %TIME%) === > .benchmarks\016_t4_g2_run.log
systeminfo | findstr /C:"Total Physical Memory" /C:"Available Physical Memory" /C:"OS Name" >> .benchmarks\016_t4_g2_run.log
wmic OS get FreePhysicalMemory,TotalVirtualMemorySize /format:list >> .benchmarks\016_t4_g2_run.log 2>nul
tasklist | findstr /I "python cargo kv_ vk_" >> .benchmarks\016_t4_g2_run.log 2>nul

echo === cell A: context ~1025 === >> .benchmarks\016_t4_g2_run.log
target-rel\release\kv_reconstruct_gate.exe ^
  ..\riir-train\data\gemma-2-2b-it-f16.gguf ^
  ..\riir-train\data\chat_probe ^
  --table .benchmarks\012_kv_table_residual.bin ^
  --skip-g1 --seq-len 1024 --eval-tokens 2048 ^
  --tg-pairs 4 --tg-prefill 1024 --tg-decode 64 ^
  --report .benchmarks\016a_t4_g2_cellA_report.md ^
  --box-note "4090 workstation i7-13700K, CPU lane, AC; cell A of the T4 scaling pair (decision evidence, not a GOAT)" ^
  >> .benchmarks\016_t4_g2_run.log 2>&1
echo === cell A exited rc=%ERRORLEVEL% (%DATE% %TIME%) === >> .benchmarks\016_t4_g2_run.log

echo === cell B: window edge ~4097 === >> .benchmarks\016_t4_g2_run.log
target-rel\release\kv_reconstruct_gate.exe ^
  ..\riir-train\data\gemma-2-2b-it-f16.gguf ^
  ..\riir-train\data\chat_probe ^
  --table .benchmarks\012_kv_table_residual.bin ^
  --skip-g1 --seq-len 4096 --eval-tokens 4096 ^
  --tg-pairs 2 --tg-prefill 4096 --tg-decode 64 ^
  --report .benchmarks\016b_t4_g2_cellB_report.md ^
  --box-note "4090 workstation i7-13700K, CPU lane, AC; cell B of the T4 scaling pair (the promotion-decision cell)" ^
  >> .benchmarks\016_t4_g2_run.log 2>&1
echo === cell B exited rc=%ERRORLEVEL% (%DATE% %TIME%) === >> .benchmarks\016_t4_g2_run.log
